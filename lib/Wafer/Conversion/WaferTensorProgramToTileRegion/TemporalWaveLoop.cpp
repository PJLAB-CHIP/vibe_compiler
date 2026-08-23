//===- TemporalWaveLoop.cpp - Compact iterator wave loops ------------===//

#include "TemporalWaveLoop.h"

#include "Internal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalWaveLoopNest(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<int64_t> iterationSizes,
    llvm::ArrayRef<int64_t> iteratorTileSizes,
    llvm::ArrayRef<uint32_t> waveLoopOrder, mlir::ValueRange initialValues,
    TemporalWaveLeafBuilder buildLeaf, std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> {
    setFailureReason(failureReason, message);
    return mlir::failure();
  };
  if (iterationOffsets.size() != iterationSizes.size() ||
      iterationSizes.size() != iteratorTileSizes.size() ||
      llvm::any_of(llvm::zip_equal(iterationSizes, iteratorTileSizes),
                   [](auto values) {
                     auto [extent, tile] = values;
                     return extent <= 0 || tile <= 0;
                   }))
    return fail("temporal wave traversal has a malformed iterator domain");

  llvm::SmallVector<uint32_t, 4> active;
  for (auto [dimension, extent, tile] :
       llvm::enumerate(iterationSizes, iteratorTileSizes))
    if (std::min(extent, tile) < extent)
      active.push_back(static_cast<uint32_t>(dimension));
  llvm::SmallVector<uint32_t, 4> order;
  if (waveLoopOrder.empty()) {
    order = active;
  } else {
    llvm::SmallVector<uint32_t, 4> seen;
    for (uint32_t dimension : waveLoopOrder) {
      if (dimension >= iterationSizes.size() ||
          llvm::is_contained(seen, dimension))
        return fail("temporal wave-loop order is malformed");
      seen.push_back(dimension);
      if (std::min(iterationSizes[dimension], iteratorTileSizes[dimension]) <
          iterationSizes[dimension])
        order.push_back(dimension);
    }
  }
  llvm::SmallVector<uint32_t, 4> sortedOrder = order;
  llvm::sort(sortedOrder);
  if (sortedOrder != active)
    return fail("temporal wave-loop order does not cover active iterators");

  llvm::SmallVector<mlir::OpFoldResult, 4> offsets(iterationOffsets.begin(),
                                                   iterationOffsets.end());
  llvm::SmallVector<int64_t, 4> sizes(iterationSizes.begin(),
                                      iterationSizes.end());
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;

  using Traversal =
      std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>(
          mlir::OpBuilder &, unsigned, mlir::ValueRange)>;
  Traversal traverse = [&](mlir::OpBuilder &nested, unsigned depth,
                           mlir::ValueRange values)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> {
    if (depth == order.size())
      return buildLeaf(nested, offsets, sizes, values, loops);
    const uint32_t dimension = order[depth];
    const int64_t extent = iterationSizes[dimension];
    const int64_t tileSize = std::min(extent, iteratorTileSizes[dimension]);
    const int64_t tailSize = extent % tileSize;
    const int64_t mainSize = extent - tailSize;
    llvm::SmallVector<mlir::Value, 2> current(values.begin(), values.end());

    auto materializeAt = [&](mlir::OpBuilder &bodyBuilder,
                             mlir::OpFoldResult offset, int64_t size,
                             mlir::ValueRange carried)
        -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> {
      mlir::OpFoldResult oldOffset = offsets[dimension];
      int64_t oldSize = sizes[dimension];
      offsets[dimension] = offset;
      sizes[dimension] = size;
      auto result = traverse(bodyBuilder, depth + 1, carried);
      offsets[dimension] = oldOffset;
      sizes[dimension] = oldSize;
      return result;
    };

    auto addOffset = [&](mlir::OpBuilder &offsetBuilder,
                         int64_t delta) -> mlir::FailureOr<mlir::OpFoldResult> {
      mlir::OpFoldResult base = iterationOffsets[dimension];
      if (delta == 0)
        return base;
      if (std::optional<int64_t> constant = mlir::getConstantIntValue(base)) {
        const __int128 sum = static_cast<__int128>(*constant) + delta;
        if (sum < std::numeric_limits<int64_t>::min() ||
            sum > std::numeric_limits<int64_t>::max())
          return mlir::failure();
        return mlir::OpFoldResult(
            offsetBuilder.getIndexAttr(static_cast<int64_t>(sum)));
      }
      mlir::Value baseValue =
          mlir::getValueOrCreateConstantIndexOp(offsetBuilder, loc, base);
      mlir::Value deltaValue =
          offsetBuilder.create<mlir::arith::ConstantIndexOp>(loc, delta);
      return mlir::OpFoldResult(
          offsetBuilder.create<mlir::arith::AddIOp>(loc, baseValue, deltaValue)
              .getResult());
    };

    auto prologueOffset = addOffset(nested, 0);
    if (mlir::failed(prologueOffset))
      return fail("temporal wave offset overflows index");
    auto prologue = materializeAt(nested, *prologueOffset, tileSize, current);
    if (mlir::failed(prologue))
      return mlir::failure();
    current = std::move(*prologue);
    if (mainSize > tileSize) {
      auto lowerOffset = addOffset(nested, tileSize);
      auto upperOffset = addOffset(nested, mainSize);
      if (mlir::failed(lowerOffset) || mlir::failed(upperOffset))
        return fail("temporal steady-loop offset overflows index");
      mlir::Value lower =
          mlir::getValueOrCreateConstantIndexOp(nested, loc, *lowerOffset);
      mlir::Value upper =
          mlir::getValueOrCreateConstantIndexOp(nested, loc, *upperOffset);
      auto step = nested.create<mlir::arith::ConstantIndexOp>(loc, tileSize);
      auto loop =
          nested.create<mlir::scf::ForOp>(loc, lower, upper, step, current);
      loops.push_back(
          mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
      if (!loop.getBody()->empty() &&
          mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
        loop.getBody()->back().erase();
      mlir::OpBuilder bodyBuilder =
          mlir::OpBuilder::atBlockBegin(loop.getBody());
      auto steady = materializeAt(bodyBuilder, loop.getInductionVar(), tileSize,
                                  loop.getRegionIterArgs());
      loops.pop_back();
      if (mlir::failed(steady))
        return mlir::failure();
      bodyBuilder.create<mlir::scf::YieldOp>(loc, *steady);
      nested.setInsertionPointAfter(loop);
      current.assign(loop.getResults().begin(), loop.getResults().end());
    }
    if (tailSize > 0) {
      auto tailOffset = addOffset(nested, mainSize);
      if (mlir::failed(tailOffset))
        return fail("temporal tail offset overflows index");
      auto tail = materializeAt(nested, *tailOffset, tailSize, current);
      if (mlir::failed(tail))
        return mlir::failure();
      current = std::move(*tail);
    }
    return current;
  };

  return traverse(builder, /*depth=*/0, initialValues);
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalWaveLoopNest(mlir::OpBuilder &builder, mlir::Location loc,
                                llvm::ArrayRef<int64_t> iterationOffsets,
                                llvm::ArrayRef<int64_t> iterationSizes,
                                llvm::ArrayRef<int64_t> iteratorTileSizes,
                                llvm::ArrayRef<uint32_t> waveLoopOrder,
                                mlir::ValueRange initialValues,
                                TemporalWaveLeafBuilder buildLeaf,
                                std::string *failureReason) {
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(iterationOffsets.size());
  for (int64_t offset : iterationOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  return materializeTemporalWaveLoopNest(
      builder, loc, mixedOffsets, iterationSizes, iteratorTileSizes,
      waveLoopOrder, initialValues, buildLeaf, failureReason);
}

} // namespace wafer::tensor_program_to_tile_region
