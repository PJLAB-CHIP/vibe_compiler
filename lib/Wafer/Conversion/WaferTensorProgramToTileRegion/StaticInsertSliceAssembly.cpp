//===- StaticInsertSliceAssembly.cpp - Exact insert coverage ----------===//

#include "Internal.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

struct StaticBox {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

bool checkedVolume(llvm::ArrayRef<int64_t> shape, uint64_t &volume) {
  volume = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || static_cast<uint64_t>(extent) >
                           std::numeric_limits<uint64_t>::max() / volume)
      return false;
    volume *= static_cast<uint64_t>(extent);
  }
  return true;
}

bool boxesOverlap(const StaticBox &lhs, const StaticBox &rhs) {
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes))
    if (lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset)
      return false;
  return true;
}

} // namespace

mlir::FailureOr<std::optional<StaticInsertSliceAssembly>>
analyzeCompleteStaticInsertSliceAssembly(mlir::Value value) {
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!resultType || !resultType.hasStaticShape())
    return std::optional<StaticInsertSliceAssembly>{};

  StaticInsertSliceAssembly result;
  mlir::Value current = value;
  while (auto insert = current.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
    result.inserts.push_back(insert);
    current = insert.getDest();
  }
  auto empty = current.getDefiningOp<mlir::tensor::EmptyOp>();
  if (!empty || result.inserts.empty() || current.getType() != resultType ||
      !empty.getDynamicSizes().empty())
    return std::optional<StaticInsertSliceAssembly>{};

  uint64_t resultVolume = 0;
  if (!checkedVolume(resultType.getShape(), resultVolume))
    return mlir::failure();
  uint64_t assembledVolume = 0;
  llvm::SmallVector<StaticBox, 4> boxes;
  for (mlir::tensor::InsertSliceOp insert : llvm::reverse(result.inserts)) {
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getRank() != resultType.getRank() ||
        insert.getDestType() != resultType || insert.getType() != resultType ||
        llvm::is_contained(insert.getStaticOffsets(),
                           mlir::ShapedType::kDynamic) ||
        llvm::is_contained(insert.getStaticSizes(),
                           mlir::ShapedType::kDynamic) ||
        llvm::is_contained(insert.getStaticStrides(),
                           mlir::ShapedType::kDynamic) ||
        !llvm::all_of(insert.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return std::optional<StaticInsertSliceAssembly>{};

    StaticBox box;
    box.offsets.assign(insert.getStaticOffsets().begin(),
                       insert.getStaticOffsets().end());
    box.sizes.assign(insert.getStaticSizes().begin(),
                     insert.getStaticSizes().end());
    if (box.offsets.size() != static_cast<size_t>(resultType.getRank()) ||
        box.sizes.size() != static_cast<size_t>(resultType.getRank()) ||
        sourceType.getShape() != llvm::ArrayRef<int64_t>(box.sizes))
      return std::optional<StaticInsertSliceAssembly>{};
    for (auto [offset, size, bound] :
         llvm::zip_equal(box.offsets, box.sizes, resultType.getShape()))
      if (offset < 0 || size <= 0 || size > bound || offset > bound - size)
        return std::optional<StaticInsertSliceAssembly>{};
    if (llvm::any_of(boxes, [&](const StaticBox &other) {
          return boxesOverlap(box, other);
        }))
      return std::optional<StaticInsertSliceAssembly>{};

    uint64_t boxVolume = 0;
    if (!checkedVolume(box.sizes, boxVolume) ||
        boxVolume > std::numeric_limits<uint64_t>::max() - assembledVolume)
      return mlir::failure();
    assembledVolume += boxVolume;
    boxes.push_back(std::move(box));
  }
  if (assembledVolume != resultVolume)
    return std::optional<StaticInsertSliceAssembly>{};

  result.base = current;
  return std::optional<StaticInsertSliceAssembly>(std::move(result));
}

} // namespace wafer::tensor_program_to_tile_region
