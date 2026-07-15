//===- CandidateSupport.cpp - Candidate clone and geometry support ----===//

#include "Internal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include <algorithm>
#include <limits>

using namespace wafer;

namespace wafer::group_to_tile_region {

static mlir::OwningOpRef<mlir::ModuleOp>
cloneGroupToStandaloneModuleImpl(GroupOp group) {
  mlir::Location loc = group.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> standaloneModule =
      mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(standaloneModule->getBodyRegion());

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  for (mlir::Value input : group.getInputs())
    inputTypes.push_back(input.getType());
  for (mlir::Value output : group.getOuts())
    inputTypes.push_back(output.getType());

  auto funcType =
      moduleBuilder.getFunctionType(inputTypes, group.getResultTypes());
  auto func = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "group_to_tile_region", funcType);
  mlir::Block *entry = func.addEntryBlock();

  mlir::OpBuilder builder(entry, entry->end());
  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : group.getInputs()) {
    mlir::Value replacement = entry->getArgument(argumentIndex++);
    if (auto constant = input.getDefiningOp<mlir::arith::ConstantOp>()) {
      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      replacement = cloned->getResult(0);
    }
    mapping.map(input, replacement);
  }
  for (mlir::Value output : group.getOuts())
    mapping.map(output, entry->getArgument(argumentIndex++));

  auto clonedGroup =
      mlir::cast<GroupOp>(builder.clone(*group.getOperation(), mapping));
  builder.create<mlir::func::ReturnOp>(loc, clonedGroup.getResults());
  return standaloneModule;
}

GroupOp findSingleStandaloneGroup(mlir::ModuleOp module) {
  GroupOp found;
  module.walk([&](GroupOp group) {
    if (!found)
      found = group;
  });
  return found;
}

bool isGroupOutputBoundary(GroupOp group, mlir::Value value,
                           unsigned outputIndex) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &group.getBody().front())
    return false;
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  return blockArg.getArgNumber() == inputCount + outputIndex;
}

mlir::LogicalResult validateCandidateTile(mlir::RankedTensorType resultType,
                                          llvm::ArrayRef<int64_t> offsets,
                                          llvm::ArrayRef<int64_t> sizes,
                                          std::string *failureReason) {
  if (offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match group result rank");
    return mlir::failure();
  }

  for (auto [dim, values] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    int64_t offset = std::get<0>(values);
    int64_t size = std::get<1>(values);
    int64_t bound = resultType.getDimSize(dim);
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "candidate tile requires static result shape");
      return mlir::failure();
    }
    if (offset < 0 || size <= 0 || offset + size > bound) {
      setFailureReason(failureReason,
                       "candidate tile is outside group result bounds");
      return mlir::failure();
    }
  }
  return mlir::success();
}

llvm::SmallVector<unsigned, 2> getReductionLoopDims(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<unsigned, 2> dims;
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      dims.push_back(static_cast<unsigned>(index));
  }
  return dims;
}

mlir::LogicalResult
buildCandidateLoopTile(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::linalg::LinalgOp op, mlir::AffineMap outputMap,
                       llvm::ArrayRef<int64_t> candidateOffsets,
                       llvm::ArrayRef<int64_t> candidateSizes,
                       llvm::ArrayRef<int64_t> candidateReductionOffsets,
                       llvm::ArrayRef<int64_t> candidateReductionSizes,
                       CandidateLoopTile &tile, std::string *failureReason) {
  llvm::SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t value) {
        return mlir::ShapedType::isDynamic(value);
      })) {
    setFailureReason(failureReason,
                     "candidate tile requires static linalg loop ranges");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(op);
  bool hasReductionSplit =
      !candidateReductionOffsets.empty() || !candidateReductionSizes.empty();
  if (candidateReductionOffsets.size() != candidateReductionSizes.size() ||
      (hasReductionSplit &&
       candidateReductionOffsets.size() != reductionLoopDims.size())) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::DenseMap<unsigned, unsigned> resultDimForLoopDim;
  for (auto [resultDim, expr] : llvm::enumerate(outputMap.getResults())) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) {
      setFailureReason(failureReason,
                       "candidate tile requires permutation-only output map");
      return mlir::failure();
    }
    resultDimForLoopDim[dimExpr.getPosition()] =
        static_cast<unsigned>(resultDim);
  }

  llvm::DenseMap<unsigned, unsigned> reductionOrdinalForLoopDim;
  for (auto [ordinal, loopDim] : llvm::enumerate(reductionLoopDims))
    reductionOrdinalForLoopDim[loopDim] = static_cast<unsigned>(ordinal);

  mlir::OpFoldResult zero = builder.getIndexAttr(0);
  for (auto [loopDim, loopRange] : llvm::enumerate(loopRanges)) {
    tile.sizeBounds.push_back(builder.getIndexAttr(loopRange));
    tile.loopOffsets.push_back(zero);
    tile.tileSizes.push_back(zero);

    auto resultDimIt = resultDimForLoopDim.find(static_cast<unsigned>(loopDim));
    if (resultDimIt != resultDimForLoopDim.end()) {
      unsigned resultDim = resultDimIt->second;
      tile.loopOffsets.back() =
          builder.getIndexAttr(candidateOffsets[resultDim]);
      tile.tileSizes.back() = builder.getIndexAttr(candidateSizes[resultDim]);
      tile.ivs.push_back(tile.loopOffsets.back());
      continue;
    }

    auto reductionDimIt =
        reductionOrdinalForLoopDim.find(static_cast<unsigned>(loopDim));
    if (reductionDimIt == reductionOrdinalForLoopDim.end())
      continue;
    if (!hasReductionSplit)
      continue;

    unsigned reductionOrdinal = reductionDimIt->second;
    int64_t reductionOffset = candidateReductionOffsets[reductionOrdinal];
    int64_t reductionSize = candidateReductionSizes[reductionOrdinal];
    if (reductionOffset < 0 || reductionSize <= 0 ||
        reductionOffset + reductionSize > loopRange) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
    tile.loopOffsets.back() = builder.getIndexAttr(reductionOffset);
    tile.tileSizes.back() = builder.getIndexAttr(reductionSize);
    tile.ivs.push_back(tile.loopOffsets.back());
  }

  return mlir::success();
}

mlir::FailureOr<uint64_t> getCandidateReductionChunkCount(
    mlir::linalg::LinalgOp root,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (candidateReductionTileSizes.empty())
    return uint64_t{1};

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  if (candidateReductionTileSizes.size() != reductionLoopDims.size()) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  llvm::SmallVector<int64_t, 2> reductionRanges;
  reductionRanges.reserve(reductionLoopDims.size());
  for (unsigned loopDim : reductionLoopDims) {
    if (loopDim >= loopRanges.size() ||
        mlir::ShapedType::isDynamic(loopRanges[loopDim])) {
      setFailureReason(failureReason,
                       "candidate reduction split requires static loop ranges");
      return mlir::failure();
    }
    reductionRanges.push_back(loopRanges[loopDim]);
  }

  for (auto [range, splitSize] :
       llvm::zip(reductionRanges, candidateReductionTileSizes)) {
    if (range <= 0 || splitSize <= 0 || splitSize > range) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
  }

  uint64_t chunkCount = 0;
  switch (wafer::detail::checkedStaticTileProduct(
      reductionRanges, candidateReductionTileSizes, chunkCount)) {
  case wafer::detail::CheckedStaticTileProductStatus::Success:
    return chunkCount;
  case wafer::detail::CheckedStaticTileProductStatus::Overflow:
    setFailureReason(
        failureReason,
        "candidate reduction split expansion count is not representable");
    return mlir::failure();
  case wafer::detail::CheckedStaticTileProductStatus::InvalidInput:
    setFailureReason(failureReason,
                     "candidate reduction split has invalid static ranges");
    return mlir::failure();
  }
  llvm_unreachable("unknown checked tile product status");
}

static void
buildReductionChunkProducts(llvm::ArrayRef<int64_t> ranges,
                            llvm::ArrayRef<int64_t> splitSizes, unsigned dim,
                            llvm::SmallVectorImpl<int64_t> &currentOffsets,
                            llvm::SmallVectorImpl<int64_t> &currentSizes,
                            llvm::SmallVectorImpl<ReductionChunk> &chunks) {
  if (dim == ranges.size()) {
    chunks.push_back(
        ReductionChunk{llvm::SmallVector<int64_t, 2>(currentOffsets.begin(),
                                                     currentOffsets.end()),
                       llvm::SmallVector<int64_t, 2>(currentSizes.begin(),
                                                     currentSizes.end())});
    return;
  }

  for (int64_t offset = 0; offset < ranges[dim];) {
    int64_t size = std::min(splitSizes[dim], ranges[dim] - offset);
    currentOffsets.push_back(offset);
    currentSizes.push_back(size);
    buildReductionChunkProducts(ranges, splitSizes, dim + 1, currentOffsets,
                                currentSizes, chunks);
    currentOffsets.pop_back();
    currentSizes.pop_back();
    offset += size;
  }
}

mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>>
buildReductionChunks(mlir::linalg::LinalgOp root,
                     llvm::ArrayRef<int64_t> candidateReductionTileSizes,
                     std::string *failureReason) {
  mlir::FailureOr<uint64_t> chunkCount = getCandidateReductionChunkCount(
      root, candidateReductionTileSizes, failureReason);
  if (mlir::failed(chunkCount))
    return mlir::failure();
  if (candidateReductionTileSizes.empty())
    return llvm::SmallVector<ReductionChunk, 8>{ReductionChunk{}};

  if (*chunkCount > wafer::detail::kCompleteCandidateMaterializationBudget) {
    setFailureReason(
        failureReason,
        "candidate reduction split exceeds the eager materialization budget "
        "(4096 chunks); this is an implementation resource limit, not an IR "
        "or target legality restriction");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  llvm::SmallVector<int64_t, 2> reductionRanges;
  for (unsigned loopDim : reductionLoopDims)
    reductionRanges.push_back(loopRanges[loopDim]);

  llvm::SmallVector<ReductionChunk, 8> chunks;
  chunks.reserve(static_cast<size_t>(*chunkCount));
  llvm::SmallVector<int64_t, 2> currentOffsets;
  llvm::SmallVector<int64_t, 2> currentSizes;
  buildReductionChunkProducts(reductionRanges, candidateReductionTileSizes,
                              /*dim=*/0, currentOffsets, currentSizes, chunks);
  return chunks;
}

} // namespace wafer::group_to_tile_region

mlir::OwningOpRef<mlir::ModuleOp>
wafer::detail::cloneGroupToStandaloneModule(GroupOp group) {
  return group_to_tile_region::cloneGroupToStandaloneModuleImpl(group);
}
