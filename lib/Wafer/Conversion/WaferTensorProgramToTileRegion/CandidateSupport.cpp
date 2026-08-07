//===- CandidateSupport.cpp - Candidate clone and geometry support ----===//

#include "Internal.h"

#include "Wafer/IR/Target/TopologyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <limits>

using namespace wafer;

namespace wafer {

CandidateTraversalRootCapability
classifyCandidateTraversalRoot(mlir::Operation *operation) {
  if (!operation)
    return CandidateTraversalRootCapability::Unsupported;

  // Ordinary structured roots are admitted by the interfaces that define
  // their tile relation.  Do not maintain an operation allowlist here: a new
  // DPS operation that implements TilingInterface must enter the same search
  // and materialization path without a coordinator or conversion edit.
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
  if (dps && mlir::isa<mlir::TilingInterface>(operation) &&
      dps.getNumDpsInits() == 1 && operation->getNumResults() == 1)
    return CandidateTraversalRootCapability::Tiled;

  // A typed collective without a tile relation remains legal only as one
  // full traversal.  This is a capability distinction, not recognition of a
  // workload or an invitation to invent a relation in the coordinator.
  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(operation))
    return CandidateTraversalRootCapability::FullTraversalOnly;
  return CandidateTraversalRootCapability::Unsupported;
}

} // namespace wafer

namespace wafer::tensor_program_to_tile_region {

static mlir::OwningOpRef<mlir::ModuleOp>
cloneTensorProgramToStandaloneModuleImpl(mlir::func::FuncOp function) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(function.getLoc());
  if (mlir::ModuleOp sourceModule =
          function->getParentOfType<mlir::ModuleOp>())
    cloneTargetExecutionFacts(sourceModule, *module);
  mlir::OpBuilder builder(module->getBodyRegion());
  builder.clone(*function.getOperation());
  return module;
}

mlir::func::FuncOp findSingleStandaloneTensorProgram(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  bool multiple = false;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal())
      return;
    if (found) {
      multiple = true;
      return;
    }
    found = function;
  });
  return multiple ? mlir::func::FuncOp{} : found;
}

mlir::LogicalResult verifyTensorProgramScope(mlir::func::FuncOp function,
                                             std::string *failureReason) {
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody())) {
    setFailureReason(
        failureReason,
        "tensor program must be one defined single-block function");
    return mlir::failure();
  }
  if (function.getNumResults() == 0 ||
      function.getNumArguments() < function.getNumResults()) {
    setFailureReason(failureReason,
                     "tensor program boundary has invalid input/output arity");
    return mlir::failure();
  }

  TensorProgramScope scope(function);
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != scope.getOutputCount()) {
    setFailureReason(failureReason,
                     "tensor program func.return/result arity mismatch");
    return mlir::failure();
  }
  for (auto [output, resultType, returned] :
       llvm::zip(scope.getOutputs(), scope.getResultTypes(),
                 returnOp.getOperands())) {
    if (!mlir::isa<mlir::RankedTensorType>(resultType) ||
        output.getType() != resultType || returned.getType() != resultType) {
      setFailureReason(
          failureReason,
          "tensor program output arguments, results, and returns must be "
          "matching ranked tensors");
      return mlir::failure();
    }
  }

  return mlir::success();
}

bool isTensorProgramOutputBoundary(TensorProgramScope scope, mlir::Value value,
                                   unsigned outputIndex) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &scope.getBody())
    return false;
  return blockArg.getArgNumber() == scope.getInputCount() + outputIndex;
}

mlir::LogicalResult validateCandidateTile(mlir::RankedTensorType resultType,
                                          llvm::ArrayRef<int64_t> offsets,
                                          llvm::ArrayRef<int64_t> sizes,
                                          std::string *failureReason) {
  if (offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match program result rank");
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
                       "candidate tile is outside program result bounds");
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
                       llvm::ArrayRef<mlir::OpFoldResult> candidateOffsets,
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
  if (candidateOffsets.size() != candidateSizes.size() ||
      candidateOffsets.size() != outputMap.getNumResults()) {
    setFailureReason(failureReason, "candidate output tile rank mismatch");
    return mlir::failure();
  }
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
      if (candidateSizes[resultDim] <= 0) {
        setFailureReason(failureReason,
                         "candidate output tile size must be positive");
        return mlir::failure();
      }
      tile.loopOffsets.back() = candidateOffsets[resultDim];
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

  // Static materialization must remain bounded independently of candidate
  // ranking so a direct public API caller cannot allocate an unbounded chunk
  // sequence.
  constexpr uint64_t kStaticReductionChunkMaterializationBudget = 4096;
  if (*chunkCount > kStaticReductionChunkMaterializationBudget) {
    setFailureReason(
        failureReason,
        "candidate reduction split exceeds the static chunk implementation "
        "budget (4096 chunks); this is not an IR or target legality "
        "restriction");
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

} // namespace wafer::tensor_program_to_tile_region

mlir::LogicalResult wafer::verifyCandidateReductionSplitNumericLegality(
    mlir::linalg::LinalgOp root, std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  if (!root || root->getNumResults() != 1 || root.getNumDpsInits() != 1) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split requires one result and one accumulator");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::ShapedType>(root->getResult(0).getType());
  if (!resultType || !mlir::isa<mlir::IntegerType, mlir::FloatType>(
                         resultType.getElementType())) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split requires shaped integer or floating-point "
        "results");
    return mlir::failure();
  }
  if (tensor_program_to_tile_region::getReductionLoopDims(root).empty()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split requires a reduction iteration dimension");
    return mlir::failure();
  }
  if (root.getRegionOutputArgs().size() != 1) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split requires one reduction accumulator");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue = mlir::matchReduction(root.getRegionOutputArgs(),
                                                  /*redPos=*/0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split requires one exact combiner wired to the "
        "reduced value and accumulator");
    return mlir::failure();
  }

  mlir::Operation *combiner = combinerOps.front();
  if (mlir::isa<mlir::arith::AddFOp>(combiner))
    return mlir::success();
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(combiner)) {
    if (addi.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none)
      return mlir::success();
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve integer overflow flags");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MinimumFOp,
                mlir::arith::MaxSIOp, mlir::arith::MinSIOp>(combiner))
    return mlir::success();
  if (mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::MinNumFOp>(combiner)) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve maxnum/minnum NaN "
        "semantics with the current reduce kind");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(combiner)) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve unsigned min/max semantics "
        "with the current reduce kind");
    return mlir::failure();
  }

  tensor_program_to_tile_region::setFailureReason(
      failureReason,
      "candidate reduction split requires an exact sum, signed min/max, or "
      "IEEE minimum/maximum combiner");
  return mlir::failure();
}

mlir::OwningOpRef<mlir::ModuleOp>
wafer::detail::cloneTensorProgramToStandaloneModule(
    mlir::func::FuncOp function) {
  return tensor_program_to_tile_region::
      cloneTensorProgramToStandaloneModuleImpl(function);
}
