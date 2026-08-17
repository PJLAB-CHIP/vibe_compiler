//===- CandidateSupport.cpp - Candidate clone and geometry support ----===//

#include "Internal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include <algorithm>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

StructuredRootCapability classifyStructuredRoot(mlir::Operation *operation) {
  if (!operation)
    return StructuredRootCapability::Unsupported;

  // Ordinary structured roots are admitted by the interfaces that define
  // their tile relation.  Do not maintain an operation allowlist here: a new
  // DPS operation that implements TilingInterface must enter the same search
  // and materialization path without a coordinator or conversion edit.
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
  if (dps && mlir::isa<mlir::TilingInterface>(operation) &&
      dps.getNumDpsInits() == 1 && operation->getNumResults() == 1)
    return StructuredRootCapability::Tiled;

  // A typed collective without a tile relation remains legal only as one
  // full traversal.  This is a capability distinction, not recognition of a
  // workload or an invitation to invent a relation in the coordinator.
  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(operation))
    return StructuredRootCapability::FullTraversalOnly;
  return StructuredRootCapability::Unsupported;
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
                                             unsigned functionalArgumentCount,
                                             std::string *failureReason) {
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody())) {
    setFailureReason(
        failureReason,
        "tensor program must be one defined single-block function");
    return mlir::failure();
  }
  if (function.getNumResults() == 0 ||
      function.getNumArguments() !=
          functionalArgumentCount + function.getNumResults()) {
    setFailureReason(
        failureReason,
        "private scheduling boundary does not exactly extend the functional "
        "arguments with one destination per result");
    return mlir::failure();
  }

  TensorProgramScope scope(function, functionalArgumentCount);
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
    if (offset < 0 || size <= 0 || size > bound || offset > bound - size) {
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
                       llvm::ArrayRef<mlir::OpFoldResult>
                           candidateReductionOffsets,
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
    if (mlir::isa<mlir::AffineConstantExpr>(expr))
      // A constant-position extent-one result dimension is not mapped from
      // any loop dimension; its complete slice is part of every output tile.
      continue;
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
    mlir::OpFoldResult reductionOffset =
        candidateReductionOffsets[reductionOrdinal];
    int64_t reductionSize = candidateReductionSizes[reductionOrdinal];
    std::optional<int64_t> staticOffset =
        mlir::getConstantIntValue(reductionOffset);
    if (reductionSize <= 0 || reductionSize > loopRange ||
        (staticOffset &&
         (*staticOffset < 0 || *staticOffset > loopRange - reductionSize))) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
    tile.loopOffsets.back() = reductionOffset;
    tile.tileSizes.back() = builder.getIndexAttr(reductionSize);
    tile.ivs.push_back(tile.loopOffsets.back());
  }

  return mlir::success();
}

mlir::LogicalResult
verifyReductionSplitNumericLegality(mlir::linalg::LinalgOp root,
                                    bool preservesSequentialReductionOrder,
                                    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  if (!root || root->getNumResults() != 1 || root.getNumDpsInits() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one result and one accumulator");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::ShapedType>(root->getResult(0).getType());
  if (!resultType || !mlir::isa<mlir::IntegerType, mlir::FloatType>(
                         resultType.getElementType())) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires shaped integer or floating-point "
        "results");
    return mlir::failure();
  }
  if (getReductionLoopDims(root).empty()) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires a reduction iteration dimension");
    return mlir::failure();
  }
  if (root.getRegionOutputArgs().size() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one reduction accumulator");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue = mlir::matchReduction(root.getRegionOutputArgs(),
                                                  /*redPos=*/0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one exact combiner wired to the "
        "reduced value and accumulator");
    return mlir::failure();
  }

  mlir::Operation *combiner = combinerOps.front();
  if (mlir::isa<mlir::arith::AddFOp>(combiner)) {
    // Floating-point reassociation (including reduction split and tree) is a
    // supported numeric transformation per the architecture numeric policy:
    // acceptance is owned by the typed comparator, and no fast-math flag is
    // consumed as a semantics switch.
    return mlir::success();
  }
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(combiner)) {
    if (preservesSequentialReductionOrder ||
        addi.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none)
      return mlir::success();
    setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve integer overflow flags");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MinimumFOp,
                mlir::arith::MaxSIOp, mlir::arith::MinSIOp>(combiner))
    return mlir::success();
  if (mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::MinNumFOp>(combiner)) {
    setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve maxnum/minnum NaN "
        "semantics with the current reduce kind");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(combiner)) {
    setFailureReason(
        failureReason,
        "candidate reduction split cannot preserve unsigned min/max semantics "
        "with the current reduce kind");
    return mlir::failure();
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires an exact sum, signed min/max, or "
      "IEEE minimum/maximum combiner");
  return mlir::failure();
}

} // namespace wafer::tensor_program_to_tile_region
