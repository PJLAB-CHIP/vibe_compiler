//===- StructuredTiling.cpp - Interface-driven tile traversal ----------===//

#include "Wafer/Transforms/Linalg/StructuredTiling.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

using namespace wafer;

namespace {

void setFailureReason(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

mlir::LogicalResult
verifyOperandTileRequest(mlir::Operation *consumer, unsigned operandNumber,
                         llvm::ArrayRef<mlir::OpFoldResult> offsets,
                         llvm::ArrayRef<mlir::OpFoldResult> sizes,
                         std::string *failureReason) {
  if (!consumer) {
    setFailureReason(failureReason,
                     "operand-driven tiling requires a consumer operation");
    return mlir::failure();
  }
  if (operandNumber >= consumer->getNumOperands()) {
    setFailureReason(failureReason,
                     "operand-driven tile operand number is out of range");
    return mlir::failure();
  }
  if (offsets.size() != sizes.size()) {
    setFailureReason(failureReason,
                     "operand-driven tile offset/size rank mismatch");
    return mlir::failure();
  }

  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(
      consumer->getOperand(operandNumber).getType());
  if (!shapedType || !shapedType.hasRank() ||
      offsets.size() != static_cast<size_t>(shapedType.getRank())) {
    setFailureReason(
        failureReason,
        "operand-driven tile rank does not match the selected operand");
    return mlir::failure();
  }
  if (!mlir::isa<mlir::TilingInterface>(consumer)) {
    setFailureReason(
        failureReason,
        "operand-driven tile consumer does not implement TilingInterface");
    return mlir::failure();
  }
  return mlir::success();
}

mlir::FailureOr<OperandTileIterationDomain> mapOperandTileToIterationDomain(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (mlir::failed(verifyOperandTileRequest(consumer, operandNumber, offsets,
                                            sizes, failureReason)))
    return mlir::failure();

  auto tiling = mlir::cast<mlir::TilingInterface>(consumer);
  OperandTileIterationDomain result;
  if (mlir::failed(tiling.getIterationDomainTileFromOperandTile(
          builder, operandNumber, offsets, sizes, result.offsets,
          result.sizes))) {
    setFailureReason(
        failureReason,
        "TilingInterface rejected the operand-to-iteration tile relation");
    return mlir::failure();
  }
  size_t expectedRank = tiling.getLoopIteratorTypes().size();
  if (result.offsets.size() != expectedRank ||
      result.sizes.size() != expectedRank) {
    setFailureReason(
        failureReason,
        "TilingInterface returned an incomplete operand-to-iteration tile "
        "relation");
    return mlir::failure();
  }
  return result;
}

} // namespace

mlir::FailureOr<wafer::IterationTileMaterialization>
wafer::materializeOperationFromIterationTile(
    mlir::Operation *operation, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling) {
    setFailureReason(
        failureReason,
        "fixed iteration tile requires an operation with TilingInterface");
    return mlir::failure();
  }
  if (iterationOffsets.size() != iterationSizes.size() ||
      iterationOffsets.size() != tiling.getLoopIteratorTypes().size()) {
    setFailureReason(
        failureReason,
        "fixed iteration tile rank does not match the operation iterator "
        "domain");
    return mlir::failure();
  }
  IterationTileMaterialization result;
  for (unsigned resultNumber = 0; resultNumber < operation->getNumResults();
       ++resultNumber) {
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    if (mlir::failed(tiling.getResultTilePosition(
            builder, resultNumber, iterationOffsets, iterationSizes, offsets,
            sizes))) {
      setFailureReason(failureReason,
                       "TilingInterface cannot map one fixed result tile");
      return mlir::failure();
    }
    result.resultOffsets.push_back(std::move(offsets));
    result.resultSizes.push_back(std::move(sizes));
  }
  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, iterationOffsets, iterationSizes);
  if (mlir::failed(tiled) || tiled->tiledOps.empty() ||
      tiled->tiledValues.size() != operation->getNumResults()) {
    if (mlir::succeeded(tiled)) {
      for (mlir::Operation *tiledOperation : llvm::reverse(tiled->tiledOps))
        if (tiledOperation->use_empty())
          tiledOperation->erase();
      for (mlir::Operation *slice : llvm::reverse(tiled->generatedSlices))
        if (slice->use_empty())
          slice->erase();
    }
    setFailureReason(failureReason,
                     "TilingInterface rejected the fixed iteration tile");
    return mlir::failure();
  }
  result.tiledOperations.assign(tiled->tiledOps.begin(), tiled->tiledOps.end());
  result.tiledValues.assign(tiled->tiledValues.begin(),
                            tiled->tiledValues.end());
  result.generatedSlices.assign(tiled->generatedSlices.begin(),
                                tiled->generatedSlices.end());
  return result;
}

mlir::FailureOr<wafer::OperandTileMaterialization>
wafer::materializeConsumerFromOperandTile(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason) {
  mlir::FailureOr<OperandTileIterationDomain> iterationDomain =
      mapOperandTileToIterationDomain(consumer, builder, operandNumber, offsets,
                                      sizes, failureReason);
  if (mlir::failed(iterationDomain))
    return mlir::failure();

  auto tiling = mlir::cast<mlir::TilingInterface>(consumer);
  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementationFromOperandTile(builder, operandNumber,
                                                   offsets, sizes);
  if (mlir::failed(tiled)) {
    setFailureReason(
        failureReason,
        "TilingInterface rejected consumer materialization from operand tile");
    return mlir::failure();
  }
  if (tiled->tiledOps.empty() || tiled->tiledValues.empty()) {
    setFailureReason(
        failureReason,
        "operand-driven tiling produced no consumer implementation");
    return mlir::failure();
  }

  OperandTileMaterialization result;
  result.iterationDomain = std::move(*iterationDomain);
  result.tiledOperations.assign(tiled->tiledOps.begin(), tiled->tiledOps.end());
  result.tiledValues.assign(tiled->tiledValues.begin(),
                            tiled->tiledValues.end());
  result.generatedSlices.assign(tiled->generatedSlices.begin(),
                                tiled->generatedSlices.end());
  return result;
}

mlir::FailureOr<wafer::PartialReductionTileMaterialization>
wafer::materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason) {
  return materializePartialReductionTile(
      reduction, builder, iterationOffsets, iterationSizes,
      /*resultTileDestinations=*/{}, failureReason);
}

mlir::FailureOr<wafer::PartialReductionTileMaterialization>
wafer::materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    mlir::ValueRange requestedResultTileDestinations,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!reduction) {
    setFailureReason(failureReason,
                     "partial-reduction tiling requires a reduction operation");
    return mlir::failure();
  }

  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(reduction);
  auto partial = mlir::dyn_cast<mlir::PartialReductionOpInterface>(reduction);
  if (!tiling || !partial) {
    setFailureReason(failureReason,
                     "partial-reduction tiling requires TilingInterface and "
                     "PartialReductionOpInterface");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      tiling.getLoopIteratorTypes();
  if (iterationOffsets.size() != iteratorTypes.size() ||
      iterationSizes.size() != iteratorTypes.size()) {
    setFailureReason(
        failureReason,
        "partial-reduction tile rank does not match the iteration domain");
    return mlir::failure();
  }

  PartialReductionTileMaterialization result;
  for (auto [dimension, iteratorType] : llvm::enumerate(iteratorTypes))
    if (iteratorType == mlir::utils::IteratorType::reduction)
      result.reductionDimensions.push_back(static_cast<int>(dimension));
  if (result.reductionDimensions.empty()) {
    setFailureReason(
        failureReason,
        "partial-reduction tiling requires at least one reduction dimension");
    return mlir::failure();
  }

  mlir::Location loc = reduction->getLoc();
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(reduction);
  if (!dps || dps.getNumDpsInits() != reduction->getNumResults()) {
    setFailureReason(
        failureReason,
        "partial-reduction tiling requires one destination per result");
    return mlir::failure();
  }

  if (!requestedResultTileDestinations.empty() &&
      requestedResultTileDestinations.size() != reduction->getNumResults()) {
    setFailureReason(
        failureReason,
        "partial-reduction destination count does not match result count");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 2> resultTileDestinations;
  resultTileDestinations.reserve(reduction->getNumResults());
  for (unsigned resultNumber = 0; resultNumber < reduction->getNumResults();
       ++resultNumber) {
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    if (mlir::failed(tiling.getResultTilePosition(
            builder, resultNumber, iterationOffsets, iterationSizes,
            resultOffsets, resultSizes))) {
      setFailureReason(
          failureReason,
          "TilingInterface failed to map the partial-reduction result tile");
      return mlir::failure();
    }
    mlir::Value destination = dps.getDpsInitOperand(resultNumber)->get();
    auto destinationType =
        mlir::dyn_cast<mlir::RankedTensorType>(destination.getType());
    if (!destinationType ||
        resultOffsets.size() !=
            static_cast<size_t>(destinationType.getRank()) ||
        resultSizes.size() != static_cast<size_t>(destinationType.getRank())) {
      setFailureReason(
          failureReason,
          "partial-reduction result tile does not match its destination");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::OpFoldResult, 4> strides(destinationType.getRank(),
                                                     builder.getIndexAttr(1));
    mlir::RankedTensorType expectedType =
        mlir::tensor::ExtractSliceOp::inferResultType(
            destinationType, resultOffsets, resultSizes, strides);
    if (!requestedResultTileDestinations.empty()) {
      mlir::Value requested = requestedResultTileDestinations[resultNumber];
      if (requested.getType() != expectedType) {
        setFailureReason(
            failureReason,
            "partial-reduction destination type does not match result tile");
        return mlir::failure();
      }
      resultTileDestinations.push_back(requested);
      continue;
    }
    resultTileDestinations.push_back(
        builder
            .create<mlir::tensor::ExtractSliceOp>(
                loc, destination, resultOffsets, resultSizes, strides)
            .getResult());
  }

  // mergeReductions uses the interface operation's destinations. The clone is
  // only a transient adapter so a smaller output tile merges into its matching
  // destination slice instead of recreating a full-result reduction.
  mlir::Operation *interfaceOperation = builder.clone(*reduction);
  auto clonedDps =
      mlir::cast<mlir::DestinationStyleOpInterface>(interfaceOperation);
  for (auto [initOperand, destination] :
       llvm::zip(clonedDps.getDpsInitsMutable(), resultTileDestinations))
    initOperand.set(destination);
  for (auto [resultValue, destination] :
       llvm::zip(interfaceOperation->getResults(), resultTileDestinations))
    resultValue.setType(destination.getType());
  auto interface =
      mlir::cast<mlir::PartialReductionOpInterface>(interfaceOperation);

  mlir::FailureOr<llvm::SmallVector<mlir::Value>> initial =
      interface.generateInitialTensorForPartialReduction(
          builder, loc, iterationSizes, result.reductionDimensions);
  if (mlir::failed(initial) || initial->empty()) {
    interfaceOperation->erase();
    setFailureReason(
        failureReason,
        "PartialReductionOpInterface failed to generate initial values");
    return mlir::failure();
  }
  result.initialValues.assign(initial->begin(), initial->end());

  mlir::FailureOr<mlir::TilingResult> partialTiling =
      interface.tileToPartialReduction(builder, loc, result.initialValues,
                                       iterationOffsets, iterationSizes,
                                       result.reductionDimensions);
  if (mlir::failed(partialTiling) || partialTiling->tiledOps.empty() ||
      partialTiling->tiledValues.empty()) {
    interfaceOperation->erase();
    setFailureReason(
        failureReason,
        "PartialReductionOpInterface failed to materialize partial values");
    return mlir::failure();
  }
  result.partialOperations.assign(partialTiling->tiledOps.begin(),
                                  partialTiling->tiledOps.end());
  result.partialValues.assign(partialTiling->tiledValues.begin(),
                              partialTiling->tiledValues.end());
  result.generatedSlices.assign(partialTiling->generatedSlices.begin(),
                                partialTiling->generatedSlices.end());

  // The pinned Linalg external model appends each reduction dimension to an
  // init indexing map, while generateInitialTensorForPartialReduction inserts
  // the corresponding tensor dimension at its original iterator position.
  // Those two choices coincide for trailing reductions but produce invalid IR
  // for a legal non-trailing reduction. Rebuild only the just-created actual
  // partial generic with the same tiled inputs, current identity tensors and
  // a map whose result order matches those tensors. This remains one local
  // current-IR transformation; no future operation or replay record is kept.
  auto sourceLinalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(reduction);
  if (sourceLinalg && result.partialOperations.size() == 1) {
    auto partialGeneric =
        mlir::dyn_cast<mlir::linalg::GenericOp>(result.partialOperations[0]);
    if (!partialGeneric) {
      interfaceOperation->erase();
      setFailureReason(failureReason,
                       "Linalg partial reduction did not produce a generic");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::AffineMap, 4> maps =
        partialGeneric.getIndexingMapsArray();
    llvm::DenseSet<int> reductionDimensionSet(
        result.reductionDimensions.begin(), result.reductionDimensions.end());
    for (unsigned initIndex = 0; initIndex < sourceLinalg.getNumDpsInits();
         ++initIndex) {
      mlir::AffineMap original = sourceLinalg.getMatchingIndexingMap(
          sourceLinalg.getDpsInitOperand(initIndex));
      llvm::SmallVector<mlir::AffineExpr, 4> orderedResults;
      unsigned originalResult = 0;
      const unsigned partialRank =
          original.getNumResults() + result.reductionDimensions.size();
      for (unsigned position = 0; position < partialRank; ++position) {
        if (reductionDimensionSet.contains(position)) {
          orderedResults.push_back(builder.getAffineDimExpr(position));
          continue;
        }
        if (originalResult >= original.getNumResults()) {
          interfaceOperation->erase();
          setFailureReason(
              failureReason,
              "partial-reduction init map cannot preserve iterator order");
          return mlir::failure();
        }
        orderedResults.push_back(original.getResult(originalResult++));
      }
      if (originalResult != original.getNumResults()) {
        interfaceOperation->erase();
        setFailureReason(
            failureReason,
            "partial-reduction init map does not cover its output dimensions");
        return mlir::failure();
      }
      mlir::OpOperand *partialInit =
          partialGeneric.getDpsInitOperand(initIndex);
      auto partialLinalg =
          mlir::cast<mlir::linalg::LinalgOp>(partialGeneric.getOperation());
      maps[partialLinalg.getIndexingMapIndex(partialInit)] =
          mlir::AffineMap::get(original.getNumDims(), original.getNumSymbols(),
                               orderedResults, original.getContext());
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(partialGeneric);
    auto rebuilt = builder.create<mlir::linalg::GenericOp>(
        loc, mlir::ValueRange(result.initialValues).getTypes(),
        partialGeneric.getDpsInputs(), result.initialValues, maps,
        partialGeneric.getIteratorTypesArray());
    mlir::IRMapping mapping;
    partialGeneric.getRegion().cloneInto(&rebuilt.getRegion(),
                                         rebuilt.getRegion().begin(), mapping);
    partialGeneric.erase();
    result.partialOperations.assign(1, rebuilt.getOperation());
    result.partialValues.assign(rebuilt->getResults().begin(),
                                rebuilt->getResults().end());
    llvm::SmallVector<mlir::Operation *, 4> liveSlices;
    for (mlir::Operation *slice : result.generatedSlices) {
      if (slice->use_empty()) {
        slice->erase();
        continue;
      }
      liveSlices.push_back(slice);
    }
    result.generatedSlices = std::move(liveSlices);
  }

  mlir::FailureOr<mlir::MergeResult> merged = interface.mergeReductions(
      builder, loc, result.partialValues, result.reductionDimensions);
  if (mlir::failed(merged) || merged->mergeOps.empty() ||
      merged->replacements.empty()) {
    interfaceOperation->erase();
    setFailureReason(
        failureReason,
        "PartialReductionOpInterface failed to merge partial values");
    return mlir::failure();
  }
  // The official Linalg interface currently emits named linalg.reduce merge
  // operations. The existing Wafer structured-to-tile boundary consumes the
  // equivalent generic form, so generalize the actual merge in place and keep
  // its replacement SSA values. The interface result is not discarded.
  mlir::IRRewriter rewriter(builder);
  for (mlir::Operation *mergeOperation : merged->mergeOps) {
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(mergeOperation);
    if (!linalg) {
      interfaceOperation->erase();
      setFailureReason(
          failureReason,
          "PartialReductionOpInterface produced a non-linalg merge");
      return mlir::failure();
    }
    if (auto generic =
            mlir::dyn_cast<mlir::linalg::GenericOp>(mergeOperation)) {
      result.mergeOperations.push_back(generic);
      result.mergedValues.append(generic->getResults().begin(),
                                 generic->getResults().end());
      continue;
    }
    rewriter.setInsertionPoint(mergeOperation);
    mlir::FailureOr<mlir::linalg::GenericOp> generic =
        mlir::linalg::generalizeNamedOp(rewriter, linalg);
    if (mlir::failed(generic)) {
      interfaceOperation->erase();
      setFailureReason(
          failureReason,
          "partial-reduction merge cannot be represented as linalg.generic");
      return mlir::failure();
    }
    result.mergeOperations.push_back(generic->getOperation());
    result.mergedValues.append(generic->getResults().begin(),
                               generic->getResults().end());
  }
  if (result.mergedValues.size() != merged->replacements.size()) {
    interfaceOperation->erase();
    setFailureReason(
        failureReason,
        "partial-reduction merge result count changed during generalization");
    return mlir::failure();
  }
  interfaceOperation->erase();
  return result;
}
