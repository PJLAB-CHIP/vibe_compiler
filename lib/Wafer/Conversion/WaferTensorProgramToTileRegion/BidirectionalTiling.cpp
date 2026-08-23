//===- BidirectionalTiling.cpp - Interface-driven tile traversal ----------===//

#include "Internal.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace wafer;

namespace {

mlir::LogicalResult
verifyOperandTileRequest(mlir::Operation *consumer, unsigned operandNumber,
                         llvm::ArrayRef<mlir::OpFoldResult> offsets,
                         llvm::ArrayRef<mlir::OpFoldResult> sizes,
                         std::string *failureReason) {
  if (!consumer) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason, "operand-driven tiling requires a consumer operation");
    return mlir::failure();
  }
  if (operandNumber >= consumer->getNumOperands()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason, "operand-driven tile operand number is out of range");
    return mlir::failure();
  }
  if (offsets.size() != sizes.size()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason, "operand-driven tile offset/size rank mismatch");
    return mlir::failure();
  }

  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(
      consumer->getOperand(operandNumber).getType());
  if (!shapedType || !shapedType.hasRank() ||
      offsets.size() != static_cast<size_t>(shapedType.getRank())) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "operand-driven tile rank does not match the selected operand");
    return mlir::failure();
  }
  if (!mlir::isa<mlir::TilingInterface>(consumer)) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "operand-driven tile consumer does not implement TilingInterface");
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::FailureOr<wafer::OperandTileIterationDomain>
wafer::mapOperandTileToIterationDomain(
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
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "TilingInterface rejected the operand-to-iteration tile relation");
    return mlir::failure();
  }
  size_t expectedRank = tiling.getLoopIteratorTypes().size();
  if (result.offsets.size() != expectedRank ||
      result.sizes.size() != expectedRank) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "TilingInterface returned an incomplete operand-to-iteration tile "
        "relation");
    return mlir::failure();
  }
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
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "TilingInterface rejected consumer materialization from operand tile");
    return mlir::failure();
  }
  if (tiled->tiledOps.empty() || tiled->tiledValues.empty()) {
    tensor_program_to_tile_region::setFailureReason(
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
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "partial-reduction tiling requires a reduction operation");
    return mlir::failure();
  }

  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(reduction);
  auto partial = mlir::dyn_cast<mlir::PartialReductionOpInterface>(reduction);
  if (!tiling || !partial) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason, "partial-reduction tiling requires TilingInterface and "
                       "PartialReductionOpInterface");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      tiling.getLoopIteratorTypes();
  if (iterationOffsets.size() != iteratorTypes.size() ||
      iterationSizes.size() != iteratorTypes.size()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "partial-reduction tile rank does not match the iteration domain");
    return mlir::failure();
  }

  PartialReductionTileMaterialization result;
  for (auto [dimension, iteratorType] : llvm::enumerate(iteratorTypes))
    if (iteratorType == mlir::utils::IteratorType::reduction)
      result.reductionDimensions.push_back(static_cast<int>(dimension));
  if (result.reductionDimensions.empty()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "partial-reduction tiling requires at least one reduction dimension");
    return mlir::failure();
  }

  mlir::Location loc = reduction->getLoc();
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(reduction);
  if (!dps || dps.getNumDpsInits() != reduction->getNumResults()) {
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "partial-reduction tiling requires one destination per result");
    return mlir::failure();
  }

  if (!requestedResultTileDestinations.empty() &&
      requestedResultTileDestinations.size() != reduction->getNumResults()) {
    tensor_program_to_tile_region::setFailureReason(
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
      tensor_program_to_tile_region::setFailureReason(
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
      tensor_program_to_tile_region::setFailureReason(
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
        tensor_program_to_tile_region::setFailureReason(
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
    tensor_program_to_tile_region::setFailureReason(
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
    tensor_program_to_tile_region::setFailureReason(
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

  mlir::FailureOr<mlir::MergeResult> merged = interface.mergeReductions(
      builder, loc, result.partialValues, result.reductionDimensions);
  if (mlir::failed(merged) || merged->mergeOps.empty() ||
      merged->replacements.empty()) {
    interfaceOperation->erase();
    tensor_program_to_tile_region::setFailureReason(
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
      tensor_program_to_tile_region::setFailureReason(
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
      tensor_program_to_tile_region::setFailureReason(
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
    tensor_program_to_tile_region::setFailureReason(
        failureReason,
        "partial-reduction merge result count changed during generalization");
    return mlir::failure();
  }
  interfaceOperation->erase();
  return result;
}
