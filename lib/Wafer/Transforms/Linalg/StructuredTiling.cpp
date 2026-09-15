//===- StructuredTiling.cpp - Interface-driven tile traversal ----------===//

#include "Wafer/Transforms/Linalg/StructuredTiling.h"
#include "Wafer/Transforms/Linalg/ContractionAccumulation.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

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

static mlir::FailureOr<mlir::Operation *>
getReductionCombiner(mlir::linalg::LinalgOp operation, unsigned resultNumber,
                     std::string *failureReason) {
  if (!operation || resultNumber >= operation.getNumDpsInits()) {
    setFailureReason(failureReason,
                     "local reduction requires a Linalg destination");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Operation *, 2> combiners;
  auto accumulators = operation.getRegionOutputArgs();
  if (!mlir::matchReduction(accumulators, resultNumber, combiners) ||
      combiners.size() != 1 || combiners[0]->getNumOperands() != 2 ||
      combiners[0]->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "local reduction requires one binary scalar combiner");
    return mlir::failure();
  }
  auto *combiner = combiners[0];
  if ((combiner->getOperand(0) == accumulators[resultNumber]) ==
      (combiner->getOperand(1) == accumulators[resultNumber])) {
    setFailureReason(failureReason,
                     "reduction combiner must consume its accumulator once");
    return mlir::failure();
  }
  return combiner;
}

mlir::FailureOr<mlir::Value> wafer::combineReductionPartial(
    mlir::Operation *reduction, unsigned resultNumber, mlir::Value partial,
    mlir::Value destination, mlir::OpBuilder &builder,
    std::string *failureReason) {
  auto operation = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(reduction);
  auto combiner = getReductionCombiner(operation, resultNumber, failureReason);
  auto type = partial
                  ? mlir::dyn_cast<mlir::RankedTensorType>(partial.getType())
                  : mlir::RankedTensorType{};
  if (mlir::failed(combiner) || !type || !destination ||
      destination.getType() != type) {
    setFailureReason(
        failureReason,
        "partial and merge destination must have the same tensor type");
    return mlir::failure();
  }
  auto identity = builder.getMultiDimIdentityMap(type.getRank());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      type.getRank(), mlir::utils::IteratorType::parallel);
  auto merge = builder.create<mlir::linalg::GenericOp>(
      reduction->getLoc(), mlir::TypeRange{type}, mlir::ValueRange{partial},
      mlir::ValueRange{destination},
      llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location loc, mlir::ValueRange args) {
        mlir::IRMapping mapping;
        auto accumulator = operation.getRegionOutputArgs()[resultNumber];
        for (mlir::Value operand : (*combiner)->getOperands())
          mapping.map(operand, operand == accumulator ? args[1] : args[0]);
        if (requiresWideContractionState(operation) &&
            type.getElementType().isF32()) {
          auto add = mlir::cast<mlir::arith::AddFOp>(*combiner);
          auto combined = nested.create<mlir::arith::AddFOp>(
              loc, mapping.lookup(add.getLhs()), mapping.lookup(add.getRhs()),
              add.getFastmathAttr());
          nested.create<mlir::linalg::YieldOp>(loc, combined.getResult());
        } else {
          auto *combined = nested.clone(**combiner, mapping);
          nested.create<mlir::linalg::YieldOp>(loc, combined->getResult(0));
        }
      });
  return merge.getResult(0);
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
  auto operation = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(reduction);
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(reduction);
  if (!operation || !tiling || !operation.hasPureTensorSemantics() ||
      operation.getNumDpsInits() != reduction->getNumResults() ||
      iterationOffsets.size() != operation.getNumLoops() ||
      iterationSizes.size() != operation.getNumLoops() ||
      (!requestedResultTileDestinations.empty() &&
       requestedResultTileDestinations.size() !=
           static_cast<size_t>(operation.getNumDpsInits()))) {
    setFailureReason(failureReason, "local partial requires a tensor Linalg "
                                    "reduction and an exact iteration tile");
    return mlir::failure();
  }
  PartialReductionTileMaterialization result;
  for (auto [dimension, kind] :
       llvm::enumerate(operation.getIteratorTypesArray()))
    if (kind == mlir::utils::IteratorType::reduction)
      result.reductionDimensions.push_back(dimension);
  if (result.reductionDimensions.empty()) {
    setFailureReason(
        failureReason,
        "partial-reduction tiling requires at least one reduction dimension");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::TypedAttr, 2> identities;
  for (unsigned index = 0; index != operation.getNumDpsInits(); ++index) {
    auto combiner = getReductionCombiner(operation, index, failureReason);
    auto identity = mlir::succeeded(combiner)
                        ? mlir::arith::getNeutralElement(*combiner)
                        : std::optional<mlir::TypedAttr>{};
    if (!identity) {
      setFailureReason(failureReason,
                       "local reduction combiner has no neutral element");
      return mlir::failure();
    }
    identities.push_back(*identity);
  }
  auto tiled = materializeOperationFromIterationTile(
      reduction, builder, iterationOffsets, iterationSizes, failureReason);
  if (mlir::failed(tiled) || tiled->tiledOperations.size() != 1)
    return mlir::failure();
  auto local =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(tiled->tiledOperations[0]);
  if (!local ||
      static_cast<size_t>(local.getNumDpsInits()) != identities.size()) {
    setFailureReason(failureReason,
                     "tiled reduction omitted its local Linalg destinations");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Value, 2> destinations;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(local);
    for (int64_t index = 0; index != local.getNumDpsInits(); ++index) {
      auto *init = local.getDpsInitOperand(index);
      auto type = mlir::cast<mlir::RankedTensorType>(init->get().getType());
      mlir::Value destination = requestedResultTileDestinations.empty()
                                    ? init->get()
                                    : requestedResultTileDestinations[index];
      if (destination.getType() != type) {
        setFailureReason(
            failureReason,
            "partial-reduction destination type does not match result tile");
        return mlir::failure();
      }
      destinations.push_back(destination);
      auto sizes = mlir::tensor::getMixedSizes(builder, reduction->getLoc(),
                                               init->get());
      mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
          reduction->getLoc(), sizes, type.getElementType());
      mlir::Value scalar = builder.create<mlir::arith::ConstantOp>(
          reduction->getLoc(), identities[index]);
      mlir::Value identity =
          builder
              .create<mlir::linalg::FillOp>(reduction->getLoc(), scalar, empty)
              .getResult(0);
      result.initialValues.push_back(identity);
      init->set(identity);
    }
  }
  result.partialOperations = tiled->tiledOperations;
  result.partialValues = tiled->tiledValues;
  result.generatedSlices = tiled->generatedSlices;
  builder.setInsertionPointAfter(local);
  for (auto [index, value] : llvm::enumerate(result.partialValues)) {
    auto merged = combineReductionPartial(
        reduction, index, value, destinations[index], builder, failureReason);
    if (mlir::failed(merged))
      return mlir::failure();
    result.mergeOperations.push_back(merged->getDefiningOp());
    result.mergedValues.push_back(*merged);
  }
  return result;
}
