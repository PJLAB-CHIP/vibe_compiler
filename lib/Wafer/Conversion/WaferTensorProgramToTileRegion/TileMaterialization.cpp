//===- TileMaterialization.cpp - Candidate root tile materialization -===//

#include "Internal.h"
#include "StructuredIterationTile.h"

#include <optional>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::Operation *root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one result");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  return materializeCandidateRootTileValue(
      builder, scope, root, outputIndex, mixedOffsets, candidateTileSizes,
      loops, operationTemporalTiles, failureReason, operationNodes);
}

mlir::FailureOr<mlir::Value> materializeCandidateOperandConsumerTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  if (!root || root->getNumResults() != 1 ||
      !mlir::isa<mlir::TilingInterface>(root)) {
    setFailureReason(
        failureReason,
        "operand-driven candidate requires one tiled consumer result");
    return mlir::failure();
  }

  mlir::FailureOr<OperandTileMaterialization> materialized =
      materializeConsumerFromOperandTile(root, builder, operandNumber,
                                         operandTileOffsets, operandTileSizes,
                                         failureReason);
  if (mlir::failed(materialized))
    return mlir::failure();
  if (materialized->tiledOperations.size() != 1 ||
      materialized->tiledValues.size() != 1 ||
      materialized->tiledOperations.front()->getNumResults() != 1 ||
      materialized->tiledOperations.front()->getResult(0) !=
          materialized->tiledValues.front()) {
    setFailureReason(
        failureReason,
        "operand-driven candidate must materialize one consumer result");
    return mlir::failure();
  }

  auto tiling = mlir::cast<mlir::TilingInterface>(root);
  llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultSizes;
  if (mlir::failed(tiling.getResultTilePosition(
          builder, /*resultNumber=*/0, materialized->iterationDomain.offsets,
          materialized->iterationDomain.sizes, resultOffsets, resultSizes))) {
    setFailureReason(
        failureReason,
        "operand-driven candidate cannot map its consumer result tile");
    return mlir::failure();
  }

  // A boundary-seeded complete traversal may carry a single output only when
  // every seed tile maps one-to-one onto the same result tile. More general
  // broadcast/permutation cover needs an independent exact-cover proof; it is
  // deliberately rejected here instead of assuming non-overlap.
  if (!llvm::equal(resultOffsets, operandTileOffsets) ||
      !llvm::equal(resultSizes, operandTileSizes)) {
    setFailureReason(
        failureReason,
        "operand-driven candidate requires an exact one-to-one result tile "
        "relation");
    return mlir::failure();
  }

  mlir::Operation *tiledConsumer = materialized->tiledOperations.front();
  if (mlir::failed(fuseCandidateProducerSlices(
          tiledConsumer, root, scope, loops,
          /*operationTemporalTiles=*/{}, /*nestedTemporalTiles=*/{},
          builder.getListener(), failureReason)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledConsumer);
  return materialized->tiledValues.front();
}

static mlir::FailureOr<mlir::linalg::LinalgOp>
getCandidateReductionComputeRoot(mlir::Operation *root,
                                 std::string *failureReason) {
  if (auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(root))
    return linalg;

  auto allReduce = mlir::dyn_cast_or_null<LinalgExtCollectiveAllReduceOp>(root);
  if (!allReduce || allReduce.getInputs().size() != 1 ||
      allReduce.getOuts().size() != 1 || root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "reduction traversal requires a direct reduction or one "
                     "typed all-reduce wrapper");
    return mlir::failure();
  }

  mlir::Value input = allReduce.getInputs().front();
  auto producer =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(input.getDefiningOp());
  if (!producer || producer->getBlock() != root->getBlock() ||
      producer->getNumResults() != 1 || producer->getResult(0) != input ||
      !input.hasOneUse()) {
    setFailureReason(failureReason,
                     "reduction typed all-reduce requires one exact single-use "
                     "same-block Linalg producer");
    return mlir::failure();
  }
  return producer;
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileIntoDestination(
    TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::Value destination, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (!root || root->getNumResults() != 1 || !destination) {
    setFailureReason(
        failureReason,
        "destination-backed candidate traversal requires one result");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || destination.getType() != resultType ||
      mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();
  mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
      getCandidateReductionComputeRoot(root, failureReason);
  if (mlir::failed(computeRoot))
    return mlir::failure();
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  return materializeConfiguredStructuredTraversal(
      builder, scope, root, *computeRoot, mixedOffsets, candidateTileSizes,
      loops, operationTemporalTiles, /*nestedTemporalTiles=*/{}, failureReason,
      destination, mixedOffsets, operationNodes);
}

static mlir::FailureOr<mlir::Value>
materializeConfiguredStructuredRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
      getCandidateReductionComputeRoot(root, failureReason);
  if (mlir::failed(computeRoot))
    return mlir::failure();
  return materializeConfiguredStructuredTraversal(
      builder, scope, root, *computeRoot, candidateTileOffsets,
      candidateTileSizes, loops, operationTemporalTiles,
      /*nestedTemporalTiles=*/{}, failureReason, /*outputDestination=*/{},
      /*destinationBaseOffsets=*/{}, operationNodes);
}

static mlir::FailureOr<mlir::Value> materializeCandidateInterfaceRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  if (!dps || !tiling || dps.getNumDpsInits() != 1 ||
      root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate interface root requires one DPS output and "
                     "TilingInterface");
    return mlir::failure();
  }
  (void)scope;
  (void)outputIndex;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || candidateTileOffsets.size() != candidateTileSizes.size() ||
      candidateTileSizes.size() != static_cast<size_t>(resultType.getRank()) ||
      llvm::any_of(candidateTileSizes,
                   [](int64_t size) { return size <= 0; })) {
    setFailureReason(failureReason,
                     "candidate interface tile rank or size mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  mixedSizes.reserve(candidateTileSizes.size());
  for (int64_t size : candidateTileSizes)
    mixedSizes.push_back(builder.getIndexAttr(size));

  // Candidate coordinates are expressed in the result domain. Recover an
  // exact iteration-domain tile through TilingInterface result/operand
  // relations and verify the round trip before materialization. Passing
  // result offsets directly to getTiledImplementation without that proof
  // would silently assume an identity indexing map.
  std::optional<OperandTileIterationDomain> iteration;
  auto hasExactResultTile = [&](llvm::ArrayRef<mlir::OpFoldResult> offsets,
                                llvm::ArrayRef<mlir::OpFoldResult> sizes) {
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    return mlir::succeeded(tiling.getResultTilePosition(
               builder, /*resultNumber=*/0, offsets, sizes, resultOffsets,
               resultSizes)) &&
           llvm::equal(resultOffsets, candidateTileOffsets) &&
           llvm::equal(resultSizes, mixedSizes);
  };
  auto retainIteration = [&](llvm::ArrayRef<mlir::OpFoldResult> offsets,
                             llvm::ArrayRef<mlir::OpFoldResult> sizes) {
    if (!hasExactResultTile(offsets, sizes))
      return false;
    iteration.emplace();
    iteration->offsets.assign(offsets.begin(), offsets.end());
    iteration->sizes.assign(sizes.begin(), sizes.end());
    return true;
  };

  // Prefer the interface's explicit result-to-iteration relation. The
  // verified round-trip fallback admits identity iteration/result domains
  // exposed by simpler TilingInterface implementations without assuming that
  // every equal-rank operation is identity-mapped.
  llvm::SmallVector<mlir::OpFoldResult> resultIterationOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultIterationSizes;
  if (mlir::succeeded(tiling.getIterationDomainTileFromResultTile(
          builder, /*resultNumber=*/0, candidateTileOffsets, mixedSizes,
          resultIterationOffsets, resultIterationSizes)))
    (void)retainIteration(resultIterationOffsets, resultIterationSizes);
  if (!iteration &&
      candidateTileOffsets.size() == tiling.getLoopIteratorTypes().size())
    (void)retainIteration(candidateTileOffsets, mixedSizes);

  llvm::SmallVector<unsigned, 4> relationOperands;
  if (mlir::OpOperand *resultDestination = dps.getDpsInitOperand(0))
    relationOperands.push_back(resultDestination->getOperandNumber());
  for (unsigned operandNumber = 0; operandNumber < root->getNumOperands();
       ++operandNumber)
    if (!llvm::is_contained(relationOperands, operandNumber))
      relationOperands.push_back(operandNumber);
  for (unsigned operandNumber : relationOperands) {
    if (iteration)
      break;
    std::string relationFailure;
    mlir::FailureOr<OperandTileIterationDomain> candidateIteration =
        mapOperandTileToIterationDomain(root, builder, operandNumber,
                                        candidateTileOffsets, mixedSizes,
                                        &relationFailure);
    if (mlir::failed(candidateIteration))
      continue;
    (void)retainIteration(candidateIteration->offsets,
                          candidateIteration->sizes);
  }
  if (!iteration) {
    setFailureReason(
        failureReason,
        "TilingInterface provides no operand relation that exactly covers "
        "the requested result tile");
    return mlir::failure();
  }

  mlir::FailureOr<StructuredIterationTile> tiled =
      materializeStructuredIterationTile(root, builder, iteration->offsets,
                                         iteration->sizes, failureReason);
  if (mlir::failed(tiled))
    return mlir::failure();
  if (tiled->operations.size() != 1 || tiled->values.size() != 1 ||
      tiled->operations.front()->getNumResults() != 1 ||
      tiled->operations.front()->getResult(0) != tiled->values.front()) {
    setFailureReason(
        failureReason,
        "candidate interface root must materialize one tiled op and result");
    return mlir::failure();
  }

  mlir::Operation *tiledRoot = tiled->operations.front();
  recordStructuredOperationNodeMaterialization(root, tiledRoot, operationNodes);
  if (mlir::failed(fuseCandidateProducerSlices(
          tiledRoot, root, scope, loops, operationTemporalTiles,
          /*nestedTemporalTiles=*/{}, builder.getListener(), failureReason,
          operationNodes)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledRoot);
  return tiled->values.front();
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (classifyStructuredRoot(root) != StructuredRootCapability::Tiled) {
    setFailureReason(failureReason,
                     "candidate traversal root does not support tiling");
    return mlir::failure();
  }
  mlir::linalg::LinalgOp reductionCompute =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(root);
  if (!reductionCompute && mlir::isa<LinalgExtCollectiveAllReduceOp>(root)) {
    mlir::FailureOr<mlir::linalg::LinalgOp> resolved =
        getCandidateReductionComputeRoot(root, failureReason);
    if (mlir::failed(resolved))
      return mlir::failure();
    reductionCompute = *resolved;
  }
  if (reductionCompute) {
    auto selected = llvm::find_if(
        operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
          return tile.operation == reductionCompute.getOperation();
        });
    if (selected != operationTemporalTiles.end())
      return materializeConfiguredStructuredRootTileValue(
          builder, scope, root, candidateTileOffsets, candidateTileSizes, loops,
          operationTemporalTiles, failureReason, operationNodes);
  }
  return materializeCandidateInterfaceRootTileValue(
      builder, scope, root, outputIndex, candidateTileOffsets,
      candidateTileSizes, loops, operationTemporalTiles, failureReason,
      operationNodes);
}

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(TensorProgramScope scope, unsigned outputIndex,
                           std::string *failureReason) {
  unsigned inputCount = scope.getInputCount();
  mlir::Block &body = scope.getBody();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }
  return body.getArgument(inputCount + outputIndex);
}

} // namespace wafer::tensor_program_to_tile_region
