//===- DependentDataflow.cpp - Selected edge-action lowering ------------===//

#include "EdgeDomain.h"
#include "SelectedEdgeLoweringInternal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include <utility>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

mlir::LogicalResult
wafer::tensor_program_to_tile_region::reportSelectedEdgeFailure(
    std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

mlir::LogicalResult wafer::deriveSpatialEdgeConsumerResultDomain(
    const SpatialEdgeStrategy &strategy,
    llvm::SmallVectorImpl<int64_t> &consumerOffsets,
    llvm::SmallVectorImpl<int64_t> &consumerSizes, std::string *failureReason) {
  consumerOffsets.clear();
  consumerSizes.clear();
  return deriveConsumerDomainFromProducerDemand(
      strategy.producer, strategy.producerResult, strategy.consumer,
      strategy.consumerOperand, strategy.producerOffsets,
      strategy.producerSizes, consumerOffsets, consumerSizes, failureReason);
}

bool wafer::isSpatialEdgeStrategyIncidentOnTile(
    const SpatialEdgeStrategy &strategy, TileId tile) {
  if (strategy.destinationTile == tile)
    return true;
  if (strategy.action != SpatialEdgeAction::PeerFragments)
    return false;
  return llvm::any_of(strategy.fragments, [&](const SpatialEdgeFragment &item) {
    return item.kind == SpatialEdgeFragmentKind::Peer &&
           item.sourceTile == tile;
  });
}

mlir::LogicalResult wafer::lowerSpatialEdgeStrategiesToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    bool requireOneStructuredRootPerRegion) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || currentLogicalPartition < 0)
    return reportSelectedEdgeFailure(
        failureReason, "edge-action lowering requires a source module and "
                       "logical card partition");
  if (!edgeFacts.empty() && edgeFacts.size() != edgeStrategies.size())
    return reportSelectedEdgeFailure(
        failureReason,
        "edge materialization facts must cover every selected edge");

  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone(cloneMapping));
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(*candidate);
  mlir::func::FuncOp sourceFunction =
      findSingleStandaloneTensorProgram(sourceModule);
  if (!function || !sourceFunction || sourceFunction.isExternal() ||
      !sourceFunction.getBody().hasOneBlock())
    return reportSelectedEdgeFailure(
        failureReason,
        "edge-action lowering requires one pristine support template body");
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
  TensorProgramScope scope(function, functionalArgumentCount);
  if (mlir::failed(verifyTensorProgramScope(function, functionalArgumentCount,
                                            failureReason,
                                            scope.getBoundaryArgumentCount())))
    return mlir::failure();
  mlir::Block &sourceBody = sourceFunction.getBody().front();

  mlir::FailureOr<SelectedEdgeProgramMapping> mappedEdges =
      mapSelectedEdgesToCandidate(sourceBody, cloneMapping, scope, currentTile,
                                  materializationMode, edgeStrategies,
                                  operationTemporalTiles, operationNodes,
                                  edgeFacts, operandDemands, failureReason);
  if (mlir::failed(mappedEdges))
    return mlir::failure();
  SelectedEdgeLoweringState state(
      sourceModule, functionalArgumentCount, outputShards, currentTile,
      currentLogicalPartition, requireOneStructuredRootPerRegion,
      operationNodes, std::move(candidate), function, std::move(*mappedEdges),
      failureReason);
  if (mlir::failed(materializeSelectedDirectActions(state)) ||
      mlir::failed(materializeSelectedPeerReceives(state)) ||
      mlir::failed(verifySelectedReceiveOwners(state)) ||
      mlir::failed(materializeSelectedSourceStages(state)) ||
      mlir::failed(materializeSelectedConsumerStages(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "internal consumer materialization")) ||
      mlir::failed(materializeSelectedPeerSends(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "outgoing peer materialization")) ||
      mlir::failed(materializeSelectedOutputs(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "output traversal materialization")) ||
      mlir::failed(verifySelectedReceiveExecutionSinks(state)))
    return mlir::failure();
  return finishSelectedEdgeLowering(state, module, materializationRelations);
}
