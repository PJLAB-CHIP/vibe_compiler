//===- DependentDataflow.cpp - Selected edge-action lowering ------------===//

#include "SelectedEdgeLoweringInternal.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"

#include <iterator>
#include <utility>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

namespace {

mlir::FailureOr<mlir::presburger::PresburgerSet>
getRectangleSet(llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
                std::string *failureReason) {
  analysis::IndexSetResult rectangle =
      analysis::IndexRelation::staticRectangularDomain(offsets, sizes);
  if (!rectangle.isExact()) {
    setFailureReason(failureReason, rectangle.reason);
    return mlir::failure();
  }
  return std::move(*rectangle.set);
}

mlir::LogicalResult
validateStrategyDemand(const SpatialEdgeStrategy &strategy,
                       const analysis::ExactIndexSet &requiredDomain,
                       std::string *failureReason) {
  const analysis::IndexRelationLimits limits;
  const auto &required = requiredDomain.getPresburgerSet();
  if (required.getNumDisjuncts() > limits.maxDisjuncts) {
    setFailureReason(failureReason,
                     "selected carrier demand exceeds comparison work limit");
    return mlir::failure();
  }

  std::optional<mlir::presburger::PresburgerSet> covered;
  auto addPiece = [&](llvm::ArrayRef<int64_t> offsets,
                      llvm::ArrayRef<int64_t> sizes,
                      bool peerFragment) -> mlir::LogicalResult {
    mlir::FailureOr<mlir::presburger::PresburgerSet> piece =
        getRectangleSet(offsets, sizes, failureReason);
    if (mlir::failed(piece)) {
      if (peerFragment)
        setFailureReason(
            failureReason,
            "dependent fragment extends outside its consumer demand");
      return mlir::failure();
    }
    if (piece->getNumDisjuncts() + required.getNumDisjuncts() >
        limits.maxDisjuncts) {
      setFailureReason(failureReason,
                       "selected carrier demand exceeds comparison work limit");
      return mlir::failure();
    }
    if (peerFragment && !piece->intersect(required).isEqual(*piece)) {
      setFailureReason(
          failureReason,
          "dependent fragment extends outside its consumer demand");
      return mlir::failure();
    }
    if (covered && !covered->intersect(*piece).isIntegerEmpty()) {
      setFailureReason(failureReason,
                       peerFragment
                           ? "dependent fragments overlap within one consumer "
                             "demand"
                           : "selected producer domain overlaps itself");
      return mlir::failure();
    }
    covered = covered ? covered->unionSet(*piece) : std::move(*piece);
    return mlir::success();
  };

  if (strategy.action == SpatialEdgeAction::PeerFragments) {
    if (strategy.fragments.size() > limits.maxRectangularPieces) {
      setFailureReason(failureReason,
                       "selected carrier fragment count exceeds work limit");
      return mlir::failure();
    }
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (mlir::failed(addPiece(fragment.offsets, fragment.sizes,
                                /*peerFragment=*/true)))
        return mlir::failure();
  } else if (mlir::failed(addPiece(strategy.producerOffsets,
                                   strategy.producerSizes,
                                   /*peerFragment=*/false))) {
    return mlir::failure();
  }
  if (!covered) {
    setFailureReason(failureReason,
                     "selected physical carrier has no demand domain");
    return mlir::failure();
  }
  if (!covered->isEqual(required)) {
    setFailureReason(
        failureReason,
        strategy.action == SpatialEdgeAction::PeerFragments
            ? "dependent fragments do not exactly cover the consumer demand"
            : "selected producer domain differs from the exact relation image");
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult
wafer::tensor_program_to_tile_region::reportSelectedEdgeFailure(
    std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
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

mlir::FailureOr<llvm::SmallVector<SpatialEdgeMaterializationFacts, 16>>
wafer::deriveSpatialEdgeMaterializationFacts(
    mlir::Block &sourceBody, llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    std::string *failureReason) {
  llvm::SmallVector<SpatialEdgeMaterializationFacts, 16> facts;
  facts.reserve(edgeStrategies.size());
  for (const SpatialEdgeStrategy &strategy : edgeStrategies) {
    auto consumerPosition =
        llvm::find_if(sourceBody, [&](mlir::Operation &operation) {
          return &operation == strategy.consumer;
        });
    if (consumerPosition == sourceBody.end()) {
      setFailureReason(failureReason,
                       "selected edge consumer is outside source body");
      return mlir::failure();
    }
    auto dependency = llvm::find_if(
        operandDemands, [&](const analysis::DependencyDemand &candidate) {
          return candidate.consumerOperation == strategy.consumer &&
                 candidate.consumerOperand == strategy.consumerOperand;
        });
    if (dependency == operandDemands.end()) {
      setFailureReason(failureReason,
                       "selected edge has no grouped exact-demand dependency");
      return mlir::failure();
    }
    auto destination = llvm::find_if(
        dependency->perDestination,
        [&](const analysis::DestinationDemand &candidate) {
          return candidate.destinationTile == strategy.destinationTile;
        });
    if (destination == dependency->perDestination.end()) {
      setFailureReason(failureReason,
                       "selected edge has no destination demand recipe");
      return mlir::failure();
    }
    auto source = llvm::find_if(
        destination->sources, [&](const analysis::SourceDemand &candidate) {
          const auto *structured =
              std::get_if<analysis::StructuredResultSource>(&candidate.source);
          return structured && structured->operation == strategy.producer &&
                 structured->result == strategy.producerResult;
        });
    if (source == destination->sources.end() ||
        source->requiredDomain.isEmpty()) {
      setFailureReason(
          failureReason,
          "selected edge has no nonempty structured source demand");
      return mlir::failure();
    }
    if (mlir::failed(validateStrategyDemand(strategy, source->requiredDomain,
                                            failureReason)))
      return mlir::failure();
    facts.push_back(SpatialEdgeMaterializationFacts{
        /*requiresConsumerInputReconstruction=*/
        !destination->reconstruction.steps.empty(),
        /*consumerScheduleOrdinal=*/
        static_cast<uint64_t>(
            std::distance(sourceBody.begin(), consumerPosition))});
  }
  return facts;
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
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
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
