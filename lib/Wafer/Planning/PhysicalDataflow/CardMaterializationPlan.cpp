//===- CardMaterializationPlan.cpp - Selected Card construction -------===//

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

#include "Wafer/Planning/PhysicalDataflow/CardDataflowConstruction.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

namespace wafer::compiler::detail {
namespace {

llvm::StringRef stringifyEdgeAction(SpatialEdgeAction action) {
  switch (action) {
  case SpatialEdgeAction::RecursiveProducerTiling:
    return "recursive-producer-tiling";
  case SpatialEdgeAction::LocalShardResidency:
    return "local-shard-residency";
  case SpatialEdgeAction::PeerFragments:
    return "peer-fragments";
  case SpatialEdgeAction::SpillReload:
    return "spill-reload";
  case SpatialEdgeAction::Recompute:
    return "recompute";
  case SpatialEdgeAction::RegionCut:
    return "region-cut";
  case SpatialEdgeAction::CardDDRTransfer:
    return "card-ddr-transfer";
  }
  llvm_unreachable("unknown spatial edge action");
}

mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 16>>
getNodePlacements(const StructuredDAGAnalysis &dag,
                  const SpatialAssignment &spatial,
                  const analysis::ExactDemandProof &demand,
                  std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    const NodeExecutionPartition *partition = view->getNode(node.id);
    if (!partition || partition->shards.empty())
      return mlir::failure();
    StructuredDAGNodePlacement placement;
    placement.node = node.id;
    placement.iteratorPartitionFactors.assign(
        partition->shards.front().shard.coordinate.size(), 0);
    for (const ExecutionShard &shard : partition->shards) {
      placement.tiles.push_back(shard.tile);
      for (auto [iterator, coordinate] :
           llvm::enumerate(shard.shard.coordinate))
        placement.iteratorPartitionFactors[iterator] = std::max<uint32_t>(
            placement.iteratorPartitionFactors[iterator], coordinate + 1);
    }
    placements.push_back(std::move(placement));
  }
  return placements;
}

mlir::FailureOr<llvm::SmallVector<OutputTileMapping, 4>> getOutputMappings(
    const CardProgramAnalysis &program, const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand, std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(program.dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  llvm::SmallVector<OutputTileMapping, 4> outputs;
  for (auto [output, roots] :
       llvm::enumerate(program.dag.getObservableOutputRootNodes())) {
    if (roots.size() != 1) {
      if (failureReason)
        *failureReason =
            "candidate output requires one structured semantic root";
      return mlir::failure();
    }
    OutputTileMapping mapping;
    mapping.outputIndex = output;
    llvm::SmallVector<const analysis::FinalResultOwner *, 4> owners =
        view->getFinalOwners(roots.front(), 0);
    if (owners.empty())
      return mlir::failure();
    for (const analysis::FinalResultOwner *owner : owners) {
      if (owner->domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
          owner->domain.getBoxes().size() != 1) {
        if (failureReason)
          *failureReason = "candidate output owner is not one finite rectangle";
        return mlir::failure();
      }
      const analysis::StaticRectangularIndexSet &box =
          owner->domain.getBoxes().front();
      mapping.shards.push_back({owner->tile, box.offsets, box.sizes});
    }
    outputs.push_back(std::move(mapping));
  }
  return outputs;
}

} // namespace

mlir::FailureOr<CardMaterializationPlan>
buildCardMaterializationPlan(const CardProgramAnalysis &program,
                             const CompleteCandidatePlan &plan,
                             SpatialDataflowMaterializationMode mode,
                             CandidateMaterializationStatistics *statistics,
                             llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  CardMaterializationPlan assignment;
  assignment.spatial = plan.spatial;
  assignment.demand = plan.demand;
  mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 16>>
      placements = getNodePlacements(program.dag, assignment.spatial,
                                     assignment.demand, &failureReason);
  if (mlir::failed(placements))
    return mlir::failure();
  assignment.nodePlacements = std::move(*placements);
  mlir::FailureOr<llvm::SmallVector<OutputTileMapping, 4>> outputs =
      getOutputMappings(program, assignment.spatial, assignment.demand,
                        &failureReason);
  if (mlir::failed(outputs)) {
    diagnostics << "wafer-compile: candidate output mapping failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  assignment.mapping.outputs = std::move(*outputs);
  assignment.mapping.materializationMode = mode;

  std::map<mlir::Operation *, llvm::SmallVector<int64_t, 4>> temporalByRoot;
  for (const TemporalScopePlan &scope : plan.temporal.scopes) {
    const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
    const auto *root =
        execution && isTopLevelScope(scope.id)
            ? std::get_if<RequiredRootExecution>(&execution->source)
            : nullptr;
    if (!root)
      continue;
    auto work = llvm::find_if(plan.rootWorks,
                              [&](const analysis::RootRegionWork &candidate) {
                                return candidate.id == root->work;
                              });
    if (work == plan.rootWorks.end() || !work->rootOperation)
      return mlir::failure();
    auto [position, inserted] = temporalByRoot.try_emplace(
        work->rootOperation, scope.iteratorTileSizes);
    if (!inserted) {
      if (position->second.size() != scope.iteratorTileSizes.size())
        return mlir::failure();
      for (auto [current, candidate] :
           llvm::zip_equal(position->second, scope.iteratorTileSizes))
        current = std::max(current, candidate);
    }
  }
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    auto temporal = temporalByRoot.find(node.operation);
    if (temporal == temporalByRoot.end())
      return mlir::failure();
    assignment.mapping.operationTemporalTiles.push_back(
        {node.operation, temporal->second, {}});
    if (wafer::support::getActiveCompileTimingSession()) {
      auto placement = llvm::find_if(
          assignment.nodePlacements, [&](const auto &candidate) {
            return candidate.node == node.id;
          });
      diagnostics << "wafer-compile: ir-selected-node node=" << node.id
                  << " op=" << node.operation->getName()
                  << " spatial_tiles="
                  << (placement == assignment.nodePlacements.end()
                          ? 0
                          : placement->tiles.size())
                  << " loop_ranges=[";
      if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation))
        llvm::interleaveComma(linalg.getStaticLoopRanges(), diagnostics);
      diagnostics << "] temporal_tiles=[";
      llvm::interleaveComma(temporal->second, diagnostics);
      diagnostics << "]\n";
    }
  }

  llvm::ArrayRef<llvm::SmallVector<StructuredDAGNodeID, 2>> outputRoots =
      program.dag.getObservableOutputRootNodes();
  for (OutputTileMapping &output : assignment.mapping.outputs) {
    if (output.outputIndex >= program.outputDomains.size() ||
        output.outputIndex >= outputRoots.size() || output.shards.empty())
      return mlir::failure();
    output.temporalTileSizes.assign(
        program.outputDomains[output.outputIndex].size(), 0);
    for (const OutputTileShard &shard : output.shards)
      for (auto [dimension, size] : llvm::enumerate(shard.sizes))
        output.temporalTileSizes[dimension] =
            std::max(output.temporalTileSizes[dimension], size);
    if (outputRoots[output.outputIndex].size() != 1)
      continue;
    const StructuredDAGNode *root =
        program.dag.getNode(outputRoots[output.outputIndex].front());
    auto temporal =
        root ? temporalByRoot.find(root->operation) : temporalByRoot.end();
    if (!root || temporal == temporalByRoot.end())
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> resultTile =
        getStructuredResultTileShape(root->operation, 0, temporal->second);
    if (!resultTile || resultTile->size() != output.temporalTileSizes.size())
      continue;
    for (auto [outputSize, rootSize] :
         llvm::zip_equal(output.temporalTileSizes, *resultTile))
      outputSize = std::min(outputSize, rootSize);
  }
  if (mlir::failed(addCardDataflowConstruction(
          assignment, program.dag, plan.movement, {}, {}, &failureReason))) {
    diagnostics << "wafer-compile: candidate movement projection failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (wafer::support::getActiveCompileTimingSession()) {
    std::map<SpatialEdgeAction, uint64_t> actions;
    std::map<mlir::Operation *, uint64_t> strategiesByProducer;
    for (const SpatialEdgeStrategy &strategy :
         assignment.mapping.edgeStrategies) {
      ++actions[strategy.action];
      ++strategiesByProducer[strategy.producer];
    }
    for (const auto &[action, count] : actions)
      diagnostics << "wafer-compile: ir-edge-actions action="
                  << stringifyEdgeAction(action) << " count=" << count
                  << '\n';
    for (const StructuredDAGNode &node : program.dag.getNodes())
      diagnostics << "wafer-compile: ir-edge-producer node=" << node.id
                  << " strategies=" << strategiesByProducer[node.operation]
                  << '\n';
  }
  if (statistics) {
    ++statistics->spatialCoordinateQueries;
    statistics->exactDemandSatisfiedEdges = program.dag.getEdges().size();
  }
  return assignment;
}

} // namespace wafer::compiler::detail
