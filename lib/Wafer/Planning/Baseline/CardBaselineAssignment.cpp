//===- CardBaselineAssignment.cpp ------------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"
#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"
#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

namespace wafer::compiler::detail {
namespace {

std::string getOutcomeDetail(const analysis::ExactDemandOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
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
            "baseline output requires one structured semantic root";
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
          *failureReason = "baseline output owner is not one finite rectangle";
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

mlir::FailureOr<CardBaselineAssignment>
computeCardBaselineAssignment(const CardProgramAnalysis &program, CardId cardId,
                              BaselineStatistics *statistics,
                              llvm::raw_ostream &diagnostics) {
  (void)cardId;
  std::string failureReason;
  mlir::FailureOr<CanonicalSpatialCoordinate> coordinate =
      buildCanonicalSpatialAssignment(program.dag, program.availableTileIds,
                                      &failureReason);
  if (mlir::failed(coordinate)) {
    diagnostics << "wafer-compile: canonical spatial assignment failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  mlir::FailureOr<DemandPlanningSession> session =
      DemandPlanningSession::create(
          program.dag, analysis::IndexRelationLimits(), &failureReason);
  if (mlir::failed(session)) {
    diagnostics << "wafer-compile: exact demand facts failed: " << failureReason
                << '\n';
    return mlir::failure();
  }
  analysis::ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(outcome);
  if (!proof) {
    diagnostics << "wafer-compile: exact demand failed: "
                << getOutcomeDetail(outcome) << '\n';
    return mlir::failure();
  }
  session->close();

  CardBaselineAssignment assignment;
  assignment.spatial = std::move(coordinate->assignment);
  assignment.demand = *proof;
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
    diagnostics << "wafer-compile: baseline output mapping failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  assignment.mapping.outputs = std::move(*outputs);
  assignment.mapping.materializationMode =
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  if (statistics) {
    ++statistics->spatialCoordinateQueries;
    statistics->exactDemandSatisfiedEdges = program.dag.getEdges().size();
  }
  if (mlir::failed(
          setCardBaselineTemporalTiles(assignment, program, &failureReason)) ||
      mlir::failed(addCardBaselineDataMovement(assignment, program.dag,
                                               &failureReason))) {
    diagnostics << "wafer-compile: baseline assignment failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  return assignment;
}

mlir::FailureOr<CardBaselineAssignment>
buildCardBaselineMaterializationAssignment(const CardProgramAnalysis &program,
                                           const CanonicalBaselinePlan &plan,
                                           BaselineStatistics *statistics,
                                           llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  CardBaselineAssignment assignment;
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
    diagnostics << "wafer-compile: baseline output mapping failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  assignment.mapping.outputs = std::move(*outputs);
  assignment.mapping.materializationMode =
      SpatialDataflowMaterializationMode::IndependentDDRStages;

  std::map<mlir::Operation *, llvm::SmallVector<int64_t, 4>> temporalByRoot;
  for (const TemporalScopePlan &scope : plan.temporal.scopes) {
    const auto *root =
        std::get_if<RequiredRootExecution>(&scope.execution.source);
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
  if (mlir::failed(addCardBaselineDataMovement(assignment, program.dag,
                                               &failureReason))) {
    diagnostics << "wafer-compile: baseline movement projection failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics) {
    ++statistics->spatialCoordinateQueries;
    statistics->exactDemandSatisfiedEdges = program.dag.getEdges().size();
  }
  return assignment;
}

} // namespace wafer::compiler::detail
