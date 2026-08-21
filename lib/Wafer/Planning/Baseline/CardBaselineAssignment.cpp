//===- CardBaselineAssignment.cpp ------------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"
#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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
        placement.iteratorPartitionFactors[iterator] =
            std::max<uint32_t>(placement.iteratorPartitionFactors[iterator],
                               coordinate + 1);
    }
    placements.push_back(std::move(placement));
  }
  return placements;
}

mlir::FailureOr<llvm::SmallVector<OutputTileMapping, 4>> getOutputMappings(
    const CardProgramAnalysis &program,
    const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand,
    std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view = StructuredDemandView::create(
      program.dag, spatial, demand, failureReason);
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
          *failureReason =
              "baseline output owner is not one finite rectangle";
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
      DemandPlanningSession::create(program.dag, analysis::IndexRelationLimits(),
                                    &failureReason);
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
  mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 16>> placements =
      getNodePlacements(program.dag, assignment.spatial, assignment.demand,
                        &failureReason);
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
  if (mlir::failed(setCardBaselineTemporalTiles(assignment, program,
                                                &failureReason)) ||
      mlir::failed(addCardBaselineDataMovement(
          assignment, program.dag, &failureReason))) {
    diagnostics << "wafer-compile: baseline assignment failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  return assignment;
}

} // namespace wafer::compiler::detail
