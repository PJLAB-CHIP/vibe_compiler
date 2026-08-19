//===- CardBaselineAssignment.cpp ------------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"
#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"
#include "Wafer/Planning/Baseline/CardBaselinePlacement.h"
#include "Wafer/Planning/Baseline/CardBaselineTemporalTiling.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 16>>
makeInitialPlacements(const CardProgramAnalysis &program,
                      std::string *failureReason) {
  if (program.availableTileIds.empty()) {
    if (failureReason)
      *failureReason = "card has no available Tile";
    return mlir::failure();
  }
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    StructuredDAGNodePlacement placement;
    placement.node = node.id;
    std::optional<llvm::SmallVector<CardBaselineSpatialAxis, 4>> axes =
        getCardBaselineSpatialAxes(node);
    if (!axes || axes->empty()) {
      auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
      if (!tiling) {
        if (failureReason)
          *failureReason = "structured node has no typed iterator domain";
        return mlir::failure();
      }
      placement.iteratorPartitionFactors.assign(
          tiling.getLoopIteratorTypes().size(), 1);
      placement.tiles = {program.availableTileIds.front()};
      placements.push_back(std::move(placement));
      continue;
    }
    const CardBaselineSpatialAxis &axis = axes->front();
    const uint64_t participants = std::min<uint64_t>(
        program.availableTileIds.size(), std::max<uint64_t>(1, axis.extent));
    placement.iteratorPartitionFactors = axis.unitPartitionFactors;
    placement.iteratorPartitionFactors[axis.iteratorDimension] =
        static_cast<uint32_t>(participants);
    placement.tiles.assign(program.availableTileIds.begin(),
                           program.availableTileIds.begin() + participants);
    placements.push_back(std::move(placement));
  }
  return placements;
}

mlir::FailureOr<CardBaselinePlacementClosure>
closePlacements(const CardProgramAnalysis &program,
                llvm::SmallVector<StructuredDAGNodePlacement, 16> placements,
                BaselineStatistics *statistics, std::string *failureReason) {
  while (true) {
    if (statistics)
      ++statistics->spatialCoordinateQueries;
    CardBaselinePlacementVerdict legality;
    mlir::FailureOr<CardBaselinePlacementClosure> closure =
        closeCardBaselinePlacement(program.dag, placements, program.epoch,
                                   failureReason, &legality);
    if (mlir::succeeded(closure))
      return std::move(*closure);
    if (legality.status != analysis::ExactDemandStatus::ProvenLogicalInfeasible)
      return mlir::failure();
    mlir::FailureOr<DeterministicSpatialAdvance> advance =
        advanceDeterministicSpatialCoordinate(placements, failureReason);
    if (mlir::failed(advance) ||
        *advance == DeterministicSpatialAdvance::Exhausted)
      return mlir::failure();
    if (statistics)
      ++statistics->spatialLegalizationTransitions;
  }
}

CardBaselineAssignment buildAssignment(CardBaselinePlacementClosure closure) {
  CardBaselineAssignment assignment;
  assignment.nodePlacements = std::move(closure.nodePlacements);
  for (const CardBaselineObservablePlacement &selected :
       closure.outputPlacements) {
    OutputTileMapping output;
    output.outputIndex = selected.outputIndex;
    output.shardDimension = selected.shardDimension;
    output.activeTileIds = selected.tiles;
    assignment.mapping.outputs.push_back(std::move(output));
  }
  llvm::sort(assignment.mapping.outputs,
             [](const OutputTileMapping &lhs, const OutputTileMapping &rhs) {
               return lhs.outputIndex < rhs.outputIndex;
             });
  assignment.mapping.materializationMode =
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  return assignment;
}

} // namespace

mlir::FailureOr<CardBaselineAssignment>
computeCardBaselineAssignment(const CardProgramAnalysis &program, CardId cardId,
                              BaselineStatistics *statistics,
                              llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 16>>
      placements = [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "deterministic-baseline", "initial-spatial-placement");
        return makeInitialPlacements(program, &failureReason);
      }();
  if (mlir::failed(placements)) {
    diagnostics << "wafer-compile: baseline placement failed: " << failureReason
                << '\n';
    return mlir::failure();
  }
  mlir::FailureOr<CardBaselinePlacementClosure> closure = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "close-spatial-placement");
    return closePlacements(program, std::move(*placements), statistics,
                           &failureReason);
  }();
  if (mlir::failed(closure)) {
    diagnostics << "wafer-compile: baseline placement closure failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    statistics->exactDemandSatisfiedEdges = program.dag.getEdges().size();
  CardBaselineAssignment assignment = buildAssignment(std::move(*closure));
  // Temporal wave shapes are part of physical carrier construction: remote
  // producer demand is fragmented at those exact wave boundaries so no peer
  // endpoint materializes a whole logical shard in SPM.
  mlir::LogicalResult temporal = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "select-temporal-waves");
    return setCardBaselineTemporalTiles(assignment, program, &failureReason);
  }();
  mlir::LogicalResult movement = mlir::failure();
  if (mlir::succeeded(temporal)) {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "construct-data-movement");
    movement = addCardBaselineDataMovement(assignment, program.dag,
                                           program.epoch, &failureReason);
  }
  if (mlir::failed(temporal) || mlir::failed(movement)) {
    diagnostics << "wafer-compile: baseline assignment failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  return assignment;
}

} // namespace wafer::compiler::detail
