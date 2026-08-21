//===- CanonicalBaselinePlan.cpp - Resolved deterministic plan --------===//

#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalTemporalPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<CanonicalBaselinePlan> fail(std::string *failureReason,
                                            llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

std::string demandDetail(const analysis::ExactDemandOutcome &outcome) {
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

} // namespace

mlir::FailureOr<CanonicalBaselinePlan>
buildCanonicalBaselinePlan(const CardProgramAnalysis &program,
                           const TargetMemoryPolicy &memory,
                           std::string *failureReason) {
  mlir::FailureOr<CanonicalSpatialCoordinate> spatial =
      buildCanonicalSpatialAssignment(program.dag, program.availableTileIds,
                                      failureReason);
  if (mlir::failed(spatial))
    return mlir::failure();
  mlir::FailureOr<DemandPlanningSession> demandSession =
      DemandPlanningSession::create(
          program.dag, analysis::IndexRelationLimits(), failureReason);
  if (mlir::failed(demandSession))
    return mlir::failure();
  analysis::ExactDemandOutcome demandOutcome =
      demandSession->query(spatial->assignment);
  const analysis::ExactDemandProof *demand =
      analysis::getExactDemandProof(demandOutcome);
  if (!demand)
    return fail(failureReason, demandDetail(demandOutcome));
  demandSession->close();

  mlir::FailureOr<RootRegionWorkAnalysis> rootAnalysis =
      RootRegionWorkAnalysis::create(program.dag, spatial->assignment, *demand,
                                     failureReason);
  if (mlir::failed(rootAnalysis))
    return mlir::failure();
  std::vector<analysis::RootRegionWork> rootWorks;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    const SemanticRootKey *root = rootAnalysis->getRoot(node.id);
    if (!root)
      return fail(failureReason, "canonical root has no semantic identity");
    for (TileId tile : program.availableTileIds) {
      analysis::RootRegionWorkOutcome outcome =
          rootAnalysis->query(*root, tile);
      if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
        continue;
      const analysis::RootRegionWork *work =
          analysis::getRootRegionWork(outcome);
      if (!work)
        return fail(failureReason,
                    "canonical root work query did not return exact work");
      rootWorks.push_back(*work);
    }
  }

  CanonicalRegionPlanOutcome regionOutcome =
      buildCanonicalRegionPlan(rootWorks);
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  if (!regions)
    return fail(failureReason,
                std::get<BrokenRegionPlan>(regionOutcome).detail);
  CanonicalTemporalPlanOutcome temporalOutcome =
      buildCanonicalTemporalPlan(*regions, rootWorks);
  const TemporalPlan *temporal = getTemporalPlan(temporalOutcome);
  if (!temporal)
    return fail(failureReason,
                std::get<BrokenTemporalPlan>(temporalOutcome).detail);
  CanonicalRepresentationPlanOutcome representationOutcome =
      buildCanonicalRepresentationPlan(*regions, *temporal, rootWorks);
  const CanonicalRepresentationCoordinate *representations =
      getCanonicalRepresentationCoordinate(representationOutcome);
  if (!representations)
    return fail(failureReason,
                "canonical representation plan is not available");
  CanonicalMovementPlanOutcome movementOutcome =
      buildCanonicalMovementPlan(*regions, *representations, rootWorks);
  const CanonicalMovementCoordinate *movements =
      getCanonicalMovementCoordinate(movementOutcome);
  if (!movements)
    return fail(failureReason, "canonical movement plan is not available");
  CanonicalSerializedExecutionPlanOutcome serializedOutcome =
      buildCanonicalSerializedExecutionPlan(*regions, *temporal);
  const SerializedExecutionPlan *serialized =
      getSerializedExecutionPlan(serializedOutcome);
  if (!serialized)
    return fail(failureReason,
                "canonical serialized execution plan is not available");
  CanonicalStoragePlanOutcome storageOutcome =
      buildCanonicalStoragePlan(*representations, *movements, *serialized);
  const CanonicalStorageCoordinate *storage =
      getCanonicalStorageCoordinate(storageOutcome);
  if (!storage)
    return fail(failureReason, "canonical storage plan is not available");
  CanonicalSchedulePlanOutcome scheduleOutcome =
      buildCanonicalSchedulePlan(*storage, *serialized);
  const CanonicalScheduleCoordinate *schedule =
      getCanonicalScheduleCoordinate(scheduleOutcome);
  if (!schedule)
    return fail(failureReason, "canonical schedule plan is not available");
  CanonicalAttentionWorkProjectionOutcome attentionOutcome =
      buildCanonicalAttentionWorkProjection(rootWorks, *representations,
                                            *movements, *storage, *schedule);
  const CanonicalAttentionWorkCoordinate *attention =
      getCanonicalAttentionWorkCoordinate(attentionOutcome);
  if (!attention)
    return fail(failureReason,
                "canonical attention projection is not available");
  CanonicalFeasibilityOutcome feasibilityOutcome =
      buildCanonicalFeasibilityProof(*storage, *movements, *schedule,
                                     *attention, memory);
  const CanonicalFeasibilityCoordinate *feasibility =
      getCanonicalFeasibilityCoordinate(feasibilityOutcome);
  if (!feasibility)
    return fail(failureReason,
                "canonical coordinate has no full feasibility proof");
  PreparedAttentionDecompositionOutcome preparedOutcome =
      prepareSelectedAttentionDecomposition(*attention, feasibility->proof);
  const auto *prepared =
      std::get_if<PreparedAttentionDecomposition>(&preparedOutcome);
  if (!prepared)
    return fail(failureReason,
                "canonical attention decomposition cannot be prepared");

  return CanonicalBaselinePlan{spatial->assignment,
                               *demand,
                               std::move(rootWorks),
                               *regions,
                               *temporal,
                               *representations,
                               *movements,
                               *serialized,
                               *storage,
                               *schedule,
                               *attention,
                               feasibility->problem,
                               feasibility->proof,
                               *prepared};
}

} // namespace wafer::compiler::detail
