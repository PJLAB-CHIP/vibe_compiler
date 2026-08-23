//===- CanonicalBaselinePlan.cpp - Deterministic candidate plan ------===//

#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
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

template <typename T, typename = void> struct HasDetail : std::false_type {};
template <typename T>
struct HasDetail<T, std::void_t<decltype(std::declval<T>().detail)>>
    : std::true_type {};

template <typename Success, typename Outcome>
std::string outcomeDetail(const Outcome &outcome,
                          llvm::StringRef defaultDetail) {
  return std::visit(
      [&](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Success>)
          return defaultDetail.str();
        else if constexpr (HasDetail<T>::value)
          return value.detail;
        else
          return defaultDetail.str();
      },
      outcome);
}

} // namespace

mlir::LogicalResult recloseCanonicalBaselinePlan(CanonicalBaselinePlan &plan,
                                                 std::string *failureReason) {
  CanonicalRepresentationPlanOutcome representationOutcome =
      buildCanonicalRepresentationPlan(plan.regions, plan.temporal,
                                       plan.rootWorks);
  const CanonicalRepresentationCoordinate *representations =
      getCanonicalRepresentationCoordinate(representationOutcome);
  if (!representations) {
    if (failureReason)
      *failureReason = outcomeDetail<CanonicalRepresentationCoordinate>(
          representationOutcome,
          "canonical representation plan is not available");
    return mlir::failure();
  }

  CanonicalMovementPlanOutcome movementOutcome = buildCanonicalMovementPlan(
      plan.regions, *representations, plan.rootWorks);
  const CanonicalMovementCoordinate *movements =
      getCanonicalMovementCoordinate(movementOutcome);
  if (!movements) {
    if (failureReason)
      *failureReason = outcomeDetail<CanonicalMovementCoordinate>(
          movementOutcome, "canonical movement plan is not available");
    return mlir::failure();
  }

  CanonicalSerializedExecutionPlanOutcome serializedOutcome =
      buildCanonicalSerializedExecutionPlan(plan.regions, plan.temporal);
  const SerializedExecutionPlan *serialized =
      getSerializedExecutionPlan(serializedOutcome);
  if (!serialized) {
    if (failureReason)
      *failureReason = "canonical serialized execution plan is not available";
    return mlir::failure();
  }

  CanonicalStoragePlanOutcome storageOutcome =
      buildCanonicalStoragePlan(*representations, *movements, *serialized);
  const CanonicalStorageCoordinate *storage =
      getCanonicalStorageCoordinate(storageOutcome);
  if (!storage) {
    if (failureReason)
      *failureReason = std::get<BrokenStoragePlan>(storageOutcome).detail;
    return mlir::failure();
  }
  CanonicalStoragePlanOutcome residentStorageOutcome =
      recloseCanonicalStorageForTemporal(*storage, plan.temporal,
                                         plan.rootWorks);
  const CanonicalStorageCoordinate *residentStorage =
      getCanonicalStorageCoordinate(residentStorageOutcome);
  if (!residentStorage) {
    if (failureReason)
      *failureReason =
          std::get<BrokenStoragePlan>(residentStorageOutcome).detail;
    return mlir::failure();
  }

  CanonicalSchedulePlanOutcome scheduleOutcome =
      buildCanonicalSchedulePlan(*residentStorage, *serialized);
  const CanonicalScheduleCoordinate *schedule =
      getCanonicalScheduleCoordinate(scheduleOutcome);
  if (!schedule) {
    if (failureReason)
      *failureReason = "canonical schedule plan is not available";
    return mlir::failure();
  }

  CanonicalAttentionWorkProjectionOutcome attentionOutcome =
      buildCanonicalAttentionWorkProjection(plan.rootWorks, *representations,
                                            *movements, *residentStorage,
                                            *schedule, &plan.temporal);
  const CanonicalAttentionWorkCoordinate *attention =
      getCanonicalAttentionWorkCoordinate(attentionOutcome);
  if (!attention) {
    if (failureReason)
      *failureReason = outcomeDetail<CanonicalAttentionWorkCoordinate>(
          attentionOutcome, "canonical attention projection is not available");
    return mlir::failure();
  }

  PreparedAttentionDecompositionOutcome preparedOutcome =
      prepareSelectedAttentionDecomposition(*attention);
  const auto *prepared =
      std::get_if<PreparedAttentionDecomposition>(&preparedOutcome);
  if (!prepared) {
    if (failureReason)
      *failureReason = outcomeDetail<PreparedAttentionDecomposition>(
          preparedOutcome,
          "canonical attention decomposition cannot be prepared");
    return mlir::failure();
  }

  plan.representations = *representations;
  plan.movements = *movements;
  plan.serialized = *serialized;
  plan.storage = *residentStorage;
  plan.schedule = *schedule;
  plan.attention = *attention;
  plan.preparedAttention = *prepared;
  return mlir::success();
}

mlir::FailureOr<CanonicalBaselinePlan>
buildCanonicalBaselinePlan(const CardProgramAnalysis &program,
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
      buildBaselineTemporalPlan(*regions, rootWorks);
  const TemporalPlan *temporal = getTemporalPlan(temporalOutcome);
  if (!temporal)
    return fail(failureReason,
                std::get<BrokenTemporalPlan>(temporalOutcome).detail);

  CanonicalBaselinePlan plan;
  plan.spatial = spatial->assignment;
  plan.demand = *demand;
  plan.rootWorks = std::move(rootWorks);
  plan.regions = *regions;
  plan.temporal = *temporal;
  if (mlir::failed(recloseCanonicalBaselinePlan(plan, failureReason)))
    return mlir::failure();
  return plan;
}

} // namespace wafer::compiler::detail
