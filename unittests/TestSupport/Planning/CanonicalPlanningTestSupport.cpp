//===- CanonicalPlanningTestSupport.cpp -------------------------------===//

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"

#include <type_traits>
#include <utility>

namespace wafer::test {

mlir::FailureOr<CanonicalPlanningPrefix>
buildCanonicalPlanningPrefix(const compiler::detail::StructuredDAGAnalysis &dag,
                             llvm::ArrayRef<TileId> tiles,
                             std::string *failureReason) {
  auto coordinate = compiler::detail::buildCanonicalSpatialAssignment(
      dag, tiles, failureReason);
  if (mlir::failed(coordinate))
    return mlir::failure();
  auto demandSession = compiler::detail::DemandPlanningSession::create(
      dag, analysis::IndexRelationLimits(), failureReason);
  if (mlir::failed(demandSession))
    return mlir::failure();
  analysis::ExactDemandOutcome demandOutcome =
      demandSession->query(coordinate->assignment);
  const analysis::ExactDemandProof *demand =
      analysis::getExactDemandProof(demandOutcome);
  if (!demand) {
    if (failureReason)
      *failureReason = std::visit(
          [](const auto &value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
              return {};
            else
              return value.detail;
          },
          demandOutcome);
    return mlir::failure();
  }
  auto rootAnalysis = compiler::detail::RootRegionWorkAnalysis::create(
      dag, coordinate->assignment, *demand, failureReason);
  if (mlir::failed(rootAnalysis))
    return mlir::failure();
  std::vector<analysis::RootRegionWork> rootWorks;
  for (const compiler::detail::StructuredDAGNode &node : dag.getNodes()) {
    const compiler::detail::SemanticRootKey *root =
        rootAnalysis->getRoot(node.id);
    if (!root)
      return mlir::failure();
    for (TileId tile : tiles) {
      analysis::RootRegionWorkOutcome outcome =
          rootAnalysis->query(*root, tile);
      if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
        continue;
      const analysis::RootRegionWork *work =
          analysis::getRootRegionWork(outcome);
      if (!work) {
        if (failureReason)
          *failureReason = std::visit(
              [](const auto &value) -> std::string {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, analysis::RootRegionWork> ||
                              std::is_same_v<T, analysis::NoRootRegionWork>)
                  return {};
                else
                  return value.detail;
              },
              outcome);
        return mlir::failure();
      }
      rootWorks.push_back(*work);
    }
  }
  compiler::detail::CanonicalRegionPlanOutcome regionOutcome =
      compiler::detail::buildCanonicalRegionPlan(rootWorks);
  const compiler::detail::RegionPlan *regions =
      compiler::detail::getRegionPlan(regionOutcome);
  if (!regions) {
    if (failureReason)
      *failureReason =
          std::get<compiler::detail::BrokenRegionPlan>(regionOutcome).detail;
    return mlir::failure();
  }
  compiler::detail::CanonicalTemporalPlanOutcome temporalOutcome =
      compiler::detail::buildCanonicalTemporalPlan(*regions, rootWorks);
  const compiler::detail::TemporalPlan *temporal =
      compiler::detail::getTemporalPlan(temporalOutcome);
  if (!temporal) {
    if (failureReason)
      *failureReason =
          std::get<compiler::detail::BrokenTemporalPlan>(temporalOutcome)
              .detail;
    return mlir::failure();
  }
  return CanonicalPlanningPrefix{coordinate->assignment, *demand,
                                 std::move(rootWorks), *regions, *temporal};
}

} // namespace wafer::test
