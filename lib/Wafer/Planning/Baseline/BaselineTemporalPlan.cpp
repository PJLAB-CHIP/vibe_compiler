//===- BaselineTemporalPlan.cpp - Deterministic temporal start -------===//

#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

namespace wafer::compiler::detail {
namespace {

CanonicalTemporalPlanOutcome
broken(BrokenTemporalPlanReason reason, llvm::StringRef detail,
       std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return BrokenTemporalPlan{reason, std::move(work), detail.str()};
}

} // namespace

CanonicalTemporalPlanOutcome
buildBaselineTemporalPlan(const RegionPlan &regions,
                          llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  for (const RegionGroupPlan &group : regions.groups)
    if (group.mandatoryRoots.size() != 1)
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "baseline temporal input is not singleton regions");
  TemporalDomainResult domain = buildTemporalDomain(regions, rootWorks);
  if (!domain.succeeded()) {
    const std::string detail =
        domain.failure ? domain.failure->detail
                       : "baseline temporal domain returned no detail";
    const BrokenTemporalPlanReason reason =
        detail.find("non-positive") != std::string::npos
            ? BrokenTemporalPlanReason::InvalidLocalExtent
            : BrokenTemporalPlanReason::RegionWorkMismatch;
    return broken(reason, detail);
  }
  TemporalSuccessor first = domain.domain->getFirstPlan();
  if (first.getKind() != TemporalSuccessorKind::Plan || !first.getPlan())
    return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                  first.getDetail().empty()
                      ? "baseline temporal domain has no full-local point"
                      : first.getDetail());
  return *first.getPlan();
}

} // namespace wafer::compiler::detail
