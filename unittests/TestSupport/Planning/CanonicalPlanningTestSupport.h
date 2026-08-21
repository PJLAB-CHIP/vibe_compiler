//===- CanonicalPlanningTestSupport.h ------------------------*- C++ -*-===//

#ifndef WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H
#define WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalTemporalPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace wafer::test {

struct CanonicalPlanningPrefix {
  compiler::detail::SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  std::vector<analysis::RootRegionWork> rootWorks;
  compiler::detail::RegionPlan regions;
  compiler::detail::TemporalPlan temporal;
};

mlir::FailureOr<CanonicalPlanningPrefix>
buildCanonicalPlanningPrefix(const compiler::detail::StructuredDAGAnalysis &dag,
                             llvm::ArrayRef<TileId> tiles,
                             std::string *failureReason = nullptr);

} // namespace wafer::test

#endif // WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H
