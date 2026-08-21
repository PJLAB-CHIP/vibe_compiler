//===- CanonicalPlanningTestSupport.h ------------------------*- C++ -*-===//

#ifndef WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H
#define WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalTemporalPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
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

/// Builds the shared rank-5 aligned/ragged flash-decoding fixture used to
/// verify coupled planning artifacts across canonical boundaries.
std::string buildFlashDecodingPlanningFixture(int64_t queryExtent,
                                              int64_t keyValueExtent);

/// Builds the shared rank-5 flash-attention fixture. The optional mask keeps
/// aligned/no-mask and ragged/mask coverage on the same semantic op contract.
std::string buildFlashAttentionPlanningFixture(int64_t queryExtent,
                                               int64_t keyValueExtent,
                                               bool withMask);

} // namespace wafer::test

#endif // WAFER_UNITTESTS_TESTSUPPORT_PLANNING_CANONICALPLANNINGTESTSUPPORT_H
