//===- SpatialDemandTestSupport.h - Closed test spatial demand -*- C++ -*-===//

#ifndef WAFER_UNITTESTS_TESTSUPPORT_PLANNING_SPATIALDEMANDTESTSUPPORT_H
#define WAFER_UNITTESTS_TESTSUPPORT_PLANNING_SPATIALDEMANDTESTSUPPORT_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Support/LogicalResult.h"

#include <string>

namespace wafer::test {

struct TestSpatialDemand {
  compiler::detail::SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
};

mlir::FailureOr<TestSpatialDemand> buildTestSpatialDemand(
    const compiler::detail::StructuredDAGAnalysis &dag,
    llvm::ArrayRef<compiler::detail::StructuredDAGNodePlacement> placements,
    std::string *failureReason = nullptr);

} // namespace wafer::test

#endif // WAFER_UNITTESTS_TESTSUPPORT_PLANNING_SPATIALDEMANDTESTSUPPORT_H
