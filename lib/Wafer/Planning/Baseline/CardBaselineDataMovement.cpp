//===- CardBaselineDataMovement.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Planning/Baseline/CardBaselineConsumerInputs.h"
#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"
#include "Wafer/Support/CompileTiming.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardBaselineAssignment &assignment, const StructuredDAGAnalysis &dag,
    analysis::IREpoch epoch, std::string *failureReason) {
  mlir::FailureOr<analysis::LogicalShardTrial> trial = buildLogicalShardTrial(
      dag, assignment.nodePlacements, epoch, failureReason);
  if (mlir::failed(trial))
    return mlir::failure();

  StructuredDAGExactDemandQuery query(dag, epoch);
  mlir::LogicalResult inputs = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "construct-consumer-inputs");
    return addCardBaselineConsumerInputs(assignment.mapping, dag, *trial, query,
                                         failureReason);
  }();
  if (mlir::failed(inputs))
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "query", "deterministic-baseline", "construct-edge-carriers");
  return addCardBaselineEdgeCarriers(assignment.mapping, *trial, dag, query,
                                     failureReason);
}

} // namespace wafer::compiler::detail
