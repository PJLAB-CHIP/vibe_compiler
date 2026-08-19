//===- CardBaselineDataMovement.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"

#include "Wafer/Planning/Baseline/CardBaselineConsumerInputs.h"
#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardBaselineAssignment &assignment, const StructuredDAGAnalysis &dag,
    analysis::IREpoch epoch, std::string *failureReason) {
  mlir::FailureOr<analysis::LogicalShardTrial> trial = buildLogicalShardTrial(
      dag, assignment.nodePlacements, epoch, failureReason);
  if (mlir::failed(trial))
    return mlir::failure();

  StructuredDAGExactDemandQuery query(dag, epoch);
  if (mlir::failed(addCardBaselineConsumerInputs(
          assignment.mapping, dag, *trial, query, failureReason)))
    return mlir::failure();
  return addCardBaselineEdgeCarriers(assignment.mapping, *trial, dag, query,
                                     failureReason);
}

} // namespace wafer::compiler::detail
