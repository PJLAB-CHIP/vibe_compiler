//===- CardBaselineDataMovement.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineDataMovement.h"

#include "Wafer/Planning/Baseline/CardBaselineConsumerInputs.h"
#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardBaselineAssignment &assignment, const StructuredDAGAnalysis &dag,
    std::string *failureReason) {
  if (mlir::failed(addCardBaselineConsumerInputs(
          assignment.mapping, assignment.demand, failureReason)))
    return mlir::failure();
  return addCardBaselineEdgeCarriers(
      assignment.mapping, assignment.spatial, assignment.demand, dag,
      failureReason);
}

} // namespace wafer::compiler::detail
