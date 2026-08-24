//===- CardDataflowConstruction.cpp - Selected edge construction ------===//

#include "Wafer/Planning/PhysicalDataflow/CardDataflowConstruction.h"

#include "Wafer/Planning/PhysicalDataflow/CardConsumerInputs.h"
#include "Wafer/Planning/PhysicalDataflow/CardEdgeCarriers.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardDataflowConstruction(
    CardMaterializationPlan &assignment, const StructuredDAGAnalysis &dag,
    const MovementPlan &movement, std::string *failureReason) {
  if (mlir::failed(addCardConsumerInputs(assignment.mapping, assignment.demand,
                                         failureReason)))
    return mlir::failure();
  return addCardEdgeCarriers(assignment.mapping, assignment.spatial,
                             assignment.demand, dag, movement, failureReason);
}

} // namespace wafer::compiler::detail
