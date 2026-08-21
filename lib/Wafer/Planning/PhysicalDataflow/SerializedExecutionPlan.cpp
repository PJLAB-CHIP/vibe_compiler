//===- SerializedExecutionPlan.cpp - Serialized plan schema -----------===//

#include "Wafer/Planning/PhysicalDataflow/SerializedExecutionPlan.h"

namespace wafer::compiler::detail {

const SerializedExecutionPlan *getSerializedExecutionPlan(
    const CanonicalSerializedExecutionPlanOutcome &outcome) {
  return std::get_if<SerializedExecutionPlan>(&outcome);
}

} // namespace wafer::compiler::detail
