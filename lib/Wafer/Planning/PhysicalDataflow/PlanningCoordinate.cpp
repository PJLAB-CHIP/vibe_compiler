//===- PlanningCoordinate.cpp - Physical planning coordinates --------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningCoordinate.h"

namespace wafer::compiler::detail {

llvm::StringRef
stringifyRequiredPlanningCoordinate(RequiredPlanningCoordinate coordinate) {
  switch (coordinate) {
  case RequiredPlanningCoordinate::Region:
    return "region";
  case RequiredPlanningCoordinate::Temporal:
    return "temporal";
  case RequiredPlanningCoordinate::PartialFeasibility:
    return "partial-feasibility";
  case RequiredPlanningCoordinate::Representation:
    return "representation";
  case RequiredPlanningCoordinate::Movement:
    return "movement";
  case RequiredPlanningCoordinate::Storage:
    return "storage";
  case RequiredPlanningCoordinate::EventResource:
    return "event-resource";
  case RequiredPlanningCoordinate::ExecutionStructure:
    return "execution-structure";
  case RequiredPlanningCoordinate::StructureSpecificStorage:
    return "structure-specific-storage";
  case RequiredPlanningCoordinate::Schedule:
    return "schedule";
  }
  return "unknown";
}

} // namespace wafer::compiler::detail
