//===- PlanningCoordinate.h - Physical planning coordinates -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGCOORDINATE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGCOORDINATE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace wafer::compiler::detail {

enum class RequiredPlanningCoordinate : uint8_t {
  Region,
  Temporal,
  PartialFeasibility,
  Representation,
};

llvm::StringRef
stringifyRequiredPlanningCoordinate(RequiredPlanningCoordinate coordinate);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGCOORDINATE_H
