//===- StructuralReadiness.cpp - Closed-prefix structural query -------===//

#include "Wafer/Planning/PhysicalDataflow/StructuralReadiness.h"

namespace wafer::compiler::detail {

StructuralReadinessResult
checkStructuralReadiness(const TemporalDomain &domain,
                         const TemporalPlan &temporal) {
  if (!domain.contains(temporal))
    return {StructuralReadinessKind::CompilerBug,
            {},
            "temporal prefix is outside its current structural domain"};
  return {StructuralReadinessKind::ReadyForNextCoordinate,
          RequiredPlanningCoordinate::Representation};
}

} // namespace wafer::compiler::detail
