//===- StructuralReadiness.h - Closed-prefix structural query -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURALREADINESS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURALREADINESS_H

#include "Wafer/Planning/PhysicalDataflow/PlanningCoordinate.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

enum class StructuralReadinessKind : uint8_t {
  MissingCoordinate,
  ReadyForNextCoordinate,
  Unsupported,
  CompilerBug,
};

/// Pure result of validating the currently closed planning prefix. Its closed
/// payload is only the classification, required coordinate and detail.
class StructuralReadinessResult {
public:
  StructuralReadinessKind getKind() const { return kind; }
  std::optional<RequiredPlanningCoordinate> getRequiredCoordinate() const {
    return requiredCoordinate;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  StructuralReadinessResult(
      StructuralReadinessKind kind,
      std::optional<RequiredPlanningCoordinate> requiredCoordinate,
      std::string detail = {})
      : kind(kind), requiredCoordinate(requiredCoordinate),
        detail(std::move(detail)) {}

  StructuralReadinessKind kind;
  std::optional<RequiredPlanningCoordinate> requiredCoordinate;
  std::string detail;

  friend StructuralReadinessResult
  checkStructuralReadiness(const TemporalDomain &, const TemporalPlan &);
};

StructuralReadinessResult
checkStructuralReadiness(const TemporalDomain &domain,
                         const TemporalPlan &temporal);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURALREADINESS_H
