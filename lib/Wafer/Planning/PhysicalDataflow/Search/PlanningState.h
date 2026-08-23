//===- PlanningState.h - Closed physical planning prefixes -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H

#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>

namespace wafer::compiler::detail {

class PhysicalDataflowPlanningProblem;

/// The only not-yet-closed coordinate after a SpatialState in the current
/// planning pipeline. New coordinates extend the closed state sequence in
/// their owning work item; they are not nullable fields in this state.
enum class RequiredPlanningCoordinate : uint8_t { Region };

llvm::StringRef
stringifyRequiredPlanningCoordinate(RequiredPlanningCoordinate coordinate);

/// A validated spatial prefix. It owns only the compact semantic value: no
/// parent, operation handle, proposal source, score, history, or derived
/// demand proof participates in state identity.
class SpatialState {
public:
  static mlir::FailureOr<SpatialState>
  create(const PhysicalDataflowPlanningProblem &problem, SpatialPlan plan,
         std::string *failureReason = nullptr);

  const SpatialPlan &getPlan() const { return plan; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::Region;
  }

  friend bool operator==(const SpatialState &lhs, const SpatialState &rhs) {
    return lhs.plan == rhs.plan;
  }
  friend bool operator<(const SpatialState &lhs, const SpatialState &rhs) {
    return lhs.plan < rhs.plan;
  }

private:
  explicit SpatialState(SpatialPlan plan) : plan(std::move(plan)) {}

  SpatialPlan plan;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
