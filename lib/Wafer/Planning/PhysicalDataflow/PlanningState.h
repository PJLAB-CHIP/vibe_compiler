//===- PlanningState.h - Closed structural planning prefixes -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGSTATE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGSTATE_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

class PhysicalDataflowPlanningProblem;
class RegionDomain;

class SpatialState {
public:
  static mlir::FailureOr<SpatialState>
  create(const PhysicalDataflowPlanningProblem &problem, SpatialPlan plan,
         std::string *failureReason = nullptr);

  const SpatialPlan &getPlan() const { return plan; }
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

class RegionState {
public:
  static mlir::FailureOr<RegionState>
  create(const RegionDomain &domain, SpatialState spatial, RegionPlan regions,
         std::string *failureReason = nullptr);

  const SpatialState &getSpatialState() const { return spatial; }
  const SpatialPlan &getSpatialPlan() const { return spatial.getPlan(); }
  const RegionPlan &getRegionPlan() const { return regions; }
  friend bool operator==(const RegionState &lhs, const RegionState &rhs) {
    return lhs.spatial == rhs.spatial && lhs.regions == rhs.regions;
  }
  friend bool operator<(const RegionState &lhs, const RegionState &rhs) {
    if (lhs.spatial < rhs.spatial)
      return true;
    if (rhs.spatial < lhs.spatial)
      return false;
    return lhs.regions < rhs.regions;
  }

private:
  RegionState(SpatialState spatial, RegionPlan regions)
      : spatial(std::move(spatial)), regions(std::move(regions)) {}

  SpatialState spatial;
  RegionPlan regions;
};

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGSTATE_H
