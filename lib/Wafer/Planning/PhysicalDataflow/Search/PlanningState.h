//===- PlanningState.h - Closed structural planning prefixes -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

class PhysicalDataflowPlanningProblem;
class RegionDomain;
class TemporalDomain;

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

class TemporalState {
public:
  static mlir::FailureOr<TemporalState>
  create(const TemporalDomain &domain, RegionState region,
         TemporalPlan temporal, std::string *failureReason = nullptr);

  const RegionState &getRegionState() const { return region; }
  const SpatialState &getSpatialState() const {
    return region.getSpatialState();
  }
  const SpatialPlan &getSpatialPlan() const { return region.getSpatialPlan(); }
  const RegionPlan &getRegionPlan() const { return region.getRegionPlan(); }
  const TemporalPlan &getTemporalPlan() const { return temporal; }
  friend bool operator==(const TemporalState &lhs, const TemporalState &rhs) {
    return lhs.region == rhs.region && lhs.temporal == rhs.temporal;
  }
  friend bool operator<(const TemporalState &lhs, const TemporalState &rhs) {
    if (lhs.region < rhs.region)
      return true;
    if (rhs.region < lhs.region)
      return false;
    return lhs.temporal < rhs.temporal;
  }

private:
  TemporalState(RegionState region, TemporalPlan temporal)
      : region(std::move(region)), temporal(std::move(temporal)) {}

  RegionState region;
  TemporalPlan temporal;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
