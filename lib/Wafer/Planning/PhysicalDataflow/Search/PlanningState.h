//===- PlanningState.h - Closed physical planning prefixes -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>

namespace wafer::compiler::detail {

class PhysicalDataflowPlanningProblem;
class RegionDomain;
class TemporalDomain;

/// Not-yet-closed coordinate immediately following the deepest current state.
/// New coordinates extend the closed state sequence in their owning work item;
/// they are not nullable fields in an earlier state.
enum class RequiredPlanningCoordinate : uint8_t {
  Region,
  Temporal,
  PartialFeasibility,
};

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

/// A validated region prefix extending one SpatialState. RootRegionWork and
/// exact demand remain derived session facts and are not copied into identity.
class RegionState {
public:
  static mlir::FailureOr<RegionState>
  create(const RegionDomain &domain, SpatialState spatial, RegionPlan regions,
         std::string *failureReason = nullptr);

  const SpatialState &getSpatialState() const { return spatial; }
  const SpatialPlan &getSpatialPlan() const { return spatial.getPlan(); }
  const RegionPlan &getRegionPlan() const { return regions; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::Temporal;
  }

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

/// A validated temporal prefix extending one RegionState. Scope descriptors,
/// exact waves and root work remain derived session facts; identity contains
/// only the closed TemporalPlan coordinate.
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
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::PartialFeasibility;
  }

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
