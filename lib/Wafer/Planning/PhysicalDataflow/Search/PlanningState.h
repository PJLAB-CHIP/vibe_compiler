//===- PlanningState.h - Closed physical planning prefixes -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H

#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"
#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningCoordinate.h"
#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

class PhysicalDataflowPlanningProblem;
class MovementDomain;
class StorageDomain;
class ExecutionStructureDomain;
class StructureSpecificStorageDomain;
class RegionDomain;
class RepresentationDomain;
class TemporalDomain;

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

/// A validated physical-version coordinate extending one TemporalState.
/// Constraint graph, resources and solver records remain derived domain facts.
class RepresentationState {
public:
  static mlir::FailureOr<RepresentationState>
  create(const RepresentationDomain &domain, TemporalState temporal,
         RepresentationPlan representations,
         std::string *failureReason = nullptr);

  const TemporalState &getTemporalState() const { return temporal; }
  const SpatialPlan &getSpatialPlan() const {
    return temporal.getSpatialPlan();
  }
  const RegionPlan &getRegionPlan() const { return temporal.getRegionPlan(); }
  const TemporalPlan &getTemporalPlan() const {
    return temporal.getTemporalPlan();
  }
  const RepresentationPlan &getRepresentationPlan() const {
    return representations;
  }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::Movement;
  }

  friend bool operator==(const RepresentationState &lhs,
                         const RepresentationState &rhs) {
    return lhs.temporal == rhs.temporal &&
           lhs.representations == rhs.representations;
  }
  friend bool operator<(const RepresentationState &lhs,
                        const RepresentationState &rhs) {
    if (lhs.temporal < rhs.temporal)
      return true;
    if (rhs.temporal < lhs.temporal)
      return false;
    return lhs.representations < rhs.representations;
  }

private:
  RepresentationState(TemporalState temporal,
                      RepresentationPlan representations)
      : temporal(std::move(temporal)),
        representations(std::move(representations)) {}

  TemporalState temporal;
  RepresentationPlan representations;
};

/// A validated movement coordinate extending one RepresentationState.
class MovementState {
public:
  static mlir::FailureOr<MovementState>
  create(const MovementDomain &domain, RepresentationState representations,
         MovementPlan movement, std::string *failureReason = nullptr);

  const RepresentationState &getRepresentationState() const {
    return representations;
  }
  const SpatialPlan &getSpatialPlan() const {
    return representations.getSpatialPlan();
  }
  const RegionPlan &getRegionPlan() const {
    return representations.getRegionPlan();
  }
  const TemporalPlan &getTemporalPlan() const {
    return representations.getTemporalPlan();
  }
  const RepresentationPlan &getRepresentationPlan() const {
    return representations.getRepresentationPlan();
  }
  const MovementPlan &getMovementPlan() const { return movement; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::Storage;
  }

  friend bool operator==(const MovementState &lhs, const MovementState &rhs) {
    return lhs.representations == rhs.representations &&
           lhs.movement == rhs.movement;
  }
  friend bool operator<(const MovementState &lhs, const MovementState &rhs) {
    if (lhs.representations < rhs.representations)
      return true;
    if (rhs.representations < lhs.representations)
      return false;
    return lhs.movement < rhs.movement;
  }

private:
  MovementState(RepresentationState representations, MovementPlan movement)
      : representations(std::move(representations)),
        movement(std::move(movement)) {}

  RepresentationState representations;
  MovementPlan movement;
};

/// Pre-structure storage coordinate. A later execution-structure choice may
/// invalidate occurrence families and require a new BufferState.
class InitialBufferState {
public:
  static mlir::FailureOr<InitialBufferState>
  create(const StorageDomain &domain, MovementState movement,
         BufferPlan buffers, std::string *failureReason = nullptr);

  const MovementState &getMovementState() const { return movement; }
  const SpatialPlan &getSpatialPlan() const {
    return movement.getSpatialPlan();
  }
  const RegionPlan &getRegionPlan() const { return movement.getRegionPlan(); }
  const TemporalPlan &getTemporalPlan() const {
    return movement.getTemporalPlan();
  }
  const RepresentationPlan &getRepresentationPlan() const {
    return movement.getRepresentationPlan();
  }
  const MovementPlan &getMovementPlan() const {
    return movement.getMovementPlan();
  }
  const BufferPlan &getBufferPlan() const { return buffers; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::EventResource;
  }

  friend bool operator==(const InitialBufferState &lhs,
                         const InitialBufferState &rhs) {
    return lhs.movement == rhs.movement && lhs.buffers == rhs.buffers;
  }
  friend bool operator<(const InitialBufferState &lhs,
                        const InitialBufferState &rhs) {
    if (lhs.movement < rhs.movement)
      return true;
    if (rhs.movement < lhs.movement)
      return false;
    return lhs.buffers < rhs.buffers;
  }

private:
  InitialBufferState(MovementState movement, BufferPlan buffers)
      : movement(std::move(movement)), buffers(std::move(buffers)) {}

  MovementState movement;
  BufferPlan buffers;
};

/// A fixed execution-structure coordinate. Pre-structure storage and the
/// foundation EventGraph are upstream facts only; I and J must be rebuilt for
/// this exact structure before the state can reach scheduling.
class ExecutionStructureState {
public:
  static mlir::FailureOr<ExecutionStructureState>
  create(const ExecutionStructureDomain &domain, InitialBufferState buffers,
         ExecutionStructurePlan structure,
         std::string *failureReason = nullptr);

  const InitialBufferState &getInitialBufferState() const { return buffers; }
  const SpatialPlan &getSpatialPlan() const { return buffers.getSpatialPlan(); }
  const RegionPlan &getRegionPlan() const { return buffers.getRegionPlan(); }
  const TemporalPlan &getTemporalPlan() const {
    return buffers.getTemporalPlan();
  }
  const RepresentationPlan &getRepresentationPlan() const {
    return buffers.getRepresentationPlan();
  }
  const MovementPlan &getMovementPlan() const {
    return buffers.getMovementPlan();
  }
  const BufferPlan &getInitialBufferPlan() const {
    return buffers.getBufferPlan();
  }
  const BufferPlan &getBufferPlan() const { return getInitialBufferPlan(); }
  const ExecutionStructurePlan &getExecutionStructurePlan() const {
    return structure;
  }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::StructureSpecificStorage;
  }

  friend bool operator==(const ExecutionStructureState &lhs,
                         const ExecutionStructureState &rhs) {
    return lhs.buffers == rhs.buffers && lhs.structure == rhs.structure;
  }
  friend bool operator<(const ExecutionStructureState &lhs,
                        const ExecutionStructureState &rhs) {
    if (lhs.buffers < rhs.buffers)
      return true;
    if (rhs.buffers < lhs.buffers)
      return false;
    return lhs.structure < rhs.structure;
  }

private:
  ExecutionStructureState(InitialBufferState buffers,
                          ExecutionStructurePlan structure)
      : buffers(std::move(buffers)), structure(std::move(structure)) {}

  InitialBufferState buffers;
  ExecutionStructurePlan structure;
};

/// Storage reclosed for one exact execution structure. The parent structure
/// is the generation identity; a plan from another K sibling cannot create
/// this state even when its object bindings happen to compare equal.
class BufferState {
public:
  static mlir::FailureOr<BufferState>
  create(const StructureSpecificStorageDomain &domain,
         ExecutionStructureState structure, BufferPlan buffers,
         std::string *failureReason = nullptr);

  const ExecutionStructureState &getExecutionStructureState() const {
    return structure;
  }
  const SpatialPlan &getSpatialPlan() const {
    return structure.getSpatialPlan();
  }
  const RegionPlan &getRegionPlan() const { return structure.getRegionPlan(); }
  const TemporalPlan &getTemporalPlan() const {
    return structure.getTemporalPlan();
  }
  const RepresentationPlan &getRepresentationPlan() const {
    return structure.getRepresentationPlan();
  }
  const MovementPlan &getMovementPlan() const {
    return structure.getMovementPlan();
  }
  const ExecutionStructurePlan &getExecutionStructurePlan() const {
    return structure.getExecutionStructurePlan();
  }
  const BufferPlan &getBufferPlan() const { return buffers; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return RequiredPlanningCoordinate::Schedule;
  }

  friend bool operator==(const BufferState &lhs, const BufferState &rhs) {
    return lhs.structure == rhs.structure && lhs.buffers == rhs.buffers;
  }
  friend bool operator<(const BufferState &lhs, const BufferState &rhs) {
    if (lhs.structure < rhs.structure)
      return true;
    if (rhs.structure < lhs.structure)
      return false;
    return lhs.buffers < rhs.buffers;
  }

private:
  BufferState(ExecutionStructureState structure, BufferPlan buffers)
      : structure(std::move(structure)), buffers(std::move(buffers)) {}

  ExecutionStructureState structure;
  BufferPlan buffers;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSTATE_H
