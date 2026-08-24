//===- CompleteCandidateKey.h - Complete physical assignment -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_COMPLETECANDIDATEKEY_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_COMPLETECANDIDATEKEY_H

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {

/// Stable value identity of one complete physical assignment. The key carries
/// every search axis but no IR pointer, ordinal, actual offset, diagnostic,
/// objective, or feedback history.
class CompleteCandidateKey {
public:
  /// Joins plans already validated by their owning domains and rechecks the
  /// only cross-generation relations stored in the key itself. Production
  /// callers normally use the ScheduledState overload below.
  static mlir::FailureOr<CompleteCandidateKey> createFromValidatedPlans(
      SpatialPlan spatial, RegionPlan regions, TemporalPlan temporal,
      RepresentationPlan representations, MovementPlan movement,
      BufferPlan initialBuffers, ExecutionStructurePlan structure,
      BufferPlan buffers, ClosedSchedulePlan schedule,
      std::string *failureReason = nullptr);

  static mlir::FailureOr<CompleteCandidateKey>
  create(const ScheduledState &state, std::string *failureReason = nullptr);

  const SpatialPlan &getSpatialPlan() const { return spatial; }
  const RegionPlan &getRegionPlan() const { return regions; }
  const TemporalPlan &getTemporalPlan() const { return temporal; }
  const RepresentationPlan &getRepresentationPlan() const {
    return representations;
  }
  const MovementPlan &getMovementPlan() const { return movement; }
  const BufferPlan &getInitialBufferPlan() const { return initialBuffers; }
  const ExecutionStructurePlan &getExecutionStructurePlan() const {
    return structure;
  }
  const BufferPlan &getBufferPlan() const { return buffers; }
  const ClosedSchedulePlan &getSchedulePlan() const { return schedule; }

  friend bool operator==(const CompleteCandidateKey &lhs,
                         const CompleteCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions, lhs.temporal, lhs.representations,
                    lhs.movement, lhs.initialBuffers, lhs.structure,
                    lhs.buffers, lhs.schedule) ==
           std::tie(rhs.spatial, rhs.regions, rhs.temporal, rhs.representations,
                    rhs.movement, rhs.initialBuffers, rhs.structure,
                    rhs.buffers, rhs.schedule);
  }
  friend bool operator<(const CompleteCandidateKey &lhs,
                        const CompleteCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions, lhs.temporal, lhs.representations,
                    lhs.movement, lhs.initialBuffers, lhs.structure,
                    lhs.buffers, lhs.schedule) <
           std::tie(rhs.spatial, rhs.regions, rhs.temporal, rhs.representations,
                    rhs.movement, rhs.initialBuffers, rhs.structure,
                    rhs.buffers, rhs.schedule);
  }

private:
  CompleteCandidateKey(SpatialPlan spatial, RegionPlan regions,
                       TemporalPlan temporal,
                       RepresentationPlan representations,
                       MovementPlan movement, BufferPlan initialBuffers,
                       ExecutionStructurePlan structure, BufferPlan buffers,
                       ClosedSchedulePlan schedule)
      : spatial(std::move(spatial)), regions(std::move(regions)),
        temporal(std::move(temporal)),
        representations(std::move(representations)),
        movement(std::move(movement)),
        initialBuffers(std::move(initialBuffers)),
        structure(std::move(structure)), buffers(std::move(buffers)),
        schedule(std::move(schedule)) {}

  SpatialPlan spatial;
  RegionPlan regions;
  TemporalPlan temporal;
  RepresentationPlan representations;
  MovementPlan movement;
  BufferPlan initialBuffers;
  ExecutionStructurePlan structure;
  BufferPlan buffers;
  ClosedSchedulePlan schedule;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_COMPLETECANDIDATEKEY_H
