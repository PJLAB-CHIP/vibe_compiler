//===- CompleteCandidateKey.cpp - Complete physical assignment -------===//

#include "Wafer/Planning/PhysicalDataflow/Search/CompleteCandidateKey.h"

#include "llvm/ADT/StringRef.h"

#include <utility>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<CompleteCandidateKey> fail(std::string *failureReason,
                                           llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

} // namespace

mlir::FailureOr<CompleteCandidateKey>
CompleteCandidateKey::createFromValidatedPlans(
    SpatialPlan spatial, RegionPlan regions, TemporalPlan temporal,
    RepresentationPlan representations, MovementPlan movement,
    BufferPlan initialBuffers, ExecutionStructurePlan structure,
    BufferPlan buffers, ClosedSchedulePlan schedule,
    std::string *failureReason) {
  if (spatial.nodes.empty() || regions.groups.empty() ||
      structure.scopes.empty() || buffers.storageObjects.empty() ||
      schedule.controlOrders.empty())
    return fail(failureReason,
                "complete candidate key has an incomplete physical axis");
  if (!(schedule.structure == structure) || !(schedule.buffers == buffers))
    return fail(failureReason,
                "complete candidate key has a stale schedule generation");
  BufferPlan initialBase = initialBuffers;
  BufferPlan selectedBase = buffers;
  initialBase.slotFamilies.clear();
  selectedBase.slotFamilies.clear();
  if (!(initialBase == selectedBase))
    return fail(failureReason,
                "complete candidate key changed storage outside post-K slots");
  return CompleteCandidateKey(std::move(spatial), std::move(regions),
                              std::move(temporal), std::move(representations),
                              std::move(movement), std::move(initialBuffers),
                              std::move(structure), std::move(buffers),
                              std::move(schedule));
}

mlir::FailureOr<CompleteCandidateKey>
CompleteCandidateKey::create(const ScheduledState &state,
                             std::string *failureReason) {
  const ExecutionStructureState &structure =
      state.getBufferState().getExecutionStructureState();
  return createFromValidatedPlans(
      state.getSpatialPlan(), state.getRegionPlan(), state.getTemporalPlan(),
      state.getRepresentationPlan(), state.getMovementPlan(),
      structure.getInitialBufferPlan(), state.getExecutionStructurePlan(),
      state.getBufferPlan(), state.getSchedulePlan(), failureReason);
}

} // namespace wafer::compiler::detail
