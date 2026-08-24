//===- CompleteCandidatePreparation.h - Selected candidate stages -*- C++
//-*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEPREPARATION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEPREPARATION_H

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"
#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

/// Current candidate execution identity mapped to the structured node emitted
/// for it. Replica nodes are candidate-local and therefore remain distinct.
struct CandidateExecutionNodeRelation {
  RegionExecutionId execution;
  uint32_t structuredNodeId = 0;
};

struct PreparedCandidateEvents {
  std::vector<PlannedEvent> events;
  std::vector<EventDependency> hardDependencies;
  std::vector<CompletionObligation> completionObligations;
};

PreparedCandidateEvents prepareCandidateEvents(const EventGraph &eventGraph);

/// Applies the already-selected post-K schedule inside Q50.0's owned
/// candidate. The object owns only immutable/query-local plan facts and never
/// retains IR pointers after either preparation callback returns.
class CompleteCandidatePreparation final : public CardExecutablePreparation {
public:
  CompleteCandidatePreparation(
      ScheduleDomain scheduleDomain, PreparedCandidateEvents events,
      RepresentationPlan representations, MovementPlan movement,
      std::vector<MovementResourceDescription> movementResources,
      BufferPlan buffers,
      std::vector<StorageResourceDescription> storageResources,
      ExecutionStructurePlan structure, ClosedSchedulePlan schedule,
      std::vector<CandidateExecutionNodeRelation> executionNodes)
      : scheduleDomain(std::move(scheduleDomain)),
        events(std::move(events)),
        representations(std::move(representations)),
        movement(std::move(movement)),
        movementResources(std::move(movementResources)),
        buffers(std::move(buffers)),
        storageResources(std::move(storageResources)),
        structure(std::move(structure)), schedule(std::move(schedule)),
        executionNodes(std::move(executionNodes)) {}

  bool ownsInstructionCompletion() const final { return true; }

  mlir::LogicalResult
  prepareTileDataflow(llvm::MutableArrayRef<CandidateTileDataflowIR> tiles,
                      CardExecutablePreparationFailure &failure) final;

  mlir::LogicalResult
  prepareInstructionIR(llvm::MutableArrayRef<CandidateInstructionIR> tiles,
                       CardExecutablePreparationFailure &failure) final;

private:
  ScheduleDomain scheduleDomain;
  PreparedCandidateEvents events;
  RepresentationPlan representations;
  MovementPlan movement;
  std::vector<MovementResourceDescription> movementResources;
  BufferPlan buffers;
  std::vector<StorageResourceDescription> storageResources;
  ExecutionStructurePlan structure;
  ClosedSchedulePlan schedule;
  std::vector<CandidateExecutionNodeRelation> executionNodes;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEPREPARATION_H
