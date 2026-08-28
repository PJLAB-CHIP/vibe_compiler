//===- Internal.h - Schedule cost analysis internals ----------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H
#define WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H

#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis::detail {

unsigned getKnowledgeSeverity(ScheduleCostKnowledge knowledge);

void degrade(ScheduleCostMetric &metric, ScheduleCostKnowledge knowledge,
             ScheduleCostReason reason);

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result);

struct Quantity {
  uint64_t value = 0;
  ScheduleCostKnowledge knowledge = ScheduleCostKnowledge::Known;
  ScheduleCostReason reason = ScheduleCostReason::None;

  static Quantity unavailable(ScheduleCostReason reason);
  static Quantity unsupported(ScheduleCostReason reason);
  static Quantity overflow();
};

struct ExecutionMultiplicity {
  Quantity exact{1};
  Quantity lowerBound{1};
  Quantity upperBound{1};
};

using InstructionWorkCountMember =
    InstructionExecutionCount InstructionProgramWork::*;

/// Canonical enumeration of every execution-count dimension. Aggregation and
/// knowledge degradation use this list so adding a typed work field cannot
/// silently create a second, partial fact path.
inline constexpr InstructionWorkCountMember kInstructionWorkCountMembers[] = {
    &InstructionProgramWork::instructions,
    &InstructionProgramWork::asynchronousEvents,
    &InstructionProgramWork::rdmaIssues,
    &InstructionProgramWork::wdmaIssues,
    &InstructionProgramWork::tdmaIssues,
    &InstructionProgramWork::ctIssues,
    &InstructionProgramWork::neIssues,
    &InstructionProgramWork::dteOperations,
    &InstructionProgramWork::gatherScatterOperations,
    &InstructionProgramWork::dteSendOperations,
    &InstructionProgramWork::dteReceiveOperations,
    &InstructionProgramWork::dteWaitOperations,
    &InstructionProgramWork::nccJoins,
    &InstructionProgramWork::steadyStateNCCJoins,
    &InstructionProgramWork::nonTerminalNCCJoins,
    &InstructionProgramWork::nccParticipantWaits,
    &InstructionProgramWork::steadyStateNCCParticipantWaits,
    &InstructionProgramWork::nonTerminalNCCParticipantWaits,
    &InstructionProgramWork::intrinsicNCCDrains,
};

Quantity multiply(Quantity lhs, Quantity rhs);
Quantity multiply(Quantity lhs, uint64_t rhs);
void add(ScheduleCostMetric &metric, Quantity quantity);

void walkInstructionProgramWork(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, ExecutionMultiplicity)>
        onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow);

/// Visits the same statically executable instruction stream used by the
/// Tile-local cost collector. The callback receives the current static
/// multiplicity. Unsupported recursive/call control flow invokes
/// `onUnsupportedControlFlow` because it may hide instructions.
void walkInstructionProgram(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow);

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost);
void collectExecutionCost(
    mlir::Operation *root, InstructionProgramCost &cost,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation);
void collectSPMHighWater(mlir::Operation *root, InstructionProgramCost &cost,
                         const TargetMemoryPolicy &policy);
void collectSPMHighWater(
    mlir::Operation *root, InstructionProgramCost &cost,
    const TargetMemoryPolicy &policy,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation);
void collectDDRHighWater(mlir::Operation *root, InstructionProgramCost &cost);
void collectDDRHighWater(
    mlir::Operation *root, InstructionProgramCost &cost,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation);

} // namespace wafer::analysis::detail

#endif // WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H
