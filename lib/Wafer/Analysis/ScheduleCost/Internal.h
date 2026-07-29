//===- Internal.h - Schedule cost analysis internals ----------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H
#define WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

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

  static Quantity unknown(ScheduleCostReason reason);
  static Quantity unsupported(ScheduleCostReason reason);
  static Quantity overflow();
};

Quantity multiply(Quantity lhs, Quantity rhs);
Quantity multiply(Quantity lhs, uint64_t rhs);
void add(ScheduleCostMetric &metric, Quantity quantity);

/// Visits the same statically executable instruction stream used by the
/// rank-local cost collector. The callback receives the current static
/// multiplicity. Unsupported recursive/call control flow invokes
/// `onUnsupportedControlFlow` because it may hide instructions.
void walkInstructionProgram(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow);

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost);
void collectDataDependencyDepth(mlir::Operation *root,
                                InstructionProgramCost &cost);
void collectQualifiedOverlapWindows(mlir::Operation *root,
                                    InstructionProgramCost &cost,
                                    const TargetScheduleCostPolicy &policy);
void collectSPMHighWater(mlir::Operation *root, InstructionProgramCost &cost,
                         const TargetScheduleCostPolicy &policy);

} // namespace wafer::analysis::detail

#endif // WAFER_ANALYSIS_SCHEDULECOST_INTERNAL_H
