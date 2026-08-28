//===- CurrentIRLayoutOptimization.h - Current layout cleanup -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H
#define WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H

#include "Wafer/Analysis/Structured/StructuredMaterializationRelations.h"
#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

struct CurrentIRLayoutModule {
  mlir::ModuleOp module;
  StructuredMaterializationRelations *relations = nullptr;
};

struct CurrentIRLayoutOptimizationStatistics {
  uint64_t invocations = 0;
  uint64_t solverWork = 0;
  uint64_t layoutMaterializationsBefore = 0;
  uint64_t layoutMaterializationsAfter = 0;
  uint64_t unusedMaterializationsErased = 0;
  uint64_t sharedMaterializationsReused = 0;
  uint64_t hardOnlyInvocations = 0;
};

struct CurrentIRLayoutOptimizationResult {
  ExactPBQPStatus status = ExactPBQPStatus::BrokenContract;
  CurrentIRLayoutOptimizationStatistics statistics;
  std::string detail;

  bool succeeded() const { return status == ExactPBQPStatus::Optimal; }
};

/// Builds one query-local PBQP directly from the supplied current Tile IR and
/// immediately applies the selected exact reuse/dead-materialization choices.
/// Operation and Value handles never leave this call. The current domain has
/// no performance-comparable soft term until the concrete target descriptor
/// count is known, so this query uses only hard feasibility and the solver's
/// complete semantic tie-break.
CurrentIRLayoutOptimizationResult
optimizeCurrentIRLayouts(llvm::MutableArrayRef<CurrentIRLayoutModule> modules,
                         uint64_t workLimit = UINT64_C(1048576));

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H
