//===- CurrentIRLayoutOptimization.h - Current layout cleanup -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H
#define WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H

#include "Wafer/Planning/PhysicalDataflow/RepresentationPBQPSolver.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

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
  RepresentationPBQPStatus status = RepresentationPBQPStatus::BrokenContract;
  CurrentIRLayoutOptimizationStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return status == RepresentationPBQPStatus::Optimal;
  }
};

/// Builds one query-local PBQP directly from the supplied current Tile IR and
/// immediately applies the selected exact reuse/dead-materialization choices.
/// Operation and Value handles never leave this call. The current domain has
/// no performance-comparable soft term until the concrete target descriptor
/// count is known, so this query uses only hard feasibility and the solver's
/// complete semantic tie-break.
CurrentIRLayoutOptimizationResult optimizeCurrentIRLayouts(
    llvm::MutableArrayRef<CurrentIRLayoutModule> modules,
    uint64_t workLimit = UINT64_C(1048576));

/// Re-establishes explicit semantic owners for result-producing current Tile
/// operations after a function-boundary bufferization epoch. Owners come only
/// from current buffer relations or already-recorded current consumers;
/// absence of that evidence is a contract failure.
mlir::LogicalResult closeCurrentTileDataflowOwnerRelations(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_CURRENTIRLAYOUTOPTIMIZATION_H
