//===- LayoutOptimization.h - Current layout cleanup -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H
#define WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

struct LayoutOptimizationInput {
  mlir::ModuleOp module;
  StructuredMaterializationRelations *relations = nullptr;
};

struct LayoutOptimizationStatistics {
  uint64_t invocations = 0;
  uint64_t solverWork = 0;
  uint64_t layoutMaterializationsBefore = 0;
  uint64_t layoutMaterializationsAfter = 0;
  uint64_t unusedMaterializationsErased = 0;
  uint64_t sharedMaterializationsReused = 0;
  uint64_t hardOnlyInvocations = 0;
};

struct LayoutOptimizationResult {
  ExactPBQPStatus status = ExactPBQPStatus::BrokenContract;
  LayoutOptimizationStatistics statistics;
  std::string detail;

  bool succeeded() const { return status == ExactPBQPStatus::Optimal; }
};

/// Builds one query-local PBQP directly from the supplied current Tile IR and
/// immediately applies the selected exact reuse/dead-materialization choices.
/// Operation and Value handles never leave this call. The current domain has
/// no performance-comparable soft term until the concrete target descriptor
/// count is known, so this query uses only hard feasibility and the solver's
/// complete semantic tie-break.
LayoutOptimizationResult
optimizeTileLayouts(llvm::MutableArrayRef<LayoutOptimizationInput> modules,
                    uint64_t workLimit = UINT64_C(1048576));

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H
