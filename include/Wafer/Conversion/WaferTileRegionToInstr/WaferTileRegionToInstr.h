//===- WaferTileRegionToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
#define WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <string>

namespace wafer {

namespace detail {

/// Maximum number of statically materialized terminal instruction issues and
/// explicit local completions accepted for one logical rank.  This is an
/// independent target-program bound, not the candidate traversal expansion
/// budget used by structured tensor scheduling.
inline constexpr uint64_t kStaticTerminalOperationBudget = 4096;

enum class StaticTerminalOperationBudgetStatus {
  WithinBudget,
  CountOverflow,
  BudgetExceeded,
};

/// Counts every terminal instruction issue and explicit local completion in
/// `root` without mutating IR, then checks the per-rank static program bound.
/// `operationCount` is reset before traversal and contains the exact count
/// unless the count itself overflows uint64_t.
StaticTerminalOperationBudgetStatus
checkStaticTerminalOperationBudget(mlir::Operation *root,
                                   uint64_t &operationCount);

} // namespace detail

/// Recompute NCC pending/completion state from the module's current typed
/// instruction, SSA and memory-effect IR. Orphan and overbroad typed joins are
/// erased or narrowed; only cross-worker/external-observer conflicts and
/// terminal publication materialize the minimum participant join.
///
/// Candidate rewrites that remove, clone or reorder instruction issues must
/// rerun this normalizer before lifetime/resource planning. The operation is
/// intentionally module-level because pending NCC state can cross
/// tile-region and static-loop boundaries.
mlir::LogicalResult normalizeMinimumNCCJoins(mlir::ModuleOp module);

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
