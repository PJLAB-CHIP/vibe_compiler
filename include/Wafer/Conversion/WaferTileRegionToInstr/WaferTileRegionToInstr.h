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
/// budget used by group tiling.
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

enum class AllGatherSchedule {
  Ring,
  Direct,
};

enum class AllReduceSchedule {
  Ring,
  Tree,
};

enum class ReduceScatterSchedule {
  Direct,
};

struct TileRegionToInstrOptions {
  AllGatherSchedule allGatherSchedule = AllGatherSchedule::Ring;
  AllReduceSchedule allReduceSchedule = AllReduceSchedule::Ring;
  ReduceScatterSchedule reduceScatterSchedule = ReduceScatterSchedule::Direct;
};

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               std::string *failureReason = nullptr);

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               const TileRegionToInstrOptions &options,
                               std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
