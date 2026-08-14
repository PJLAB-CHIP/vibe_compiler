//===- WaferTileRegionToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
#define WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <memory>
#include <string>

namespace wafer {

namespace detail {

enum class StaticExecutableOperationCountStatus {
  Counted,
  CountOverflow,
};

/// Counts every instruction issue and explicit local completion in `root`
/// without mutating IR.  Program size is a ranking/cost fact rather than a
/// fixed workload-legality gate; only an unrepresentable count fails closed.
/// `operationCount` is reset before traversal and contains the exact count
/// unless the count itself overflows uint64_t.
StaticExecutableOperationCountStatus
countStaticExecutableOperations(mlir::Operation *root,
                                uint64_t &operationCount);

} // namespace detail

/// Recompute NCC pending/completion state from one function's current typed
/// instruction, SSA and memory-effect IR. Orphan and overbroad typed joins are
/// erased or narrowed; only cross-worker/external-observer conflicts and
/// terminal publication materialize the minimum participant join.
///
/// Candidate rewrites that remove, clone or reorder instruction issues must
/// rerun this normalizer before lifetime/resource planning. The operation is
/// intentionally function-level because pending NCC state can cross
/// tile-region and static-loop boundaries, but never crosses a call boundary.
mlir::LogicalResult normalizeMinimumNCCJoins(mlir::func::FuncOp function);

/// Compatibility adapter for owned module transformations. New pass
/// pipelines should use the function-anchored operation above.
mlir::LogicalResult normalizeMinimumNCCJoins(mlir::ModuleOp module);

/// Erase every compiler-derived NCC participant join and rebuild completion
/// solely from the module's current worker order, typed issues/effects,
/// aliases, ranges, event tokens, and observer boundaries. Terminal candidate
/// finalization uses this after function-boundary bufferization; action
/// construction uses the incremental normalizer above while its loop-carried
/// handoff topology is still being formed.
mlir::LogicalResult rebuildMinimumNCCJoins(mlir::func::FuncOp function);

/// Compatibility adapter for owned module transformations.
mlir::LogicalResult rebuildMinimumNCCJoins(mlir::ModuleOp module);

/// Returns true when `root` contains a typed Tile dataflow operation consumed
/// by Tile-region-to-Instr conversion. TileRegionOp and TileYieldOp are
/// structural boundaries and therefore do not by themselves require
/// instruction lowering.
bool containsTileDataflowOperations(mlir::Operation *root);

/// Immutable, request-scoped lowering infrastructure for one MLIRContext.
/// Frozen patterns and conversion legality may be shared by concurrent
/// conversions; the session owns no IR and records no per-conversion state.
class TileRegionToInstrLoweringSession {
public:
  explicit TileRegionToInstrLoweringSession(mlir::MLIRContext &context);
  ~TileRegionToInstrLoweringSession();

  TileRegionToInstrLoweringSession(const TileRegionToInstrLoweringSession &) =
      delete;
  TileRegionToInstrLoweringSession &
  operator=(const TileRegionToInstrLoweringSession &) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  friend mlir::LogicalResult
  convertTileRegionToInstr(TileRegionOp, TileRegionToInstrLoweringSession &,
                           std::string *);
};

/// Lower exactly one isolated TileRegion body. This operation does not run
/// function-wide NCC completion, bufferization, or memory planning.
mlir::LogicalResult
convertTileRegionToInstr(TileRegionOp region,
                         TileRegionToInstrLoweringSession &session,
                         std::string *failureReason = nullptr);

/// One-shot compatibility adapter. Multi-region compiler requests should
/// construct one request-scoped session and use the overload above.
mlir::LogicalResult
convertTileRegionToInstr(TileRegionOp region,
                         std::string *failureReason = nullptr);

/// Compatibility adapter that lowers every TileRegion in an owned module and
/// then runs function-wide NCC completion. Production pass pipelines should
/// use the region-anchored conversion and function-anchored completion passes.
mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
