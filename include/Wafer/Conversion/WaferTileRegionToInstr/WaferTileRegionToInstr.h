//===- WaferTileRegionToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
#define WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <memory>

namespace wafer {

namespace detail {

enum class StaticExecutableOperationCountStatus {
  Counted,
  CountOverflow,
};

/// Counts every instruction op and explicit NCC join in `root`
/// without mutating IR.  Program size is a ranking/cost fact rather than a
/// fixed workload-legality gate; only an unrepresentable count fails closed.
/// `operationCount` is reset before traversal and contains the exact count
/// unless the count itself overflows uint64_t.
StaticExecutableOperationCountStatus
countStaticExecutableOperations(mlir::Operation *root,
                                uint64_t &operationCount);

} // namespace detail

/// Place the required NCC joins from one function's current typed instruction,
/// SSA, and memory-effect IR. Orphan and overbroad joins are
/// erased or narrowed; only cross-worker/external-observer conflicts and
/// function-result observation materialize the minimum participant join.
///
/// Candidate rewrites that remove, clone or reorder instruction issues must
/// rerun this normalizer before lifetime/resource planning. The operation is
/// intentionally function-level because pending NCC state can cross
/// tile-region and static-loop boundaries, but never crosses a call boundary.
/// The caller owns the mutable function and discards it if the operation
/// fails; this API does not copy the function to manufacture rollback.
mlir::LogicalResult placeRequiredNCCJoins(mlir::func::FuncOp function);

/// Applies the same operation to every directly nested function in a
/// caller-owned mutable module. The caller discards the module on failure.
/// New pass pipelines should use the function-anchored operation above.
mlir::LogicalResult placeRequiredNCCJoins(mlir::ModuleOp module);

/// Erase every compiler-derived NCC participant join and rebuild required joins
/// solely from the module's current worker order, typed issues/effects,
/// aliases, ranges, event tokens, and observer boundaries. The current Instr
/// owner uses this after function-boundary bufferization and final order; the
/// memory/target leaf only validates the result. Action construction may use
/// the incremental normalizer above while loop-carried communication topology
/// is still being formed. The operation mutates the caller-owned function once
/// and does not promise rollback on failure.
mlir::LogicalResult rebuildRequiredNCCJoins(mlir::func::FuncOp function);

/// Applies the same rebuild to every directly nested function in a
/// caller-owned mutable module. The caller discards the module on failure.
mlir::LogicalResult rebuildRequiredNCCJoins(mlir::ModuleOp module);

/// Rebuild joins required by one isolated TileRegion's local SPM roots.
/// The query starts with no access to the clone's private allocations and
/// drains those roots before they leave their owning region. It does not
/// model unrelated function-level outstanding accesses and must not replace
/// the function-anchored production pass.
///
/// The caller owns a private disposable TileRegion. The rebuild mutates that
/// region once; on failure the caller discards the complete query scope.
mlir::LogicalResult
rebuildRequiredNCCJoinsForIsolatedTileRegion(TileRegionOp tileRegion);

/// Returns true when `root` contains a typed Tile dataflow operation consumed
/// by Tile-region-to-Instr conversion. TileRegionOp and TileYieldOp are
/// structural boundaries and therefore do not by themselves require
/// instruction lowering.
bool containsTileDataflowOperations(mlir::Operation *root);

/// Caller-owned sink for actual SPM allocations created while lowering one
/// source Tile operation. Implementations must derive ownership from current
/// typed relations of `sourceOperation`; operation names, locations, shapes
/// and insertion order are never ownership evidence.
class TileRegionToInstrBufferRecorder {
public:
  virtual ~TileRegionToInstrBufferRecorder() = default;
  virtual void recordScratchAllocation(mlir::Operation *sourceOperation,
                                       mlir::Value allocation) = 0;
  virtual void recordLoweredOperation(mlir::Operation *sourceOperation,
                                      mlir::Operation *loweredOperation) = 0;
};

/// Immutable, request-scoped lowering infrastructure for one MLIRContext.
/// Frozen patterns and conversion legality may be shared by concurrent
/// conversions; the session owns no IR and records no per-conversion state.
class TileRegionToInstrLoweringSession {
public:
  explicit TileRegionToInstrLoweringSession(
      mlir::MLIRContext &context,
      TileRegionToInstrBufferRecorder *bufferRecorder = nullptr);
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
                           mlir::RewriterBase::Listener *);
  friend mlir::LogicalResult
  convertBufferizationCopiesToInstr(mlir::ModuleOp,
                                    TileRegionToInstrLoweringSession &,
                                    mlir::RewriterBase::Listener *);
};

/// Lower exactly one isolated TileRegion body. This operation does not run
/// function-wide required NCC join placement, bufferization, or memory
/// planning.
mlir::LogicalResult
convertTileRegionToInstr(TileRegionOp region,
                         TileRegionToInstrLoweringSession &session,
                         mlir::RewriterBase::Listener *listener = nullptr);

/// Lowers standard materializing copies introduced by function-boundary
/// bufferization after all TileRegion bodies have been converted. Copies are
/// classified from their current Wafer memory spaces and exact relation.
mlir::LogicalResult convertBufferizationCopiesToInstr(
    mlir::ModuleOp module, TileRegionToInstrLoweringSession &session,
    mlir::RewriterBase::Listener *listener = nullptr);

namespace detail {

/// Place joins directly in one caller-owned private function. The caller must
/// discard the complete function on failure.
/// Production pipelines use the function-anchored pass, which consumes this
/// same operation-scoped kernel.
mlir::LogicalResult
placeRequiredNCCJoinsInPrivateFunction(mlir::func::FuncOp function);

} // namespace detail

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
