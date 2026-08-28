//===- TileToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_TILETOINSTR_TILETOINSTR_H
#define WAFER_CONVERSION_TILETOINSTR_TILETOINSTR_H

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

} // namespace wafer

#endif // WAFER_CONVERSION_TILETOINSTR_TILETOINSTR_H
