//===- GatherScatterWork.h - Materialize bounded GS work -------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_GATHERSCATTERWORK_H
#define WAFER_TRANSFORMS_INSTR_GATHERSCATTERWORK_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdint>

namespace wafer {

enum class GatherScatterWorkFailure { None, UnsupportedAliasing, InvalidIR };

struct GatherScatterWorkResult {
  GatherScatterWorkFailure failure = GatherScatterWorkFailure::None;
  uint64_t rewrittenOperations = 0;
  uint64_t issuedSegments = 0;
  bool succeeded() const { return failure == GatherScatterWorkFailure::None; }
};

/// Operates on current descriptors before final completion/memory planning.
/// Does not create storage or choose completion. Failure emits a diagnostic;
/// the caller owns and discards a failed candidate, as for other transforms.
GatherScatterWorkResult
materializeGatherScatterWork(mlir::func::FuncOp function);

} // namespace wafer

#endif // WAFER_TRANSFORMS_INSTR_GATHERSCATTERWORK_H
