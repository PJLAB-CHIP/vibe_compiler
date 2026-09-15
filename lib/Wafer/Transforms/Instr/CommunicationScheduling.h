//===- CommunicationScheduling.h - Construct a current issue order --------===//
#ifndef WAFER_TRANSFORMS_INSTR_COMMUNICATIONSCHEDULING_H
#define WAFER_TRANSFORMS_INSTR_COMMUNICATIONSCHEDULING_H

#include "Wafer/IR/Topology/TargetTopology.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>

namespace wafer::compiler::detail {
enum class CommunicationSchedulingStatus {
  Scheduled,
  Blocked,
  Unsupported,
  WorkLimit,
  Contract
};

struct CommunicationSchedulingResult {
  CommunicationSchedulingStatus status =
      CommunicationSchedulingStatus::Contract;
  // Only a blocked read-only query returns these current-epoch handles.
  // Each endpoint's local dependencies have all been scheduled. Its peer's
  // dependencies may still be blocked; no incomplete pair was committed.
  llvm::SmallVector<mlir::Operation *> readySenders;
  llvm::SmallVector<mlir::Operation *> readyReceivers;
  uint64_t groups = 0;
  uint64_t movedIssues = 0;
  uint64_t work = 0;
  uint64_t effectSummaries = 0;
  std::string detail;
};

/// Construct compatible full send/receive groups over verified actual
/// operations. Ordinary local operations retain their relative order. SSA,
/// precise byte conflicts, scope and DDR publication edges constrain movable
/// issues. Success commits that permutation to the same IR and erases obsolete
/// DTE waits; the caller immediately runs the common completion implementation.
/// Other outcomes leave the IR unchanged. No operation is predicted/created.
CommunicationSchedulingResult
scheduleCurrentCommunication(llvm::ArrayRef<mlir::ModuleOp> modules,
                             llvm::ArrayRef<TileId> tileIds,
                             uint64_t maximumWork = 16777216);
} // namespace wafer::compiler::detail
#endif
