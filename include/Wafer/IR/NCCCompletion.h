//===- NCCCompletion.h - Instr NCC completion semantics -----*- C++ -*-===//

#ifndef WAFER_IR_NCCCOMPLETION_H
#define WAFER_IR_NCCCOMPLETION_H

#include "Wafer/IR/WaferEnums.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
}

namespace wafer {

inline constexpr uint32_t kNCCWorkerCount = getMaxEnumValForNCCWorker() + 1;
inline constexpr uint32_t kAllNCCWorkersMask =
    (uint32_t{1} << kNCCWorkerCount) - 1;
inline constexpr char kWaferNCCWorkerAttrName[] = "worker";

enum class NCCCompletionKind : uint8_t {
  None,
  OrderedAsynchronousIssue,
  ParticipantJoin,
  SynchronousWriteback,
};

/// Completion semantics of one current Instr operation. Ordinary issues carry
/// one worker and no participant mask. A join carries only its nonempty worker
/// mask. Synchronous writeback carries the issue worker and completes that same
/// worker before the operation returns.
struct NCCOperationCompletion {
  NCCCompletionKind kind = NCCCompletionKind::None;
  std::optional<NCCWorker> issueWorker;
  uint32_t participantMask = 0;
};

/// Reads only typed operation interfaces. Concrete operation switches and
/// target ABI constants are intentionally outside this adapter.
NCCOperationCompletion getNCCOperationCompletion(mlir::Operation *operation);

/// Return or update the typed worker of an ordinary NCC issue. Mutation is
/// rejected for non-issue operations and out-of-domain values.
std::optional<NCCWorker> getNCCIssueWorker(mlir::Operation *operation);
mlir::LogicalResult setNCCIssueWorker(mlir::Operation *operation,
                                      NCCWorker worker);

} // namespace wafer

#endif // WAFER_IR_NCCCOMPLETION_H
