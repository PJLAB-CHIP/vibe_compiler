//===- NCCCompletion.h - Pure target NCC completion protocol -*- C++ -*-===//

#ifndef WAFER_TARGET_NCCCOMPLETION_H
#define WAFER_TARGET_NCCCOMPLETION_H

#include "Wafer/ABI/Tx81NCCABI.h"

#include <cstdint>

namespace wafer {

/// Pure target worker identity used by target calls, runtime, and models.
/// MLIR's NCCWorker attribute maps to this protocol type only in lowering.
enum class TargetNCCWorker : uint8_t { Worker0 = 0, Worker1 = 1, Worker2 = 2 };

inline constexpr uint32_t kTargetNCCWorkerCount = WAFER_TX81_NCC_WORKER_COUNT;
inline constexpr uint32_t kAllTargetNCCWorkersMask =
    WAFER_TX81_NCC_ALL_WORKERS_MASK;

/// Completion behavior of one decoded target command in an NCC worker domain.
enum class TargetNCCCompletionBehavior : uint8_t {
  None,
  OrderedAsynchronousIssue,
  ParticipantJoin,
  SynchronousWriteback,
};

} // namespace wafer

#endif // WAFER_TARGET_NCCCOMPLETION_H
