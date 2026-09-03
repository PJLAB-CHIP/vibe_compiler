//===- NativeDirectDTEMultiSend.h - Form native DTE sends ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_NATIVEDIRECTDTEMULTISEND_H
#define WAFER_TRANSFORMS_INSTR_NATIVEDIRECTDTEMULTISEND_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class NativeDirectDTEMultiSendFailureKind : uint8_t {
  None,
  BrokenContract,
  CompilerFailure,
};

struct NativeDirectDTEMultiSendStatistics {
  uint64_t coalescedP2PTransfers = 0;
  uint64_t unicastReceiveOperationsRemoved = 0;
  uint64_t broadcastOperations = 0;
  uint64_t scatterOperations = 0;
  uint64_t unicastSendOperationsRemoved = 0;
};

struct NativeDirectDTEMultiSendResult {
  NativeDirectDTEMultiSendFailureKind failure =
      NativeDirectDTEMultiSendFailureKind::None;
  NativeDirectDTEMultiSendStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return failure == NativeDirectDTEMultiSendFailureKind::None;
  }
};

/// Coalesces matched unicast send/receive fragments only when both physical
/// source and destination ranges form the same contiguous ordered union.
/// The transformation runs before native multi-destination grouping and fresh
/// completion.
NativeDirectDTEMultiSendResult
coalesceExactDirectDTETransfers(llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Replaces complete groups of actual unicast Instr sends with the target's
/// qualified native broadcast/scatter operations. The supplied modules must
/// be Tile-to-Instr lowered and must not yet contain compiler-derived DTE
/// waits or physical bindings. All groups are preflighted before mutation.
NativeDirectDTEMultiSendResult materializeNativeDirectDTEMultiSends(
    llvm::ArrayRef<mlir::ModuleOp> tileModules);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_INSTR_NATIVEDIRECTDTEMULTISEND_H
