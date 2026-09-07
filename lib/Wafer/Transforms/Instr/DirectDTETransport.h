//===- DirectDTETransport.h - Physical Direct DTE binding -----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_DIRECTDTETRANSPORT_H
#define WAFER_TRANSFORMS_INSTR_DIRECTDTETRANSPORT_H

#include "Wafer/Target/TransportContract.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class DirectDTECompletionFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  CompilerFailure,
};

struct DirectDTECompletionStatistics {
  uint64_t sends = 0;
  uint64_t receives = 0;
  uint64_t waitsErased = 0;
  uint64_t waitsPlaced = 0;
  uint64_t senderSlotReuseWaits = 0;
  uint64_t receiverFSMReuseWaits = 0;
  uint64_t receiverReadySlotReuseWaits = 0;
};

struct DirectDTECompletionResult {
  DirectDTECompletionFailureKind failure = DirectDTECompletionFailureKind::None;
  DirectDTECompletionStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return failure == DirectDTECompletionFailureKind::None;
  }
};

/// Erases compiler-derived Direct-DTE waits and rebuilds them from the actual
/// Instr issues, SSA tokens, buffer effects and current target sender/FSM/ready
/// limits. The request-local placement choices are consumed in this call and
/// are never retained as a schedule plan.
DirectDTECompletionResult
rebuildRequiredDirectDTEWaits(llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Recomputes logical message matching, structured dynamic occurrences and the
/// card Direct-DTE wait graph from the supplied Tile modules.
/// This validation does not attach physical bindings or retain analysis state.
mlir::LogicalResult
verifyDirectDTETransportSchedule(llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Matches and validates every logical Direct DTE issue across the supplied
/// Tile domain, then writes typed physical bindings into the modules.
/// Analysis completes for the full domain before any binding is written.
mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_INSTR_DIRECTDTETRANSPORT_H
