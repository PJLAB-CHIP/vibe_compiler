//===- NoCIntermediateDataflow.h - Typed intermediate handoff -*- C++ -*-===//

#ifndef WAFER_LIB_COMPILER_NOCINTERMEDIATEDATAFLOW_H
#define WAFER_LIB_COMPILER_NOCINTERMEDIATEDATAFLOW_H

#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace wafer::compiler::detail {

enum class NoCFanoutKind {
  Direct,
  ReceiveForward,
};

/// Rewrites exact, replicated producer/intermediate spill cuts in one actual
/// complete-rank tuple. Equivalence is proved from typed boundary tiles,
/// current SSA, operation interfaces, and value-associated effects. The
/// caller owns tuple cloning and all late gates.
unsigned materializeNoCIntermediateHandoffs(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind);

/// Routes one proven equivalent produced value to every frontend-required
/// replicated output publisher while preserving every exact full-buffer WDMA.
/// Required publishers are recovered from the current tile-region boundary
/// and frontend ABI binding; incomplete descriptors, duplicate/missing
/// publishers, or a later output writer reject the opportunity atomically.
unsigned materializeNoCOutputPublications(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind);

} // namespace wafer::compiler::detail

#endif // WAFER_LIB_COMPILER_NOCINTERMEDIATEDATAFLOW_H
