//===- ProgramIngestion.h - Portable StableHLO source ingestion -*- C++ -*-===//

#ifndef WAFER_FRONTEND_STABLEHLO_PROGRAMINGESTION_H
#define WAFER_FRONTEND_STABLEHLO_PROGRAMINGESTION_H

#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend {

inline constexpr llvm::StringLiteral kStableHLOProgramArtifact =
    "functions/forward.stablehlo.bc";

/// Reads the single portable StableHLO authority from a program directory.
/// Legalizes supported public XLA math extensions in the owned module before
/// returning it for strict source verification and downstream consumption.
/// Text MLIR and generic MLIR bytecode are not fallback inputs.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
deserializeStableHLOProgramDirectory(llvm::StringRef programDirectory,
                                     mlir::MLIRContext &context,
                                     llvm::raw_ostream &diagnostics);

/// Verifies the production source-IR boundary: one static program, only
/// builtin/func/StableHLO semantics, and no external custom-call fallback.
mlir::LogicalResult verifyStableHLOSourceModule(
    mlir::ModuleOp module, llvm::raw_ostream &diagnostics);

struct VerifiedStableHLOProgram {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  FrontendProgramVerificationResult metadata;
};

/// Shared advisory/compiler ingestion. The compiler still snapshots and
/// invokes this again with transaction-owned payload sources.
mlir::FailureOr<VerifiedStableHLOProgram> ingestStableHLOProgramDirectory(
    llvm::StringRef programDirectory, mlir::MLIRContext &context,
    llvm::raw_ostream &diagnostics,
    const ProgramPayloadResolver *resolver = nullptr);

} // namespace wafer::frontend

#endif // WAFER_FRONTEND_STABLEHLO_PROGRAMINGESTION_H
