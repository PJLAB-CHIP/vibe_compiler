//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

/// Test-only entry to the production all-rank Direct DTE acceptance gate.
/// Successful calls attach typed bindings; failed calls leave every candidate
/// issue unbound.
mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules);

/// Runs the production transaction while injecting a failure only after the
/// selected logical rank has completed lowering and verification. This API is
/// callable only through test drivers and is not a production command-line
/// option.
mlir::LogicalResult compileProgramWithRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics);

mlir::LogicalResult compileProgramWithTargetRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics);

mlir::LogicalResult compileProgramWithPackageRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::testing

#endif // WAFER_COMPILER_TESTING_H
