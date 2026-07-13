//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"

namespace wafer::compiler::testing {

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

} // namespace wafer::compiler::testing

#endif // WAFER_COMPILER_TESTING_H
