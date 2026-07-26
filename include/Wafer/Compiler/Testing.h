//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

/// Test-only entry to the production all-rank Direct DTE acceptance gate.
/// Successful calls attach typed bindings; failed calls leave every candidate
/// issue unbound.
mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> rankModules);

/// Test-only entry to the production whole-variant resource gate. The summary
/// is recomputed from the supplied accepted rank IR and is not persisted as a
/// second scheduling representation.
mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeVariantResources(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                            const ExecutionConfig &executionConfig);

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

/// Runs the production publication transaction while committing the unique
/// reserved-baseline member of each rank frontier. The source, lowering,
/// whole-variant legality, target-artifact and package gates are otherwise
/// identical to production compilation.
mlir::FailureOr<ExecutableBundle> compileProgramWithReservedBaseline(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::testing

#endif // WAFER_COMPILER_TESTING_H
