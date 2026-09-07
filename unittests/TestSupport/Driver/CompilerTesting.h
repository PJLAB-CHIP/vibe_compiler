//===- CompilerTesting.h - Compilation test-only hooks -------------*- C++
//-*-===//

#ifndef WAFER_TESTSUPPORT_COMPILERTESTING_H
#define WAFER_TESTSUPPORT_COMPILERTESTING_H

#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Driver/CompilationResult.h"
#include "Wafer/Driver/CompiledProgram.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

enum class CommunicationCandidate {
  Peer,
  SharedDDR,
  RecursiveDoubling,
  DimensionOrderedAllToAll,
  RingReduction,
};

/// Selects one realization for source-derived qualification. This adapter is
/// linked only into test drivers; production none/search expose no selector.
llvm::Expected<CompiledProgram> compileProgramWithCommunicationCandidate(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    CommunicationCandidate candidate, llvm::raw_ostream &diagnostics);

/// Test-only entry to card Direct DTE binding. Successful calls attach
/// typed bindings; failed calls leave every issue unbound.
mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Test-only entry to card resource verification. The summary is
/// recomputed from the supplied Tile IR and is not persisted as a
/// second scheduling representation.
mlir::FailureOr<analysis::InstructionProgramAggregateCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                       llvm::ArrayRef<TileId> tileIds,
                       const ExecutionConfig &executionConfig);

/// Runs the production transaction while injecting a failure after the
/// selected Tile executable launch slot has completed lowering and
/// verification. This API is callable only through test drivers.
llvm::Expected<CompilationResult> compileProgramWithExecutableLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

llvm::Expected<CompilationResult> compileProgramWithTargetLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

llvm::Expected<CompilationResult> compileProgramWithPackageLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

/// Runs the production transaction while injecting a failure after the
/// commit-stage verification and before the single publication rename. The
/// target output must remain invisible.
llvm::Expected<CompilationResult> compileProgramWithCommitVerificationFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics);

/// Corrupts the staged program-data member immediately before package binding.
/// The public facade must return a PackageCommit CompilationFailure and publish
/// no output.
llvm::Expected<CompilationResult> compileProgramWithPackageBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics);

/// Corrupts the staged profile plan immediately before instrumentation
/// binding. The public facade must return a PackageCommit CompilationFailure
/// and publish no delivery root.
llvm::Expected<CompilationResult> compileProgramWithProfileBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::testing

#endif // WAFER_TESTSUPPORT_COMPILERTESTING_H
