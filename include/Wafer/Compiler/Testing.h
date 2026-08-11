//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

/// Test-only entry to the production whole-card Direct DTE acceptance gate.
/// Successful calls attach typed bindings; failed calls leave every candidate
/// issue unbound.
mlir::FailureOr<TransportContract>
acceptDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules);

/// Test-only entry to the production whole-card resource gate. The summary is
/// recomputed from the supplied accepted physical Tile IR and is not persisted
/// as a second scheduling representation.
mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeCardResources(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
                         llvm::ArrayRef<PhysicalTileId> physicalTileIds,
                         const ExecutionConfig &executionConfig);

/// Runs the production transaction while injecting a failure after the
/// selected physical-Tile executable launch slot has completed lowering and
/// verification. This API is callable only through test drivers.
mlir::LogicalResult compileProgramWithExecutableLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

mlir::LogicalResult compileProgramWithTargetLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

mlir::LogicalResult compileProgramWithPackageLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::testing

#endif // WAFER_COMPILER_TESTING_H
