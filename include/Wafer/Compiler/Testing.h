//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetCodeGen.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

/// Test-only entry to card Direct DTE binding. Successful calls attach
/// typed bindings; failed calls leave every issue unbound.
mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules);

/// Test-only entry to card resource verification. The summary is
/// recomputed from the supplied Tile IR and is not persisted as a
/// second scheduling representation.
mlir::FailureOr<analysis::CardInstructionProgramCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                         llvm::ArrayRef<TileId> tileIds,
                         const ExecutionConfig &executionConfig);

/// Runs the production transaction while injecting a failure after the
/// selected Tile executable launch slot has completed lowering and
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
