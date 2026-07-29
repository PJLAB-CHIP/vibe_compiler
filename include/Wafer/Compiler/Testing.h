//===- Testing.h - Wafer compiler test-only hooks -------------*- C++ -*-===//

#ifndef WAFER_COMPILER_TESTING_H
#define WAFER_COMPILER_TESTING_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::testing {

/// Test-only collective implementation requested from the fully accepted
/// whole-rank frontier. Selection is verified from final instruction DTE
/// phases; an absent or mixed implementation fails closed.
enum class CollectiveCharacterizationAlgorithm {
  AllGatherDirect,
  AllGatherRing,
  ReduceScatterDirect,
  ReduceScatterRing,
  AllReduceRing,
  NoCResidentAllReduceRing,
  AllReduceTree,
};

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

/// Target-model form of reserved-baseline publication. It retains the exact
/// accepted executable and target LLVM modules consumed by package
/// publication so the model gate executes the selected baseline without
/// reconstructing either side of the target-call contract.
mlir::FailureOr<TargetCompilationProduct>
compileProgramWithReservedBaselineTargetCompilation(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Runs the production publication transaction while committing a fully
/// accepted static fixed-slot realization from each rank frontier. The
/// selected all-rank tuple passes the same lowering, resource,
/// target-artifact and canonical package readback gates as production. This
/// qualification seam is not a production command-line mode and does not
/// change production selection.
mlir::FailureOr<ExecutableBundle> compileProgramForStaticFixedSlotQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Target-model form of static fixed-slot qualification. It retains the exact
/// accepted executable and target LLVM modules already consumed by package
/// publication and qualification attestation.
mlir::FailureOr<TargetCompilationProduct>
compileProgramForStaticFixedSlotTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Runs the production publication transaction while committing a fully
/// accepted disjoint-component NCC worker realization from each rank
/// frontier. The selected tuple passes the same lowering, resource,
/// target-artifact and canonical package readback gates as production.
mlir::FailureOr<ExecutableBundle> compileProgramForWorkerPlacementQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// The target-model form of the worker-placement qualification transaction.
/// It retains the exact accepted executable and target LLVM modules already
/// consumed by package publication so the ordinary target-model gate does not
/// reconstruct either side of the target-call contract.
mlir::FailureOr<TargetCompilationProduct>
compileProgramForWorkerPlacementTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Runs the production publication transaction while committing one
/// nonreserved all-rank tuple that simultaneously has boundary-only DDR
/// movement, actual Direct DTE traffic on every rank and in both directions
/// across the tuple, static fixed-slot scheduling, and disjoint multi-worker
/// NCC placement in the accepted instruction IR. The predicates are never
/// combined across sibling candidates.
mlir::FailureOr<ExecutableBundle>
compileProgramForNoCResidentFixedSlotWorkerQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Target-model form of the compound NoC-resident fixed-slot worker
/// qualification. It retains the exact executable and target LLVM modules
/// already consumed by canonical package and qualification-companion
/// publication.
mlir::FailureOr<TargetCompilationProduct>
compileProgramForNoCResidentFixedSlotWorkerTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Runs the production transaction while committing a fully accepted variant
/// whose final instruction DTE phases identify the requested collective
/// implementation. All normal lowering, resource, target and package gates
/// still run; absence or ambiguity is a compilation failure.
mlir::FailureOr<ExecutableBundle> compileProgramForCollectiveCharacterization(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    CollectiveCharacterizationAlgorithm algorithm, llvm::StringRef reportPath,
    llvm::raw_ostream &diagnostics);

/// Target-model form of collective characterization. It retains the exact
/// accepted executable and target LLVM modules consumed by package
/// publication, while preserving the same atomic characterization-report
/// publication contract.
mlir::FailureOr<TargetCompilationProduct>
compileProgramForCollectiveCharacterizationTargetCompilation(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    CollectiveCharacterizationAlgorithm algorithm, llvm::StringRef reportPath,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::testing

#endif // WAFER_COMPILER_TESTING_H
