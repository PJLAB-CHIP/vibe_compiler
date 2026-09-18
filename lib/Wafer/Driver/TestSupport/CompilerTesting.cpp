//===- CompilerTesting.cpp - Compiler test-only adapters ----------------===//

#include "TestSupport/Driver/CompilerTesting.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CompilationQualification.h"
#include "Wafer/Driver/PhysicalDataflow/BaselineCurrentIR.h"
#include "Wafer/Driver/ProgramResourceVerification.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace wafer::compiler::testing {

llvm::Expected<CompiledProgram> compileProgramWithCommunicationCandidate(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    CommunicationCandidate candidate, llvm::raw_ostream &diagnostics) {
  if (!options.getOptimizationConfig().isNone())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "explicit communication qualification requires none");
  detail::CommunicationCandidateSelection selection;
  switch (candidate) {
  case CommunicationCandidate::Peer:
    break;
  case CommunicationCandidate::SharedDDR:
    selection.mergeRegions = false;
    selection.movement.transport = detail::BoundaryMovementTransport::SharedDDR;
    break;
  case CommunicationCandidate::RecursiveDoubling:
    selection.movement.allGather =
        detail::CompleteAllGatherAlgorithm::RecursiveDoubling;
    break;
  case CommunicationCandidate::DimensionOrderedAllToAll:
    selection.movement.allToAll =
        detail::CompleteAllToAllAlgorithm::DimensionOrdered;
    break;
  case CommunicationCandidate::RingReduction:
    selection.movement.reduction = detail::DistributedReductionAlgorithm::Ring;
    break;
  case CommunicationCandidate::AccessReusePeer:
    selection.reusePeerInputs = true;
    break;
  case CommunicationCandidate::PipelinedLoads:
    selection.pipelineLoads = true;
    break;
  }
  detail::CompilationQualification qualification{selection};
  return detail::compileProgramWithTargetLLVMModulesImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, options, diagnostics, &qualification);
}

llvm::Expected<CompiledProgram> compileProgramWithSearchCandidate(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    uint64_t candidateIndex, llvm::raw_ostream &diagnostics) {
  if (!options.getOptimizationConfig().isSearch())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "candidate inspection requires search");
  auto observe = [&](uint64_t index, const detail::ActualCandidateResult &) {
    return index == candidateIndex
               ? detail::CandidateObservationAction::Inspect
               : detail::CandidateObservationAction::Continue;
  };
  detail::CompilationQualification qualification{
      detail::SearchCandidateInspection{observe}};
  return detail::compileProgramWithTargetLLVMModulesImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, options, diagnostics, &qualification);
}

mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules) {
  return detail::bindDirectDTETransport(tileModules);
}

mlir::FailureOr<analysis::InstructionProgramAggregateCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                       llvm::ArrayRef<TileId> tileIds,
                       const ExecutionConfig &executionConfig) {
  return detail::verifyProgramResources(tileModules, tileIds, executionConfig);
}

llvm::Expected<CompilationResult> compileProgramWithExecutableLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics, "test-only executable launch slot is outside "
                                "ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only executable launch slot is outside ExecutionConfig");
  }
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      failAfterLaunchSlot, std::nullopt, std::nullopt,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithTargetLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics,
                   "test-only target launch slot is outside ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only target launch slot is outside ExecutionConfig");
  }
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      std::nullopt, failAfterLaunchSlot, std::nullopt,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithPackageLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics,
                   "test-only package launch slot is outside ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only package launch slot is outside ExecutionConfig");
  }
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      std::nullopt, std::nullopt, failAfterLaunchSlot,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithCommitVerificationFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::FailAfterVerification);
}

llvm::Expected<CompilationResult> compileProgramWithPackageBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::CorruptPackageProgramData);
}

llvm::Expected<CompilationResult> compileProgramWithProfileBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  if (!options.shouldProduceProfileInstrumentation())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only profile binding failure requires profile instrumentation");
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::CorruptProfilePlan);
}

} // namespace wafer::compiler::testing
