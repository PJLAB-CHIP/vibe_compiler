//===- CandidateEvaluation.cpp - Tile candidate implementation
//-----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Support/CanonicalIRSnapshot.h"
#include "Wafer/Support/OptimizationMechanism.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/OperationSupport.h"

#include <limits>

namespace wafer::tensor_program_scheduling {

static mlir::FailureOr<uint64_t>
getCandidateInvocationOrdinalBase(const SelectionConfig &config) {
  constexpr uint64_t rankCountLimit = 16;
  constexpr uint64_t rankVariantCount = 6;
  constexpr uint64_t scopeCountLimit = 4096;
  constexpr uint64_t candidateCountLimit = 64;
  // Each candidate owns four materialization subranges.  Three are currently
  // assigned to screening, retained-artifact materialization and commit proof;
  // the fourth remains reserved so adding a bounded typed attempt does not
  // renumber later candidates.
  constexpr uint64_t invocationRangeSize = uint64_t{1} << 14;
  if (config.logicalRank < 0 ||
      static_cast<uint64_t>(config.logicalRank) >= rankCountLimit ||
      config.rankVariantOrdinal >= rankVariantCount ||
      config.scopeOrdinal >= scopeCountLimit ||
      config.candidateOrdinal >= candidateCountLimit)
    return mlir::failure();
  uint64_t range = static_cast<uint64_t>(config.logicalRank);
  range = range * rankVariantCount + config.rankVariantOrdinal;
  range = range * scopeCountLimit + config.scopeOrdinal;
  range = range * candidateCountLimit + config.candidateOrdinal;
  return range * invocationRangeSize;
}

static mlir::FailureOr<uint64_t> getCandidateAttemptInvocationOrdinalBase(
    const SelectionConfig &config, CandidateEvaluationAttempt attempt) {
  constexpr uint64_t attemptRangeSize = uint64_t{1} << 12;
  mlir::FailureOr<uint64_t> candidateBase =
      getCandidateInvocationOrdinalBase(config);
  if (mlir::failed(candidateBase) ||
      static_cast<uint64_t>(attempt) >= 4)
    return mlir::failure();
  return *candidateBase + static_cast<uint64_t>(attempt) * attemptRangeSize;
}

static std::string
takeDiagnostics(mlir::MLIRContext *context,
                llvm::function_ref<mlir::LogicalResult()> callback,
                mlir::LogicalResult &result) {
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  result = callback();
  return diagnostics;
}

static std::string joinFailure(llvm::StringRef gate, llvm::StringRef reason,
                               llvm::StringRef diagnostics) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << gate << ": ";
  if (!reason.empty())
    os << reason;
  else if (!diagnostics.empty())
    os << diagnostics.trim();
  else
    os << "failed";
  return os.str();
}

static std::string joinInstructionFailure(llvm::StringRef gate,
                                          llvm::StringRef reason,
                                          llvm::StringRef diagnostics) {
  llvm::StringRef detail = reason.empty() ? diagnostics.trim() : reason.trim();
  constexpr llvm::StringLiteral targetNarrowingKey = "target_abi_narrowing:";
  size_t keyOffset = detail.find(targetNarrowingKey);
  if (keyOffset != llvm::StringRef::npos)
    return detail.drop_front(keyOffset).split('\n').first.str();
  return joinFailure(gate, reason, diagnostics);
}

static CandidateEvaluation
finishCandidateEvaluation(CandidateEvaluation evaluation,
                          const SelectionConfig &config,
                          CandidateEvaluationAttempt attempt) {
  mlir::MLIRContext *context = evaluation.module->getContext();
  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  mlir::FailureOr<uint64_t> invocationOrdinal =
      getCandidateAttemptInvocationOrdinalBase(config, attempt);
  if (mlir::failed(invocationOrdinal)) {
    evaluation.failureReason =
        "instr-lowering: invocation identity is out of range";
    return evaluation;
  }
  CanonicalIRSnapshotV1 inputSnapshot;
  if (!createCanonicalIRSnapshotV1(
          *evaluation.module, CanonicalIRSnapshotMode::SemanticStructure,
          inputSnapshot, &failureReason)) {
    evaluation.failureReason =
        joinInstructionFailure("instr-lowering-snapshot", failureReason, "");
    return evaluation;
  }
  OptimizationInvocationTokenV1 invocationToken;
  if (!beginOptimizationInvocationV1(
          mechanism::InstructionLowering,
          OptimizationCutPoint::SelectedPhysicalPayloadModule,
          *invocationOrdinal, inputSnapshot.sha256Digest, invocationToken,
          &failureReason)) {
    evaluation.failureReason =
        joinInstructionFailure("instr-lowering-begin", failureReason, "");
    return evaluation;
  }
  mlir::OperationFingerPrint before(*evaluation.module);
  uint64_t beforeWork = 0;
  evaluation.module->walk([&](mlir::Operation *) { ++beforeWork; });
  failureReason.clear();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return convertTileRegionToInstrModule(*evaluation.module,
                                              &failureReason);
      },
      result);
  mlir::OperationFingerPrint after(*evaluation.module);
  uint64_t afterWork = 0;
  evaluation.module->walk([&](mlir::Operation *) { ++afterWork; });
  OptimizationInvocationTelemetry telemetry;
  telemetry.key = mechanism::InstructionLowering;
  telemetry.cutPoint = OptimizationCutPoint::SelectedPhysicalPayloadModule;
  telemetry.invocationOrdinal = *invocationOrdinal;
  telemetry.inputSnapshotDigest = inputSnapshot.sha256Digest;
  telemetry.workUnits =
      std::numeric_limits<uint64_t>::max() - beforeWork < afterWork
          ? std::numeric_limits<uint64_t>::max()
          : beforeWork + afterWork;
  telemetry.outcome = mlir::failed(result)
                          ? InvocationOutcome::Invalid
                          : (before == after ? InvocationOutcome::NoChange
                                             : InvocationOutcome::Applied);
  telemetry.rewriteCount = telemetry.outcome == InvocationOutcome::Applied;
  std::string telemetryDiagnostic;
  if (!commitOptimizationInvocationV1(invocationToken, telemetry,
                                      &telemetryDiagnostic)) {
    evaluation.failureReason = joinInstructionFailure(
        "instr-lowering-terminal", telemetryDiagnostic, "");
    return evaluation;
  }
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinInstructionFailure("instr-lowering", failureReason, diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planSPMMemoryModule(*evaluation.module, config.spmBase,
                                   config.spmLimit, config.spmAlignment);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("spm-offsets", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planDDRMemoryModule(*evaluation.module, config.ddrAlignmentBytes,
                                   config.ddrCapacityBytes,
                                   config.ddrLargestContiguousBytes,
                                   config.ddrBandwidthLimitBytes);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("ddr-offsets", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context, [&]() { return mlir::verify(*evaluation.module); }, result);
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinInstructionFailure("verifier", "", diagnostics);
    return evaluation;
  }

  evaluation.stats = estimateStats(*evaluation.module);
  return evaluation;
}

static bool canUseFullTraversalFallback(
    const CandidateSpec &candidate,
    llvm::ArrayRef<int64_t> traversalShape) {
  return candidate.reductionSplitSizes.empty() &&
         candidate.tileSizes.size() == traversalShape.size() &&
         std::equal(candidate.tileSizes.begin(), candidate.tileSizes.end(),
                    traversalShape.begin(), traversalShape.end());
}

CandidateEvaluation evaluateCompleteCandidate(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config,
    CandidateEvaluationAttempt attempt) {
  CandidateEvaluation evaluation;
  mlir::FailureOr<uint64_t> invocationOrdinalBase =
      getCandidateAttemptInvocationOrdinalBase(config, attempt);
  if (mlir::failed(invocationOrdinalBase)) {
    evaluation.failureReason =
        "invocation-identity: candidate ordinal reservation is out of range";
    return evaluation;
  }
  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  std::string diagnostics = takeDiagnostics(
      task.getContext(),
      [&]() {
        return lowerCompleteCandidateTensorProgramToTileRegionModule(
            task, candidate.tileSizes, candidate.reductionSplitSizes,
            evaluation.module, &failureReason, config.logicalRank,
            *invocationOrdinalBase);
      },
      result);
  evaluation.artifactSource = CandidateArtifactSource::CompleteTraversalAPI;
  if (mlir::failed(result) &&
      canUseFullTraversalFallback(candidate, traversalShape)) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        task.getContext(),
        [&]() {
          return lowerTensorProgramToTileRegionModule(
              task, evaluation.module, &failureReason, config.logicalRank);
        },
        result);
    if (mlir::succeeded(result))
      evaluation.artifactSource =
          CandidateArtifactSource::FullTraversalFallback;
  }
  if (mlir::failed(result)) {
    evaluation.failureReason =
        joinFailure("complete-tile-region", failureReason, diagnostics);
    return evaluation;
  }
  return finishCandidateEvaluation(std::move(evaluation), config, attempt);
}

static void
registerSelectionEvaluationDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseStandaloneTaskModule(llvm::StringRef standaloneTaskModuleText,
                          mlir::MLIRContext &context,
                          std::string &failureReason) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::LogicalResult parseResult = mlir::success();
  std::string diagnostics = takeDiagnostics(
      &context,
      [&]() {
        module = mlir::parseSourceString<mlir::ModuleOp>(
            standaloneTaskModuleText, &context);
        return module ? mlir::success() : mlir::failure();
      },
      parseResult);
  if (mlir::failed(parseResult) || !module) {
    failureReason = joinFailure("parse-standalone", "", diagnostics);
    return nullptr;
  }
  return module;
}

static CandidateCheckResult evaluateTaskCandidate(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config,
    bool retainAcceptedModule) {
  CandidateCheckResult result;
  result.spec = candidate;
  result.candidateOrdinal = config.candidateOrdinal;
  llvm::SmallVector<TileInstance, 8> reps =
      buildRepresentativeTiles(traversalShape, candidate.tileSizes);
  result.representativeCount = static_cast<int64_t>(reps.size());

  // The compact complete traversal contains the main tile, every static tail
  // and every tail corner in the same artifact that can be committed.  Running
  // each representative as an independent function is neither an additional
  // coverage proof nor a sound resource bound: a partial function must
  // reconstruct its full output destination and can therefore allocate a
  // full-shape SPM buffer that does not exist in the complete traversal.
  // Keep the representative count for deterministic diagnostics, but run all
  // instruction, SPM, DDR and cost gates exactly once on the all-and-only
  // complete artifact.
  CandidateEvaluation acceptedEvaluation =
      evaluateCompleteCandidate(
          task, traversalShape, candidate, config,
          retainAcceptedModule ? CandidateEvaluationAttempt::RetainedArtifact
                               : CandidateEvaluationAttempt::Screening);
  if (!acceptedEvaluation.failureReason.empty()) {
    result.failureReason = acceptedEvaluation.failureReason;
    return result;
  }
  if (!isCompleteArtifactSource(acceptedEvaluation.artifactSource)) {
    result.failureReason =
        "complete-artifact: accepted candidate has representative-only "
        "provenance";
    return result;
  }
  result.stats = acceptedEvaluation.stats;
  result.artifactSource = acceptedEvaluation.artifactSource;
  if (retainAcceptedModule)
    result.module = std::move(acceptedEvaluation.module);
  return result;
}

CandidateCheckResult evaluateCandidateOnOriginalTask(
    mlir::func::FuncOp task, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const SelectionConfig &config) {
  return evaluateTaskCandidate(task, traversalShape, candidate, config,
                               /*retainAcceptedModule=*/true);
}

CandidateCheckResult evaluateCandidateOnStandaloneTaskText(
    llvm::StringRef standaloneTaskModuleText,
    llvm::ArrayRef<int64_t> traversalShape, const CandidateSpec &candidate,
    const SelectionConfig &config) {
  CandidateCheckResult result;
  result.spec = candidate;
  result.candidateOrdinal = config.candidateOrdinal;
  mlir::DialectRegistry registry;
  registerSelectionEvaluationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string parseFailure;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseStandaloneTaskModule(
      standaloneTaskModuleText, context, parseFailure);
  if (!module) {
    result.failureReason = parseFailure;
    return result;
  }
  mlir::func::FuncOp task = findSingleSelectionTask(*module);
  if (!task) {
    result.failureReason =
        "parse-standalone: standalone module has no scheduling task";
    return result;
  }
  return evaluateTaskCandidate(task, traversalShape, candidate, config,
                               /*retainAcceptedModule=*/false);
}

} // namespace wafer::tensor_program_scheduling
