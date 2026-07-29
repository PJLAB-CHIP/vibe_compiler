//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "CompilationInternal.h"
#include "CompilationStatistics.h"
#include "NoCResidentDataflow.h"
#include "ScheduledRankFinalization.h"
#include "WholeVariantAttemptPlan.h"
#include "WholeVariantCoordinator.h"

#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

llvm::Expected<std::vector<detail::RankVariantFrontier>>
detail::importRankVariantFrontiersIntoOwnerContext(
    mlir::MLIRContext &ownerContext,
    llvm::ArrayRef<SerializedRankVariantFrontier> serializedFrontiers,
    int64_t expectedRankCount) {
  std::vector<RankVariantMetadataFrontier> frontierMetadata;
  frontierMetadata.reserve(serializedFrontiers.size());
  for (const SerializedRankVariantFrontier &frontier : serializedFrontiers) {
    RankVariantMetadataFrontier metadata;
    metadata.reserve(frontier.size());
    for (const SerializedRankVariantCandidate &candidate : frontier)
      metadata.push_back({candidate.stableOrdinal, candidate.artifactKind,
                          candidate.reservedBaseline, candidate.bufferingKind,
                          candidate.bufferingPlanOrdinal,
                          candidate.workerPlacementKind,
                          candidate.workerPlacementPlanOrdinal});
    frontierMetadata.push_back(std::move(metadata));
  }

  WholeVariantAttemptPlan attemptPlan =
      buildWholeVariantAttemptPlan(frontierMetadata, expectedRankCount);
  if (attemptPlan.requiredModuleIndices.size() != serializedFrontiers.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "internal rank-frontier import plan has an incomplete rank domain");

  std::vector<RankVariantFrontier> frontiers;
  frontiers.reserve(serializedFrontiers.size());
  for (auto [rankIndex, serialized] : llvm::enumerate(serializedFrontiers)) {
    RankVariantFrontier imported;
    imported.reserve(serialized.size());
    std::vector<bool> required(serialized.size(), false);
    for (size_t index : attemptPlan.requiredModuleIndices[rankIndex]) {
      if (index >= required.size())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "internal rank-frontier import plan references an invalid "
            "candidate slot");
      required[index] = true;
    }
    for (auto [candidateIndex, candidate] : llvm::enumerate(serialized)) {
      mlir::OwningOpRef<mlir::ModuleOp> module;
      if (required[candidateIndex]) {
        if (!candidate.moduleData)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "required rank-frontier candidate has no serialized module "
              "data");
        module = mlir::parseSourceString<mlir::ModuleOp>(*candidate.moduleData,
                                                         &ownerContext);
        if (!module)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "failed to import a lowered scheduling candidate for logical "
              "rank %zu",
              static_cast<size_t>(rankIndex));
      }
      imported.push_back(
          {std::move(module), candidate.stableOrdinal, candidate.artifactKind,
           candidate.reservedBaseline, candidate.bufferingKind,
           candidate.bufferingPlanOrdinal, candidate.workerPlacementKind,
           candidate.workerPlacementPlanOrdinal});
    }
    frontiers.push_back(std::move(imported));
  }
  return frontiers;
}

void detail::compactImportedRankVariantFrontiers(
    std::vector<RankVariantFrontier> &frontiers) {
  for (RankVariantFrontier &frontier : frontiers)
    llvm::erase_if(frontier, [](const RankVariantCandidate &candidate) {
      return !candidate.module;
    });
}

static llvm::Expected<ExecutableBundle> buildExecutableBundleImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    detail::WholeVariantSelectionMode selectionMode) {
  const detail::CompileClock::time_point totalStart =
      detail::CompileClock::now();
  constexpr int64_t candidateEvaluationWorkerLimit = 4;
  constexpr uint32_t rankRequestShardLimit = 16;
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  struct RankSearchShardResult {
    int64_t logicalRank;
    uint32_t generationClass;
    uint32_t requestShardIndex;
    bool succeeded = false;
    int64_t wallMs = 0;
    // Modules must be destroyed before the context that owns their uniqued
    // state. Member destruction is reverse declaration order.
    std::unique_ptr<mlir::MLIRContext> context;
    std::vector<wafer::ScheduledRankCandidate> candidates;
    std::string diagnostics;
  };
  struct RankLoweringResult {
    int64_t logicalRank;
    std::unique_ptr<mlir::MLIRContext> context;
    std::vector<detail::FinalizedRankCandidate> candidates;
  };

  std::string tensorModuleData;
  llvm::raw_string_ostream tensorModuleStream(tensorModuleData);
  if (mlir::failed(mlir::writeBytecodeToFile(tensorModule.getOperation(),
                                             tensorModuleStream)))
    return fail("failed to encode the verified tensor program for rank "
                "lowering");
  tensorModuleStream.flush();

  const bool rankInvariant =
      wafer::isTensorProgramSchedulingRankInvariant(tensorModule);
  const int64_t rankGenerationClassCount =
      rankInvariant ? 1 : executionConfig.getRankCount();
  const uint32_t requestShardCount = rankInvariant ? rankRequestShardLimit : 1;
  std::vector<RankSearchShardResult> searchShardResults;
  searchShardResults.reserve(static_cast<size_t>(rankGenerationClassCount) *
                             requestShardCount);
  for (int64_t generationClass = 0; generationClass < rankGenerationClassCount;
       ++generationClass)
    for (uint32_t requestShardIndex = 0; requestShardIndex < requestShardCount;
         ++requestShardIndex)
      searchShardResults.push_back({generationClass,
                                    static_cast<uint32_t>(generationClass),
                                    requestShardIndex});

  // Candidate evaluation installs diagnostic handlers and creates transient
  // IR. Give every rank-dependent generation class its own request shard(s)
  // and context. A tensor program without typed collectives has one generation
  // class whose deterministic request groups can run independently. Shard
  // results are merged in the original request order and pass the original
  // global admission policy before rank finalization.
  const detail::CompileClock::time_point rankGenerationStart =
      detail::CompileClock::now();
  llvm::parallelFor(0, searchShardResults.size(), [&](size_t index) {
    RankSearchShardResult &result = searchShardResults[index];
    const detail::CompileClock::time_point shardStart =
        detail::CompileClock::now();
    mlir::DialectRegistry registry;
    detail::registerCompilationDialects(registry);
    auto rankContext = std::make_unique<mlir::MLIRContext>(registry);
    // Rank pipelines are already the unit of parallelism. Avoid creating a
    // nested context thread pool for each of the 16 concurrent workers.
    rankContext->disableMultithreading();
    rankContext->loadAllAvailableDialects();
    mlir::ScopedDiagnosticHandler diagnosticHandler(
        rankContext.get(), [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream os(result.diagnostics);
          diagnostic.print(os);
          os << "\n";
          return mlir::success();
        });

    mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
        mlir::parseSourceString<mlir::ModuleOp>(tensorModuleData,
                                                rankContext.get());
    if (!sourceModule)
      return;
    wafer::TensorProgramSchedulingConfig schedulingConfig;
    schedulingConfig.logicalRank = result.logicalRank;
    schedulingConfig.candidateParallelism = candidateEvaluationWorkerLimit;
    schedulingConfig.requestShardIndex = result.requestShardIndex;
    schedulingConfig.requestShardCount = requestShardCount;
    schedulingConfig.targetProfile = executionConfig.getTargetProfileId();
    mlir::FailureOr<std::vector<wafer::ScheduledRankCandidate>> frontier =
        wafer::buildScheduledRankCandidateFrontier(*sourceModule,
                                                   schedulingConfig);
    if (mlir::failed(frontier) || frontier->empty()) {
      if (result.requestShardIndex == 0)
        return;
    }
    if (mlir::failed(frontier))
      return;

    result.candidates = std::move(*frontier);
    result.context = std::move(rankContext);
    result.wallMs = detail::elapsedCompileMilliseconds(shardStart);
    result.succeeded = true;
  });

  for (RankSearchShardResult &result : searchShardResults) {
    if (!result.succeeded) {
      diagnostics << result.diagnostics;
      return fail("rank scheduling search failed for logical rank " +
                  std::to_string(result.logicalRank) + " request shard " +
                  std::to_string(result.requestShardIndex));
    }
    diagnostics << "wafer-compile: compile-stats stage=rank-request-shard"
                << " logical_rank=" << result.logicalRank
                << " request_shard=" << result.requestShardIndex
                << " request_shard_count=" << requestShardCount
                << " wall_ms=" << result.wallMs
                << " candidate_count=" << result.candidates.size() << "\n";
  }
  if (failAfterLogicalRank && *failAfterLogicalRank >= 0 &&
      *failAfterLogicalRank < executionConfig.getRankCount())
    return fail("test-only injected failure after logical rank " +
                std::to_string(*failAfterLogicalRank));

  std::vector<RankLoweringResult> loweringResults;
  loweringResults.reserve(rankGenerationClassCount);
  for (int64_t generationClass = 0; generationClass < rankGenerationClassCount;
       ++generationClass) {
    mlir::DialectRegistry registry;
    detail::registerCompilationDialects(registry);
    auto generationContext = std::make_unique<mlir::MLIRContext>(registry);
    generationContext->disableMultithreading();
    generationContext->loadAllAvailableDialects();

    std::vector<wafer::ScheduledRankCandidate *> shardCandidates;
    for (RankSearchShardResult &shard : searchShardResults)
      if (shard.generationClass == static_cast<uint32_t>(generationClass))
        for (wafer::ScheduledRankCandidate &candidate : shard.candidates)
          shardCandidates.push_back(&candidate);
    llvm::stable_sort(shardCandidates, [](const auto *lhs, const auto *rhs) {
      return lhs->frontierOrderOrdinal < rhs->frontierOrderOrdinal;
    });

    wafer::RankFrontierAdmissionState admission;
    std::vector<wafer::ScheduledRankCandidate> merged;
    merged.reserve(wafer::kMaximumScheduledRankFrontierSize);
    for (wafer::ScheduledRankCandidate *candidate : shardCandidates) {
      if (!candidate->reservedBaseline &&
          !admission.tryAdmit(candidate->bufferingKind,
                              candidate->workerPlacementKind))
        continue;
      std::string moduleData;
      llvm::raw_string_ostream moduleStream(moduleData);
      if (mlir::failed(mlir::writeBytecodeToFile(
              candidate->module.get().getOperation(), moduleStream)))
        return fail("failed to encode a rank scheduling search-shard "
                    "candidate");
      moduleStream.flush();
      mlir::OwningOpRef<mlir::ModuleOp> imported =
          mlir::parseSourceString<mlir::ModuleOp>(moduleData,
                                                  generationContext.get());
      if (!imported)
        return fail("failed to import a rank scheduling search-shard "
                    "candidate");
      merged.emplace_back(
          std::move(imported), candidate->stableOrdinal,
          candidate->artifactKind, candidate->reservedBaseline,
          candidate->bufferingKind, candidate->bufferingPlanOrdinal,
          candidate->workerPlacementKind, candidate->workerPlacementPlanOrdinal,
          candidate->frontierOrderOrdinal);
    }

    mlir::FailureOr<std::vector<detail::FinalizedRankCandidate>> finalized =
        detail::finalizeScheduledRankCandidateFrontier(
            std::move(merged), executionConfig.getTargetProfileId());
    if (mlir::failed(finalized))
      return fail("rank scheduling finalization failed for logical rank " +
                  std::to_string(generationClass));
    loweringResults.push_back(
        {generationClass, std::move(generationContext), std::move(*finalized)});
  }
  for (RankSearchShardResult &result : searchShardResults) {
    result.candidates.clear();
    result.context.reset();
  }
  const int64_t rankGenerationWallMs =
      detail::elapsedCompileMilliseconds(rankGenerationStart);

  uint64_t generatedCandidateCount = 0;
  for (RankLoweringResult &result : loweringResults) {
    const int64_t logicalRank = result.logicalRank;
    if (result.candidates.empty())
      return fail("rank lowering failed for logical rank " +
                  std::to_string(logicalRank));
    generatedCandidateCount += result.candidates.size();
  }

  // Compute the exact bounded context-transfer plan from cheap metadata while
  // every finalized module still belongs to its worker context. Only modules
  // reachable through that plan are encoded; original slot positions and all
  // metadata remain available to the owner import audit.
  const detail::CompileClock::time_point transferStart =
      detail::CompileClock::now();
  std::vector<detail::RankVariantMetadataFrontier> generatedMetadata;
  generatedMetadata.reserve(loweringResults.size());
  for (const RankLoweringResult &result : loweringResults) {
    detail::RankVariantMetadataFrontier metadata;
    metadata.reserve(result.candidates.size());
    for (const detail::FinalizedRankCandidate &candidate : result.candidates)
      metadata.push_back({candidate.stableOrdinal, candidate.artifactKind,
                          candidate.reservedBaseline, candidate.bufferingKind,
                          candidate.bufferingPlanOrdinal,
                          candidate.workerPlacementKind,
                          candidate.workerPlacementPlanOrdinal});
    generatedMetadata.push_back(std::move(metadata));
  }
  uint64_t frontierCandidateSlotCount = 0;
  std::vector<detail::RankVariantMetadataFrontier> frontierMetadata;
  frontierMetadata.reserve(executionConfig.getRankCount());
  for (int64_t logicalRank = 0; logicalRank < executionConfig.getRankCount();
       ++logicalRank) {
    const size_t generationClass =
        rankInvariant ? 0 : static_cast<size_t>(logicalRank);
    frontierCandidateSlotCount += generatedMetadata[generationClass].size();
    frontierMetadata.push_back(generatedMetadata[generationClass]);
  }
  detail::WholeVariantAttemptPlan importPlan =
      detail::buildWholeVariantAttemptPlan(frontierMetadata,
                                           executionConfig.getRankCount());

  std::vector<std::vector<bool>> requiredByGeneration;
  requiredByGeneration.reserve(loweringResults.size());
  for (const RankLoweringResult &result : loweringResults)
    requiredByGeneration.emplace_back(result.candidates.size(), false);
  for (size_t rankIndex = 0;
       rankIndex < importPlan.requiredModuleIndices.size(); ++rankIndex) {
    const size_t generationClass = rankInvariant ? 0 : rankIndex;
    std::vector<bool> &required = requiredByGeneration[generationClass];
    for (size_t candidateIndex : importPlan.requiredModuleIndices[rankIndex]) {
      if (candidateIndex >= required.size())
        return fail("internal rank-frontier transfer plan references an "
                    "invalid candidate slot");
      required[candidateIndex] = true;
    }
  }

  uint64_t serializedModuleBytes = 0;
  uint64_t uniqueEncodedModuleCount = 0;
  std::vector<detail::SerializedRankVariantFrontier>
      serializedGenerationFrontiers;
  serializedGenerationFrontiers.reserve(loweringResults.size());
  for (auto [generationClass, result] : llvm::enumerate(loweringResults)) {
    const std::vector<bool> &required = requiredByGeneration[generationClass];
    detail::SerializedRankVariantFrontier serializedFrontier;
    serializedFrontier.reserve(result.candidates.size());
    for (auto [candidateIndex, candidate] :
         llvm::enumerate(result.candidates)) {
      detail::SerializedRankVariantCandidate serialized;
      serialized.stableOrdinal = candidate.stableOrdinal;
      serialized.artifactKind = candidate.artifactKind;
      serialized.reservedBaseline = candidate.reservedBaseline;
      serialized.bufferingKind = candidate.bufferingKind;
      serialized.bufferingPlanOrdinal = candidate.bufferingPlanOrdinal;
      serialized.workerPlacementKind = candidate.workerPlacementKind;
      serialized.workerPlacementPlanOrdinal =
          candidate.workerPlacementPlanOrdinal;
      if (required[candidateIndex]) {
        std::string moduleData;
        llvm::raw_string_ostream moduleStream(moduleData);
        if (mlir::failed(mlir::writeBytecodeToFile(
                candidate.module.get().getOperation(), moduleStream)))
          return fail("failed to encode a required rank scheduling candidate");
        moduleStream.flush();
        serializedModuleBytes += moduleData.size();
        serialized.moduleData =
            std::make_shared<const std::string>(std::move(moduleData));
        ++uniqueEncodedModuleCount;
      }
      serializedFrontier.push_back(std::move(serialized));
    }
    serializedGenerationFrontiers.push_back(std::move(serializedFrontier));
    result.candidates.clear();
    result.context.reset();
  }

  std::vector<detail::SerializedRankVariantFrontier> serializedFrontiers;
  serializedFrontiers.reserve(executionConfig.getRankCount());
  for (int64_t logicalRank = 0; logicalRank < executionConfig.getRankCount();
       ++logicalRank) {
    const size_t generationClass =
        rankInvariant ? 0 : static_cast<size_t>(logicalRank);
    serializedFrontiers.push_back(
        serializedGenerationFrontiers[generationClass]);
  }
  const int64_t transferWallMs =
      detail::elapsedCompileMilliseconds(transferStart);

  // Reproduce the coordinator's original bounded attempt sequence from cheap
  // metadata before parsing any finalized module into the bundle-owner
  // context. Original slot positions remain intact only for this import audit:
  // slots referenced by a correspondence-valid attempted tuple (plus every
  // reserved baseline) materialize their module.
  const detail::CompileClock::time_point importStart =
      detail::CompileClock::now();
  llvm::Expected<std::vector<detail::RankVariantFrontier>> imported =
      detail::importRankVariantFrontiersIntoOwnerContext(
          *context, serializedFrontiers, executionConfig.getRankCount());
  if (!imported)
    return fail(llvm::toString(imported.takeError()));
  std::vector<detail::RankVariantFrontier> frontiers = std::move(*imported);
  const int64_t importWallMs = detail::elapsedCompileMilliseconds(importStart);

  // The import boundary preserves metadata-only slots long enough to audit
  // the exact bounded context-transfer plan and diagnose a required parse
  // failure. They are not artifacts, however, and must not enter a later
  // attempt-plan recomputation after semantic transformations append fresh
  // materialized candidates. Keep only actual owner-context IR before the
  // transformation/selection frontier starts evolving.
  detail::compactImportedRankVariantFrontiers(frontiers);

  const detail::CompileClock::time_point dataflowStart =
      detail::CompileClock::now();
  std::string dataflowFailure;
  if (mlir::failed(detail::appendNoCResidentDataflowCandidates(
          frontiers, program, executionConfig, &dataflowFailure)))
    return fail("all-rank NoC-resident candidate construction failed: " +
                dataflowFailure);
  const int64_t dataflowWallMs =
      detail::elapsedCompileMilliseconds(dataflowStart);

  const detail::CompileClock::time_point selectionStart =
      detail::CompileClock::now();
  detail::WholeVariantSelectionStatistics selectionStatistics;
  mlir::FailureOr<detail::AcceptedWholeVariant> accepted =
      detail::selectAcceptedWholeVariant(frontiers, program, executionConfig,
                                         diagnostics, selectionMode,
                                         &selectionStatistics);
  if (mlir::failed(accepted))
    return fail("whole-variant coordination failed");
  const int64_t selectionWallMs =
      detail::elapsedCompileMilliseconds(selectionStart);

  diagnostics << "wafer-compile: compile-stats stage=rank-candidate-generation"
              << " wall_ms=" << rankGenerationWallMs
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
              << " rank_count=" << executionConfig.getRankCount()
              << " rank_generation_classes=" << rankGenerationClassCount
              << " rank_invariant=" << (rankInvariant ? "true" : "false")
              << " request_shards_per_class=" << requestShardCount
              << " candidate_workers=" << candidateEvaluationWorkerLimit
              << " candidate_count=" << generatedCandidateCount
              << " per_rank_candidate_limit="
              << wafer::kMaximumScheduledRankFrontierSize << "\n";
  diagnostics << "wafer-compile: compile-stats stage=frontier-transfer"
              << " wall_ms=" << transferWallMs
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
              << " candidate_slots=" << frontierCandidateSlotCount
              << " encoded_modules=" << uniqueEncodedModuleCount
              << " imported_modules=" << importPlan.getRequiredModuleCount()
              << " encoded_bytes=" << serializedModuleBytes << "\n";
  diagnostics << "wafer-compile: compile-stats stage=owner-import"
              << " wall_ms=" << importWallMs
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
              << " imported_modules=" << importPlan.getRequiredModuleCount()
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=noc-candidate-expansion"
              << " wall_ms=" << dataflowWallMs
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
              << " frontier_candidates="
              << selectionStatistics.frontierCandidateCount << "\n";
  diagnostics << "wafer-compile: compile-stats stage=whole-variant-selection"
              << " wall_ms=" << selectionWallMs
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
              << " planned_attempts=" << selectionStatistics.plannedAttemptCount
              << " planned_attempt_limit="
              << selectionStatistics.plannedAttemptLimit
              << " pretarget_attempts=" << selectionStatistics.preTargetAttempts
              << " pretarget_accepted=" << selectionStatistics.preTargetAccepted
              << " target_gate_attempts="
              << selectionStatistics.targetGateInvocations
              << " target_rank_lowerings="
              << selectionStatistics.targetRankGateInvocations
              << " fully_accepted=" << selectionStatistics.fullyAcceptedVariants
              << " pareto_retained="
              << selectionStatistics.paretoRetainedVariants
              << " rank_clone_upper_bound="
              << selectionStatistics.preTargetAttempts *
                     static_cast<uint64_t>(executionConfig.getRankCount())
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=executable-bundle"
              << " wall_ms=" << detail::elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB() << "\n";

  auto makeBundle = [&](detail::AcceptedWholeVariant variant)
      -> llvm::Expected<ExecutableBundle> {
    std::vector<RankExecutable> ranks = std::move(variant.ranks);
    if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
      return fail("executable bundle rank domain is incomplete");
    for (auto [expectedRank, rank] : llvm::enumerate(ranks))
      if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
        return fail("executable bundle rank domain is not canonical");
    return ExecutableBundleBuilder::makeBundle(
        executionConfig, std::move(variant.runtimeLaunchContract), context,
        std::move(ranks));
  };

  return makeBundle(std::move(*accepted));
}

llvm::Expected<ExecutableBundle> detail::buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode) {
  return buildExecutableBundleImpl(context, tensorModule, std::move(program),
                                   executionConfig, diagnostics,
                                   failAfterLogicalRank, selectionMode);
}

} // namespace wafer::compiler
