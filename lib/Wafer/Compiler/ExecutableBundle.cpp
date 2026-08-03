//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "CompilationInternal.h"
#include "CompilationStatistics.h"
#include "NoCResidentDataflow.h"
#include "ScheduledRankFinalization.h"
#include "WholeVariantAttemptPlan.h"
#include "WholeVariantCoordinator.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Threading.h"
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
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    detail::WholeVariantSelectionMode selectionMode) {
  const detail::CompileClock::time_point totalStart =
      detail::CompileClock::now();
  wafer::support::ScopedCompileTimingSpan executableBundleTiming(
      "stage", "tensor-program-to-executable", "executable-bundle");
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
    std::vector<detail::FinalizedRankCandidate> finalizedCandidates;
    std::string diagnostics;
    bool finalizationSucceeded = false;
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
  const unsigned heavyweightThreadCount =
      llvm::heavyweight_hardware_concurrency().compute_thread_count();
  const uint32_t requestShardCount = std::min<uint32_t>(
      rankRequestShardLimit,
      std::max<uint32_t>(1,
                         heavyweightThreadCount /
                             static_cast<uint32_t>(rankGenerationClassCount)));
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
  auto rankGenerationTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "tensor-program-to-executable", "rank-candidate-generation");
  std::shared_ptr<wafer::support::CompileTimingSession> timingSession =
      wafer::support::getActiveCompileTimingSession();
  llvm::parallelFor(0, searchShardResults.size(), [&](size_t index) {
    wafer::support::ScopedCompileTimingActivation timingActivation(
        timingSession);
    RankSearchShardResult &result = searchShardResults[index];
    const detail::CompileClock::time_point shardStart =
        detail::CompileClock::now();
    std::string timingDetail;
    llvm::raw_string_ostream timingDetailStream(timingDetail);
    timingDetailStream << "logical-rank=" << result.logicalRank
                       << ",generation-class=" << result.generationClass
                       << ",request-shard=" << result.requestShardIndex;
    wafer::support::ScopedCompileTimingSpan shardTiming(
        "search", "rank-candidate-generation", "rank-request-shard",
        timingDetailStream.str());
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
    schedulingConfig.optimizations = optimizations;
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

  // Reapply the one canonical rank-frontier admission policy across request
  // shards before finalization. The selected candidates stay in their
  // disjoint shard contexts, so expensive bufferization and SPM replanning can
  // use the same request-shard parallelism as search without changing which
  // frontier members are admitted.
  struct CandidateLocation {
    size_t shardIndex;
    size_t candidateIndex;
  };
  std::vector<std::vector<bool>> admittedMasks;
  admittedMasks.reserve(searchShardResults.size());
  for (const RankSearchShardResult &result : searchShardResults)
    admittedMasks.emplace_back(result.candidates.size(), false);
  for (size_t generationClass = 0;
       generationClass < static_cast<size_t>(rankGenerationClassCount);
       ++generationClass) {
    std::vector<CandidateLocation> locations;
    for (auto [shardIndex, shard] : llvm::enumerate(searchShardResults))
      if (shard.generationClass == generationClass)
        for (size_t candidateIndex = 0;
             candidateIndex < shard.candidates.size(); ++candidateIndex)
          locations.push_back({shardIndex, candidateIndex});
    llvm::stable_sort(
        locations, [&](CandidateLocation lhs, CandidateLocation rhs) {
          return searchShardResults[lhs.shardIndex]
                     .candidates[lhs.candidateIndex]
                     .frontierOrderOrdinal < searchShardResults[rhs.shardIndex]
                                                 .candidates[rhs.candidateIndex]
                                                 .frontierOrderOrdinal;
        });
    wafer::RankFrontierAdmissionState admission;
    unsigned baselineCount = 0;
    for (CandidateLocation location : locations) {
      const wafer::ScheduledRankCandidate &candidate =
          searchShardResults[location.shardIndex]
              .candidates[location.candidateIndex];
      baselineCount += candidate.reservedBaseline;
      if (!candidate.reservedBaseline &&
          !admission.tryAdmit(candidate.bufferingKind,
                              candidate.workerPlacementKind))
        continue;
      admittedMasks[location.shardIndex][location.candidateIndex] = true;
    }
    if (baselineCount != 1)
      return fail("rank scheduling search did not produce exactly one "
                  "reserved baseline for generation class " +
                  std::to_string(generationClass));
  }
  for (auto [shardIndex, result] : llvm::enumerate(searchShardResults)) {
    std::vector<wafer::ScheduledRankCandidate> admitted;
    admitted.reserve(result.candidates.size());
    for (auto [candidateIndex, candidate] : llvm::enumerate(result.candidates))
      if (admittedMasks[shardIndex][candidateIndex])
        admitted.push_back(std::move(candidate));
    result.candidates = std::move(admitted);
  }

  llvm::parallelFor(0, searchShardResults.size(), [&](size_t shardIndex) {
    RankSearchShardResult &result = searchShardResults[shardIndex];
    wafer::support::ScopedCompileTimingActivation timingActivation(
        timingSession);
    std::string timingDetail;
    llvm::raw_string_ostream timingDetailStream(timingDetail);
    timingDetailStream << "generation-class=" << result.generationClass
                       << ",request-shard=" << result.requestShardIndex;
    wafer::support::ScopedCompileTimingSpan finalizationTiming(
        "search", "rank-candidate-generation", "rank-frontier-finalization",
        timingDetailStream.str());
    mlir::ScopedDiagnosticHandler diagnosticHandler(
        result.context.get(), [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream os(result.diagnostics);
          diagnostic.print(os);
          os << "\n";
          return mlir::success();
        });
    mlir::FailureOr<std::vector<detail::FinalizedRankCandidate>> finalized =
        detail::finalizeScheduledRankCandidateFrontier(
            std::move(result.candidates), executionConfig.getTargetProfileId(),
            /*requireReservedBaseline=*/false);
    if (mlir::failed(finalized))
      return;
    result.finalizedCandidates = std::move(*finalized);
    result.finalizationSucceeded = true;
  });

  struct FinalizedCandidateLocation {
    RankSearchShardResult *shard;
    size_t candidateIndex;
  };
  std::vector<std::vector<FinalizedCandidateLocation>>
      finalizedGenerationCandidates(
          static_cast<size_t>(rankGenerationClassCount));
  for (RankSearchShardResult &result : searchShardResults) {
    if (!result.finalizationSucceeded) {
      diagnostics << result.diagnostics;
      return fail("rank scheduling finalization failed for logical rank " +
                  std::to_string(result.logicalRank) + " request shard " +
                  std::to_string(result.requestShardIndex));
    }
    for (size_t candidateIndex = 0;
         candidateIndex < result.finalizedCandidates.size(); ++candidateIndex)
      finalizedGenerationCandidates[result.generationClass].push_back(
          {&result, candidateIndex});
  }
  uint64_t generatedCandidateCount = 0;
  for (auto [generationClass, locations] :
       llvm::enumerate(finalizedGenerationCandidates)) {
    llvm::stable_sort(locations, [](FinalizedCandidateLocation lhs,
                                    FinalizedCandidateLocation rhs) {
      return lhs.shard->finalizedCandidates[lhs.candidateIndex]
                 .frontierOrderOrdinal <
             rhs.shard->finalizedCandidates[rhs.candidateIndex]
                 .frontierOrderOrdinal;
    });
    unsigned baselineCount = 0;
    for (FinalizedCandidateLocation location : locations)
      baselineCount +=
          location.shard->finalizedCandidates[location.candidateIndex]
              .reservedBaseline;
    if (locations.empty() || baselineCount != 1)
      return fail("rank lowering failed for logical rank " +
                  std::to_string(generationClass));
    generatedCandidateCount += locations.size();
  }
  const int64_t rankGenerationWallMs =
      detail::elapsedCompileMilliseconds(rankGenerationStart);
  rankGenerationTiming.reset();

  // Compute the exact bounded context-transfer plan from cheap metadata while
  // every finalized module still belongs to its worker context. Only modules
  // reachable through that plan are encoded; original slot positions and all
  // metadata remain available to the owner import audit.
  const detail::CompileClock::time_point transferStart =
      detail::CompileClock::now();
  auto transferTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "tensor-program-to-executable", "frontier-transfer");
  std::vector<detail::RankVariantMetadataFrontier> generatedMetadata;
  generatedMetadata.reserve(finalizedGenerationCandidates.size());
  for (const std::vector<FinalizedCandidateLocation> &locations :
       finalizedGenerationCandidates) {
    detail::RankVariantMetadataFrontier metadata;
    metadata.reserve(locations.size());
    for (FinalizedCandidateLocation location : locations) {
      const detail::FinalizedRankCandidate &candidate =
          location.shard->finalizedCandidates[location.candidateIndex];
      metadata.push_back({candidate.stableOrdinal, candidate.artifactKind,
                          candidate.reservedBaseline, candidate.bufferingKind,
                          candidate.bufferingPlanOrdinal,
                          candidate.workerPlacementKind,
                          candidate.workerPlacementPlanOrdinal});
    }
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
  requiredByGeneration.reserve(finalizedGenerationCandidates.size());
  for (const std::vector<FinalizedCandidateLocation> &locations :
       finalizedGenerationCandidates)
    requiredByGeneration.emplace_back(locations.size(), false);
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
  serializedGenerationFrontiers.reserve(finalizedGenerationCandidates.size());
  for (auto [generationClass, locations] :
       llvm::enumerate(finalizedGenerationCandidates)) {
    const std::vector<bool> &required = requiredByGeneration[generationClass];
    detail::SerializedRankVariantFrontier serializedFrontier;
    serializedFrontier.reserve(locations.size());
    for (auto [candidateIndex, location] : llvm::enumerate(locations)) {
      detail::FinalizedRankCandidate &candidate =
          location.shard->finalizedCandidates[location.candidateIndex];
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
  }
  for (RankSearchShardResult &result : searchShardResults) {
    result.candidates.clear();
    result.finalizedCandidates.clear();
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
  transferTiming.reset();

  // Reproduce the coordinator's original bounded attempt sequence from cheap
  // metadata before parsing any finalized module into the bundle-owner
  // context. Original slot positions remain intact only for this import audit:
  // slots referenced by a correspondence-valid attempted tuple (plus every
  // reserved baseline) materialize their module.
  const detail::CompileClock::time_point importStart =
      detail::CompileClock::now();
  auto importTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "stage", "tensor-program-to-executable", "owner-import");
  llvm::Expected<std::vector<detail::RankVariantFrontier>> imported =
      detail::importRankVariantFrontiersIntoOwnerContext(
          *context, serializedFrontiers, executionConfig.getRankCount());
  if (!imported)
    return fail(llvm::toString(imported.takeError()));
  std::vector<detail::RankVariantFrontier> frontiers = std::move(*imported);
  const int64_t importWallMs = detail::elapsedCompileMilliseconds(importStart);
  importTiming.reset();

  // The import boundary preserves metadata-only slots long enough to audit
  // the exact bounded context-transfer plan and diagnose a required parse
  // failure. They are not artifacts, however, and must not enter a later
  // attempt-plan recomputation after semantic transformations append fresh
  // materialized candidates. Keep only actual owner-context IR before the
  // transformation/selection frontier starts evolving.
  detail::compactImportedRankVariantFrontiers(frontiers);

  const detail::CompileClock::time_point dataflowStart =
      detail::CompileClock::now();
  auto dataflowTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "tensor-program-to-executable", "noc-candidate-expansion");
  std::string dataflowFailure;
  if (optimizations.isEnabled(OptimizationKind::NoCResidentDataflow) &&
      mlir::failed(detail::appendNoCResidentDataflowCandidates(
          frontiers, program, executionConfig, &dataflowFailure)))
    return fail("all-rank NoC-resident candidate construction failed: " +
                dataflowFailure);
  const int64_t dataflowWallMs =
      detail::elapsedCompileMilliseconds(dataflowStart);
  dataflowTiming.reset();

  const detail::CompileClock::time_point selectionStart =
      detail::CompileClock::now();
  auto selectionTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "tensor-program-to-executable", "whole-variant-selection");
  detail::WholeVariantSelectionStatistics selectionStatistics;
  mlir::FailureOr<detail::AcceptedWholeVariant> accepted =
      detail::selectAcceptedWholeVariant(frontiers, program, executionConfig,
                                         diagnostics, selectionMode,
                                         &selectionStatistics);
  if (mlir::failed(accepted))
    return fail("whole-variant coordination failed");
  const int64_t selectionWallMs =
      detail::elapsedCompileMilliseconds(selectionStart);
  selectionTiming.reset();

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
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode) {
  return buildExecutableBundleImpl(context, tensorModule, std::move(program),
                                   executionConfig, optimizations, diagnostics,
                                   failAfterLogicalRank, selectionMode);
}

} // namespace wafer::compiler
