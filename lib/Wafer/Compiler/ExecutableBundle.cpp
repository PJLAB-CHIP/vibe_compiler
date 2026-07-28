//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "CompilationInternal.h"
#include "ScheduledRankFinalization.h"
#include "WholeVariantAttemptPlan.h"
#include "WholeVariantCoordinator.h"

#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

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
                          candidate.bufferingPlanOrdinal});
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
        module = mlir::parseSourceString<mlir::ModuleOp>(candidate.moduleText,
                                                         &ownerContext);
        if (!module)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "failed to import a lowered scheduling candidate for logical "
              "rank %zu",
              static_cast<size_t>(rankIndex));
      }
      imported.push_back({std::move(module), candidate.stableOrdinal,
                          candidate.artifactKind, candidate.reservedBaseline,
                          candidate.bufferingKind,
                          candidate.bufferingPlanOrdinal});
    }
    frontiers.push_back(std::move(imported));
  }
  return frontiers;
}

static llvm::Expected<ExecutableBundle> buildExecutableBundleImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    detail::WholeVariantSelectionMode selectionMode) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  struct RankLoweringResult {
    int64_t logicalRank;
    bool succeeded = false;
    detail::SerializedRankVariantFrontier candidates;
    std::string diagnostics;
  };

  std::string tensorModuleText;
  llvm::raw_string_ostream tensorModuleStream(tensorModuleText);
  tensorModule.print(tensorModuleStream);
  tensorModuleStream.flush();

  std::vector<RankLoweringResult> loweringResults;
  loweringResults.reserve(executionConfig.getRankCount());
  for (int64_t logicalRank = 0; logicalRank < executionConfig.getRankCount();
       ++logicalRank)
    loweringResults.push_back({logicalRank});

  // Candidate evaluation installs diagnostic handlers and creates transient
  // IR. Give each concurrent rank its own context, then parse successful
  // results back into the bundle-owner context in canonical rank order before
  // cross-rank transport acceptance.
  llvm::parallelFor(0, loweringResults.size(), [&](size_t index) {
    RankLoweringResult &result = loweringResults[index];
    mlir::DialectRegistry registry;
    detail::registerCompilationDialects(registry);
    mlir::MLIRContext rankContext(registry);
    // Rank pipelines are already the unit of parallelism. Avoid creating a
    // nested context thread pool for each of the 16 concurrent workers.
    rankContext.disableMultithreading();
    rankContext.loadAllAvailableDialects();
    mlir::ScopedDiagnosticHandler diagnosticHandler(
        &rankContext, [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream os(result.diagnostics);
          diagnostic.print(os);
          os << "\n";
          return mlir::success();
        });

    mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
        mlir::parseSourceString<mlir::ModuleOp>(tensorModuleText, &rankContext);
    if (!sourceModule)
      return;
    wafer::TensorProgramSchedulingConfig schedulingConfig;
    schedulingConfig.logicalRank = result.logicalRank;
    schedulingConfig.candidateParallelism = 4;
    schedulingConfig.targetProfile = executionConfig.getTargetProfileId();
    mlir::FailureOr<std::vector<wafer::ScheduledRankCandidate>> frontier =
        wafer::buildScheduledRankCandidateFrontier(*sourceModule,
                                                   schedulingConfig);
    if (mlir::failed(frontier) || frontier->empty())
      return;

    mlir::FailureOr<std::vector<detail::FinalizedRankCandidate>> finalized =
        detail::finalizeScheduledRankCandidateFrontier(
            std::move(*frontier), executionConfig.getTargetProfileId());
    if (mlir::failed(finalized))
      return;

    result.candidates.reserve(finalized->size());
    for (detail::FinalizedRankCandidate &candidate : *finalized) {
      detail::SerializedRankVariantCandidate serialized;
      serialized.stableOrdinal = candidate.stableOrdinal;
      serialized.artifactKind = candidate.artifactKind;
      serialized.reservedBaseline = candidate.reservedBaseline;
      serialized.bufferingKind = candidate.bufferingKind;
      serialized.bufferingPlanOrdinal = candidate.bufferingPlanOrdinal;
      llvm::raw_string_ostream moduleStream(serialized.moduleText);
      candidate.module->print(moduleStream);
      moduleStream.flush();
      result.candidates.push_back(std::move(serialized));
    }
    result.succeeded = true;
  });

  std::vector<detail::SerializedRankVariantFrontier> serializedFrontiers;
  serializedFrontiers.reserve(loweringResults.size());
  for (RankLoweringResult &result : loweringResults) {
    const int64_t logicalRank = result.logicalRank;
    if (!result.succeeded || result.candidates.empty()) {
      diagnostics << result.diagnostics;
      return fail("rank lowering failed for logical rank " +
                  std::to_string(logicalRank));
    }
    if (failAfterLogicalRank && logicalRank == *failAfterLogicalRank)
      return fail("test-only injected failure after logical rank " +
                  std::to_string(logicalRank));
    serializedFrontiers.push_back(std::move(result.candidates));
  }

  // Reproduce the coordinator's original bounded attempt sequence from cheap
  // metadata before parsing any finalized module into the bundle-owner
  // context. The original frontier slots remain intact: only slots referenced
  // by a correspondence-valid attempted tuple (plus every reserved baseline)
  // materialize their module.
  llvm::Expected<std::vector<detail::RankVariantFrontier>> imported =
      detail::importRankVariantFrontiersIntoOwnerContext(
          *context, serializedFrontiers, executionConfig.getRankCount());
  if (!imported)
    return fail(llvm::toString(imported.takeError()));
  std::vector<detail::RankVariantFrontier> frontiers = std::move(*imported);

  mlir::FailureOr<detail::AcceptedWholeVariant> accepted =
      detail::selectAcceptedWholeVariant(frontiers, program, executionConfig,
                                         diagnostics, selectionMode);
  if (mlir::failed(accepted))
    return fail("whole-variant coordination failed");

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
