//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "CompilationInternal.h"
#include "ScheduledRankFinalization.h"
#include "WholeVariantCoordinator.h"

#include "Wafer/Transforms/TensorProgramScheduling.h"

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

llvm::Expected<ExecutableBundle> detail::buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    const CompilationOptimizationPolicyV1 &optimizationPolicy) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  struct RankLoweringResult {
    struct SerializedCandidate {
      std::string moduleText;
      int64_t estimatedTimePs = 0;
      int64_t discoveryOrder = 0;
    };

    int64_t logicalRank;
    bool succeeded = false;
    std::vector<SerializedCandidate> candidates;
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
    std::optional<bool> candidateCleanup = isOptimizationMechanismEnabled(
        optimizationPolicy.proposal,
        optimizationPolicy.optimizationConfiguration,
        mechanism::CandidateCommitCleanup, nullptr);
    // An optional mechanism omitted from the immutable proposal is disabled.
    // The finalization pipeline still validates the complete typed proposal
    // and configuration before consuming the scheduled candidate.
    schedulingConfig.enableCandidateCommitCleanup =
        candidateCleanup.value_or(false);
    mlir::FailureOr<std::vector<wafer::ScheduledRankCandidate>> frontier =
        wafer::buildScheduledRankCandidateFrontier(*sourceModule,
                                                   schedulingConfig);
    if (mlir::failed(frontier) || frontier->empty())
      return;

    mlir::FailureOr<std::vector<detail::FinalizedRankCandidate>> finalized =
        detail::finalizeScheduledRankCandidateFrontier(
            std::move(*frontier), result.logicalRank,
            optimizationPolicy.proposal,
            optimizationPolicy.optimizationConfiguration);
    if (mlir::failed(finalized))
      return;

    result.candidates.reserve(finalized->size());
    for (detail::FinalizedRankCandidate &candidate : *finalized) {
      RankLoweringResult::SerializedCandidate serialized;
      serialized.estimatedTimePs = candidate.estimatedTimePs;
      serialized.discoveryOrder = candidate.discoveryOrder;
      llvm::raw_string_ostream moduleStream(serialized.moduleText);
      candidate.module->print(moduleStream);
      moduleStream.flush();
      result.candidates.push_back(std::move(serialized));
    }
    result.succeeded = true;
  });

  std::vector<detail::RankVariantFrontier> frontiers;
  frontiers.reserve(loweringResults.size());
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
    detail::RankVariantFrontier imported;
    imported.reserve(result.candidates.size());
    for (RankLoweringResult::SerializedCandidate &candidate :
         result.candidates) {
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::parseSourceString<mlir::ModuleOp>(candidate.moduleText,
                                                  context.get());
      if (!module)
        return fail("failed to import a lowered scheduling candidate for "
                    "logical rank " +
                    std::to_string(logicalRank));
      imported.push_back({std::move(module), candidate.estimatedTimePs,
                          candidate.discoveryOrder});
    }
    frontiers.push_back(std::move(imported));
  }

  mlir::FailureOr<detail::AcceptedWholeVariant> accepted =
      detail::selectAcceptedWholeVariant(frontiers, program, executionConfig,
                                         diagnostics);
  if (mlir::failed(accepted))
    return fail("whole-variant coordination failed");
  std::vector<RankExecutable> ranks = std::move(accepted->ranks);
  if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return fail("executable bundle rank domain is incomplete");
  for (auto [expectedRank, rank] : llvm::enumerate(ranks))
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail("executable bundle rank domain is not canonical");

  return ExecutableBundleBuilder::makeBundle(
      executionConfig, std::move(context), std::move(ranks));
}

} // namespace wafer::compiler
