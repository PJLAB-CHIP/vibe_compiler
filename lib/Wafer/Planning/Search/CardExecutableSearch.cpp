//===- CardExecutableSearch.cpp - Card executable selection -----------===//

#include "Wafer/Planning/Search/CardExecutableSearch.h"

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/Search/TensorProgramAlternative.h"
#include "Wafer/Planning/Search/UnifiedPhysicalDataflow.h"

#include <array>

namespace wafer::compiler::detail {

mlir::FailureOr<CardExecutableLoweringResult> runCardExecutableSearch(
    mlir::ModuleOp tensorProgram, const CardProgramAnalysis &programAnalysis,
    CardExecutableLoweringResult baseline,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, SearchWorkBudget budget,
    const TargetMemoryPolicy &memory, CardExecutableSearchSummary *summary) {
  if (summary)
    *summary = {};
  mlir::FailureOr<TensorProgramAlternativeDomain> alternatives =
      getTensorProgramAlternativeDomain(tensorProgram);
  if (mlir::failed(alternatives))
    return mlir::failure();
  SearchWorkCounts work;
  bool exhausted = false;
  auto isBetter = [](const CardExecutableLoweringResult &candidate,
                     const CardExecutableLoweringResult &incumbent) {
    auto values = [](const analysis::CardInstructionProgramCost &cost)
        -> std::optional<std::array<uint64_t, 6>> {
      const std::array<const analysis::ScheduleCostMetric *, 6> metrics{
          &cost.aggregateDDRReadBytes,
          &cost.aggregateDDRWriteBytes,
          &cost.aggregateNoC.aggregateTransmitBytes,
          &cost.aggregateInstructionCount,
          &cost.maximumTileSPMHighWaterBytes,
          &cost.maximumTileDDRHighWaterBytes};
      if (llvm::any_of(metrics,
                       [](const auto *metric) { return !metric->isKnown(); }))
        return std::nullopt;
      return std::array<uint64_t, 6>{metrics[0]->value, metrics[1]->value,
                                     metrics[2]->value, metrics[3]->value,
                                     metrics[4]->value, metrics[5]->value};
    };
    auto candidateValues = values(candidate.resourceCost);
    auto incumbentValues = values(incumbent.resourceCost);
    return candidateValues && incumbentValues &&
           *candidateValues < *incumbentValues;
  };

  std::string failureReason;
  auto evaluateRoot = [&](mlir::ModuleOp root,
                          const CardProgramAnalysis &analysis) {
    auto domain =
        UnifiedPhysicalDataflowDomain::create(analysis, CardId(0), memory);
    if (mlir::failed(domain)) {
      ++work.indeterminate;
      return false;
    }
    auto current = domain->getFirstAssignment(&failureReason);
    if (mlir::failed(current)) {
      ++work.indeterminate;
      return false;
    }
    while (true) {
      if (budget.maximumEvaluations &&
          work.evaluated >= *budget.maximumEvaluations)
        return false;
      ++work.generated;
      ++work.evaluated;
      auto materialized = domain->materialize(root, *current, &failureReason);
      auto scopes = domain->buildBufferingScopes(*current, &failureReason);
      if (mlir::failed(materialized) || mlir::failed(scopes)) {
        ++work.indeterminate;
      } else {
        CardExecutableCompilationResult compiled =
            compileCardModuleToExecutable(
                std::move(materialized->module), CardId(0),
                analysis.availableTileIds, *scopes, materialized->relations,
                program, executionConfig, diagnostics, programData,
                /*statistics=*/nullptr, /*tilePipelineParallelism=*/0,
                /*captureTileIRTrace=*/false,
                /*applySelectedInstructionSchedule=*/true);
        if (compiled.isAccepted()) {
          ++work.accepted;
          CardExecutableLoweringResult candidate = compiled.takeExecutable();
          if (isBetter(candidate, baseline))
            baseline = std::move(candidate);
        } else if (compiled.isProvenExactRejection()) {
          ++work.exactRejected;
        } else {
          ++work.indeterminate;
        }
      }
      auto next = domain->getNextAssignment(*current, &failureReason);
      if (mlir::failed(next)) {
        ++work.indeterminate;
        return false;
      }
      if (!*next)
        return true;
      current = std::move(**next);
    }
  };

  TensorProgramAlternativeAssignment alternative =
      alternatives->getFirstAssignment();
  while (true) {
    bool rootExhausted = false;
    if (alternative.kind == TensorProgramAlternativeKind::Original) {
      rootExhausted = evaluateRoot(tensorProgram, programAnalysis);
    } else {
      TensorProgramAlternativeMaterialization materialized =
          materializeTensorProgramAlternative(tensorProgram, alternative);
      if (materialized.getKind() !=
          TensorProgramAlternativeMaterializationKind::Materialized) {
        ++work.indeterminate;
      } else {
        mlir::OwningOpRef<mlir::ModuleOp> root = materialized.takeModule();
        auto analysis =
            analyzeCardProgram(*root, program, executionConfig, diagnostics);
        if (mlir::failed(analysis))
          ++work.indeterminate;
        else
          rootExhausted = evaluateRoot(*root, **analysis);
      }
    }
    if (!rootExhausted)
      break;
    auto nextAlternative = alternatives->getNextAssignment(alternative);
    if (mlir::failed(nextAlternative)) {
      ++work.indeterminate;
      break;
    }
    if (!*nextAlternative) {
      exhausted = true;
      break;
    }
    alternative = std::move(**nextAlternative);
  }
  if (summary) {
    summary->work = work;
    summary->coverage = exhausted ? CardExecutableSearchCoverage::Exhausted
                                  : CardExecutableSearchCoverage::BudgetLimited;
  }
  return baseline;
}

} // namespace wafer::compiler::detail
