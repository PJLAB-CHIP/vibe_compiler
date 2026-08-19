//===- CardExecutableSearch.h - Card executable selection -----*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESEARCH_H
#define WAFER_COMPILER_CARDEXECUTABLESEARCH_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/CodeGen/Executable/CardExecutableLowering.h"
#include "Wafer/Planning/Search/SearchControl.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {

enum class CardExecutableSearchCoverage : uint8_t {
  Exhausted,
  BudgetLimited,
};

struct CardExecutableSearchSummary {
  SearchWorkCounts work;
  CardExecutableSearchCoverage coverage =
      CardExecutableSearchCoverage::BudgetLimited;
};

/// Evaluates complete original-root physical assignments through the shared
/// Q50.0 exact gate. The accepted baseline is an incumbent only. The explicit
/// budget controls complete candidate evaluations; no partial state is lowered
/// or cloned. Semantic graph alternatives join this same closure before Q51 is
/// marked complete.
mlir::FailureOr<CardExecutableLoweringResult> runCardExecutableSearch(
    mlir::ModuleOp tensorProgram, const CardProgramAnalysis &programAnalysis,
    CardExecutableLoweringResult baseline,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, SearchWorkBudget budget,
    const TargetMemoryPolicy &memory,
    CardExecutableSearchSummary *summary = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESEARCH_H
