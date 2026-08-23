//===- CardExecutableSearch.h - Card executable selection -----*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESEARCH_H
#define WAFER_COMPILER_CARDEXECUTABLESEARCH_H

#include "Wafer/CodeGen/Executable/CardExecutableLowering.h"
#include "Wafer/Planning/Search/SearchWork.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class CardExecutableSearchCoverage : uint8_t {
  OptimalCertified,
  FeasibleWithBound,
  BudgetedFeasible,
};

struct CardExecutableSearchSummary {
  SearchWorkCounts work;
  CardExecutableSearchCoverage coverage =
      CardExecutableSearchCoverage::BudgetedFeasible;
  uint64_t winnerUpdates = 0;
  std::string proposalDetail;
  std::string lastDetail;
};

/// One search-owned accepted result. Diagnostic IR snapshots are present only
/// when explicitly requested and always move with the executable selected from
/// the same candidate compilation.
struct CardExecutableSearchResult {
  CardExecutableLoweringResult executable;
  std::vector<std::string> tileDataflowIRTrace;
};

/// Independently analyzes the immutable TensorProgram and evaluates complete
/// physical assignments through the shared CardModule-to-executable exact
/// gate. This entry does not invoke, receive or return the deterministic
/// baseline. The explicit budget controls complete search-candidate
/// evaluations; no partial state is lowered or cloned. Semantic graph
/// alternatives join this same search-owned candidate closure.
mlir::FailureOr<CardExecutableSearchResult> runCardExecutableSearch(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, SearchWorkBudget budget,
    CardExecutableSearchSummary *summary = nullptr,
    bool captureTileDataflowIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESEARCH_H
