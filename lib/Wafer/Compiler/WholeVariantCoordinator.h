//===- WholeVariantCoordinator.h - All-rank candidate commit -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H
#define WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

/// One compiler-private member of a rank-local scheduling frontier.  Cost and
/// discovery order participate only in deterministic selection; the module is
/// the sole semantic artifact.
struct RankVariantCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t estimatedTimePs = 0;
  int64_t discoveryOrder = 0;
};

using RankVariantFrontier = std::vector<RankVariantCandidate>;

struct AcceptedWholeVariant {
  std::vector<RankExecutable> ranks;
  analysis::WholeCardInstructionProgramCost resourceCost;
  std::vector<int64_t> selectedDiscoveryOrders;
};

/// Selects a complete rank-domain combination from independently planned
/// frontiers.  Every attempted combination is cloned, then passes exact
/// execution-config verification, Direct DTE binding, whole-card resources,
/// accepted-rank/resource projection, and target ABI lowering.  Only a fully
/// passing combination is returned, so no failed attempt can partially commit
/// a rank or typed transport binding.
mlir::FailureOr<AcceptedWholeVariant> selectAcceptedWholeVariant(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H
