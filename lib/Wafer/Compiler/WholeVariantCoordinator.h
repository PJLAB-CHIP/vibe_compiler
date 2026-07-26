//===- WholeVariantCoordinator.h - All-rank candidate commit -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H
#define WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "WholeVariantSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace wafer::compiler::detail {

/// One compiler-private member of a rank-local scheduling frontier. Stable
/// ordinal plus artifact kind form the cross-rank correspondence key; the
/// module remains the sole semantic artifact.
struct RankVariantCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t stableOrdinal = 0;
  wafer::RankArtifactKind artifactKind = wafer::RankArtifactKind::Spill;
  bool reservedBaseline = false;
};

using RankVariantFrontier = std::vector<RankVariantCandidate>;

struct AcceptedWholeVariant {
  std::vector<RankExecutable> ranks;
  analysis::WholeCardInstructionProgramCost resourceCost;
  std::vector<int64_t> selectedStableOrdinals;
  std::vector<wafer::RankArtifactKind> selectedArtifactKinds;
  std::vector<bool> selectedReservedBaselines;
};

/// Same-frontier result used by the profiling product. When production selects
/// the reserved baseline, `reservedBaseline` is absent and both product roles
/// refer to `production`; no accepted module is cloned after selection.
struct AcceptedProductionAndBaseline {
  AcceptedWholeVariant production;
  std::optional<AcceptedWholeVariant> reservedBaseline;
  bool productionIsReservedBaseline = false;
};

/// Selects a complete rank-domain combination from independently planned
/// frontiers. Every attempted combination must carry one same-generation
/// stable ordinal and one physical artifact kind across all ranks before it is
/// cloned, then passes exact
/// execution-config verification, Direct DTE binding, whole-card resources,
/// accepted-rank/resource projection, and target ABI lowering.  Only a fully
/// passing combination is returned, so no failed attempt can partially commit
/// a rank or typed transport binding.
mlir::FailureOr<AcceptedWholeVariant> selectAcceptedWholeVariant(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production);

/// Selects the production winner while retaining the already accepted
/// reserved baseline from the same generated rank frontiers and acceptance
/// transaction.
mlir::FailureOr<AcceptedProductionAndBaseline>
selectAcceptedProductionAndBaseline(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H
