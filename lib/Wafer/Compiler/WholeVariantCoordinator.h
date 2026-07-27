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
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

/// One compiler-private member of a rank-local scheduling frontier. Stable
/// ordinal plus artifact kind form the cross-rank correspondence key; the
/// module remains the sole semantic artifact. During invocation-local context
/// transfer, a slot that cannot occur in the already-fixed bounded attempt
/// sequence may keep a null module so its original index and metadata still
/// reproduce that sequence. Such a slot can only be observed by the
/// correspondence precheck and never crosses the accepted-bundle boundary.
struct RankVariantCandidate {
  /// Nullable only for an invocation-local, non-attemptable context-transfer
  /// slot. Every correspondence-valid attempted tuple requires a real module.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t stableOrdinal = 0;
  wafer::RankArtifactKind artifactKind = wafer::RankArtifactKind::Spill;
  bool reservedBaseline = false;
};

using RankVariantFrontier = std::vector<RankVariantCandidate>;

struct AcceptedWholeVariant {
  AcceptedWholeVariant(std::vector<RankExecutable> ranks,
                       RuntimeLaunchContract runtimeLaunchContract,
                       analysis::WholeCardInstructionProgramCost resourceCost)
      : ranks(std::move(ranks)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<RankExecutable> ranks;
  RuntimeLaunchContract runtimeLaunchContract;
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

/// Invocation-local test instrumentation for the expensive target gate. It is
/// never stored in selected IR, an executable bundle, or a package artifact.
struct WholeVariantSelectionStatistics {
  uint64_t targetGateInvocations = 0;
  uint64_t targetRankGateInvocations = 0;
};

/// Selects a complete rank-domain combination from independently planned
/// frontiers. Every attempted combination must carry one same-generation
/// stable ordinal and one physical artifact kind across all ranks before it is
/// cloned. The reserved baseline passes exact execution-config verification,
/// Direct DTE binding, whole-card resources, accepted-rank/resource projection,
/// and target ABI/LLVM lowering before optimized tuples are considered.
/// Optimized tuples run the target gate only when the existing Pareto rule
/// would retain them against the already fully target-gated optimized
/// frontier. Only a fully passing combination can be returned, so no failed
/// attempt can partially commit a rank or typed transport binding.
mlir::FailureOr<AcceptedWholeVariant> selectAcceptedWholeVariant(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production,
    WholeVariantSelectionStatistics *statistics = nullptr);

/// Selects the production winner while retaining the already accepted
/// reserved baseline from the same generated rank frontiers and acceptance
/// transaction.
mlir::FailureOr<AcceptedProductionAndBaseline>
selectAcceptedProductionAndBaseline(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionStatistics *statistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTCOORDINATOR_H
