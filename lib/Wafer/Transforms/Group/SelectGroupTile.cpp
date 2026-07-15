//===- SelectGroupTile.cpp - Closed-loop group tile selection -------------===//

#include "Group/SelectGroupTileInternal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace wafer {
#define GEN_PASS_DEF_SELECTGROUPTILEPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

using namespace group_tile_selection;

namespace {

struct SelectGroupTilePass
    : public impl::SelectGroupTilePassBase<SelectGroupTilePass> {
  using impl::SelectGroupTilePassBase<
      SelectGroupTilePass>::SelectGroupTilePassBase;

  void runOnOperation() final {
    mlir::FailureOr<TileSearchMode> parsedMode =
        parseTileSearchMode(tileSearch, getOperation());
    mlir::FailureOr<TileSearchEffort> parsedEffort =
        parseTileSearchEffort(tileSearchEffort, getOperation());
    if (mlir::failed(parsedMode) || mlir::failed(parsedEffort)) {
      signalPassFailure();
      return;
    }
    std::optional<llvm::SmallVector<int64_t, 8>> parsedPreferred;
    if (!llvm::StringRef(preferredTileSizes).trim().empty()) {
      mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsed = parseI64List(
          preferredTileSizes, "preferred-tile-sizes", getOperation());
      if (mlir::failed(parsed)) {
        signalPassFailure();
        return;
      }
      parsedPreferred = std::move(*parsed);
    }

    if (maxCandidatesPerDim < -1 || maxSearchCandidates < -1 ||
        searchBeamWidth < -1) {
      getOperation()->emitError()
          << "invalid_tile_search_config: search-space overrides must be -1 "
             "or non-negative";
      signalPassFailure();
      return;
    }
    if (*parsedMode == TileSearchMode::MinEstimatedTime &&
        candidateParallelism <= 0) {
      getOperation()->emitError()
          << "invalid_tile_search_config: min-estimated-time requires "
             "positive candidate-parallelism";
      signalPassFailure();
      return;
    }

    WaferTargetPolicy targetPolicy = getDefaultWaferTargetPolicy(*parsedEffort);
    SelectionConfig config(targetPolicy);
    config.mode = *parsedMode;
    config.logicalRank = logicalRank;
    if (parsedPreferred && parsedPreferred->empty()) {
      getOperation()->emitError()
          << "invalid_tile_search_config: preferred-tile-sizes cannot be empty "
             "when explicitly provided";
      signalPassFailure();
      return;
    }
    if (parsedPreferred)
      config.preferredTileSizes = *parsedPreferred;
    if (maxCandidatesPerDim >= 0)
      config.maxCandidatesPerDim = maxCandidatesPerDim;
    if (maxSearchCandidates >= 0)
      config.maxSearchCandidates = maxSearchCandidates;
    if (searchBeamWidth >= 0)
      config.searchBeamWidth = searchBeamWidth;
    if (config.mode == TileSearchMode::MinEstimatedTime)
      config.candidateParallelism = candidateParallelism;

    mlir::OwningOpRef<mlir::ModuleOp> stagedModule =
        mlir::cast<mlir::ModuleOp>(getOperation()->clone());
    llvm::SmallVector<GroupOp, 8> groups;
    stagedModule->walk([&](GroupOp group) { groups.push_back(group); });
    if (groups.empty()) {
      markAllAnalysesPreserved();
      return;
    }
    if (config.logicalRank < 0) {
      getOperation()->emitError()
          << "missing_logical_rank: tile selection requires an explicit "
             "non-negative logical-rank";
      signalPassFailure();
      return;
    }

    llvm::DenseMap<mlir::Operation *, unsigned> groupOrdinals;
    llvm::SmallVector<SelectedCandidate, 8> selectedCandidates;
    for (GroupOp group : groups) {
      std::string symbolName = getNearestSymbolName(group.getOperation());
      unsigned ordinal = groupOrdinals[group->getParentOp()]++;
      std::string label;
      llvm::raw_string_ostream labelOs(label);
      labelOs << symbolName << "#" << ordinal;

      mlir::FailureOr<SelectedCandidate> selected =
          selectCandidateForGroup(group, labelOs.str(), config);
      if (mlir::failed(selected)) {
        signalPassFailure();
        return;
      }
      selectedCandidates.push_back(std::move(*selected));
    }

    // The staged module is not mutated until the complete per-rank terminal
    // program is known to fit. Existing terminal operations outside groups
    // remain live; each selected standalone artifact replaces one group body.
    uint64_t terminalOperationCount = 0;
    if (mlir::failed(accumulateStaticTerminalOperations(
            (*stagedModule).getOperation(), /*skipGroupBodies=*/true,
            terminalOperationCount))) {
      getOperation()->emitError()
          << "static_terminal_budget_exceeded: selected rank terminal "
             "operation count overflows";
      signalPassFailure();
      return;
    }
    for (const SelectedCandidate &selected : selectedCandidates) {
      if (!selected.module) {
        getOperation()->emitError()
            << "static_terminal_budget_exceeded: selected rank terminal "
               "operation count is unavailable";
        signalPassFailure();
        return;
      }
      uint64_t selectedOperationCount = 0;
      detail::StaticTerminalOperationBudgetStatus status =
          detail::checkStaticTerminalOperationBudget(
              (*selected.module).getOperation(), selectedOperationCount);
      if (status ==
              detail::StaticTerminalOperationBudgetStatus::CountOverflow ||
          selectedOperationCount >
              std::numeric_limits<uint64_t>::max() - terminalOperationCount) {
        getOperation()->emitError()
            << "static_terminal_budget_exceeded: selected rank terminal "
               "operation count overflows";
        signalPassFailure();
        return;
      }
      terminalOperationCount += selectedOperationCount;
    }
    if (terminalOperationCount >
        wafer::detail::kStaticTerminalOperationBudget) {
      getOperation()->emitError()
          << "static_terminal_budget_exceeded: selected rank requires "
          << terminalOperationCount
          << " terminal instruction issue/completion operations; limit is "
          << wafer::detail::kStaticTerminalOperationBudget;
      signalPassFailure();
      return;
    }

    for (SelectedCandidate &selected : selectedCandidates) {
      if (mlir::failed(commitSelectedCandidate(selected, config))) {
        signalPassFailure();
        return;
      }
    }

    // Commit only changes the private staged clone. Recount the actual staged
    // rank so the gate does not rely on determinism between candidate proof
    // materialization and commit-time materialization.
    uint64_t stagedTerminalOperationCount = 0;
    detail::StaticTerminalOperationBudgetStatus stagedBudgetStatus =
        detail::checkStaticTerminalOperationBudget(
            (*stagedModule).getOperation(), stagedTerminalOperationCount);
    if (stagedBudgetStatus ==
        detail::StaticTerminalOperationBudgetStatus::CountOverflow) {
      getOperation()->emitError()
          << "static_terminal_budget_exceeded: committed staged rank terminal "
             "operation count overflows";
      signalPassFailure();
      return;
    }
    if (stagedBudgetStatus ==
        detail::StaticTerminalOperationBudgetStatus::BudgetExceeded) {
      getOperation()->emitError()
          << "static_terminal_budget_exceeded: committed staged rank requires "
          << stagedTerminalOperationCount
          << " terminal instruction issue/completion operations; limit is "
          << wafer::detail::kStaticTerminalOperationBudget;
      signalPassFailure();
      return;
    }

    if (mlir::failed(mlir::verify(*stagedModule))) {
      stagedModule->emitError()
          << "selected candidate commit produced invalid IR";
      signalPassFailure();
      return;
    }

    if (printCandidateSummary)
      for (const SelectedCandidate &selected : selectedCandidates)
        printSelectedSummary(selected, config.mode);

    getOperation()->setAttrs((*stagedModule)->getAttrs());
    getOperation().getBodyRegion().takeBody(stagedModule->getBodyRegion());
  }
};

} // namespace

} // namespace wafer
