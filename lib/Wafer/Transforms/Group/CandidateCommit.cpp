//===- CandidateCommit.cpp - Tile candidate implementation
//-----------------===//

#include "Group/SelectGroupTileInternal.h"

namespace wafer::group_tile_selection {

void printSelectedSummary(const SelectedCandidate &selected,
                          TileSearchMode mode) {
  llvm::errs() << "wafer.select_group_tile selected group " << selected.label
               << " mode="
               << (mode == TileSearchMode::FirstLegal ? "first-legal"
                                                      : "min-estimated-time")
               << " tile=";
  printI64List(selected.spec.tileSizes, llvm::errs());
  llvm::errs() << " split=";
  printI64List(selected.spec.reductionSplitSizes, llvm::errs());
  llvm::errs() << " estimated_cycles=" << selected.estimatedCycles
               << " candidates=" << selected.candidateCount
               << " rejected=" << selected.rejectedCount
               << " representatives=" << selected.representativeCount << "\n";
}

static mlir::FailureOr<mlir::func::FuncOp>
getStandaloneSelectedFunction(GroupOp group, mlir::ModuleOp selectedModule) {
  mlir::func::FuncOp selectedFunc;
  for (auto func : selectedModule.getOps<mlir::func::FuncOp>()) {
    if (selectedFunc) {
      group.emitError()
          << "selected candidate commit expected one lowered function";
      return mlir::failure();
    }
    selectedFunc = func;
  }
  if (!selectedFunc) {
    group.emitError() << "selected candidate commit found no lowered function";
    return mlir::failure();
  }
  return selectedFunc;
}

mlir::LogicalResult commitSelectedCandidate(SelectedCandidate &selected,
                                            const SelectionConfig &config) {
  GroupOp group = selected.group;
  if (!group || !selected.module ||
      !isCompleteArtifactSource(selected.artifactSource)) {
    if (group)
      group.emitError()
          << "selected candidate commit missing complete accepted artifact";
    return mlir::failure();
  }

  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> traversalShape =
      getStaticTraversalShape(group);
  if (mlir::failed(traversalShape)) {
    group.emitError() << "selected candidate commit has no static traversal "
                         "shape";
    return mlir::failure();
  }

  CandidateEvaluation commitProof =
      evaluateCompleteCandidate(group, *traversalShape, selected.spec, config);
  if (!commitProof.failureReason.empty() || !commitProof.module ||
      !isCompleteArtifactSource(commitProof.artifactSource)) {
    group.emitError() << "selected candidate complete artifact proof failed"
                      << (commitProof.failureReason.empty() ? "" : ": ")
                      << commitProof.failureReason;
    return mlir::failure();
  }
  selected.module = std::move(commitProof.module);
  selected.artifactSource = commitProof.artifactSource;

  mlir::FailureOr<mlir::func::FuncOp> selectedFunc =
      getStandaloneSelectedFunction(group, *selected.module);
  if (mlir::failed(selectedFunc))
    return mlir::failure();
  if (!selectedFunc->getBody().hasOneBlock()) {
    group.emitError() << "selected candidate commit requires one-block "
                         "lowered function";
    return mlir::failure();
  }

  mlir::Block &entry = selectedFunc->getBody().front();
  mlir::Operation *terminator = entry.getTerminator();
  if (!terminator) {
    group.emitError()
        << "selected candidate commit requires function terminator";
    return mlir::failure();
  }
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(terminator);
  if (!returnOp) {
    group.emitError()
        << "selected candidate commit requires func.return terminator";
    return mlir::failure();
  }

  unsigned expectedArgCount =
      static_cast<unsigned>(group.getInputs().size() + group.getOuts().size());
  if (entry.getNumArguments() != expectedArgCount) {
    group.emitError() << "selected candidate commit argument count mismatch";
    return mlir::failure();
  }
  if (returnOp.getNumOperands() != group->getNumResults()) {
    group.emitError() << "selected candidate commit result count mismatch";
    return mlir::failure();
  }

  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : group.getInputs())
    mapping.map(entry.getArgument(argumentIndex++), input);
  for (mlir::Value output : group.getOuts())
    mapping.map(entry.getArgument(argumentIndex++), output);

  mlir::OpBuilder builder(group.getOperation());
  for (mlir::Operation &op : entry.getOperations()) {
    if (&op == terminator)
      break;
    builder.clone(op, mapping);
  }

  llvm::SmallVector<mlir::Value, 2> replacements;
  replacements.reserve(returnOp.getNumOperands());
  for (mlir::Value returned : returnOp.getOperands()) {
    mlir::Value mapped = mapping.lookupOrNull(returned);
    if (!mapped) {
      group.emitError()
          << "selected candidate commit could not map returned value";
      return mlir::failure();
    }
    replacements.push_back(mapped);
  }

  group->replaceAllUsesWith(replacements);
  group->erase();
  return mlir::success();
}

mlir::LogicalResult accumulateStaticTerminalOperations(mlir::Operation *root,
                                                       bool skipGroupBodies,
                                                       uint64_t &count) {
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow || (skipGroupBodies && operation->getParentOfType<GroupOp>()))
      return;
    if (!mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(operation))
      return;
    if (count == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++count;
  });
  return mlir::failure(overflow);
}

} // namespace wafer::group_tile_selection
