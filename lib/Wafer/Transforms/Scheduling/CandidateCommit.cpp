//===- CandidateCommit.cpp - Tile candidate implementation
//-----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

namespace wafer::tensor_program_scheduling {

static mlir::FailureOr<mlir::func::FuncOp>
getStandaloneSelectedFunction(mlir::Operation *anchor,
                              mlir::ModuleOp selectedModule) {
  mlir::func::FuncOp selectedFunc;
  for (auto func : selectedModule.getOps<mlir::func::FuncOp>()) {
    if (selectedFunc) {
      anchor->emitError()
          << "selected candidate commit expected one lowered function";
      return mlir::failure();
    }
    selectedFunc = func;
  }
  if (!selectedFunc) {
    anchor->emitError()
        << "selected candidate commit found no lowered function";
    return mlir::failure();
  }
  return selectedFunc;
}

mlir::LogicalResult commitSelectedTaskCandidate(SelectedCandidate &selected,
                                                const SelectionConfig &config) {
  structured_scheduler::StructuredSchedulingScope &scope = selected.scope;
  mlir::Operation *anchor = scope.insertionPoint;
  if (!anchor || !selected.sourceTask || !selected.module ||
      !isCompleteArtifactSource(selected.artifactSource)) {
    if (anchor)
      anchor->emitError(
          "selected task commit missing complete accepted artifact");
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> traversalShape =
      getStaticTraversalShape(selected.sourceTask);
  if (mlir::failed(traversalShape)) {
    anchor->emitError("selected task commit has no static traversal shape");
    return mlir::failure();
  }
  CandidateEvaluation commitProof = evaluateCompleteCandidate(
      selected.sourceTask, *traversalShape, selected.spec, config);
  if (!commitProof.failureReason.empty() || !commitProof.module ||
      !isCompleteArtifactSource(commitProof.artifactSource)) {
    anchor->emitError() << "selected task complete artifact proof failed"
                        << (commitProof.failureReason.empty() ? "" : ": ")
                        << commitProof.failureReason;
    return mlir::failure();
  }
  selected.module = std::move(commitProof.module);
  selected.artifactSource = commitProof.artifactSource;

  mlir::FailureOr<mlir::func::FuncOp> selectedFunc =
      getStandaloneSelectedFunction(anchor, *selected.module);
  if (mlir::failed(selectedFunc))
    return mlir::failure();
  return structured_scheduler::replaceStructuredSchedulingScopeWithFunction(
      scope, *selectedFunc);
}

mlir::LogicalResult accumulateStaticTerminalOperations(mlir::Operation *root,
                                                       uint64_t &count) {
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow)
      return;
    if (!mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation))
      return;
    if (count == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++count;
  });
  return mlir::failure(overflow);
}

} // namespace wafer::tensor_program_scheduling
