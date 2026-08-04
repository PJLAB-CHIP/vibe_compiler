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
  // A consumer-first commit can rewrite an operand of a non-contiguous task
  // scope (for example when shared-input peers straddle another producer).
  // Keep the transformation-local selected-op set, but re-derive its boundary
  // from the current staged SSA graph instead of retaining stale Value handles.
  if (mlir::failed(
          structured_scheduler::refreshStructuredSchedulingScopeBoundary(
              scope))) {
    anchor->emitError(
        "selected task commit cannot refresh its current SSA boundary");
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
  if (!selectedFunc->getBody().hasOneBlock()) {
    anchor->emitError(
        "selected task commit requires one-block lowered function");
    return mlir::failure();
  }
  mlir::Block &entry = selectedFunc->getBody().front();
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(entry.getTerminator());
  if (!returnOp) {
    anchor->emitError("selected task commit requires func.return terminator");
    return mlir::failure();
  }
  unsigned expectedArgCount =
      static_cast<unsigned>(scope.inputs.size() + scope.outs.size());
  if (entry.getNumArguments() != expectedArgCount ||
      returnOp.getNumOperands() != scope.yieldedValues.size()) {
    anchor->emitError("selected task commit boundary cardinality mismatch");
    return mlir::failure();
  }

  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : scope.inputs) {
    mlir::BlockArgument argument = entry.getArgument(argumentIndex++);
    if (argument.getType() != input.getType()) {
      anchor->emitError("selected task commit input boundary type mismatch");
      return mlir::failure();
    }
    mapping.map(argument, input);
  }
  for (mlir::Value out : scope.outs) {
    mlir::BlockArgument argument = entry.getArgument(argumentIndex++);
    if (argument.getType() != out.getType()) {
      anchor->emitError("selected task commit output boundary type mismatch");
      return mlir::failure();
    }
    mapping.map(argument, out);
  }

  mlir::OpBuilder builder(anchor);
  for (mlir::Operation &op : entry.getOperations()) {
    if (&op == entry.getTerminator())
      break;
    builder.clone(op, mapping);
  }
  llvm::SmallVector<mlir::Value, 4> replacements;
  for (auto [returned, yielded] :
       llvm::zip_equal(returnOp.getOperands(), scope.yieldedValues)) {
    if (returned.getType() != yielded.getType()) {
      anchor->emitError("selected task commit result boundary type mismatch");
      return mlir::failure();
    }
    mlir::Value mapped = mapping.lookupOrNull(returned);
    if (!mapped) {
      anchor->emitError("selected task commit could not map returned value");
      return mlir::failure();
    }
    if (mapped.getType() != yielded.getType()) {
      anchor->emitError(
          "selected task commit mapped result boundary type mismatch");
      return mlir::failure();
    }
    replacements.push_back(mapped);
  }

  for (auto [yielded, replacement] :
       llvm::zip_equal(scope.yieldedValues, replacements)) {
    llvm::SmallVector<mlir::OpOperand *> externalUses;
    for (mlir::OpOperand &use : yielded.getUses())
      if (!structured_scheduler::isInsideStructuredSchedulingScope(
              use.getOwner(), scope))
        externalUses.push_back(&use);
    for (mlir::OpOperand *use : externalUses)
      use->set(replacement);
  }
  for (mlir::Operation *op : llvm::reverse(scope.orderedOps))
    op->erase();
  return mlir::success();
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
