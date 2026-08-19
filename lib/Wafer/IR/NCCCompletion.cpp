//===- NCCCompletion.cpp - Instr NCC completion adapter ---------------===//

#include "Wafer/IR/NCCCompletion.h"

#include "Wafer/IR/WaferDialect.h"

wafer::NCCOperationCompletion
wafer::getNCCOperationCompletion(mlir::Operation *operation) {
  if (!operation)
    return {};
  if (auto completion =
          mlir::dyn_cast<WaferNCCCompletionOpInterface>(operation))
    return completion.getNCCCompletion();
  if (auto issue = mlir::dyn_cast<WaferNCCIssueOpInterface>(operation))
    return {NCCCompletionKind::OrderedAsynchronousIssue, issue.getIssueWorker(),
            0};
  return {};
}

std::optional<wafer::NCCWorker>
wafer::getNCCIssueWorker(mlir::Operation *operation) {
  NCCOperationCompletion completion = getNCCOperationCompletion(operation);
  return completion.issueWorker;
}

mlir::LogicalResult wafer::setNCCIssueWorker(mlir::Operation *operation,
                                             NCCWorker worker) {
  if (!operation || !mlir::isa<WaferNCCIssueOpInterface>(operation) ||
      static_cast<uint32_t>(worker) >= kNCCWorkerCount)
    return mlir::failure();
  operation->setAttr(kWaferNCCWorkerAttrName,
                     NCCWorkerAttr::get(operation->getContext(), worker));
  return mlir::success();
}
