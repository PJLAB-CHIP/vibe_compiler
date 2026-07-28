//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "Wafer/IR/WaferDialect.h"

#include "Wafer/IR/WaferInterfaces.cpp.inc"

wafer::NCCCompletionContract
wafer::getNCCCompletionContract(mlir::Operation *operation) {
  if (!operation)
    return {};
  if (mlir::isa<SyncLocalFenceOp>(operation))
    return {LocalInstructionCompletion::ParticipantJoin, std::nullopt,
            uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0)};
  if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
    uint32_t participants = 0;
    for (int64_t worker : join.getParticipants()) {
      if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount))
        return {LocalInstructionCompletion::ParticipantJoin, std::nullopt, 0};
      participants |= uint32_t{1} << static_cast<uint32_t>(worker);
    }
    return {LocalInstructionCompletion::ParticipantJoin, std::nullopt,
            participants};
  }

  if (auto peripheral = mlir::dyn_cast<InstrPeripheralOp>(operation)) {
    switch (peripheral.getKindAttr().getValue()) {
    case InstrPeripheralKind::ArgMax:
    case InstrPeripheralKind::ArgMin:
      // The production target wrapper issues CT, drains the default worker's
      // local queues, then writes both scalar results to SPM.
      return {LocalInstructionCompletion::SynchronousWriteback,
              peripheral.getIssueWorker(),
              uint32_t{1}
                  << static_cast<uint32_t>(peripheral.getIssueWorker())};
    default:
      break;
    }
  }

  if (auto issue = mlir::dyn_cast<WaferNCCIssueOpInterface>(operation))
    return {LocalInstructionCompletion::OrderedPending,
            issue.getIssueWorker(), 0};
  return {};
}

wafer::LocalInstructionCompletion
wafer::classifyLocalInstructionCompletion(mlir::Operation *operation) {
  return getNCCCompletionContract(operation).behavior;
}

std::optional<wafer::NCCWorker>
wafer::getNCCIssueWorker(mlir::Operation *operation) {
  return getNCCCompletionContract(operation).issueWorker;
}

llvm::StringRef
wafer::stringifyTargetImplementationKind(TargetImplementationKind kind) {
  switch (kind) {
  case TargetImplementationKind::Fill:
    return "fill";
  case TargetImplementationKind::Gemm:
    return "gemm";
  case TargetImplementationKind::BatchGemm:
    return "batch-gemm";
  case TargetImplementationKind::Generic:
    return "generic";
  case TargetImplementationKind::GenericReciprocal:
    return "generic-reciprocal";
  }
  llvm_unreachable("unknown target implementation kind");
}
