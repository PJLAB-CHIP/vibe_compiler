//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

#include "Wafer/IR/WaferInterfaces.cpp.inc"

namespace {

static void analyzeNCCWorkerBlock(
    mlir::Block &block, uint32_t &pendingWorkers,
    wafer::NCCWorkerWindowSummary &summary);

static void analyzeNCCWorkerOperation(
    mlir::Operation *operation, uint32_t &pendingWorkers,
    wafer::NCCWorkerWindowSummary &summary) {
  if (auto tileRegion = mlir::dyn_cast<wafer::TileRegionOp>(operation)) {
    if (!tileRegion.getBody().empty())
      analyzeNCCWorkerBlock(tileRegion.getBody().front(), pendingWorkers,
                            summary);
    return;
  }
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
    uint32_t thenPending = pendingWorkers;
    analyzeNCCWorkerBlock(ifOp.getThenRegion().front(), thenPending, summary);
    uint32_t elsePending = pendingWorkers;
    if (!ifOp.getElseRegion().empty())
      analyzeNCCWorkerBlock(ifOp.getElseRegion().front(), elsePending, summary);
    pendingWorkers = thenPending | elsePending;
    return;
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
    uint32_t bodyPending = pendingWorkers;
    analyzeNCCWorkerBlock(*forOp.getBody(), bodyPending, summary);
    pendingWorkers = bodyPending;
    return;
  }

  wafer::NCCCompletionContract contract =
      wafer::getNCCCompletionContract(operation);
  if (contract.issueWorker) {
    uint32_t worker = static_cast<uint32_t>(*contract.issueWorker);
    if (worker < wafer::kNCCWorkerCount) {
      uint32_t workerMask = uint32_t{1} << worker;
      summary.issuedWorkerMask |= workerMask;
      pendingWorkers |= workerMask;
      summary.hasCrossWorkerWindow |= llvm::popcount(pendingWorkers) >= 2;
    }
  }
  if (contract.behavior ==
          wafer::LocalInstructionCompletion::ParticipantJoin ||
      contract.behavior ==
          wafer::LocalInstructionCompletion::SynchronousWriteback)
    pendingWorkers &= ~(contract.participantMask &
                        wafer::kAllNCCWorkersMask);
}

static void analyzeNCCWorkerBlock(
    mlir::Block &block, uint32_t &pendingWorkers,
    wafer::NCCWorkerWindowSummary &summary) {
  for (mlir::Operation &operation : block)
    analyzeNCCWorkerOperation(&operation, pendingWorkers, summary);
}

} // namespace

wafer::NCCCompletionContract
wafer::getNCCCompletionContract(mlir::Operation *operation) {
  if (!operation)
    return {};
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
      return {
          LocalInstructionCompletion::SynchronousWriteback,
          peripheral.getIssueWorker(),
          uint32_t{1} << static_cast<uint32_t>(peripheral.getIssueWorker())};
    default:
      break;
    }
  }

  if (auto issue = mlir::dyn_cast<WaferNCCIssueOpInterface>(operation))
    return {LocalInstructionCompletion::OrderedPending, issue.getIssueWorker(),
            0};
  return {};
}

wafer::NCCWorkerWindowSummary
wafer::analyzeNCCWorkerWindows(mlir::ModuleOp module) {
  NCCWorkerWindowSummary summary;
  if (!module)
    return summary;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    for (mlir::Block &block : function.getBody()) {
      uint32_t pendingWorkers = 0;
      analyzeNCCWorkerBlock(block, pendingWorkers, summary);
    }
  }
  return summary;
}

wafer::LocalInstructionCompletion
wafer::classifyLocalInstructionCompletion(mlir::Operation *operation) {
  return getNCCCompletionContract(operation).behavior;
}

std::optional<wafer::NCCWorker>
wafer::getNCCIssueWorker(mlir::Operation *operation) {
  return getNCCCompletionContract(operation).issueWorker;
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
