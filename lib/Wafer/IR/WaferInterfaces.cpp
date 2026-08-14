//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

#include "Wafer/IR/WaferInterfaces.cpp.inc"

namespace {

static void analyzeNCCWorkerBlock(mlir::Block &block, uint32_t &pendingWorkers,
                                  wafer::NCCWorkerWindowSummary &summary);

static void analyzeNCCWorkerOperation(mlir::Operation *operation,
                                      uint32_t &pendingWorkers,
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

  wafer::NCCSynchronizationContract contract =
      wafer::getNCCSynchronizationContract(operation);
  if (contract.issueWorker) {
    uint32_t worker = static_cast<uint32_t>(*contract.issueWorker);
    if (worker < wafer::kNCCWorkerCount) {
      uint32_t workerMask = uint32_t{1} << worker;
      summary.issuedWorkerMask |= workerMask;
      pendingWorkers |= workerMask;
      summary.hasCrossWorkerWindow |= llvm::popcount(pendingWorkers) >= 2;
    }
  }
  if (contract.behavior == wafer::NCCSynchronizationBehavior::ParticipantJoin ||
      contract.behavior ==
          wafer::NCCSynchronizationBehavior::SynchronousWriteback)
    pendingWorkers &= ~(contract.participantMask & wafer::kAllNCCWorkersMask);
}

static void analyzeNCCWorkerBlock(mlir::Block &block, uint32_t &pendingWorkers,
                                  wafer::NCCWorkerWindowSummary &summary) {
  for (mlir::Operation &operation : block)
    analyzeNCCWorkerOperation(&operation, pendingWorkers, summary);
}

} // namespace

wafer::NCCSynchronizationContract
wafer::getNCCSynchronizationContract(mlir::Operation *operation) {
  if (!operation)
    return {};
  if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
    uint32_t participants = 0;
    for (int64_t worker : join.getParticipants()) {
      if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount))
        return {NCCSynchronizationBehavior::ParticipantJoin, std::nullopt, 0};
      participants |= uint32_t{1} << static_cast<uint32_t>(worker);
    }
    return {NCCSynchronizationBehavior::ParticipantJoin, std::nullopt,
            participants};
  }

  if (auto peripheral = mlir::dyn_cast<InstrPeripheralOp>(operation)) {
    switch (peripheral.getKindAttr().getValue()) {
    case InstrPeripheralKind::ArgMax:
    case InstrPeripheralKind::ArgMin:
      // The production target wrapper issues CT, drains the default worker's
      // local queues, then writes both scalar results to SPM.
      return {
          NCCSynchronizationBehavior::SynchronousWriteback,
          peripheral.getIssueWorker(),
          uint32_t{1} << static_cast<uint32_t>(peripheral.getIssueWorker())};
    default:
      break;
    }
  }

  if (auto issue = mlir::dyn_cast<WaferNCCIssueOpInterface>(operation))
    return {NCCSynchronizationBehavior::OrderedAsynchronousIssue, issue.getIssueWorker(),
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

wafer::NCCSynchronizationBehavior
wafer::classifyNCCSynchronizationBehavior(mlir::Operation *operation) {
  return getNCCSynchronizationContract(operation).behavior;
}

std::optional<wafer::NCCWorker>
wafer::getNCCIssueWorker(mlir::Operation *operation) {
  return getNCCSynchronizationContract(operation).issueWorker;
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
