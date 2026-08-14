//===- TargetSchedulingAnalysis.cpp - Scheduling query from Wafer IR ----===//

#include "Wafer/Analysis/TargetSchedulingAnalysis.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace wafer::analysis {
namespace {

static llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

static std::optional<TargetSchedulingEngine> convertEngine(InstrFamily family) {
  switch (family) {
  case InstrFamily::CT:
    return TargetSchedulingEngine::CT;
  case InstrFamily::NE:
    return TargetSchedulingEngine::NE;
  case InstrFamily::RDMA:
    return TargetSchedulingEngine::RDMA;
  case InstrFamily::WDMA:
    return TargetSchedulingEngine::WDMA;
  case InstrFamily::TDMA:
    return TargetSchedulingEngine::TDMA;
  case InstrFamily::DTE:
    return TargetSchedulingEngine::DTE;
  }
  return std::nullopt;
}

static void recordMemrefGeometry(mlir::Value value,
                                 llvm::DenseSet<mlir::Value> &visibleBuffers,
                                 TargetSchedulingWindowQuery &query) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !isWaferMemRefType(type))
    return;
  if (isWaferSPMMemRefType(type))
    visibleBuffers.insert(value);
  std::optional<WaferPhysicalTensorInfo> physical =
      computeWaferPhysicalTensorInfo(type);
  if (!physical || physical->physicalBytes < 0) {
    query.geometryKnown = false;
    return;
  }
  query.maximumPayloadBytes =
      std::max(query.maximumPayloadBytes,
               static_cast<uint64_t>(physical->physicalBytes));
}

} // namespace

llvm::Expected<TargetSchedulingWindowQuery>
analyzeTargetSchedulingWindow(mlir::ModuleOp module,
                              TargetSchedulingMechanism mechanism) {
  if (!module)
    return invalid("target scheduling analysis requires a module");

  TargetSchedulingWindowQuery query(mechanism);
  llvm::DenseSet<mlir::Value> visibleBuffers;
  bool hasNCC = false;
  bool hasDTE = false;
  bool countOverflow = false;

  module.walk([&](mlir::Operation *operation) {
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
    if (instruction) {
      std::optional<TargetSchedulingEngine> engine =
          convertEngine(instruction.getInstructionFamily());
      if (!engine) {
        query.geometryKnown = false;
      } else {
        query.engines |= targetSchedulingEngineBit(*engine);
        hasDTE |= *engine == TargetSchedulingEngine::DTE;
        hasNCC |= *engine != TargetSchedulingEngine::DTE;
      }
      if (instruction.getInstructionFamily() == InstrFamily::DTE &&
          operation->getNumResults() != 0) {
        if (query.visibleTokenCount == std::numeric_limits<uint32_t>::max())
          countOverflow = true;
        else
          ++query.visibleTokenCount;
      }
    }

    NCCSynchronizationContract completion =
        getNCCSynchronizationContract(operation);
    if (completion.behavior == NCCSynchronizationBehavior::SynchronousWriteback)
      query.observer = TargetSchedulingObserverKind::SynchronousHostWriteback;

    for (mlir::Value operand : operation->getOperands())
      recordMemrefGeometry(operand, visibleBuffers, query);
    for (mlir::Value result : operation->getResults())
      recordMemrefGeometry(result, visibleBuffers, query);
  });

  if (countOverflow ||
      visibleBuffers.size() > std::numeric_limits<uint32_t>::max())
    return invalid("target scheduling analysis structural bound overflows");
  query.visibleBufferCount = static_cast<uint32_t>(visibleBuffers.size());
  if (query.engines == 0)
    return invalid("target scheduling analysis found no typed instruction "
                   "engine");
  NCCWorkerWindowSummary workerWindows = analyzeNCCWorkerWindows(module);

  if (hasDTE && hasNCC) {
    query.workerRelation = TargetSchedulingWorkerRelation::MixedNCCAndDTE;
    query.completion =
        TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent;
  } else if (hasDTE) {
    query.workerRelation = TargetSchedulingWorkerRelation::ExactDTEEvent;
    query.completion = TargetSchedulingCompletionKind::ExactEvent;
  } else if (workerWindows.hasCrossWorkerWindow) {
    query.workerRelation = TargetSchedulingWorkerRelation::CrossNCCWorkers;
    query.completion = TargetSchedulingCompletionKind::ParticipantJoin;
  } else {
    query.workerRelation = TargetSchedulingWorkerRelation::SameNCCWorker;
    query.completion = TargetSchedulingCompletionKind::SameWorkerIssueOrder;
  }
  return query;
}

} // namespace wafer::analysis
