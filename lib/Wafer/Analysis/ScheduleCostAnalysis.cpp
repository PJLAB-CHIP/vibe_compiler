//===- ScheduleCostAnalysis.cpp - Instruction cost API ---------*- C++ -*-===//

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "ScheduleCost/Internal.h"
#include "Wafer/Analysis/ExecutionTopologyAnalysis.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace wafer::analysis {
namespace detail {

unsigned getKnowledgeSeverity(ScheduleCostKnowledge knowledge) {
  switch (knowledge) {
  case ScheduleCostKnowledge::Known:
    return 0;
  case ScheduleCostKnowledge::Unknown:
    return 1;
  case ScheduleCostKnowledge::Unsupported:
    return 2;
  case ScheduleCostKnowledge::Overflow:
    return 3;
  }
  llvm_unreachable("unhandled schedule cost knowledge");
}

void degrade(ScheduleCostMetric &metric, ScheduleCostKnowledge knowledge,
             ScheduleCostReason reason) {
  if (getKnowledgeSeverity(knowledge) < getKnowledgeSeverity(metric.knowledge))
    return;
  if (getKnowledgeSeverity(knowledge) ==
          getKnowledgeSeverity(metric.knowledge) &&
      metric.reason != ScheduleCostReason::None)
    return;
  metric.value = 0;
  metric.knowledge = knowledge;
  metric.reason = reason;
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

Quantity Quantity::unknown(ScheduleCostReason reason) {
  return {0, ScheduleCostKnowledge::Unknown, reason};
}

Quantity Quantity::unsupported(ScheduleCostReason reason) {
  return {0, ScheduleCostKnowledge::Unsupported, reason};
}

Quantity Quantity::overflow() {
  return {0, ScheduleCostKnowledge::Overflow,
          ScheduleCostReason::ArithmeticOverflow};
}

Quantity multiply(Quantity lhs, Quantity rhs) {
  if (lhs.knowledge != ScheduleCostKnowledge::Known ||
      rhs.knowledge != ScheduleCostKnowledge::Known) {
    if (getKnowledgeSeverity(rhs.knowledge) >
        getKnowledgeSeverity(lhs.knowledge))
      return rhs;
    return lhs;
  }
  uint64_t result = 0;
  if (!checkedMul(lhs.value, rhs.value, result))
    return Quantity::overflow();
  return {result};
}

Quantity multiply(Quantity lhs, uint64_t rhs) {
  return multiply(lhs, Quantity{rhs});
}

void add(ScheduleCostMetric &metric, Quantity quantity) {
  if (quantity.knowledge != ScheduleCostKnowledge::Known) {
    degrade(metric, quantity.knowledge, quantity.reason);
    return;
  }
  if (!metric.isKnown())
    return;
  uint64_t result = 0;
  if (!checkedAdd(metric.value, quantity.value, result)) {
    degrade(metric, ScheduleCostKnowledge::Overflow,
            ScheduleCostReason::ArithmeticOverflow);
    return;
  }
  metric.value = result;
}

} // namespace detail

TargetScheduleCostPolicy
getTargetScheduleCostPolicy(TargetProfileId targetProfile) {
  TargetScheduleCostPolicy policy(targetProfile);
  if (targetProfile == TargetProfileId::waferTx81SingleCardKernelV1() ||
      targetProfile == TargetProfileId::waferTx81SingleCardKernelV2() ||
      targetProfile == TargetProfileId::waferTx81SingleCardKernelV3()) {
    policy.qualifiedOverlapFamilyMask =
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::CT)) |
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::RDMA)) |
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::WDMA));
    policy.qualifiedOverlapWorker = static_cast<uint32_t>(NCCWorker::Worker0);
  }
  if (targetProfile == TargetProfileId::waferTx81SingleCardKernelV3()) {
    // Only V3 has a real prepare/explicit-issue/exact-wait target contract.
    // This mask recognizes an auditable structural opportunity. The separate
    // target scheduling profitability registry intentionally remains Unknown
    // until matched board evidence exists.
    policy.qualifiedDirectDTEOverlapFamilyMask =
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::DTE)) |
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::CT)) |
        (uint32_t{1} << static_cast<uint32_t>(InstrFamily::NE));
  }
  return policy;
}

const ScheduleCostMetric &
ScheduleNoCCost::directional(NoCDirection direction) const {
  return directionalTransmitBytes[static_cast<size_t>(direction)];
}

const ScheduleCostMetric &
ScheduleNoCCost::collective(NoCCollectiveKind kind) const {
  return collectiveTransmitBytes[static_cast<size_t>(kind)];
}

InstructionProgramCost
analyzeInstructionProgramCost(mlir::Operation *root,
                              const TargetScheduleCostPolicy &policy) {
  InstructionProgramCost cost;
  if (!root)
    return cost;
  detail::collectExecutionCost(root, cost);
  detail::collectDataDependencyDepth(root, cost);
  detail::collectQualifiedOverlapWindows(root, cost, policy);
  detail::collectSPMHighWater(root, cost, policy);
  return cost;
}

namespace {

static void addMetric(ScheduleCostMetric &aggregate,
                      const ScheduleCostMetric &rankMetric) {
  if (!rankMetric.isKnown()) {
    detail::degrade(aggregate, rankMetric.knowledge, rankMetric.reason);
    return;
  }
  if (!aggregate.isKnown())
    return;
  uint64_t result = 0;
  if (!detail::checkedAdd(aggregate.value, rankMetric.value, result)) {
    detail::degrade(aggregate, ScheduleCostKnowledge::Overflow,
                    ScheduleCostReason::ArithmeticOverflow);
    return;
  }
  aggregate.value = result;
}

static void maximizeMetric(ScheduleCostMetric &aggregate,
                           const ScheduleCostMetric &rankMetric) {
  if (!rankMetric.isKnown()) {
    detail::degrade(aggregate, rankMetric.knowledge, rankMetric.reason);
    return;
  }
  if (aggregate.isKnown())
    aggregate.value = std::max(aggregate.value, rankMetric.value);
}

static void addComputeCost(ScheduleComputeCost &aggregate,
                           const ScheduleComputeCost &rankCost) {
  addMetric(aggregate.npuF16Bf16LogicalOps, rankCost.npuF16Bf16LogicalOps);
  addMetric(aggregate.npuOtherLogicalOps, rankCost.npuOtherLogicalOps);
  addMetric(aggregate.vectorF16Bf16LogicalOps,
            rankCost.vectorF16Bf16LogicalOps);
  addMetric(aggregate.vectorF32LogicalOps, rankCost.vectorF32LogicalOps);
  addMetric(aggregate.vectorOtherLogicalOps, rankCost.vectorOtherLogicalOps);
}

static void addNoCCost(ScheduleNoCCost &aggregate,
                       const ScheduleNoCCost &rankCost) {
  addMetric(aggregate.staticIssueSiteCount, rankCost.staticIssueSiteCount);
  addMetric(aggregate.aggregateTransmitBytes, rankCost.aggregateTransmitBytes);
  addMetric(aggregate.aggregateReceiveBytes, rankCost.aggregateReceiveBytes);
  addMetric(aggregate.transmitMessageCount, rankCost.transmitMessageCount);
  addMetric(aggregate.receiveMessageCount, rankCost.receiveMessageCount);
  addMetric(aggregate.waitOperationCount, rankCost.waitOperationCount);
  addMetric(aggregate.waitedEventCount, rankCost.waitedEventCount);
  for (size_t index = 0; index < aggregate.directionalTransmitBytes.size();
       ++index)
    addMetric(aggregate.directionalTransmitBytes[index],
              rankCost.directionalTransmitBytes[index]);
  for (size_t index = 0; index < aggregate.collectiveTransmitBytes.size();
       ++index)
    addMetric(aggregate.collectiveTransmitBytes[index],
              rankCost.collectiveTransmitBytes[index]);
}

struct PendingHopTransmit {
  size_t sourceRank = 0;
  int64_t peer = -1;
  detail::Quantity payloadBytes;
  detail::Quantity messageCount;
};

struct ModeledDirectedLinkLoad {
  ExecutionDirectedLink link;
  uint64_t bytes = 0;
};

static mlir::ModuleOp getContainingModule(mlir::Operation *root) {
  if (!root)
    return {};
  if (auto module = mlir::dyn_cast<mlir::ModuleOp>(root))
    return module;
  return root->getParentOfType<mlir::ModuleOp>();
}

static void collectMinimumHopLinkByteDemand(
    llvm::ArrayRef<mlir::Operation *> rankRoots,
    ScheduleCostMetric &minimumHopLinkByteDemand,
    ScheduleCostMetric &minimumHopMessageDemand,
    ScheduleCostMetric &directedNoCLinkCount,
    ScheduleCostMetric &idealizedMinimumPeakLinkByteDemand,
    ModeledNoCRouteCost &modeledNoCRoute,
    ScheduleCostMetric &maximumNoCHopCount) {
  llvm::SmallVector<PendingHopTransmit, 32> transmits;
  for (auto [sourceRank, root] : llvm::enumerate(rankRoots)) {
    if (!root) {
      detail::degrade(minimumHopLinkByteDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(minimumHopMessageDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      continue;
    }
    auto collect = [&](mlir::Operation *operation,
                       detail::Quantity multiplicity) {
      auto send = mlir::dyn_cast<InstrDTESendOp>(operation);
      if (!send)
        return;
      detail::Quantity bytes =
          send.getBytes() < 0
              ? detail::Quantity::unsupported(
                    ScheduleCostReason::UnsupportedInstructionSemantics)
              : detail::Quantity{static_cast<uint64_t>(send.getBytes())};
      detail::Quantity payload = detail::multiply(bytes, multiplicity);
      if (payload.knowledge != ScheduleCostKnowledge::Known) {
        detail::add(minimumHopLinkByteDemand, payload);
        detail::add(idealizedMinimumPeakLinkByteDemand, payload);
        detail::add(modeledNoCRoute.peakDirectedLinkByteDemand, payload);
      }
      if (multiplicity.knowledge == ScheduleCostKnowledge::Known &&
          multiplicity.value == 0)
        return;
      transmits.push_back({static_cast<size_t>(sourceRank),
                           send.getPeerAttr().getInt(), payload, multiplicity});
    };
    auto markUnsupported = [&]() {
      detail::degrade(minimumHopLinkByteDemand,
                      ScheduleCostKnowledge::Unsupported,
                      ScheduleCostReason::UnsupportedControlFlow);
      detail::degrade(minimumHopMessageDemand,
                      ScheduleCostKnowledge::Unsupported,
                      ScheduleCostReason::UnsupportedControlFlow);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unsupported,
                      ScheduleCostReason::UnsupportedControlFlow);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unsupported,
                      ScheduleCostReason::UnsupportedControlFlow);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unsupported,
                      ScheduleCostReason::UnsupportedControlFlow);
    };
    detail::walkInstructionProgram(root, collect, markUnsupported);
  }

  // Zero final sends imply exact zero traffic independently of topology, but
  // directedNoCLinkCount is still a topology fact and must not remain a
  // fabricated Known(0). A missing topology therefore degrades only that fact
  // for an otherwise exact NoC-free program.
  if (transmits.empty() && !minimumHopLinkByteDemand.isKnown())
    detail::degrade(idealizedMinimumPeakLinkByteDemand,
                    minimumHopLinkByteDemand.knowledge,
                    minimumHopLinkByteDemand.reason);
  if (rankRoots.empty()) {
    detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Unknown,
                    ScheduleCostReason::InvalidExecutionTopology);
    return;
  }

  llvm::SmallVector<ExecutionTopologyAnalysis, 16> rankTopologies;
  rankTopologies.reserve(rankRoots.size());
  for (mlir::Operation *root : rankRoots) {
    mlir::ModuleOp module = getContainingModule(root);
    mlir::FailureOr<ExecutionTopologyAnalysis> topology =
        ExecutionTopologyAnalysis::create(module);
    if (mlir::failed(topology)) {
      detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      if (transmits.empty())
        return;
      detail::degrade(minimumHopLinkByteDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(minimumHopMessageDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      return;
    }
    if (topology->getRankCount() != static_cast<int64_t>(rankRoots.size()) ||
        (!rankTopologies.empty() &&
         !rankTopologies.front().isEquivalentTo(*topology))) {
      detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      if (transmits.empty())
        return;
      detail::degrade(minimumHopLinkByteDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(minimumHopMessageDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      return;
    }
    rankTopologies.push_back(std::move(*topology));
  }

  const ExecutionTopologyAnalysis &topology = rankTopologies.front();
  directedNoCLinkCount.value = topology.getDirectedLinkCount();
  if (transmits.empty())
    return;
  llvm::SmallVector<ModeledDirectedLinkLoad, 32> modeledLinkLoads;
  for (const PendingHopTransmit &transmit : transmits) {
    std::optional<uint64_t> hops = topology.getShortestHopDistance(
        static_cast<int64_t>(transmit.sourceRank), transmit.peer);
    mlir::FailureOr<llvm::SmallVector<ExecutionDirectedLink, 8>> route =
        topology.getCanonicalShortestPath(
            static_cast<int64_t>(transmit.sourceRank), transmit.peer);
    if (!hops || mlir::failed(route) || route->size() != *hops) {
      detail::degrade(minimumHopLinkByteDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(minimumHopMessageDemand, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
      return;
    }
    if (maximumNoCHopCount.isKnown())
      maximumNoCHopCount.value = std::max(maximumNoCHopCount.value, *hops);
    detail::add(minimumHopLinkByteDemand,
                detail::multiply(transmit.payloadBytes, *hops));
    detail::add(minimumHopMessageDemand,
                detail::multiply(transmit.messageCount, *hops));
    if (transmit.payloadBytes.knowledge != ScheduleCostKnowledge::Known)
      continue;
    for (const ExecutionDirectedLink &link : *route) {
      auto existing =
          std::find_if(modeledLinkLoads.begin(), modeledLinkLoads.end(),
                       [&](const ModeledDirectedLinkLoad &load) {
                         return load.link == link;
                       });
      if (existing == modeledLinkLoads.end()) {
        modeledLinkLoads.push_back({link, transmit.payloadBytes.value});
        continue;
      }
      uint64_t sum = 0;
      if (!detail::checkedAdd(existing->bytes, transmit.payloadBytes.value,
                              sum)) {
        detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                        ScheduleCostKnowledge::Overflow,
                        ScheduleCostReason::ArithmeticOverflow);
        continue;
      }
      existing->bytes = sum;
    }
  }
  if (modeledNoCRoute.peakDirectedLinkByteDemand.isKnown())
    for (const ModeledDirectedLinkLoad &load : modeledLinkLoads)
      modeledNoCRoute.peakDirectedLinkByteDemand.value = std::max(
          modeledNoCRoute.peakDirectedLinkByteDemand.value, load.bytes);
  if (!minimumHopLinkByteDemand.isKnown()) {
    detail::degrade(idealizedMinimumPeakLinkByteDemand,
                    minimumHopLinkByteDemand.knowledge,
                    minimumHopLinkByteDemand.reason);
    return;
  }
  const uint64_t links = directedNoCLinkCount.value;
  if (links == 0) {
    if (minimumHopLinkByteDemand.value != 0)
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unknown,
                      ScheduleCostReason::InvalidExecutionTopology);
    return;
  }
  idealizedMinimumPeakLinkByteDemand.value =
      minimumHopLinkByteDemand.value / links +
      static_cast<uint64_t>(minimumHopLinkByteDemand.value % links != 0);
}

} // namespace

WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCost(
    llvm::ArrayRef<mlir::Operation *> rankRoots,
    const TargetScheduleCostPolicy &policy) {
  WholeCardInstructionProgramCost result;
  result.rankCosts.reserve(rankRoots.size());
  for (mlir::Operation *root : rankRoots) {
    InstructionProgramCost rankCost =
        analyzeInstructionProgramCost(root, policy);
    addComputeCost(result.aggregateCompute, rankCost.compute);
    addMetric(result.aggregateDDRReadBytes, rankCost.ddrReadBytes);
    addMetric(result.aggregateDDRWriteBytes, rankCost.ddrWriteBytes);
    addMetric(result.aggregateSPMMovementBytes, rankCost.spmMovementBytes);
    addNoCCost(result.aggregateNoC, rankCost.noc);
    maximizeMetric(result.maximumRankNoCTransmitBytes,
                   rankCost.noc.aggregateTransmitBytes);
    maximizeMetric(result.maximumRankNoCReceiveBytes,
                   rankCost.noc.aggregateReceiveBytes);
    maximizeMetric(result.maximumRankNoCTransmitMessageCount,
                   rankCost.noc.transmitMessageCount);
    maximizeMetric(result.maximumRankNoCReceiveMessageCount,
                   rankCost.noc.receiveMessageCount);
    addMetric(result.aggregateInstructionCount, rankCost.instructionCount);
    addMetric(result.aggregateEventCount, rankCost.eventCount);
    addMetric(result.aggregateNCCJoinCount, rankCost.nccJoinCount);
    addMetric(result.aggregateSteadyStateNCCJoinCount,
              rankCost.steadyStateNCCJoinCount);
    addMetric(result.aggregateNonTerminalNCCJoinCount,
              rankCost.nonTerminalNCCJoinCount);
    addMetric(result.aggregateNCCParticipantWaitCount,
              rankCost.nccParticipantWaitCount);
    addMetric(result.aggregateSteadyStateNCCParticipantWaitCount,
              rankCost.steadyStateNCCParticipantWaitCount);
    addMetric(result.aggregateNonTerminalNCCParticipantWaitCount,
              rankCost.nonTerminalNCCParticipantWaitCount);
    addMetric(result.aggregateIntrinsicNCCDrainCount,
              rankCost.intrinsicNCCDrainCount);
    maximizeMetric(result.maximumRankDataDependencyDepth,
                   rankCost.dataDependencyDepth);
    addMetric(result.aggregateReadyOrderPriorityInversions,
              rankCost.readyOrderPriorityInversions);
    addMetric(result.aggregateQualifiedOverlapWindowCount,
              rankCost.qualifiedOverlapWindowCount);
    addMetric(result.aggregateDirectDTEComputeOverlapWindowCount,
              rankCost.directDTEComputeOverlapWindowCount);
    maximizeMetric(result.maximumRankSPMHighWaterBytes,
                   rankCost.spmHighWaterBytes);
    addMetric(result.summedRankSPMHighWaterBytes, rankCost.spmHighWaterBytes);
    result.rankCosts.push_back(std::move(rankCost));
  }
  collectMinimumHopLinkByteDemand(
      rankRoots, result.minimumHopLinkByteDemand,
      result.minimumHopMessageDemand, result.directedNoCLinkCount,
      result.idealizedMinimumPeakLinkByteDemand, result.modeledNoCRoute,
      result.maximumNoCHopCount);
  return result;
}

llvm::StringRef
stringifyScheduleCostKnowledge(ScheduleCostKnowledge knowledge) {
  switch (knowledge) {
  case ScheduleCostKnowledge::Known:
    return "known";
  case ScheduleCostKnowledge::Unknown:
    return "unknown";
  case ScheduleCostKnowledge::Unsupported:
    return "unsupported";
  case ScheduleCostKnowledge::Overflow:
    return "overflow";
  }
  llvm_unreachable("unhandled schedule cost knowledge");
}

llvm::StringRef stringifyScheduleCostReason(ScheduleCostReason reason) {
  switch (reason) {
  case ScheduleCostReason::None:
    return "none";
  case ScheduleCostReason::DynamicLoopTripCount:
    return "dynamic-loop-trip-count";
  case ScheduleCostReason::InvalidLoopStep:
    return "invalid-loop-step";
  case ScheduleCostReason::ConditionalControlFlow:
    return "conditional-control-flow";
  case ScheduleCostReason::UnsupportedControlFlow:
    return "unsupported-control-flow";
  case ScheduleCostReason::UnknownPhysicalGeometry:
    return "unknown-physical-geometry";
  case ScheduleCostReason::UnknownResourceBytes:
    return "unknown-resource-bytes";
  case ScheduleCostReason::MissingAcceptedSPMOffset:
    return "missing-accepted-spm-offset";
  case ScheduleCostReason::InvalidAcceptedSPMOffset:
    return "invalid-accepted-spm-offset";
  case ScheduleCostReason::UnsupportedSPMRoot:
    return "unsupported-spm-root";
  case ScheduleCostReason::UnresolvedNoCRoute:
    return "unresolved-noc-route";
  case ScheduleCostReason::InvalidExecutionTopology:
    return "invalid-execution-topology";
  case ScheduleCostReason::UnsupportedInstructionSemantics:
    return "unsupported-instruction-semantics";
  case ScheduleCostReason::UnsupportedComputeType:
    return "unsupported-compute-type";
  case ScheduleCostReason::MissingPerformanceCalibration:
    return "missing-performance-calibration";
  case ScheduleCostReason::ArithmeticOverflow:
    return "arithmetic-overflow";
  }
  llvm_unreachable("unhandled schedule cost reason");
}

} // namespace wafer::analysis
