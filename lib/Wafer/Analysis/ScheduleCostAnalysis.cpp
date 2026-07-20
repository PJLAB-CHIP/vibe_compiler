//===- ScheduleCostAnalysis.cpp - Instruction cost API ---------*- C++ -*-===//

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "ScheduleCost/Internal.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
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
  return TargetScheduleCostPolicy(targetProfile);
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
  addMetric(aggregate.aggregateTransmitBytes, rankCost.aggregateTransmitBytes);
  addMetric(aggregate.aggregateReceiveBytes, rankCost.aggregateReceiveBytes);
  for (size_t index = 0; index < aggregate.directionalTransmitBytes.size();
       ++index)
    addMetric(aggregate.directionalTransmitBytes[index],
              rankCost.directionalTransmitBytes[index]);
  for (size_t index = 0; index < aggregate.collectiveTransmitBytes.size();
       ++index)
    addMetric(aggregate.collectiveTransmitBytes[index],
              rankCost.collectiveTransmitBytes[index]);
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
    addMetric(result.aggregateInstructionCount, rankCost.instructionCount);
    addMetric(result.aggregateEventCount, rankCost.eventCount);
    maximizeMetric(result.maximumRankDataDependencyDepth,
                   rankCost.dataDependencyDepth);
    addMetric(result.aggregateReadyOrderPriorityInversions,
              rankCost.readyOrderPriorityInversions);
    maximizeMetric(result.maximumRankSPMHighWaterBytes,
                   rankCost.spmHighWaterBytes);
    addMetric(result.summedRankSPMHighWaterBytes, rankCost.spmHighWaterBytes);
    result.rankCosts.push_back(std::move(rankCost));
  }
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
  case ScheduleCostReason::UnsupportedInstructionSemantics:
    return "unsupported-instruction-semantics";
  case ScheduleCostReason::UnsupportedComputeType:
    return "unsupported-compute-type";
  case ScheduleCostReason::ArithmeticOverflow:
    return "arithmetic-overflow";
  }
  llvm_unreachable("unhandled schedule cost reason");
}

} // namespace wafer::analysis
