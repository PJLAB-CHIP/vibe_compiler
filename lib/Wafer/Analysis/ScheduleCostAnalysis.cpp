//===- ScheduleCostAnalysis.cpp - Instruction cost API ---------*- C++ -*-===//

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "ScheduleCost/Internal.h"
#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Parallel.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::analysis {
namespace detail {

unsigned getKnowledgeSeverity(ScheduleCostKnowledge knowledge) {
  switch (knowledge) {
  case ScheduleCostKnowledge::Known:
    return 0;
  case ScheduleCostKnowledge::Unavailable:
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

Quantity Quantity::unavailable(ScheduleCostReason reason) {
  return {0, ScheduleCostKnowledge::Unavailable, reason};
}

Quantity Quantity::unsupported(ScheduleCostReason reason) {
  return {0, ScheduleCostKnowledge::Unsupported, reason};
}

Quantity Quantity::overflow() {
  return {0, ScheduleCostKnowledge::Overflow,
          ScheduleCostReason::ArithmeticOverflow};
}

Quantity multiply(Quantity lhs, Quantity rhs) {
  // A statically unreachable region contributes no work even when a nested
  // bound or path is dynamic. Preserve that exact zero instead of allowing an
  // unavailable fact to leak out of dead structured control flow.
  if ((lhs.knowledge == ScheduleCostKnowledge::Known && lhs.value == 0) ||
      (rhs.knowledge == ScheduleCostKnowledge::Known && rhs.value == 0))
    return Quantity{0};
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

TargetScheduleCostPolicy getTargetScheduleCostPolicy() {
  return TargetScheduleCostPolicy{};
}

const ScheduleCostMetric &
ScheduleNoCCost::directional(NoCDirection direction) const {
  return directionalTransmitBytes[static_cast<size_t>(direction)];
}

InstructionProgramCost
analyzeInstructionProgramCost(mlir::Operation *root,
                              const TargetScheduleCostPolicy &policy) {
  InstructionProgramCost cost;
  if (!root)
    return cost;
  detail::collectExecutionCost(root, cost);
  detail::collectSPMHighWater(root, cost, policy);
  detail::collectDDRHighWater(root, cost);
  return cost;
}

static InstructionProgramCost analyzeInstructionProgramCostSlice(
    mlir::Operation *root, const TargetScheduleCostPolicy &policy,
    const llvm::DenseSet<mlir::Operation *> &includedOperations) {
  InstructionProgramCost cost;
  if (!root)
    return cost;
  auto include = [&](mlir::Operation *operation) {
    return includedOperations.contains(operation);
  };
  detail::collectExecutionCost(root, cost, include);
  detail::collectSPMHighWater(root, cost, policy, include);
  detail::collectDDRHighWater(root, cost, include);
  return cost;
}

namespace {

static void addMetric(ScheduleCostMetric &aggregate,
                      const ScheduleCostMetric &tileMetric) {
  if (!tileMetric.isKnown()) {
    detail::degrade(aggregate, tileMetric.knowledge, tileMetric.reason);
    return;
  }
  if (!aggregate.isKnown())
    return;
  uint64_t result = 0;
  if (!detail::checkedAdd(aggregate.value, tileMetric.value, result)) {
    detail::degrade(aggregate, ScheduleCostKnowledge::Overflow,
                    ScheduleCostReason::ArithmeticOverflow);
    return;
  }
  aggregate.value = result;
}

static void maximizeMetric(ScheduleCostMetric &aggregate,
                           const ScheduleCostMetric &tileMetric) {
  if (!tileMetric.isKnown()) {
    detail::degrade(aggregate, tileMetric.knowledge, tileMetric.reason);
    return;
  }
  if (aggregate.isKnown())
    aggregate.value = std::max(aggregate.value, tileMetric.value);
}

static void addComputeCost(ScheduleComputeCost &aggregate,
                           const ScheduleComputeCost &tileCost) {
  addMetric(aggregate.npuF16Bf16LogicalOps, tileCost.npuF16Bf16LogicalOps);
  addMetric(aggregate.npuOtherLogicalOps, tileCost.npuOtherLogicalOps);
  addMetric(aggregate.vectorF16Bf16LogicalOps,
            tileCost.vectorF16Bf16LogicalOps);
  addMetric(aggregate.vectorF32LogicalOps, tileCost.vectorF32LogicalOps);
  addMetric(aggregate.vectorOtherLogicalOps, tileCost.vectorOtherLogicalOps);
}

static void maximizeComputeCost(ScheduleComputeCost &maximum,
                                const ScheduleComputeCost &tileCost) {
  maximizeMetric(maximum.npuF16Bf16LogicalOps, tileCost.npuF16Bf16LogicalOps);
  maximizeMetric(maximum.npuOtherLogicalOps, tileCost.npuOtherLogicalOps);
  maximizeMetric(maximum.vectorF16Bf16LogicalOps,
                 tileCost.vectorF16Bf16LogicalOps);
  maximizeMetric(maximum.vectorF32LogicalOps, tileCost.vectorF32LogicalOps);
  maximizeMetric(maximum.vectorOtherLogicalOps, tileCost.vectorOtherLogicalOps);
}

static void addNoCCost(ScheduleNoCCost &aggregate,
                       const ScheduleNoCCost &tileCost) {
  addMetric(aggregate.staticIssueSiteCount, tileCost.staticIssueSiteCount);
  addMetric(aggregate.aggregateTransmitBytes, tileCost.aggregateTransmitBytes);
  addMetric(aggregate.aggregateReceiveBytes, tileCost.aggregateReceiveBytes);
  addMetric(aggregate.transmitMessageCount, tileCost.transmitMessageCount);
  addMetric(aggregate.receiveMessageCount, tileCost.receiveMessageCount);
  addMetric(aggregate.waitOperationCount, tileCost.waitOperationCount);
  addMetric(aggregate.waitedEventCount, tileCost.waitedEventCount);
  for (size_t index = 0; index < aggregate.directionalTransmitBytes.size();
       ++index)
    addMetric(aggregate.directionalTransmitBytes[index],
              tileCost.directionalTransmitBytes[index]);
}

static void addExecutionCount(InstructionExecutionCount &aggregate,
                              const InstructionExecutionCount &tileCount) {
  addMetric(aggregate.staticSites, tileCount.staticSites);
  addMetric(aggregate.exactExecutions, tileCount.exactExecutions);
  addMetric(aggregate.lowerBound, tileCount.lowerBound);
  addMetric(aggregate.upperBound, tileCount.upperBound);
}

static void maximizeExecutionCount(InstructionExecutionCount &maximum,
                                   const InstructionExecutionCount &tileCount) {
  maximizeMetric(maximum.staticSites, tileCount.staticSites);
  maximizeMetric(maximum.exactExecutions, tileCount.exactExecutions);
  maximizeMetric(maximum.lowerBound, tileCount.lowerBound);
  maximizeMetric(maximum.upperBound, tileCount.upperBound);
}

static void addWork(InstructionProgramWork &aggregate,
                    const InstructionProgramWork &tileWork) {
  for (detail::InstructionWorkCountMember member :
       detail::kInstructionWorkCountMembers)
    addExecutionCount(aggregate.*member, tileWork.*member);
}

static void maximizeWork(InstructionProgramWork &maximum,
                         const InstructionProgramWork &tileWork) {
  for (detail::InstructionWorkCountMember member :
       detail::kInstructionWorkCountMembers)
    maximizeExecutionCount(maximum.*member, tileWork.*member);
}

struct PendingTileTransmit {
  PhysicalTileId sourceTile = PhysicalTileId(-1);
  int64_t peer = -1;
  detail::Quantity payloadBytes;
  detail::Quantity messageCount;
};

struct ModeledDirectedLinkLoad {
  PhysicalTileDirectedLink link;
  uint64_t bytes = 0;
};

static mlir::ModuleOp getContainingModule(mlir::Operation *root) {
  if (!root)
    return {};
  if (auto module = mlir::dyn_cast<mlir::ModuleOp>(root))
    return module;
  return root->getParentOfType<mlir::ModuleOp>();
}

static bool
hasEquivalentOnCardTopology(const PhysicalTopology &lhs,
                            const PhysicalTopology &rhs,
                            llvm::ArrayRef<PhysicalTileId> expectedTileIds) {
  if (lhs.getCardCount() != 1 || rhs.getCardCount() != 1 ||
      lhs.getTilesPerCard() != rhs.getTilesPerCard() ||
      lhs.getCardGrid() != rhs.getCardGrid() ||
      lhs.getTileGrid() != rhs.getTileGrid())
    return false;
  std::optional<llvm::ArrayRef<PhysicalTileId>> available =
      rhs.getAvailableTileIds(PhysicalCardId(0));
  return available && *available == expectedTileIds;
}

static void degradeInvalidPhysicalTopology(
    bool hasTransmit, ScheduleCostMetric &minimumHopLinkByteDemand,
    ScheduleCostMetric &minimumHopMessageDemand,
    ScheduleCostMetric &directedNoCLinkCount,
    ScheduleCostMetric &idealizedMinimumPeakLinkByteDemand,
    ModeledNoCRouteCost &modeledNoCRoute,
    ScheduleCostMetric &maximumNoCHopCount) {
  detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
  if (!hasTransmit)
    return;
  detail::degrade(minimumHopLinkByteDemand, ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
  detail::degrade(minimumHopMessageDemand, ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
  detail::degrade(idealizedMinimumPeakLinkByteDemand,
                  ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
  detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                  ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
  detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unavailable,
                  ScheduleCostReason::InvalidExecutionTopology);
}

static void collectMinimumHopLinkByteDemand(
    llvm::ArrayRef<PhysicalTileInstructionProgram> tilePrograms,
    llvm::ArrayRef<llvm::DenseSet<mlir::Operation *>> includedOperations,
    ScheduleCostMetric &minimumHopLinkByteDemand,
    ScheduleCostMetric &minimumHopMessageDemand,
    ScheduleCostMetric &directedNoCLinkCount,
    ScheduleCostMetric &idealizedMinimumPeakLinkByteDemand,
    ModeledNoCRouteCost &modeledNoCRoute,
    ScheduleCostMetric &maximumNoCHopCount) {
  // The caller supplies the complete single-card Tile programs with explicit
  // physical identity.  The direct topology validates that domain; neither
  // vector position nor logical partition identity is interpreted as tile_id.
  std::optional<PhysicalTopology> topology;
  bool validTopology = !tilePrograms.empty();
  for (const PhysicalTileInstructionProgram &program : tilePrograms) {
    mlir::ModuleOp module = getContainingModule(program.root);
    mlir::FailureOr<PhysicalTopology> current =
        PhysicalTopology::create(module);
    if (mlir::failed(current) || current->getCardCount() != 1) {
      validTopology = false;
      continue;
    }
    std::optional<llvm::ArrayRef<PhysicalTileId>> available =
        current->getAvailableTileIds(PhysicalCardId(0));
    if (!available || available->size() != tilePrograms.size()) {
      validTopology = false;
      continue;
    }
    if (!topology) {
      topology.emplace(std::move(*current));
      continue;
    }
    std::optional<llvm::ArrayRef<PhysicalTileId>> expected =
        topology->getAvailableTileIds(PhysicalCardId(0));
    if (!expected ||
        !hasEquivalentOnCardTopology(*topology, *current, *expected))
      validTopology = false;
  }

  if (!topology)
    validTopology = false;
  llvm::SmallVector<PhysicalTileId, 16> providedTileIds;
  providedTileIds.reserve(tilePrograms.size());
  for (const PhysicalTileInstructionProgram &program : tilePrograms)
    providedTileIds.push_back(program.tileId);
  llvm::sort(providedTileIds, [](PhysicalTileId lhs, PhysicalTileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTileIds =
      topology ? topology->getAvailableTileIds(PhysicalCardId(0))
               : std::nullopt;
  if (!availableTileIds || !llvm::equal(providedTileIds, *availableTileIds) ||
      std::adjacent_find(providedTileIds.begin(), providedTileIds.end()) !=
          providedTileIds.end())
    validTopology = false;
  if (validTopology) {
    for (PhysicalTileId tile : *availableTileIds) {
      mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 4>> neighbors =
          topology->getOnCardNeighbors(PhysicalCardId(0), tile);
      if (mlir::failed(neighbors)) {
        validTopology = false;
        break;
      }
      uint64_t count = 0;
      if (!detail::checkedAdd(directedNoCLinkCount.value, neighbors->size(),
                              count)) {
        detail::degrade(directedNoCLinkCount, ScheduleCostKnowledge::Overflow,
                        ScheduleCostReason::ArithmeticOverflow);
        validTopology = false;
        break;
      }
      directedNoCLinkCount.value = count;
    }
  }

  llvm::SmallVector<PendingTileTransmit, 32> transmits;
  for (auto [tileIndex, program] : llvm::enumerate(tilePrograms)) {
    mlir::Operation *root = program.root;
    if (!root)
      continue;
    auto collect = [&](mlir::Operation *operation,
                       detail::Quantity multiplicity) {
      if (!includedOperations.empty() &&
          !includedOperations[tileIndex].contains(operation))
        return;
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
      transmits.push_back(
          {program.tileId, send.getPeerAttr().getInt(), payload, multiplicity});
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

  if (!validTopology) {
    degradeInvalidPhysicalTopology(
        !transmits.empty(), minimumHopLinkByteDemand, minimumHopMessageDemand,
        directedNoCLinkCount, idealizedMinimumPeakLinkByteDemand,
        modeledNoCRoute, maximumNoCHopCount);
    return;
  }
  if (transmits.empty())
    return;

  llvm::SmallVector<ModeledDirectedLinkLoad, 32> modeledLinkLoads;
  for (const PendingTileTransmit &transmit : transmits) {
    PhysicalTileId peer(transmit.peer);
    std::optional<uint64_t> hops = topology->getOnCardShortestHopDistance(
        PhysicalCardId(0), transmit.sourceTile, peer);
    mlir::FailureOr<llvm::SmallVector<PhysicalTileDirectedLink, 8>> route =
        topology->getCanonicalOnCardPath(PhysicalCardId(0), transmit.sourceTile,
                                         peer);
    if (!hops || mlir::failed(route) || route->size() != *hops) {
      detail::degrade(minimumHopLinkByteDemand,
                      ScheduleCostKnowledge::Unavailable,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(minimumHopMessageDemand,
                      ScheduleCostKnowledge::Unavailable,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(idealizedMinimumPeakLinkByteDemand,
                      ScheduleCostKnowledge::Unavailable,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(modeledNoCRoute.peakDirectedLinkByteDemand,
                      ScheduleCostKnowledge::Unavailable,
                      ScheduleCostReason::InvalidExecutionTopology);
      detail::degrade(maximumNoCHopCount, ScheduleCostKnowledge::Unavailable,
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
    for (const PhysicalTileDirectedLink &link : *route) {
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
                      ScheduleCostKnowledge::Unavailable,
                      ScheduleCostReason::InvalidExecutionTopology);
    return;
  }
  idealizedMinimumPeakLinkByteDemand.value =
      minimumHopLinkByteDemand.value / links +
      static_cast<uint64_t>(minimumHopLinkByteDemand.value % links != 0);
}

} // namespace

static WholeCardInstructionProgramCost
analyzeWholeCardInstructionProgramCostImpl(
    llvm::ArrayRef<PhysicalTileInstructionProgram> tilePrograms,
    const TargetScheduleCostPolicy &policy,
    llvm::ArrayRef<llvm::DenseSet<mlir::Operation *>> includedOperations) {
  WholeCardInstructionProgramCost result;
  std::vector<InstructionProgramCost> tileCosts(tilePrograms.size());
  llvm::parallelFor(0, tilePrograms.size(), [&](size_t tileIndex) {
    tileCosts[tileIndex] =
        includedOperations.empty()
            ? analyzeInstructionProgramCost(tilePrograms[tileIndex].root,
                                            policy)
            : analyzeInstructionProgramCostSlice(tilePrograms[tileIndex].root,
                                                 policy,
                                                 includedOperations[tileIndex]);
  });
  result.tileCosts.reserve(tilePrograms.size());
  for (InstructionProgramCost &tileCost : tileCosts) {
    addWork(result.aggregateWork, tileCost.work);
    maximizeWork(result.maximumTileWork, tileCost.work);
    addComputeCost(result.aggregateCompute, tileCost.compute);
    maximizeComputeCost(result.maximumTileCompute, tileCost.compute);
    addMetric(result.aggregateDDRReadBytes, tileCost.ddrReadBytes);
    addMetric(result.aggregateDDRWriteBytes, tileCost.ddrWriteBytes);
    addMetric(result.aggregateSPMMovementBytes, tileCost.spmMovementBytes);
    addMetric(result.aggregateGatherScatterBytes, tileCost.gatherScatterBytes);
    maximizeMetric(result.maximumTileSPMMovementBytes,
                   tileCost.spmMovementBytes);
    maximizeMetric(result.maximumTileGatherScatterBytes,
                   tileCost.gatherScatterBytes);
    addNoCCost(result.aggregateNoC, tileCost.noc);
    maximizeMetric(result.maximumTileNoCTransmitBytes,
                   tileCost.noc.aggregateTransmitBytes);
    maximizeMetric(result.maximumTileNoCReceiveBytes,
                   tileCost.noc.aggregateReceiveBytes);
    maximizeMetric(result.maximumTileNoCTransmitMessageCount,
                   tileCost.noc.transmitMessageCount);
    maximizeMetric(result.maximumTileNoCReceiveMessageCount,
                   tileCost.noc.receiveMessageCount);
    maximizeMetric(result.maximumTileSPMHighWaterBytes,
                   tileCost.spmHighWaterBytes);
    addMetric(result.summedTileSPMHighWaterBytes, tileCost.spmHighWaterBytes);
    maximizeMetric(result.maximumTileDDRHighWaterBytes,
                   tileCost.ddrHighWaterBytes);
    addMetric(result.summedTileDDRHighWaterBytes, tileCost.ddrHighWaterBytes);
    addMetric(result.aggregateCompilerOwnedSPMBufferCount,
              tileCost.compilerOwnedSPMBufferCount);
    addMetric(result.aggregateCompilerOwnedDDRBufferCount,
              tileCost.compilerOwnedDDRBufferCount);
    maximizeMetric(result.maximumTileCompilerOwnedSPMBufferCount,
                   tileCost.compilerOwnedSPMBufferCount);
    maximizeMetric(result.maximumTileCompilerOwnedDDRBufferCount,
                   tileCost.compilerOwnedDDRBufferCount);
    result.tileCosts.push_back(std::move(tileCost));
  }
  result.aggregateInstructionCount =
      result.aggregateWork.instructions.exactExecutions;
  result.aggregateEventCount =
      result.aggregateWork.asynchronousEvents.exactExecutions;
  result.aggregateNCCJoinCount = result.aggregateWork.nccJoins.exactExecutions;
  result.aggregateSteadyStateNCCJoinCount =
      result.aggregateWork.steadyStateNCCJoins.exactExecutions;
  result.aggregateNonTerminalNCCJoinCount =
      result.aggregateWork.nonTerminalNCCJoins.exactExecutions;
  result.aggregateNCCParticipantWaitCount =
      result.aggregateWork.nccParticipantWaits.exactExecutions;
  result.aggregateSteadyStateNCCParticipantWaitCount =
      result.aggregateWork.steadyStateNCCParticipantWaits.exactExecutions;
  result.aggregateNonTerminalNCCParticipantWaitCount =
      result.aggregateWork.nonTerminalNCCParticipantWaits.exactExecutions;
  result.aggregateIntrinsicNCCDrainCount =
      result.aggregateWork.intrinsicNCCDrains.exactExecutions;
  collectMinimumHopLinkByteDemand(
      tilePrograms, includedOperations, result.minimumHopLinkByteDemand,
      result.minimumHopMessageDemand, result.directedNoCLinkCount,
      result.idealizedMinimumPeakLinkByteDemand, result.modeledNoCRoute,
      result.maximumNoCHopCount);
  return result;
}

WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCost(
    llvm::ArrayRef<PhysicalTileInstructionProgram> tilePrograms,
    const TargetScheduleCostPolicy &policy) {
  return analyzeWholeCardInstructionProgramCostImpl(tilePrograms, policy, {});
}

WholeCardInstructionProgramCost analyzeWholeCardInstructionProgramCostSlice(
    llvm::ArrayRef<PhysicalTileInstructionProgramSlice> tilePrograms,
    const TargetScheduleCostPolicy &policy) {
  llvm::SmallVector<PhysicalTileInstructionProgram, 16> programs;
  std::vector<llvm::DenseSet<mlir::Operation *>> includedOperations;
  programs.reserve(tilePrograms.size());
  includedOperations.reserve(tilePrograms.size());
  for (const PhysicalTileInstructionProgramSlice &slice : tilePrograms) {
    programs.push_back({slice.tileId, slice.root});
    includedOperations.emplace_back(slice.includedOperations.begin(),
                                    slice.includedOperations.end());
  }
  return analyzeWholeCardInstructionProgramCostImpl(programs, policy,
                                                    includedOperations);
}

llvm::StringRef
stringifyScheduleCostKnowledge(ScheduleCostKnowledge knowledge) {
  switch (knowledge) {
  case ScheduleCostKnowledge::Known:
    return "known";
  case ScheduleCostKnowledge::Unavailable:
    return "unavailable";
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
  case ScheduleCostReason::UnavailablePhysicalGeometry:
    return "unavailable-physical-geometry";
  case ScheduleCostReason::UnavailableResourceBytes:
    return "unavailable-resource-bytes";
  case ScheduleCostReason::MissingAcceptedSPMOffset:
    return "missing-accepted-spm-offset";
  case ScheduleCostReason::InvalidAcceptedSPMOffset:
    return "invalid-accepted-spm-offset";
  case ScheduleCostReason::MissingAcceptedDDROffset:
    return "missing-accepted-ddr-offset";
  case ScheduleCostReason::InvalidAcceptedDDROffset:
    return "invalid-accepted-ddr-offset";
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
  case ScheduleCostReason::ArithmeticOverflow:
    return "arithmetic-overflow";
  }
  llvm_unreachable("unhandled schedule cost reason");
}

} // namespace wafer::analysis
