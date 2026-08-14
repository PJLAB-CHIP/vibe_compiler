//===- WholeDAGSchedulePlan.cpp - Accepted-IR candidate plan ------------===//

#include "WholeDAGSchedulePlan.h"

#include "AcceptedCallClosure.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Location.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

using analysis::InstructionExecutionCount;
using analysis::InstructionProgramCost;
using analysis::InstructionProgramWork;
using analysis::ScheduleComputeCost;
using analysis::ScheduleCostKnowledge;
using analysis::ScheduleCostMetric;
using analysis::ScheduleNoCCost;
using analysis::StaticScheduleBranch;
using analysis::StaticSchedulePlan;
using analysis::StaticScheduleResource;
using analysis::StaticScheduleStage;
using analysis::StaticScheduleStep;
using analysis::StaticScheduleWork;
using analysis::WholeCardInstructionProgramCost;

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

unsigned severity(ScheduleCostKnowledge knowledge) {
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

void addMetric(ScheduleCostMetric &target, const ScheduleCostMetric &source) {
  if (!source.isKnown()) {
    if (severity(source.knowledge) > severity(target.knowledge) ||
        (severity(source.knowledge) == severity(target.knowledge) &&
         target.reason == analysis::ScheduleCostReason::None)) {
      target.value = 0;
      target.knowledge = source.knowledge;
      target.reason = source.reason;
    }
    return;
  }
  if (!target.isKnown())
    return;
  if (source.value > std::numeric_limits<uint64_t>::max() - target.value) {
    target.value = 0;
    target.knowledge = ScheduleCostKnowledge::Overflow;
    target.reason = analysis::ScheduleCostReason::ArithmeticOverflow;
    return;
  }
  target.value += source.value;
}

void maximizeMetric(ScheduleCostMetric &target,
                    const ScheduleCostMetric &source) {
  if (!source.isKnown()) {
    if (severity(source.knowledge) > severity(target.knowledge) ||
        (severity(source.knowledge) == severity(target.knowledge) &&
         target.reason == analysis::ScheduleCostReason::None)) {
      target.value = 0;
      target.knowledge = source.knowledge;
      target.reason = source.reason;
    }
    return;
  }
  if (target.isKnown())
    target.value = std::max(target.value, source.value);
}

bool equalMetric(const ScheduleCostMetric &lhs, const ScheduleCostMetric &rhs) {
  return lhs.value == rhs.value && lhs.knowledge == rhs.knowledge &&
         lhs.reason == rhs.reason;
}

using WorkMember = InstructionExecutionCount InstructionProgramWork::*;
constexpr WorkMember kWorkMembers[] = {
    &InstructionProgramWork::instructions,
    &InstructionProgramWork::asynchronousEvents,
    &InstructionProgramWork::rdmaIssues,
    &InstructionProgramWork::wdmaIssues,
    &InstructionProgramWork::tdmaIssues,
    &InstructionProgramWork::ctIssues,
    &InstructionProgramWork::neIssues,
    &InstructionProgramWork::dteOperations,
    &InstructionProgramWork::gatherScatterOperations,
    &InstructionProgramWork::dteSendOperations,
    &InstructionProgramWork::dteReceiveOperations,
    &InstructionProgramWork::dteWaitOperations,
    &InstructionProgramWork::nccJoins,
    &InstructionProgramWork::steadyStateNCCJoins,
    &InstructionProgramWork::nonTerminalNCCJoins,
    &InstructionProgramWork::nccParticipantWaits,
    &InstructionProgramWork::steadyStateNCCParticipantWaits,
    &InstructionProgramWork::nonTerminalNCCParticipantWaits,
    &InstructionProgramWork::intrinsicNCCDrains,
};

void addExecutionCount(InstructionExecutionCount &target,
                       const InstructionExecutionCount &source) {
  addMetric(target.staticSites, source.staticSites);
  addMetric(target.exactExecutions, source.exactExecutions);
  addMetric(target.lowerBound, source.lowerBound);
  addMetric(target.upperBound, source.upperBound);
}

bool equalExecutionCount(const InstructionExecutionCount &lhs,
                         const InstructionExecutionCount &rhs) {
  return equalMetric(lhs.staticSites, rhs.staticSites) &&
         equalMetric(lhs.exactExecutions, rhs.exactExecutions) &&
         equalMetric(lhs.lowerBound, rhs.lowerBound) &&
         equalMetric(lhs.upperBound, rhs.upperBound);
}

void addWork(InstructionProgramWork &target,
             const InstructionProgramWork &source) {
  for (WorkMember member : kWorkMembers)
    addExecutionCount(target.*member, source.*member);
}

bool equalWork(const InstructionProgramWork &lhs,
               const InstructionProgramWork &rhs) {
  return llvm::all_of(kWorkMembers, [&](WorkMember member) {
    return equalExecutionCount(lhs.*member, rhs.*member);
  });
}

void addCompute(ScheduleComputeCost &target,
                const ScheduleComputeCost &source) {
  addMetric(target.npuF16Bf16LogicalOps, source.npuF16Bf16LogicalOps);
  addMetric(target.npuOtherLogicalOps, source.npuOtherLogicalOps);
  addMetric(target.vectorF16Bf16LogicalOps, source.vectorF16Bf16LogicalOps);
  addMetric(target.vectorF32LogicalOps, source.vectorF32LogicalOps);
  addMetric(target.vectorOtherLogicalOps, source.vectorOtherLogicalOps);
}

bool equalCompute(const ScheduleComputeCost &lhs,
                  const ScheduleComputeCost &rhs) {
  return equalMetric(lhs.npuF16Bf16LogicalOps, rhs.npuF16Bf16LogicalOps) &&
         equalMetric(lhs.npuOtherLogicalOps, rhs.npuOtherLogicalOps) &&
         equalMetric(lhs.vectorF16Bf16LogicalOps,
                     rhs.vectorF16Bf16LogicalOps) &&
         equalMetric(lhs.vectorF32LogicalOps, rhs.vectorF32LogicalOps) &&
         equalMetric(lhs.vectorOtherLogicalOps, rhs.vectorOtherLogicalOps);
}

void addNoC(ScheduleNoCCost &target, const ScheduleNoCCost &source) {
  addMetric(target.staticIssueSiteCount, source.staticIssueSiteCount);
  addMetric(target.aggregateTransmitBytes, source.aggregateTransmitBytes);
  addMetric(target.aggregateReceiveBytes, source.aggregateReceiveBytes);
  addMetric(target.transmitMessageCount, source.transmitMessageCount);
  addMetric(target.receiveMessageCount, source.receiveMessageCount);
  addMetric(target.waitOperationCount, source.waitOperationCount);
  addMetric(target.waitedEventCount, source.waitedEventCount);
  for (auto [result, value] : llvm::zip_equal(target.directionalTransmitBytes,
                                              source.directionalTransmitBytes))
    addMetric(result, value);
}

bool equalNoC(const ScheduleNoCCost &lhs, const ScheduleNoCCost &rhs) {
  if (!equalMetric(lhs.staticIssueSiteCount, rhs.staticIssueSiteCount) ||
      !equalMetric(lhs.aggregateTransmitBytes, rhs.aggregateTransmitBytes) ||
      !equalMetric(lhs.aggregateReceiveBytes, rhs.aggregateReceiveBytes) ||
      !equalMetric(lhs.transmitMessageCount, rhs.transmitMessageCount) ||
      !equalMetric(lhs.receiveMessageCount, rhs.receiveMessageCount) ||
      !equalMetric(lhs.waitOperationCount, rhs.waitOperationCount) ||
      !equalMetric(lhs.waitedEventCount, rhs.waitedEventCount))
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs.directionalTransmitBytes,
                                            rhs.directionalTransmitBytes))
    if (!equalMetric(left, right))
      return false;
  return true;
}

void addTileCost(InstructionProgramCost &target,
                 const InstructionProgramCost &source) {
  addWork(target.work, source.work);
  addCompute(target.compute, source.compute);
  addMetric(target.ddrReadBytes, source.ddrReadBytes);
  addMetric(target.ddrWriteBytes, source.ddrWriteBytes);
  addMetric(target.spmMovementBytes, source.spmMovementBytes);
  addMetric(target.gatherScatterBytes, source.gatherScatterBytes);
  addNoC(target.noc, source.noc);
  addMetric(target.instructionCount, source.instructionCount);
  addMetric(target.eventCount, source.eventCount);
  addMetric(target.nccJoinCount, source.nccJoinCount);
  addMetric(target.steadyStateNCCJoinCount, source.steadyStateNCCJoinCount);
  addMetric(target.nonTerminalNCCJoinCount, source.nonTerminalNCCJoinCount);
  addMetric(target.nccParticipantWaitCount, source.nccParticipantWaitCount);
  addMetric(target.steadyStateNCCParticipantWaitCount,
            source.steadyStateNCCParticipantWaitCount);
  addMetric(target.nonTerminalNCCParticipantWaitCount,
            source.nonTerminalNCCParticipantWaitCount);
  addMetric(target.intrinsicNCCDrainCount, source.intrinsicNCCDrainCount);
  maximizeMetric(target.spmHighWaterBytes, source.spmHighWaterBytes);
  maximizeMetric(target.ddrHighWaterBytes, source.ddrHighWaterBytes);
  addMetric(target.compilerOwnedSPMBufferCount,
            source.compilerOwnedSPMBufferCount);
  addMetric(target.compilerOwnedDDRBufferCount,
            source.compilerOwnedDDRBufferCount);
}

bool equalTileCost(const InstructionProgramCost &lhs,
                   const InstructionProgramCost &rhs) {
  return equalWork(lhs.work, rhs.work) &&
         equalCompute(lhs.compute, rhs.compute) &&
         equalMetric(lhs.ddrReadBytes, rhs.ddrReadBytes) &&
         equalMetric(lhs.ddrWriteBytes, rhs.ddrWriteBytes) &&
         equalMetric(lhs.spmMovementBytes, rhs.spmMovementBytes) &&
         equalMetric(lhs.gatherScatterBytes, rhs.gatherScatterBytes) &&
         equalNoC(lhs.noc, rhs.noc) &&
         equalMetric(lhs.instructionCount, rhs.instructionCount) &&
         equalMetric(lhs.eventCount, rhs.eventCount) &&
         equalMetric(lhs.nccJoinCount, rhs.nccJoinCount) &&
         equalMetric(lhs.steadyStateNCCJoinCount,
                     rhs.steadyStateNCCJoinCount) &&
         equalMetric(lhs.nonTerminalNCCJoinCount,
                     rhs.nonTerminalNCCJoinCount) &&
         equalMetric(lhs.nccParticipantWaitCount,
                     rhs.nccParticipantWaitCount) &&
         equalMetric(lhs.steadyStateNCCParticipantWaitCount,
                     rhs.steadyStateNCCParticipantWaitCount) &&
         equalMetric(lhs.nonTerminalNCCParticipantWaitCount,
                     rhs.nonTerminalNCCParticipantWaitCount) &&
         equalMetric(lhs.intrinsicNCCDrainCount, rhs.intrinsicNCCDrainCount) &&
         equalMetric(lhs.spmHighWaterBytes, rhs.spmHighWaterBytes) &&
         equalMetric(lhs.ddrHighWaterBytes, rhs.ddrHighWaterBytes) &&
         equalMetric(lhs.compilerOwnedSPMBufferCount,
                     rhs.compilerOwnedSPMBufferCount) &&
         equalMetric(lhs.compilerOwnedDDRBufferCount,
                     rhs.compilerOwnedDDRBufferCount);
}

bool conservesRawCost(const WholeCardInstructionProgramCost &whole,
                      llvm::ArrayRef<WholeCardInstructionProgramCost> slices) {
  if (whole.tileCosts.empty() || llvm::any_of(slices, [&](const auto &slice) {
        return slice.tileCosts.size() != whole.tileCosts.size();
      }))
    return false;
  for (size_t tile = 0; tile < whole.tileCosts.size(); ++tile) {
    InstructionProgramCost sum;
    for (const WholeCardInstructionProgramCost &slice : slices)
      addTileCost(sum, slice.tileCosts[tile]);
    if (!equalTileCost(sum, whole.tileCosts[tile]))
      return false;
  }
  ScheduleCostMetric linkBytes;
  ScheduleCostMetric linkMessages;
  for (const WholeCardInstructionProgramCost &slice : slices) {
    addMetric(linkBytes, slice.minimumHopLinkByteDemand);
    addMetric(linkMessages, slice.minimumHopMessageDemand);
  }
  return equalMetric(linkBytes, whole.minimumHopLinkByteDemand) &&
         equalMetric(linkMessages, whole.minimumHopMessageDemand);
}

bool tileSetsIntersect(llvm::ArrayRef<PhysicalTileId> lhs,
                       llvm::ArrayRef<PhysicalTileId> rhs) {
  return llvm::any_of(
      lhs, [&](PhysicalTileId tile) { return llvm::is_contained(rhs, tile); });
}

std::vector<llvm::BitVector>
computeDAGReachability(const CardDAGAnalysis &dag) {
  const size_t nodeCount = dag.getNodes().size();
  std::vector<llvm::BitVector> reachable(nodeCount, llvm::BitVector(nodeCount));
  for (const CardDAGEdge &edge : dag.getEdges())
    reachable[edge.producer].set(edge.consumer);

  // Compute the finite transitive closure without relying on the incidental
  // direct-operation order used to number the current DAG nodes.
  for (size_t intermediate = 0; intermediate < nodeCount; ++intermediate)
    for (size_t source = 0; source < nodeCount; ++source)
      if (reachable[source].test(intermediate))
        reachable[source] |= reachable[intermediate];
  return reachable;
}

llvm::BitVector getObservableDAGNodes(const CardDAGAnalysis &dag) {
  llvm::BitVector observable(dag.getNodes().size());
  llvm::SmallVector<CardDAGNodeID, 16> worklist;
  for (llvm::ArrayRef<CardDAGNodeID> roots : dag.getObservableOutputRootNodes())
    worklist.append(roots.begin(), roots.end());

  while (!worklist.empty()) {
    CardDAGNodeID node = worklist.pop_back_val();
    if (observable.test(node))
      continue;
    observable.set(node);
    for (CardDAGEdgeID edgeId : dag.getNodes()[node].incomingEdges)
      worklist.push_back(dag.getEdges()[edgeId].producer);
  }
  return observable;
}

/// A source node may disappear from accepted Instr IR when its pure tensor
/// work is eliminated or absorbed by a surviving downstream operation. Such a
/// node remains schedule-covered only when every observable successor path is
/// already covered. Observable terminal nodes therefore still require actual
/// accepted-IR buffer use, while source-only dead nodes do not become artificial
/// schedule obligations.
bool hasCompleteObservableNodeCoverage(const CardDAGAnalysis &dag,
                                       const llvm::BitVector &observed) {
  llvm::BitVector observable = getObservableDAGNodes(dag);
  llvm::BitVector covered = observed;
  covered.resize(observable.size());

  bool changed = true;
  while (changed) {
    changed = false;
    for (const CardDAGNode &node : dag.getNodes()) {
      if (!observable.test(node.id) || covered.test(node.id))
        continue;
      bool hasObservableSuccessor = false;
      bool allObservableSuccessorsCovered = true;
      for (CardDAGEdgeID edgeId : node.outgoingEdges) {
        CardDAGNodeID successor = dag.getEdges()[edgeId].consumer;
        if (!observable.test(successor))
          continue;
        hasObservableSuccessor = true;
        allObservableSuccessorsCovered &= covered.test(successor);
      }
      if (hasObservableSuccessor && allObservableSuccessorsCovered) {
        covered.set(node.id);
        changed = true;
      }
    }
  }

  observable.reset(covered);
  return observable.none();
}

bool belongsToResidualPhase(mlir::Operation *operation) {
  // Card-shared DDR and NoC service are modeled from the complete accepted
  // cost exactly once. Their actual issue/wait operations, plus explicit
  // completion joins, remain in the residual operation partition rather than
  // being mistaken for Tile-private node lanes.
  return mlir::isa<InstrRDMAOp, InstrWDMAOp, InstrDTESendOp, InstrDTERecvOp,
                   InstrDTEWaitOp, SyncNCCJoinOp>(operation);
}

StaticScheduleStage
oneWholeCardResourceStage(const WholeCardInstructionProgramCost &cost,
                          StaticScheduleResource resource) {
  StaticScheduleBranch branch;
  branch.dependentWork.push_back(
      {&cost, analysis::staticScheduleResourceMask(resource)});
  StaticScheduleStage stage;
  stage.independentBranches.push_back(std::move(branch));
  return stage;
}

} // namespace

mlir::FailureOr<StaticSchedulePlan> buildAcceptedWholeDAGSchedulePlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    llvm::ArrayRef<AcceptedOperationNodeRelation> operationNodeRelations,
    AcceptedWholeCardExecutable &executable,
    llvm::SmallVectorImpl<WholeCardInstructionProgramCost> &phaseCosts,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (nodePlacements.size() != dag.getNodes().size() ||
      executable.tiles.empty()) {
    setFailure(failureReason,
               "accepted schedule plan lacks a complete DAG/Tile domain");
    return mlir::failure();
  }

  llvm::DenseMap<mlir::Operation *, llvm::BitVector> operationNodes;
  for (const AcceptedOperationNodeRelation &relation : operationNodeRelations) {
    if (!relation.operation || relation.structuredNodeId >= dag.getNodes().size()) {
      setFailure(failureReason,
                 "accepted operation-node relation is null or out of range");
      return mlir::failure();
    }
    auto [entry, inserted] = operationNodes.try_emplace(
        relation.operation, llvm::BitVector(dag.getNodes().size()));
    entry->second.set(relation.structuredNodeId);
  }

  llvm::SmallVector<const WholeDAGNodePlacement *, 16> placements(
      dag.getNodes().size(), nullptr);
  for (const WholeDAGNodePlacement &placement : nodePlacements) {
    if (placement.node >= placements.size() || placements[placement.node] ||
        placement.tiles.empty()) {
      setFailure(failureReason,
                 "accepted schedule plan has invalid node placement");
      return mlir::failure();
    }
    placements[placement.node] = &placement;
  }
  if (llvm::is_contained(placements, nullptr)) {
    setFailure(failureReason,
               "accepted schedule plan does not place every DAG node");
    return mlir::failure();
  }

  const size_t phaseCount = dag.getNodes().size() + 1;
  std::vector<std::vector<llvm::SmallVector<mlir::Operation *, 32>>> operations(
      phaseCount);
  for (auto &phase : operations)
    phase.resize(executable.tiles.size());

  llvm::SmallVector<mlir::func::FuncOp, 16> entries;
  entries.reserve(executable.tiles.size());
  llvm::BitVector observedNodes(dag.getNodes().size());
  const std::vector<llvm::BitVector> reachable = computeDAGReachability(dag);
  for (auto [tileIndex, tile] : llvm::enumerate(executable.tiles)) {
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(tile.getModule(), tile.getEntrySymbol());
    if (!closure) {
      llvm::consumeError(closure.takeError());
      setFailure(failureReason,
                 "accepted schedule plan cannot recover the Tile call closure");
      return mlir::failure();
    }
    entries.push_back(closure->entry);
    for (mlir::func::FuncOp function : closure->functions) {
      std::string walkFailure;
      mlir::WalkResult walk = function.walk([&](mlir::Operation *operation) {
        llvm::BitVector operationNodeSet(dag.getNodes().size());
        if (auto found = operationNodes.find(operation);
            found != operationNodes.end())
          operationNodeSet = found->second;
        observedNodes |= operationNodeSet;

        const bool residual = belongsToResidualPhase(operation);
        std::optional<CardDAGNodeID> node;
        if (!operationNodeSet.none() && !residual) {
          // Fusion may touch buffers from several source nodes. Attribute its
          // exact work once to the unique node downstream of every other
          // retained node. Incomparable sets without a retained sink remain a
          // hard error.
          unsigned eligibleOwnerCount = 0;
          for (int relationNode = operationNodeSet.find_first();
               relationNode >= 0;
               relationNode = operationNodeSet.find_next(relationNode)) {
            CardDAGNodeID candidate = static_cast<CardDAGNodeID>(relationNode);
            bool downstreamOfAll = true;
            for (int other = operationNodeSet.find_first(); other >= 0;
                 other = operationNodeSet.find_next(other)) {
              if (other == relationNode)
                continue;
              downstreamOfAll &=
                  reachable[static_cast<size_t>(other)].test(candidate);
            }
            if (!downstreamOfAll)
              continue;
            ++eligibleOwnerCount;
            node = candidate;
          }
          if (eligibleOwnerCount != 1) {
            llvm::raw_string_ostream message(walkFailure);
            message << "accepted operation has no unique downstream DAG "
                       "node"
                    << " tile_id=" << tile.getPhysicalTileId().getValue()
                    << " op=" << operation->getName().getStringRef()
                    << " eligible_owners=" << eligibleOwnerCount
                    << " relation_nodes=[";
            bool first = true;
            for (int relationNode = operationNodeSet.find_first();
                 relationNode >= 0;
                 relationNode = operationNodeSet.find_next(relationNode)) {
              if (!first)
                message << ',';
              first = false;
              message << relationNode;
            }
            message << ']';
            return mlir::WalkResult::interrupt();
          }
        }
        if (node && !llvm::is_contained(placements[*node]->tiles,
                                        tile.getPhysicalTileId())) {
          llvm::raw_string_ostream message(walkFailure);
          message << "accepted operation is outside its DAG node placement"
                  << " tile_id=" << tile.getPhysicalTileId().getValue()
                  << " op=" << operation->getName().getStringRef()
                  << " structured_node=" << *node
                  << " source_op="
                  << dag.getNodes()[*node].operation->getName().getStringRef()
                  << " placement_tiles=[";
          for (auto [index, physicalTile] :
               llvm::enumerate(placements[*node]->tiles)) {
            if (index != 0)
              message << ',';
            message << physicalTile.getValue();
          }
          message << "] result_types=[";
          for (auto [index, type] :
               llvm::enumerate(operation->getResultTypes())) {
            if (index != 0)
              message << ',';
            type.print(message);
          }
          message << "] users=[";
          bool firstUser = true;
          for (mlir::Operation *user : operation->getUsers()) {
            if (!firstUser)
              message << ',';
            firstUser = false;
            message << user->getName().getStringRef();
          }
          message << ']';
          return mlir::WalkResult::interrupt();
        }
        const size_t phase = node ? static_cast<size_t>(*node) + 1 : 0;
        operations[phase][tileIndex].push_back(operation);
        return mlir::WalkResult::advance();
      });
      if (walk.wasInterrupted()) {
        setFailure(failureReason,
                   walkFailure.empty()
                       ? "accepted operation has an ambiguous DAG node relation"
                       : llvm::StringRef(walkFailure));
        return mlir::failure();
      }
    }
  }
  if (!hasCompleteObservableNodeCoverage(dag, observedNodes)) {
    setFailure(failureReason,
               "accepted lowering lost an observable terminal DAG node");
    return mlir::failure();
  }

  phaseCosts.clear();
  phaseCosts.reserve(phaseCount);
  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  for (size_t phase = 0; phase < phaseCount; ++phase) {
    llvm::SmallVector<analysis::PhysicalTileInstructionProgramSlice, 16>
        tileSlices;
    tileSlices.reserve(executable.tiles.size());
    for (auto [tileIndex, tile] : llvm::enumerate(executable.tiles))
      tileSlices.push_back({tile.getPhysicalTileId(),
                            entries[tileIndex].getOperation(),
                            operations[phase][tileIndex]});
    phaseCosts.push_back(analysis::analyzeWholeCardInstructionProgramCostSlice(
        tileSlices, policy));
  }
  if (!conservesRawCost(executable.resourceCost, phaseCosts)) {
    setFailure(failureReason,
               "accepted DAG cost slices do not conserve final raw cost");
    phaseCosts.clear();
    return mlir::failure();
  }

  llvm::SmallVector<StaticScheduleStep, 16> steps;
  steps.push_back(StaticScheduleStep::forStage(oneWholeCardResourceStage(
      executable.resourceCost, StaticScheduleResource::DDR)));
  steps.push_back(StaticScheduleStep::forStage(oneWholeCardResourceStage(
      executable.resourceCost, StaticScheduleResource::NoC)));

  const auto localResources =
      analysis::staticScheduleResourceMask(StaticScheduleResource::Compute) |
      analysis::staticScheduleResourceMask(StaticScheduleResource::SPMMovement);
  StaticScheduleBranch residualBranch;
  residualBranch.dependentWork.push_back({&phaseCosts.front(), localResources});
  StaticScheduleStage residualStage;
  residualStage.independentBranches.push_back(std::move(residualBranch));
  steps.push_back(StaticScheduleStep::forStage(std::move(residualStage)));

  llvm::SmallVector<uint32_t, 16> indegree(dag.getNodes().size(), 0);
  llvm::SmallVector<llvm::SmallVector<CardDAGNodeID, 4>, 16> successors(
      dag.getNodes().size());
  for (const CardDAGEdge &edge : dag.getEdges()) {
    ++indegree[edge.consumer];
    successors[edge.producer].push_back(edge.consumer);
  }
  llvm::BitVector scheduled(dag.getNodes().size());
  while (!scheduled.all()) {
    llvm::SmallVector<CardDAGNodeID, 8> ready;
    for (const CardDAGNode &node : dag.getNodes())
      if (!scheduled.test(node.id) && indegree[node.id] == 0)
        ready.push_back(node.id);
    if (ready.empty()) {
      setFailure(failureReason,
                 "accepted schedule plan encountered a cyclic DAG");
      phaseCosts.clear();
      return mlir::failure();
    }

    llvm::SmallVector<CardDAGNodeID, 8> dispatched;
    for (CardDAGNodeID node : ready) {
      const WholeDAGNodePlacement &placement = *placements[node];
      if (llvm::any_of(dispatched, [&](CardDAGNodeID other) {
            return tileSetsIntersect(placement.tiles, placements[other]->tiles);
          }))
        continue;
      dispatched.push_back(node);
    }
    StaticScheduleStage stage;
    for (CardDAGNodeID node : dispatched) {
      StaticScheduleBranch branch;
      branch.dependentWork.push_back(
          {&phaseCosts[static_cast<size_t>(node) + 1], localResources});
      stage.independentBranches.push_back(std::move(branch));
    }
    steps.push_back(StaticScheduleStep::forStage(std::move(stage)));
    for (CardDAGNodeID node : dispatched) {
      scheduled.set(node);
      for (CardDAGNodeID successor : successors[node]) {
        if (indegree[successor] == 0) {
          setFailure(failureReason,
                     "accepted schedule DAG indegree is inconsistent");
          phaseCosts.clear();
          return mlir::failure();
        }
        --indegree[successor];
      }
    }
  }

  std::optional<StaticSchedulePlan> plan =
      StaticSchedulePlan::create(executable.resourceCost, steps);
  if (!plan) {
    setFailure(failureReason,
               "accepted schedule plan failed structural validation");
    phaseCosts.clear();
    return mlir::failure();
  }
  return std::move(*plan);
}

} // namespace wafer::compiler::detail
