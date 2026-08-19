//===- SelectedBufferMaterialization.cpp - Joint buffer actualization --===//

#include "Wafer/Transforms/Bufferization/SelectedBufferMaterialization.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct SelectedBufferMaterialization {
  StructuredMaterializationRelations materializationRelations;
  unsigned stageCount = 0;
  unsigned slotAllocationCount = 0;
  unsigned maximumSlotCount = 0;
};

struct DependencyEdge {
  unsigned predecessor = 0;
  bool advancesStage = false;
};

struct BufferAccess {
  mlir::Value root;
  bool write = false;
  unsigned issueOperation = 0;
  unsigned completionOperation = 0;
};

struct AllocationPlan {
  mlir::memref::AllocOp allocation;
  unsigned firstStage = 0;
  unsigned lastStage = 0;
  unsigned slotCount = 0;
};

struct SelectedBufferPlan {
  llvm::SmallVector<mlir::Operation *, 16> operations;
  llvm::DenseMap<mlir::Operation *, unsigned> operationIndices;
  llvm::DenseMap<mlir::Operation *, InstrFamily> instructionFamilies;
  llvm::DenseMap<mlir::Operation *, NCCWorker> instructionWorkers;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> nccIssueJoins;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      explicitJoinProducers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      backedgeJoinProducers;
  llvm::DenseSet<mlir::Operation *> removableBackedgeJoins;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> directDTEWaits;
  llvm::SmallVector<llvm::SmallVector<DependencyEdge, 4>, 16> predecessors;
  llvm::DenseMap<mlir::Operation *, unsigned> stages;
  llvm::SmallVector<AllocationPlan, 4> allocations;
  llvm::SmallVector<mlir::memref::DeallocOp, 4> deallocations;
  uint64_t tripCount = 0;
  unsigned maxStage = 0;
  unsigned slotAllocationCount = 0;
  unsigned maximumSlotCount = 0;
};

static constexpr llvm::StringLiteral kPointerPermutationMarker =
    "wafer.selected_buffer_pointer_permutation";

static mlir::FailureOr<SelectedBufferPlan> failPlan(std::string *failureReason,
                                                    llvm::Twine message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static mlir::FailureOr<SelectedBufferPlan>
failPlan(std::string *failureReason,
         SelectedBufferMaterializationFailureKind *failureKind,
         SelectedBufferMaterializationFailureKind kind, llvm::Twine message) {
  if (failureKind)
    *failureKind = kind;
  return failPlan(failureReason, message);
}

static mlir::FailureOr<SelectedBufferMaterialization>
failMaterialization(std::string *failureReason, llvm::Twine message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static bool matchesMessage(mlir::Operation *operation,
                           const SelectedBufferMessage &expected) {
  DTEMessageAttr actual;
  if (expected.direction == SelectedBufferMessageDirection::Send) {
    auto send = mlir::dyn_cast<InstrDTESendOp>(operation);
    if (!send)
      return false;
    actual = send.getMessageAttr();
  } else {
    auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation);
    if (!receive)
      return false;
    actual = receive.getMessageAttr();
  }
  return actual && actual.getCommunicationId() == expected.communicationId &&
         actual.getPayloadSlice() == expected.payloadSlice;
}

static bool loopMayContainRequest(
    mlir::scf::ForOp loop, const SelectedBufferRequest &request,
    const StructuredMaterializationRelations &materializationRelations) {
  bool producer = false;
  bool consumer = false;
  llvm::SmallVector<bool, 2> messages(request.messages.size(), false);
  loop.walk([&](mlir::Operation *operation) {
    producer |= request.producerNode &&
                operationUsesStructuredNode(operation, *request.producerNode,
                                            materializationRelations);
    consumer |= request.consumerNode &&
                operationUsesStructuredNode(operation, *request.consumerNode,
                                            materializationRelations);
    for (auto [index, message] : llvm::enumerate(request.messages))
      messages[index] = messages[index] || matchesMessage(operation, message);
  });
  if (request.requireLocalDataflow && (!producer || !consumer))
    return false;
  for (auto [index, message] : llvm::enumerate(request.messages)) {
    if (!messages[index])
      return false;
    if (message.direction == SelectedBufferMessageDirection::Send && !producer)
      return false;
    if (message.direction == SelectedBufferMessageDirection::Receive &&
        !consumer)
      return false;
  }
  return request.requireLocalDataflow || !request.messages.empty();
}

static std::optional<uint64_t>
getPositiveStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
    return std::nullopt;
  __int128 span = static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  // The pinned SCF pipeliner computes `upper - lower` in int64_t before
  // divideCeilSigned. Keep its precondition explicit so an extreme but valid
  // index interval cannot trigger signed overflow inside the utility.
  if (span > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  __int128 count =
      (span + static_cast<__int128>(*step) - 1) / static_cast<__int128>(*step);
  if (count <= 0 ||
      count > static_cast<__int128>(std::numeric_limits<uint64_t>::max()))
    return std::nullopt;
  return static_cast<uint64_t>(count);
}

static bool hasPhysicalPlacementFacts(mlir::ModuleOp module) {
  bool placed = false;
  module.walk([&](mlir::memref::AllocOp allocation) {
    placed |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
              allocation->hasAttr(kWaferDDROffsetAttrName);
  });
  return placed;
}

static mlir::FailureOr<mlir::Value> getStaticAliasRoot(mlir::Value value,
                                                       mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto type = mlir::dyn_cast<mlir::BaseMemRefType>(value.getType());
    if (!type)
      return mlir::failure();
    if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
            value.getDefiningOp())) {
      value = view.getViewSource();
      continue;
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
      if (mlir::Value entry =
              analysis::getSingleExecutionRegionEntryOperand(blockArg)) {
        value = entry;
        continue;
      }
      if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (owner == forOp.getBody() && blockArg.getArgNumber() > 0) {
          unsigned index = blockArg.getArgNumber() - 1;
          auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
              forOp.getBody()->getTerminator());
          if (index < forOp.getInitArgs().size() && yield &&
              index < yield.getResults().size() &&
              yield.getResults()[index] == blockArg) {
            value = forOp.getInitArgs()[index];
            continue;
          }
        }
      }
      return value;
    }
    if (value.getDefiningOp<mlir::memref::AllocOp>())
      return value;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !loop->isAncestor(definition))
      return value;
    return mlir::failure();
  }
  return mlir::failure();
}

static bool isFunctionEntryArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return false;
  auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front();
}

static bool isFunctionEntryBackedMemref(mlir::Value value) {
  if (isFunctionEntryArgument(value))
    return true;
  auto toMemref = value.getDefiningOp<mlir::bufferization::ToMemrefOp>();
  return toMemref && isFunctionEntryArgument(toMemref.getTensor());
}

static bool areProvenDistinct(mlir::Value lhs, mlir::Value rhs,
                              mlir::scf::ForOp loop) {
  if (lhs == rhs)
    return false;
  auto lhsAllocation = lhs.getDefiningOp<mlir::memref::AllocOp>();
  auto rhsAllocation = rhs.getDefiningOp<mlir::memref::AllocOp>();
  // Two distinct allocation roots are disjoint. Do not generalize one fresh
  // allocation against an arbitrary SSA producer: an unrecognized producer
  // may itself be an alias of that allocation.
  if (lhsAllocation && rhsAllocation)
    return true;
  // An allocation dynamically created in the selected loop cannot alias a
  // value available before that allocation in the same iteration. Likewise a
  // function argument predates every memref.alloc in the function. Unknown
  // SSA producers remain unproven because they may return an alias.
  if ((lhsAllocation && loop->isAncestor(lhsAllocation.getOperation())) ||
      (rhsAllocation && loop->isAncestor(rhsAllocation.getOperation())) ||
      (lhsAllocation && isFunctionEntryBackedMemref(rhs)) ||
      (rhsAllocation && isFunctionEntryBackedMemref(lhs)))
    return true;

  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  if (!lhsType || !rhsType)
    return false;
  MemoryAttr lhsMemory = getWaferMemoryAttr(lhsType);
  MemoryAttr rhsMemory = getWaferMemoryAttr(rhsType);
  return lhsMemory && rhsMemory && lhsMemory.getSpace() != rhsMemory.getSpace();
}

static bool isSupportedPureBodyOperation(mlir::Operation *operation) {
  return operation->getNumRegions() == 0 && mlir::isMemoryEffectFree(operation);
}

static bool isStaticLoopAllocation(mlir::Operation *operation) {
  auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation);
  return allocation && allocation.getDynamicSizes().empty() &&
         allocation.getSymbolOperands().empty() &&
         allocation.getType().hasStaticShape();
}

static void addDependency(SelectedBufferPlan &plan, unsigned predecessor,
                          unsigned successor, bool advancesStage) {
  if (predecessor == successor)
    return;
  auto &edges = plan.predecessors[successor];
  auto found = llvm::find_if(edges, [&](const DependencyEdge &edge) {
    return edge.predecessor == predecessor;
  });
  if (found == edges.end()) {
    edges.push_back({predecessor, advancesStage});
    return;
  }
  found->advancesStage |= advancesStage;
}

static bool isCrossEngineDependency(const SelectedBufferPlan &plan,
                                    unsigned predecessor, unsigned successor) {
  auto predecessorFamily =
      plan.instructionFamilies.find(plan.operations[predecessor]);
  auto successorFamily =
      plan.instructionFamilies.find(plan.operations[successor]);
  return predecessorFamily != plan.instructionFamilies.end() &&
         successorFamily != plan.instructionFamilies.end() &&
         predecessorFamily->second != successorFamily->second;
}

static bool isDirectDTEIssue(mlir::Operation *operation) {
  return mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation);
}

static bool dependencyAdvancesStage(const SelectedBufferPlan &plan,
                                    unsigned predecessor, unsigned successor) {
  mlir::Operation *predecessorOperation = plan.operations[predecessor];
  mlir::Operation *successorOperation = plan.operations[successor];
  auto wait = plan.directDTEWaits.find(predecessorOperation);
  if (wait != plan.directDTEWaits.end() && wait->second == successorOperation)
    return false;
  if (mlir::isa<SyncNCCJoinOp>(predecessorOperation) &&
      isDirectDTEIssue(successorOperation))
    return true;
  // Keep issue and exact wait in one stage so the post-memory Direct DTE
  // acceptance contract continues to see one direct same-block SSA use.
  // An explicit NCC participant join is the distinct producer-to-DTE
  // visibility boundary. A following NCC consumer/reuser still advances
  // through the ordinary cross-engine rule and runs as the preceding tile in
  // the steady kernel.
  return isCrossEngineDependency(plan, predecessor, successor);
}

static bool dependencyPathExists(const SelectedBufferPlan &plan,
                                 unsigned predecessor, unsigned successor) {
  if (predecessor == successor || predecessor >= plan.operations.size() ||
      successor >= plan.operations.size())
    return false;
  llvm::BitVector visited(plan.operations.size());
  llvm::SmallVector<unsigned, 16> worklist{successor};
  while (!worklist.empty()) {
    unsigned current = worklist.pop_back_val();
    if (current >= visited.size() || visited.test(current))
      continue;
    visited.set(current);
    for (const DependencyEdge &edge : plan.predecessors[current]) {
      if (edge.predecessor == predecessor)
        return true;
      worklist.push_back(edge.predecessor);
    }
  }
  return false;
}

static bool stagedDependencyPathExists(const SelectedBufferPlan &plan,
                                       llvm::ArrayRef<unsigned> predecessors,
                                       llvm::ArrayRef<unsigned> successors) {
  for (unsigned predecessor : predecessors) {
    unsigned predecessorStage =
        plan.stages.lookup(plan.operations[predecessor]);
    for (unsigned successor : successors) {
      unsigned successorStage = plan.stages.lookup(plan.operations[successor]);
      if (successorStage > predecessorStage &&
          dependencyPathExists(plan, predecessor, successor))
        return true;
    }
  }
  return false;
}

static llvm::SmallVector<unsigned, 8> collectNodeInstructionIndices(
    const SelectedBufferPlan &plan, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &materializationRelations) {
  llvm::SmallVector<unsigned, 8> result;
  for (auto [operationIndex, operation] : llvm::enumerate(plan.operations))
    if (plan.instructionFamilies.contains(operation) &&
        operationUsesStructuredNode(operation, structuredNodeId,
                                    materializationRelations))
      result.push_back(static_cast<unsigned>(operationIndex));
  return result;
}

static llvm::SmallVector<unsigned, 2>
collectMessageInstructionIndices(const SelectedBufferPlan &plan,
                                 const SelectedBufferMessage &message) {
  llvm::SmallVector<unsigned, 2> result;
  for (auto [operationIndex, operation] : llvm::enumerate(plan.operations))
    if (matchesMessage(operation, message))
      result.push_back(static_cast<unsigned>(operationIndex));
  return result;
}

static bool
markOneSelectedStageBoundary(SelectedBufferPlan &plan,
                             llvm::ArrayRef<unsigned> sources,
                             llvm::ArrayRef<unsigned> destinations) {
  for (unsigned destination : destinations) {
    for (DependencyEdge &edge : plan.predecessors[destination]) {
      if (llvm::any_of(sources, [&](unsigned source) {
            return source == edge.predecessor ||
                   dependencyPathExists(plan, source, edge.predecessor);
          })) {
        edge.advancesStage = true;
        return true;
      }
    }
  }
  return false;
}

static mlir::LogicalResult markSelectedBufferStageBoundaries(
    SelectedBufferPlan &plan, llvm::ArrayRef<SelectedBufferRequest> requests,
    const StructuredMaterializationRelations &materializationRelations,
    size_t *failedRequest) {
  if (failedRequest)
    *failedRequest = std::numeric_limits<size_t>::max();
  for (auto [requestIndex, request] : llvm::enumerate(requests)) {
    llvm::SmallVector<unsigned, 8> producers = collectNodeInstructionIndices(
        plan, *request.producerNode, materializationRelations);
    llvm::SmallVector<unsigned, 8> consumers = collectNodeInstructionIndices(
        plan, *request.consumerNode, materializationRelations);
    bool witnessed = !request.requireLocalDataflow ||
                     markOneSelectedStageBoundary(plan, producers, consumers);
    for (const SelectedBufferMessage &message : request.messages) {
      llvm::SmallVector<unsigned, 2> endpoints =
          collectMessageInstructionIndices(plan, message);
      witnessed &=
          message.direction == SelectedBufferMessageDirection::Send
              ? markOneSelectedStageBoundary(plan, producers, endpoints)
              : markOneSelectedStageBoundary(plan, endpoints, consumers);
    }
    if (!witnessed) {
      if (failedRequest)
        *failedRequest = requestIndex;
      return mlir::failure();
    }
  }
  return mlir::success();
}

static mlir::LogicalResult validateSelectedBufferRequests(
    const SelectedBufferPlan &plan,
    llvm::ArrayRef<SelectedBufferRequest> requests,
    const StructuredMaterializationRelations &materializationRelations,
    size_t *failedRequest) {
  if (failedRequest)
    *failedRequest = std::numeric_limits<size_t>::max();
  for (auto [requestIndex, request] : llvm::enumerate(requests)) {
    llvm::SmallVector<unsigned, 8> producers = collectNodeInstructionIndices(
        plan, *request.producerNode, materializationRelations);
    llvm::SmallVector<unsigned, 8> consumers = collectNodeInstructionIndices(
        plan, *request.consumerNode, materializationRelations);

    bool witnessed = !request.requireLocalDataflow ||
                     stagedDependencyPathExists(plan, producers, consumers);
    for (const SelectedBufferMessage &message : request.messages) {
      llvm::SmallVector<unsigned, 2> endpoints =
          collectMessageInstructionIndices(plan, message);
      witnessed &= message.direction == SelectedBufferMessageDirection::Send
                       ? stagedDependencyPathExists(plan, producers, endpoints)
                       : stagedDependencyPathExists(plan, endpoints, consumers);
    }
    if (!witnessed) {
      if (failedRequest)
        *failedRequest = requestIndex;
      return mlir::failure();
    }
  }
  return mlir::success();
}

static mlir::LogicalResult collectDirectDTEWaits(SelectedBufferPlan &plan,
                                                 std::string *failureReason) {
  for (mlir::Operation *operation : plan.operations) {
    auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation);
    if (!wait)
      continue;
    unsigned senderIssues = 0;
    for (mlir::Value token : wait.getTokens()) {
      mlir::Operation *issue = token.getDefiningOp();
      if (!isDirectDTEIssue(issue) || !plan.operationIndices.contains(issue) ||
          issue->getBlock() != operation->getBlock()) {
        if (failureReason)
          *failureReason =
              "selected-buffer Direct DTE wait requires same-loop issue tokens";
        return mlir::failure();
      }
      senderIssues += mlir::isa<InstrDTESendOp>(issue) ? 1U : 0U;
    }
    if (senderIssues > 1) {
      if (failureReason)
        *failureReason = "selected-buffer Direct DTE issue/wait window cannot "
                         "contain multiple "
                         "sender issues";
      return mlir::failure();
    }
  }

  for (mlir::Operation *operation : plan.operations) {
    if (!isDirectDTEIssue(operation))
      continue;
    mlir::Value token = operation->getResult(0);
    if (!token.hasOneUse()) {
      if (failureReason)
        *failureReason =
            "selected-buffer Direct DTE issue requires exactly one wait use";
      return mlir::failure();
    }
    mlir::Operation *wait = token.use_begin()->getOwner();
    auto issueIndex = plan.operationIndices.find(operation);
    auto waitIndex = plan.operationIndices.find(wait);
    if (!mlir::isa<InstrDTEWaitOp>(wait) ||
        wait->getBlock() != operation->getBlock() ||
        waitIndex == plan.operationIndices.end() ||
        issueIndex->second >= waitIndex->second) {
      if (failureReason)
        *failureReason =
            "selected-buffer Direct DTE issue requires one following same-loop "
            "exact wait";
      return mlir::failure();
    }
    plan.directDTEWaits.try_emplace(operation, wait);
  }

  struct DirectDTEWindow {
    mlir::Operation *wait = nullptr;
    unsigned firstIssue = 0;
    unsigned waitIndex = 0;
  };
  llvm::DenseMap<mlir::Operation *, unsigned> firstIssues;
  for (const auto &[issue, wait] : plan.directDTEWaits) {
    unsigned issueIndex = plan.operationIndices.lookup(issue);
    auto [found, inserted] = firstIssues.try_emplace(wait, issueIndex);
    if (!inserted)
      found->second = std::min(found->second, issueIndex);
  }
  llvm::SmallVector<DirectDTEWindow, 4> windows;
  for (const auto &[wait, firstIssue] : firstIssues)
    windows.push_back({wait, firstIssue, plan.operationIndices.lookup(wait)});
  llvm::sort(windows,
             [](const DirectDTEWindow &lhs, const DirectDTEWindow &rhs) {
               return lhs.firstIssue < rhs.firstIssue;
             });
  for (auto pair : llvm::zip(windows, llvm::drop_begin(windows))) {
    const DirectDTEWindow &prior = std::get<0>(pair);
    const DirectDTEWindow &next = std::get<1>(pair);
    if (prior.waitIndex >= next.firstIssue) {
      if (failureReason)
        *failureReason =
            "selected-buffer Direct DTE issue/wait windows must not overlap";
      return mlir::failure();
    }
  }
  return mlir::success();
}

static mlir::LogicalResult
collectInstructionAccesses(mlir::Operation *operation, unsigned operationIndex,
                           unsigned accessEndIndex, mlir::scf::ForOp loop,
                           llvm::SmallVectorImpl<BufferAccess> &accesses,
                           std::string *failureReason) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects) {
    if (failureReason)
      *failureReason =
          "selected-buffer instruction has no typed memory-effect interface";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);

  // Engine occupancy is owned by the typed instruction family/worker. Coarse
  // SPM/DDR resource effects are accepted only when a value-associated effect
  // of the same kind and address space witnesses the actual storage root.
  // Unknown/rootless storage must never disappear from the dependency DAG.
  for (const auto &instance : instances) {
    if (instance.getValue())
      continue;
    mlir::SideEffects::Resource *resource = instance.getResource();
    if (resource == WaferComputeResource::get() ||
        resource == WaferMovementResource::get())
      continue;
    // Direct DTE transport identity and readiness are carried by the async
    // token and its exact wait. Its rootless communication resource is not a
    // second buffer alias domain.
    if (resource == WaferCommunicationResource::get() &&
        mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation))
      continue;

    std::optional<MemorySpace> memorySpace;
    if (resource == WaferSPMResource::get())
      memorySpace = MemorySpace::SPM;
    else if (resource == WaferDDRResource::get())
      memorySpace = MemorySpace::DDR;
    if (!memorySpace) {
      if (failureReason)
        *failureReason =
            "selected-buffer instruction has an unsupported rootless resource "
            "effect";
      return mlir::failure();
    }

    bool rootlessRead =
        mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool rootlessWrite =
        mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
    if (!rootlessRead && !rootlessWrite) {
      if (failureReason)
        *failureReason =
            "selected-buffer instruction has an unsupported rootless storage "
            "effect";
      return mlir::failure();
    }
    bool witnessed = llvm::any_of(instances, [&](const auto &candidate) {
      mlir::Value value = candidate.getValue();
      auto type =
          value ? mlir::dyn_cast<mlir::MemRefType>(value.getType()) : nullptr;
      MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
      if (!memory || memory.getSpace() != *memorySpace)
        return false;
      return (rootlessRead &&
              mlir::isa<mlir::MemoryEffects::Read>(candidate.getEffect())) ||
             (rootlessWrite &&
              mlir::isa<mlir::MemoryEffects::Write>(candidate.getEffect()));
    });
    if (!witnessed) {
      if (failureReason)
        *failureReason =
            "selected-buffer instruction has an unwitnessed rootless storage "
            "effect";
      return mlir::failure();
    }
  }

  for (const auto &instance : instances) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect()))
      continue;
    mlir::Value value = instance.getValue();
    if (!value)
      continue;
    mlir::FailureOr<mlir::Value> root = getStaticAliasRoot(value, loop);
    if (mlir::failed(root)) {
      if (failureReason)
        *failureReason =
            "selected-buffer instruction has an unsupported alias producer";
      return mlir::failure();
    }
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool write = mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect()) ||
                 mlir::isa<mlir::MemoryEffects::Free>(instance.getEffect());
    if (!read && !write) {
      if (failureReason)
        *failureReason =
            "selected-buffer instruction has an unsupported memory effect";
      return mlir::failure();
    }
    auto found = llvm::find_if(accesses, [&](const BufferAccess &access) {
      return access.issueOperation == operationIndex && access.root == *root;
    });
    if (found == accesses.end())
      accesses.push_back({*root, write, operationIndex, accessEndIndex});
    else
      found->write |= write;
  }
  return mlir::success();
}

static mlir::LogicalResult validateLeadingBackedgeJoins(
    SelectedBufferPlan &plan, llvm::ArrayRef<BufferAccess> accesses,
    mlir::scf::ForOp loop, std::string *failureReason) {
  auto fail = [&](llvm::Twine message) {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  for (const auto &[join, producers] : plan.backedgeJoinProducers) {
    unsigned joinIndex = plan.operationIndices.lookup(join);
    llvm::DenseMap<uint32_t, unsigned> firstFollowingWorkerIssue;
    for (unsigned index = joinIndex + 1; index < plan.operations.size();
         ++index) {
      auto worker = plan.instructionWorkers.find(plan.operations[index]);
      if (worker == plan.instructionWorkers.end())
        continue;
      firstFollowingWorkerIssue.try_emplace(
          static_cast<uint32_t>(worker->second), index);
    }

    uint32_t conflictParticipants = 0;
    for (mlir::Operation *producer : producers) {
      auto worker = plan.instructionWorkers.find(producer);
      if (worker == plan.instructionWorkers.end())
        return mlir::failure();
      uint32_t workerOrdinal = static_cast<uint32_t>(worker->second);
      auto firstIssue = firstFollowingWorkerIssue.find(workerOrdinal);
      if (firstIssue == firstFollowingWorkerIssue.end())
        return mlir::failure();
      unsigned producerIndex = plan.operationIndices.lookup(producer);
      for (const BufferAccess &tailAccess : accesses) {
        if (tailAccess.issueOperation != producerIndex)
          continue;
        for (const BufferAccess &prefixAccess : accesses) {
          if (prefixAccess.issueOperation <= joinIndex ||
              prefixAccess.issueOperation >= firstIssue->second ||
              (!prefixAccess.write && !tailAccess.write))
            continue;
          bool provenDistinct =
              areProvenDistinct(prefixAccess.root, tailAccess.root, loop);
          if (provenDistinct)
            continue;
          if (prefixAccess.root != tailAccess.root)
            return fail("selected-buffer leading NCC join has an unproven "
                        "backedge alias");
          auto allocation =
              prefixAccess.root.getDefiningOp<mlir::memref::AllocOp>();
          if (!allocation || allocation->getParentOp() != loop.getOperation())
            return fail(
                "selected-buffer leading NCC join backedge conflict is not a "
                "slotizable loop-local allocation");
          conflictParticipants |= uint32_t{1} << workerOrdinal;
        }
      }
    }

    uint32_t participants =
        getNCCOperationCompletion(join).participantMask & kAllNCCWorkersMask;
    if (conflictParticipants != participants)
      return fail(
          "selected-buffer leading NCC join has a participant without an exact "
          "slotizable backedge conflict");
    plan.removableBackedgeJoins.insert(join);
  }
  return mlir::success();
}

static mlir::FailureOr<SelectedBufferPlan> buildSelectedBufferPlan(
    mlir::scf::ForOp loop, std::string *failureReason,
    SelectedBufferMaterializationFailureKind *failureKind = nullptr,
    llvm::ArrayRef<SelectedBufferRequest> requests = {},
    const StructuredMaterializationRelations *materializationRelations =
        nullptr,
    size_t *failedRequest = nullptr) {
  if (failureReason)
    failureReason->clear();
  if (failureKind)
    *failureKind =
        SelectedBufferMaterializationFailureKind::UnsupportedStructure;
  if (!loop || !loop.getBody())
    return failPlan(failureReason, failureKind,
                    SelectedBufferMaterializationFailureKind::NoExactLoop,
                    "selected buffering requires an scf.for");
  if (!loop.getRegion().hasOneBlock())
    return failPlan(failureReason,
                    "selected buffering requires a single-block scf.for");

  std::optional<uint64_t> tripCount = getPositiveStaticTripCount(loop);
  if (!tripCount)
    return failPlan(
        failureReason, failureKind,
        SelectedBufferMaterializationFailureKind::NoExactLoop,
        "selected buffering requires a static positive trip count and step");

  SelectedBufferPlan plan;
  plan.tripCount = *tripCount;
  llvm::SmallVector<mlir::memref::AllocOp, 4> allocations;
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 4>, kNCCWorkerCount>
      pendingNCCIssues(kNCCWorkerCount);
  SyncNCCJoinOp deferredLeadingJoin;
  uint32_t deferredLeadingParticipants = 0;
  bool sawInstructionOrJoin = false;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0)
      return failPlan(failureReason, failureKind,
                      SelectedBufferMaterializationFailureKind::NestedRegion,
                      "selected buffering rejects nested region operation " +
                          operation.getName().getStringRef() +
                          " in the loop body");
    if (mlir::isa<mlir::memref::AllocOp>(operation)) {
      if (!isStaticLoopAllocation(&operation))
        return failPlan(
            failureReason,
            "selected buffering requires static loop-local allocations");
      allocations.push_back(mlir::cast<mlir::memref::AllocOp>(operation));
      continue;
    }
    if (auto deallocation =
            mlir::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      plan.deallocations.push_back(deallocation);
      continue;
    }

    unsigned index = static_cast<unsigned>(plan.operations.size());
    plan.operationIndices.try_emplace(&operation, index);
    plan.operations.push_back(&operation);

    NCCOperationCompletion contract = getNCCOperationCompletion(&operation);
    if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
      llvm::SmallVector<mlir::Operation *, 4> producers;
      bool missingParticipantProducer = false;
      for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
        if ((contract.participantMask & (uint32_t{1} << worker)) == 0)
          continue;
        if (pendingNCCIssues[worker].empty()) {
          missingParticipantProducer = true;
          continue;
        }
        producers.append(pendingNCCIssues[worker]);
      }
      if (missingParticipantProducer) {
        if (!producers.empty() || sawInstructionOrJoin || deferredLeadingJoin)
          return failPlan(
              failureReason,
              "selected-buffer NCC join participant has no preceding pending "
              "producer");
        deferredLeadingJoin = join;
        deferredLeadingParticipants = contract.participantMask;
        sawInstructionOrJoin = true;
        continue;
      }
      for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
        if ((contract.participantMask & (uint32_t{1} << worker)) == 0)
          continue;
        for (mlir::Operation *producer : pendingNCCIssues[worker]) {
          plan.nccIssueJoins.try_emplace(producer, join.getOperation());
        }
        pendingNCCIssues[worker].clear();
      }
      if (producers.empty())
        return failPlan(
            failureReason,
            "selected-buffer NCC join requires exact pending producer "
            "participants");
      plan.explicitJoinProducers.try_emplace(join.getOperation(),
                                             std::move(producers));
      sawInstructionOrJoin = true;
      continue;
    }

    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(&operation);
    if (!instruction) {
      if (mlir::isa<SyncNCCJoinOp>(operation))
        continue;
      if (!isSupportedPureBodyOperation(&operation))
        return failPlan(
            failureReason,
            "selected buffering encountered unknown effectful operation " +
                operation.getName().getStringRef());
      continue;
    }
    InstrFamily family = instruction.getInstructionFamily();
    if (family == InstrFamily::DTE) {
      if (!mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation))
        return failPlan(
            failureReason,
            "selected buffering encountered an unsupported Direct DTE "
            "instruction");
      plan.instructionFamilies.try_emplace(&operation, family);
      sawInstructionOrJoin = true;
      continue;
    }
    if (contract.kind != NCCCompletionKind::OrderedAsynchronousIssue ||
        !contract.issueWorker)
      return failPlan(
          failureReason,
          "selected buffering rejects join/wait and synchronous islands");
    plan.instructionFamilies.try_emplace(&operation, family);
    plan.instructionWorkers.try_emplace(&operation, *contract.issueWorker);
    pendingNCCIssues[static_cast<uint32_t>(*contract.issueWorker)].push_back(
        &operation);
    sawInstructionOrJoin = true;
  }
  if (deferredLeadingJoin) {
    uint32_t tailParticipants = 0;
    llvm::SmallVector<mlir::Operation *, 4> tailProducers;
    for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
      if (pendingNCCIssues[worker].empty())
        continue;
      tailParticipants |= uint32_t{1} << worker;
      tailProducers.append(pendingNCCIssues[worker]);
    }
    if (tailParticipants != deferredLeadingParticipants)
      return failPlan(failureReason, "selected-buffer leading NCC join "
                                     "participants do not exactly match the "
                                     "loop-tail pending issue set");
    plan.backedgeJoinProducers.try_emplace(deferredLeadingJoin.getOperation(),
                                           std::move(tailProducers));
  }
  if (plan.operations.empty() || plan.instructionFamilies.size() < 2)
    return failPlan(
        failureReason,
        "selected buffering requires at least two typed instructions");
  for (mlir::memref::DeallocOp deallocation : plan.deallocations) {
    mlir::FailureOr<mlir::Value> root =
        getStaticAliasRoot(deallocation.getMemref(), loop);
    auto allocation = mlir::succeeded(root)
                          ? root->getDefiningOp<mlir::memref::AllocOp>()
                          : mlir::memref::AllocOp{};
    if (!allocation || allocation->getParentOp() != loop.getOperation() ||
        !llvm::is_contained(allocations, allocation))
      return failPlan(
          failureReason,
          "selected buffering requires loop-local deallocation ownership");
  }
  if (mlir::failed(collectDirectDTEWaits(plan, failureReason)))
    return mlir::failure();
  plan.predecessors.resize(plan.operations.size());
  for (const auto &[join, producers] : plan.explicitJoinProducers) {
    unsigned joinIndex = plan.operationIndices.lookup(join);
    for (mlir::Operation *producer : producers)
      addDependency(plan, plan.operationIndices.lookup(producer), joinIndex,
                    /*advancesStage=*/false);
  }

  // Preserve all direct SSA definitions. Memory dependencies below add the
  // value-associated RAW/WAR/WAW edges that instruction ops express through
  // effects rather than SSA results.
  for (auto [successor, operation] : llvm::enumerate(plan.operations)) {
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = plan.operationIndices.find(definition);
      if (found == plan.operationIndices.end())
        continue;
      addDependency(plan, found->second, static_cast<unsigned>(successor),
                    requests.empty() && dependencyAdvancesStage(
                                            plan, found->second,
                                            static_cast<unsigned>(successor)));
    }
  }

  // Preserve typed NCC issue/join order explicitly. A participant join
  // stays in the producer stage; a later cross-family DTE buffer dependency
  // advances the transport to the next stage. This lets the steady kernel
  // issue the prior tile's DTE, launch independent work for the next tile,
  // then wait the exact DTE event without carrying an event across iterations.
  std::array<llvm::SmallVector<unsigned, 4>, kNCCWorkerCount>
      pendingWorkerIssues;
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    NCCOperationCompletion contract = getNCCOperationCompletion(operation);
    if (contract.kind == NCCCompletionKind::OrderedAsynchronousIssue) {
      if (!contract.issueWorker)
        return failPlan(failureReason,
                        "selected-buffer NCC issue has no typed worker");
      unsigned worker = static_cast<unsigned>(*contract.issueWorker);
      if (worker >= kNCCWorkerCount)
        return failPlan(
            failureReason,
            "selected-buffer NCC issue worker is outside the typed domain");
      pendingWorkerIssues[worker].push_back(static_cast<unsigned>(index));
      continue;
    }
    if (contract.kind != NCCCompletionKind::ParticipantJoin)
      continue;
    if (contract.participantMask == 0 ||
        (contract.participantMask & ~kAllNCCWorkersMask) != 0)
      return failPlan(
          failureReason,
          "selected-buffer participant join has an invalid worker mask");
    for (unsigned worker = 0; worker < kNCCWorkerCount; ++worker) {
      if ((contract.participantMask & (uint32_t{1} << worker)) == 0)
        continue;
      for (unsigned predecessor : pendingWorkerIssues[worker])
        addDependency(plan, predecessor, static_cast<unsigned>(index),
                      /*advancesStage=*/false);
      pendingWorkerIssues[worker].clear();
    }
  }

  // The current target exposes one Direct-DTE sender slot. Preserve
  // its typed issue/release chain independently of buffer aliasing so
  // pipelining may overlap one send with NCC work but never overlaps two
  // sender events that the ABI cannot represent.
  std::optional<unsigned> pendingSenderIssue;
  std::optional<unsigned> lastSenderWait;
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    if (mlir::isa<InstrDTESendOp>(operation)) {
      if (pendingSenderIssue)
        return failPlan(
            failureReason,
            "selected buffering has overlapping Direct DTE senders");
      if (lastSenderWait)
        addDependency(plan, *lastSenderWait, static_cast<unsigned>(index),
                      /*advancesStage=*/false);
      pendingSenderIssue = static_cast<unsigned>(index);
      continue;
    }
    auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation);
    if (!wait || !pendingSenderIssue)
      continue;
    bool completesSender =
        llvm::any_of(wait.getTokens(), [&](mlir::Value token) {
          return token.getDefiningOp() == plan.operations[*pendingSenderIssue];
        });
    if (!completesSender)
      continue;
    addDependency(plan, *pendingSenderIssue, static_cast<unsigned>(index),
                  /*advancesStage=*/false);
    pendingSenderIssue.reset();
    lastSenderWait = static_cast<unsigned>(index);
  }
  if (pendingSenderIssue)
    return failPlan(failureReason,
                    "selected-buffer Direct DTE sender has no matching wait");

  // Receiver FSM ids are also finite target resources. Keep each source
  // batch intact and require its exact wait before the next batch can enter
  // the software pipeline; a batch larger than the four-FSM normal profile is
  // not a legal candidate.
  llvm::SmallVector<unsigned, 4> pendingReceiverIssues;
  std::optional<unsigned> lastReceiverWait;
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    if (mlir::isa<InstrDTERecvOp>(operation)) {
      if (pendingReceiverIssues.empty() && lastReceiverWait)
        addDependency(plan, *lastReceiverWait, static_cast<unsigned>(index),
                      /*advancesStage=*/false);
      pendingReceiverIssues.push_back(static_cast<unsigned>(index));
      if (pendingReceiverIssues.size() > 4)
        return failPlan(
            failureReason,
            "selected buffering exceeds the Direct DTE receiver FSM "
            "profile");
      continue;
    }
    auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation);
    if (!wait || pendingReceiverIssues.empty())
      continue;
    for (mlir::Value token : wait.getTokens()) {
      mlir::Operation *definition = token.getDefiningOp();
      auto found = llvm::find_if(pendingReceiverIssues, [&](unsigned issue) {
        return plan.operations[issue] == definition;
      });
      if (found == pendingReceiverIssues.end())
        continue;
      addDependency(plan, *found, static_cast<unsigned>(index),
                    /*advancesStage=*/false);
      pendingReceiverIssues.erase(found);
    }
    if (pendingReceiverIssues.empty())
      lastReceiverWait = static_cast<unsigned>(index);
  }
  if (!pendingReceiverIssues.empty())
    return failPlan(failureReason,
                    "selected-buffer Direct DTE receiver has no matching wait");

  llvm::SmallVector<BufferAccess, 16> accesses;
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    if (!plan.instructionFamilies.contains(operation))
      continue;
    unsigned accessEndIndex = static_cast<unsigned>(index);
    auto dteWait = plan.directDTEWaits.find(operation);
    if (dteWait != plan.directDTEWaits.end())
      accessEndIndex = plan.operationIndices.lookup(dteWait->second);
    auto nccJoin = plan.nccIssueJoins.find(operation);
    if (nccJoin != plan.nccIssueJoins.end())
      accessEndIndex = plan.operationIndices.lookup(nccJoin->second);
    llvm::SmallVector<BufferAccess, 8> current;
    if (mlir::failed(collectInstructionAccesses(
            operation, static_cast<unsigned>(index), accessEndIndex, loop,
            current, failureReason)))
      return mlir::failure();
    for (const BufferAccess &access : current) {
      for (const BufferAccess &prior : accesses) {
        bool provenDistinct = areProvenDistinct(access.root, prior.root, loop);
        if (!access.write && !prior.write)
          continue;
        mlir::Operation *priorIssue = plan.operations[prior.issueOperation];
        bool priorDirectDTE = isDirectDTEIssue(priorIssue);
        bool currentDirectDTE = isDirectDTEIssue(operation);

        if (access.root != prior.root && !provenDistinct)
          return failPlan(
              failureReason,
              "selected buffering cannot prove two accessed roots distinct");
        if (provenDistinct)
          continue;

        // A DTE issue owns every write hazard on its buffer until the exact
        // token wait. An overwrite or receive before that wait would force a
        // backwards dependency; read/read pairs were intentionally skipped
        // above.
        if (priorDirectDTE && prior.completionOperation >
                                  static_cast<unsigned>(access.issueOperation))
          return failPlan(failureReason, "selected-buffer Direct DTE buffer "
                                         "access precedes its exact wait");

        auto priorWorker = plan.instructionWorkers.find(priorIssue);
        auto currentWorker = plan.instructionWorkers.find(operation);
        if (!priorDirectDTE && currentDirectDTE &&
            (priorWorker == plan.instructionWorkers.end() ||
             prior.completionOperation == prior.issueOperation ||
             prior.completionOperation >= access.issueOperation))
          return failPlan(
              failureReason,
              "selected-buffer NCC-to-Direct-DTE access requires an "
              "explicit preceding participant join");
        if (!priorDirectDTE && !currentDirectDTE &&
            (priorWorker == plan.instructionWorkers.end() ||
             currentWorker == plan.instructionWorkers.end()))
          return failPlan(failureReason,
                          "selected buffering lost an NCC issue worker");
        if (!priorDirectDTE && !currentDirectDTE &&
            priorWorker->second != currentWorker->second &&
            (prior.completionOperation == prior.issueOperation ||
             prior.completionOperation >= access.issueOperation))
          return failPlan(
              failureReason,
              "selected-buffer cross-worker memory hazard requires an explicit "
              "preceding participant join");
        unsigned predecessor = prior.issueOperation;
        if (prior.completionOperation != prior.issueOperation &&
            prior.completionOperation < access.issueOperation)
          predecessor = prior.completionOperation;
        addDependency(plan, predecessor, access.issueOperation,
                      requests.empty() &&
                          dependencyAdvancesStage(plan, predecessor,
                                                  access.issueOperation));
      }
    }
    // Operand effects on one typed instruction describe one atomic issue.
    // Compare its normalized roots only with earlier issues; the operation's
    // own verifier/interface owns source/destination alias legality.
    accesses.append(current);
  }
  if (mlir::failed(
          validateLeadingBackedgeJoins(plan, accesses, loop, failureReason)))
    return mlir::failure();

  // The joint state selected a logical edge, not an instruction-family-wide
  // pipeline. For a selected logical edge, advance exactly one dependency cut
  // for each requested local/transport endpoint. All other engine transitions
  // preserve their original within-iteration order in the same stage.
  if (!requests.empty() &&
      mlir::failed(markSelectedBufferStageBoundaries(
          plan, requests, *materializationRelations, failedRequest))) {
    if (failureKind)
      *failureKind = SelectedBufferMaterializationFailureKind::EdgeNotWitnessed;
    return failPlan(
        failureReason,
        "selected buffering could not bind a stage boundary to the requested "
        "logical edge");
  }

  // Original block order is already a topological order for direct SSA and
  // effect dependencies. A cross-engine dependency advances one iteration
  // stage; same-engine ordering remains in the same stage.
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    unsigned stage = 0;
    for (const DependencyEdge &edge :
         plan.predecessors[static_cast<unsigned>(index)]) {
      mlir::Operation *predecessor = plan.operations[edge.predecessor];
      auto found = plan.stages.find(predecessor);
      if (found == plan.stages.end())
        return failPlan(failureReason,
                        "selected-buffer dependency DAG is not topological");
      stage = std::max(stage, found->second + (edge.advancesStage ? 1U : 0U));
    }
    plan.stages.try_emplace(operation, stage);
    plan.maxStage = std::max(plan.maxStage, stage);
  }
  llvm::DenseMap<mlir::Operation *, unsigned> directDTEIssueStages;
  for (const auto &[issue, wait] : plan.directDTEWaits) {
    unsigned stage = plan.stages.lookup(issue);
    auto [found, inserted] = directDTEIssueStages.try_emplace(wait, stage);
    if (!inserted && found->second != stage)
      return failPlan(failureReason, "selected-buffer Direct DTE issues "
                                     "sharing one exact wait must occupy "
                                     "one issue stage");
  }
  if (plan.maxStage == 0)
    return failPlan(
        failureReason, failureKind,
        SelectedBufferMaterializationFailureKind::NoCrossEngineStage,
        "selected buffering has no proven cross-engine pipeline stage");
  const uint64_t minimumTripCount =
      requests.empty()
          ? static_cast<uint64_t>(plan.maxStage) + 1
          : std::max<uint64_t>(plan.maxStage + 1, requests.front().bufferCount);
  if (plan.tripCount < minimumTripCount)
    return failPlan(
        failureReason, failureKind,
        SelectedBufferMaterializationFailureKind::TripCountTooSmall,
        "selected-buffer trip count is smaller than the derived stage count");

  // Loop-external roots are not renamed by this transform. If a written root
  // is touched in more than one stage, the steady kernel can reverse an
  // original loop-carried RAW/WAR/WAW order (for example, stage 1 of
  // iteration i+1 can issue before stage 2 of iteration i). Same-stage
  // accesses retain source and iteration order, and read-only roots have no
  // such carried hazard. Until an address/range analysis proves iterations
  // disjoint, reject the cross-stage case rather than relying on the
  // busytable to serialize an already incorrect issue order.
  struct ExternalRootStageSummary {
    bool hasWrite = false;
    std::optional<unsigned> firstStage;
    bool crossesStage = false;
  };
  llvm::DenseMap<mlir::Value, ExternalRootStageSummary> externalRoots;
  for (const BufferAccess &access : accesses) {
    auto allocation = access.root.getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && allocation->getParentOp() == loop.getOperation())
      continue;
    unsigned issueStage =
        plan.stages.lookup(plan.operations[access.issueOperation]);
    unsigned completionStage =
        plan.stages.lookup(plan.operations[access.completionOperation]);
    ExternalRootStageSummary &summary = externalRoots[access.root];
    summary.hasWrite |= access.write;
    if (!summary.firstStage)
      summary.firstStage = issueStage;
    else
      summary.crossesStage |= *summary.firstStage != issueStage;
    summary.crossesStage |= issueStage != completionStage;
  }
  for (const auto &[root, summary] : externalRoots) {
    (void)root;
    if (summary.hasWrite && summary.crossesStage)
      return failPlan(
          failureReason,
          "selected buffering rejects a loop-external cross-stage write "
          "hazard without iteration-disjoint ranges");
  }

  llvm::DenseMap<mlir::Operation *, std::pair<unsigned, unsigned>>
      allocationStageSpans;
  for (const BufferAccess &access : accesses) {
    auto allocation = access.root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || allocation->getParentOp() != loop.getOperation())
      continue;
    unsigned issueStage =
        plan.stages.lookup(plan.operations[access.issueOperation]);
    unsigned completionStage =
        plan.stages.lookup(plan.operations[access.completionOperation]);
    unsigned firstStage = std::min(issueStage, completionStage);
    unsigned lastStage = std::max(issueStage, completionStage);
    auto [found, inserted] = allocationStageSpans.try_emplace(
        allocation.getOperation(), std::make_pair(firstStage, lastStage));
    if (!inserted) {
      found->second.first = std::min(found->second.first, firstStage);
      found->second.second = std::max(found->second.second, lastStage);
    }
  }

  for (mlir::memref::AllocOp allocation : allocations) {
    auto span = allocationStageSpans.find(allocation.getOperation());
    if (span == allocationStageSpans.end())
      return failPlan(
          failureReason,
          "selected-buffer loop allocation has no proven instruction lifetime");
    unsigned slotCount = span->second.second - span->second.first + 1;
    if (!requests.empty() && slotCount > 1)
      slotCount = std::max<unsigned>(slotCount, requests.front().bufferCount);
    if (slotCount == 0 || plan.slotAllocationCount >
                              std::numeric_limits<unsigned>::max() - slotCount)
      return failPlan(failureReason,
                      "selected-buffer allocation count is not representable");
    plan.allocations.push_back(
        {allocation, span->second.first, span->second.second, slotCount});
    plan.slotAllocationCount += slotCount;
    plan.maximumSlotCount = std::max(plan.maximumSlotCount, slotCount);
  }
  if (plan.allocations.empty() || plan.slotAllocationCount < 2)
    return failPlan(
        failureReason,
        "selected buffering did not derive a real multi-buffer allocation");

  // The SCF utility supports only distance-zero/one recurrences whose yielded
  // value is defined in the loop or outside it. Reject unsupported original
  // recurrence before cloning; slot rotation is added below with explicit
  // stage-zero pointer identities, while slot count separately represents the
  // allocation's cross-stage memory lifetime.
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getNumRegionIterArgs())
    return failPlan(failureReason,
                    "selected buffering has an invalid scf.for recurrence");
  for (mlir::Value value : yield.getOperands()) {
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
      bool supportedPassThrough =
          blockArg && blockArg.getOwner() == loop.getBody() &&
          blockArg.getArgNumber() > 0 &&
          mlir::isa<mlir::BaseMemRefType>(blockArg.getType());
      if (!supportedPassThrough)
        return failPlan(
            failureReason,
            "selected buffering rejects unsupported pass-through loop "
            "recurrence");
      continue;
    }
    if (definition && loop->isAncestor(definition) &&
        !plan.operationIndices.contains(definition))
      return failPlan(
          failureReason,
          "selected buffering has an unsupported recurrence definition");
  }
  return plan;
}

static mlir::FailureOr<mlir::scf::ForOp> materializeSlotsAndRotation(
    mlir::scf::ForOp loop, const SelectedBufferPlan &plan,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason) {
  mlir::IRRewriter rewriter(loop.getContext());
  rewriter.setInsertionPoint(loop);

  llvm::SmallVector<mlir::Value, 8> initArgs(loop.getInitArgs().begin(),
                                             loop.getInitArgs().end());
  llvm::SmallVector<mlir::memref::AllocOp, 8> materializedSlots;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 3>> slotsBySource;
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    mlir::memref::AllocOp sourceAllocation = allocationPlan.allocation;
    for (unsigned index = 0; index < allocationPlan.slotCount; ++index) {
      mlir::Operation *clone = rewriter.clone(*sourceAllocation.getOperation());
      auto clonedAllocation = mlir::cast<mlir::memref::AllocOp>(clone);
      initArgs.push_back(clonedAllocation.getResult());
      materializedSlots.push_back(clonedAllocation);
      slotsBySource[sourceAllocation.getResult()].push_back(
          clonedAllocation.getResult());
    }
  }

  if (materializationRelations) {
    auto retarget = [&](auto &relations) {
      using Relation = typename std::decay_t<decltype(relations)>::value_type;
      llvm::SmallVector<Relation, 16> updated;
      for (const Relation &relation : relations) {
        bool replaced = false;
        for (const auto &[source, slots] : slotsBySource) {
          if (!shareStructuredBufferStorage(relation.buffer, source))
            continue;
          for (mlir::Value slot : slots) {
            Relation copy = relation;
            copy.buffer = slot;
            updated.push_back(copy);
          }
          replaced = true;
          break;
        }
        if (replaced)
          continue;
        mlir::Operation *definition = relation.buffer.getDefiningOp();
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(relation.buffer);
        const bool erasedWithLoop =
            (definition && loop->isAncestor(definition)) ||
            (argument && argument.getOwner() == loop.getBody());
        if (!erasedWithLoop)
          updated.push_back(relation);
      }
      relations.assign(updated.begin(), updated.end());
    };
    retarget(materializationRelations->operationResultBuffers);
    retarget(materializationRelations->operandBuffers);
    retarget(materializationRelations->outputBuffers);
  }

  auto preparedLoop = rewriter.create<mlir::scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      initArgs);
  mlir::Block *preparedBody = preparedLoop.getBody();
  if (!preparedBody->empty() &&
      mlir::isa<mlir::scf::YieldOp>(preparedBody->back()))
    rewriter.eraseOp(&preparedBody->back());
  rewriter.setInsertionPointToStart(preparedBody);

  mlir::IRMapping mapping;
  mapping.map(loop.getInductionVar(), preparedLoop.getInductionVar());
  mapping.map(
      loop.getRegionIterArgs(),
      preparedLoop.getRegionIterArgs().take_front(loop.getNumRegionIterArgs()));

  unsigned slotArgument = loop.getNumRegionIterArgs();
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    mlir::memref::AllocOp sourceAllocation = allocationPlan.allocation;
    mapping.map(sourceAllocation.getResult(),
                preparedLoop.getRegionIterArgs()[slotArgument]);
    slotArgument += allocationPlan.slotCount;
  }

  std::vector<std::pair<mlir::Operation *, unsigned>> schedule;
  schedule.reserve(plan.operations.size() + plan.slotAllocationCount);
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> operationClones;
  for (mlir::Operation *operation : plan.operations) {
    if (plan.removableBackedgeJoins.contains(operation))
      continue;
    mlir::Operation *clone = rewriter.clone(*operation, mapping);
    operationClones.try_emplace(operation, clone);
  }

  auto appendScheduledOperation = [&](mlir::Operation *operation) {
    if (plan.removableBackedgeJoins.contains(operation))
      return;
    schedule.emplace_back(operationClones.lookup(operation),
                          plan.stages.lookup(operation));
  };
  if (plan.directDTEWaits.empty()) {
    for (mlir::Operation *operation : plan.operations)
      appendScheduledOperation(operation);
  } else {
    unsigned firstIssueIndex = std::numeric_limits<unsigned>::max();
    unsigned lastWaitIndex = 0;
    unsigned firstWindowStage = std::numeric_limits<unsigned>::max();
    for (const auto &[issue, wait] : plan.directDTEWaits) {
      firstIssueIndex =
          std::min(firstIssueIndex, plan.operationIndices.lookup(issue));
      lastWaitIndex =
          std::max(lastWaitIndex, plan.operationIndices.lookup(wait));
      firstWindowStage = std::min(firstWindowStage, plan.stages.lookup(issue));
    }
    if (firstIssueIndex == std::numeric_limits<unsigned>::max() ||
        lastWaitIndex >= plan.operations.size()) {
      if (failureReason)
        *failureReason =
            "selected-buffer Direct DTE window lost its issue owner";
      return mlir::failure();
    }
    // Every exact wait remains in the same stage and direct SSA block as its
    // issue group. Rotate stage-later work after the final window ahead of the
    // complete DTE sequence, so the preceding tile's NCC work is pending while
    // the current tile executes one or more endpoint-safe issue/wait segments.
    llvm::DenseSet<mlir::Operation *> rotated;
    for (unsigned index = lastWaitIndex + 1; index < plan.operations.size();
         ++index) {
      mlir::Operation *operation = plan.operations[index];
      if (plan.stages.lookup(operation) <= firstWindowStage)
        continue;
      appendScheduledOperation(operation);
      rotated.insert(operation);
    }
    for (mlir::Operation *operation : plan.operations)
      if (!rotated.contains(operation))
        appendScheduledOperation(operation);
  }

  std::vector<std::pair<mlir::Operation *, unsigned>> pointerPermutations;
  pointerPermutations.reserve(loop.getNumRegionIterArgs() +
                              plan.slotAllocationCount);
  auto originalYield =
      mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value, 8> yieldValues;
  yieldValues.reserve(originalYield.getNumOperands() +
                      plan.slotAllocationCount);
  for (mlir::Value value : originalYield.getOperands()) {
    mlir::Value mapped = mapping.lookupOrDefault(value);
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (blockArg && blockArg.getOwner() == loop.getBody() &&
        blockArg.getArgNumber() > 0) {
      auto identity = rewriter.create<mlir::memref::CastOp>(
          loop.getLoc(), mapped.getType(), mapped);
      identity->setAttr(kPointerPermutationMarker, rewriter.getUnitAttr());
      yieldValues.push_back(identity.getResult());
      pointerPermutations.emplace_back(identity.getOperation(), 0);
      continue;
    }
    yieldValues.push_back(mapped);
  }

  slotArgument = loop.getNumRegionIterArgs();
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    llvm::SmallVector<mlir::Value, 4> rotated;
    rotated.reserve(allocationPlan.slotCount);
    for (unsigned index = 0; index < allocationPlan.slotCount; ++index) {
      unsigned sourceIndex = (index + 1) % allocationPlan.slotCount;
      mlir::Value source =
          preparedLoop.getRegionIterArgs()[slotArgument + sourceIndex];
      auto identity = rewriter.create<mlir::memref::CastOp>(
          loop.getLoc(), source.getType(), source);
      identity->setAttr(kPointerPermutationMarker, rewriter.getUnitAttr());
      rotated.push_back(identity.getResult());
      // This is a pointer permutation, not the buffer's final memory use.
      // Slot count encodes reuse distance; stage zero keeps the distance-one
      // SCF recurrence valid for slot families spanning three or more stages.
      pointerPermutations.emplace_back(identity.getOperation(), 0);
    }
    yieldValues.append(rotated);
    slotArgument += allocationPlan.slotCount;
  }
  rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), yieldValues);

  for (auto [oldResult, newResult] :
       llvm::zip(loop.getResults(),
                 preparedLoop.getResults().take_front(loop.getNumResults())))
    oldResult.replaceAllUsesWith(newResult);
  rewriter.eraseOp(loop);

  // Rotation definitions produce the values observed through next
  // iteration's region arguments. SCF's cyclic schedule verifier therefore
  // needs them before every current-iteration consumer in operation order.
  // They are pure pointer permutations at stage zero; buffer memory lifetime
  // is represented separately by the number of rotated slots. For a Direct
  // DTE window, stage-later work is ordered first so its nonblocking NCC issue
  // overlaps the following tile's direct issue+exact wait without carrying a
  // DTE token across the loop backedge.
  schedule.insert(schedule.begin(), pointerPermutations.begin(),
                  pointerPermutations.end());
  mlir::scf::PipeliningOption options;
  options.getScheduleFn =
      [schedule = std::move(schedule)](
          mlir::scf::ForOp,
          std::vector<std::pair<mlir::Operation *, unsigned>> &result) {
        result = schedule;
      };
  options.supportDynamicLoops = false;
  options.peelEpilogue = true;

  rewriter.setInsertionPoint(preparedLoop);
  bool modifiedIR = false;
  mlir::FailureOr<mlir::scf::ForOp> pipelined =
      mlir::scf::pipelineForLoop(rewriter, preparedLoop, options, &modifiedIR);
  if (mlir::failed(pipelined)) {
    if (failureReason)
      *failureReason = modifiedIR
                           ? "SCF pipelining failed after mutating the private "
                             "candidate clone"
                           : "SCF pipelining rejected the verified schedule";
    return mlir::failure();
  }

  // Loop-local allocations became loop-external rotating slot families.
  // Their last uses include the generated epilogue, so release them at the
  // owning block boundary rather than scheduling the original per-iteration
  // deallocation as pipeline work.
  mlir::Block *ownerBlock = pipelined->getOperation()->getBlock();
  if (!ownerBlock || !ownerBlock->getTerminator()) {
    if (failureReason)
      *failureReason = "selected buffering lost the slot owner block";
    return mlir::failure();
  }
  rewriter.setInsertionPoint(ownerBlock->getTerminator());
  for (mlir::memref::AllocOp allocation : materializedSlots)
    rewriter.create<mlir::memref::DeallocOp>(allocation.getLoc(),
                                             allocation.getResult());

  return *pipelined;
}

static void eraseSynthesizedPointerPermutations(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::memref::CastOp, 8> identities;
  module.walk([&](mlir::memref::CastOp cast) {
    if (!cast->hasAttr(kPointerPermutationMarker))
      return;
    cast->removeAttr(kPointerPermutationMarker);
    if (cast.getSource().getType() == cast.getResult().getType())
      identities.push_back(cast);
  });
  for (mlir::memref::CastOp identity : identities) {
    identity.getResult().replaceAllUsesWith(identity.getSource());
    identity.erase();
  }
}

static void splitDirectDTEWaitGroups(mlir::scf::ForOp loop) {
  llvm::SmallVector<InstrDTEWaitOp, 4> groupedWaits;
  for (mlir::Operation &operation : loop.getBody()->without_terminator())
    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(&operation);
        wait && wait.getTokens().size() > 1)
      groupedWaits.push_back(wait);

  for (InstrDTEWaitOp wait : groupedWaits) {
    mlir::OpBuilder builder(wait);
    for (mlir::Value token : wait.getTokens())
      builder.create<InstrDTEWaitOp>(wait.getLoc(), mlir::ValueRange{token});
    wait.erase();
  }
}

/// The pinned SCF utility rebuilds a static kernel upper bound as
/// `upper - maxStage * step`. Canonicalize exactly that mechanical expression;
/// a whole-module canonicalizer here would be too broad because this transform
/// must preserve unrelated source operations and pre-existing casts.
static mlir::LogicalResult
canonicalizePipelinedKernelUpperBound(mlir::scf::ForOp loop,
                                      int64_t originalUpper,
                                      int64_t originalStep, unsigned maxStage) {
  __int128 upper = static_cast<__int128>(originalUpper) -
                   static_cast<__int128>(originalStep) * maxStage;
  if (upper < std::numeric_limits<int64_t>::min() ||
      upper > std::numeric_limits<int64_t>::max())
    return mlir::failure();

  mlir::Value mechanicalUpper = loop.getUpperBound();
  mlir::OpBuilder builder(loop);
  auto canonicalUpper = builder.create<mlir::arith::ConstantIndexOp>(
      loop.getLoc(), static_cast<int64_t>(upper));
  loop.setUpperBound(canonicalUpper);

  // Only erase the two operation kinds emitted for this bound by the pinned
  // utility. Original loop bounds are direct constants in this transform's
  // input contract and therefore cannot be consumed by this cleanup.
  llvm::SmallVector<mlir::Value, 2> worklist{mechanicalUpper};
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    mlir::Operation *operation = value.getDefiningOp();
    if (!operation || !operation->use_empty() ||
        !mlir::isa<mlir::arith::SubIOp, mlir::arith::MulIOp>(operation))
      continue;
    worklist.append(operation->operand_begin(), operation->operand_end());
    operation->erase();
  }
  return mlir::success();
}

} // namespace

static mlir::FailureOr<SelectedBufferMaterialization>
materializeSelectedBufferCandidateInPlace(
    mlir::ModuleOp module, mlir::scf::ForOp loop, std::string *failureReason,
    llvm::ArrayRef<SelectedBufferRequest> requests = {},
    const StructuredMaterializationRelations *materializationRelations =
        nullptr,
    SelectedBufferMaterializationFailureKind *failureKind = nullptr,
    size_t *failedRequest = nullptr) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "static-selected-buffer-buffering",
      "materializeSelectedBufferCandidateInPlace");
  if (failureReason)
    failureReason->clear();
  if (failureKind)
    *failureKind =
        SelectedBufferMaterializationFailureKind::UnsupportedStructure;
  if (failedRequest)
    *failedRequest = std::numeric_limits<size_t>::max();
  if (!module || !loop || !module->isAncestor(loop.getOperation()))
    return failMaterialization(
        failureReason,
        "selected buffering loop is not owned by the source module");
  if (!requests.empty() && !materializationRelations)
    return failMaterialization(
        failureReason,
        "selected buffering edge requests require current buffer relations");
  if (hasPhysicalPlacementFacts(module))
    return failMaterialization(
        failureReason,
        "selected buffering requires unplaced input without physical "
        "SPM/DDR offset facts");

  std::optional<uint64_t> sourceTripCount = getPositiveStaticTripCount(loop);
  if (sourceTripCount && *sourceTripCount == 1) {
    if (!requests.empty()) {
      if (failureKind)
        *failureKind =
            SelectedBufferMaterializationFailureKind::NoCrossEngineStage;
      return failMaterialization(
          failureReason,
          "selected buffering exact edge loop has only one iteration");
    }
    if (!loop.getRegion().hasOneBlock())
      return failMaterialization(
          failureReason,
          "selected buffering identity requires a single-block scf.for");
    SelectedBufferMaterialization result;
    if (materializationRelations)
      result.materializationRelations = *materializationRelations;
    result.stageCount = 1;
    result.slotAllocationCount = 0;
    return result;
  }

  mlir::FailureOr<SelectedBufferPlan> sourcePlan;
  {
    wafer::support::ScopedCompileTimingSpan planTiming(
        "analysis-phase", "materializeSelectedBufferCandidateInPlace",
        "buildSelectedBufferPlan(source)");
    sourcePlan =
        buildSelectedBufferPlan(loop, failureReason, failureKind, requests,
                                materializationRelations, failedRequest);
  }
  if (mlir::failed(sourcePlan))
    return mlir::failure();
  if (!requests.empty() &&
      mlir::failed(validateSelectedBufferRequests(
          *sourcePlan, requests, *materializationRelations, failedRequest))) {
    if (failureKind)
      *failureKind = SelectedBufferMaterializationFailureKind::EdgeNotWitnessed;
    return failMaterialization(
        failureReason,
        "selected buffering loop has no staged dependency for the requested "
        "logical edge");
  }

  StructuredMaterializationRelations currentRelations;
  if (materializationRelations)
    currentRelations = *materializationRelations;

  // A grouped wait couples tokens whose producers may belong to different
  // pipeline stages. Split it inside the private candidate so each exact
  // event stays with its own issue stage; this preserves the normal
  // single-sender ABI and keeps transport acceptance on direct SSA edges.
  splitDirectDTEWaitGroups(loop);
  mlir::FailureOr<SelectedBufferPlan> currentPlan;
  {
    wafer::support::ScopedCompileTimingSpan planTiming(
        "analysis-phase", "materializeSelectedBufferCandidateInPlace",
        "buildSelectedBufferPlan(current)");
    currentPlan =
        buildSelectedBufferPlan(loop, failureReason, failureKind, requests,
                                &currentRelations, failedRequest);
  }
  if (mlir::failed(currentPlan))
    return mlir::failure();
  if (!requests.empty() &&
      mlir::failed(validateSelectedBufferRequests(
          *currentPlan, requests, currentRelations, failedRequest))) {
    if (failureKind)
      *failureKind = SelectedBufferMaterializationFailureKind::EdgeNotWitnessed;
    return failMaterialization(
        failureReason,
        "selected buffering current loop lost the requested logical-edge "
        "dependency");
  }
  std::optional<int64_t> currentUpper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> currentStep =
      mlir::getConstantIntValue(loop.getStep());
  if (!currentUpper || !currentStep)
    return failMaterialization(
        failureReason,
        "selected buffering loop lost its admitted static bounds");
  mlir::FailureOr<mlir::scf::ForOp> pipelined;
  {
    wafer::support::ScopedCompileTimingSpan materializeTiming(
        "transformation-phase", "materializeSelectedBufferCandidateInPlace",
        "materializeSlotsAndRotation");
    pipelined = materializeSlotsAndRotation(loop, *currentPlan,
                                            &currentRelations, failureReason);
  }
  if (mlir::failed(pipelined))
    return mlir::failure();
  {
    wafer::support::ScopedCompileTimingSpan canonicalizeTiming(
        "transformation-phase", "materializeSelectedBufferCandidateInPlace",
        "canonicalize");
    if (mlir::failed(canonicalizePipelinedKernelUpperBound(
            *pipelined, *currentUpper, *currentStep, currentPlan->maxStage)))
      return failMaterialization(
          failureReason,
          "selected buffering failed kernel-bound canonicalization");
    eraseSynthesizedPointerPermutations(module);
  }
  {
    wafer::support::ScopedCompileTimingSpan normalizeTiming(
        "transformation-phase", "materializeSelectedBufferCandidateInPlace",
        "placeRequiredNCCJoins");
    if (mlir::failed(checkStructuredBufferRelationsCurrent(
            module.getOperation(), currentRelations)))
      return failMaterialization(
          failureReason,
          "selected buffering produced relations outside its current IR");
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
      if (mlir::failed(
              wafer::detail::placeRequiredNCCJoinsInPrivateFunction(function)))
        return failMaterialization(
            failureReason,
            "selected buffering failed required NCC join placement");
    if (mlir::failed(checkStructuredBufferRelationsCurrent(
            module.getOperation(), currentRelations)))
      return failMaterialization(
          failureReason,
          "selected buffering NCC join placement changed its buffer relation "
          "domain");
  }
  {
    wafer::support::ScopedCompileTimingSpan verifyTiming(
        "analysis-phase", "materializeSelectedBufferCandidateInPlace",
        "verify");
    if (mlir::failed(mlir::verify(module)))
      return failMaterialization(
          failureReason,
          "selected buffering failed verification after materialization");
  }

  SelectedBufferMaterialization result;
  result.materializationRelations = std::move(currentRelations);
  result.stageCount = currentPlan->maxStage + 1;
  result.slotAllocationCount = currentPlan->slotAllocationCount;
  result.maximumSlotCount = currentPlan->maximumSlotCount;
  return result;
}

mlir::FailureOr<SelectedBufferingResult> materializeSelectedBuffering(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<SelectedBufferRequest> requests,
    StructuredMaterializationRelations materializationRelations,
    SelectedBufferMaterializationFailure *failure) {
  if (failure)
    *failure = {};

  auto fail = [&](SelectedBufferMaterializationFailureKind kind,
                  llvm::Twine detail,
                  size_t requestIndex = std::numeric_limits<size_t>::max()) {
    if (failure) {
      failure->kind = kind;
      failure->requestIndex = requestIndex;
      if (requestIndex < requests.size()) {
        failure->producerNode = requests[requestIndex].producerNode;
        failure->consumerNode = requests[requestIndex].consumerNode;
        failure->bufferCount = requests[requestIndex].bufferCount;
      }
      failure->detail = detail.str();
    }
    return mlir::failure();
  };
  if (!module || requests.empty())
    return fail(SelectedBufferMaterializationFailureKind::InvalidRequest,
                "selected buffering requires a live module and exact edge "
                "request");

  const uint32_t requestedBufferCount = requests.front().bufferCount;
  for (auto [index, request] : llvm::enumerate(requests)) {
    if (!request.producerNode || !request.consumerNode ||
        request.bufferCount < 2 ||
        request.bufferCount != requestedBufferCount ||
        (!request.requireLocalDataflow && request.messages.empty()))
      return fail(
          SelectedBufferMaterializationFailureKind::InvalidRequest,
          "selected buffering edge requests must carry two DAG nodes, one "
          "common count of at least two and actual local or Direct-DTE "
          "dataflow",
          index);
  }

  llvm::SmallVector<mlir::scf::ForOp, 8> loops;
  module->walk([&](mlir::scf::ForOp loop) {
    if (llvm::all_of(requests, [&](const SelectedBufferRequest &request) {
          return loopMayContainRequest(loop, request, materializationRelations);
        }))
      loops.push_back(loop);
  });
  if (loops.empty()) {
    return fail(
        SelectedBufferMaterializationFailureKind::NoExactLoop,
        "selected buffering found no static loop containing every exact "
        "logical-edge endpoint; requests=" +
            std::to_string(requests.size()));
  }

  SelectedBufferMaterializationFailure lastFailure;
  lastFailure.kind = SelectedBufferMaterializationFailureKind::NoExactLoop;
  lastFailure.detail =
      "selected buffering found no exact loop for the requested logical edge";
  mlir::scf::ForOp selectedLoop;
  for (mlir::scf::ForOp loop : loops) {
    std::string attemptFailure;
    SelectedBufferMaterializationFailureKind attemptKind =
        SelectedBufferMaterializationFailureKind::UnsupportedStructure;
    size_t failedRequest = std::numeric_limits<size_t>::max();
    mlir::FailureOr<SelectedBufferPlan> plan =
        buildSelectedBufferPlan(loop, &attemptFailure, &attemptKind, requests,
                                &materializationRelations, &failedRequest);
    if (mlir::succeeded(plan) &&
        mlir::failed(validateSelectedBufferRequests(
            *plan, requests, materializationRelations, &failedRequest))) {
      attemptKind = SelectedBufferMaterializationFailureKind::EdgeNotWitnessed;
      attemptFailure =
          "selected buffering loop has no staged dependency for the requested "
          "logical edge";
      plan = mlir::failure();
    }
    if (mlir::failed(plan)) {
      if (failedRequest == std::numeric_limits<size_t>::max() &&
          requests.size() == 1)
        failedRequest = 0;
      lastFailure.kind = attemptKind;
      lastFailure.requestIndex = failedRequest;
      if (failedRequest < requests.size()) {
        lastFailure.producerNode = requests[failedRequest].producerNode;
        lastFailure.consumerNode = requests[failedRequest].consumerNode;
        lastFailure.bufferCount = requests[failedRequest].bufferCount;
      }
      lastFailure.detail = std::move(attemptFailure);
      continue;
    }
    if (plan->maximumSlotCount != requestedBufferCount) {
      lastFailure.kind =
          SelectedBufferMaterializationFailureKind::MultiplicityMismatch;
      lastFailure.requestIndex = std::numeric_limits<size_t>::max();
      lastFailure.bufferCount = requestedBufferCount;
      lastFailure.detail = "actual rotating-slot multiplicity " +
                           std::to_string(plan->maximumSlotCount) +
                           " differs from structured-DAG edge selection " +
                           std::to_string(requestedBufferCount);
      continue;
    }
    selectedLoop = loop;
    break;
  }
  if (selectedLoop) {
    std::string attemptFailure;
    SelectedBufferMaterializationFailureKind attemptKind =
        SelectedBufferMaterializationFailureKind::UnsupportedStructure;
    size_t failedRequest = std::numeric_limits<size_t>::max();
    mlir::FailureOr<SelectedBufferMaterialization> materialized =
        materializeSelectedBufferCandidateInPlace(
            *module, selectedLoop, &attemptFailure, requests,
            &materializationRelations, &attemptKind, &failedRequest);
    if (mlir::failed(materialized) ||
        materialized->maximumSlotCount != requestedBufferCount) {
      if (failure) {
        failure->kind = mlir::failed(materialized)
                            ? attemptKind
                            : SelectedBufferMaterializationFailureKind::
                                  MultiplicityMismatch;
        failure->requestIndex = failedRequest;
        if (failedRequest < requests.size()) {
          failure->producerNode = requests[failedRequest].producerNode;
          failure->consumerNode = requests[failedRequest].consumerNode;
        }
        failure->bufferCount = requestedBufferCount;
        failure->detail =
            mlir::failed(materialized)
                ? std::move(attemptFailure)
                : "selected buffering plan changed while materializing the "
                  "chosen loop";
      }
      return mlir::failure();
    }
    SelectedBufferingResult result;
    result.module = std::move(module);
    result.materializationRelations =
        std::move(materialized->materializationRelations);
    result.stageCount = materialized->stageCount;
    result.slotAllocationCount = materialized->slotAllocationCount;
    result.maximumSlotCount = materialized->maximumSlotCount;
    return result;
  }
  if (failure)
    *failure = std::move(lastFailure);
  return mlir::failure();
}

} // namespace wafer::compiler::detail
