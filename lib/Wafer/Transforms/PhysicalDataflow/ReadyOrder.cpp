//===- ReadyOrder.cpp - Actual instruction ready ordering ----------------===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <limits>

namespace wafer {
namespace {

struct BufferAccess {
  mlir::Value value;
  bool write = false;
};

static mlir::Value getAccessBase(mlir::Value value) {
  while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
             value.getDefiningOp()))
    value = view.getViewSource();
  return value;
}

static bool isReadyOrderOperation(mlir::Operation *operation) {
  return mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation);
}

static bool isFailClosedCompletionBoundary(mlir::Operation *operation) {
  return classifyNCCSynchronizationBehavior(operation) ==
         NCCSynchronizationBehavior::SynchronousWriteback;
}

static bool waitMayReleaseDTESender(mlir::Operation *operation) {
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation);
  if (!wait)
    return false;
  return llvm::any_of(wait.getTokens(), [](mlir::Value token) {
    return token.getDefiningOp<InstrDTESendOp>() ||
           mlir::isa<mlir::BlockArgument>(token);
  });
}

static bool canReorderCompletionDomains(mlir::Operation *lhs,
                                        mlir::Operation *rhs) {
  // The current Direct-DTE ABI owns one sender slot. A send's exact
  // wait is therefore also a typed resource release: another send may not
  // move above it even when the two payload buffers are disjoint.
  if ((waitMayReleaseDTESender(lhs) && mlir::isa<InstrDTESendOp>(rhs)) ||
      (mlir::isa<InstrDTESendOp>(lhs) && waitMayReleaseDTESender(rhs)))
    return false;

  NCCSynchronizationContract lhsContract = getNCCSynchronizationContract(lhs);
  NCCSynchronizationContract rhsContract = getNCCSynchronizationContract(rhs);
  if (lhsContract.behavior ==
          NCCSynchronizationBehavior::SynchronousWriteback ||
      rhsContract.behavior == NCCSynchronizationBehavior::SynchronousWriteback)
    return false;

  auto joinAllowsIssue = [](const NCCSynchronizationContract &join,
                            const NCCSynchronizationContract &issue) {
    if (join.behavior != NCCSynchronizationBehavior::ParticipantJoin)
      return true;
    if (join.participantMask == 0 ||
        (join.participantMask & ~kAllNCCWorkersMask) != 0)
      return false;
    if (issue.behavior != NCCSynchronizationBehavior::OrderedAsynchronousIssue)
      return true;
    if (!issue.issueWorker)
      return false;
    uint32_t worker = static_cast<uint32_t>(*issue.issueWorker);
    return worker < kNCCWorkerCount &&
           (join.participantMask & (uint32_t{1} << worker)) == 0;
  };
  return joinAllowsIssue(lhsContract, rhsContract) &&
         joinAllowsIssue(rhsContract, lhsContract);
}

static void
collectBufferAccesses(mlir::Operation *operation,
                      llvm::SmallVectorImpl<BufferAccess> &accesses) {
  auto recordAccess = [&](mlir::Value value, bool write) {
    value = getAccessBase(value);
    auto found = llvm::find_if(accesses, [&](const BufferAccess &access) {
      return access.value == value;
    });
    if (found == accesses.end())
      accesses.push_back({value, write});
    else
      found->write |= write;
  };

  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (interface) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
    interface.getEffects(effects);
    for (const auto &effect : effects) {
      mlir::Value value = effect.getValue();
      if (!value)
        continue;
      bool read = llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
      // Write, allocate, free, and any future non-read effect are mutating for
      // buffer ordering.  In particular, a DTE wait must not cross a standard
      // memref deallocation of its in-flight buffer.
      recordAccess(value, /*write=*/!read);
    }
  }

  // A Direct DTE issue only starts an asynchronous buffer access.  Recover
  // the in-flight access from the wait's SSA tokens so ready ordering cannot
  // move a receive consumer, or a send-source overwrite, before completion.
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation);
  if (!wait)
    return;
  for (mlir::Value token : wait.getTokens()) {
    if (auto send = token.getDefiningOp<InstrDTESendOp>())
      recordAccess(send.getBuffer(), /*write=*/false);
    else if (auto recv = token.getDefiningOp<InstrDTERecvOp>())
      recordAccess(recv.getBuffer(), /*write=*/true);
  }
}

static unsigned getReadyPriority(mlir::Operation *operation) {
  // Start Direct DTE as soon as its source/destination is ready, but defer its
  // exact completion wait behind independent NCC work. Buffer hazards and the
  // token edge still prevent consumers or reuse from crossing the wait.
  if (mlir::isa<InstrDTEWaitOp>(operation))
    return 3;
  if (mlir::isa<SyncNCCJoinOp>(operation))
    return 4;
  auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
  if (!instruction)
    return 3;
  switch (instruction.getInstructionFamily()) {
  case InstrFamily::RDMA:
  case InstrFamily::WDMA:
  case InstrFamily::TDMA:
    return 0;
  case InstrFamily::DTE:
    return 1;
  case InstrFamily::CT:
  case InstrFamily::NE:
    return 2;
  }
  llvm_unreachable("unknown instruction family");
}

static unsigned scheduleRun(llvm::ArrayRef<mlir::Operation *> operations) {
  if (operations.size() < 2)
    return 0;

  const unsigned count = operations.size();
  llvm::DenseMap<mlir::Operation *, unsigned> indices;
  for (auto [index, operation] : llvm::enumerate(operations))
    indices.try_emplace(operation, static_cast<unsigned>(index));

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> successors(count);
  llvm::SmallVector<unsigned, 16> indegrees(count, 0);
  auto addEdge = [&](unsigned from, unsigned to) {
    if (from == to || llvm::is_contained(successors[from], to))
      return;
    successors[from].push_back(to);
    ++indegrees[to];
  };

  // A normal-profile rank block has four receiver FSMs. Preserve the source
  // receiver order, but make the fifth and every later receive wait for the
  // completion that frees the oldest slot. This is a scheduling resource
  // edge, independent of buffer aliasing and message shape.
  struct DTEReceiverLifetime {
    unsigned issue = 0;
    unsigned completion = 0;
  };
  llvm::SmallVector<DTEReceiverLifetime, 8> receiverLifetimes;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation);
    if (!recv || !recv.getToken().hasOneUse())
      continue;
    auto wait =
        mlir::dyn_cast<InstrDTEWaitOp>(*recv.getToken().getUsers().begin());
    auto waitIndex = wait ? indices.find(wait.getOperation()) : indices.end();
    if (!wait || waitIndex == indices.end() ||
        static_cast<unsigned>(index) >= waitIndex->second)
      return 0;
    receiverLifetimes.push_back(
        {static_cast<unsigned>(index), waitIndex->second});
  }
  for (unsigned index = 4; index < receiverLifetimes.size(); ++index)
    addEdge(receiverLifetimes[index - 4].completion,
            receiverLifetimes[index].issue);

  llvm::DenseMap<mlir::Value, unsigned> lastWriters;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned, 4>> readers;
  std::array<llvm::SmallVector<unsigned, 4>, kNCCWorkerCount>
      pendingWorkerIssues;
  std::array<std::optional<unsigned>, kNCCWorkerCount> lastWorkerCompletions;
  std::optional<unsigned> lastFailClosedCompletion;
  std::optional<unsigned> pendingDTESenderIssue;
  std::optional<unsigned> lastDTESenderCompletion;
  for (unsigned index = 0; index < count; ++index) {
    mlir::Operation *operation = operations[index];
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = indices.find(definition);
      if (found != indices.end())
        addEdge(found->second, index);
    }
    if (lastFailClosedCompletion)
      addEdge(*lastFailClosedCompletion, index);

    if (mlir::isa<InstrDTESendOp>(operation)) {
      if (pendingDTESenderIssue)
        return 0;
      if (lastDTESenderCompletion)
        addEdge(*lastDTESenderCompletion, index);
      pendingDTESenderIssue = index;
    }
    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
      bool completesPendingSender =
          pendingDTESenderIssue &&
          llvm::any_of(wait.getTokens(), [&](mlir::Value token) {
            return token.getDefiningOp() == operations[*pendingDTESenderIssue];
          });
      if (completesPendingSender) {
        addEdge(*pendingDTESenderIssue, index);
        pendingDTESenderIssue.reset();
        lastDTESenderCompletion = index;
      } else if (!pendingDTESenderIssue && waitMayReleaseDTESender(operation)) {
        lastDTESenderCompletion = index;
      }
    }

    NCCSynchronizationContract contract = getNCCSynchronizationContract(operation);
    std::optional<unsigned> issueWorker;
    if (contract.issueWorker) {
      unsigned worker = static_cast<unsigned>(*contract.issueWorker);
      if (worker >= kNCCWorkerCount)
        return 0;
      issueWorker = worker;
      if (lastWorkerCompletions[worker])
        addEdge(*lastWorkerCompletions[worker], index);
    }

    bool isCompletion =
        contract.behavior == NCCSynchronizationBehavior::ParticipantJoin ||
        contract.behavior == NCCSynchronizationBehavior::SynchronousWriteback;
    if (isCompletion) {
      if (contract.participantMask == 0 ||
          (contract.participantMask & ~kAllNCCWorkersMask) != 0)
        return 0;
      for (unsigned worker = 0; worker < kNCCWorkerCount; ++worker) {
        if ((contract.participantMask & (uint32_t{1} << worker)) == 0)
          continue;
        for (unsigned predecessor : pendingWorkerIssues[worker])
          addEdge(predecessor, index);
        pendingWorkerIssues[worker].clear();
        lastWorkerCompletions[worker] = index;
      }
    }
    if (contract.behavior == NCCSynchronizationBehavior::SynchronousWriteback) {
      for (unsigned predecessor = 0; predecessor < index; ++predecessor)
        addEdge(predecessor, index);
      lastFailClosedCompletion = index;
    }

    llvm::SmallVector<BufferAccess, 4> accesses;
    collectBufferAccesses(operation, accesses);
    for (const BufferAccess &access : accesses) {
      auto writer = lastWriters.find(access.value);
      if (writer != lastWriters.end())
        addEdge(writer->second, index);
      if (!access.write) {
        readers[access.value].push_back(index);
        continue;
      }
      auto activeReaders = readers.find(access.value);
      if (activeReaders != readers.end()) {
        for (unsigned reader : activeReaders->second)
          addEdge(reader, index);
        activeReaders->second.clear();
      }
      lastWriters[access.value] = index;
    }

    if (contract.behavior == NCCSynchronizationBehavior::OrderedAsynchronousIssue) {
      if (!issueWorker)
        return 0;
      pendingWorkerIssues[*issueWorker].push_back(index);
    }
  }

  llvm::SmallVector<unsigned, 16> order;
  llvm::SmallVector<bool, 16> emitted(count, false);
  while (order.size() != count) {
    unsigned selected = count;
    unsigned selectedPriority = std::numeric_limits<unsigned>::max();
    for (unsigned index = 0; index < count; ++index) {
      if (emitted[index] || indegrees[index] != 0)
        continue;
      unsigned priority = getReadyPriority(operations[index]);
      if (selected == count || priority < selectedPriority) {
        selected = index;
        selectedPriority = priority;
      }
    }
    if (selected == count)
      return 0;
    emitted[selected] = true;
    order.push_back(selected);
    for (unsigned successor : successors[selected])
      --indegrees[successor];
  }

  unsigned moved = 0;
  for (auto [position, originalIndex] : llvm::enumerate(order))
    moved += position != originalIndex;
  if (moved == 0)
    return 0;

  mlir::Block *block = operations.front()->getBlock();
  auto insertionPoint = operations.back()->getIterator();
  ++insertionPoint;
  for (unsigned originalIndex : order)
    operations[originalIndex]->moveBefore(block, insertionPoint);
  return moved;
}

static bool
isMutatingEffect(const mlir::MemoryEffects::EffectInstance &effect) {
  return !llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
}

static bool areKnownDistinctValues(mlir::Value lhs, mlir::Value rhs) {
  lhs = getAccessBase(lhs);
  rhs = getAccessBase(rhs);
  if (lhs == rhs)
    return false;
  // Each memref.alloc result denotes a fresh allocation. This is the only
  // non-alias proof needed for crossing the setup emitted by the production
  // tile/instruction lowering; all other different values fail closed.
  return lhs.getDefiningOp<mlir::memref::AllocOp>() ||
         rhs.getDefiningOp<mlir::memref::AllocOp>();
}

static bool effectsMayConflict(const mlir::MemoryEffects::EffectInstance &lhs,
                               const mlir::MemoryEffects::EffectInstance &rhs) {
  if (!isMutatingEffect(lhs) && !isMutatingEffect(rhs))
    return false;
  if (lhs.getResource() != rhs.getResource())
    return false;
  mlir::Value lhsValue = lhs.getValue();
  mlir::Value rhsValue = rhs.getValue();
  if (lhsValue)
    lhsValue = getAccessBase(lhsValue);
  if (rhsValue)
    rhsValue = getAccessBase(rhsValue);
  if (lhsValue && rhsValue && areKnownDistinctValues(lhsValue, rhsValue))
    return false;
  return true;
}

static bool canReorderEffects(mlir::Operation *lhs, mlir::Operation *rhs) {
  // Compare concrete buffer effects for every operation pair, not only two
  // Wafer instructions. scheduleWindow may move an instruction across
  // standard memref operations that sit between two ready-order operations.
  llvm::SmallVector<BufferAccess, 4> lhsAccesses;
  llvm::SmallVector<BufferAccess, 4> rhsAccesses;
  collectBufferAccesses(lhs, lhsAccesses);
  collectBufferAccesses(rhs, rhsAccesses);
  for (const BufferAccess &lhsAccess : lhsAccesses)
    for (const BufferAccess &rhsAccess : rhsAccesses)
      if (lhsAccess.value == rhsAccess.value &&
          (lhsAccess.write || rhsAccess.write))
        return false;

  if (mlir::isa<WaferInstructionOpInterface>(lhs) &&
      mlir::isa<WaferInstructionOpInterface>(rhs))
    return true;
  auto lhsEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(lhs);
  auto rhsEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(rhs);
  if (!lhsEffects || !rhsEffects)
    return mlir::isMemoryEffectFree(lhs) && mlir::isMemoryEffectFree(rhs);
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> lhsInstances;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> rhsInstances;
  lhsEffects.getEffects(lhsInstances);
  rhsEffects.getEffects(rhsInstances);
  for (const auto &lhsEffect : lhsInstances)
    for (const auto &rhsEffect : rhsInstances)
      if (effectsMayConflict(lhsEffect, rhsEffect))
        return false;
  return true;
}

static bool canMoveAfter(mlir::Operation *operation,
                         mlir::Operation *destination) {
  if (operation->getBlock() != destination->getBlock() ||
      operation == destination || operation->getNumRegions() != 0)
    return false;
  llvm::DenseSet<mlir::Value> results;
  for (mlir::Value result : operation->getResults())
    results.insert(result);
  for (mlir::Operation *crossed = operation->getNextNode(); crossed;
       crossed = crossed->getNextNode()) {
    if (crossed->hasTrait<mlir::OpTrait::IsTerminator>() ||
        crossed->getNumRegions() != 0 ||
        !canReorderCompletionDomains(operation, crossed) ||
        !canReorderEffects(operation, crossed))
      return false;
    if (llvm::any_of(crossed->getOperands(), [&](mlir::Value operand) {
          return results.contains(operand);
        }))
      return false;
    if (crossed == destination)
      return true;
  }
  return false;
}

static unsigned scheduleWindow(llvm::ArrayRef<mlir::Operation *> operations) {
  if (operations.size() < 2)
    return 0;
  llvm::SmallVector<mlir::Operation *, 16> order(operations);
  unsigned moved = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned index = 0; index + 1 < order.size(); ++index) {
      mlir::Operation *earlier = order[index];
      mlir::Operation *later = order[index + 1];
      if (getReadyPriority(earlier) <= getReadyPriority(later) ||
          !canMoveAfter(earlier, later))
        continue;
      earlier->moveAfter(later);
      std::swap(order[index], order[index + 1]);
      ++moved;
      changed = true;
    }
  }
  return moved;
}

static unsigned scheduleBlock(mlir::Block &block) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization-phase", "scheduleIndependentInstructionsByReadyOrder",
      "schedule-block");
  unsigned moved = 0;
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 16>, 4> windows(1);
  for (mlir::Operation &operation : block) {
    if (isFailClosedCompletionBoundary(&operation)) {
      windows.emplace_back();
      continue;
    }
    if (isReadyOrderOperation(&operation))
      windows.back().push_back(&operation);
  }
  // A full fence-bounded window permits a lower-priority instruction to move
  // later across effect-independent cast/allocation setup.
  for (llvm::ArrayRef<mlir::Operation *> window : windows)
    moved += scheduleWindow(window);

  // Keep the contiguous DAG scheduler as the stronger path for runs where
  // multiple legal ready nodes can be selected without crossing setup ops.
  llvm::SmallVector<mlir::Operation *, 16> run;
  auto flushRun = [&] {
    moved += scheduleRun(run);
    run.clear();
  };
  for (mlir::Operation &operation : block) {
    if (isReadyOrderOperation(&operation))
      run.push_back(&operation);
    else
      flushRun();
  }
  flushRun();
  return moved;
}

static unsigned scheduleRegion(mlir::Region &region) {
  unsigned moved = 0;
  for (mlir::Block &block : region) {
    for (mlir::Operation &operation : block)
      for (mlir::Region &nested : operation.getRegions())
        moved += scheduleRegion(nested);
    moved += scheduleBlock(block);
  }
  return moved;
}

} // namespace

unsigned scheduleIndependentInstructionsByReadyOrder(mlir::Operation *scope) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "ready-order-scheduling",
      "scheduleIndependentInstructionsByReadyOrder");
  if (!scope)
    return 0;
  unsigned moved = 0;
  for (mlir::Region &region : scope->getRegions())
    moved += scheduleRegion(region);
  return moved;
}

} // namespace wafer
