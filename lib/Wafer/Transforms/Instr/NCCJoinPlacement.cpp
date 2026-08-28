//===- NCCJoinPlacement.cpp - Current-IR required NCC joins ------------===//

#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;
namespace wafer {
#define GEN_PASS_DEF_PLACEREQUIREDNCCJOINSPASS
#define GEN_PASS_DEF_REBUILDREQUIREDNCCJOINSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace {

struct NCCOutstandingAccessSummary {
  uint32_t workers = 0;
  llvm::DenseMap<mlir::Value, uint32_t> readers;
  llvm::DenseMap<mlir::Value, uint32_t> writers;
};

static void addMask(llvm::DenseMap<mlir::Value, uint32_t> &masks,
                    mlir::Value root, uint32_t mask) {
  if (root && mask)
    masks[root] |= mask;
}

static void clearMask(llvm::DenseMap<mlir::Value, uint32_t> &masks,
                      uint32_t completed) {
  for (auto iterator = masks.begin(); iterator != masks.end();) {
    iterator->second &= ~completed;
    if (iterator->second != 0) {
      ++iterator;
      continue;
    }
    auto dead = iterator++;
    masks.erase(dead);
  }
}

static void clearSynchronizedWorkers(NCCOutstandingAccessSummary &state,
                                     uint32_t completed) {
  state.workers &= ~completed;
  clearMask(state.readers, completed);
  clearMask(state.writers, completed);
}

static void
mergeOutstandingAccessSummaries(NCCOutstandingAccessSummary &destination,
                                const NCCOutstandingAccessSummary &source) {
  destination.workers |= source.workers;
  for (const auto &entry : source.readers)
    destination.readers[entry.first] |= entry.second;
  for (const auto &entry : source.writers)
    destination.writers[entry.first] |= entry.second;
}

static bool haveEqualMasks(const llvm::DenseMap<mlir::Value, uint32_t> &lhs,
                           const llvm::DenseMap<mlir::Value, uint32_t> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(lhs, [&](const auto &entry) {
    auto found = rhs.find(entry.first);
    return found != rhs.end() && found->second == entry.second;
  });
}

static bool
haveEqualOutstandingAccessSummaries(const NCCOutstandingAccessSummary &lhs,
                                    const NCCOutstandingAccessSummary &rhs) {
  return lhs.workers == rhs.workers &&
         haveEqualMasks(lhs.readers, rhs.readers) &&
         haveEqualMasks(lhs.writers, rhs.writers);
}

static void collectAccessRoots(mlir::Value value,
                               llvm::SmallVectorImpl<mlir::Value> &roots,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;

  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArgument.getOwner();
    mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(parent);
        tileRegion && tileRegion.getBody().hasOneBlock() &&
        owner == &tileRegion.getBody().front() &&
        blockArgument.getArgNumber() < tileRegion.getInputs().size()) {
      collectAccessRoots(tileRegion.getInputs()[blockArgument.getArgNumber()],
                         roots, visited);
      return;
    }
    // Keep same-iteration SCF state variables distinct here. Expanding each
    // one independently through init/yield loses the relational fact that a
    // rotating-buffer recurrence is a permutation of disjoint allocations.
    // `areKnownDistinctRoots` proves such pairs by induction below.
    roots.push_back(value);
    return;
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *definition = result ? result.getOwner() : nullptr;
  if (!definition) {
    roots.push_back(value);
    return;
  }
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
    collectAccessRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(definition);
      tileRegion && tileRegion.getBody().hasOneBlock()) {
    auto yield = mlir::dyn_cast<TileYieldOp>(
        tileRegion.getBody().front().getTerminator());
    if (yield && result.getResultNumber() < yield.getValues().size()) {
      collectAccessRoots(yield.getValues()[result.getResultNumber()], roots,
                         visited);
      return;
    }
  }
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(definition)) {
    for (mlir::Region &region : ifOp->getRegions()) {
      if (region.empty())
        continue;
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
      if (yield && result.getResultNumber() < yield.getOperands().size())
        collectAccessRoots(yield.getOperands()[result.getResultNumber()], roots,
                           visited);
    }
    if (!roots.empty())
      return;
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(definition)) {
    unsigned resultNumber = result.getResultNumber();
    if (resultNumber < forOp.getInitArgs().size())
      collectAccessRoots(forOp.getInitArgs()[resultNumber], roots, visited);
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yield && resultNumber < yield.getOperands().size())
      collectAccessRoots(yield.getOperands()[resultNumber], roots, visited);
    if (!roots.empty())
      return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(definition);
      cast && cast.getInputs().size() == 1) {
    collectAccessRoots(cast.getInputs().front(), roots, visited);
    return;
  }
  roots.push_back(value);
}

static llvm::SmallVector<mlir::Value, 4> getAccessRoots(mlir::Value value) {
  llvm::SmallVector<mlir::Value, 4> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectAccessRoots(value, roots, visited);
  return roots;
}

static bool areKnownDistinctRootsImpl(
    mlir::Value lhs, mlir::Value rhs,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> &activePairs) {
  if (lhs == rhs)
    return false;
  if (lhs.getDefiningOp<mlir::memref::AllocOp>() &&
      rhs.getDefiningOp<mlir::memref::AllocOp>())
    return true;
  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  MemoryAttr lhsMemory = lhsType ? getWaferMemoryAttr(lhsType) : MemoryAttr{};
  MemoryAttr rhsMemory = rhsType ? getWaferMemoryAttr(rhsType) : MemoryAttr{};
  if (lhsMemory && rhsMemory && lhsMemory.getSpace() != rhsMemory.getSpace())
    return true;

  auto lhsArgument = mlir::dyn_cast<mlir::BlockArgument>(lhs);
  auto rhsArgument = mlir::dyn_cast<mlir::BlockArgument>(rhs);
  if (!lhsArgument || !rhsArgument ||
      lhsArgument.getOwner() != rhsArgument.getOwner() ||
      lhsArgument.getArgNumber() == 0 || rhsArgument.getArgNumber() == 0)
    return false;
  auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      lhsArgument.getOwner()->getParentOp());
  if (!loop || lhsArgument.getOwner() != loop.getBody())
    return false;

  auto active = llvm::find_if(activePairs, [&](const auto &pair) {
    return (pair.first == lhs && pair.second == rhs) ||
           (pair.first == rhs && pair.second == lhs);
  });
  // Coinductive backedge: every pair on the active chain has already proved
  // its distinct init state. Re-entering that pair therefore closes the
  // induction over the loop recurrence.
  if (active != activePairs.end())
    return true;

  unsigned lhsIndex = lhsArgument.getArgNumber() - 1;
  unsigned rhsIndex = rhsArgument.getArgNumber() - 1;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (lhsIndex >= loop.getInitArgs().size() ||
      rhsIndex >= loop.getInitArgs().size() || !yield ||
      lhsIndex >= yield.getOperands().size() ||
      rhsIndex >= yield.getOperands().size())
    return false;

  activePairs.push_back({lhs, rhs});
  bool distinct =
      areKnownDistinctRootsImpl(loop.getInitArgs()[lhsIndex],
                                loop.getInitArgs()[rhsIndex], activePairs) &&
      areKnownDistinctRootsImpl(yield.getOperands()[lhsIndex],
                                yield.getOperands()[rhsIndex], activePairs);
  activePairs.pop_back();
  return distinct;
}

static bool areKnownDistinctRoots(mlir::Value lhs, mlir::Value rhs) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> activePairs;
  return areKnownDistinctRootsImpl(lhs, rhs, activePairs);
}

static uint32_t
getAliasingMask(mlir::Value value,
                const llvm::DenseMap<mlir::Value, uint32_t> &masks) {
  uint32_t result = 0;
  for (mlir::Value root : getAccessRoots(value)) {
    for (const auto &entry : masks) {
      if (!areKnownDistinctRoots(root, entry.first))
        result |= entry.second;
    }
  }
  return result;
}

static llvm::SmallVector<int64_t, kNCCWorkerCount>
getParticipants(uint32_t mask) {
  llvm::SmallVector<int64_t, kNCCWorkerCount> participants;
  for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
    if (mask & (uint32_t{1} << worker))
      participants.push_back(static_cast<int64_t>(worker));
  }
  return participants;
}

static void insertNCCJoinBefore(mlir::Operation *operation, uint32_t mask,
                                NCCOutstandingAccessSummary &state) {
  mask &= state.workers;
  if (mask == 0)
    return;
  mlir::OpBuilder builder(operation);
  builder.create<SyncNCCJoinOp>(operation->getLoc(), getParticipants(mask),
                                mlir::UnitAttr{});
  clearSynchronizedWorkers(state, mask);
}

static void recordNCCIssue(mlir::Operation *operation, uint32_t workerMask,
                           NCCOutstandingAccessSummary &state) {
  state.workers |= workerMask;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances) {
    mlir::Value value = instance.getValue();
    if (!value)
      continue;
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool allocate =
        mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect());
    if (allocate)
      continue;
    for (mlir::Value root : getAccessRoots(value)) {
      if (read)
        addMask(state.readers, root, workerMask);
      else
        addMask(state.writers, root, workerMask);
    }
  }
}

static uint32_t
getExternalConflictMask(mlir::Operation *operation,
                        const NCCOutstandingAccessSummary &state,
                        bool ignoreTypedIssueResources = false) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return 0;
  // Once an NCC address lifetime has advanced through a same-worker issue
  // chain, static packing may reuse that address for a later NCC allocation.
  // Direct DTE is a different completion domain and does not participate in
  // the NCC busytable, so every still-pending NCC worker must be completed
  // before a DTE issue can become the first access to such a reused address.
  // The exact DTE wait remains independent and never completes NCC work.
  uint32_t conflicts =
      mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation) ? state.workers : 0;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  bool ignoreTypedResources =
      ignoreTypedIssueResources ||
      mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation);
  for (const auto &instance : instances) {
    mlir::Value value = instance.getValue();
    if (!value) {
      // An NCC issue's custom Wafer resource effects describe typed engine
      // and memory-space occupancy. Its value-associated effects below are
      // the address hazard contract, so the custom resource must not turn two
      // proven-distinct roots into a blocking cross-worker join. A Direct DTE
      // issue has the same property: its value-associated buffer
      // effect is the NCC visibility boundary, while its rootless
      // communication resource describes transport occupancy. The exact DTE
      // wait carries no NCC buffer observation and must not become an implicit
      // NCC join. Unknown and other synchronous observers retain the
      // conservative all-pending conflict.
      if (ignoreTypedResources &&
          instance.getResource() != mlir::SideEffects::DefaultResource::get())
        continue;
      if (!mlir::isa<mlir::MemoryEffects::Allocate, mlir::MemoryEffects::Free>(
              instance.getEffect()))
        conflicts |= state.workers;
      continue;
    }
    // alloc/free delimit compiler-owned lifetime but do not execute a Kcore or
    // host memory access. Actual reuse is ordered by the next typed issue:
    // same-worker ranges use busytable order and cross-worker conflicts join
    // immediately before that issue.
    if (mlir::isa<mlir::MemoryEffects::Allocate, mlir::MemoryEffects::Free>(
            instance.getEffect()))
      continue;
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    conflicts |= getAliasingMask(value, state.writers);
    if (!read)
      conflicts |= getAliasingMask(value, state.readers);
  }
  return conflicts;
}

/// Resolve a scalar carried into a tile residency region back to the enclosing
/// SSA value. Region partitioning may turn a previously local loop bound into
/// a region input; that transport does not make a static bound dynamic and
/// must not change required-join legality.
static mlir::Value resolveTileRegionScalarForwarding(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = argument.getOwner();
      auto region =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!region || region.getBody().empty() ||
          owner != &region.getBody().front() ||
          argument.getArgNumber() >= region.getInputs().size())
        break;
      value = region.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto region = result ? mlir::dyn_cast<TileRegionOp>(result.getOwner())
                         : TileRegionOp{};
    if (!region || region.getBody().empty())
      break;
    auto yield =
        mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
    if (!yield || result.getResultNumber() >= yield.getValues().size())
      break;
    value = yield.getValues()[result.getResultNumber()];
  }
  return value;
}

class RequiredNCCJoinPlacement {
public:
  mlir::LogicalResult run(mlir::func::FuncOp function) {
    if (function.isExternal())
      return mlir::success();
    if (!function.getBody().hasOneBlock()) {
      bool hasIssue = false;
      function.walk([&](WaferNCCIssueOpInterface) { hasIssue = true; });
      if (!hasIssue)
        return mlir::success();
      return function.emitError()
             << "required_ncc_join_failure: exact NCC synchronization "
                "placement requires structured single-block function IR";
    }
    NCCOutstandingAccessSummary state;
    return processBlock(function.getBody().front(), state);
  }

  mlir::LogicalResult run(TileRegionOp tileRegion) {
    NCCOutstandingAccessSummary state;
    return processOperation(tileRegion.getOperation(), state);
  }

private:
  mlir::LogicalResult processBlock(mlir::Block &block,
                                   NCCOutstandingAccessSummary &state) {
    for (auto iterator = block.begin(); iterator != block.end();) {
      mlir::Operation *operation = &*iterator++;
      if (mlir::failed(processOperation(operation, state)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult processOperation(mlir::Operation *operation,
                                       NCCOutstandingAccessSummary &state) {
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(operation)) {
      if (!tileRegion.getBody().hasOneBlock())
        return tileRegion.emitError()
               << "required_ncc_join_failure: outstanding NCC access "
                  "placement requires a single-block wafer.tile.region";
      if (mlir::failed(processBlock(tileRegion.getBody().front(), state)))
        return mlir::failure();
      return mlir::success();
    }

    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      NCCOutstandingAccessSummary thenState = state;
      if (mlir::failed(processBlock(ifOp.getThenRegion().front(), thenState)))
        return mlir::failure();
      NCCOutstandingAccessSummary elseState = state;
      if (!ifOp.getElseRegion().empty() &&
          mlir::failed(processBlock(ifOp.getElseRegion().front(), elseState)))
        return mlir::failure();
      state = std::move(thenState);
      mergeOutstandingAccessSummaries(state, elseState);
      return mlir::success();
    }

    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      auto getConstantIndex = [](mlir::Value value) -> std::optional<int64_t> {
        value = resolveTileRegionScalarForwarding(value);
        if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
          return constant;
        if (!value || !value.getType().isIndex())
          return std::nullopt;
        mlir::FailureOr<int64_t> constant =
            mlir::ValueBoundsConstraintSet::computeConstantBound(
                mlir::presburger::BoundType::EQ,
                mlir::ValueBoundsConstraintSet::Variable(value));
        return mlir::succeeded(constant) ? std::optional<int64_t>(*constant)
                                         : std::nullopt;
      };
      std::optional<int64_t> lower = getConstantIndex(forOp.getLowerBound());
      std::optional<int64_t> upper = getConstantIndex(forOp.getUpperBound());
      std::optional<int64_t> step = getConstantIndex(forOp.getStep());
      if (lower && upper && step && *step > 0 && *lower >= *upper)
        return mlir::success();

      NCCOutstandingAccessSummary bodyState = state;
      if (mlir::failed(processBlock(*forOp.getBody(), bodyState)))
        return mlir::failure();
      bool guaranteedToExecute =
          lower && upper && step && *step > 0 && *lower < *upper;
      std::optional<__int128> tripCount;
      if (guaranteedToExecute) {
        __int128 span =
            static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
        tripCount = (span + static_cast<__int128>(*step) - 1) /
                    static_cast<__int128>(*step);
      }
      if (tripCount && *tripCount == 1) {
        state = std::move(bodyState);
        return mlir::success();
      }

      // A later iteration observes the prior iteration's pending access state.
      // Reprocess to a finite typed-worker fixed point so a cross-worker
      // loop-carried RAW/WAR/WAW cut is explicit in the loop body, while a
      // same-worker chain remains a join-free ordered issue stream.
      const unsigned convergenceLimit =
          static_cast<unsigned>(kNCCWorkerCount) *
              (static_cast<unsigned>(std::distance(forOp.getBody()->begin(),
                                                   forOp.getBody()->end())) +
               1) +
          1;
      bool converged = false;
      for (unsigned iteration = 0; iteration < convergenceLimit; ++iteration) {
        // This is a sequential backedge, not an alternative control-flow
        // merge. `bodyState` already contains the entry state followed by one
        // complete iteration; feed that exact pending state into the next.
        // Optional same-worker issues remain ordered on every path where they
        // occur, and paths without an issue require no completion.
        NCCOutstandingAccessSummary nextState = bodyState;
        if (mlir::failed(processBlock(*forOp.getBody(), nextState)))
          return mlir::failure();
        if (haveEqualOutstandingAccessSummaries(nextState, bodyState)) {
          converged = true;
          bodyState = std::move(nextState);
          break;
        }
        bodyState = std::move(nextState);
      }
      if (!converged)
        return forOp.emitError()
               << "required_ncc_join_failure: NCC loop backedge "
                  "access state did not reach a finite fixed point";
      if (guaranteedToExecute)
        state = std::move(bodyState);
      else
        // Preserve the zero-trip path while carrying every pending access
        // from one-or-more iterations to the first actual observer/terminal.
        mergeOutstandingAccessSummaries(state, bodyState);
      return mlir::success();
    }

    if (mlir::isa<mlir::func::ReturnOp, mlir::async::YieldOp>(operation)) {
      insertNCCJoinBefore(operation, state.workers, state);
      return mlir::success();
    }
    if (mlir::isa<TileYieldOp, mlir::scf::YieldOp>(operation))
      return mlir::success();

    // These operations only change the tensor/memref SSA view of one storage
    // object. They do not execute a Kcore/host read and therefore cannot turn
    // a pending NCC issue into a completion boundary.
    if (mlir::isa<mlir::bufferization::ToMemrefOp,
                  mlir::bufferization::ToTensorOp>(operation))
      return mlir::success();

    NCCOperationCompletion contract = getNCCOperationCompletion(operation);
    if (contract.issueWorker) {
      uint32_t worker = static_cast<uint32_t>(*contract.issueWorker);
      if (worker >= kNCCWorkerCount)
        return operation->emitError()
               << "required_ncc_join_failure: NCC issue worker is "
                  "outside the typed worker domain";
      uint32_t workerMask = uint32_t{1} << worker;
      // The target busytable orders an NCC worker's own RAW/WAR/WAW chain,
      // including compiler-managed store/reload and offset reuse. Complete
      // only prior conflicting workers before issuing the new access;
      // operation kind and residency structure are not completion events.
      uint32_t crossWorkerConflicts =
          getExternalConflictMask(operation, state,
                                  /*ignoreTypedIssueResources=*/true) &
          ~workerMask;
      insertNCCJoinBefore(operation, crossWorkerConflicts, state);
      recordNCCIssue(operation, workerMask, state);
    }

    if (contract.kind == NCCCompletionKind::ParticipantJoin) {
      uint32_t requested = contract.participantMask & kAllNCCWorkersMask;
      if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
        uint32_t effective = requested & state.workers;
        if (effective == 0) {
          join.erase();
          return mlir::success();
        }
        if (effective != requested)
          join.setParticipants(getParticipants(effective));
        clearSynchronizedWorkers(state, effective);
        return mlir::success();
      }
      clearSynchronizedWorkers(state, requested);
      return mlir::success();
    }
    if (contract.kind == NCCCompletionKind::SynchronousWriteback) {
      clearSynchronizedWorkers(state, contract.participantMask);
      return mlir::success();
    }
    if (contract.kind == NCCCompletionKind::OrderedAsynchronousIssue)
      return mlir::success();

    // A Direct DTE wait observes only its own typed event. It neither consumes
    // nor joins pending NCC worker issues, so independent NCC work
    // may remain in flight while the transport is awaited.
    if (mlir::isa<InstrDTEWaitOp>(operation))
      return mlir::success();

    // A Direct DTE issue observes only its explicit SPM buffer. Ignore the
    // rootless communication-resource effect here and complete exactly the
    // NCC workers whose value-associated accesses conflict with that buffer.
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation)) {
      uint32_t conflicts = getExternalConflictMask(
          operation, state, /*ignoreTypedIssueResources=*/true);
      insertNCCJoinBefore(operation, conflicts, state);
      return mlir::success();
    }

    if (mlir::isa<mlir::CallOpInterface>(operation)) {
      insertNCCJoinBefore(operation, state.workers, state);
      return mlir::success();
    }

    if (!mlir::isa<mlir::MemoryEffectOpInterface>(operation) &&
        !mlir::isMemoryEffectFree(operation)) {
      if (state.workers == 0)
        return mlir::success();
      return operation->emitError()
             << "required_ncc_join_failure: operation without a typed "
                "memory-effect contract may observe pending NCC work";
    }

    uint32_t conflicts = getExternalConflictMask(operation, state);
    insertNCCJoinBefore(operation, conflicts, state);
    return mlir::success();
  }
};

static void eraseDerivedNCCJoins(mlir::Operation *root) {
  llvm::SmallVector<SyncNCCJoinOp, 16> joins;
  root->walk([&](SyncNCCJoinOp join) {
    if (!join.getScheduleBoundary())
      joins.push_back(join);
  });
  for (SyncNCCJoinOp join : llvm::reverse(joins))
    join.erase();
}

static mlir::LogicalResult
normalizeRequiredNCCJoinsInPlace(mlir::func::FuncOp function,
                                 bool rebuildDerivedJoins) {
  if (rebuildDerivedJoins)
    eraseDerivedNCCJoins(function.getOperation());
  return RequiredNCCJoinPlacement().run(function);
}

static mlir::LogicalResult
normalizeRequiredNCCJoinsInPlace(mlir::ModuleOp module,
                                 bool rebuildDerivedJoins) {
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    if (mlir::failed(
            normalizeRequiredNCCJoinsInPlace(function, rebuildDerivedJoins)))
      return mlir::failure();
  return mlir::success();
}

struct PlaceRequiredNCCJoinsPass
    : public wafer::impl::PlaceRequiredNCCJoinsPassBase<
          PlaceRequiredNCCJoinsPass> {
  using wafer::impl::PlaceRequiredNCCJoinsPassBase<
      PlaceRequiredNCCJoinsPass>::PlaceRequiredNCCJoinsPassBase;

  void runOnOperation() final {
    unsigned before = 0;
    getOperation().walk([&](SyncNCCJoinOp) { ++before; });
    if (mlir::succeeded(normalizeRequiredNCCJoinsInPlace(
            getOperation(), /*rebuildDerivedJoins=*/false))) {
      unsigned after = 0;
      getOperation().walk([&](SyncNCCJoinOp) { ++after; });
      if (after > before)
        numRequiredJoinsInserted += after - before;
      return;
    }
    signalPassFailure();
  }
};

struct RebuildRequiredNCCJoinsPass
    : public wafer::impl::RebuildRequiredNCCJoinsPassBase<
          RebuildRequiredNCCJoinsPass> {
  using wafer::impl::RebuildRequiredNCCJoinsPassBase<
      RebuildRequiredNCCJoinsPass>::RebuildRequiredNCCJoinsPassBase;

  void runOnOperation() final {
    unsigned before = 0;
    getOperation().walk([&](SyncNCCJoinOp) { ++before; });
    if (mlir::succeeded(normalizeRequiredNCCJoinsInPlace(
            getOperation(), /*rebuildDerivedJoins=*/true))) {
      unsigned after = 0;
      getOperation().walk([&](SyncNCCJoinOp) { ++after; });
      numDerivedJoinsRemoved += before;
      numRequiredJoinsInserted += after;
      return;
    }
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult wafer::placeRequiredNCCJoins(mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr", "required-ncc-join-placement");
  if (mlir::failed(
          normalizeRequiredNCCJoinsInPlace(function,
                                           /*rebuildDerivedJoins=*/false))) {
    timing.markFailed();
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult wafer::placeRequiredNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(module,
                                          /*rebuildDerivedJoins=*/false);
}

mlir::LogicalResult
wafer::rebuildRequiredNCCJoins(mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr", "fresh-ncc-join-rebuild");
  if (mlir::failed(
          normalizeRequiredNCCJoinsInPlace(function,
                                           /*rebuildDerivedJoins=*/true))) {
    timing.markFailed();
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult wafer::rebuildRequiredNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(module, /*rebuildDerivedJoins=*/true);
}

mlir::LogicalResult
wafer::rebuildRequiredNCCJoinsForIsolatedTileRegion(TileRegionOp tileRegion) {
  if (!tileRegion)
    return mlir::failure();
  eraseDerivedNCCJoins(tileRegion.getOperation());
  return RequiredNCCJoinPlacement().run(tileRegion);
}

mlir::LogicalResult wafer::detail::placeRequiredNCCJoinsInPrivateFunction(
    mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(function,
                                          /*rebuildDerivedJoins=*/false);
}
