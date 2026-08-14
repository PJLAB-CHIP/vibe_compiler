//===- WaferTileRegionToInstr.cpp - Tile-region to instr conversion ------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "Internal.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace wafer {
#define GEN_PASS_DEF_CONVERTTILEREGIONTOINSTRPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace {

static void configureTileRegionToInstrTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncNCCJoinOp,
                    InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrBit2FpOp, InstrMaskMoveOp,
                    InstrReduceOp, InstrConvertOp, InstrGemmOp, InstrConvOp,
                    InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>();
  target.addDynamicallyLegalOp<InstrTDMADataMoveOp>([](InstrTDMADataMoveOp op) {
    return !requiresGatherScatterMaterialization(op.getKindAttr().getValue());
  });
  target.addDynamicallyLegalOp<mlir::async::AwaitOp>(
      [](mlir::async::AwaitOp op) {
        mlir::Value operand = op.getOperand();
        return !operand.getDefiningOp<CommPeerSendOp>() &&
               !operand.getDefiningOp<CommPeerRecvOp>() &&
               !operand.getDefiningOp<InstrDTESendOp>() &&
               !operand.getDefiningOp<InstrDTERecvOp>();
      });
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *operation) {
    return !mlir::isa<WaferTileDataflowOpInterface>(operation);
  });
}

static void populateTileRegionToInstrPatterns(
    mlir::RewritePatternSet &patterns) {
  // Match failures stay inside the PatternRewriter transaction. The public
  // compiler adapter reports a deterministic stage-level failure after the
  // conversion driver returns; it does not expose the last attempted pattern
  // through rollback-external mutable state.
  populateMovementLoweringPatterns(patterns, /*failureReason=*/nullptr);
  populateComputeLoweringPatterns(patterns, /*failureReason=*/nullptr);
  populateViewReshapeLoweringPattern(patterns, /*failureReason=*/nullptr);
  populateFillLoweringPattern(patterns);
  populatePeerLoweringPatterns(patterns, /*failureReason=*/nullptr);
}

/// Remove a private fill whose destination has no reader.  Constant folding
/// of tensor-level select expressions can make the predicate buffer dead
/// after the tile body has already materialized its splat.  Keeping that
/// write would turn a dead i1 value into a real target TDMA command, where the
/// hardware profile correctly rejects the unproven BOOL encoding.
static void eraseDeadPrivateFills(mlir::ModuleOp module) {
  llvm::SmallVector<InstrFillOp, 4> deadFills;
  module.walk([&](InstrFillOp fill) {
    mlir::Value dest = fill.getDest();
    if (dest.hasOneUse() && dest.getDefiningOp<mlir::memref::AllocOp>())
      deadFills.push_back(fill);
  });
  for (InstrFillOp fill : deadFills) {
    mlir::Value dest = fill.getDest();
    mlir::Value scalar = fill.getValue();
    auto alloc = dest.getDefiningOp<mlir::memref::AllocOp>();
    fill->erase();
    alloc->erase();
    if (auto constant = scalar.getDefiningOp<mlir::arith::ConstantOp>();
        constant && constant->use_empty())
      constant->erase();
  }
}

struct PendingNCCState {
  uint32_t workers = 0;
  llvm::DenseMap<mlir::Value, uint32_t> readers;
  llvm::DenseMap<mlir::Value, uint32_t> writers;
  std::array<mlir::Operation *, kNCCWorkerCount> latestIssues{};
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

static void clearCompletedWorkers(PendingNCCState &state, uint32_t completed) {
  state.workers &= ~completed;
  clearMask(state.readers, completed);
  clearMask(state.writers, completed);
  for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker)
    if ((completed & (uint32_t{1} << worker)) != 0)
      state.latestIssues[worker] = nullptr;
}

static void mergePendingState(PendingNCCState &destination,
                              const PendingNCCState &source) {
  uint32_t destinationWorkers = destination.workers;
  for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
    uint32_t mask = uint32_t{1} << worker;
    bool destinationHasWorker = (destinationWorkers & mask) != 0;
    bool sourceHasWorker = (source.workers & mask) != 0;
    if (destinationHasWorker != sourceHasWorker) {
      // This is an alternative-path frontier: one path can reach the merge
      // without the issue.  Retaining the other path's operation pointer
      // would falsely make that issue an unconditional same-worker successor
      // on a surrounding loop backedge.
      destination.latestIssues[worker] = nullptr;
      continue;
    }
    if (destinationHasWorker && sourceHasWorker &&
        destination.latestIssues[worker] != source.latestIssues[worker])
      destination.latestIssues[worker] = nullptr;
  }
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

static bool haveEqualPendingState(const PendingNCCState &lhs,
                                  const PendingNCCState &rhs) {
  return lhs.workers == rhs.workers &&
         haveEqualMasks(lhs.readers, rhs.readers) &&
         haveEqualMasks(lhs.writers, rhs.writers) &&
         lhs.latestIssues == rhs.latestIssues;
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

static bool areManagedRoots(llvm::ArrayRef<mlir::Value> roots,
                            bool (*isExpectedType)(mlir::Type)) {
  return !roots.empty() && llvm::all_of(roots, [&](mlir::Value root) {
    return root.getDefiningOp<mlir::memref::AllocOp>() &&
           isExpectedType(root.getType());
  });
}

static bool hasPriorLocalMaterializationStore(mlir::Value ddrRoot,
                                              InstrRDMAOp reload) {
  auto allocation = ddrRoot.getDefiningOp<mlir::memref::AllocOp>();
  mlir::Block *block = reload->getBlock();
  if (!allocation || !block || allocation->getBlock() != block)
    return false;

  for (mlir::Operation &operation : *block) {
    if (&operation == reload.getOperation())
      break;
    auto store = mlir::dyn_cast<InstrWDMAOp>(operation);
    if (!store)
      continue;
    llvm::SmallVector<mlir::Value, 4> sourceRoots =
        getAccessRoots(store.getSource());
    llvm::SmallVector<mlir::Value, 4> destinationRoots =
        getAccessRoots(store.getDest());
    if (areManagedRoots(sourceRoots, isWaferSPMMemRefType) &&
        llvm::is_contained(destinationRoots, ddrRoot))
      return true;
  }
  return false;
}

static bool isLocalManagedMaterializationReload(
    InstrRDMAOp reload, llvm::ArrayRef<mlir::Value> sourceRoots,
    llvm::ArrayRef<mlir::Value> destinationRoots) {
  // A selective spill is represented by one finite local interval: a store
  // into a fresh DDR allocation and a later reload into a fresh SPM root in
  // the same block. Ordinary producer/consumer DDR edges across traversal
  // loops are not lifetime cuts; same-worker busytable ordering is sufficient
  // for those edges and fresh completion must not invent loop-local joins.
  return areManagedRoots(sourceRoots, isWaferDDRMemRefType) &&
         areManagedRoots(destinationRoots, isWaferSPMMemRefType) &&
         llvm::all_of(sourceRoots, [&](mlir::Value root) {
           return hasPriorLocalMaterializationStore(root, reload);
         });
}

static bool isManagedMaterializationReload(
    mlir::Operation *operation,
    const llvm::DenseSet<mlir::Value> &materializationRoots) {
  auto rdma = mlir::dyn_cast<InstrRDMAOp>(operation);
  if (!rdma)
    return false;
  llvm::SmallVector<mlir::Value, 4> sourceRoots =
      getAccessRoots(rdma.getSource());
  llvm::SmallVector<mlir::Value, 4> destinationRoots =
      getAccessRoots(rdma.getDest());
  return areManagedRoots(sourceRoots, isWaferDDRMemRefType) &&
         areManagedRoots(destinationRoots, isWaferSPMMemRefType) &&
         llvm::all_of(sourceRoots, [&](mlir::Value root) {
           return materializationRoots.contains(root);
         });
}

static bool isManagedMaterializationStore(
    mlir::Operation *operation,
    const llvm::DenseSet<mlir::Value> &materializationRoots) {
  auto wdma = mlir::dyn_cast<InstrWDMAOp>(operation);
  if (!wdma)
    return false;
  llvm::SmallVector<mlir::Value, 4> destinationRoots =
      getAccessRoots(wdma.getDest());
  return areManagedRoots(destinationRoots, isWaferDDRMemRefType) &&
         llvm::any_of(destinationRoots, [&](mlir::Value root) {
           return materializationRoots.contains(root);
         });
}

static mlir::Operation *
getManagedReloadCompletionAnchor(mlir::Operation *operation,
                                 const PendingNCCState &state,
                                 uint32_t workerMask) {
  auto rdma = mlir::dyn_cast<InstrRDMAOp>(operation);
  if (!rdma || llvm::popcount(workerMask) != 1)
    return operation;
  uint32_t worker = llvm::countr_zero(workerMask);
  mlir::Operation *latestIssue = state.latestIssues[worker];
  if (!latestIssue || latestIssue->getBlock() != operation->getBlock())
    return operation;

  mlir::Operation *anchor = nullptr;
  for (mlir::Value root : getAccessRoots(rdma.getDest())) {
    auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || allocation->getBlock() != operation->getBlock() ||
        !latestIssue->isBeforeInBlock(allocation) ||
        !allocation->isBeforeInBlock(operation))
      return operation;
    if (!anchor || allocation->isBeforeInBlock(anchor))
      anchor = allocation;
  }
  return anchor ? anchor : operation;
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
                                PendingNCCState &state) {
  mask &= state.workers;
  if (mask == 0)
    return;
  mlir::OpBuilder builder(operation);
  builder.create<SyncNCCJoinOp>(operation->getLoc(), getParticipants(mask));
  clearCompletedWorkers(state, mask);
}

static void insertNCCJoinAfter(mlir::Operation *operation, uint32_t mask,
                               PendingNCCState &state) {
  mask &= state.workers;
  if (mask == 0)
    return;
  mlir::OpBuilder builder(operation);
  builder.setInsertionPointAfter(operation);
  builder.create<SyncNCCJoinOp>(operation->getLoc(), getParticipants(mask));
  clearCompletedWorkers(state, mask);
}

static void recordNCCIssue(mlir::Operation *operation, uint32_t workerMask,
                           PendingNCCState &state) {
  state.workers |= workerMask;
  for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker)
    if ((workerMask & (uint32_t{1} << worker)) != 0)
      state.latestIssues[worker] = operation;
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
                        const PendingNCCState &state,
                        bool ignoreTypedIssueResources = false) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return 0;
  uint32_t conflicts = 0;
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
      // NCC completion. Unknown and other synchronous observers retain the
      // conservative all-pending conflict.
      if (ignoreTypedResources &&
          instance.getResource() != mlir::SideEffects::DefaultResource::get())
        continue;
      if (!mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect()))
        conflicts |= state.workers;
      continue;
    }
    if (mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect()))
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
/// must not change completion legality.
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

static uint32_t getRegionLocalPendingWorkerMask(TileRegionOp region,
                                                const PendingNCCState &state) {
  auto isRegionLocalRoot = [&](mlir::Value root) {
    auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
    return allocation && allocation->getParentOfType<TileRegionOp>() == region;
  };
  uint32_t workers = 0;
  for (const auto &entry : state.readers)
    if (isRegionLocalRoot(entry.first))
      workers |= entry.second;
  for (const auto &entry : state.writers)
    if (isRegionLocalRoot(entry.first))
      workers |= entry.second;
  return workers;
}

class NCCJoinPlacement {
public:
  mlir::LogicalResult run(mlir::ModuleOp module) {
    materializationRoots.clear();
    module.walk([&](InstrRDMAOp rdma) {
      llvm::SmallVector<mlir::Value, 4> sourceRoots =
          getAccessRoots(rdma.getSource());
      llvm::SmallVector<mlir::Value, 4> destinationRoots =
          getAccessRoots(rdma.getDest());
      if (!isLocalManagedMaterializationReload(rdma, sourceRoots,
                                               destinationRoots))
        return;
      materializationRoots.insert(sourceRoots.begin(), sourceRoots.end());
    });

    mlir::WalkResult result = module.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return mlir::WalkResult::skip();
      if (!function.getBody().hasOneBlock()) {
        bool hasIssue = false;
        function.walk([&](WaferNCCIssueOpInterface) { hasIssue = true; });
        if (hasIssue) {
          function.emitError()
              << "instruction_completion_failure: exact NCC completion "
                 "placement requires structured single-block function IR";
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::skip();
      }
      PendingNCCState state;
      if (mlir::failed(processBlock(function.getBody().front(), state)))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::skip();
    });
    return result.wasInterrupted() ? mlir::failure() : mlir::success();
  }

private:
  mlir::LogicalResult processBlock(mlir::Block &block, PendingNCCState &state) {
    for (auto iterator = block.begin(); iterator != block.end();) {
      mlir::Operation *operation = &*iterator++;
      if (mlir::failed(processOperation(operation, state)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult processOperation(mlir::Operation *operation,
                                       PendingNCCState &state) {
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(operation)) {
      if (!tileRegion.getBody().hasOneBlock())
        return tileRegion.emitError()
               << "instruction_completion_failure: NCC pending-frontier "
                  "placement requires a single-block wafer.tile.region";
      if (mlir::failed(processBlock(tileRegion.getBody().front(), state)))
        return mlir::failure();
      // Region partitioning is a selected residency cut, but the structural
      // boundary alone is not a completion event. Complete exactly those
      // participant domains that still access roots owned by this region;
      // unrelated pending work remains live across the boundary.
      insertNCCJoinBefore(tileRegion.getBody().front().getTerminator(),
                          getRegionLocalPendingWorkerMask(tileRegion, state),
                          state);
      return mlir::success();
    }

    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      PendingNCCState thenState = state;
      if (mlir::failed(processBlock(ifOp.getThenRegion().front(), thenState)))
        return mlir::failure();
      PendingNCCState elseState = state;
      if (!ifOp.getElseRegion().empty() &&
          mlir::failed(processBlock(ifOp.getElseRegion().front(), elseState)))
        return mlir::failure();
      state = std::move(thenState);
      mergePendingState(state, elseState);
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

      PendingNCCState bodyState = state;
      if (mlir::failed(processBlock(*forOp.getBody(), bodyState)))
        return mlir::failure();
      bool guaranteedToExecute =
          lower && upper && step && *step > 0 && *lower < *upper;
      if (!guaranteedToExecute && (bodyState.workers & ~state.workers) != 0)
        return forOp.emitError()
               << "instruction_completion_failure: dynamic or optional loop "
                  "would require a potentially empty NCC terminal join";
      if (!guaranteedToExecute) {
        mergePendingState(state, bodyState);
        return mlir::success();
      }

      __int128 span =
          static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
      __int128 tripCount = (span + static_cast<__int128>(*step) - 1) /
                           static_cast<__int128>(*step);
      if (tripCount == 1) {
        state = std::move(bodyState);
        return mlir::success();
      }

      // A later iteration observes the prior iteration's pending frontier.
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
        PendingNCCState nextState = state;
        mergePendingState(nextState, bodyState);
        if (mlir::failed(processBlock(*forOp.getBody(), nextState)))
          return mlir::failure();
        if (haveEqualPendingState(nextState, bodyState)) {
          converged = true;
          bodyState = std::move(nextState);
          break;
        }
        bodyState = std::move(nextState);
      }
      if (!converged)
        return forOp.emitError()
               << "instruction_completion_failure: NCC loop backedge "
                  "frontier did not reach a finite fixed point";
      // A pending worker with no unique latest issue came from alternative
      // paths where an NCC issue is optional or differs by branch.  The next
      // iteration therefore cannot prove a same-worker successor on every
      // path.  Complete only those ambiguous workers at the current
      // backedge; unconditional homogeneous streams remain join-free.
      uint32_t ambiguousBackedgeWorkers = 0;
      for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
        const uint32_t mask = uint32_t{1} << worker;
        if ((bodyState.workers & mask) != 0 &&
            bodyState.latestIssues[worker] == nullptr)
          ambiguousBackedgeWorkers |= mask;
      }
      insertNCCJoinBefore(forOp.getBody()->getTerminator(),
                          ambiguousBackedgeWorkers, bodyState);
      state = std::move(bodyState);
      return mlir::success();
    }

    if (mlir::isa<mlir::func::ReturnOp, mlir::async::YieldOp>(operation)) {
      insertNCCJoinBefore(operation, state.workers, state);
      return mlir::success();
    }
    if (mlir::isa<TileYieldOp, mlir::scf::YieldOp>(operation))
      return mlir::success();

    NCCCompletionContract contract = getNCCCompletionContract(operation);
    if (contract.issueWorker) {
      uint32_t worker = static_cast<uint32_t>(*contract.issueWorker);
      if (worker >= kNCCWorkerCount)
        return operation->emitError()
               << "instruction_completion_failure: NCC issue worker is "
                  "outside the typed worker domain";
      uint32_t workerMask = uint32_t{1} << worker;
      // The target busytable orders an NCC worker's own RAW/WAR/WAW chain,
      // but does not prove visibility across workers. Complete only prior
      // conflicting workers before issuing the new access; disjoint workers
      // and ordinary same-worker chains remain in one nonblocking issue
      // window. An explicit compiler-managed spill/reload additionally owns
      // a real residency cut. Its store completion is emitted immediately
      // below; before its reload, complete any intervening work on the same
      // worker so the fresh SPM root can reuse that worker's prior ranges.
      uint32_t crossWorkerConflicts =
          getExternalConflictMask(operation, state,
                                  /*ignoreTypedIssueResources=*/true) &
          ~workerMask;
      insertNCCJoinBefore(operation, crossWorkerConflicts, state);
      if (isManagedMaterializationReload(operation, materializationRoots)) {
        mlir::Operation *anchor =
            getManagedReloadCompletionAnchor(operation, state, workerMask);
        insertNCCJoinBefore(anchor, workerMask, state);
      }
      recordNCCIssue(operation, workerMask, state);
      if (isManagedMaterializationStore(operation, materializationRoots))
        insertNCCJoinAfter(operation, workerMask, state);
    }

    if (contract.behavior == LocalInstructionCompletion::ParticipantJoin) {
      uint32_t requested = contract.participantMask & kAllNCCWorkersMask;
      if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
        uint32_t effective = requested & state.workers;
        if (effective == 0) {
          join.erase();
          return mlir::success();
        }
        if (effective != requested)
          join.setParticipants(getParticipants(effective));
        clearCompletedWorkers(state, effective);
        return mlir::success();
      }
      clearCompletedWorkers(state, requested);
      return mlir::success();
    }
    if (contract.behavior == LocalInstructionCompletion::SynchronousWriteback) {
      clearCompletedWorkers(state, contract.participantMask);
      return mlir::success();
    }
    if (contract.behavior == LocalInstructionCompletion::OrderedPending)
      return mlir::success();

    // Direct DTE completion observes only its own typed event. It neither
    // consumes nor completes an NCC worker frontier, so independent NCC work
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
             << "instruction_completion_failure: operation without a typed "
                "memory-effect contract may observe pending NCC work";
    }

    uint32_t conflicts = getExternalConflictMask(operation, state);
    insertNCCJoinBefore(operation, conflicts, state);
    return mlir::success();
  }

  llvm::DenseSet<mlir::Value> materializationRoots;
};

struct ConvertTileRegionToInstrPass
    : public wafer::impl::ConvertTileRegionToInstrPassBase<
          ConvertTileRegionToInstrPass> {
  using wafer::impl::ConvertTileRegionToInstrPassBase<
      ConvertTileRegionToInstrPass>::ConvertTileRegionToInstrPassBase;

  void runOnOperation() final {
    std::string failureReason;
    if (mlir::succeeded(wafer::convertTileRegionToInstrModule(getOperation(),
                                                              &failureReason)))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("tile-region to instruction conversion failed");
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult wafer::normalizeMinimumNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr",
      "minimum-ncc-join-normalization");
  mlir::LogicalResult result = NCCJoinPlacement().run(module);
  if (mlir::failed(result))
    timing.markFailed();
  return result;
}

mlir::LogicalResult wafer::rebuildMinimumNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr", "fresh-ncc-join-rebuild");
  llvm::SmallVector<SyncNCCJoinOp, 16> staleJoins;
  module.walk([&](SyncNCCJoinOp join) { staleJoins.push_back(join); });
  for (SyncNCCJoinOp join : llvm::reverse(staleJoins))
    join.erase();
  mlir::LogicalResult result = NCCJoinPlacement().run(module);
  if (mlir::failed(result))
    timing.markFailed();
  return result;
}

bool wafer::containsTileDataflowOperations(mlir::Operation *root) {
  if (!root)
    return false;
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    if (!mlir::isa<WaferTileDataflowOpInterface>(operation))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

wafer::detail::StaticExecutableOperationCountStatus
wafer::detail::countStaticExecutableOperations(mlir::Operation *root,
                                               uint64_t &operationCount) {
  operationCount = 0;
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow ||
        !mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation))
      return;
    if (operationCount == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++operationCount;
  });
  if (overflow)
    return StaticExecutableOperationCountStatus::CountOverflow;
  return StaticExecutableOperationCountStatus::Counted;
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      std::string *failureReason) {
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TileToInstructionLowering);
  wafer::support::ScopedCompileTimingSpan conversionTiming(
      "conversion", "tile-region-to-instr", "module-conversion");
  if (failureReason)
    failureReason->clear();

  mlir::MLIRContext *context = module.getContext();
  llvm::SmallVector<mlir::Operation *, 4> selectCandidates;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "select-discovery");
    module.walk([&](ComputeElementwiseOp op) {
      if (op.getKind() == ComputeElementwiseKind::Select)
        selectCandidates.push_back(op);
    });
  }
  if (!selectCandidates.empty()) {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr",
        "constant-select-canonicalization");
    mlir::RewritePatternSet canonicalizationPatterns(context);
    populateConstantPredicateSelectCanonicalizationPattern(
        canonicalizationPatterns);
    mlir::FrozenRewritePatternSet frozenPatterns(
        std::move(canonicalizationPatterns));
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyOpPatternsAndFold(selectCandidates,
                                                  frozenPatterns, config))) {
      timing.markFailed();
      setFailureReason(
          failureReason,
          "tile constant-predicate select canonicalization failed");
      return mlir::failure();
    }
  }

  mlir::ConversionTarget target(*context);
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr",
        "conversion-target-configuration");
    configureTileRegionToInstrTarget(target);
  }

  mlir::RewritePatternSet patterns(context);
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "pattern-population");
    populateTileRegionToInstrPatterns(patterns);
  }

  bool conversionSucceeded = false;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "full-conversion");
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
    if (!conversionSucceeded)
      timing.markFailed();
  }

  if (!conversionSucceeded) {
    conversionTiming.markFailed();
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason,
                       "tile-region to instruction conversion failed");
    return mlir::failure();
  }
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "dead-private-fill-erasure");
    eraseDeadPrivateFills(module);
  }
  mlir::LogicalResult result = wafer::normalizeMinimumNCCJoins(module);
  if (mlir::failed(result))
    conversionTiming.markFailed();
  return result;
}
