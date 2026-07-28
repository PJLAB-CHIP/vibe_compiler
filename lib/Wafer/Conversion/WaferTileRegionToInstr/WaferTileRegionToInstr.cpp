//===- WaferTileRegionToInstr.cpp - Tile-region to instr conversion ------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "Internal.h"
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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

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
  target
      .addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncLocalFenceOp,
                  SyncNCCJoinOp, InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp,
                  InstrFillOp, InstrElementwiseOp, InstrBit2FpOp,
                  InstrMaskMoveOp, InstrReduceOp, InstrConvertOp, InstrGemmOp,
                  InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>();
  target.addDynamicallyLegalOp<InstrTDMADataMoveOp>([](InstrTDMADataMoveOp op) {
    return !requiresGatherScatterMaterialization(op.getKindAttr().getValue());
  });
  target.addIllegalOp<StorageLoadOp, StorageStoreOp, LayoutMaterializeOp,
                      ComputeFillOp, ComputeConvertOp, ComputeGemmOp,
                      ComputeElementwiseOp, ComputeReduceOp, MoveCopyOp,
                      MoveExtractSliceOp, MoveInsertSliceOp, MoveTransposeOp,
                      MoveBroadcastOp, ViewReshapeOp, CommAllGatherOp,
                      CommReduceScatterOp, CommAllReduceOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static void
populateTileRegionToInstrPatterns(mlir::RewritePatternSet &patterns,
                                  const TileRegionToInstrOptions &options,
                                  std::string *failureReason) {
  populateMovementLoweringPatterns(patterns, failureReason);
  populateComputeLoweringPatterns(patterns, failureReason);
  populateViewReshapeLoweringPattern(patterns, failureReason);
  populateFillLoweringPattern(patterns);
  populateCollectiveLoweringPatterns(patterns, options, failureReason);
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
}

static void mergePendingState(PendingNCCState &destination,
                              const PendingNCCState &source) {
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
    if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
        forOp && owner == forOp.getBody() && blockArgument.getArgNumber() > 0) {
      unsigned resultNumber = blockArgument.getArgNumber() - 1;
      if (resultNumber < forOp.getInitArgs().size())
        collectAccessRoots(forOp.getInitArgs()[resultNumber], roots, visited);
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
      if (yield && resultNumber < yield.getOperands().size())
        collectAccessRoots(yield.getOperands()[resultNumber], roots, visited);
      return;
    }
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

static bool areKnownDistinctRoots(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return false;
  return lhs.getDefiningOp<mlir::memref::AllocOp>() &&
         rhs.getDefiningOp<mlir::memref::AllocOp>();
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
                                PendingNCCState &state) {
  mask &= state.workers;
  if (mask == 0)
    return;
  mlir::OpBuilder builder(operation);
  builder.create<SyncNCCJoinOp>(operation->getLoc(), getParticipants(mask));
  clearCompletedWorkers(state, mask);
}

static void recordNCCIssue(mlir::Operation *operation, uint32_t workerMask,
                           PendingNCCState &state) {
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
                        const PendingNCCState &state,
                        bool ignoreTypedIssueResources = false) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return 0;
  uint32_t conflicts = 0;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances) {
    mlir::Value value = instance.getValue();
    if (!value) {
      // An NCC issue's custom Wafer resource effects describe typed engine
      // and memory-space occupancy. Its value-associated effects below are
      // the address hazard contract, so the custom resource must not turn two
      // proven-distinct roots into a blocking cross-worker join. For a
      // non-NCC observer (including DTE/Kcore/unknown resource users), retain
      // the conservative all-pending conflict.
      if (ignoreTypedIssueResources &&
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

class NCCJoinPlacement {
public:
  mlir::LogicalResult run(mlir::ModuleOp module) {
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
      return processBlock(tileRegion.getBody().front(), state);
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
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(forOp.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(forOp.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(forOp.getStep());
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
      // and same-worker chains remain in one nonblocking issue window.
      uint32_t crossWorkerConflicts =
          getExternalConflictMask(operation, state,
                                  /*ignoreTypedIssueResources=*/true) &
          ~workerMask;
      insertNCCJoinBefore(operation, crossWorkerConflicts, state);
      recordNCCIssue(operation, workerMask, state);
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
  return NCCJoinPlacement().run(module);
}

wafer::detail::StaticTerminalOperationBudgetStatus
wafer::detail::checkStaticTerminalOperationBudget(mlir::Operation *root,
                                                  uint64_t &operationCount) {
  operationCount = 0;
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow || !mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp,
                               SyncNCCJoinOp>(operation))
      return;
    if (operationCount == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++operationCount;
  });
  if (overflow)
    return StaticTerminalOperationBudgetStatus::CountOverflow;
  if (operationCount > kStaticTerminalOperationBudget)
    return StaticTerminalOperationBudgetStatus::BudgetExceeded;
  return StaticTerminalOperationBudgetStatus::WithinBudget;
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      std::string *failureReason) {
  return wafer::tile_region_to_instr::convertTileRegionToInstrModule(
      module, TileRegionToInstrOptions{}, failureReason);
}

mlir::LogicalResult wafer::tile_region_to_instr::convertTileRegionToInstrModule(
    mlir::ModuleOp module, const TileRegionToInstrOptions &options,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  mlir::MLIRContext *context = module.getContext();
  llvm::SmallVector<mlir::Operation *, 4> selectCandidates;
  module.walk([&](ComputeElementwiseOp op) {
    if (op.getKind() == ComputeElementwiseKind::Select)
      selectCandidates.push_back(op);
  });
  if (!selectCandidates.empty()) {
    mlir::RewritePatternSet canonicalizationPatterns(context);
    populateConstantPredicateSelectCanonicalizationPattern(
        canonicalizationPatterns);
    mlir::FrozenRewritePatternSet frozenPatterns(
        std::move(canonicalizationPatterns));
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyOpPatternsAndFold(selectCandidates,
                                                  frozenPatterns, config))) {
      setFailureReason(
          failureReason,
          "tile constant-predicate select canonicalization failed");
      return mlir::failure();
    }
  }

  mlir::ConversionTarget target(*context);
  configureTileRegionToInstrTarget(target);

  mlir::RewritePatternSet patterns(context);
  populateTileRegionToInstrPatterns(patterns, options, failureReason);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason,
                       "tile-region to instruction conversion failed");
    return mlir::failure();
  }
  eraseDeadPrivateFills(module);
  return wafer::normalizeMinimumNCCJoins(module);
}
