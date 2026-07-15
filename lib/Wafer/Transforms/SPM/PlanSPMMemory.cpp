//===- PlanSPMMemory.cpp - Plan Wafer SPM memory --------------------------===//

#include "Wafer/Transforms/Passes.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace wafer {
#define GEN_PASS_DEF_PLANSPMMEMORYPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

namespace mp = memory_planning::detail;

struct DTECompletionRef {
  mlir::Value originToken;
  mp::PathCondition path = mp::PathCondition::root();
};

struct PendingSPMPlacement {
  mlir::memref::AllocOp allocation;
  int64_t offsetBytes = 0;
};

static bool hasAsyncTokenType(mlir::Value value) {
  return mlir::isa<mlir::async::TokenType>(value.getType());
}

static mlir::scf::YieldOp getSingleBlockYield(mlir::Region &region) {
  if (region.empty())
    return {};
  return mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
}

static bool belongsToTileRegion(TileRegionOp tileRegion, mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == tileRegion.getOperation())
      return true;
    if (mlir::isa<TileRegionOp>(parent))
      return false;
  }
  return false;
}

static mlir::LogicalResult verifyCompletionControlFlow(mlir::Operation *scope) {
  mlir::WalkResult result = scope->walk([&](mlir::scf::ForOp forOp) {
    bool carriesToken = llvm::any_of(forOp.getInitArgs(), hasAsyncTokenType) ||
                        llvm::any_of(forOp.getResults(), hasAsyncTokenType);
    if (!carriesToken)
      return mlir::WalkResult::advance();
    forOp.emitError()
        << "unsupported_completion_control_flow: SPM memory planning cannot "
           "prove loop-carried DTE token completion";
    return mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::LogicalResult
verifySPMRegionExecutionScopes(mlir::ModuleOp moduleOp) {
  mlir::WalkResult result = moduleOp.walk([&](TileRegionOp tileRegion) {
    for (mlir::Operation *parent = tileRegion->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (mlir::isa<mlir::ModuleOp>(parent))
        break;
      if (mlir::isa<TileRegionOp>(parent)) {
        tileRegion.emitError()
            << "unsupported_spm_scope_nesting: nested wafer.tile.region "
               "scopes cannot independently plan the same physical SPM "
               "arena";
        return mlir::WalkResult::interrupt();
      }
      if (mlir::isa<mlir::func::FuncOp, mlir::scf::IfOp, mlir::scf::ForOp>(
              parent))
        continue;
      tileRegion.emitError()
          << "unsupported_spm_planning_scope: wafer.tile.region must be "
             "owned only by sequential func.func/scf.if/scf.for "
             "execution scopes";
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::LogicalResult verifySPMValueScopes(mlir::ModuleOp moduleOp) {
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    bool hasSPMBoundary =
        llvm::any_of(
            funcOp.getArgumentTypes(),
            [](mlir::Type type) { return isWaferSPMMemRefType(type); }) ||
        llvm::any_of(funcOp.getResultTypes(), [](mlir::Type type) {
          return isWaferSPMMemRefType(type);
        });
    if (hasSPMBoundary)
      return funcOp.emitError()
             << "unsupported_spm_planning_scope: func.func boundaries cannot "
                "own SPM values outside wafer.tile.region";
  }

  mlir::WalkResult result = moduleOp.walk([&](mlir::Operation *op) {
    if (op->getParentOfType<TileRegionOp>() ||
        op->getParentOfType<mlir::async::FuncOp>())
      return mlir::WalkResult::advance();
    bool hasSPMValue = llvm::any_of(op->getOperandTypes(),
                                    [](mlir::Type type) {
                                      return isWaferSPMMemRefType(type);
                                    }) ||
                       llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
                         return isWaferSPMMemRefType(type);
                       });
    if (!hasSPMValue)
      return mlir::WalkResult::advance();
    op->emitError()
        << "unsupported_spm_planning_scope: SPM values and uses must be "
           "contained by wafer.tile.region (async.func formals are checked "
           "at their tile-local call sites)";
    return mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static bool isPotentiallyOverlappingSPMCall(mlir::Operation *call) {
  for (mlir::Operation *parent = call->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (mlir::isa<mlir::func::FuncOp, mlir::ModuleOp>(parent))
      return false;
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(parent))
      continue;
    return true;
  }
  return true;
}

static mlir::LogicalResult verifySPMCallScopes(mlir::ModuleOp moduleOp) {
  llvm::DenseSet<mlir::Operation *> mayExecuteTileRegion;
  bool moduleHasTileRegion = false;
  moduleOp.walk([&](TileRegionOp tileRegion) {
    moduleHasTileRegion = true;
    if (mlir::func::FuncOp owner =
            tileRegion->getParentOfType<mlir::func::FuncOp>())
      mayExecuteTileRegion.insert(owner.getOperation());
  });
  moduleOp.walk([&](mlir::func::CallIndirectOp call) {
    if (mlir::func::FuncOp owner = call->getParentOfType<mlir::func::FuncOp>())
      mayExecuteTileRegion.insert(owner.getOperation());
  });

  bool changed = true;
  while (changed) {
    changed = false;
    moduleOp.walk([&](mlir::func::CallOp call) {
      mlir::func::FuncOp caller = call->getParentOfType<mlir::func::FuncOp>();
      mlir::func::FuncOp callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (caller && callee &&
          mayExecuteTileRegion.contains(callee.getOperation()) &&
          mayExecuteTileRegion.insert(caller.getOperation()).second)
        changed = true;
    });
  }

  mlir::WalkResult result =
      moduleOp.walk(
          [&](mlir::func::CallOp call) {
            mlir::func::FuncOp callee =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                    call, call.getCalleeAttr());
            if (!isPotentiallyOverlappingSPMCall(call.getOperation()))
              return mlir::WalkResult::advance();
            if (callee && !callee.isExternal() &&
                !mayExecuteTileRegion.contains(callee.getOperation()))
              return mlir::WalkResult::advance();
            call.emitError()
                << "unsupported_spm_planning_scope: a call from an active or "
                   "asynchronous SPM scope may dynamically execute another "
                   "wafer.tile.region";
            return mlir::WalkResult::interrupt();
          });
  if (result.wasInterrupted())
    return mlir::failure();

  result = moduleOp.walk([&](mlir::func::CallIndirectOp call) {
    if (!isPotentiallyOverlappingSPMCall(call.getOperation()))
      return mlir::WalkResult::advance();
    call.emitError()
        << "unsupported_spm_planning_scope: an indirect call from an active "
           "or asynchronous SPM scope may execute another "
           "wafer.tile.region";
    return mlir::WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return mlir::failure();

  if (!moduleHasTileRegion)
    return mlir::success();
  result = moduleOp.walk([&](mlir::async::CallOp call) {
    mlir::async::FuncOp callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::async::FuncOp>(
            call, call.getCalleeAttr());
    if (callee && !callee.isExternal())
      return mlir::WalkResult::advance();
    call.emitError()
        << "unsupported_spm_planning_scope: external async.call in a module "
           "with SPM tile regions has no verifiable arena/resource summary";
    return mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

class DTECompletionTracker {
public:
  explicit DTECompletionTracker(const mp::StructuredTimeline &timeline)
      : timeline(timeline) {}

  mlir::LogicalResult run(mlir::Operation *scope) {
    for (mlir::Region &region : scope->getRegions())
      if (mlir::failed(processRegion(region)))
        return mlir::failure();
    if (pendingCompletions.empty())
      return mlir::success();

    mlir::Operation *origin =
        pendingCompletions.front().originToken.getDefiningOp();
    mlir::InFlightDiagnostic diagnostic =
        (origin ? origin : scope)->emitError()
        << "missing_dte_completion: DTE token has a reachable path to ";
    diagnostic << (mlir::isa<TileRegionOp>(scope) ? "wafer.tile.region exit"
                                                  : "async function exit")
               << " without wafer.instr.dte_wait";
    return mlir::failure();
  }

private:
  llvm::SmallVector<DTECompletionRef, 2>
  completionsAt(mlir::Value token, mp::PathCondition usePath) const {
    llvm::SmallVector<DTECompletionRef, 2> refs;
    auto it = completionRefs.find(token);
    if (it == completionRefs.end())
      return refs;
    for (DTECompletionRef ref : it->second) {
      if (std::optional<mp::PathCondition> path = ref.path.intersect(usePath))
        refs.push_back(DTECompletionRef{ref.originToken, *path});
    }
    return refs;
  }

  void processWait(InstrDTEWaitOp op) {
    std::optional<mp::ProgramPoint> point = timeline.lookup(op);
    if (!point)
      return;

    for (mlir::Value token : op.getTokens()) {
      for (DTECompletionRef waited : completionsAt(token, point->path)) {
        llvm::SmallVector<DTECompletionRef, 8> remaining;
        for (DTECompletionRef pending : pendingCompletions) {
          if (pending.originToken != waited.originToken ||
              !pending.path.intersect(waited.path)) {
            remaining.push_back(pending);
            continue;
          }

          llvm::SmallVector<mp::PathCondition, 2> remainingPaths;
          pending.path.subtract(waited.path, remainingPaths);
          for (mp::PathCondition path : remainingPaths)
            remaining.push_back(DTECompletionRef{pending.originToken, path});
        }
        pendingCompletions = std::move(remaining);
      }
    }
  }

  void recordCompletion(mlir::Operation *op) {
    if (!mlir::isa<InstrDTESendOp, InstrDTERecvOp>(op))
      return;
    std::optional<mp::ProgramPoint> point = timeline.lookup(op);
    if (!point)
      return;
    for (mlir::Value result : op->getResults()) {
      if (!hasAsyncTokenType(result))
        continue;
      DTECompletionRef ref{result, point->path};
      completionRefs[result] = llvm::SmallVector<DTECompletionRef, 2>{ref};
      pendingCompletions.push_back(ref);
    }
  }

  void appendYieldCompletions(mlir::scf::YieldOp yield, unsigned index,
                              llvm::SmallVectorImpl<DTECompletionRef> &refs) {
    if (index >= yield.getResults().size())
      return;
    std::optional<mp::ProgramPoint> point = timeline.lookup(yield);
    if (!point)
      return;
    llvm::SmallVector<DTECompletionRef, 2> yielded =
        completionsAt(yield.getResults()[index], point->path);
    refs.append(yielded.begin(), yielded.end());
  }

  void mapIfResults(mlir::scf::IfOp ifOp) {
    mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
    mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
    if (!thenYield || !elseYield)
      return;

    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (!hasAsyncTokenType(result))
        continue;
      llvm::SmallVector<DTECompletionRef, 2> refs;
      appendYieldCompletions(thenYield, index, refs);
      appendYieldCompletions(elseYield, index, refs);
      if (!refs.empty())
        completionRefs[result] = std::move(refs);
    }
  }

  mlir::LogicalResult processRegion(mlir::Region &region) {
    for (mlir::Block &block : region)
      if (mlir::failed(processBlock(block)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult processBlock(mlir::Block &block) {
    for (mlir::Operation &op : block) {
      if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
        processWait(wait);
      } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (mlir::failed(processRegion(forOp.getRegion())))
          return mlir::failure();
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        if (mlir::failed(processRegion(ifOp.getThenRegion())) ||
            mlir::failed(processRegion(ifOp.getElseRegion())))
          return mlir::failure();
        mapIfResults(ifOp);
      } else {
        for (mlir::Region &region : op.getRegions())
          if (mlir::failed(processRegion(region)))
            return mlir::failure();
      }
      recordCompletion(&op);
    }
    return mlir::success();
  }

  const mp::StructuredTimeline &timeline;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<DTECompletionRef, 2>>
      completionRefs;
  llvm::SmallVector<DTECompletionRef, 8> pendingCompletions;
};

static mlir::LogicalResult
initializeSPMDemands(TileRegionOp tileRegion, int64_t defaultAlignment,
                     llvm::SmallVectorImpl<mp::LifetimeDemand> &demands) {
  mlir::LogicalResult result = mlir::success();
  tileRegion.walk([&](mlir::memref::AllocOp alloc) {
    if (mlir::failed(result) ||
        !belongsToTileRegion(tileRegion, alloc.getOperation()))
      return;

    mlir::MemRefType memrefType = alloc.getType();
    if (!isWaferSPMMemRefType(memrefType))
      return;

    if (!alloc.getDynamicSizes().empty() ||
        !alloc.getSymbolOperands().empty()) {
      alloc.emitError() << "unsupported_layout_conversion: SPM memory planning "
                           "requires static "
                           "memref.alloc sizes and symbols";
      result = mlir::failure();
      return;
    }

    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(memrefType);
    if (!info || info->physicalBytes < 0) {
      alloc.emitError()
          << "unsupported_layout_conversion: cannot compute physical SPM "
             "byte size for "
          << memrefType;
      result = mlir::failure();
      return;
    }

    int64_t requiredAlignment = defaultAlignment;
    if (std::optional<uint64_t> allocAlignment = alloc.getAlignment()) {
      if (*allocAlignment >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        alloc.emitError()
            << "alignment_unsatisfied: memref.alloc alignment exceeds int64";
        result = mlir::failure();
        return;
      }
      std::optional<int64_t> combined = mp::combineAlignmentRequirements(
          requiredAlignment, static_cast<int64_t>(*allocAlignment));
      if (!combined) {
        alloc.emitError()
            << "alignment_unsatisfied: combined SPM alignment exceeds int64";
        result = mlir::failure();
        return;
      }
      requiredAlignment = *combined;
    }

    mp::LifetimeDemand demand;
    demand.allocation = alloc;
    demand.sizeBytes = info->physicalBytes;
    demand.alignmentBytes = requiredAlignment;
    demand.stableOrdinal = demands.size();
    demands.push_back(std::move(demand));
  });
  return result;
}

static mlir::LogicalResult
emitLifetimeFailure(TileRegionOp tileRegion,
                    const mp::LifetimeFailure &failure) {
  mlir::Operation *origin =
      failure.origin ? failure.origin : tileRegion.getOperation();
  switch (failure.kind) {
  case mp::LifetimeFailureKind::MissingAllocationEvent:
    return origin->emitError()
           << "lifetime_overlap_conflict: missing event for SPM allocation";
  case mp::LifetimeFailureKind::UnsupportedTrackedValueProducer:
    return origin->emitError()
           << "unsupported_lifetime_alias: SPM memref producers must be "
              "memref.alloc or implement a supported alias/control-flow "
              "interface";
  case mp::LifetimeFailureKind::UnsupportedTrackedValueEscape:
    return origin->emitError()
           << "unsupported_lifetime_alias: tracked SPM storage cannot escape "
              "through raw metadata or an operation without supported "
              "alias/effect semantics (operation "
           << origin->getName() << ")";
  case mp::LifetimeFailureKind::LoopCarriedAllocationInstance:
    return origin->emitError()
           << "unsupported_lifetime_alias: loop-body SPM allocation cannot be "
              "loop-carried without multi-instance placement";
  case mp::LifetimeFailureKind::MissingAsyncCompletion:
    return origin->emitError()
           << "missing_async_completion: asynchronous SPM access has a "
              "reachable path to wafer.tile.region exit without a proven "
              "completion wait";
  case mp::LifetimeFailureKind::UnsupportedAsyncCompletionFlow:
    return origin->emitError()
           << "unsupported_async_completion_flow: SPM memory planning cannot "
              "prove completion identity through this async handle flow";
  case mp::LifetimeFailureKind::MissingLocalCompletion:
    return origin->emitError()
           << "missing_local_completion: local Compute/Movement issue has a "
              "reachable path to wafer.tile.region exit without "
              "wafer.instr.local_fence";
  case mp::LifetimeFailureKind::LoopBackedgeCompletion:
    return origin->emitError()
           << "missing_local_completion: local Compute/Movement issue has a "
              "reachable loop backedge without wafer.instr.local_fence";
  case mp::LifetimeFailureKind::InconsistentCompletionState:
    return tileRegion.emitError()
           << "completion_proof_failure: local issue lifetime state remains "
              "after all local issues were fenced";
  }
  llvm_unreachable("unknown lifetime failure kind");
}

static mlir::LogicalResult
emitAsyncFunctionLifetimeFailure(mlir::async::FuncOp funcOp,
                                 const mp::LifetimeFailure &failure) {
  mlir::Operation *origin =
      failure.origin ? failure.origin : funcOp.getOperation();
  switch (failure.kind) {
  case mp::LifetimeFailureKind::MissingAllocationEvent:
    return origin->emitError()
           << "lifetime_overlap_conflict: missing event in SPM async callee";
  case mp::LifetimeFailureKind::UnsupportedTrackedValueProducer:
  case mp::LifetimeFailureKind::UnsupportedTrackedValueEscape:
  case mp::LifetimeFailureKind::LoopCarriedAllocationInstance:
    return origin->emitError()
           << "unsupported_lifetime_alias: SPM async callee contains an "
              "unsupported storage alias or dynamic instance";
  case mp::LifetimeFailureKind::MissingAsyncCompletion:
    return origin->emitError()
           << "missing_async_completion: SPM async callee returns before a "
              "nested asynchronous access completes";
  case mp::LifetimeFailureKind::UnsupportedAsyncCompletionFlow:
    return origin->emitError()
           << "unsupported_async_completion_flow: SPM async callee has an "
              "unprovable completion-handle flow";
  case mp::LifetimeFailureKind::MissingLocalCompletion:
  case mp::LifetimeFailureKind::LoopBackedgeCompletion:
  case mp::LifetimeFailureKind::InconsistentCompletionState:
    return origin->emitError()
           << "missing_local_completion: SPM async callee returns with an "
              "unfenced local engine access";
  }
  llvm_unreachable("unknown async callee lifetime failure kind");
}

static mlir::LogicalResult
verifySPMAsyncFunctionClosures(mlir::ModuleOp moduleOp) {
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::async::FuncOp funcOp) {
    if (mlir::failed(result) || funcOp.isExternal())
      return;
    if (mlir::failed(verifyCompletionControlFlow(funcOp.getOperation()))) {
      result = mlir::failure();
      return;
    }

    mp::TimelineFailure timelineFailure;
    mlir::FailureOr<mp::StructuredTimeline> timeline =
        mp::StructuredTimeline::build(funcOp.getOperation(), &timelineFailure);
    if (mlir::failed(timeline)) {
      mlir::Operation *origin = timelineFailure.origin ? timelineFailure.origin
                                                       : funcOp.getOperation();
      result = origin->emitError()
               << "unsupported_async_completion_flow: SPM async callee "
                  "completion proof requires supported single-block "
                  "structured control flow";
      return;
    }

    DTECompletionTracker dteCompletion(*timeline);
    if (mlir::failed(dteCompletion.run(funcOp.getOperation()))) {
      result = mlir::failure();
      return;
    }

    llvm::SmallVector<mp::LifetimeDemand, 0> demands;
    mp::LocalCompletionTracker localCompletion(WaferResourceKind::SPM);
    mp::LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
      return isWaferSPMMemRefType(type);
    });
    mp::LifetimeFailure lifetimeFailure;
    if (mlir::failed(dataflow.run(funcOp.getOperation(), &localCompletion,
                                  &lifetimeFailure)))
      result = emitAsyncFunctionLifetimeFailure(funcOp, lifetimeFailure);
  });
  return result;
}

static mlir::LogicalResult
planRegion(TileRegionOp tileRegion, int64_t spmBase, int64_t spmLimit,
           int64_t spmAlignment,
           llvm::SmallVectorImpl<PendingSPMPlacement> &pendingPlacements) {
  if (mlir::failed(verifyCompletionControlFlow(tileRegion.getOperation())))
    return mlir::failure();

  mp::TimelineFailure timelineFailure;
  mlir::FailureOr<mp::StructuredTimeline> timeline =
      mp::StructuredTimeline::build(tileRegion.getOperation(),
                                    &timelineFailure);
  if (mlir::failed(timeline)) {
    mlir::Operation *origin = timelineFailure.origin
                                  ? timelineFailure.origin
                                  : tileRegion.getOperation();
    if (timelineFailure.kind == mp::TimelineFailureKind::TooManyDecisions)
      return origin->emitError()
             << "lifetime_overlap_conflict: SPM memory planning supports at "
                "most 64 control-flow decision points";
    return origin->emitError()
           << "unsupported_lifetime_control_flow: SPM memory planning only "
              "supports single-block wafer.tile.region, scf.if and scf.for "
              "structured regions";
  }

  llvm::SmallVector<mp::LifetimeDemand, 8> demands;
  if (mlir::failed(initializeSPMDemands(tileRegion, spmAlignment, demands)))
    return mlir::failure();

  // Preserve the owner-specific DTE completion contract and diagnostic before
  // the shared generic async terminal proof handles other async producers.
  DTECompletionTracker dteCompletion(*timeline);
  if (mlir::failed(dteCompletion.run(tileRegion.getOperation())))
    return mlir::failure();

  mp::LocalCompletionTracker localCompletion(WaferResourceKind::SPM);
  mp::LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return isWaferSPMMemRefType(type);
  });
  mp::LifetimeFailure lifetimeFailure;
  if (mlir::failed(dataflow.run(tileRegion.getOperation(), &localCompletion,
                                &lifetimeFailure)))
    return emitLifetimeFailure(tileRegion, lifetimeFailure);

  auto tileYield =
      mlir::dyn_cast<TileYieldOp>(tileRegion.getBody().front().getTerminator());
  std::optional<mp::ProgramPoint> yieldPoint =
      tileYield ? timeline->lookup(tileYield.getOperation()) : std::nullopt;
  if (tileYield && yieldPoint) {
    for (mlir::Value value : tileYield.getOperands()) {
      if (dataflow.rootsAt(value, yieldPoint->path).empty())
        continue;
      mp::LifetimeFailure escape{
          mp::LifetimeFailureKind::UnsupportedTrackedValueEscape,
          tileYield.getOperation()};
      return emitLifetimeFailure(tileRegion, escape);
    }
  }

  mp::PackingResult packing =
      mp::packFirstFit(demands, mp::ArenaRange{spmBase, spmLimit});
  if (!packing.succeeded()) {
    const mp::PackingFailure &failure = *packing.failure;
    if (failure.demandIndex >= demands.size())
      return tileRegion.emitError()
             << "capacity_overflow: SPM planning failed without a demand";
    mp::LifetimeDemand &demand = demands[failure.demandIndex];
    return demand.allocation.emitError()
           << "capacity_overflow: SPM planning range [" << spmBase << ", "
           << spmLimit << ") cannot fit " << demand.sizeBytes
           << " byte buffer with IR-derived lifetime";
  }

  for (const mp::Placement &placement : packing.placements) {
    mp::LifetimeDemand &demand = demands[placement.demandIndex];
    pendingPlacements.push_back(
        PendingSPMPlacement{demand.allocation, placement.offsetBytes});
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult planSPMMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t spmBase, int64_t spmLimit,
                                        int64_t spmAlignment) {
  if (spmBase < 0 || spmLimit <= spmBase)
    return moduleOp->emitError()
           << "invalid_spm_range: expected 0 <= spm-base < spm-limit";
  if (spmAlignment <= 0)
    return moduleOp->emitError()
           << "alignment_unsatisfied: spm-alignment must be positive";
  mlir::WalkResult escapedAllocation =
      moduleOp.walk(
          [&](mlir::memref::AllocOp alloc) {
            if (!isWaferSPMMemRefType(alloc.getType()) ||
                alloc->getParentOfType<TileRegionOp>())
              return mlir::WalkResult::advance();
            alloc.emitError()
                << "unsupported_spm_planning_scope: compiler-managed SPM "
                   "allocation must be owned by a wafer.tile.region";
            return mlir::WalkResult::interrupt();
          });
  if (escapedAllocation.wasInterrupted())
    return mlir::failure();
  if (mlir::failed(verifySPMRegionExecutionScopes(moduleOp)) ||
      mlir::failed(verifySPMValueScopes(moduleOp)) ||
      mlir::failed(verifySPMCallScopes(moduleOp)))
    return mlir::failure();
  if (mlir::failed(verifySPMAsyncFunctionClosures(moduleOp)))
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  llvm::SmallVector<PendingSPMPlacement, 16> pendingPlacements;
  moduleOp.walk([&](TileRegionOp tileRegion) {
    if (mlir::failed(result))
      return;
    result = planRegion(tileRegion, spmBase, spmLimit, spmAlignment,
                        pendingPlacements);
  });
  if (mlir::failed(result))
    return result;

  for (PendingSPMPlacement placement : pendingPlacements) {
    placement.allocation->setAttr(
        kWaferSPMOffsetAttrName,
        SPMOffsetAttr::get(placement.allocation.getContext(),
                           placement.offsetBytes));
  }
  return mlir::success();
}

namespace {

struct PlanSPMMemoryPass
    : public impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass> {
  using impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass>::PlanSPMMemoryPassBase;

  void runOnOperation() final {
    if (mlir::failed(planSPMMemoryModule(getOperation(), spmBase, spmLimit,
                                         spmAlignment)))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
