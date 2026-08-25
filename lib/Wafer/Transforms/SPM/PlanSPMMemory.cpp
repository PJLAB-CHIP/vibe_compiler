//===- PlanSPMMemory.cpp - Plan Wafer SPM memory --------------------------===//

#include "Wafer/Transforms/MemoryPlanning.h"
#include "Wafer/Transforms/Passes.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "MemoryPlanning/StaticMemoryPacking.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Analysis/CallGraph/DirectCallGraphAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer {
#define GEN_PASS_DEF_PLANSPMMEMORYPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

namespace mp = memory_planning::detail;

using analysis::getSingleExecutionRegionFlow;

using ManagedTimelineMap =
    llvm::DenseMap<mlir::Operation *, const mp::StructuredTimelineAnalysis *>;

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

static bool isStaticallyNonEmpty(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getLowerBound()));
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getUpperBound()));
  std::optional<int64_t> step =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
  return lower && upper && step && *step > 0 && *lower < *upper;
}

static mlir::LogicalResult
verifySPMRegionExecutionScopes(mlir::ModuleOp moduleOp) {
  mlir::WalkResult result = moduleOp.walk([&](TileRegionOp tileRegion) {
    bool hasFunctionOwner = false;
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
      if (mlir::isa<mlir::func::FuncOp>(parent)) {
        hasFunctionOwner = true;
        continue;
      }
      if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(parent))
        continue;
      tileRegion.emitError()
          << "unsupported_spm_planning_scope: wafer.tile.region must be "
             "owned only by sequential func.func/scf.if/scf.for "
             "execution scopes";
      return mlir::WalkResult::interrupt();
    }
    if (!hasFunctionOwner) {
      tileRegion.emitError()
          << "unsupported_spm_planning_scope: wafer.tile.region requires a "
             "func.func owner for whole-function SPM planning";
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

    bool isSupportedSSAEdge =
        mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::YieldOp>(op) ||
        mlir::isa<mlir::ViewLikeOpInterface, mlir::SelectLikeOpInterface>(op) ||
        getSingleExecutionRegionFlow(op).has_value();
    if (isSupportedSSAEdge)
      return mlir::WalkResult::advance();
    op->emitError()
        << "unsupported_spm_planning_scope: SPM values outside "
           "wafer.tile.region may only flow through explicit tile-region "
           "operands/results or supported structured alias/control-flow SSA "
           "edges (async.func formals are checked at their tile-local call "
           "sites)";
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

static llvm::DenseSet<mlir::Operation *>
collectFunctionsThatMayClobberSPMArena(
    mlir::ModuleOp moduleOp,
    const analysis::DirectCallGraphAnalysis &callGraph) {
  llvm::DenseSet<mlir::Operation *> mayClobberFunctions;
  moduleOp.walk([&](TileRegionOp tileRegion) {
    if (mlir::func::FuncOp owner =
            tileRegion->getParentOfType<mlir::func::FuncOp>())
      mayClobberFunctions.insert(owner.getOperation());
  });
  for (mlir::Operation *call : callGraph.getUnsupportedCallOperations())
    if (mlir::func::FuncOp owner = call->getParentOfType<mlir::func::FuncOp>())
      mayClobberFunctions.insert(owner.getOperation());
  for (mlir::func::CallOp call : callGraph.getUnresolvedCalls())
    if (mlir::func::FuncOp caller =
            call->getParentOfType<mlir::func::FuncOp>())
      mayClobberFunctions.insert(caller.getOperation());
  for (mlir::func::FuncOp function : callGraph.getFunctions())
    for (mlir::func::CallOp call : callGraph.getCalls(function)) {
    mlir::func::FuncOp caller = call->getParentOfType<mlir::func::FuncOp>();
    mlir::func::FuncOp callee = callGraph.getCallee(call);
    if (caller && (!callee || callee.isExternal()))
      mayClobberFunctions.insert(caller.getOperation());
    }

  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::func::FuncOp caller : callGraph.getFunctions())
      for (mlir::func::FuncOp callee : callGraph.getCallees(caller))
        if (mayClobberFunctions.contains(callee.getOperation()) &&
            mayClobberFunctions.insert(caller.getOperation()).second)
          changed = true;
  }
  return mayClobberFunctions;
}

static bool mayClobberSPMArena(
    mlir::func::CallOp call,
    const analysis::DirectCallGraphAnalysis &callGraph,
    const llvm::DenseSet<mlir::Operation *> &mayClobberFunctions) {
  mlir::func::FuncOp callee = callGraph.getCallee(call);
  return !callee || callee.isExternal() ||
         mayClobberFunctions.contains(callee.getOperation());
}

static mlir::LogicalResult verifySPMCallScopes(
    mlir::ModuleOp moduleOp,
    const analysis::DirectCallGraphAnalysis &callGraph,
    const llvm::DenseSet<mlir::Operation *> &mayClobberFunctions) {
  bool moduleHasTileRegion = false;
  moduleOp.walk([&](TileRegionOp) { moduleHasTileRegion = true; });

  mlir::WalkResult result =
      moduleOp.walk(
          [&](mlir::func::CallOp call) {
            if (!isPotentiallyOverlappingSPMCall(call.getOperation()))
              return mlir::WalkResult::advance();
            if (!mayClobberSPMArena(call, callGraph, mayClobberFunctions))
              return mlir::WalkResult::advance();
            call.emitError()
                << "unsupported_spm_planning_scope: a call from an active or "
                   "asynchronous SPM scope may dynamically execute another "
                   "wafer.tile.region or otherwise clobber the independently "
                   "planned physical SPM arena";
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

static bool
hasLiveSPMStorageAcrossCall(mlir::Operation *call,
                            const mp::StructuredTimeline &timeline,
                            llvm::ArrayRef<mp::LifetimeDemand> demands) {
  std::optional<mp::ProgramPoint> callPoint = timeline.lookup(call);
  std::optional<int64_t> callEnd = timeline.lookupSubtreeEnd(call);
  if (!callPoint || !callEnd)
    return false;

  // Opposite branches inside a loop may be selected in different dynamic
  // iterations, so use the same repeatable-path compatibility as packing.
  return llvm::any_of(demands, [&](const mp::LifetimeDemand &demand) {
    return llvm::any_of(demand.segments, [&](const mp::LiveSegment &segment) {
      return segment.beginEvent < callPoint->event &&
             segment.endEvent > *callEnd &&
             segment.path.compatibleForPacking(callPoint->path);
    });
  });
}

static mlir::LogicalResult verifyLiveSPMAcrossCalls(
    mlir::func::FuncOp funcOp, const mp::StructuredTimeline &timeline,
    llvm::ArrayRef<mp::LifetimeDemand> demands,
    const analysis::DirectCallGraphAnalysis &callGraph,
    const llvm::DenseSet<mlir::Operation *> &mayClobberFunctions) {
  mlir::WalkResult result = funcOp.walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::func::CallOp, mlir::func::CallIndirectOp>(op))
      return mlir::WalkResult::advance();
    if (isPotentiallyOverlappingSPMCall(op) ||
        !hasLiveSPMStorageAcrossCall(op, timeline, demands))
      return mlir::WalkResult::advance();

    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      if (!mayClobberSPMArena(call, callGraph, mayClobberFunctions))
        return mlir::WalkResult::advance();
    } else if (!mlir::isa<mlir::func::CallIndirectOp>(op)) {
      return mlir::WalkResult::advance();
    }

    op->emitError()
        << "unsupported_spm_planning_scope: live SPM storage crosses a call "
           "that may clobber the independently planned physical SPM arena; "
           "interprocedural arena/resource summaries are not available";
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
    diagnostic << (mlir::isa<mlir::async::FuncOp>(scope)
                       ? "async function exit"
                       : "wafer.tile.region exit")
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
    mp::PathCondition completionPath = point->path;
    for (mlir::Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (mlir::isa<mlir::scf::IfOp>(parent))
        break;
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
        if (isStaticallyNonEmpty(loop))
          if (std::optional<mp::ProgramPoint> loopPoint = timeline.lookup(loop))
            completionPath = loopPoint->path;
        break;
      }
      if (mlir::isa<TileRegionOp, mlir::func::FuncOp, mlir::async::FuncOp>(
              parent))
        break;
    }

    for (mlir::Value token : op.getTokens()) {
      llvm::SmallVector<DTECompletionRef, 2> waitedRefs =
          completionsAt(token, completionPath);
      for (DTECompletionRef waited : waitedRefs) {
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

  void mapForRegionIterArgs(mlir::scf::ForOp forOp) {
    std::optional<mp::ProgramPoint> point = timeline.lookup(forOp);
    if (!point)
      return;
    for (auto [init, iterArg] :
         llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
      if (!hasAsyncTokenType(init))
        continue;
      llvm::SmallVector<DTECompletionRef, 2> refs =
          completionsAt(init, point->path);
      if (!refs.empty())
        completionRefs[iterArg] = std::move(refs);
    }
  }

  mlir::LogicalResult mapForResults(mlir::scf::ForOp forOp) {
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    std::optional<mp::ProgramPoint> yieldPoint =
        yield ? timeline.lookup(yield.getOperation()) : std::nullopt;
    std::optional<mp::ProgramPoint> loopPoint = timeline.lookup(forOp);
    if (!yield || !yieldPoint || !loopPoint)
      return mlir::success();

    for (auto [index, result] : llvm::enumerate(forOp.getResults())) {
      if (!hasAsyncTokenType(result))
        continue;
      if (index >= forOp.getInitArgs().size() ||
          index >= yield->getNumOperands())
        continue;
      llvm::SmallVector<DTECompletionRef, 2> refs =
          completionsAt(yield->getOperand(index), yieldPoint->path);
      bool yieldsLoopLocalCompletion =
          llvm::any_of(refs, [&](const DTECompletionRef &ref) {
            mlir::Operation *origin = ref.originToken.getDefiningOp();
            return origin && forOp->isProperAncestor(origin);
          });
      if (yieldsLoopLocalCompletion && !isStaticallyNonEmpty(forOp))
        return forOp.emitError()
               << "unsupported_async_completion_flow: dynamically optional "
                  "loop-carried DTE issue has no exact completion instance "
                  "proof";
      if (!isStaticallyNonEmpty(forOp)) {
        llvm::SmallVector<DTECompletionRef, 2> initial =
            completionsAt(forOp.getInitArgs()[index], loopPoint->path);
        auto hasSameOrigins = [](llvm::ArrayRef<DTECompletionRef> lhs,
                                 llvm::ArrayRef<DTECompletionRef> rhs) {
          return llvm::all_of(lhs, [&](const DTECompletionRef &ref) {
            return llvm::any_of(rhs, [&](const DTECompletionRef &other) {
              return ref.originToken == other.originToken;
            });
          });
        };
        if (!hasSameOrigins(refs, initial) || !hasSameOrigins(initial, refs))
          return forOp.emitError()
                 << "unsupported_async_completion_flow: dynamically optional "
                    "loop may select different DTE completion identities";
        refs = std::move(initial);
      }
      llvm::SmallVector<DTECompletionRef, 2> relaxed;
      for (DTECompletionRef ref : refs) {
        // The result is selected after the complete recurrence. Loop-internal
        // decisions no longer constrain the dynamic handle it represents;
        // outer structured decisions remain in the loop entry path.
        ref.path = loopPoint->path;
        if (!llvm::any_of(relaxed, [&](const DTECompletionRef &existing) {
              return existing.originToken == ref.originToken &&
                     existing.path == ref.path;
            }))
          relaxed.push_back(ref);
      }
      if (!relaxed.empty())
        completionRefs[result] = std::move(relaxed);
    }
    return mlir::success();
  }

  void mapSingleExecutionRegionBlockArgs(mlir::Operation *operation) {
    std::optional<analysis::SingleExecutionRegionFlow> flow =
        getSingleExecutionRegionFlow(operation);
    std::optional<mp::ProgramPoint> point = timeline.lookup(operation);
    if (!flow || !point)
      return;
    for (auto [input, blockArg] :
         llvm::zip_equal(flow->entryOperands, flow->entryArguments)) {
      if (!hasAsyncTokenType(input))
        continue;
      llvm::SmallVector<DTECompletionRef, 2> refs =
          completionsAt(input, point->path);
      if (!refs.empty())
        completionRefs[blockArg] = std::move(refs);
    }
  }

  void mapSingleExecutionRegionResults(mlir::Operation *operation) {
    std::optional<analysis::SingleExecutionRegionFlow> flow =
        getSingleExecutionRegionFlow(operation);
    if (!flow)
      return;
    std::optional<mp::ProgramPoint> point =
        timeline.lookup(flow->region->front().getTerminator());
    if (!point)
      return;
    for (auto [yielded, result] :
         llvm::zip_equal(flow->exitOperands, flow->results)) {
      if (!hasAsyncTokenType(result))
        continue;
      llvm::SmallVector<DTECompletionRef, 2> refs =
          completionsAt(yielded, point->path);
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
        mapForRegionIterArgs(forOp);
        if (mlir::failed(processRegion(forOp.getRegion())))
          return mlir::failure();
        if (mlir::failed(mapForResults(forOp)))
          return mlir::failure();
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        if (mlir::failed(processRegion(ifOp.getThenRegion())) ||
            mlir::failed(processRegion(ifOp.getElseRegion())))
          return mlir::failure();
        mapIfResults(ifOp);
      } else if (std::optional<analysis::SingleExecutionRegionFlow>
                     flow = getSingleExecutionRegionFlow(&op)) {
        mapSingleExecutionRegionBlockArgs(&op);
        if (mlir::failed(processRegion(*flow->region)))
          return mlir::failure();
        mapSingleExecutionRegionResults(&op);
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

static mlir::LogicalResult initializeSPMDemands(
    mlir::Operation *scope, int64_t defaultAlignment,
    llvm::SmallVectorImpl<mp::LifetimeDemand> &demands) {
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::memref::AllocOp alloc) {
    if (mlir::failed(result))
      return;
    TileRegionOp tileRegion = alloc->getParentOfType<TileRegionOp>();
    if (!tileRegion)
      return;
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(scope)) {
      if (tileRegion->getParentOfType<mlir::func::FuncOp>() != function)
        return;
    } else if (auto region = mlir::dyn_cast<TileRegionOp>(scope)) {
      if (tileRegion != region)
        return;
    } else {
      return;
    }

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

    llvm::SmallVector<int64_t, 2> alignmentRequirements{defaultAlignment};
    if (std::optional<uint64_t> allocAlignment = alloc.getAlignment()) {
      if (*allocAlignment >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        alloc.emitError()
            << "alignment_unsatisfied: memref.alloc alignment exceeds int64";
        result = mlir::failure();
        return;
      }
      alignmentRequirements.push_back(static_cast<int64_t>(*allocAlignment));
    }
    mlir::FailureOr<int64_t> requiredAlignment =
        computeWaferRequiredAlignmentBytes(memrefType, alignmentRequirements);
    if (mlir::failed(requiredAlignment)) {
      alloc.emitError()
          << "alignment_unsatisfied: cannot combine physical encoding, "
             "target, and allocation SPM alignment";
      result = mlir::failure();
      return;
    }

    mp::LifetimeDemand demand;
    demand.allocation = alloc;
    demand.sizeBytes = info->physicalBytes;
    demand.alignmentBytes = *requiredAlignment;
    demand.stableOrdinal = demands.size();
    demands.push_back(std::move(demand));
  });
  return result;
}

static mlir::LogicalResult emitLifetimeFailure(
    mlir::Operation *scope, const mp::LifetimeFailure &failure) {
  mlir::Operation *origin =
      failure.origin ? failure.origin : scope;
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
  case mp::LifetimeFailureKind::LoopCarriedAllocationInstance: {
    mlir::InFlightDiagnostic diagnostic = origin->emitError();
    diagnostic
        << "unsupported_lifetime_alias: loop-body SPM allocation cannot be "
           "loop-carried without multi-instance placement";
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(origin)) {
      diagnostic << "; type=" << allocation.getType();
      llvm::SmallVector<llvm::StringRef, 4> userNames;
      for (mlir::Operation *user : allocation.getResult().getUsers()) {
        if (userNames.size() == 4)
          break;
        userNames.push_back(user->getName().getStringRef());
      }
      if (!userNames.empty()) {
        diagnostic << ", users=[";
        llvm::interleaveComma(userNames, diagnostic);
        diagnostic << "]";
      }
    }
    return mlir::failure();
  }
  case mp::LifetimeFailureKind::MissingAsyncCompletion:
    return origin->emitError()
           << "missing_async_completion: asynchronous SPM access has a "
              "reachable path to wafer.tile.region exit without a proven "
              "completion wait";
  case mp::LifetimeFailureKind::UnsupportedAsyncCompletionFlow:
    return origin->emitError()
           << "unsupported_async_completion_flow: SPM memory planning cannot "
              "prove completion identity through this async handle flow";
  case mp::LifetimeFailureKind::MissingLocalCompletion: {
    std::string originIR;
    llvm::raw_string_ostream originStream(originIR);
    origin->print(originStream);
    mlir::InFlightDiagnostic diagnostic = origin->emitError();
    diagnostic
        << "missing_local_completion: local Compute/Movement issue has a "
           "reachable path to wafer.tile.region exit without "
           "a matching participant in wafer.instr.ncc_join (operation "
        << origin->getName() << "); origin_ir='";
    diagnostic << originStream.str() << "'";
    return mlir::failure();
  }
  case mp::LifetimeFailureKind::LoopBackedgeCompletion:
  {
    std::string originIR;
    llvm::raw_string_ostream originStream(originIR);
    origin->print(originStream);
    mlir::InFlightDiagnostic diagnostic = origin->emitError();
    diagnostic
        << "missing_local_completion: local Compute/Movement issue has a "
           "reachable loop backedge without a same-worker ordered successor "
           "or matching participant join (operation "
        << origin->getName() << "); origin_ir='" << originStream.str() << "'";
    return mlir::failure();
  }
  case mp::LifetimeFailureKind::InconsistentCompletionState:
    return scope->emitError()
           << "completion_proof_failure: local issue lifetime state remains "
              "after all local completion domains were discharged";
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
verifySPMAsyncFunctionClosures(
    mlir::ModuleOp moduleOp,
    const ManagedTimelineMap *managedTimelines = nullptr) {
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::async::FuncOp funcOp) {
    if (mlir::failed(result) || funcOp.isExternal())
      return;

    mp::TimelineFailure timelineFailure;
    std::optional<mp::StructuredTimeline> ownedTimeline;
    const mp::StructuredTimeline *timeline = nullptr;
    if (managedTimelines) {
      auto found = managedTimelines->find(funcOp.getOperation());
      if (found != managedTimelines->end()) {
        const mp::StructuredTimelineAnalysis &analysis = *found->second;
        if (analysis.isValid())
          timeline = &analysis.getTimeline();
        else
          timelineFailure = analysis.getFailure();
      }
    } else {
      mlir::FailureOr<mp::StructuredTimeline> built =
          mp::StructuredTimeline::build(funcOp.getOperation(),
                                        &timelineFailure);
      if (mlir::succeeded(built)) {
        ownedTimeline.emplace(std::move(*built));
        timeline = &*ownedTimeline;
      }
    }
    if (!timeline) {
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
    mp::LocalCompletionTracker localCompletion;
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
planFunction(mlir::func::FuncOp funcOp, int64_t spmBase, int64_t spmLimit,
             int64_t spmAlignment,
             const analysis::DirectCallGraphAnalysis &callGraph,
             const llvm::DenseSet<mlir::Operation *> &mayClobberFunctions,
             llvm::SmallVectorImpl<PendingSPMPlacement> &pendingPlacements,
             SPMMemoryPlanningFailure *failure,
             bool emitCapacityDiagnostics,
             const ManagedTimelineMap *managedTimelines = nullptr) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "transformation-phase", "planFunction(SPM)", "total");
  bool hasTileRegion = false;
  funcOp.walk([&](TileRegionOp) { hasTileRegion = true; });
  if (!hasTileRegion)
    return mlir::success();

  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planFunction(SPM)", "StructuredTimeline");
  mp::TimelineFailure timelineFailure;
  std::optional<mp::StructuredTimeline> ownedTimeline;
  const mp::StructuredTimeline *timeline = nullptr;
  if (managedTimelines) {
    auto found = managedTimelines->find(funcOp.getOperation());
    if (found != managedTimelines->end()) {
      const mp::StructuredTimelineAnalysis &analysis = *found->second;
      if (analysis.isValid())
        timeline = &analysis.getTimeline();
      else
        timelineFailure = analysis.getFailure();
    }
  }
  if (!timeline && !managedTimelines) {
    mlir::FailureOr<mp::StructuredTimeline> built =
        mp::StructuredTimeline::build(funcOp.getOperation(), &timelineFailure);
    if (mlir::succeeded(built)) {
      ownedTimeline.emplace(std::move(*built));
      timeline = &*ownedTimeline;
    }
  }
  if (!timeline) {
    mlir::Operation *origin =
        timelineFailure.origin ? timelineFailure.origin : funcOp.getOperation();
    if (timelineFailure.kind ==
        mp::TimelineFailureKind::DecisionDomainExhausted)
      return origin->emitError()
             << "lifetime_analysis_resource_exhausted: SPM structured "
                "decision identifier domain exhausted";
    if (timelineFailure.kind ==
        mp::TimelineFailureKind::InconsistentPathCondition)
      return origin->emitError()
             << "lifetime_overlap_conflict: SPM memory planning could not "
                "construct a consistent structured path condition";
    return origin->emitError()
           << "unsupported_lifetime_control_flow: SPM memory planning only "
              "supports single-block func.func, non-nested "
              "wafer.tile.region, scf.if and scf.for structured regions";
  }

  llvm::SmallVector<mp::LifetimeDemand, 8> demands;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planFunction(SPM)", "initializeSPMDemands");
  if (mlir::failed(initializeSPMDemands(funcOp.getOperation(), spmAlignment,
                                        demands)))
    return mlir::failure();

  // Preserve the owner-specific DTE completion contract and diagnostic before
  // the shared generic async terminal proof handles other async producers.
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planFunction(SPM)", "lifetime-dataflow");
  DTECompletionTracker dteCompletion(*timeline);
  if (mlir::failed(dteCompletion.run(funcOp.getOperation())))
    return mlir::failure();

  mp::LocalCompletionTracker localCompletion;
  mp::LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return isWaferSPMMemRefType(type);
  });
  mp::LifetimeFailure lifetimeFailure;
  if (mlir::failed(dataflow.run(funcOp.getOperation(), &localCompletion,
                                &lifetimeFailure))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    return emitLifetimeFailure(funcOp.getOperation(), lifetimeFailure);
  }
  if (mlir::failed(verifyLiveSPMAcrossCalls(
          funcOp, *timeline, demands, callGraph, mayClobberFunctions)))
    return mlir::failure();

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planFunction(SPM)", "packStaticMemory");
  mp::PackingResult packing =
      mp::packStaticMemory(demands, mp::ArenaRange{spmBase, spmLimit});
  if (!packing.succeeded()) {
    mlir::Operation *origin = funcOp.getOperation();
    mp::LifetimeDemand *demand = nullptr;
    if (packing.demandIndex && *packing.demandIndex < demands.size()) {
      demand = &demands[*packing.demandIndex];
      origin = demand->allocation.getOperation();
    }
    switch (packing.status) {
    case mp::PackingStatus::ProvenInfeasible: {
      if (failure)
        failure->kind = SPMMemoryPlanningFailureKind::CapacityOverflow;
      std::string diagnosticText;
      llvm::raw_string_ostream diagnostic(diagnosticText);
      diagnostic << "capacity_overflow: SPM planning range [" << spmBase << ", "
                 << spmLimit << ") has no valid static placement";
      if (!packing.capacityConflictDemandIndices.empty()) {
        uint64_t capacityConflictByteTotal = 0;
        if (failure)
          failure->capacityConflictDemands.reserve(
              packing.capacityConflictDemandIndices.size());
        for (unsigned demandIndex : packing.capacityConflictDemandIndices) {
          if (demandIndex >= demands.size())
            return origin->emitError()
                   << "SPM allocator returned an invalid capacity-conflict "
                      "demand index";
          mp::LifetimeDemand &conflictDemand = demands[demandIndex];
          const uint64_t bytes =
              static_cast<uint64_t>(conflictDemand.sizeBytes);
          capacityConflictByteTotal =
              capacityConflictByteTotal >
                      std::numeric_limits<uint64_t>::max() - bytes
                  ? std::numeric_limits<uint64_t>::max()
                  : capacityConflictByteTotal + bytes;
          if (!failure)
            continue;
          SPMMemoryPlanningFailure::DemandEvidence evidence{
              conflictDemand.allocation.getLoc(),
              conflictDemand.allocation.getResult(),
              conflictDemand.allocation.getType(),
              bytes,
              {}};
          for (mlir::Operation *user :
               conflictDemand.allocation.getResult().getUsers()) {
            evidence.userLocations.push_back(user->getLoc());
            evidence.userOperationNames.push_back(user->getName());
          }
          failure->capacityConflictDemands.push_back(std::move(evidence));
        }
        diagnostic << "; capacity_conflict_demands="
                   << packing.capacityConflictDemandIndices.size()
                   << ", capacity_conflict_bytes=" << capacityConflictByteTotal;
      }
      if (!packing.individuallyOversizedDemandIndices.empty()) {
        uint64_t oversizedBytes = 0;
        if (failure)
          failure->individuallyOversizedDemands.reserve(
              packing.individuallyOversizedDemandIndices.size());
        for (unsigned demandIndex :
             packing.individuallyOversizedDemandIndices) {
          if (demandIndex >= demands.size())
            return origin->emitError()
                   << "SPM allocator returned an invalid individually "
                      "oversized demand index";
          mp::LifetimeDemand &oversizedDemand = demands[demandIndex];
          const uint64_t bytes =
              static_cast<uint64_t>(oversizedDemand.sizeBytes);
          oversizedBytes =
              oversizedBytes > std::numeric_limits<uint64_t>::max() - bytes
                  ? std::numeric_limits<uint64_t>::max()
                  : oversizedBytes + bytes;
          if (!failure)
            continue;
          SPMMemoryPlanningFailure::DemandEvidence evidence{
              oversizedDemand.allocation.getLoc(),
              oversizedDemand.allocation.getResult(),
              oversizedDemand.allocation.getType(),
              bytes,
              {}};
          for (mlir::Operation *user :
               oversizedDemand.allocation.getResult().getUsers()) {
            evidence.userLocations.push_back(user->getLoc());
            evidence.userOperationNames.push_back(user->getName());
          }
          failure->individuallyOversizedDemands.push_back(std::move(evidence));
        }
        diagnostic << "; individually_oversized_demands="
                   << packing.individuallyOversizedDemandIndices.size()
                   << ", individually_oversized_bytes=" << oversizedBytes;
      }
      if (demand)
        diagnostic << " for an IR-derived lifetime demand of "
                   << demand->sizeBytes << " bytes, type "
                   << demand->allocation.getType();
      if (!demands.empty()) {
        mp::LifetimeDemand &largest =
            *llvm::max_element(demands, [](const mp::LifetimeDemand &lhs,
                                           const mp::LifetimeDemand &rhs) {
              return lhs.sizeBytes < rhs.sizeBytes;
            });
        if (failure) {
          failure->largestDemandLocation = largest.allocation.getLoc();
          failure->largestDemandType = largest.allocation.getType();
          failure->largestDemandBytes = largest.sizeBytes;
          failure->demandCount = demands.size();
          for (mp::LifetimeDemand &candidate : demands) {
            if (candidate.sizeBytes != largest.sizeBytes)
              continue;
            SPMMemoryPlanningFailure::DemandEvidence evidence{
                candidate.allocation.getLoc(),
                candidate.allocation.getResult(),
                candidate.allocation.getType(),
                static_cast<uint64_t>(candidate.sizeBytes),
                {}};
            for (mlir::Operation *user :
                 candidate.allocation.getResult().getUsers()) {
              evidence.userLocations.push_back(user->getLoc());
              evidence.userOperationNames.push_back(user->getName());
            }
            failure->largestDemands.push_back(std::move(evidence));
          }
        }
        diagnostic << "; demand_count=" << demands.size()
                   << ", largest_demand_bytes=" << largest.sizeBytes
                   << ", largest_demand_type=" << largest.allocation.getType()
                   << ", largest_demand_ties="
                   << llvm::count_if(demands, [&](const auto &candidate) {
                        return candidate.sizeBytes == largest.sizeBytes;
                      });
        llvm::SmallVector<llvm::StringRef, 4> userNames;
        for (mlir::Operation *user :
             largest.allocation.getResult().getUsers()) {
          if (userNames.size() == 4)
            break;
          userNames.push_back(user->getName().getStringRef());
        }
        if (!userNames.empty()) {
          diagnostic << ", largest_demand_users=[";
          llvm::interleaveComma(userNames, diagnostic);
          diagnostic << "]";
        }
        for (mlir::Operation *user :
             largest.allocation.getResult().getUsers()) {
          if (!mlir::isa<InstrGemmOp>(user))
            continue;
          diagnostic << ", gemm_operand_types=[";
          llvm::interleaveComma(user->getOperandTypes(), diagnostic);
          diagnostic << "], gemm_result_types=[";
          llvm::interleaveComma(user->getResultTypes(), diagnostic);
          diagnostic << "]";
          break;
        }
      }
      diagnostic.flush();
      if (emitCapacityDiagnostics)
        origin->emitError(diagnosticText);
      return mlir::failure();
    }
    case mp::PackingStatus::ResourceExhausted:
      return origin->emitError()
             << "packing_search_exhausted: MiniMalloc consumed "
             << packing.searchNodes
             << " deterministic search nodes before producing an SPM "
                "placement or proof";
    case mp::PackingStatus::ArithmeticOverflow:
      return origin->emitError()
             << "range_end_overflow: SPM static packing address arithmetic "
                "overflowed int64";
    case mp::PackingStatus::InvalidProblem:
      return origin->emitError()
             << "invalid_packing_result: SPM static packing input is invalid";
    case mp::PackingStatus::InvalidSolverResult:
      return origin->emitError()
             << "invalid_packing_result: MiniMalloc returned an invalid SPM "
                "placement";
    case mp::PackingStatus::Feasible:
      break;
    }
    llvm_unreachable("successful SPM packing entered failure handling");
  }

  for (const mp::Placement &placement : packing.placements) {
    mp::LifetimeDemand &demand = demands[placement.demandIndex];
    pendingPlacements.push_back(
        PendingSPMPlacement{demand.allocation, placement.offsetBytes});
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult checkTileRegionSPMCapacity(
    TileRegionOp region, int64_t spmBase, int64_t spmLimit,
    int64_t spmAlignment, SPMMemoryPlanningFailure *failure) {
  if (failure)
    *failure = {};
  if (!region || spmBase < 0 || spmLimit <= spmBase || spmAlignment <= 0) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return mlir::failure();
  }

  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::SPMPlanning);
  wafer::support::ScopedCompileTimingSpan timing(
      "analysis", "tile-region-spm-capacity", "region-capacity-query");

  mp::TimelineFailure timelineFailure;
  mlir::FailureOr<mp::StructuredTimeline> timeline =
      mp::StructuredTimeline::build(region.getOperation(), &timelineFailure);
  if (mlir::failed(timeline)) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    mlir::Operation *origin =
        timelineFailure.origin ? timelineFailure.origin : region.getOperation();
    timing.markFailed();
    return origin->emitError()
           << "unsupported_lifetime_control_flow: TileRegion capacity query "
              "requires structured single-block region, scf.if and scf.for "
              "control flow";
  }

  llvm::SmallVector<mp::LifetimeDemand, 8> demands;
  if (mlir::failed(initializeSPMDemands(region.getOperation(), spmAlignment,
                                        demands))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    timing.markFailed();
    return mlir::failure();
  }

  DTECompletionTracker dteCompletion(*timeline);
  if (mlir::failed(dteCompletion.run(region.getOperation()))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    timing.markFailed();
    return mlir::failure();
  }
  mp::LocalCompletionTracker localCompletion;
  mp::LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return isWaferSPMMemRefType(type);
  });
  mp::LifetimeFailure lifetimeFailure;
  if (mlir::failed(dataflow.run(region.getOperation(), &localCompletion,
                                &lifetimeFailure))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::UnsupportedLifetime;
    timing.markFailed();
    return emitLifetimeFailure(region.getOperation(), lifetimeFailure);
  }

  mp::PackingResult packing =
      mp::packStaticMemory(demands, mp::ArenaRange{spmBase, spmLimit});
  if (packing.succeeded())
    return mlir::success();

  timing.markFailed();
  mlir::Operation *origin = region.getOperation();
  if (packing.demandIndex && *packing.demandIndex < demands.size())
    origin = demands[*packing.demandIndex].allocation.getOperation();
  if (packing.status != mp::PackingStatus::ProvenInfeasible) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    if (packing.status == mp::PackingStatus::ResourceExhausted)
      return origin->emitError()
             << "packing_search_exhausted: TileRegion SPM capacity query "
                "exhausted deterministic search";
    return origin->emitError()
           << "invalid_packing_result: TileRegion SPM capacity query failed "
              "without a proven infeasibility certificate";
  }

  if (failure)
    failure->kind = SPMMemoryPlanningFailureKind::CapacityOverflow;
  auto appendEvidence = [](mp::LifetimeDemand &demand) {
    SPMMemoryPlanningFailure::DemandEvidence evidence{
        demand.allocation.getLoc(), demand.allocation.getResult(),
        demand.allocation.getType(),
        static_cast<uint64_t>(demand.sizeBytes), {}};
    for (mlir::Operation *user : demand.allocation.getResult().getUsers()) {
      evidence.userLocations.push_back(user->getLoc());
      evidence.userOperationNames.push_back(user->getName());
    }
    return evidence;
  };
  if (failure) {
    for (unsigned index : packing.capacityConflictDemandIndices)
      if (index < demands.size())
        failure->capacityConflictDemands.push_back(
            appendEvidence(demands[index]));
    for (unsigned index : packing.individuallyOversizedDemandIndices)
      if (index < demands.size())
        failure->individuallyOversizedDemands.push_back(
            appendEvidence(demands[index]));
    failure->demandCount = demands.size();
    if (!demands.empty()) {
      mp::LifetimeDemand &largest =
          *llvm::max_element(demands, [](const mp::LifetimeDemand &lhs,
                                         const mp::LifetimeDemand &rhs) {
            return lhs.sizeBytes < rhs.sizeBytes;
          });
      failure->largestDemandLocation = largest.allocation.getLoc();
      failure->largestDemandType = largest.allocation.getType();
      failure->largestDemandBytes = largest.sizeBytes;
      for (mp::LifetimeDemand &demand : demands)
        if (demand.sizeBytes == largest.sizeBytes)
          failure->largestDemands.push_back(appendEvidence(demand));
    }
  }
  return origin->emitError()
         << "capacity_overflow: TileRegion SPM planning range [" << spmBase
         << ", " << spmLimit
         << ") has no valid static placement; demand_count="
         << demands.size();
}

static mlir::LogicalResult planSPMMemoryModuleImpl(
    mlir::ModuleOp moduleOp, int64_t spmBase, int64_t spmLimit,
    int64_t spmAlignment, SPMMemoryPlanningFailure *failure,
    bool emitCapacityDiagnostics,
    const ManagedTimelineMap *managedTimelines,
    const analysis::DirectCallGraphAnalysis *managedCallGraph) {
  if (failure)
    *failure = {};
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::SPMPlanning);
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "planSPMMemoryModule", "total");
  if (spmBase < 0 || spmLimit <= spmBase) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return moduleOp->emitError()
           << "invalid_spm_range: expected 0 <= spm-base < spm-limit";
  }
  if (spmAlignment <= 0) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return moduleOp->emitError()
           << "alignment_unsatisfied: spm-alignment must be positive";
  }
  auto scopeVerificationTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "analysis-phase", "planSPMMemoryModule", "verify-scopes");
  if (mlir::failed(verifyTileRegionStorageBoundaries(moduleOp))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return mlir::failure();
  }
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
  if (escapedAllocation.wasInterrupted()) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return mlir::failure();
  }
  std::optional<analysis::DirectCallGraphAnalysis> ownedCallGraph;
  if (!managedCallGraph) {
    ownedCallGraph.emplace(moduleOp.getOperation());
    managedCallGraph = &*ownedCallGraph;
  }
  llvm::DenseSet<mlir::Operation *> mayClobberFunctions =
      collectFunctionsThatMayClobberSPMArena(moduleOp, *managedCallGraph);
  if (mlir::failed(verifySPMRegionExecutionScopes(moduleOp)) ||
      mlir::failed(verifySPMValueScopes(moduleOp)) ||
      mlir::failed(verifySPMCallScopes(moduleOp, *managedCallGraph,
                                       mayClobberFunctions))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return mlir::failure();
  }
  if (mlir::failed(
          verifySPMAsyncFunctionClosures(moduleOp, managedTimelines))) {
    if (failure)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return mlir::failure();
  }
  scopeVerificationTiming.reset();

  mlir::LogicalResult result = mlir::success();
  llvm::SmallVector<PendingSPMPlacement, 16> pendingPlacements;
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (mlir::failed(result))
      break;
    if (funcOp.isExternal())
      continue;
    result = planFunction(funcOp, spmBase, spmLimit, spmAlignment,
                          *managedCallGraph,
                          mayClobberFunctions, pendingPlacements, failure,
                          emitCapacityDiagnostics,
                          managedTimelines);
  }
  if (mlir::failed(result)) {
    if (failure && failure->kind == SPMMemoryPlanningFailureKind::None)
      failure->kind = SPMMemoryPlanningFailureKind::Other;
    return result;
  }

  for (PendingSPMPlacement placement : pendingPlacements) {
    placement.allocation->setAttr(
        kWaferSPMOffsetAttrName,
        SPMOffsetAttr::get(placement.allocation.getContext(),
                           placement.offsetBytes));
  }
  return mlir::success();
}

mlir::LogicalResult planSPMMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t spmBase, int64_t spmLimit,
                                        int64_t spmAlignment,
                                        SPMMemoryPlanningFailure *failure,
                                        bool emitCapacityDiagnostics) {
  return planSPMMemoryModuleImpl(moduleOp, spmBase, spmLimit, spmAlignment,
                                 failure, emitCapacityDiagnostics,
                                 /*managedTimelines=*/nullptr,
                                 /*managedCallGraph=*/nullptr);
}

namespace {

struct PlanSPMMemoryPass
    : public impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass> {
  using impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass>::PlanSPMMemoryPassBase;

  PlanSPMMemoryPass(const PlanSPMMemoryPassOptions &options,
                    SPMMemoryPlanningFailure *failure)
      : impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass>(options),
        failure(failure) {}

  void runOnOperation() final {
    unsigned assignedBefore = 0;
    getOperation().walk([&](mlir::memref::AllocOp allocation) {
      assignedBefore += static_cast<bool>(
          allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName));
    });
    ManagedTimelineMap managedTimelines;
    for (mlir::func::FuncOp function :
         getOperation().getOps<mlir::func::FuncOp>()) {
      bool hasTileRegion = false;
      function.walk([&](TileRegionOp) { hasTileRegion = true; });
      if (!function.isExternal() && hasTileRegion)
        managedTimelines.try_emplace(
            function.getOperation(),
            &getChildAnalysis<mp::StructuredTimelineAnalysis>(function));
    }
    getOperation().walk([&](mlir::async::FuncOp function) {
      if (!function.isExternal())
        managedTimelines.try_emplace(
            function.getOperation(),
            &getChildAnalysis<mp::StructuredTimelineAnalysis>(function));
    });
    if (mlir::failed(planSPMMemoryModuleImpl(
            getOperation(), spmBase, spmLimit, spmAlignment,
            failure, emitCapacityDiagnostics, &managedTimelines,
            &getAnalysis<analysis::DirectCallGraphAnalysis>()))) {
      signalPassFailure();
      return;
    }
    numTimelineScopes += managedTimelines.size();
    unsigned assignedAfter = 0;
    getOperation().walk([&](mlir::memref::AllocOp allocation) {
      assignedAfter += static_cast<bool>(
          allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName));
    });
    if (assignedAfter > assignedBefore)
      numAssignedAllocations += assignedAfter - assignedBefore;
    markAnalysesPreserved<mp::StructuredTimelineAnalysis>();
    markAnalysesPreserved<analysis::DirectCallGraphAnalysis>();
  }

private:
  SPMMemoryPlanningFailure *failure = nullptr;
};

} // namespace

std::unique_ptr<mlir::Pass> createPlanSPMMemoryPassWithFailure(
    const PlanSPMMemoryPassOptions &options,
    SPMMemoryPlanningFailure *failure) {
  return std::make_unique<PlanSPMMemoryPass>(options, failure);
}

} // namespace wafer
