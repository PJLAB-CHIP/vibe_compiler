//===- PlanDDRMemory.cpp - Plan Wafer DDR memory ------------===//

#include "Wafer/Transforms/MemoryPlanning.h"
#include "Wafer/Transforms/Passes.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "MemoryPlanning/StaticIndexRange.h"
#include "MemoryPlanning/StaticMemoryPacking.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"

#include "Wafer/Analysis/Module/DirectCallGraphAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace wafer {
#define GEN_PASS_DEF_PLANDDRMEMORYPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

namespace memory_planning = wafer::memory_planning::detail;
using StaticIndexRange = memory_planning::StaticIndexRange;
using ManagedTimelineMap =
    llvm::DenseMap<mlir::Operation *,
                   const memory_planning::StructuredTimelineAnalysis *>;

struct MovementDescriptor {
  int64_t byteCount = 0;
  int64_t innerBytes = 0;
  int64_t byteOffset = 0;
  mlir::DenseI64ArrayAttr strides;
  mlir::DenseI64ArrayAttr iterations;
  llvm::StringRef role;
};

struct DDRView {
  mlir::Value root;
  mlir::MemRefType rootType;
  mlir::MemRefType viewType;
  int64_t minViewOffsetBytes = 0;
  int64_t maxViewOffsetBytes = 0;
  int64_t viewSpanBytes = 0;
  int64_t rootBytes = 0;
};

struct ExternalDDRRootDemand {
  int64_t rootBytes = 0;
};

struct DDRDemandSummary {
  llvm::DenseMap<mlir::Value, ExternalDDRRootDemand> externalRootDemands;
  int64_t plannedHighWaterBytes = 0;
  int64_t bandwidthBytes = 0;
};

using PlannedDDROffsets = llvm::DenseMap<mlir::Operation *, int64_t>;

struct PendingDDRPlacement {
  mlir::memref::AllocOp allocation;
  int64_t offsetBytes = 0;
};

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t>
combinePhysicalAlignment(mlir::MemRefType type, int64_t requestedAlignment,
                         std::optional<uint64_t> explicitAlignment) {
  llvm::SmallVector<int64_t, 2> requirements{requestedAlignment};
  if (explicitAlignment) {
    if (*explicitAlignment >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    requirements.push_back(static_cast<int64_t>(*explicitAlignment));
  }
  mlir::FailureOr<int64_t> combined =
      computeWaferRequiredAlignmentBytes(type, requirements);
  if (mlir::failed(combined))
    return std::nullopt;
  return *combined;
}

static mlir::Value resolveOneTileRegionBoundary(mlir::Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Value entry =
        analysis::getSingleExecutionRegionEntryOperand(blockArg);
    return entry ? entry : value;
  }
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Value exit = result
                         ? analysis::getSingleExecutionRegionExitOperand(result)
                         : mlir::Value{};
  return exit ? exit : value;
}

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    mlir::Value resolved = resolveOneTileRegionBoundary(value);
    if (!resolved || resolved == value)
      break;
    value = resolved;
  }
  return value;
}

static bool isExplicitDDRRoot(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (auto getGlobal = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(def)) {
    auto global =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
            getGlobal, getGlobal.getNameAttr());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    return global && resultType && isWaferDDRMemRefType(resultType) &&
           resultType.hasStaticShape() && global.getType() == resultType &&
           static_cast<bool>(global.getConstantInitValue());
  }

  auto toMemref = mlir::dyn_cast_or_null<mlir::bufferization::ToMemrefOp>(def);
  if (!toMemref)
    return false;
  auto source = mlir::dyn_cast<mlir::BlockArgument>(toMemref.getTensor());
  if (!source || !source.getOwner())
    return false;
  mlir::Operation *parent = source.getOwner()->getParentOp();
  if (!parent || !mlir::isa<mlir::func::FuncOp, mlir::async::FuncOp>(parent) ||
      parent->getNumRegions() != 1 || parent->getRegion(0).empty())
    return false;
  return source.getOwner() == &parent->getRegion(0).front();
}

static bool operationTouchesDDR(mlir::Operation *op) {
  if (llvm::any_of(
          op->getOperandTypes(),
          [](mlir::Type type) { return isWaferDDRMemRefType(type); }) ||
      llvm::any_of(op->getResultTypes(),
                   [](mlir::Type type) { return isWaferDDRMemRefType(type); }))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> resources;
  effects.getEffects(resources);
  return llvm::any_of(resources, [](const auto &effect) {
    return llvm::isa<WaferDDRResource>(effect.getResource());
  });
}

static bool hasExplicitDDRResourceEffect(mlir::Operation *op) {
  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!interface)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<WaferDDRResource>(effect.getResource());
  });
}

static bool functionTouchesDDR(mlir::Operation *functionLike) {
  bool touchesDDR = false;
  functionLike->walk([&](mlir::Operation *op) {
    if (!touchesDDR && operationTouchesDDR(op))
      touchesDDR = true;
  });
  return touchesDDR;
}

static bool
isSupportedDDRAliasCall(mlir::func::CallOp call,
                        const analysis::DirectCallGraphAnalysis &callGraph) {
  if (!memory_planning::isSupportedDirectAliasCall(
          call, [](mlir::Type type) { return isWaferDDRMemRefType(type); }))
    return false;
  mlir::func::FuncOp callee = callGraph.getCallee(call);
  if (!callee)
    return false;
  bool hasDDRResourceEffect = false;
  callee.walk([&](mlir::Operation *op) {
    if (hasDDRResourceEffect)
      return;
    hasDDRResourceEffect = hasExplicitDDRResourceEffect(op);
  });
  return !hasDDRResourceEffect;
}

static mlir::LogicalResult
verifyDDRCallScopes(mlir::ModuleOp moduleOp,
                    const analysis::DirectCallGraphAnalysis &callGraph) {
  bool moduleTouchesDDR = functionTouchesDDR(moduleOp.getOperation());
  llvm::DenseSet<mlir::Operation *> mayTouchDDR;
  moduleOp.walk([&](mlir::func::FuncOp funcOp) {
    if (functionTouchesDDR(funcOp.getOperation()))
      mayTouchDDR.insert(funcOp.getOperation());
  });
  moduleOp.walk([&](mlir::async::FuncOp funcOp) {
    if (functionTouchesDDR(funcOp.getOperation()))
      mayTouchDDR.insert(funcOp.getOperation());
  });
  bool changed = true;
  while (changed) {
    changed = false;
    moduleOp.walk([&](mlir::func::CallOp call) {
      mlir::Operation *caller = call->getParentOfType<mlir::func::FuncOp>();
      if (!caller)
        caller = call->getParentOfType<mlir::async::FuncOp>();
      mlir::func::FuncOp callee = callGraph.getCallee(call);
      if (caller && callee && mayTouchDDR.contains(callee.getOperation()) &&
          mayTouchDDR.insert(caller).second)
        changed = true;
    });
  }

  mlir::WalkResult result = moduleOp.walk([&](mlir::func::CallOp call) {
    mlir::func::FuncOp callee = callGraph.getCallee(call);
    bool carriesDDR = llvm::any_of(call->getOperandTypes(),
                                   [](mlir::Type type) {
                                     return isWaferDDRMemRefType(type);
                                   }) ||
                      llvm::any_of(call->getResultTypes(), [](mlir::Type type) {
                        return isWaferDDRMemRefType(type);
                      });
    if (isSupportedDDRAliasCall(call, callGraph))
      return mlir::WalkResult::advance();
    bool lacksResourceSummary =
        moduleTouchesDDR && (!callee || callee.isExternal());
    if (!carriesDDR && !lacksResourceSummary &&
        (!callee || !mayTouchDDR.contains(callee.getOperation())))
      return mlir::WalkResult::advance();
    call.emitError()
        << "unsupported_ddr_planning_scope: func.call across a DDR planning "
           "scope requires an interprocedural arena and resource summary";
    return mlir::WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return mlir::failure();

  result = moduleOp.walk([&](mlir::func::CallIndirectOp call) {
    if (!moduleTouchesDDR)
      return mlir::WalkResult::advance();
    call.emitError()
        << "unsupported_ddr_planning_scope: indirect calls in a module with "
           "DDR demands have no verifiable arena/resource summary";
    return mlir::WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return mlir::failure();

  result = moduleOp.walk([&](mlir::async::CallOp call) {
    if (!moduleTouchesDDR)
      return mlir::WalkResult::advance();
    mlir::async::FuncOp callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::async::FuncOp>(
            call, call.getCalleeAttr());
    if (callee && !callee.isExternal())
      return mlir::WalkResult::advance();
    call.emitError()
        << "unsupported_ddr_planning_scope: external async.call in a module "
           "with DDR demands has no verifiable arena/resource summary";
    return mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::Value getRootViewSource(mlir::Value value) {
  value = resolveTileRegionBoundaryValue(value);
  while (mlir::Operation *def = value.getDefiningOp()) {
    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def);
    if (!viewLike)
      return value;
    mlir::Value source =
        resolveTileRegionBoundaryValue(viewLike.getViewSource());
    if (source == value)
      return value;
    value = source;
  }
  return value;
}

static bool isCompilerManagedDDRRoot(mlir::Value root) {
  mlir::Operation *def = root.getDefiningOp();
  return def && mlir::isa<mlir::memref::AllocOp>(def) &&
         isWaferDDRMemRefType(root.getType());
}

static mlir::LogicalResult
getStaticMemrefViewInfo(mlir::Operation *op, mlir::MemRefType type,
                        llvm::SmallVectorImpl<int64_t> &strides,
                        int64_t &offset, llvm::StringRef role,
                        bool allowDynamicOffset = false,
                        bool *hadDynamicOffset = nullptr) {
  if (!type.hasStaticShape())
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view must have static shape";
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()))
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view must have static strided layout";
  if (hadDynamicOffset)
    *hadDynamicOffset = offset == mlir::ShapedType::kDynamic;
  if (offset == mlir::ShapedType::kDynamic && allowDynamicOffset)
    offset = 0;
  else if (offset == mlir::ShapedType::kDynamic || offset < 0)
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view must have static non-negative offset";
  for (int64_t stride : strides) {
    if (stride == mlir::ShapedType::kDynamic || stride < 0)
      return op->emitError()
             << "unsupported_ddr_view: " << role
             << " DDR view must have static non-negative strides";
  }
  return mlir::success();
}

static mlir::FailureOr<StaticIndexRange>
getStaticIndexRange(mlir::Operation *anchor, mlir::OpFoldResult offset,
                    llvm::StringRef role) {
  if (mlir::Attribute attribute = offset.dyn_cast<mlir::Attribute>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(attribute);
    if (!attr)
      return anchor->emitError() << "unsupported_ddr_view: " << role
                                 << " static offset is not an integer";
    int64_t value = attr.getInt();
    if (value < 0)
      return anchor->emitError() << "unsupported_ddr_view: " << role
                                 << " offset must be non-negative";
    return StaticIndexRange{value, value};
  }

  mlir::Value dynamicOffset = mlir::cast<mlir::Value>(offset);
  memory_planning::StaticIndexRangeResult result =
      memory_planning::evaluateNonNegativeStaticIndexRange(dynamicOffset,
                                                           anchor);
  using Failure = memory_planning::StaticIndexRangeFailureKind;
  switch (result.failure) {
  case Failure::None:
    if (result.range.empty)
      return anchor->emitError()
             << "unsupported_ddr_view: " << role
             << " scf.for offset range must be non-empty and non-negative";
    return result.range;
  case Failure::DynamicLoopBounds:
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " dynamic offset requires constant scf.for bounds and step";
  case Failure::InvalidLoopBounds:
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " scf.for offset range must be non-empty and non-negative";
  case Failure::NonSingletonMultiplication:
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " dynamic offset multiplication must have a static operand";
  case Failure::InvalidUnsignedDivision:
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " unsigned-divide offset must be a non-negative static value";
  case Failure::ArithmeticOverflow:
    return anchor->emitError()
           << "range_end_overflow: dynamic DDR view offset expression "
              "overflows int64";
  case Failure::NegativeRange:
    return anchor->emitError() << "unsupported_ddr_view: " << role
                               << " dynamic offset range must be non-negative";
  case Failure::UnsupportedExpression:
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " dynamic offset must be a supported statically bounded index "
              "expression; root expression is "
           << (dynamicOffset.getDefiningOp()
                   ? dynamicOffset.getDefiningOp()->getName().getStringRef()
                   : llvm::StringRef("block argument"));
  }
  llvm_unreachable("unhandled static index range failure");
}

class ViewElementOffsetRangeEvaluator {
public:
  ViewElementOffsetRangeEvaluator(mlir::Operation *anchor,
                                  llvm::ArrayRef<mlir::Value> roots,
                                  llvm::StringRef role)
      : anchor(anchor), role(role) {
    for (mlir::Value root : roots)
      originRoots.insert(resolveTileRegionBoundaryValue(root));
  }

  mlir::FailureOr<StaticIndexRange> evaluate(mlir::Value view) {
    view = resolveTileRegionBoundaryValue(view);
    if (originRoots.contains(view))
      return StaticIndexRange{};
    if (auto it = cache.find(view); it != cache.end())
      return it->second;
    if (!active.insert(view).second)
      return anchor->emitError()
             << "unsupported_ddr_view: non-identity scf.for carried view "
                "recurrence has no finite static offset proof";

    mlir::FailureOr<StaticIndexRange> result = evaluateImpl(view);
    active.erase(view);
    if (mlir::succeeded(result))
      cache.try_emplace(view, *result);
    return result;
  }

private:
  static StaticIndexRange unite(StaticIndexRange lhs, StaticIndexRange rhs) {
    if (lhs.empty)
      return rhs;
    if (rhs.empty)
      return lhs;
    return StaticIndexRange{std::min(lhs.min, rhs.min),
                            std::max(lhs.max, rhs.max), /*empty=*/false};
  }

  static std::optional<unsigned>
  getAddressPreservingCarriedIndex(mlir::scf::ForOp forOp, mlir::Value value) {
    llvm::DenseSet<mlir::Value> visited;
    while (visited.insert(value).second) {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        if (blockArg.getOwner() == forOp.getBody() &&
            blockArg.getArgNumber() > 0) {
          unsigned index = blockArg.getArgNumber() - 1;
          if (index < forOp.getInitArgs().size())
            return index;
        }
        return std::nullopt;
      }
      if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
        value = cast.getSource();
        continue;
      }
      if (auto collapse =
              value.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
        value = collapse.getSrc();
        continue;
      }
      if (auto expand = value.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
        value = expand.getSrc();
        continue;
      }
      if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
        if (auto nestedFor =
                mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
          unsigned index = result.getResultNumber();
          auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
              nestedFor.getBody()->getTerminator());
          if (!yield || index >= nestedFor.getInitArgs().size() ||
              index >= yield.getNumOperands())
            return std::nullopt;

          mlir::Value yielded = yield.getOperand(index);
          while (true) {
            if (auto cast = yielded.getDefiningOp<mlir::memref::CastOp>()) {
              yielded = cast.getSource();
              continue;
            }
            if (auto collapse =
                    yielded.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
              yielded = collapse.getSrc();
              continue;
            }
            if (auto expand =
                    yielded.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
              yielded = expand.getSrc();
              continue;
            }
            break;
          }
          std::optional<unsigned> nestedIndex =
              getAddressPreservingCarriedIndex(nestedFor, yielded);
          if (!nestedIndex || *nestedIndex != index)
            return std::nullopt;
          value = nestedFor.getInitArgs()[index];
          continue;
        }
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  mlir::FailureOr<StaticIndexRange> evaluateCarriedValue(mlir::scf::ForOp forOp,
                                                         unsigned index,
                                                         bool includeInit,
                                                         bool includeBackedge) {
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield || index >= forOp.getInitArgs().size() ||
        index >= yield.getNumOperands())
      return anchor->emitError()
             << "unsupported_ddr_view: malformed scf.for carried view";

    StaticIndexRange combined{/*min=*/0, /*max=*/0, /*empty=*/true};
    auto includeInitAt = [&](unsigned carriedIndex) -> mlir::LogicalResult {
      mlir::FailureOr<StaticIndexRange> init =
          evaluate(forOp.getInitArgs()[carriedIndex]);
      if (mlir::failed(init))
        return mlir::failure();
      combined = unite(combined, *init);
      return mlir::success();
    };

    if (includeInit && mlir::failed(includeInitAt(index)))
      return mlir::failure();
    if (!includeBackedge)
      return combined;

    // Rotating-buffer pipelines carry a finite family of loop-external buffers
    // through scf.for iter_args.  Follow that address-preserving permutation
    // as a finite index graph and union its init ranges instead of treating
    // the graph cycle as an unbounded address recurrence.  Any subview or
    // other offset-producing edge deliberately falls through to evaluate(),
    // where a true cyclic recurrence still fails closed.
    llvm::SmallVector<unsigned, 4> pending{index};
    llvm::DenseSet<unsigned> visited;
    while (!pending.empty()) {
      unsigned carriedIndex = pending.pop_back_val();
      if (!visited.insert(carriedIndex).second)
        continue;
      mlir::Value backedge = yield.getOperand(carriedIndex);
      if (std::optional<unsigned> target =
              getAddressPreservingCarriedIndex(forOp, backedge)) {
        if (mlir::failed(includeInitAt(*target)))
          return mlir::failure();
        pending.push_back(*target);
        continue;
      }
      mlir::FailureOr<StaticIndexRange> yielded = evaluate(backedge);
      if (mlir::failed(yielded))
        return mlir::failure();
      combined = unite(combined, *yielded);
    }
    return combined;
  }

  mlir::FailureOr<StaticIndexRange> evaluateImpl(mlir::Value view) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(view)) {
      auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          blockArg.getOwner()->getParentOp());
      if (forOp && blockArg.getArgNumber() > 0) {
        unsigned index = blockArg.getArgNumber() - 1;
        // A loop body iter-arg can observe the init value on the first
        // iteration and the yielded value on every later iteration.
        return evaluateCarriedValue(forOp, index, /*includeInit=*/true,
                                    /*includeBackedge=*/true);
      }
    }

    if (auto result = mlir::dyn_cast<mlir::OpResult>(view)) {
      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
        if (result.getResultNumber() < forOp.getInitArgs().size()) {
          std::optional<int64_t> lower =
              mlir::getConstantIntValue(forOp.getLowerBound());
          std::optional<int64_t> upper =
              mlir::getConstantIntValue(forOp.getUpperBound());
          std::optional<int64_t> step =
              mlir::getConstantIntValue(forOp.getStep());
          if (lower && upper && step && *step > 0) {
            if (*lower < *upper)
              return evaluateCarriedValue(forOp, result.getResultNumber(),
                                          /*includeInit=*/false,
                                          /*includeBackedge=*/true);
            return evaluateCarriedValue(forOp, result.getResultNumber(),
                                        /*includeInit=*/true,
                                        /*includeBackedge=*/false);
          }
          // Without a static trip-count classification, the loop result may
          // be either its init value or a value observed on the backedge.
          return evaluateCarriedValue(forOp, result.getResultNumber(),
                                      /*includeInit=*/true,
                                      /*includeBackedge=*/true);
        }
      }
    }

    if (auto collapse = view.getDefiningOp<mlir::memref::CollapseShapeOp>())
      return evaluate(collapse.getSrc());
    if (auto expand = view.getDefiningOp<mlir::memref::ExpandShapeOp>())
      return evaluate(expand.getSrc());
    if (auto cast = view.getDefiningOp<mlir::memref::CastOp>())
      return evaluate(cast.getSource());

    auto subview = view.getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview)
      return anchor->emitError()
             << "unsupported_ddr_view: cannot prove " << role
             << " dynamic view offset to its DDR root";

    mlir::FailureOr<StaticIndexRange> total = evaluate(subview.getSource());
    if (mlir::failed(total))
      return mlir::failure();
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
    llvm::SmallVector<int64_t, 4> sourceStrides;
    int64_t sourceOffset = 0;
    if (!sourceType ||
        mlir::failed(getStaticMemrefViewInfo(anchor, sourceType, sourceStrides,
                                             sourceOffset, role,
                                             /*allowDynamicOffset=*/true)))
      return mlir::failure();
    if (sourceStrides.size() != subview.getMixedOffsets().size())
      return anchor->emitError()
             << "unsupported_ddr_view: dynamic subview rank mismatch";

    for (auto [offset, stride] :
         llvm::zip(subview.getMixedOffsets(), sourceStrides)) {
      mlir::FailureOr<StaticIndexRange> range =
          getStaticIndexRange(anchor, offset, role);
      if (mlir::failed(range))
        return mlir::failure();
      int64_t minContribution = 0;
      int64_t maxContribution = 0;
      if (!checkedMul(range->min, stride, minContribution) ||
          !checkedMul(range->max, stride, maxContribution) ||
          !checkedAdd(total->min, minContribution, total->min) ||
          !checkedAdd(total->max, maxContribution, total->max))
        return anchor->emitError()
               << "range_end_overflow: DDR view offset range overflows int64";
    }
    return total;
  }

  mlir::Operation *anchor;
  llvm::StringRef role;
  llvm::DenseSet<mlir::Value> originRoots;
  llvm::DenseMap<mlir::Value, StaticIndexRange> cache;
  llvm::DenseSet<mlir::Value> active;
};

static mlir::FailureOr<StaticIndexRange>
getViewElementOffsetRange(mlir::Operation *anchor, mlir::Value view,
                          llvm::ArrayRef<mlir::Value> roots,
                          llvm::StringRef role) {
  return ViewElementOffsetRangeEvaluator(anchor, roots, role).evaluate(view);
}

static mlir::FailureOr<int64_t>
getDescriptorPayloadBytes(mlir::Operation *op,
                          const MovementDescriptor &descriptor) {
  int64_t payload = descriptor.innerBytes;
  for (int64_t iteration : descriptor.iterations.asArrayRef()) {
    if (!checkedMul(payload, iteration, payload))
      return op->emitError()
             << "descriptor_payload_mismatch: " << descriptor.role
             << " descriptor payload bytes overflow int64";
  }
  return payload;
}

static mlir::FailureOr<int64_t>
getDescriptorLocalEnd(mlir::Operation *op,
                      const MovementDescriptor &descriptor) {
  int64_t end = 0;
  if (!checkedAdd(descriptor.byteOffset, descriptor.innerBytes, end))
    return op->emitError() << "range_end_overflow: " << descriptor.role
                           << " DDR descriptor byte range overflows int64";
  for (auto [stride, iteration] :
       llvm::zip(descriptor.strides.asArrayRef(),
                 descriptor.iterations.asArrayRef())) {
    int64_t span = 0;
    if (!checkedMul(stride, iteration - 1, span) || !checkedAdd(end, span, end))
      return op->emitError() << "range_end_overflow: " << descriptor.role
                             << " DDR descriptor byte range overflows int64";
  }
  return end;
}

static mlir::LogicalResult verifyDDRRoot(mlir::Operation *op, mlir::Value root,
                                         int64_t defaultAlignment,
                                         const PlannedDDROffsets &offsets) {
  mlir::Operation *def = root.getDefiningOp();
  if (!def)
    return mlir::success();

  auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(def);
  if (!alloc || !isWaferDDRMemRefType(root.getType()))
    return mlir::success();

  auto offsetIt = offsets.find(alloc.getOperation());
  if (offsetIt == offsets.end())
    return op->emitError()
           << "ddr_planned_range_missing: compiler-managed DDR allocation "
              "has no accepted wafer.ddr.offset";
  int64_t offset = offsetIt->second;

  std::optional<int64_t> physicalAlignment = combinePhysicalAlignment(
      alloc.getType(), defaultAlignment, alloc.getAlignment());
  if (!physicalAlignment)
    return op->emitError()
           << "ddr_alignment_failure: cannot combine target and physical "
              "encoding DDR alignment";
  int64_t requiredAlignment = *physicalAlignment;

  if (requiredAlignment <= 0 || offset % requiredAlignment != 0)
    return op->emitError()
           << "ddr_alignment_failure: accepted DDR offset does not satisfy "
              "required alignment";

  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<DDRView, 2>>
resolveDDRViews(mlir::Operation *op, mlir::Value ddrValue,
                const MovementDescriptor &descriptor, int64_t defaultAlignment,
                const PlannedDDROffsets &offsets,
                const memory_planning::StructuredTimeline &timeline,
                const memory_planning::LifetimeDataflow &dataflow) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(ddrValue.getType());
  if (!viewType || !isWaferDDRMemRefType(viewType))
    return op->emitError() << "unsupported_ddr_view: " << descriptor.role
                           << " operand must be a Wafer DDR memref";

  llvm::SmallVector<int64_t, 4> viewStrides;
  int64_t viewOffsetElements = 0;
  bool hasDynamicViewOffset = false;
  if (mlir::failed(getStaticMemrefViewInfo(
          op, viewType, viewStrides, viewOffsetElements, descriptor.role,
          /*allowDynamicOffset=*/true, &hasDynamicViewOffset)))
    return mlir::failure();

  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  if (!viewInfo || viewInfo->physicalBytes < 0 ||
      (!viewInfo->bitPackedElement && viewInfo->elementBytes <= 0))
    return op->emitError() << "unsupported_ddr_view: cannot compute "
                           << descriptor.role << " DDR view physical bytes";

  std::optional<memory_planning::ProgramPoint> point = timeline.lookup(op);
  if (!point)
    return op->emitError()
           << "lifetime_overlap_conflict: missing event for DDR descriptor";

  llvm::SmallVector<mlir::Value, 2> roots;
  for (memory_planning::ValueOriginRef origin :
       dataflow.originsAt(ddrValue, point->path)) {
    mlir::Value root = getRootViewSource(origin.root);
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  }
  if (roots.empty())
    return op->emitError() << "unsupported_ddr_view: cannot resolve "
                           << descriptor.role << " DDR view origin";

  StaticIndexRange staticOffsetRange{viewOffsetElements, viewOffsetElements};
  mlir::FailureOr<StaticIndexRange> dynamicOffsetRange = staticOffsetRange;
  if (hasDynamicViewOffset)
    dynamicOffsetRange =
        getViewElementOffsetRange(op, ddrValue, roots, descriptor.role);
  if (mlir::failed(dynamicOffsetRange))
    return mlir::failure();
  const StaticIndexRange &viewOffsetElementsRange = *dynamicOffsetRange;

  llvm::SmallVector<DDRView, 2> views;
  for (mlir::Value root : roots) {
    auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
    if (!rootType || !isWaferDDRMemRefType(rootType))
      return op->emitError() << "unsupported_ddr_view: " << descriptor.role
                             << " DDR view root must be a Wafer DDR memref";

    std::optional<WaferPhysicalTensorInfo> rootInfo =
        computeWaferPhysicalTensorInfo(rootType);
    if (!rootInfo || rootInfo->physicalBytes < 0 ||
        (!rootInfo->bitPackedElement && rootInfo->elementBytes <= 0))
      return op->emitError() << "unsupported_ddr_view: cannot compute "
                             << descriptor.role << " DDR root physical bytes";

    int64_t minViewOffsetBytes = 0;
    int64_t maxViewOffsetBytes = 0;
    if (viewInfo->bitPackedElement || rootInfo->bitPackedElement) {
      if (!viewInfo->bitPackedElement || !rootInfo->bitPackedElement)
        return op->emitError()
               << "unsupported_ddr_view: DDR view and root bitpacking differ";
      if (viewOffsetElementsRange.min != 0 || viewOffsetElementsRange.max != 0)
        return op->emitError()
               << "unsupported_ddr_view: bitpacked DDR view must have zero "
                  "element offset";
    } else {
      if (viewInfo->elementBytes != rootInfo->elementBytes)
        return op->emitError()
               << "unsupported_ddr_view: DDR view and root element byte sizes "
                  "differ";
      if (!checkedMul(viewOffsetElementsRange.min, viewInfo->elementBytes,
                      minViewOffsetBytes) ||
          !checkedMul(viewOffsetElementsRange.max, viewInfo->elementBytes,
                      maxViewOffsetBytes))
        return op->emitError()
               << "range_end_overflow: DDR view byte offset overflows int64";
    }

    if (mlir::failed(verifyDDRRoot(op, root, defaultAlignment, offsets)))
      return mlir::failure();
    views.push_back(DDRView{root, rootType, viewType, minViewOffsetBytes,
                            maxViewOffsetBytes, viewInfo->physicalBytes,
                            rootInfo->physicalBytes});
  }
  return views;
}

static mlir::LogicalResult
collectDDRDescriptorDemand(mlir::Operation *op, mlir::Value ddrValue,
                           const MovementDescriptor &descriptor,
                           int64_t defaultAlignment,
                           const PlannedDDROffsets &offsets,
                           const memory_planning::StructuredTimeline &timeline,
                           const memory_planning::LifetimeDataflow &dataflow,
                           DDRDemandSummary &summary) {
  mlir::FailureOr<int64_t> payload = getDescriptorPayloadBytes(op, descriptor);
  if (mlir::failed(payload))
    return mlir::failure();
  if (*payload != descriptor.byteCount)
    return op->emitError() << "descriptor_payload_mismatch: " << descriptor.role
                           << " byte_count " << descriptor.byteCount
                           << " does not match inner_bytes * iterations "
                           << *payload;

  mlir::FailureOr<int64_t> localEnd = getDescriptorLocalEnd(op, descriptor);
  if (mlir::failed(localEnd))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<DDRView, 2>> views = resolveDDRViews(
      op, ddrValue, descriptor, defaultAlignment, offsets, timeline, dataflow);
  if (mlir::failed(views))
    return mlir::failure();

  for (const DDRView &view : *views) {
    if (*localEnd > view.viewSpanBytes)
      return op->emitError() << "ddr_range_overflow: " << descriptor.role
                             << " descriptor byte range " << *localEnd
                             << " exceeds DDR view span " << view.viewSpanBytes;

    int64_t absoluteEnd = 0;
    if (!checkedAdd(view.maxViewOffsetBytes, *localEnd, absoluteEnd))
      return op->emitError()
             << "range_end_overflow: DDR access end overflows int64";
    if (absoluteEnd > view.rootBytes) {
      mlir::InFlightDiagnostic diagnostic =
          op->emitError() << "ddr_range_overflow: " << descriptor.role
                          << " access end " << absoluteEnd
                          << " exceeds DDR root byte size " << view.rootBytes
                          << " (view max offset " << view.maxViewOffsetBytes
                          << ", descriptor local end " << *localEnd
                          << ", view span " << view.viewSpanBytes
                          << ", value type " << ddrValue.getType()
                          << ", root type " << view.root.getType() << ")";
      if (auto subview = ddrValue.getDefiningOp<mlir::memref::SubViewOp>()) {
        std::string detail;
        llvm::raw_string_ostream stream(detail);
        stream << "; view definition=";
        subview->print(stream, mlir::OpPrintingFlags().skipRegions());
        for (auto [index, offset] :
             llvm::enumerate(subview.getMixedOffsets())) {
          auto dynamic = mlir::dyn_cast<mlir::Value>(offset);
          if (!dynamic)
            continue;
          stream << "; dynamic offset #" << index << '=';
          if (mlir::Operation *definition = dynamic.getDefiningOp())
            definition->print(stream, mlir::OpPrintingFlags().skipRegions());
          else if (auto argument =
                       mlir::dyn_cast<mlir::BlockArgument>(dynamic)) {
            stream << "block argument #" << argument.getArgNumber();
            if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                    argument.getOwner()->getParentOp()))
              stream << " of scf.for(lower=" << loop.getLowerBound()
                     << ", upper=" << loop.getUpperBound()
                     << ", step=" << loop.getStep() << ')';
          }
        }
        stream.flush();
        diagnostic << detail;
      }
      return mlir::failure();
    }

    if (!isCompilerManagedDDRRoot(view.root)) {
      auto [it, inserted] = summary.externalRootDemands.try_emplace(view.root);
      ExternalDDRRootDemand &rootDemand = it->second;
      if (inserted)
        rootDemand.rootBytes = view.rootBytes;
    }
  }

  if (!checkedAdd(summary.bandwidthBytes, descriptor.byteCount,
                  summary.bandwidthBytes))
    return op->emitError()
           << "bandwidth_pressure_too_high: DDR bandwidth byte sum overflows";
  return mlir::success();
}

static mlir::LogicalResult
emitLifetimeFailure(mlir::Operation *scope,
                    const memory_planning::LifetimeFailure &failure) {
  mlir::Operation *origin = failure.origin ? failure.origin : scope;
  switch (failure.kind) {
  case memory_planning::LifetimeFailureKind::MissingAllocationEvent:
    return origin->emitError()
           << "lifetime_overlap_conflict: missing event for DDR allocation";
  case memory_planning::LifetimeFailureKind::UnsupportedTrackedValueProducer: {
    mlir::InFlightDiagnostic diagnostic =
        origin->emitError()
        << "unsupported_lifetime_alias: DDR memref producers must be "
           "memref.alloc or implement a supported alias/control-flow "
           "interface (operation "
        << origin->getName();
    if (auto toMemref =
            mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(origin)) {
      mlir::Value tensor = toMemref.getTensor();
      if (mlir::Operation *producer = tensor.getDefiningOp())
        diagnostic << ", tensor producer " << producer->getName();
      else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
        mlir::Operation *parent = argument.getOwner()->getParentOp();
        diagnostic << ", tensor block argument parent "
                   << (parent ? parent->getName().getStringRef()
                              : llvm::StringRef("<none>"));
      }
    }
    diagnostic << ")";
    return mlir::failure();
  }
  case memory_planning::LifetimeFailureKind::UnsupportedTrackedValueEscape:
    return origin->emitError()
           << "unsupported_lifetime_alias: tracked DDR storage cannot escape "
              "through raw metadata or an operation without supported "
              "alias/effect semantics (operation "
           << origin->getName() << ")";
  case memory_planning::LifetimeFailureKind::LoopCarriedAllocationInstance:
    return origin->emitError()
           << "unsupported_lifetime_alias: loop-body DDR allocation cannot be "
              "loop-carried without multi-instance placement";
  case memory_planning::LifetimeFailureKind::MissingAsyncCompletion:
    return origin->emitError()
           << "missing_async_completion: asynchronous DDR access has a "
              "reachable path to entry exit without a proven completion "
              "wait";
  case memory_planning::LifetimeFailureKind::UnsupportedAsyncCompletionFlow:
    return origin->emitError()
           << "unsupported_async_completion_flow: DDR memory planning cannot "
              "prove completion identity through this async handle flow";
  case memory_planning::LifetimeFailureKind::MissingLocalCompletion:
    return origin->emitError()
           << "missing_local_completion: DDR-touching local Movement issue "
              "has a reachable path to entry exit without "
              "a matching participant in wafer.instr.ncc_join";
  case memory_planning::LifetimeFailureKind::LoopBackedgeCompletion:
    return origin->emitError()
           << "missing_local_completion: DDR-touching local Movement issue "
              "reaches an scf.for backedge without a same-worker ordered "
              "successor or matching participant join";
  case memory_planning::LifetimeFailureKind::InconsistentCompletionState:
    return origin->emitError()
           << "completion_proof_failure: DDR local issue lifetime state "
              "remains after all local completion domains were discharged";
  }
  llvm_unreachable("unknown DDR lifetime failure");
}

static mlir::LogicalResult verifyDDRAsyncFunctionClosures(
    mlir::ModuleOp moduleOp,
    const ManagedTimelineMap *managedTimelines = nullptr) {
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::async::FuncOp funcOp) {
    if (mlir::failed(result) || funcOp.isExternal())
      return;

    memory_planning::TimelineFailure timelineFailure;
    std::optional<memory_planning::StructuredTimeline> ownedTimeline;
    const memory_planning::StructuredTimeline *timeline = nullptr;
    if (managedTimelines) {
      auto found = managedTimelines->find(funcOp.getOperation());
      if (found != managedTimelines->end()) {
        const memory_planning::StructuredTimelineAnalysis &analysis =
            *found->second;
        if (analysis.isValid())
          timeline = &analysis.getTimeline();
        else
          timelineFailure = analysis.getFailure();
      }
    } else {
      mlir::FailureOr<memory_planning::StructuredTimeline> built =
          memory_planning::StructuredTimeline::build(funcOp.getOperation(),
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
               << "unsupported_async_completion_flow: DDR async callee "
                  "completion proof requires supported single-block "
                  "structured control flow";
      return;
    }

    llvm::SmallVector<memory_planning::LifetimeDemand, 0> demands;
    memory_planning::LocalCompletionTracker localCompletion;
    memory_planning::LifetimeDataflow dataflow(
        *timeline, demands,
        [](mlir::Type type) { return isWaferDDRMemRefType(type); });
    memory_planning::LifetimeFailure lifetimeFailure;
    if (mlir::failed(dataflow.run(funcOp.getOperation(), &localCompletion,
                                  &lifetimeFailure))) {
      result = emitLifetimeFailure(funcOp.getOperation(), lifetimeFailure);
      return;
    }

    mlir::Operation *unsupportedDescriptor = nullptr;
    funcOp.walk([&](mlir::Operation *op) {
      if (!unsupportedDescriptor && operationTouchesDDR(op) &&
          hasExplicitDDRResourceEffect(op))
        unsupportedDescriptor = op;
    });
    if (unsupportedDescriptor)
      result = unsupportedDescriptor->emitError()
               << "unsupported_ddr_planning_scope: DDR resource effects in "
                  "async.func require a call-aware descriptor and bandwidth "
                  "summary";
  });
  return result;
}

static mlir::LogicalResult initializeDDRDemand(
    mlir::memref::AllocOp alloc, int64_t defaultAlignment,
    const memory_planning::StructuredTimeline &timeline,
    llvm::SmallVectorImpl<memory_planning::LifetimeDemand> &demands) {
  mlir::MemRefType memrefType = alloc.getType();
  if (!isWaferDDRMemRefType(memrefType))
    return mlir::success();

  if (!alloc.getDynamicSizes().empty() || !alloc.getSymbolOperands().empty())
    return alloc.emitError()
           << "unsupported_compiler_managed_ddr: DDR memory planning requires "
              "static memref.alloc sizes and symbols";

  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0)
    return alloc.emitError()
           << "unsupported_compiler_managed_ddr: cannot compute physical DDR "
              "byte size for "
           << memrefType;

  std::optional<int64_t> physicalAlignment = combinePhysicalAlignment(
      memrefType, defaultAlignment, alloc.getAlignment());
  if (!physicalAlignment)
    return alloc.emitError()
           << "ddr_alignment_failure: cannot combine target and physical "
              "encoding DDR alignment";
  int64_t requiredAlignment = *physicalAlignment;

  std::optional<memory_planning::ProgramPoint> allocationPoint =
      timeline.lookup(alloc.getOperation());
  if (!allocationPoint)
    return alloc.emitError()
           << "lifetime_overlap_conflict: missing event for DDR allocation";

  demands.push_back(
      memory_planning::LifetimeDemand{alloc,
                                      info->physicalBytes,
                                      requiredAlignment,
                                      static_cast<unsigned>(demands.size()),
                                      *allocationPoint,
                                      {}});
  return mlir::success();
}

static mlir::LogicalResult collectDDRDemands(
    mlir::Operation *scope, int64_t defaultAlignment,
    const memory_planning::StructuredTimeline &timeline,
    llvm::SmallVectorImpl<memory_planning::LifetimeDemand> &demands) {
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::memref::AllocOp alloc) {
    if (mlir::failed(result))
      return;
    result = initializeDDRDemand(alloc, defaultAlignment, timeline, demands);
  });
  return result;
}

static mlir::LogicalResult planManagedDDROffsets(
    mlir::Operation *scope,
    llvm::MutableArrayRef<memory_planning::LifetimeDemand> demands,
    int64_t capacityBytes, int64_t largestContiguousBytes,
    int64_t &plannedHighWaterBytes, PlannedDDROffsets &plannedOffsets,
    llvm::SmallVectorImpl<PendingDDRPlacement> &pendingPlacements) {
  for (memory_planning::LifetimeDemand &demand : demands) {
    if (demand.sizeBytes > largestContiguousBytes)
      return demand.allocation.emitError()
             << "largest_contiguous_range_too_small: DDR planned range "
             << demand.sizeBytes << " exceeds largest contiguous range "
             << largestContiguousBytes;
  }

  memory_planning::PackingResult packing = memory_planning::packStaticMemory(
      demands, memory_planning::ArenaRange{0, capacityBytes});
  if (!packing.succeeded()) {
    mlir::Operation *origin = scope;
    memory_planning::LifetimeDemand *demand = nullptr;
    if (packing.demandIndex && *packing.demandIndex < demands.size()) {
      demand = &demands[*packing.demandIndex];
      origin = demand->allocation.getOperation();
    }
    switch (packing.status) {
    case memory_planning::PackingStatus::InvalidProblem:
      return origin->emitError()
             << "invalid_packing_result: DDR static packing input is invalid";
    case memory_planning::PackingStatus::ArithmeticOverflow:
      return origin->emitError()
             << "range_end_overflow: DDR planning end address overflows int64";
    case memory_planning::PackingStatus::ProvenInfeasible:
      return origin->emitError()
             << "memory_capacity_overflow: DDR planning capacity "
             << capacityBytes << " has no valid static placement"
             << (demand ? " for an IR-derived lifetime demand" : "");
    case memory_planning::PackingStatus::ResourceExhausted:
      return origin->emitError()
             << "packing_search_exhausted: MiniMalloc consumed "
             << packing.searchNodes
             << " deterministic search nodes before producing a DDR "
                "placement or proof";
    case memory_planning::PackingStatus::InvalidSolverResult:
      return origin->emitError()
             << "invalid_packing_result: MiniMalloc returned an invalid DDR "
                "placement";
    case memory_planning::PackingStatus::Feasible:
      break;
    }
    llvm_unreachable("successful DDR packing entered failure handling");
  }

  plannedHighWaterBytes = 0;
  for (const memory_planning::Placement &placement : packing.placements) {
    memory_planning::LifetimeDemand &demand = demands[placement.demandIndex];
    if (placement.offsetBytes % demand.alignmentBytes != 0)
      return demand.allocation.emitError()
             << "ddr_alignment_failure: selected DDR offset "
             << placement.offsetBytes << " is not aligned to "
             << demand.alignmentBytes;
    plannedOffsets[demand.allocation.getOperation()] = placement.offsetBytes;
    pendingPlacements.push_back(
        PendingDDRPlacement{demand.allocation, placement.offsetBytes});
    plannedHighWaterBytes = std::max(plannedHighWaterBytes, placement.endBytes);
  }
  return mlir::success();
}

static mlir::LogicalResult
collectDDRDescriptorDemands(mlir::Operation *scope, int64_t defaultAlignment,
                            const PlannedDDROffsets &plannedOffsets,
                            const memory_planning::StructuredTimeline &timeline,
                            const memory_planning::LifetimeDataflow &dataflow,
                            DDRDemandSummary &summary) {
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::Operation *op) {
    if (mlir::failed(result))
      return;

    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
      MovementDescriptor descriptor{
          rdma.getByteCountAttr().getInt(),
          rdma.getInnerBytesAttr().getInt(),
          rdma.getSrcOffsetAttr() ? rdma.getSrcOffsetAttr().getInt() : 0,
          rdma.getSrcStridesAttr(),
          rdma.getSrcIterationsAttr(),
          "source"};
      result = collectDDRDescriptorDemand(op, rdma.getSource(), descriptor,
                                          defaultAlignment, plannedOffsets,
                                          timeline, dataflow, summary);
      return;
    }

    if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
      MovementDescriptor descriptor{
          wdma.getByteCountAttr().getInt(),
          wdma.getInnerBytesAttr().getInt(),
          wdma.getDstOffsetAttr() ? wdma.getDstOffsetAttr().getInt() : 0,
          wdma.getDstStridesAttr(),
          wdma.getDstIterationsAttr(),
          "dest"};
      result = collectDDRDescriptorDemand(op, wdma.getDest(), descriptor,
                                          defaultAlignment, plannedOffsets,
                                          timeline, dataflow, summary);
      return;
    }
  });
  return result;
}

static mlir::LogicalResult verifyResourceLimits(mlir::Operation *op,
                                                const DDRDemandSummary &summary,
                                                int64_t capacityBytes,
                                                int64_t largestContiguousBytes,
                                                int64_t bandwidthLimitBytes) {
  int64_t totalRootBytes = 0;
  for (auto entry : summary.externalRootDemands) {
    int64_t rootBytes = entry.second.rootBytes;
    if (rootBytes > largestContiguousBytes)
      return op->emitError()
             << "largest_contiguous_range_too_small: DDR root demand "
             << rootBytes << " exceeds largest contiguous range "
             << largestContiguousBytes;
    if (!checkedAdd(totalRootBytes, rootBytes, totalRootBytes))
      return op->emitError()
             << "memory_capacity_overflow: DDR root byte sum overflows";
  }

  int64_t totalDemandBytes = 0;
  if (!checkedAdd(totalRootBytes, summary.plannedHighWaterBytes,
                  totalDemandBytes))
    return op->emitError()
           << "memory_capacity_overflow: DDR byte demand sum overflows";

  if (totalDemandBytes > capacityBytes)
    return op->emitError() << "memory_capacity_overflow: DDR demand "
                           << totalDemandBytes << " exceeds capacity "
                           << capacityBytes;

  if (summary.bandwidthBytes > bandwidthLimitBytes)
    return op->emitError() << "bandwidth_pressure_too_high: DDR movement bytes "
                           << summary.bandwidthBytes << " exceed limit "
                           << bandwidthLimitBytes;

  return mlir::success();
}

static mlir::LogicalResult planScopeDDRMemory(
    mlir::Operation *scope, int64_t defaultAlignment, int64_t capacityBytes,
    int64_t largestContiguousBytes, int64_t bandwidthLimitBytes,
    llvm::SmallVectorImpl<PendingDDRPlacement> &pendingPlacements,
    const ManagedTimelineMap *managedTimelines = nullptr) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "transformation-phase", "planScopeDDRMemory", "total");
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "StructuredTimeline::build");
  memory_planning::TimelineFailure timelineFailure;
  std::optional<memory_planning::StructuredTimeline> ownedTimeline;
  const memory_planning::StructuredTimeline *timeline = nullptr;
  if (managedTimelines) {
    auto found = managedTimelines->find(scope);
    if (found != managedTimelines->end()) {
      const memory_planning::StructuredTimelineAnalysis &analysis =
          *found->second;
      if (analysis.isValid())
        timeline = &analysis.getTimeline();
      else
        timelineFailure = analysis.getFailure();
    }
  } else {
    mlir::FailureOr<memory_planning::StructuredTimeline> built =
        memory_planning::StructuredTimeline::build(scope, &timelineFailure);
    if (mlir::succeeded(built)) {
      ownedTimeline.emplace(std::move(*built));
      timeline = &*ownedTimeline;
    }
  }
  if (!timeline) {
    mlir::Operation *origin =
        timelineFailure.origin ? timelineFailure.origin : scope;
    if (timelineFailure.kind ==
        memory_planning::TimelineFailureKind::DecisionDomainExhausted)
      return origin->emitError()
             << "lifetime_analysis_resource_exhausted: DDR structured "
                "decision identifier domain exhausted";
    if (timelineFailure.kind ==
        memory_planning::TimelineFailureKind::InconsistentPathCondition)
      return origin->emitError()
             << "lifetime_overlap_conflict: DDR memory planning could not "
                "construct a consistent structured path condition";
    return origin->emitError()
           << "unsupported_lifetime_control_flow: DDR memory planning only "
              "supports single-block wafer.tile.region, scf.if and scf.for "
              "structured regions";
  }
  llvm::SmallVector<memory_planning::LifetimeDemand, 8> demands;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "collectDDRDemands");
  if (mlir::failed(
          collectDDRDemands(scope, defaultAlignment, *timeline, demands)))
    return mlir::failure();
  memory_planning::LocalCompletionTracker localCompletion;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "lifetime-dataflow");
  memory_planning::LifetimeDataflow dataflow(
      *timeline, demands,
      [](mlir::Type type) { return isWaferDDRMemRefType(type); },
      resolveOneTileRegionBoundary, isExplicitDDRRoot);
  memory_planning::LifetimeFailure lifetimeFailure;
  if (mlir::failed(dataflow.run(scope, &localCompletion, &lifetimeFailure)))
    return emitLifetimeFailure(scope, lifetimeFailure);
  DDRDemandSummary summary;
  PlannedDDROffsets plannedOffsets;
  llvm::SmallVector<PendingDDRPlacement, 8> scopePlacements;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "planManagedDDROffsets");
  if (mlir::failed(planManagedDDROffsets(
          scope, demands, capacityBytes, largestContiguousBytes,
          summary.plannedHighWaterBytes, plannedOffsets, scopePlacements)))
    return mlir::failure();
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "collectDDRDescriptorDemands");
  if (mlir::failed(collectDDRDescriptorDemands(scope, defaultAlignment,
                                               plannedOffsets, *timeline,
                                               dataflow, summary)))
    return mlir::failure();
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "planScopeDDRMemory", "verifyResourceLimits");
  if (mlir::failed(verifyResourceLimits(scope, summary, capacityBytes,
                                        largestContiguousBytes,
                                        bandwidthLimitBytes)))
    return mlir::failure();

  pendingPlacements.append(scopePlacements.begin(), scopePlacements.end());
  return mlir::success();
}

static mlir::LogicalResult
planModuleDDRMemory(mlir::ModuleOp moduleOp, int64_t defaultAlignment,
                    int64_t capacityBytes, int64_t largestContiguousBytes,
                    int64_t bandwidthLimitBytes,
                    const ManagedTimelineMap *managedTimelines,
                    const analysis::DirectCallGraphAnalysis *managedCallGraph) {
  llvm::SmallVector<mlir::func::FuncOp, 4> functions;
  moduleOp.walk(
      [&](mlir::func::FuncOp funcOp) { functions.push_back(funcOp); });

  // Function scopes and a top-level module scope cannot share one static
  // planning timeline. Reject compiler-managed allocations outside func.func
  // instead of silently skipping them once any entry function exists. DDR
  // arguments used inside async helper functions remain caller-owned and are
  // deliberately not separate planning demands.
  if (!functions.empty()) {
    mlir::WalkResult mixedScope =
        moduleOp.walk([&](mlir::memref::AllocOp alloc) {
          if (alloc->getParentOfType<mlir::func::FuncOp>() ||
              !isWaferDDRMemRefType(alloc.getType()))
            return mlir::WalkResult::advance();
          alloc.emitError()
              << "unsupported_ddr_planning_scope: compiler-managed DDR "
                 "allocation outside func.func cannot be planned together with "
                 "func.func entry scopes";
          return mlir::WalkResult::interrupt();
        });
    if (mixedScope.wasInterrupted())
      return mlir::failure();
  }
  std::optional<analysis::DirectCallGraphAnalysis> ownedCallGraph;
  if (!managedCallGraph) {
    ownedCallGraph.emplace(moduleOp.getOperation());
    managedCallGraph = &*ownedCallGraph;
  }
  if (mlir::failed(verifyDDRCallScopes(moduleOp, *managedCallGraph)))
    return mlir::failure();
  if (mlir::failed(verifyDDRAsyncFunctionClosures(moduleOp, managedTimelines)))
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  llvm::SmallVector<PendingDDRPlacement, 16> pendingPlacements;
  for (mlir::func::FuncOp funcOp : functions) {
    if (mlir::failed(result))
      break;
    result = planScopeDDRMemory(funcOp.getOperation(), defaultAlignment,
                                capacityBytes, largestContiguousBytes,
                                bandwidthLimitBytes, pendingPlacements,
                                managedTimelines);
  }
  if (mlir::failed(result))
    return result;
  if (functions.empty() &&
      mlir::failed(planScopeDDRMemory(moduleOp.getOperation(), defaultAlignment,
                                      capacityBytes, largestContiguousBytes,
                                      bandwidthLimitBytes, pendingPlacements,
                                      managedTimelines)))
    return mlir::failure();

  for (PendingDDRPlacement placement : pendingPlacements) {
    placement.allocation->setAttr(
        kWaferDDROffsetAttrName,
        DDROffsetAttr::get(placement.allocation.getContext(),
                           placement.offsetBytes));
  }
  return mlir::success();
}

static mlir::LogicalResult planDDRMemoryModuleImpl(
    mlir::ModuleOp moduleOp, int64_t ddrAlignmentBytes,
    int64_t ddrCapacityBytes, int64_t ddrLargestContiguousBytes,
    int64_t ddrBandwidthLimitBytes, const ManagedTimelineMap *managedTimelines,
    const analysis::DirectCallGraphAnalysis *managedCallGraph) {
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::DDRPlanning);
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "planDDRMemoryModule", "total");
  if (ddrCapacityBytes < 0 || ddrLargestContiguousBytes < 0 ||
      ddrBandwidthLimitBytes < 0 || ddrAlignmentBytes <= 0)
    return moduleOp->emitError()
           << "invalid_ddr_resource_limit: DDR resource limits must be "
              "non-negative and DDR alignment must be positive";

  return planModuleDDRMemory(moduleOp, ddrAlignmentBytes, ddrCapacityBytes,
                             ddrLargestContiguousBytes, ddrBandwidthLimitBytes,
                             managedTimelines, managedCallGraph);
}

} // namespace

mlir::LogicalResult planDDRMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t ddrAlignmentBytes,
                                        int64_t ddrCapacityBytes,
                                        int64_t ddrLargestContiguousBytes,
                                        int64_t ddrBandwidthLimitBytes) {
  return planDDRMemoryModuleImpl(
      moduleOp, ddrAlignmentBytes, ddrCapacityBytes, ddrLargestContiguousBytes,
      ddrBandwidthLimitBytes,
      /*managedTimelines=*/nullptr, /*managedCallGraph=*/nullptr);
}

namespace {

struct PlanDDRMemoryPass
    : public impl::PlanDDRMemoryPassBase<PlanDDRMemoryPass> {
  using impl::PlanDDRMemoryPassBase<PlanDDRMemoryPass>::PlanDDRMemoryPassBase;

  void runOnOperation() final {
    unsigned assignedBefore = 0;
    getOperation().walk([&](mlir::memref::AllocOp allocation) {
      assignedBefore += static_cast<bool>(
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName));
    });
    ManagedTimelineMap managedTimelines;
    bool hasFunction = false;
    for (mlir::func::FuncOp function :
         getOperation().getOps<mlir::func::FuncOp>()) {
      hasFunction = true;
      managedTimelines.try_emplace(
          function.getOperation(),
          &getChildAnalysis<memory_planning::StructuredTimelineAnalysis>(
              function));
    }
    getOperation().walk([&](mlir::async::FuncOp function) {
      if (!function.isExternal())
        managedTimelines.try_emplace(
            function.getOperation(),
            &getChildAnalysis<memory_planning::StructuredTimelineAnalysis>(
                function));
    });
    if (!hasFunction)
      managedTimelines.try_emplace(
          getOperation().getOperation(),
          &getAnalysis<memory_planning::StructuredTimelineAnalysis>());

    if (mlir::failed(planDDRMemoryModuleImpl(
            getOperation(), ddrAlignmentBytes, ddrCapacityBytes,
            ddrLargestContiguousBytes, ddrBandwidthLimitBytes,
            &managedTimelines,
            &getAnalysis<analysis::DirectCallGraphAnalysis>()))) {
      signalPassFailure();
      return;
    }
    numTimelineScopes += managedTimelines.size();
    unsigned assignedAfter = 0;
    getOperation().walk([&](mlir::memref::AllocOp allocation) {
      assignedAfter += static_cast<bool>(
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName));
    });
    if (assignedAfter > assignedBefore)
      numAssignedAllocations += assignedAfter - assignedBefore;
    markAnalysesPreserved<memory_planning::StructuredTimelineAnalysis>();
    markAnalysesPreserved<analysis::DirectCallGraphAnalysis>();
  }
};

} // namespace

} // namespace wafer
