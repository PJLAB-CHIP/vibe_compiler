//===- PlanDDRMemory.cpp - Plan Wafer DDR memory ------------===//

#include "Wafer/Transforms/Passes.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "MemoryPlanning/StaticMemoryPacking.h"

#include "Wafer/IR/WaferDialect.h"

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

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (true) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!tileRegion || tileRegion.getBody().empty() ||
          owner != &tileRegion.getBody().front() ||
          blockArg.getArgNumber() >= tileRegion.getInputs().size())
        return value;
      value = tileRegion.getInputs()[blockArg.getArgNumber()];
      continue;
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto tileRegion =
        result ? mlir::dyn_cast_or_null<TileRegionOp>(result.getOwner())
               : TileRegionOp{};
    if (!tileRegion || tileRegion.getBody().empty())
      return value;
    auto yield = mlir::dyn_cast<TileYieldOp>(
        tileRegion.getBody().front().getTerminator());
    if (!yield || result.getResultNumber() >= yield.getValues().size())
      return value;
    value = yield.getValues()[result.getResultNumber()];
  }
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

static bool isSupportedDDRAliasCall(mlir::func::CallOp call) {
  if (!memory_planning::isSupportedDirectAliasCall(
          call, [](mlir::Type type) { return isWaferDDRMemRefType(type); }))
    return false;
  mlir::func::FuncOp callee =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          call, call.getCalleeAttr());
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

static mlir::LogicalResult verifyDDRCallScopes(mlir::ModuleOp moduleOp) {
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
      mlir::func::FuncOp callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (caller && callee && mayTouchDDR.contains(callee.getOperation()) &&
          mayTouchDDR.insert(caller).second)
        changed = true;
    });
  }

  mlir::WalkResult result = moduleOp.walk([&](mlir::func::CallOp call) {
    mlir::func::FuncOp callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
            call, call.getCalleeAttr());
    bool carriesDDR = llvm::any_of(call->getOperandTypes(),
                                   [](mlir::Type type) {
                                     return isWaferDDRMemRefType(type);
                                   }) ||
                      llvm::any_of(call->getResultTypes(), [](mlir::Type type) {
                        return isWaferDDRMemRefType(type);
                      });
    if (isSupportedDDRAliasCall(call))
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

struct StaticIndexRange {
  int64_t min = 0;
  int64_t max = 0;
};

static mlir::FailureOr<int64_t> getConstantIndex(mlir::Operation *anchor,
                                                 mlir::Value value,
                                                 llvm::StringRef role) {
  std::optional<int64_t> matched = mlir::getConstantIntValue(value);
  if (!matched)
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " dynamic offset requires constant scf.for bounds and step";
  return *matched;
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

  mlir::Value value = mlir::cast<mlir::Value>(offset);
  auto iv = mlir::dyn_cast<mlir::BlockArgument>(value);
  auto forOp = iv && iv.getArgNumber() == 0
                   ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                         iv.getOwner()->getParentOp())
                   : mlir::scf::ForOp{};
  if (!forOp || iv != forOp.getInductionVar())
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " dynamic offset must be a direct scf.for induction variable";

  mlir::FailureOr<int64_t> lower =
      getConstantIndex(anchor, forOp.getLowerBound(), role);
  mlir::FailureOr<int64_t> upper =
      getConstantIndex(anchor, forOp.getUpperBound(), role);
  mlir::FailureOr<int64_t> step =
      getConstantIndex(anchor, forOp.getStep(), role);
  if (mlir::failed(lower) || mlir::failed(upper) || mlir::failed(step))
    return mlir::failure();
  if (*lower < 0 || *upper <= *lower || *step <= 0)
    return anchor->emitError()
           << "unsupported_ddr_view: " << role
           << " scf.for offset range must be non-empty and non-negative";

  int64_t distance = *upper - *lower - 1;
  int64_t iterationsFromLower = distance / *step;
  int64_t max = 0;
  int64_t delta = 0;
  if (!checkedMul(iterationsFromLower, *step, delta) ||
      !checkedAdd(*lower, delta, max))
    return anchor->emitError()
           << "range_end_overflow: dynamic DDR view offset overflows int64";
  return StaticIndexRange{*lower, max};
}

static mlir::Value resolveCarriedViewValue(mlir::Value value) {
  while (true) {
    mlir::Value resolved = resolveTileRegionBoundaryValue(value);
    if (resolved != value) {
      value = resolved;
      continue;
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          blockArg.getOwner()->getParentOp());
      if (forOp && blockArg.getArgNumber() > 0 &&
          blockArg.getArgNumber() - 1 < forOp.getInitArgs().size()) {
        value = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
        continue;
      }
    }
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
        if (result.getResultNumber() < forOp.getInitArgs().size()) {
          value = forOp.getInitArgs()[result.getResultNumber()];
          continue;
        }
      }
    }
    return value;
  }
}

static mlir::FailureOr<StaticIndexRange>
getViewElementOffsetRange(mlir::Operation *anchor, mlir::Value view,
                          mlir::Value expectedRoot, llvm::StringRef role) {
  StaticIndexRange total;
  llvm::DenseSet<mlir::Value> visited;
  while (true) {
    view = resolveCarriedViewValue(view);
    expectedRoot = resolveCarriedViewValue(expectedRoot);
    if (view == expectedRoot)
      return total;
    if (!visited.insert(view).second)
      return anchor->emitError()
             << "unsupported_ddr_view: cyclic DDR view provenance";

    if (auto collapse = view.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
      view = collapse.getSrc();
      continue;
    }
    if (auto expand = view.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
      view = expand.getSrc();
      continue;
    }
    if (auto cast = view.getDefiningOp<mlir::memref::CastOp>()) {
      // memref.cast only weakens static type information.  It preserves the
      // underlying address, so it contributes no element offset to the DDR
      // range proof.
      view = cast.getSource();
      continue;
    }

    auto subview = view.getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview)
      return anchor->emitError()
             << "unsupported_ddr_view: cannot prove " << role
             << " dynamic view offset to its DDR root";
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
          !checkedAdd(total.min, minContribution, total.min) ||
          !checkedAdd(total.max, maxContribution, total.max))
        return anchor->emitError()
               << "range_end_overflow: DDR view offset range overflows int64";
    }
    view = subview.getSource();
  }
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

  int64_t requiredAlignment = defaultAlignment;
  if (std::optional<uint64_t> allocAlignment = alloc.getAlignment()) {
    if (*allocAlignment >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return op->emitError()
             << "ddr_alignment_failure: memref.alloc alignment exceeds int64";
    std::optional<int64_t> combined =
        memory_planning::combineAlignmentRequirements(
            requiredAlignment, static_cast<int64_t>(*allocAlignment));
    if (!combined)
      return op->emitError()
             << "ddr_alignment_failure: combined DDR alignment exceeds int64";
    requiredAlignment = *combined;
  }

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

    StaticIndexRange staticOffsetRange{viewOffsetElements, viewOffsetElements};
    mlir::FailureOr<StaticIndexRange> dynamicOffsetRange = staticOffsetRange;
    if (hasDynamicViewOffset)
      dynamicOffsetRange =
          getViewElementOffsetRange(op, ddrValue, root, descriptor.role);
    if (mlir::failed(dynamicOffsetRange))
      return mlir::failure();
    const StaticIndexRange &viewOffsetElementsRange = *dynamicOffsetRange;

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
    if (absoluteEnd > view.rootBytes)
      return op->emitError()
             << "ddr_range_overflow: " << descriptor.role << " access end "
             << absoluteEnd << " exceeds DDR root byte size " << view.rootBytes;

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
              "wafer.instr.local_fence";
  case memory_planning::LifetimeFailureKind::LoopBackedgeCompletion:
    return origin->emitError()
           << "missing_local_completion: DDR-touching local Movement issue "
              "reaches an scf.for backedge without a body-local "
              "wafer.instr.local_fence";
  case memory_planning::LifetimeFailureKind::InconsistentCompletionState:
    return origin->emitError()
           << "completion_proof_failure: DDR local issue lifetime state "
              "remains after all local issues were fenced";
  }
  llvm_unreachable("unknown DDR lifetime failure");
}

static mlir::LogicalResult
verifyDDRAsyncFunctionClosures(mlir::ModuleOp moduleOp) {
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::async::FuncOp funcOp) {
    if (mlir::failed(result) || funcOp.isExternal())
      return;

    memory_planning::TimelineFailure timelineFailure;
    mlir::FailureOr<memory_planning::StructuredTimeline> timeline =
        memory_planning::StructuredTimeline::build(funcOp.getOperation(),
                                                   &timelineFailure);
    if (mlir::failed(timeline)) {
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

  int64_t requiredAlignment = defaultAlignment;
  if (std::optional<uint64_t> allocAlignment = alloc.getAlignment()) {
    if (*allocAlignment >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return alloc.emitError()
             << "ddr_alignment_failure: memref.alloc alignment exceeds int64";
    std::optional<int64_t> combined =
        memory_planning::combineAlignmentRequirements(
            requiredAlignment, static_cast<int64_t>(*allocAlignment));
    if (!combined)
      return alloc.emitError()
             << "ddr_alignment_failure: combined DDR alignment exceeds int64";
    requiredAlignment = *combined;
  }

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
             << " deterministic search nodes and the first-fit safety "
                "fallback could not produce a verified DDR placement";
    case memory_planning::PackingStatus::InvalidSolverResult:
      return origin->emitError()
             << "invalid_packing_result: MiniMalloc returned an invalid DDR "
                "placement";
    case memory_planning::PackingStatus::HeuristicNoFit:
      return origin->emitError()
             << "invalid_packing_result: first-fit NoFit escaped the shared "
                "packing policy";
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
    llvm::SmallVectorImpl<PendingDDRPlacement> &pendingPlacements) {
  memory_planning::TimelineFailure timelineFailure;
  mlir::FailureOr<memory_planning::StructuredTimeline> timeline =
      memory_planning::StructuredTimeline::build(scope, &timelineFailure);
  if (mlir::failed(timeline)) {
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
  if (mlir::failed(
          collectDDRDemands(scope, defaultAlignment, *timeline, demands)))
    return mlir::failure();

  memory_planning::LocalCompletionTracker localCompletion;
  memory_planning::LifetimeDataflow dataflow(
      *timeline, demands,
      [](mlir::Type type) { return isWaferDDRMemRefType(type); },
      resolveTileRegionBoundaryValue, isExplicitDDRRoot);
  memory_planning::LifetimeFailure lifetimeFailure;
  if (mlir::failed(dataflow.run(scope, &localCompletion, &lifetimeFailure)))
    return emitLifetimeFailure(scope, lifetimeFailure);

  DDRDemandSummary summary;
  PlannedDDROffsets plannedOffsets;
  llvm::SmallVector<PendingDDRPlacement, 8> scopePlacements;
  if (mlir::failed(planManagedDDROffsets(
          scope, demands, capacityBytes, largestContiguousBytes,
          summary.plannedHighWaterBytes, plannedOffsets, scopePlacements)))
    return mlir::failure();

  if (mlir::failed(collectDDRDescriptorDemands(scope, defaultAlignment,
                                               plannedOffsets, *timeline,
                                               dataflow, summary)))
    return mlir::failure();

  if (mlir::failed(verifyResourceLimits(scope, summary, capacityBytes,
                                        largestContiguousBytes,
                                        bandwidthLimitBytes)))
    return mlir::failure();

  pendingPlacements.append(scopePlacements.begin(), scopePlacements.end());
  return mlir::success();
}

static mlir::LogicalResult planModuleDDRMemory(mlir::ModuleOp moduleOp,
                                               int64_t defaultAlignment,
                                               int64_t capacityBytes,
                                               int64_t largestContiguousBytes,
                                               int64_t bandwidthLimitBytes) {
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
  if (mlir::failed(verifyDDRCallScopes(moduleOp)))
    return mlir::failure();
  if (mlir::failed(verifyDDRAsyncFunctionClosures(moduleOp)))
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  llvm::SmallVector<PendingDDRPlacement, 16> pendingPlacements;
  for (mlir::func::FuncOp funcOp : functions) {
    if (mlir::failed(result))
      break;
    result = planScopeDDRMemory(funcOp.getOperation(), defaultAlignment,
                                capacityBytes, largestContiguousBytes,
                                bandwidthLimitBytes, pendingPlacements);
  }
  if (mlir::failed(result))
    return result;
  if (functions.empty() &&
      mlir::failed(planScopeDDRMemory(moduleOp.getOperation(), defaultAlignment,
                                      capacityBytes, largestContiguousBytes,
                                      bandwidthLimitBytes, pendingPlacements)))
    return mlir::failure();

  for (PendingDDRPlacement placement : pendingPlacements) {
    placement.allocation->setAttr(
        kWaferDDROffsetAttrName,
        DDROffsetAttr::get(placement.allocation.getContext(),
                           placement.offsetBytes));
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult planDDRMemoryModule(mlir::ModuleOp moduleOp,
                                        int64_t ddrAlignmentBytes,
                                        int64_t ddrCapacityBytes,
                                        int64_t ddrLargestContiguousBytes,
                                        int64_t ddrBandwidthLimitBytes) {
  if (ddrCapacityBytes < 0 || ddrLargestContiguousBytes < 0 ||
      ddrBandwidthLimitBytes < 0 || ddrAlignmentBytes <= 0)
    return moduleOp->emitError()
           << "invalid_ddr_resource_limit: DDR resource limits must be "
              "non-negative and DDR alignment must be positive";

  return planModuleDDRMemory(moduleOp, ddrAlignmentBytes, ddrCapacityBytes,
                             ddrLargestContiguousBytes, ddrBandwidthLimitBytes);
}

namespace {

struct PlanDDRMemoryPass
    : public impl::PlanDDRMemoryPassBase<PlanDDRMemoryPass> {
  using impl::PlanDDRMemoryPassBase<PlanDDRMemoryPass>::PlanDDRMemoryPassBase;

  void runOnOperation() final {
    if (mlir::failed(planDDRMemoryModule(
            getOperation(), ddrAlignmentBytes, ddrCapacityBytes,
            ddrLargestContiguousBytes, ddrBandwidthLimitBytes)))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
