//===- PlanDDRMemory.cpp - Plan Wafer DDR memory ------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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

struct PathCondition {
  uint64_t trueBranches = 0;
  uint64_t falseBranches = 0;
};

struct LiveSegment {
  int64_t startEvent = 0;
  int64_t endEvent = 0;
  PathCondition condition;
};

struct DDRDemand {
  mlir::memref::AllocOp alloc;
  int64_t size = 0;
  int64_t alignment = 0;
  int64_t allocEvent = 0;
  PathCondition allocCondition;
  unsigned ordinal = 0;
  int64_t conflictBytes = 0;
  int64_t firstLiveEvent = 0;
  int64_t lastLiveEvent = 0;
  llvm::SmallVector<LiveSegment, 4> segments;
};

struct AssignedDDROffset {
  unsigned demandIndex = 0;
  int64_t offset = 0;
  int64_t end = 0;
};

struct RootRef {
  unsigned demandIndex = 0;
  PathCondition condition;
};

struct EventInfo {
  llvm::DenseMap<mlir::Operation *, int64_t> operationEvents;
  llvm::DenseMap<mlir::Operation *, PathCondition> operationConditions;
  llvm::DenseMap<mlir::Operation *, int64_t> subtreeEndEvents;
  int64_t nextEvent = 0;
  unsigned nextBranch = 0;
  bool branchLimitExceeded = false;
};

struct MovementDescriptor {
  int64_t byteCount = 0;
  int64_t innerBytes = 0;
  mlir::DenseI64ArrayAttr strides;
  mlir::DenseI64ArrayAttr iterations;
  llvm::StringRef role;
};

struct DDRView {
  mlir::Value root;
  mlir::MemRefType rootType;
  mlir::MemRefType viewType;
  int64_t viewOffsetBytes = 0;
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

static std::optional<int64_t> alignUp(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;
  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t result = 0;
  if (!checkedAdd(value, alignment - remainder, result))
    return std::nullopt;
  return result;
}

static bool areCompatible(PathCondition lhs, PathCondition rhs) {
  return (lhs.trueBranches & rhs.falseBranches) == 0 &&
         (lhs.falseBranches & rhs.trueBranches) == 0;
}

static std::optional<PathCondition> mergeConditions(PathCondition lhs,
                                                    PathCondition rhs) {
  if (!areCompatible(lhs, rhs))
    return std::nullopt;
  return PathCondition{lhs.trueBranches | rhs.trueBranches,
                       lhs.falseBranches | rhs.falseBranches};
}

static std::optional<PathCondition> withBranch(PathCondition condition,
                                               unsigned branch, bool thenPath) {
  if (branch >= 64)
    return std::nullopt;
  uint64_t bit = uint64_t{1} << branch;
  PathCondition branchCondition =
      thenPath ? PathCondition{bit, 0} : PathCondition{0, bit};
  return mergeConditions(condition, branchCondition);
}

static bool segmentsOverlap(const LiveSegment &lhs, const LiveSegment &rhs) {
  if (!areCompatible(lhs.condition, rhs.condition))
    return false;
  return lhs.startEvent <= rhs.endEvent && rhs.startEvent <= lhs.endEvent;
}

static bool lifetimesOverlap(const DDRDemand &lhs, const DDRDemand &rhs) {
  for (const LiveSegment &lhsSegment : lhs.segments)
    for (const LiveSegment &rhsSegment : rhs.segments)
      if (segmentsOverlap(lhsSegment, rhsSegment))
        return true;
  return false;
}

static bool byteRangesOverlap(int64_t lhsBegin, int64_t lhsEnd,
                              int64_t rhsBegin, int64_t rhsEnd) {
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

static void assignRegionEvents(mlir::Region &region, PathCondition condition,
                               EventInfo &events);

static void assignBlockEvents(mlir::Block &block, PathCondition condition,
                              EventInfo &events) {
  for (mlir::Operation &op : block) {
    events.operationEvents[&op] = events.nextEvent++;
    events.operationConditions[&op] = condition;

    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      unsigned branch = events.nextBranch++;
      std::optional<PathCondition> thenCondition =
          withBranch(condition, branch, /*thenPath=*/true);
      std::optional<PathCondition> elseCondition =
          withBranch(condition, branch, /*thenPath=*/false);
      if (!thenCondition || !elseCondition) {
        events.branchLimitExceeded = true;
      } else {
        assignRegionEvents(ifOp.getThenRegion(), *thenCondition, events);
        assignRegionEvents(ifOp.getElseRegion(), *elseCondition, events);
      }
    } else {
      for (mlir::Region &region : op.getRegions())
        assignRegionEvents(region, condition, events);
    }

    events.subtreeEndEvents[&op] = events.nextEvent - 1;
  }
}

static void assignRegionEvents(mlir::Region &region, PathCondition condition,
                               EventInfo &events) {
  for (mlir::Block &block : region)
    assignBlockEvents(block, condition, events);
}

static mlir::LogicalResult assignOperationEvents(mlir::Operation *scope,
                                                 EventInfo &events) {
  for (mlir::Region &region : scope->getRegions())
    assignRegionEvents(region, PathCondition{}, events);
  if (events.branchLimitExceeded)
    return scope->emitError()
           << "lifetime_overlap_conflict: DDR memory planning supports at "
              "most 64 nested branch decision points";
  return mlir::success();
}

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArg.getOwner();
    if (!owner)
      return value;
    auto tileRegion =
        mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp());
    if (!tileRegion || tileRegion.getBody().empty() ||
        owner != &tileRegion.getBody().front())
      return value;
    if (blockArg.getArgNumber() >= tileRegion.getInputs().size())
      return value;
    value = tileRegion.getInputs()[blockArg.getArgNumber()];
  }
  return value;
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

static bool hasAsyncTokenType(mlir::Value value) {
  return mlir::isa<mlir::async::TokenType>(value.getType());
}

static mlir::scf::YieldOp getSingleBlockYield(mlir::Region &region) {
  if (region.empty())
    return {};
  return mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
}

static bool isCompilerManagedDDRRoot(mlir::Value root) {
  mlir::Operation *def = root.getDefiningOp();
  return def && mlir::isa<mlir::memref::AllocOp>(def) &&
         isWaferDDRMemRefType(root.getType());
}

static void addLiveSegment(DDRDemand &demand, int64_t startEvent,
                           int64_t endEvent, PathCondition condition) {
  if (endEvent < startEvent)
    return;
  demand.segments.push_back(LiveSegment{startEvent, endEvent, condition});
}

static void recordDemandUse(DDRDemand &demand, int64_t event,
                            PathCondition condition) {
  addLiveSegment(demand, demand.allocEvent, event, condition);
}

static void addSaturated(int64_t &lhs, int64_t rhs) {
  if (rhs <= 0)
    return;
  if (lhs > std::numeric_limits<int64_t>::max() - rhs) {
    lhs = std::numeric_limits<int64_t>::max();
    return;
  }
  lhs += rhs;
}

static void computeLifetimeBounds(DDRDemand &demand) {
  if (demand.segments.empty()) {
    demand.firstLiveEvent = demand.allocEvent;
    demand.lastLiveEvent = demand.allocEvent;
    return;
  }

  demand.firstLiveEvent = demand.segments.front().startEvent;
  demand.lastLiveEvent = demand.segments.front().endEvent;
  for (const LiveSegment &segment : demand.segments) {
    demand.firstLiveEvent = std::min(demand.firstLiveEvent, segment.startEvent);
    demand.lastLiveEvent = std::max(demand.lastLiveEvent, segment.endEvent);
  }
}

static int64_t getLifetimeSpan(const DDRDemand &demand) {
  if (demand.lastLiveEvent <= demand.firstLiveEvent)
    return 0;
  return demand.lastLiveEvent - demand.firstLiveEvent;
}

static void
computePlanningPriorities(llvm::MutableArrayRef<DDRDemand> demands) {
  for (DDRDemand &demand : demands) {
    demand.conflictBytes = 0;
    computeLifetimeBounds(demand);
  }

  for (size_t lhsIndex = 0; lhsIndex < demands.size(); ++lhsIndex) {
    DDRDemand &lhs = demands[lhsIndex];
    for (DDRDemand &rhs : demands.drop_front(lhsIndex + 1)) {
      if (!lifetimesOverlap(lhs, rhs))
        continue;
      addSaturated(lhs.conflictBytes, rhs.size);
      addSaturated(rhs.conflictBytes, lhs.size);
    }
  }
}

static bool hasHigherPlanningPriority(const DDRDemand &lhs,
                                      const DDRDemand &rhs) {
  if (lhs.size != rhs.size)
    return lhs.size > rhs.size;
  if (lhs.conflictBytes != rhs.conflictBytes)
    return lhs.conflictBytes > rhs.conflictBytes;

  int64_t lhsSpan = getLifetimeSpan(lhs);
  int64_t rhsSpan = getLifetimeSpan(rhs);
  if (lhsSpan != rhsSpan)
    return lhsSpan > rhsSpan;

  if (lhs.allocEvent != rhs.allocEvent)
    return lhs.allocEvent < rhs.allocEvent;
  return lhs.ordinal < rhs.ordinal;
}

static llvm::SmallVector<RootRef, 2>
getRefsAtUse(mlir::Value value, PathCondition useCondition,
             const llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>>
                 &valueRefs) {
  value = resolveTileRegionBoundaryValue(value);
  llvm::SmallVector<RootRef, 2> refs;
  auto it = valueRefs.find(value);
  if (it == valueRefs.end())
    return refs;
  for (RootRef ref : it->second) {
    std::optional<PathCondition> merged =
        mergeConditions(ref.condition, useCondition);
    if (merged)
      refs.push_back(RootRef{ref.demandIndex, *merged});
  }
  return refs;
}

static void
recordValueUse(mlir::Value value, int64_t event, PathCondition useCondition,
               llvm::MutableArrayRef<DDRDemand> demands,
               const llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>>
                   &valueRefs) {
  for (RootRef ref : getRefsAtUse(value, useCondition, valueRefs))
    recordDemandUse(demands[ref.demandIndex], event, ref.condition);
}

static void
recordTokenUse(mlir::Value value, int64_t event, PathCondition useCondition,
               llvm::MutableArrayRef<DDRDemand> demands,
               const llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>>
                   &tokenRefs) {
  auto it = tokenRefs.find(value);
  if (it == tokenRefs.end())
    return;
  for (RootRef ref : it->second) {
    std::optional<PathCondition> merged =
        mergeConditions(ref.condition, useCondition);
    if (merged)
      recordDemandUse(demands[ref.demandIndex], event, *merged);
  }
}

struct LifetimeDataflow {
  const EventInfo &events;
  llvm::MutableArrayRef<DDRDemand> demands;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> valueRefs;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> tokenRefs;

  PathCondition getOperationCondition(mlir::Operation *op) const {
    auto it = events.operationConditions.find(op);
    if (it == events.operationConditions.end())
      return {};
    return it->second;
  }

  int64_t getOperationEvent(mlir::Operation *op) const {
    auto it = events.operationEvents.find(op);
    return it == events.operationEvents.end() ? 0 : it->second;
  }

  int64_t getSubtreeEndEvent(mlir::Operation *op) const {
    auto it = events.subtreeEndEvents.find(op);
    return it == events.subtreeEndEvents.end() ? getOperationEvent(op)
                                               : it->second;
  }

  void recordOperands(mlir::Operation *op) {
    int64_t event = getOperationEvent(op);
    PathCondition condition = getOperationCondition(op);
    for (mlir::Value operand : op->getOperands()) {
      if (hasAsyncTokenType(operand)) {
        recordTokenUse(operand, event, condition, demands, tokenRefs);
        continue;
      }
      recordValueUse(operand, event, condition, demands, valueRefs);
    }
  }

  void mapViewLikeResults(mlir::Operation *op) {
    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op);
    if (!viewLike)
      return;
    llvm::SmallVector<RootRef, 2> refs = getRefsAtUse(
        viewLike.getViewSource(), getOperationCondition(op), valueRefs);
    if (refs.empty())
      return;
    for (mlir::Value result : op->getResults()) {
      if (isWaferDDRMemRefType(result.getType()))
        valueRefs[result] = refs;
    }
  }

  void mapTileRegionResults(TileRegionOp tileRegion) {
    if (tileRegion.getBody().empty())
      return;
    auto yield = mlir::dyn_cast<TileYieldOp>(
        tileRegion.getBody().front().getTerminator());
    if (!yield)
      return;

    PathCondition condition = getOperationCondition(tileRegion);
    for (auto [index, result] : llvm::enumerate(tileRegion.getResults())) {
      if (!isWaferDDRMemRefType(result.getType()) ||
          index >= yield.getValues().size())
        continue;
      llvm::SmallVector<RootRef, 2> refs =
          getRefsAtUse(yield.getValues()[index], condition, valueRefs);
      if (!refs.empty())
        valueRefs[result] = std::move(refs);
    }
  }

  void mapAsyncTokenResults(mlir::Operation *op) {
    llvm::SmallVector<RootRef, 2> refs;
    PathCondition condition = getOperationCondition(op);
    for (mlir::Value operand : op->getOperands()) {
      llvm::SmallVector<RootRef, 2> operandRefs =
          getRefsAtUse(operand, condition, valueRefs);
      refs.append(operandRefs.begin(), operandRefs.end());
    }
    if (refs.empty())
      return;
    for (mlir::Value result : op->getResults()) {
      if (hasAsyncTokenType(result))
        tokenRefs[result] = refs;
    }
  }

  void appendYieldOperandRefs(mlir::scf::YieldOp yieldOp, unsigned index,
                              llvm::SmallVectorImpl<RootRef> &refs) {
    if (index >= yieldOp.getResults().size())
      return;
    llvm::SmallVector<RootRef, 2> yieldedRefs = getRefsAtUse(
        yieldOp.getResults()[index], getOperationCondition(yieldOp), valueRefs);
    refs.append(yieldedRefs.begin(), yieldedRefs.end());
  }

  void mapIfResults(mlir::scf::IfOp ifOp) {
    mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
    mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
    if (!thenYield || !elseYield)
      return;

    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isWaferDDRMemRefType(result.getType()))
        continue;
      llvm::SmallVector<RootRef, 2> refs;
      appendYieldOperandRefs(thenYield, index, refs);
      appendYieldOperandRefs(elseYield, index, refs);
      if (!refs.empty())
        valueRefs[result] = refs;
    }
  }

  void mapForRegionIterArgs(mlir::scf::ForOp forOp) {
    for (auto [init, iterArg] :
         llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
      llvm::SmallVector<RootRef, 2> refs =
          getRefsAtUse(init, getOperationCondition(forOp), valueRefs);
      if (!refs.empty())
        valueRefs[iterArg] = refs;
    }
  }

  void mapForResultsAndBackedge(mlir::scf::ForOp forOp) {
    mlir::scf::YieldOp yieldOp =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp)
      return;

    int64_t loopStart = getOperationEvent(forOp);
    int64_t loopEnd = getSubtreeEndEvent(forOp);
    PathCondition loopCondition = getOperationCondition(forOp);

    for (auto [index, result] : llvm::enumerate(forOp.getResults())) {
      if (!isWaferDDRMemRefType(result.getType()))
        continue;

      llvm::SmallVector<RootRef, 2> refs;
      if (index < forOp.getInitArgs().size()) {
        llvm::SmallVector<RootRef, 2> initRefs =
            getRefsAtUse(forOp.getInitArgs()[index], loopCondition, valueRefs);
        refs.append(initRefs.begin(), initRefs.end());
      }
      appendYieldOperandRefs(yieldOp, index, refs);

      for (RootRef ref : refs)
        addLiveSegment(demands[ref.demandIndex], loopStart, loopEnd,
                       ref.condition);
      if (!refs.empty())
        valueRefs[result] = refs;
    }
  }

  void processRegion(mlir::Region &region) {
    for (mlir::Block &block : region)
      processBlock(block);
  }

  void processBlock(mlir::Block &block) {
    for (mlir::Operation &op : block) {
      recordOperands(&op);

      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        mapForRegionIterArgs(forOp);
        processRegion(forOp.getRegion());
        mapForResultsAndBackedge(forOp);
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        processRegion(ifOp.getThenRegion());
        processRegion(ifOp.getElseRegion());
        mapIfResults(ifOp);
      } else if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(op)) {
        processRegion(tileRegion.getBody());
        mapTileRegionResults(tileRegion);
      } else {
        for (mlir::Region &region : op.getRegions())
          processRegion(region);
        mapViewLikeResults(&op);
      }

      mapAsyncTokenResults(&op);
    }
  }
};

static mlir::LogicalResult
getStaticMemrefViewInfo(mlir::Operation *op, mlir::MemRefType type,
                        llvm::SmallVectorImpl<int64_t> &strides,
                        int64_t &offset, llvm::StringRef role) {
  if (!type.hasStaticShape())
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view must have static shape";
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()))
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view must have static strided layout";
  if (offset == mlir::ShapedType::kDynamic || offset < 0)
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
  int64_t end = descriptor.innerBytes;
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
                                         int64_t defaultAlignment) {
  mlir::Operation *def = root.getDefiningOp();
  if (!def)
    return mlir::success();

  auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(def);
  if (!alloc || !isWaferDDRMemRefType(root.getType()))
    return mlir::success();

  auto offset = alloc->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
  if (!offset)
    return op->emitError()
           << "ddr_planned_range_missing: compiler-managed DDR allocation "
              "has no accepted wafer.ddr.offset";

  int64_t requiredAlignment = defaultAlignment;
  if (std::optional<uint64_t> allocAlignment = alloc.getAlignment()) {
    if (*allocAlignment >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return op->emitError()
             << "ddr_alignment_failure: memref.alloc alignment exceeds int64";
    requiredAlignment =
        std::max(requiredAlignment, static_cast<int64_t>(*allocAlignment));
  }

  if (requiredAlignment <= 0 || offset.getOffset() % requiredAlignment != 0)
    return op->emitError()
           << "ddr_alignment_failure: accepted DDR offset does not satisfy "
              "required alignment";

  return mlir::success();
}

static mlir::FailureOr<DDRView>
resolveDDRView(mlir::Operation *op, mlir::Value ddrValue,
               const MovementDescriptor &descriptor, int64_t defaultAlignment) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(ddrValue.getType());
  if (!viewType || !isWaferDDRMemRefType(viewType))
    return op->emitError() << "unsupported_ddr_view: " << descriptor.role
                           << " operand must be a Wafer DDR memref";

  llvm::SmallVector<int64_t, 4> viewStrides;
  int64_t viewOffsetElements = 0;
  if (mlir::failed(getStaticMemrefViewInfo(
          op, viewType, viewStrides, viewOffsetElements, descriptor.role)))
    return mlir::failure();

  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  if (!viewInfo || viewInfo->physicalBytes < 0 ||
      (!viewInfo->bitPackedElement && viewInfo->elementBytes <= 0))
    return op->emitError() << "unsupported_ddr_view: cannot compute "
                           << descriptor.role << " DDR view physical bytes";

  mlir::Value root = getRootViewSource(ddrValue);
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

  int64_t viewOffsetBytes = 0;
  if (viewInfo->bitPackedElement || rootInfo->bitPackedElement) {
    if (!viewInfo->bitPackedElement || !rootInfo->bitPackedElement)
      return op->emitError()
             << "unsupported_ddr_view: DDR view and root bitpacking differ";
    if (viewOffsetElements != 0)
      return op->emitError()
             << "unsupported_ddr_view: bitpacked DDR view must have zero "
                "element offset";
  } else {
    if (viewInfo->elementBytes != rootInfo->elementBytes)
      return op->emitError()
             << "unsupported_ddr_view: DDR view and root element byte sizes "
                "differ";
    if (!checkedMul(viewOffsetElements, viewInfo->elementBytes,
                    viewOffsetBytes))
      return op->emitError()
             << "range_end_overflow: DDR view byte offset overflows int64";
  }

  if (mlir::failed(verifyDDRRoot(op, root, defaultAlignment)))
    return mlir::failure();

  return DDRView{root,
                 rootType,
                 viewType,
                 viewOffsetBytes,
                 viewInfo->physicalBytes,
                 rootInfo->physicalBytes};
}

static mlir::LogicalResult
collectDDRDescriptorDemand(mlir::Operation *op, mlir::Value ddrValue,
                           const MovementDescriptor &descriptor,
                           int64_t defaultAlignment,
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

  mlir::FailureOr<DDRView> view =
      resolveDDRView(op, ddrValue, descriptor, defaultAlignment);
  if (mlir::failed(view))
    return mlir::failure();

  if (*localEnd > view->viewSpanBytes)
    return op->emitError() << "ddr_range_overflow: " << descriptor.role
                           << " descriptor byte range " << *localEnd
                           << " exceeds DDR view span " << view->viewSpanBytes;

  int64_t absoluteEnd = 0;
  if (!checkedAdd(view->viewOffsetBytes, *localEnd, absoluteEnd))
    return op->emitError()
           << "range_end_overflow: DDR access end overflows int64";
  if (absoluteEnd > view->rootBytes)
    return op->emitError() << "ddr_range_overflow: " << descriptor.role
                           << " access end " << absoluteEnd
                           << " exceeds DDR root byte size " << view->rootBytes;

  if (!isCompilerManagedDDRRoot(view->root)) {
    auto [it, inserted] = summary.externalRootDemands.try_emplace(view->root);
    ExternalDDRRootDemand &rootDemand = it->second;
    if (inserted)
      rootDemand.rootBytes = view->rootBytes;
  }

  if (!checkedAdd(summary.bandwidthBytes, descriptor.byteCount,
                  summary.bandwidthBytes))
    return op->emitError()
           << "bandwidth_pressure_too_high: DDR bandwidth byte sum overflows";
  return mlir::success();
}

static mlir::LogicalResult initializeDDRDemand(
    mlir::memref::AllocOp alloc, int64_t defaultAlignment,
    const EventInfo &events, llvm::SmallVectorImpl<DDRDemand> &demands,
    llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> &valueRefs) {
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
    requiredAlignment =
        std::max(requiredAlignment, static_cast<int64_t>(*allocAlignment));
  }

  auto eventIt = events.operationEvents.find(alloc.getOperation());
  auto conditionIt = events.operationConditions.find(alloc.getOperation());
  if (eventIt == events.operationEvents.end() ||
      conditionIt == events.operationConditions.end())
    return alloc.emitError()
           << "lifetime_overlap_conflict: missing event for DDR allocation";

  unsigned demandIndex = demands.size();
  DDRDemand demand{alloc,           info->physicalBytes, requiredAlignment,
                   eventIt->second, conditionIt->second, demandIndex};
  demands.push_back(demand);
  valueRefs[alloc.getMemref()] =
      llvm::SmallVector<RootRef, 2>{RootRef{demandIndex, conditionIt->second}};
  alloc->removeAttr(kWaferDDROffsetAttrName);
  return mlir::success();
}

static mlir::LogicalResult
collectDDRDemand(mlir::Operation *scope, int64_t defaultAlignment,
                 const EventInfo &events,
                 llvm::SmallVectorImpl<DDRDemand> &demands) {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> valueRefs;
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::memref::AllocOp alloc) {
    if (mlir::failed(result))
      return;
    result = initializeDDRDemand(alloc, defaultAlignment, events, demands,
                                 valueRefs);
  });
  if (mlir::failed(result))
    return mlir::failure();

  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> tokenRefs;
  LifetimeDataflow dataflow{events, demands, std::move(valueRefs),
                            std::move(tokenRefs)};
  for (mlir::Region &region : scope->getRegions())
    dataflow.processRegion(region);
  return mlir::success();
}

static mlir::FailureOr<int64_t>
findFirstFitOffset(const DDRDemand &demand,
                   llvm::ArrayRef<AssignedDDROffset> assignedOffsets,
                   llvm::ArrayRef<DDRDemand> demands, int64_t capacityBytes) {
  int64_t candidate = 0;
  while (true) {
    std::optional<int64_t> alignedOffset = alignUp(candidate, demand.alignment);
    if (!alignedOffset)
      return mlir::failure();

    int64_t candidateEnd = 0;
    if (!checkedAdd(*alignedOffset, demand.size, candidateEnd))
      return mlir::failure();
    if (candidateEnd > capacityBytes)
      return mlir::failure();

    int64_t nextCandidate = *alignedOffset;
    for (const AssignedDDROffset &assigned : assignedOffsets) {
      if (!lifetimesOverlap(demand, demands[assigned.demandIndex]))
        continue;
      if (!byteRangesOverlap(*alignedOffset, candidateEnd, assigned.offset,
                             assigned.end))
        continue;
      nextCandidate = std::max(nextCandidate, assigned.end);
    }

    if (nextCandidate == *alignedOffset)
      return *alignedOffset;
    candidate = nextCandidate;
  }
}

static mlir::LogicalResult
planManagedDDROffsets(mlir::Operation *scope,
                      llvm::MutableArrayRef<DDRDemand> demands,
                      int64_t capacityBytes, int64_t largestContiguousBytes,
                      int64_t &plannedHighWaterBytes) {
  computePlanningPriorities(demands);
  llvm::sort(demands, hasHigherPlanningPriority);

  plannedHighWaterBytes = 0;
  llvm::SmallVector<AssignedDDROffset, 8> assignedOffsets;
  for (auto [demandIndex, demand] : llvm::enumerate(demands)) {
    if (demand.size > largestContiguousBytes)
      return demand.alloc.emitError()
             << "largest_contiguous_range_too_small: DDR planned range "
             << demand.size << " exceeds largest contiguous range "
             << largestContiguousBytes;

    mlir::FailureOr<int64_t> offset =
        findFirstFitOffset(demand, assignedOffsets, demands, capacityBytes);
    if (mlir::failed(offset))
      return demand.alloc.emitError()
             << "memory_capacity_overflow: DDR planning capacity "
             << capacityBytes << " cannot fit " << demand.size
             << " byte buffer with IR-derived lifetime";

    int64_t end = 0;
    if (!checkedAdd(*offset, demand.size, end))
      return demand.alloc.emitError()
             << "range_end_overflow: DDR planning end address overflows int64";
    if (*offset % demand.alignment != 0)
      return demand.alloc.emitError()
             << "ddr_alignment_failure: selected DDR offset " << *offset
             << " is not aligned to " << demand.alignment;

    demand.alloc->setAttr(
        kWaferDDROffsetAttrName,
        DDROffsetAttr::get(demand.alloc.getContext(), *offset));
    plannedHighWaterBytes = std::max(plannedHighWaterBytes, end);
    assignedOffsets.push_back(
        AssignedDDROffset{static_cast<unsigned>(demandIndex), *offset, end});
  }

  (void)scope;
  return mlir::success();
}

static mlir::LogicalResult
collectDDRDescriptorDemands(mlir::Operation *scope, int64_t defaultAlignment,
                            DDRDemandSummary &summary) {
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::Operation *op) {
    if (mlir::failed(result))
      return;

    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
      MovementDescriptor descriptor{
          rdma.getByteCountAttr().getInt(), rdma.getInnerBytesAttr().getInt(),
          rdma.getSrcStridesAttr(), rdma.getSrcIterationsAttr(), "source"};
      result = collectDDRDescriptorDemand(op, rdma.getSource(), descriptor,
                                          defaultAlignment, summary);
      return;
    }

    if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
      MovementDescriptor descriptor{
          wdma.getByteCountAttr().getInt(), wdma.getInnerBytesAttr().getInt(),
          wdma.getDstStridesAttr(), wdma.getDstIterationsAttr(), "dest"};
      result = collectDDRDescriptorDemand(op, wdma.getDest(), descriptor,
                                          defaultAlignment, summary);
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

static mlir::LogicalResult planScopeDDRMemory(mlir::Operation *scope,
                                              int64_t defaultAlignment,
                                              int64_t capacityBytes,
                                              int64_t largestContiguousBytes,
                                              int64_t bandwidthLimitBytes) {
  EventInfo events;
  if (mlir::failed(assignOperationEvents(scope, events)))
    return mlir::failure();

  llvm::SmallVector<DDRDemand, 8> demands;
  if (mlir::failed(collectDDRDemand(scope, defaultAlignment, events, demands)))
    return mlir::failure();

  DDRDemandSummary summary;
  if (mlir::failed(planManagedDDROffsets(scope, demands, capacityBytes,
                                         largestContiguousBytes,
                                         summary.plannedHighWaterBytes)))
    return mlir::failure();

  if (mlir::failed(
          collectDDRDescriptorDemands(scope, defaultAlignment, summary)))
    return mlir::failure();

  if (mlir::failed(verifyResourceLimits(scope, summary, capacityBytes,
                                        largestContiguousBytes,
                                        bandwidthLimitBytes)))
    return mlir::failure();

  return mlir::success();
}

static mlir::LogicalResult planModuleDDRMemory(mlir::ModuleOp moduleOp,
                                               int64_t defaultAlignment,
                                               int64_t capacityBytes,
                                               int64_t largestContiguousBytes,
                                               int64_t bandwidthLimitBytes) {
  bool sawFunction = false;
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::func::FuncOp funcOp) {
    sawFunction = true;
    if (mlir::failed(result))
      return;
    result = planScopeDDRMemory(funcOp.getOperation(), defaultAlignment,
                                capacityBytes, largestContiguousBytes,
                                bandwidthLimitBytes);
  });
  if (mlir::failed(result) || sawFunction)
    return result;
  return planScopeDDRMemory(moduleOp.getOperation(), defaultAlignment,
                            capacityBytes, largestContiguousBytes,
                            bandwidthLimitBytes);
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
