//===- PlanSPMMemory.cpp - Plan Wafer SPM memory --------------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
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
#define GEN_PASS_DEF_PLANSPMMEMORYPASS
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

struct SPMDemand {
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

struct AssignedSPMInterval {
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

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static std::optional<int64_t> alignUp(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;
  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t delta = alignment - remainder;
  int64_t result = 0;
  if (!checkedAdd(value, delta, result))
    return std::nullopt;
  return result;
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

static bool lifetimesOverlap(const SPMDemand &lhs, const SPMDemand &rhs) {
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

static mlir::LogicalResult assignOperationEvents(TileRegionOp tileRegion,
                                                 EventInfo &events) {
  assignRegionEvents(tileRegion.getBody(), PathCondition{}, events);
  if (events.branchLimitExceeded)
    return tileRegion.emitError() << "lifetime_overlap_conflict: SPM memory "
                                     "planning supports at most 64 "
                                     "nested branch decision points";
  return mlir::success();
}

static void addLiveSegment(SPMDemand &demand, int64_t startEvent,
                           int64_t endEvent, PathCondition condition) {
  if (endEvent < startEvent)
    return;
  demand.segments.push_back(LiveSegment{startEvent, endEvent, condition});
}

static void recordDemandUse(SPMDemand &demand, int64_t event,
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

static void computeLifetimeBounds(SPMDemand &demand) {
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

static int64_t getLifetimeSpan(const SPMDemand &demand) {
  if (demand.lastLiveEvent <= demand.firstLiveEvent)
    return 0;
  return demand.lastLiveEvent - demand.firstLiveEvent;
}

static void
computePlanningPriorities(llvm::MutableArrayRef<SPMDemand> demands) {
  for (SPMDemand &demand : demands) {
    demand.conflictBytes = 0;
    computeLifetimeBounds(demand);
  }

  for (size_t lhsIndex = 0; lhsIndex < demands.size(); ++lhsIndex) {
    SPMDemand &lhs = demands[lhsIndex];
    for (SPMDemand &rhs : demands.drop_front(lhsIndex + 1)) {
      if (!lifetimesOverlap(lhs, rhs))
        continue;
      addSaturated(lhs.conflictBytes, rhs.size);
      addSaturated(rhs.conflictBytes, lhs.size);
    }
  }
}

static bool hasHigherPlanningPriority(const SPMDemand &lhs,
                                      const SPMDemand &rhs) {
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
               llvm::MutableArrayRef<SPMDemand> demands,
               const llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>>
                   &valueRefs) {
  for (RootRef ref : getRefsAtUse(value, useCondition, valueRefs))
    recordDemandUse(demands[ref.demandIndex], event, ref.condition);
}

static void
recordTokenUse(mlir::Value value, int64_t event, PathCondition useCondition,
               llvm::MutableArrayRef<SPMDemand> demands,
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

static bool hasAsyncTokenType(mlir::Value value) {
  return mlir::isa<mlir::async::TokenType>(value.getType());
}

static mlir::scf::YieldOp getSingleBlockYield(mlir::Region &region) {
  if (region.empty())
    return {};
  return mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
}

struct LifetimeDataflow {
  const EventInfo &events;
  llvm::MutableArrayRef<SPMDemand> demands;
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
      if (isWaferSPMMemRefType(result.getType()))
        valueRefs[result] = refs;
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

  void mapIfResults(mlir::scf::IfOp ifOp) {
    mlir::scf::YieldOp thenYield = getSingleBlockYield(ifOp.getThenRegion());
    mlir::scf::YieldOp elseYield = getSingleBlockYield(ifOp.getElseRegion());
    if (!thenYield || !elseYield)
      return;

    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isWaferSPMMemRefType(result.getType()))
        continue;
      llvm::SmallVector<RootRef, 2> refs;
      appendYieldOperandRefs(thenYield, index, refs);
      appendYieldOperandRefs(elseYield, index, refs);
      if (!refs.empty())
        valueRefs[result] = refs;
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
      if (!isWaferSPMMemRefType(result.getType()))
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
      } else {
        for (mlir::Region &region : op.getRegions())
          processRegion(region);
        mapViewLikeResults(&op);
      }

      mapAsyncTokenResults(&op);
    }
  }
};

static mlir::LogicalResult initializeSPMDemands(
    TileRegionOp tileRegion, int64_t defaultAlignment, const EventInfo &events,
    llvm::SmallVectorImpl<SPMDemand> &demands,
    llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> &valueRefs) {
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
      requiredAlignment =
          std::max(requiredAlignment, static_cast<int64_t>(*allocAlignment));
    }

    auto eventIt = events.operationEvents.find(alloc.getOperation());
    auto conditionIt = events.operationConditions.find(alloc.getOperation());
    if (eventIt == events.operationEvents.end() ||
        conditionIt == events.operationConditions.end()) {
      alloc.emitError()
          << "lifetime_overlap_conflict: missing event for SPM allocation";
      result = mlir::failure();
      return;
    }

    unsigned demandIndex = demands.size();
    SPMDemand demand{alloc,           info->physicalBytes, requiredAlignment,
                     eventIt->second, conditionIt->second, demandIndex};
    demands.push_back(demand);
    valueRefs[alloc.getMemref()] = llvm::SmallVector<RootRef, 2>{
        RootRef{demandIndex, conditionIt->second}};
  });
  return result;
}

static mlir::LogicalResult
collectSPMDemands(TileRegionOp tileRegion, int64_t defaultAlignment,
                  const EventInfo &events,
                  llvm::SmallVectorImpl<SPMDemand> &demands) {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> valueRefs;
  if (mlir::failed(initializeSPMDemands(tileRegion, defaultAlignment, events,
                                        demands, valueRefs)))
    return mlir::failure();

  llvm::DenseMap<mlir::Value, llvm::SmallVector<RootRef, 2>> tokenRefs;
  LifetimeDataflow dataflow{events, demands, std::move(valueRefs),
                            std::move(tokenRefs)};
  dataflow.processRegion(tileRegion.getBody());
  return mlir::success();
}

static mlir::FailureOr<int64_t>
findFirstFitOffset(const SPMDemand &demand,
                   llvm::ArrayRef<AssignedSPMInterval> assignedIntervals,
                   llvm::ArrayRef<SPMDemand> demands, int64_t spmBase,
                   int64_t spmLimit) {
  int64_t candidate = spmBase;
  while (true) {
    std::optional<int64_t> alignedOffset = alignUp(candidate, demand.alignment);
    if (!alignedOffset)
      return mlir::failure();

    int64_t candidateEnd = 0;
    if (!checkedAdd(*alignedOffset, demand.size, candidateEnd))
      return mlir::failure();
    if (*alignedOffset < spmBase || candidateEnd > spmLimit)
      return mlir::failure();

    int64_t nextCandidate = *alignedOffset;
    for (const AssignedSPMInterval &assigned : assignedIntervals) {
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

static mlir::LogicalResult planRegion(TileRegionOp tileRegion, int64_t spmBase,
                                      int64_t spmLimit, int64_t spmAlignment) {
  EventInfo events;
  if (mlir::failed(assignOperationEvents(tileRegion, events)))
    return mlir::failure();

  llvm::SmallVector<SPMDemand, 8> demands;
  if (mlir::failed(
          collectSPMDemands(tileRegion, spmAlignment, events, demands)))
    return mlir::failure();

  computePlanningPriorities(demands);
  llvm::sort(demands, hasHigherPlanningPriority);

  llvm::SmallVector<AssignedSPMInterval, 8> assignedIntervals;
  for (auto [demandIndex, demand] : llvm::enumerate(demands)) {
    mlir::FailureOr<int64_t> offset = findFirstFitOffset(
        demand, assignedIntervals, demands, spmBase, spmLimit);
    if (mlir::failed(offset)) {
      demand.alloc.emitError()
          << "capacity_overflow: SPM planning range [" << spmBase << ", "
          << spmLimit << ") cannot fit " << demand.size
          << " byte buffer with IR-derived lifetime";
      return mlir::failure();
    }

    int64_t end = 0;
    if (!checkedAdd(*offset, demand.size, end)) {
      demand.alloc.emitError()
          << "range_end_overflow: SPM planning end address overflows int64";
      return mlir::failure();
    }

    if (*offset < spmBase || end > spmLimit) {
      demand.alloc.emitError()
          << "capacity_overflow: SPM planning range [" << spmBase << ", "
          << spmLimit << ") cannot fit " << demand.size
          << " byte buffer at aligned offset " << *offset;
      return mlir::failure();
    }

    demand.alloc->setAttr(
        kWaferSPMOffsetAttrName,
        SPMOffsetAttr::get(demand.alloc.getContext(), *offset));
    assignedIntervals.push_back(
        AssignedSPMInterval{static_cast<unsigned>(demandIndex), *offset, end});
  }

  return mlir::success();
}

struct PlanSPMMemoryPass
    : public impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass> {
  using impl::PlanSPMMemoryPassBase<PlanSPMMemoryPass>::PlanSPMMemoryPassBase;

  void runOnOperation() final {
    if (spmBase < 0 || spmLimit <= spmBase) {
      getOperation()->emitError()
          << "invalid_spm_range: expected 0 <= spm-base < spm-limit";
      signalPassFailure();
      return;
    }
    if (spmAlignment <= 0) {
      getOperation()->emitError()
          << "alignment_unsatisfied: spm-alignment must be positive";
      signalPassFailure();
      return;
    }

    mlir::LogicalResult result = mlir::success();
    getOperation().walk([&](TileRegionOp tileRegion) {
      if (mlir::failed(result))
        return;
      result = planRegion(tileRegion, spmBase, spmLimit, spmAlignment);
    });
    if (mlir::failed(result))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
