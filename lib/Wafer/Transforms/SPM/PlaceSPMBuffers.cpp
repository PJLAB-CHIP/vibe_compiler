//===- PlaceSPMBuffers.cpp - Place Wafer SPM memrefs ---------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#define GEN_PASS_DEF_PLACESPMBUFFERSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct SPMDemand {
  mlir::memref::AllocOp alloc;
  int64_t size = 0;
  int64_t alignment = 0;
  int64_t startEvent = 0;
  int64_t endEvent = 0;
  unsigned ordinal = 0;
};

struct PlacedSPMInterval {
  int64_t startEvent = 0;
  int64_t endEvent = 0;
  int64_t offset = 0;
  int64_t end = 0;
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

static std::optional<int64_t> ceilDivNonNegative(int64_t numerator,
                                                 int64_t denominator) {
  if (numerator < 0 || denominator <= 0)
    return std::nullopt;
  int64_t biased = 0;
  if (!checkedAdd(numerator, denominator - 1, biased))
    return std::nullopt;
  return biased / denominator;
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

static bool isDirectTileRegionBodyOp(TileRegionOp tileRegion,
                                     mlir::Operation *op) {
  return op->getParentRegion() == &tileRegion.getBody();
}

static bool hasNestedRegions(mlir::Operation *op) {
  return op->getNumRegions() != 0;
}

static bool lifetimesOverlap(const SPMDemand &demand,
                             const PlacedSPMInterval &placed) {
  return demand.startEvent <= placed.endEvent &&
         placed.startEvent <= demand.endEvent;
}

static bool byteRangesOverlap(int64_t lhsBegin, int64_t lhsEnd,
                              int64_t rhsBegin, int64_t rhsEnd) {
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

static void assignOperationEvents(
    TileRegionOp tileRegion,
    llvm::DenseMap<mlir::Operation *, int64_t> &operationEvents,
    int64_t &regionEndEvent) {
  int64_t nextEvent = 0;
  tileRegion.getBody().walk([&](mlir::Operation *op) {
    if (!belongsToTileRegion(tileRegion, op))
      return;
    operationEvents[op] = nextEvent++;
  });
  regionEndEvent = nextEvent;
}

static void markFullRegionLifetime(SPMDemand &demand, int64_t regionEndEvent) {
  demand.startEvent = 0;
  demand.endEvent = regionEndEvent;
}

static void recordUse(SPMDemand &demand, mlir::Operation *user,
                      int64_t userEvent, TileRegionOp tileRegion,
                      int64_t regionEndEvent) {
  if (!isDirectTileRegionBodyOp(tileRegion, user) || hasNestedRegions(user)) {
    markFullRegionLifetime(demand, regionEndEvent);
    return;
  }
  demand.endEvent = std::max(demand.endEvent, userEvent);
}

static mlir::LogicalResult collectSPMDemands(
    TileRegionOp tileRegion, int64_t defaultAlignment,
    const llvm::DenseMap<mlir::Operation *, int64_t> &operationEvents,
    int64_t regionEndEvent, llvm::SmallVectorImpl<SPMDemand> &demands) {
  mlir::LogicalResult result = mlir::success();
  llvm::DenseMap<mlir::Value, unsigned> valueToDemand;
  tileRegion.walk([&](mlir::memref::AllocOp alloc) {
    if (mlir::failed(result) ||
        !belongsToTileRegion(tileRegion, alloc.getOperation()))
      return;

    mlir::MemRefType memrefType = alloc.getType();
    if (!isWaferSPMMemRefType(memrefType))
      return;

    if (!alloc.getDynamicSizes().empty() ||
        !alloc.getSymbolOperands().empty()) {
      alloc.emitError()
          << "unsupported_layout_conversion: SPM placement requires static "
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

    auto eventIt = operationEvents.find(alloc.getOperation());
    if (eventIt == operationEvents.end()) {
      alloc.emitError()
          << "lifetime_overlap_conflict: missing event for SPM allocation";
      result = mlir::failure();
      return;
    }

    unsigned demandIndex = demands.size();
    SPMDemand demand{alloc,           info->physicalBytes, requiredAlignment,
                     eventIt->second, eventIt->second,     demandIndex};
    if (!isDirectTileRegionBodyOp(tileRegion, alloc.getOperation()))
      markFullRegionLifetime(demand, regionEndEvent);

    demands.push_back(demand);
    valueToDemand[alloc.getMemref()] = demandIndex;
  });

  if (mlir::failed(result))
    return result;

  tileRegion.getBody().walk([&](mlir::Operation *op) {
    if (!belongsToTileRegion(tileRegion, op))
      return;

    auto eventIt = operationEvents.find(op);
    if (eventIt == operationEvents.end())
      return;
    int64_t opEvent = eventIt->second;

    for (mlir::Value operand : op->getOperands()) {
      auto demandIt = valueToDemand.find(operand);
      if (demandIt == valueToDemand.end())
        continue;
      recordUse(demands[demandIt->second], op, opEvent, tileRegion,
                regionEndEvent);
    }

    if (auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      auto demandIt = valueToDemand.find(viewLike.getViewSource());
      if (demandIt == valueToDemand.end())
        return;
      for (mlir::Value resultValue : op->getResults()) {
        if (isWaferSPMMemRefType(resultValue.getType()))
          valueToDemand[resultValue] = demandIt->second;
      }
    }
  });

  return result;
}

static mlir::FailureOr<int64_t>
findFirstFitOffset(const SPMDemand &demand,
                   llvm::ArrayRef<PlacedSPMInterval> placedIntervals,
                   int64_t spmBase, int64_t spmLimit) {
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
    for (const PlacedSPMInterval &placed : placedIntervals) {
      if (!lifetimesOverlap(demand, placed))
        continue;
      if (!byteRangesOverlap(*alignedOffset, candidateEnd, placed.offset,
                             placed.end))
        continue;
      nextCandidate = std::max(nextCandidate, placed.end);
    }

    if (nextCandidate == *alignedOffset)
      return *alignedOffset;
    candidate = nextCandidate;
  }
}

static mlir::LogicalResult placeRegion(TileRegionOp tileRegion, int64_t spmBase,
                                       int64_t spmLimit, int64_t spmAlignment) {
  llvm::DenseMap<mlir::Operation *, int64_t> operationEvents;
  int64_t regionEndEvent = 0;
  assignOperationEvents(tileRegion, operationEvents, regionEndEvent);

  llvm::SmallVector<SPMDemand, 8> demands;
  if (mlir::failed(collectSPMDemands(tileRegion, spmAlignment, operationEvents,
                                     regionEndEvent, demands)))
    return mlir::failure();

  llvm::sort(demands, [](const SPMDemand &lhs, const SPMDemand &rhs) {
    if (lhs.startEvent != rhs.startEvent)
      return lhs.startEvent < rhs.startEvent;
    if (lhs.endEvent != rhs.endEvent)
      return lhs.endEvent > rhs.endEvent;
    return lhs.ordinal < rhs.ordinal;
  });

  llvm::SmallVector<PlacedSPMInterval, 8> placedIntervals;
  for (SPMDemand demand : demands) {
    mlir::FailureOr<int64_t> offset =
        findFirstFitOffset(demand, placedIntervals, spmBase, spmLimit);
    if (mlir::failed(offset)) {
      demand.alloc.emitError()
          << "capacity_overflow: SPM placement range [" << spmBase << ", "
          << spmLimit << ") cannot fit " << demand.size
          << " byte buffer with lifetime [" << demand.startEvent << ", "
          << demand.endEvent << "]";
      return mlir::failure();
    }

    int64_t end = 0;
    if (!checkedAdd(*offset, demand.size, end)) {
      demand.alloc.emitError()
          << "range_end_overflow: SPM placement end address overflows int64";
      return mlir::failure();
    }

    if (*offset < spmBase || end > spmLimit) {
      demand.alloc.emitError()
          << "capacity_overflow: SPM placement range [" << spmBase << ", "
          << spmLimit << ") cannot fit " << demand.size
          << " byte buffer at aligned offset " << *offset;
      return mlir::failure();
    }

    int64_t bankBegin = *offset / kWaferSPMBankLineBytes;
    std::optional<int64_t> bankLimit =
        ceilDivNonNegative(end, kWaferSPMBankLineBytes);
    if (!bankLimit) {
      demand.alloc.emitError()
          << "range_end_overflow: cannot compute SPM bank span";
      return mlir::failure();
    }

    demand.alloc->setAttr(
        kWaferSPMPlacementAttrName,
        SPMPlacementAttr::get(demand.alloc.getContext(), *offset, demand.size,
                              demand.alignment, bankBegin, *bankLimit));
    placedIntervals.push_back(
        PlacedSPMInterval{demand.startEvent, demand.endEvent, *offset, end});
  }

  return mlir::success();
}

struct PlaceSPMBuffersPass
    : public impl::PlaceSPMBuffersPassBase<PlaceSPMBuffersPass> {
  using impl::PlaceSPMBuffersPassBase<
      PlaceSPMBuffersPass>::PlaceSPMBuffersPassBase;

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
      result = placeRegion(tileRegion, spmBase, spmLimit, spmAlignment);
    });
    if (mlir::failed(result))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
