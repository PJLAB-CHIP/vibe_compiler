//===- PlaceSPMBuffers.cpp - Place Wafer SPM memrefs ---------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
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

static mlir::LogicalResult
collectSPMDemands(TileRegionOp tileRegion, int64_t defaultAlignment,
                  llvm::SmallVectorImpl<SPMDemand> &demands) {
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

    demands.push_back(SPMDemand{alloc, info->physicalBytes, requiredAlignment});
  });
  return result;
}

static mlir::LogicalResult placeRegion(TileRegionOp tileRegion, int64_t spmBase,
                                       int64_t spmLimit, int64_t spmAlignment) {
  llvm::SmallVector<SPMDemand, 8> demands;
  if (mlir::failed(collectSPMDemands(tileRegion, spmAlignment, demands)))
    return mlir::failure();

  int64_t cursor = spmBase;
  for (SPMDemand demand : demands) {
    std::optional<int64_t> alignedOffset = alignUp(cursor, demand.alignment);
    if (!alignedOffset) {
      demand.alloc.emitError()
          << "alignment_unsatisfied: cannot align SPM offset " << cursor
          << " to " << demand.alignment << " bytes";
      return mlir::failure();
    }

    int64_t end = 0;
    if (!checkedAdd(*alignedOffset, demand.size, end)) {
      demand.alloc.emitError()
          << "range_end_overflow: SPM placement end address overflows int64";
      return mlir::failure();
    }

    if (*alignedOffset < spmBase || end > spmLimit) {
      demand.alloc.emitError()
          << "capacity_overflow: SPM placement range [" << spmBase << ", "
          << spmLimit << ") cannot fit " << demand.size
          << " byte buffer at aligned offset " << *alignedOffset;
      return mlir::failure();
    }

    int64_t bankBegin = *alignedOffset / kWaferSPMBankLineBytes;
    std::optional<int64_t> bankLimit =
        ceilDivNonNegative(end, kWaferSPMBankLineBytes);
    if (!bankLimit) {
      demand.alloc.emitError()
          << "range_end_overflow: cannot compute SPM bank span";
      return mlir::failure();
    }

    demand.alloc->setAttr(kWaferSPMPlacementAttrName,
                          SPMPlacementAttr::get(demand.alloc.getContext(),
                                                *alignedOffset, demand.size,
                                                demand.alignment, bankBegin,
                                                *bankLimit));
    cursor = end;
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
