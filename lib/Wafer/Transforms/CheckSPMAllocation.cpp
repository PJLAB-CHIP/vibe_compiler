//===- CheckSPMAllocation.cpp - SPM allocation trial checker -------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer {
namespace {

constexpr int64_t kDefaultUsableSPMBytes = 0x2f0000;
constexpr int64_t kDefaultSPMAlignmentBytes = 256;

struct StorageResult {
  static StorageResult success(uint64_t bytes) { return {bytes, ""}; }
  static StorageResult failure(std::string reason) {
    return {0, std::move(reason)};
  }

  uint64_t bytes;
  std::string reason;
};

struct AllocationDemand {
  mlir::Value value;
  mlir::Operation *definingOp;
  uint64_t bytes;
  uint64_t alignment;
  unsigned lifetimeStart;
  unsigned lifetimeEnd;
};

struct AllocatedRange {
  const AllocationDemand *demand;
  uint64_t begin;
  uint64_t end;
};

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<uint64_t> alignUp(uint64_t value, uint64_t alignment) {
  if (alignment == 0)
    return std::nullopt;
  uint64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  uint64_t increment = alignment - remainder;
  uint64_t aligned = 0;
  if (!checkedAdd(value, increment, aligned))
    return std::nullopt;
  return aligned;
}

static std::optional<uint64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
    std::optional<uint64_t> elementBits =
        getElementBitWidth(complexType.getElementType());
    if (!elementBits)
      return std::nullopt;
    uint64_t complexBits = 0;
    if (!checkedMul(*elementBits, 2, complexBits))
      return std::nullopt;
    return complexBits;
  }
  return std::nullopt;
}

static std::optional<uint64_t>
getStaticElementCount(mlir::RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return std::nullopt;

  uint64_t count = 1;
  for (int64_t dim : tensorType.getShape()) {
    if (dim < 0)
      return std::nullopt;
    uint64_t next = 0;
    if (!checkedMul(count, static_cast<uint64_t>(dim), next))
      return std::nullopt;
    count = next;
  }
  return count;
}

static StorageResult getByteCountForElements(uint64_t elementCount,
                                             uint64_t elementBits,
                                             uint64_t alignment) {
  uint64_t totalBits = 0;
  if (!checkedMul(elementCount, elementBits, totalBits))
    return StorageResult::failure(
        "range_end_overflow while computing storage bits");

  uint64_t bytes = totalBits / 8 + (totalBits % 8 == 0 ? 0 : 1);
  std::optional<uint64_t> alignedBytes = alignUp(bytes, alignment);
  if (!alignedBytes)
    return StorageResult::failure(
        "range_end_overflow while aligning storage bytes");
  return StorageResult::success(*alignedBytes);
}

static StorageResult
getTileBufferStorageBytes(wafer::TileBufferType tileBufferType,
                          uint64_t alignment) {
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(tileBufferType.getTensorType());
  if (!tensorType)
    return StorageResult::failure("tile_buffer logical type is not ranked");
  if (!tensorType.hasStaticShape())
    return StorageResult::failure(
        "dynamic tile_buffer shape has no bounded SPM size");

  std::optional<uint64_t> elementBits =
      getElementBitWidth(tensorType.getElementType());
  if (!elementBits || *elementBits == 0)
    return StorageResult::failure("unsupported tile_buffer element type");

  auto layout =
      mlir::cast<wafer::MemLayoutAttr>(tileBufferType.getLayout()).getValue();
  if (layout == wafer::MemLayout::Tensor ||
      layout == wafer::MemLayout::NTensor) {
    std::optional<uint64_t> elementCount = getStaticElementCount(tensorType);
    if (!elementCount)
      return StorageResult::failure(
          "range_end_overflow while computing tensor elements");
    return getByteCountForElements(*elementCount, *elementBits, alignment);
  }

  if (layout != wafer::MemLayout::Cx && layout != wafer::MemLayout::NCx)
    return StorageResult::failure("unsupported tile_buffer layout");

  if (tensorType.getRank() < 2)
    return StorageResult::failure(
        "cx/ncx storage trial currently requires rank-2 or higher tensors");

  uint64_t rows = 1;
  for (int64_t dim = 0; dim + 1 < tensorType.getRank(); ++dim) {
    int64_t size = tensorType.getDimSize(dim);
    if (size < 0)
      return StorageResult::failure(
          "dynamic cx/ncx tile_buffer shape has no bounded SPM size");
    uint64_t nextRows = 0;
    if (!checkedMul(rows, static_cast<uint64_t>(size), nextRows))
      return StorageResult::failure(
          "range_end_overflow while computing cx/ncx rows");
    rows = nextRows;
  }

  int64_t cols = tensorType.getDimSize(tensorType.getRank() - 1);
  if (cols < 0)
    return StorageResult::failure(
        "dynamic cx/ncx tile_buffer shape has no bounded SPM size");

  uint64_t packingQuantum = *elementBits <= 8 ? 128 : 64;
  std::optional<uint64_t> alignedCols =
      alignUp(static_cast<uint64_t>(cols), packingQuantum);
  if (!alignedCols)
    return StorageResult::failure(
        "range_end_overflow while aligning cx/ncx columns");

  uint64_t elementCount = 0;
  if (!checkedMul(rows, *alignedCols, elementCount))
    return StorageResult::failure(
        "range_end_overflow while computing cx/ncx elements");
  return getByteCountForElements(elementCount, *elementBits, alignment);
}

static bool isSPMTileBufferType(mlir::Type type) {
  auto tileBufferType = mlir::dyn_cast<wafer::TileBufferType>(type);
  if (!tileBufferType)
    return false;
  auto memorySpace =
      mlir::cast<wafer::MemorySpaceAttr>(tileBufferType.getMemorySpace());
  return memorySpace.getValue() == wafer::MemorySpace::SPM;
}

static bool lifetimesOverlap(const AllocationDemand &lhs,
                             const AllocationDemand &rhs) {
  return lhs.lifetimeStart <= rhs.lifetimeEnd &&
         rhs.lifetimeStart <= lhs.lifetimeEnd;
}

static bool rangesOverlap(const AllocatedRange &lhs,
                          const AllocatedRange &rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

static mlir::LogicalResult
collectAllocationDemands(wafer::TileRegionOp tileRegion, uint64_t alignment,
                         llvm::SmallVectorImpl<AllocationDemand> &demands,
                         mlir::Operation *&failedOp, std::string &reason) {
  llvm::DenseMap<mlir::Operation *, unsigned> order;
  unsigned nextIndex = 0;
  tileRegion.getBody().walk(
      [&](mlir::Operation *op) { order[op] = nextIndex++; });

  for (mlir::Operation &op : tileRegion.getBody().front()) {
    for (mlir::OpResult result : op.getResults()) {
      auto tileBufferType =
          mlir::dyn_cast<wafer::TileBufferType>(result.getType());
      if (!tileBufferType)
        continue;
      if (!isSPMTileBufferType(result.getType()))
        continue;

      StorageResult storage =
          getTileBufferStorageBytes(tileBufferType, alignment);
      if (!storage.reason.empty()) {
        failedOp = &op;
        reason = storage.reason;
        return mlir::failure();
      }

      auto definingOrder = order.find(&op);
      if (definingOrder == order.end()) {
        failedOp = &op;
        reason = "internal_error: missing defining op order";
        return mlir::failure();
      }

      unsigned lifetimeEnd = definingOrder->second;
      for (mlir::Operation *user : result.getUsers()) {
        auto userOrder = order.find(user);
        if (userOrder == order.end()) {
          failedOp = &op;
          reason = "lifetime_escape outside tile_region";
          return mlir::failure();
        }
        lifetimeEnd = std::max(lifetimeEnd, userOrder->second);
      }

      demands.push_back(AllocationDemand{result, &op, storage.bytes, alignment,
                                         definingOrder->second, lifetimeEnd});
    }
  }

  return mlir::success();
}

static mlir::LogicalResult
runSequentialAllocationTrial(wafer::TileRegionOp tileRegion, uint64_t capacity,
                             uint64_t alignment) {
  llvm::SmallVector<AllocationDemand> demands;
  mlir::Operation *failedOp = tileRegion.getOperation();
  std::string reason;
  if (mlir::failed(collectAllocationDemands(tileRegion, alignment, demands,
                                            failedOp, reason))) {
    return failedOp->emitOpError("SPM allocation trial failed: ") << reason;
  }

  llvm::SmallVector<AllocatedRange> ranges;
  uint64_t cursor = 0;
  for (const AllocationDemand &demand : demands) {
    std::optional<uint64_t> alignedOffset = alignUp(cursor, demand.alignment);
    if (!alignedOffset)
      return demand.definingOp->emitOpError(
          "SPM allocation trial failed: range_end_overflow while "
          "aligning offset");

    if (*alignedOffset % demand.alignment != 0)
      return demand.definingOp->emitOpError(
          "SPM allocation trial failed: alignment_unsatisfied");

    uint64_t end = 0;
    if (!checkedAdd(*alignedOffset, demand.bytes, end))
      return demand.definingOp->emitOpError(
          "SPM allocation trial failed: range_end_overflow");

    if (end > capacity)
      return demand.definingOp->emitOpError(
                 "SPM allocation trial failed: capacity_overflow")
             << " range [" << *alignedOffset << ", " << end
             << ") exceeds SPM capacity " << capacity;

    AllocatedRange nextRange{&demand, *alignedOffset, end};
    for (const AllocatedRange &existing : ranges) {
      if (lifetimesOverlap(*nextRange.demand, *existing.demand) &&
          rangesOverlap(nextRange, existing))
        return demand.definingOp->emitOpError(
            "SPM allocation trial failed: lifetime_overlap");
    }

    ranges.push_back(nextRange);
    cursor = end;
  }

  return mlir::success();
}

struct CheckSPMAllocationPass
    : public mlir::PassWrapper<CheckSPMAllocationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<CheckSPMAllocationPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckSPMAllocationPass)

  CheckSPMAllocationPass() = default;
  CheckSPMAllocationPass(const CheckSPMAllocationPass &pass) : Base(pass) {
    spmCapacityBytes = pass.spmCapacityBytes;
    alignmentBytes = pass.alignmentBytes;
  }

  mlir::Pass::Option<int64_t> spmCapacityBytes{
      *this, "spm-capacity-bytes",
      llvm::cl::desc("usable SPM capacity in bytes for pass-local trial "
                     "allocation"),
      llvm::cl::init(kDefaultUsableSPMBytes)};

  mlir::Pass::Option<int64_t> alignmentBytes{
      *this, "alignment-bytes",
      llvm::cl::desc("SPM allocation offset and storage alignment in bytes"),
      llvm::cl::init(kDefaultSPMAlignmentBytes)};

  llvm::StringRef getArgument() const final {
    return "wafer-check-spm-allocation";
  }

  llvm::StringRef getDescription() const final {
    return "check pass-local SPM allocation feasibility for tile regions";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
  }

  void runOnOperation() final {
    if (spmCapacityBytes <= 0) {
      getOperation().emitOpError(
          "SPM allocation trial failed: capacity must be positive");
      signalPassFailure();
      return;
    }
    if (alignmentBytes <= 0) {
      getOperation().emitOpError(
          "SPM allocation trial failed: alignment must be positive");
      signalPassFailure();
      return;
    }

    bool failed = false;
    uint64_t capacity = static_cast<uint64_t>(spmCapacityBytes);
    uint64_t alignment = static_cast<uint64_t>(alignmentBytes);
    getOperation().walk([&](wafer::TileRegionOp tileRegion) {
      if (mlir::failed(
              runSequentialAllocationTrial(tileRegion, capacity, alignment))) {
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckSPMAllocationPass() {
  return std::make_unique<CheckSPMAllocationPass>();
}

} // namespace wafer
