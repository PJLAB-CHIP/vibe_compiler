//===- AcceptDDRMemoryPlan.cpp - Accept Wafer DDR memory plan ------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace wafer {
#define GEN_PASS_DEF_ACCEPTDDRMEMORYPLANPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

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

struct DDRDemandSummary {
  llvm::DenseMap<mlir::Value, int64_t> rootBytes;
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

static mlir::LogicalResult verifyPoolAndDomain(mlir::Operation *op,
                                               mlir::Value root) {
  mlir::Operation *def = root.getDefiningOp();
  if (!def)
    return mlir::success();

  if (auto pool =
          def->getAttrOfType<MemoryPoolAttr>(kWaferMemoryPoolAttrName)) {
    MemoryPool value = pool.getValue();
    if (value == MemoryPool::NPUBin || value == MemoryPool::Log)
      return op->emitError()
             << "unsupported_domain_or_pool: ordinary DDR tensor allocation "
                "cannot use "
             << stringifyMemoryPool(value) << " pool";
  }

  if (auto domain =
          def->getAttrOfType<MemoryDomainAttr>(kWaferMemoryDomainAttrName)) {
    MemoryDomain value = domain.getValue();
    if (value != MemoryDomain::LocalDRAM && value != MemoryDomain::RemoteDRAM)
      return op->emitError()
             << "unsupported_domain_or_pool: unsupported DDR domain "
             << stringifyMemoryDomain(value);
  }

  if (mlir::isa<mlir::memref::AllocOp>(def) &&
      isWaferDDRMemRefType(root.getType()))
    return op->emitError()
           << "unsupported_compiler_managed_ddr: compiler-managed DDR "
              "allocation requires explicit owner, lifetime and "
              "suballocation requirement interface";

  return mlir::success();
}

static mlir::FailureOr<DDRView> resolveDDRView(mlir::Operation *op,
                                               mlir::Value ddrValue,
                                               llvm::StringRef role) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(ddrValue.getType());
  if (!viewType || !isWaferDDRMemRefType(viewType))
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " operand must be a Wafer DDR memref";

  llvm::SmallVector<int64_t, 4> viewStrides;
  int64_t viewOffsetElements = 0;
  if (mlir::failed(getStaticMemrefViewInfo(op, viewType, viewStrides,
                                           viewOffsetElements, role)))
    return mlir::failure();

  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  if (!viewInfo || viewInfo->elementBytes <= 0 || viewInfo->bitPackedElement ||
      viewInfo->physicalBytes < 0)
    return op->emitError() << "unsupported_ddr_view: cannot compute " << role
                           << " DDR view physical bytes";

  mlir::Value root = getRootViewSource(ddrValue);
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  if (!rootType || !isWaferDDRMemRefType(rootType))
    return op->emitError() << "unsupported_ddr_view: " << role
                           << " DDR view root must be a Wafer DDR memref";

  std::optional<WaferPhysicalTensorInfo> rootInfo =
      computeWaferPhysicalTensorInfo(rootType);
  if (!rootInfo || rootInfo->elementBytes <= 0 || rootInfo->bitPackedElement ||
      rootInfo->physicalBytes < 0)
    return op->emitError() << "unsupported_ddr_view: cannot compute " << role
                           << " DDR root physical bytes";
  if (viewInfo->elementBytes != rootInfo->elementBytes)
    return op->emitError() << "unsupported_ddr_view: DDR view and root element "
                              "byte sizes differ";

  int64_t viewOffsetBytes = 0;
  if (!checkedMul(viewOffsetElements, viewInfo->elementBytes, viewOffsetBytes))
    return op->emitError()
           << "range_end_overflow: DDR view byte offset overflows int64";

  if (mlir::failed(verifyPoolAndDomain(op, root)))
    return mlir::failure();

  return DDRView{root,
                 rootType,
                 viewType,
                 viewOffsetBytes,
                 viewInfo->physicalBytes,
                 rootInfo->physicalBytes};
}

static mlir::LogicalResult acceptDDRAccess(mlir::Operation *op,
                                           mlir::Value ddrValue,
                                           const MovementDescriptor &descriptor,
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

  mlir::FailureOr<DDRView> view = resolveDDRView(op, ddrValue, descriptor.role);
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

  summary.rootBytes.try_emplace(view->root, view->rootBytes);
  if (!checkedAdd(summary.bandwidthBytes, descriptor.byteCount,
                  summary.bandwidthBytes))
    return op->emitError()
           << "bandwidth_pressure_too_high: DDR bandwidth byte sum overflows";
  return mlir::success();
}

static mlir::LogicalResult verifyResourceLimits(mlir::Operation *op,
                                                const DDRDemandSummary &summary,
                                                int64_t capacityBytes,
                                                int64_t largestContiguousBytes,
                                                int64_t bandwidthLimitBytes) {
  int64_t totalRootBytes = 0;
  for (auto entry : summary.rootBytes) {
    int64_t rootBytes = entry.second;
    if (rootBytes > largestContiguousBytes)
      return op->emitError()
             << "largest_contiguous_range_too_small: DDR root demand "
             << rootBytes << " exceeds largest contiguous range "
             << largestContiguousBytes;
    if (!checkedAdd(totalRootBytes, rootBytes, totalRootBytes))
      return op->emitError()
             << "pool_capacity_overflow: DDR root byte sum overflows";
  }

  if (totalRootBytes > capacityBytes)
    return op->emitError() << "pool_capacity_overflow: DDR root demand "
                           << totalRootBytes << " exceeds capacity "
                           << capacityBytes;

  if (summary.bandwidthBytes > bandwidthLimitBytes)
    return op->emitError() << "bandwidth_pressure_too_high: DDR movement bytes "
                           << summary.bandwidthBytes << " exceed limit "
                           << bandwidthLimitBytes;

  return mlir::success();
}

static mlir::LogicalResult acceptScope(mlir::Operation *scope,
                                       int64_t capacityBytes,
                                       int64_t largestContiguousBytes,
                                       int64_t bandwidthLimitBytes) {
  DDRDemandSummary summary;
  mlir::LogicalResult result = mlir::success();
  scope->walk([&](mlir::Operation *op) {
    if (mlir::failed(result))
      return;

    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
      MovementDescriptor descriptor{
          rdma.getByteCountAttr().getInt(), rdma.getInnerBytesAttr().getInt(),
          rdma.getSrcStridesAttr(), rdma.getSrcIterationsAttr(), "source"};
      result = acceptDDRAccess(op, rdma.getSource(), descriptor, summary);
      return;
    }

    if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
      MovementDescriptor descriptor{
          wdma.getByteCountAttr().getInt(), wdma.getInnerBytesAttr().getInt(),
          wdma.getDstStridesAttr(), wdma.getDstIterationsAttr(), "dest"};
      result = acceptDDRAccess(op, wdma.getDest(), descriptor, summary);
      return;
    }
  });
  if (mlir::failed(result))
    return mlir::failure();
  return verifyResourceLimits(scope, summary, capacityBytes,
                              largestContiguousBytes, bandwidthLimitBytes);
}

static mlir::LogicalResult acceptModule(mlir::ModuleOp moduleOp,
                                        int64_t capacityBytes,
                                        int64_t largestContiguousBytes,
                                        int64_t bandwidthLimitBytes) {
  bool sawFunction = false;
  mlir::LogicalResult result = mlir::success();
  moduleOp.walk([&](mlir::func::FuncOp funcOp) {
    sawFunction = true;
    if (mlir::failed(result))
      return;
    result = acceptScope(funcOp.getOperation(), capacityBytes,
                         largestContiguousBytes, bandwidthLimitBytes);
  });
  if (mlir::failed(result) || sawFunction)
    return result;
  return acceptScope(moduleOp.getOperation(), capacityBytes,
                     largestContiguousBytes, bandwidthLimitBytes);
}

struct AcceptDDRMemoryPlanPass
    : public impl::AcceptDDRMemoryPlanPassBase<AcceptDDRMemoryPlanPass> {
  using impl::AcceptDDRMemoryPlanPassBase<
      AcceptDDRMemoryPlanPass>::AcceptDDRMemoryPlanPassBase;

  void runOnOperation() final {
    if (ddrCapacityBytes < 0 || ddrLargestContiguousBytes < 0 ||
        ddrBandwidthLimitBytes < 0) {
      getOperation()->emitError()
          << "unsupported_domain_or_pool: DDR resource limits must be "
             "non-negative";
      signalPassFailure();
      return;
    }

    if (mlir::failed(acceptModule(getOperation(), ddrCapacityBytes,
                                  ddrLargestContiguousBytes,
                                  ddrBandwidthLimitBytes)))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
