//===- PackedMovementLowering.cpp - Exact packed BOOL movement ----------===//

#include "Internal.h"

#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <numeric>
#include <type_traits>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

struct BitOffsetTerm {
  mlir::Value index;
  int64_t stride;
};

struct PackedWindow {
  mlir::Value root;
  llvm::SmallVector<BitOffsetTerm> terms;
  llvm::SmallVector<int64_t> strides;
  int64_t staticOffset = 0;
  int64_t remainder = 0;
  int64_t rootBits = 0;
  int64_t windowBits = 0;
};

// Only explicit view chains are accepted. The local root must expose the
// complete allocation at offset zero; do not recover a larger buffer from
// names, matching shapes, or an enclosing region's unrelated operands.
std::optional<PackedWindow> provePackedWindow(mlir::Value source,
                                              mlir::Operation *owner) {
  auto src = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto srcMemory = src ? getWaferMemoryAttr(src) : MemoryAttr{};
  if (!src || !srcMemory || srcMemory.getLayout() != MemLayout::Tensor ||
      !src.getElementType().isInteger(1) || !src.hasStaticShape() ||
      llvm::any_of(src.getShape(), [](int64_t size) { return size <= 0; }))
    return std::nullopt;
  PackedWindow geometry;
  int64_t ignored;
  if (mlir::failed(mlir::getStridesAndOffset(src, geometry.strides, ignored)) ||
      llvm::any_of(geometry.strides,
                   [](int64_t stride) { return stride <= 0; }))
    return std::nullopt;
  __int128 minOffset = 0, maxOffset = 0, constantOffset = 0;
  uint64_t remainder = 0;
  mlir::Value root = source;
  while (true) {
    if (auto cast = root.getDefiningOp<mlir::memref::CastOp>()) {
      root = cast.getSource();
      continue;
    }
    if (auto collapse = root.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
      root = collapse.getSrc();
      continue;
    }
    if (auto expand = root.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
      root = expand.getSrc();
      continue;
    }
    auto subview = root.getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview)
      break;
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(mlir::getStridesAndOffset(subview.getSourceType(), strides,
                                               ignored)))
      return std::nullopt;
    for (auto [offset, stride] :
         llvm::zip_equal(subview.getMixedOffsets(), strides)) {
      if (stride <= 0)
        return std::nullopt;
      if (auto constant = mlir::getConstantIntValue(offset)) {
        if (*constant < 0)
          return std::nullopt;
        __int128 contribution = static_cast<__int128>(*constant) * stride;
        constantOffset += contribution;
        minOffset += contribution;
        maxOffset += contribution;
        remainder = (remainder + (*constant % 8) * (stride % 8)) % 8;
      } else {
        mlir::Value index = mlir::cast<mlir::Value>(offset);
        auto range =
            memory_planning::detail::evaluateNonNegativeStaticIndexRange(index,
                                                                         owner);
        auto residue = memory_planning::detail::getKnownIndexRemainder(
            index, 8 / std::gcd(static_cast<uint64_t>(stride), uint64_t(8)));
        if (!range.succeeded() || range.range.empty || !residue)
          return std::nullopt;
        minOffset += static_cast<__int128>(range.range.min) * stride;
        maxOffset += static_cast<__int128>(range.range.max) * stride;
        remainder = (remainder + *residue * (stride % 8)) % 8;
        geometry.terms.push_back({index, stride});
      }
      if (maxOffset > std::numeric_limits<int64_t>::max())
        return std::nullopt;
    }
    root = subview.getSource();
  }
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  if (!rootType ||
      (!mlir::isa<mlir::BlockArgument>(root) &&
       !root.getDefiningOp<mlir::memref::AllocOp>()) ||
      !rootType.getLayout().isIdentity() ||
      rootType.getMemorySpace() != src.getMemorySpace() ||
      rootType.getElementType() != src.getElementType())
    return std::nullopt;
  auto rootInfo = computeWaferPhysicalTensorInfo(rootType);
  if (!rootInfo || rootInfo->physicalBytes <= 0 ||
      rootInfo->physicalBytes > std::numeric_limits<int64_t>::max() / 8)
    return std::nullopt;
  __int128 span = 1;
  for (auto [size, stride] :
       llvm::zip_equal(src.getShape(), geometry.strides)) {
    span += static_cast<__int128>(size - 1) * stride;
    if (span > std::numeric_limits<int64_t>::max())
      return std::nullopt;
  }
  __int128 window = (span + remainder + 7) / 8 * 8;
  geometry.rootBits = rootInfo->physicalBytes * 8;
  // Also bound the expanded target traversal and its byte descriptors.
  if (minOffset < remainder || window <= 0 ||
      window > std::numeric_limits<uint32_t>::max() / 2 ||
      maxOffset - remainder + window > geometry.rootBits)
    return std::nullopt;
  geometry.root = root;
  geometry.staticOffset = static_cast<int64_t>(constantOffset);
  geometry.remainder = remainder;
  geometry.windowBits = static_cast<int64_t>(window);
  return geometry;
}

std::optional<PackedWindow> provePackedRead(mlir::Value source,
                                            mlir::Value destination,
                                            mlir::Operation *owner) {
  auto src = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto dst = mlir::dyn_cast<mlir::MemRefType>(destination.getType());
  auto srcMemory = src ? getWaferMemoryAttr(src) : MemoryAttr{};
  auto dstMemory = dst ? getWaferMemoryAttr(dst) : MemoryAttr{};
  if (!src || !dst || !srcMemory || !dstMemory ||
      srcMemory.getSpace() != MemorySpace::DDR ||
      dstMemory.getSpace() != MemorySpace::SPM ||
      dstMemory.getLayout() != MemLayout::Tensor ||
      !dst.getElementType().isInteger(1) || src.getShape() != dst.getShape() ||
      !dst.getLayout().isIdentity())
    return std::nullopt;
  // Repacking may set unused bits in the last destination byte. This is
  // valid only for the complete, exclusively owned destination allocation.
  mlir::Value root = destination;
  while (auto cast = root.getDefiningOp<mlir::memref::CastOp>())
    root = cast.getSource();
  auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation || allocation.getType() != dst)
    return std::nullopt;
  return provePackedWindow(source, owner);
}

// This query belongs to a function-scoped conversion, not a TileRegion pass:
// region bindings and every alias use must be visible in the same current IR.
bool hasPrivateSequentialStorage(mlir::Value localRoot,
                                 mlir::func::FuncOp function) {
  mlir::Value root = localRoot;
  while (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root)) {
    auto region =
        mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp());
    if (!region || !function->isAncestor(region) ||
        argument.getArgNumber() >= region.getInputs().size())
      return false;
    root = region.getInputs()[argument.getArgNumber()];
  }
  auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation || !function->isAncestor(allocation) ||
      allocation.getType() != localRoot.getType())
    return false;

  llvm::SmallVector<mlir::Value> pending{root};
  llvm::DenseSet<mlir::Value> visited;
  while (!pending.empty()) {
    auto value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (auto &use : value.getUses()) {
      auto *operation = use.getOwner();
      if (!function->isAncestor(operation))
        return false;
      for (auto *parent = operation->getParentOp(); parent != function;
           parent = parent->getParentOp())
        if (mlir::isa<mlir::scf::ParallelOp, mlir::scf::ForallOp,
                      mlir::async::ExecuteOp>(parent))
          return false;
      if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(operation)) {
        if (view.getViewSource() != value || operation->getNumResults() != 1)
          return false;
        pending.push_back(operation->getResult(0));
      } else if (auto region = mlir::dyn_cast<TileRegionOp>(operation)) {
        if (use.getOperandNumber() >= region.getInputs().size())
          return false;
        pending.push_back(
            region.getBody().front().getArgument(use.getOperandNumber()));
      } else if (!mlir::isa<mlir::memref::CopyOp, mlir::memref::DeallocOp,
                            StorageLoadOp, StorageStoreOp, MoveCopyIntoOp,
                            ComputeFillOp, ComputeElementwiseOp,
                            ComputeElementwiseIntoOp, InstrRDMAOp, InstrWDMAOp,
                            InstrGatherScatterOp, InstrTDMADataMoveOp,
                            InstrFillOp, InstrBit2FpOp, InstrElementwiseOp>(
                     operation)) {
        // In particular, calls, returns, pointer exposure, peer transfers and
        // unproved loop-carried aliases do not establish private storage.
        return false;
      }
    }
  }
  return true;
}

mlir::Value materializeWindow(const PackedWindow &geometry,
                              mlir::Location location,
                              mlir::PatternRewriter &rewriter) {
  auto rootType = mlir::cast<mlir::MemRefType>(geometry.root.getType());
  auto flatType = mlir::MemRefType::get(
      {geometry.rootBits}, rewriter.getI1Type(),
      mlir::MemRefLayoutAttrInterface{}, rootType.getMemorySpace());
  mlir::Value flat = rewriter.create<mlir::memref::ReinterpretCastOp>(
      location, flatType, geometry.root, int64_t(0),
      llvm::ArrayRef<int64_t>{geometry.rootBits}, llvm::ArrayRef<int64_t>{1});
  mlir::OpFoldResult offset =
      rewriter.getIndexAttr(geometry.staticOffset - geometry.remainder);
  if (!geometry.terms.empty()) {
    mlir::Value total = rewriter.create<mlir::arith::ConstantIndexOp>(
        location, geometry.staticOffset);
    for (auto term : geometry.terms) {
      mlir::Value scale =
          rewriter.create<mlir::arith::ConstantIndexOp>(location, term.stride);
      mlir::Value contribution =
          rewriter.create<mlir::arith::MulIOp>(location, term.index, scale);
      total =
          rewriter.create<mlir::arith::AddIOp>(location, total, contribution);
    }
    mlir::Value remainder = rewriter.create<mlir::arith::ConstantIndexOp>(
        location, geometry.remainder);
    offset = rewriter.create<mlir::arith::SubIOp>(location, total, remainder)
                 .getResult();
  }
  return rewriter.create<mlir::memref::SubViewOp>(
      location, flat, llvm::ArrayRef<mlir::OpFoldResult>{offset},
      llvm::ArrayRef<mlir::OpFoldResult>{
          rewriter.getIndexAttr(geometry.windowBits)},
      llvm::ArrayRef<mlir::OpFoldResult>{rewriter.getIndexAttr(1)});
}

mlir::LogicalResult materializePrivateUpdate(
    mlir::Operation *owner, mlir::Value source, mlir::Value destination,
    mlir::func::FuncOp function, mlir::PatternRewriter &rewriter,
    TileRegionToInstrBufferRecorder *recorder, MovementDescriptorCache &cache) {
  auto src = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto dst = mlir::dyn_cast<mlir::MemRefType>(destination.getType());
  auto srcMemory = src ? getWaferMemoryAttr(src) : MemoryAttr{};
  auto dstMemory = dst ? getWaferMemoryAttr(dst) : MemoryAttr{};
  if (!src || !dst || !srcMemory || !dstMemory ||
      srcMemory.getSpace() != MemorySpace::SPM ||
      (dstMemory.getSpace() != MemorySpace::SPM &&
       dstMemory.getSpace() != MemorySpace::DDR) ||
      srcMemory.getLayout() != MemLayout::Tensor ||
      !src.getElementType().isInteger(1) || src.getShape() != dst.getShape() ||
      !src.getLayout().isIdentity() ||
      mlir::failed(memory_planning::detail::proveByteAlignedPackedView(source)))
    return rewriter.notifyMatchFailure(
        owner, "requires a compact aligned BOOL source");
  // The existing descriptor path handles exact byte rows. Do not replace it
  // with read/modify/write or change the established uninstrumented packages.
  if (mlir::succeeded(
          analysis::TransferRealizability::provePackedByteRows(dst)) &&
      mlir::succeeded(
          memory_planning::detail::proveByteAlignedPackedView(destination)))
    return rewriter.notifyMatchFailure(owner, "existing packed byte movement");
  auto geometry = provePackedWindow(destination, owner);
  if (!geometry || !hasPrivateSequentialStorage(geometry->root, function))
    return rewriter.notifyMatchFailure(
        owner, "requires a bounded window in private sequential storage");
  // A complete compact allocation already owns its last padding byte.
  if (destination == geometry->root && dst.getLayout().isIdentity())
    return rewriter.notifyMatchFailure(owner, "complete packed allocation");

  auto makeType = [&](llvm::ArrayRef<int64_t> shape, mlir::Type element) {
    return mlir::MemRefType::get(shape, element,
                                 mlir::MemRefLayoutAttrInterface{},
                                 src.getMemorySpace());
  };
  auto packedType = makeType({geometry->windowBits}, rewriter.getI1Type());
  auto expandedType = makeType({geometry->windowBits}, rewriter.getF16Type());
  auto compactType = makeType(src.getShape(), rewriter.getF16Type());
  auto zeroType = makeType({1}, rewriter.getF16Type());
  auto destinationType = mlir::MemRefType::get(
      dst.getShape(), rewriter.getF16Type(),
      mlir::StridedLayoutAttr::get(rewriter.getContext(), 0, geometry->strides),
      src.getMemorySpace());
  auto relation = analysis::IndexRelation::identity(dst.getShape());
  if (!relation.isExact())
    return mlir::failure();
  auto descriptors =
      cache.getOrCreate(rewriter, owner, compactType, destinationType,
                        dst.getShape(), *relation.get(), *relation.get(),
                        MovementEngine::GatherScatter, "private packed update");
  auto bytes = getContiguousDescriptor(rewriter, owner, packedType);
  if (mlir::failed(descriptors) || mlir::failed(bytes))
    return mlir::failure();
  MovementDescriptorPlan insertion(**descriptors);
  for (auto &descriptor : insertion)
    descriptor.dest.byteOffset += geometry->remainder * 2;

  // All geometry, closure and descriptor checks precede the first mutation.
  // Reading the enclosing bytes preserves unknown neighbouring bits too;
  // none of them is assumed initialized to a particular value.
  auto location = owner->getLoc();
  auto window = materializeWindow(*geometry, location, rewriter);
  auto allocate = [&](mlir::MemRefType type) {
    return *createDestAlloc(location, type, rewriter, owner, recorder);
  };
  auto record = [&](mlir::Operation *operation) {
    if (recorder)
      recorder->recordLoweredOperation(owner, operation);
  };
  auto packed = allocate(packedType);
  auto expanded = allocate(expandedType);
  auto compact = allocate(compactType);
  auto zero = allocate(zeroType);
  bool ddr = dstMemory.getSpace() == MemorySpace::DDR;
  if (ddr)
    record(createRDMA(rewriter, location, window, packed, *bytes));
  else
    record(createGatherScatter(rewriter, location, window, packed, *bytes,
                               *bytes));
  record(rewriter.create<InstrBit2FpOp>(location, packed, expanded));
  record(rewriter.create<InstrBit2FpOp>(location, source, compact));
  if (mlir::failed(emitGatherScatterDescriptorPlan(
          rewriter, location, owner, compact, expanded, insertion, recorder)))
    return mlir::failure();
  mlir::Value falseValue = rewriter.create<mlir::arith::ConstantOp>(
      location, rewriter.getFloatAttr(rewriter.getF16Type(), 0));
  record(rewriter.create<InstrFillOp>(
      location, zero, falseValue,
      FillDomainAttr::get(rewriter.getContext(), FillDomain::PhysicalFootprint),
      getDefaultNCCWorkerAttr(rewriter)));
  auto repack = rewriter.create<InstrElementwiseOp>(
      location,
      InstrElementwiseKindAttr::get(rewriter.getContext(),
                                    InstrElementwiseKind::Ne),
      mlir::ValueRange{expanded, zero}, packed,
      getDefaultNCCWorkerAttr(rewriter));
  repack.setRhsUnitElements(1);
  record(repack);
  if (ddr)
    record(createWDMA(rewriter, location, packed, window, *bytes));
  else
    record(createGatherScatter(rewriter, location, packed, window, *bytes,
                               *bytes));
  rewriter.eraseOp(owner);
  return mlir::success();
}

template <typename Op> struct PrivatePackedUpdate : mlir::OpRewritePattern<Op> {
  PrivatePackedUpdate(mlir::MLIRContext *context, mlir::func::FuncOp function,
                      TileRegionToInstrBufferRecorder *recorder,
                      MovementDescriptorCache *cache)
      : mlir::OpRewritePattern<Op>(context), function(function),
        recorder(recorder), cache(cache) {}
  mlir::LogicalResult
  matchAndRewrite(Op operation,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value destination;
    if constexpr (std::is_same_v<Op, mlir::memref::CopyOp>)
      destination = operation.getTarget();
    else
      destination = operation.getDest();
    return materializePrivateUpdate(operation, operation.getSource(),
                                    destination, function, rewriter, recorder,
                                    *cache);
  }
  mlir::func::FuncOp function;
  TileRegionToInstrBufferRecorder *recorder;
  MovementDescriptorCache *cache;
};

} // namespace

mlir::LogicalResult wafer::tile_region_to_instr::convertPrivatePackedUpdates(
    mlir::func::FuncOp function, TileRegionToInstrBufferRecorder *recorder,
    MovementDescriptorCache &cache, mlir::RewriterBase::Listener *listener) {
  llvm::SmallVector<mlir::Operation *> operations;
  function.walk([&](mlir::Operation *operation) {
    if (!mlir::isa<StorageStoreOp, MoveCopyIntoOp, mlir::memref::CopyOp>(
            operation))
      return;
    auto source =
        mlir::dyn_cast<mlir::MemRefType>(operation->getOperand(0).getType());
    if (source && source.getElementType().isInteger(1))
      operations.push_back(operation);
  });
  if (operations.empty())
    return mlir::success();
  mlir::RewritePatternSet patterns(function.getContext());
  patterns.add<PrivatePackedUpdate<StorageStoreOp>,
               PrivatePackedUpdate<MoveCopyIntoOp>,
               PrivatePackedUpdate<mlir::memref::CopyOp>>(
      function.getContext(), function, recorder, &cache);
  mlir::GreedyRewriteConfig config;
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
  config.maxNumRewrites = operations.size();
  config.scope = &function.getBody();
  config.listener = listener;
  return mlir::applyOpPatternsAndFold(
      operations, mlir::FrozenRewritePatternSet(std::move(patterns)), config);
}

mlir::LogicalResult wafer::tile_region_to_instr::materializePackedRead(
    mlir::Operation *owner, mlir::Value source, mlir::Value destination,
    mlir::PatternRewriter &rewriter,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache) {
  auto geometry = provePackedRead(source, destination, owner);
  if (!geometry)
    return failPattern(rewriter, owner,
                       "packed read requires a bounded byte window, fixed "
                       "bit residue and complete destination allocation");
  auto dst = mlir::cast<mlir::MemRefType>(destination.getType());
  auto memory = dst.getMemorySpace();
  auto makeType = [&](llvm::ArrayRef<int64_t> shape, mlir::Type element) {
    return mlir::MemRefType::get(shape, element,
                                 mlir::MemRefLayoutAttrInterface{}, memory);
  };
  auto packedType = makeType({geometry->windowBits}, rewriter.getI1Type());
  auto expandedType = makeType({geometry->windowBits}, rewriter.getF16Type());
  auto compactType = makeType(dst.getShape(), rewriter.getF16Type());
  auto zeroType = makeType({1}, rewriter.getF16Type());
  auto sourceType = mlir::MemRefType::get(
      dst.getShape(), rewriter.getF16Type(),
      mlir::StridedLayoutAttr::get(rewriter.getContext(), 0, geometry->strides),
      memory);
  auto relation = analysis::IndexRelation::identity(dst.getShape());
  if (!relation.isExact())
    return mlir::failure();
  auto descriptors = descriptorCache->getOrCreate(
      rewriter, owner, sourceType, compactType, dst.getShape(), *relation.get(),
      *relation.get(), MovementEngine::GatherScatter, "packed read extraction");
  if (mlir::failed(descriptors))
    return mlir::failure();
  auto dma = getContiguousDescriptor(rewriter, owner, packedType);
  if (mlir::failed(dma))
    return mlir::failure();
  MovementDescriptorPlan extraction(**descriptors);
  for (auto &descriptor : extraction)
    descriptor.source.byteOffset += geometry->remainder * 2;

  // All possible geometry failures precede mutation. The window is an actual
  // read-only alias; all following allocations/effects have this Tile owner.
  auto loc = owner->getLoc();
  auto window = materializeWindow(*geometry, loc, rewriter);
  auto allocate = [&](mlir::MemRefType type) {
    return *createDestAlloc(loc, type, rewriter, owner, bufferRecorder);
  };
  auto record = [&](mlir::Operation *operation) {
    if (bufferRecorder)
      bufferRecorder->recordLoweredOperation(owner, operation);
  };
  mlir::Value packed = allocate(packedType);
  mlir::Value expanded = allocate(expandedType);
  mlir::Value compact = allocate(compactType);
  mlir::Value zero = allocate(zeroType);
  record(createRDMA(rewriter, loc, window, packed, *dma));
  record(rewriter.create<InstrBit2FpOp>(loc, packed, expanded));
  if (mlir::failed(emitGatherScatterDescriptorPlan(
          rewriter, loc, owner, expanded, compact, extraction, bufferRecorder)))
    return mlir::failure();
  mlir::Value falseValue = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(rewriter.getF16Type(), 0));
  record(rewriter.create<InstrFillOp>(
      loc, zero, falseValue,
      FillDomainAttr::get(rewriter.getContext(), FillDomain::PhysicalFootprint),
      getDefaultNCCWorkerAttr(rewriter)));
  auto repack = rewriter.create<InstrElementwiseOp>(
      loc,
      InstrElementwiseKindAttr::get(rewriter.getContext(),
                                    InstrElementwiseKind::Ne),
      mlir::ValueRange{compact, zero}, destination,
      getDefaultNCCWorkerAttr(rewriter));
  repack.setRhsUnitElements(1);
  record(repack);
  return mlir::success();
}
