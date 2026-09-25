//===- PackedMovementLowering.cpp - Exact packed BOOL reads -------------===//

#include "Internal.h"

#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <numeric>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

struct BitOffsetTerm {
  mlir::Value index;
  int64_t stride;
};

struct PackedReadGeometry {
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
std::optional<PackedReadGeometry> provePackedRead(mlir::Value source,
                                                  mlir::Value destination,
                                                  mlir::Operation *owner) {
  auto src = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto dst = mlir::dyn_cast<mlir::MemRefType>(destination.getType());
  auto srcMemory = src ? getWaferMemoryAttr(src) : MemoryAttr{};
  auto dstMemory = dst ? getWaferMemoryAttr(dst) : MemoryAttr{};
  if (!src || !dst || !srcMemory || !dstMemory ||
      srcMemory.getSpace() != MemorySpace::DDR ||
      dstMemory.getSpace() != MemorySpace::SPM ||
      srcMemory.getLayout() != MemLayout::Tensor ||
      dstMemory.getLayout() != MemLayout::Tensor ||
      !src.getElementType().isInteger(1) ||
      !dst.getElementType().isInteger(1) || src.getShape() != dst.getShape() ||
      !src.hasStaticShape() || !dst.getLayout().isIdentity() ||
      llvm::any_of(src.getShape(), [](int64_t size) { return size <= 0; }))
    return std::nullopt;
  // Repacking may set unused bits in the last destination byte. This is
  // valid only for the complete, exclusively owned destination allocation.
  mlir::Value destRoot = destination;
  while (auto cast = destRoot.getDefiningOp<mlir::memref::CastOp>())
    destRoot = cast.getSource();
  auto destAlloc = destRoot.getDefiningOp<mlir::memref::AllocOp>();
  if (!destAlloc || destAlloc.getType() != dst)
    return std::nullopt;

  PackedReadGeometry geometry;
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

} // namespace

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
  auto rootType = mlir::cast<mlir::MemRefType>(geometry->root.getType());
  auto flatType = mlir::MemRefType::get(
      {geometry->rootBits}, rewriter.getI1Type(),
      mlir::MemRefLayoutAttrInterface{}, rootType.getMemorySpace());
  mlir::Value flat = rewriter.create<mlir::memref::ReinterpretCastOp>(
      loc, flatType, geometry->root, int64_t(0),
      llvm::ArrayRef<int64_t>{geometry->rootBits}, llvm::ArrayRef<int64_t>{1});
  mlir::OpFoldResult offset =
      rewriter.getIndexAttr(geometry->staticOffset - geometry->remainder);
  if (!geometry->terms.empty()) {
    mlir::Value total = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, geometry->staticOffset);
    for (auto term : geometry->terms) {
      mlir::Value scale =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, term.stride);
      mlir::Value contribution =
          rewriter.create<mlir::arith::MulIOp>(loc, term.index, scale);
      total = rewriter.create<mlir::arith::AddIOp>(loc, total, contribution);
    }
    mlir::Value remainder =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, geometry->remainder);
    offset =
        rewriter.create<mlir::arith::SubIOp>(loc, total, remainder).getResult();
  }
  mlir::Value window = rewriter.create<mlir::memref::SubViewOp>(
      loc, flat, llvm::ArrayRef<mlir::OpFoldResult>{offset},
      llvm::ArrayRef<mlir::OpFoldResult>{
          rewriter.getIndexAttr(geometry->windowBits)},
      llvm::ArrayRef<mlir::OpFoldResult>{rewriter.getIndexAttr(1)});
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
