//===- MovementSupport.cpp - Tile-region movement support ---------------===//

#include "Internal.h"

#include "Wafer/Analysis/Tile/PhysicalAccessRelation.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>

namespace wafer::tile_region_to_instr {

namespace {

/// Local compiler-resource guard for one relation decomposition. This is not
/// an executable-program or workload legality limit: alternative tiling or a
/// more compact target descriptor can still represent the logical movement.
constexpr size_t kMaximumStaticMovementDescriptorMaterializations = 4096;

struct DescriptorLoop {
  int64_t strideBytes = 0;
  int64_t iterations = 1;
};

} // namespace

std::optional<int64_t>
getStaticPositiveElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic || dim <= 0)
      return std::nullopt;
    if (count > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    count *= dim;
  }
  return count;
}

std::optional<int64_t> checkedMulI64(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::nullopt;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return std::nullopt;
  return lhs * rhs;
}

std::optional<llvm::SmallVector<MovementDescriptorPair, 4>>
getExactTensorToBlockedDescriptors(mlir::MemRefType sourceType,
                                   mlir::MemRefType destType,
                                   mlir::AffineMap destToSource) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfo =
      computeWaferPhysicalTensorInfo(destType);
  if (!sourceMemory || !destMemory || !sourceInfo || !destInfo ||
      sourceMemory.getSpace() != MemorySpace::SPM ||
      destMemory.getSpace() != MemorySpace::SPM ||
      sourceMemory.getLayout() != MemLayout::Tensor ||
      (destMemory.getLayout() != MemLayout::Cx &&
       destMemory.getLayout() != MemLayout::NCx) ||
      sourceType.getElementType() != destType.getElementType() ||
      sourceInfo->elementBytes <= 0 ||
      sourceInfo->elementBytes != destInfo->elementBytes ||
      destInfo->cBlock <= 0 || destInfo->cxBlocks <= 0 ||
      destInfo->blockOuterElements <= 0 ||
      destInfo->outerSliceStrideElements <= 0 || sourceInfo->bitPackedElement ||
      destInfo->bitPackedElement || destType.getRank() < 2 ||
      !sourceType.hasStaticShape() || !destType.hasStaticShape() ||
      !destToSource || destToSource.getNumDims() != destType.getRank() ||
      destToSource.getNumResults() != sourceType.getRank() ||
      destToSource.getNumSymbols() != 0 ||
      !destToSource.isProjectedPermutation())
    return std::nullopt;
  const int64_t logicalChannels = destType.getShape().back();
  if (logicalChannels != destInfo->cxBlocks * destInfo->cBlock)
    return std::nullopt;

  llvm::SmallVector<int64_t, 6> sourceDimensionForDest(destType.getRank(), -1);
  for (auto [sourceDimension, expression] :
       llvm::enumerate(destToSource.getResults())) {
    auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dim || dim.getPosition() >= sourceDimensionForDest.size() ||
        sourceDimensionForDest[dim.getPosition()] != -1 ||
        sourceType.getDimSize(sourceDimension) !=
            destType.getDimSize(dim.getPosition()))
      return std::nullopt;
    sourceDimensionForDest[dim.getPosition()] = sourceDimension;
  }

  llvm::SmallVector<int64_t, 6> sourceElementStrides;
  int64_t sourceElementOffset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(sourceType, sourceElementStrides,
                                             sourceElementOffset)) ||
      sourceElementOffset != 0 ||
      llvm::is_contained(sourceElementStrides, mlir::ShapedType::kDynamic))
    return std::nullopt;

  struct Axis {
    int64_t count = 1;
    int64_t sourceStrideBytes = 0;
    int64_t destStrideBytes = 0;
  };
  llvm::SmallVector<Axis, 6> axes;
  const int64_t channelDimension = destType.getRank() - 1;
  const int64_t sourceChannelDimension =
      sourceDimensionForDest[channelDimension];
  if (sourceChannelDimension < 0)
    return std::nullopt;
  const int64_t firstOuterDimension =
      destMemory.getLayout() == MemLayout::NCx ? 1 : 0;
  int64_t laterOuterElements = 1;
  llvm::SmallVector<Axis, 6> reversedOuter;
  for (int64_t dimension = channelDimension - 1;
       dimension >= firstOuterDimension; --dimension) {
    int64_t extent = destType.getDimSize(dimension);
    if (extent > 1) {
      int64_t sourceDimension = sourceDimensionForDest[dimension];
      std::optional<int64_t> sourceStride =
          sourceDimension < 0
              ? std::optional<int64_t>(0)
              : checkedMulI64(sourceElementStrides[sourceDimension],
                              sourceInfo->elementBytes);
      std::optional<int64_t> destElements =
          checkedMulI64(laterOuterElements, destInfo->cBlock);
      std::optional<int64_t> destStride =
          destElements ? checkedMulI64(*destElements, destInfo->elementBytes)
                       : std::nullopt;
      if (!sourceStride || !destStride)
        return std::nullopt;
      reversedOuter.push_back({extent, *sourceStride, *destStride});
    }
    std::optional<int64_t> next = checkedMulI64(laterOuterElements, extent);
    if (!next)
      return std::nullopt;
    laterOuterElements = *next;
  }
  axes.append(reversedOuter.rbegin(), reversedOuter.rend());
  if (destInfo->cxBlocks > 1) {
    std::optional<int64_t> channelBlockElements = checkedMulI64(
        sourceElementStrides[sourceChannelDimension], destInfo->cBlock);
    std::optional<int64_t> channelBlockStride =
        channelBlockElements
            ? checkedMulI64(*channelBlockElements, sourceInfo->elementBytes)
            : std::nullopt;
    std::optional<int64_t> destBlockElements =
        checkedMulI64(destInfo->blockOuterElements, destInfo->cBlock);
    std::optional<int64_t> destBlockStride =
        destBlockElements
            ? checkedMulI64(*destBlockElements, destInfo->elementBytes)
            : std::nullopt;
    if (!channelBlockStride || !destBlockStride)
      return std::nullopt;
    axes.push_back({destInfo->cxBlocks, *channelBlockStride, *destBlockStride});
  }
  if (destMemory.getLayout() == MemLayout::NCx && destType.getDimSize(0) > 1) {
    int64_t sourceBatchDimension = sourceDimensionForDest[0];
    std::optional<int64_t> sourceSliceStride =
        sourceBatchDimension < 0
            ? std::optional<int64_t>(0)
            : checkedMulI64(sourceElementStrides[sourceBatchDimension],
                            sourceInfo->elementBytes);
    std::optional<int64_t> destSliceStride = checkedMulI64(
        destInfo->outerSliceStrideElements, destInfo->elementBytes);
    if (!sourceSliceStride || !destSliceStride)
      return std::nullopt;
    axes.push_back(
        {destType.getDimSize(0), *sourceSliceStride, *destSliceStride});
  }

  const size_t encodedAxes = std::min<size_t>(3, axes.size());
  const size_t splitAxes = axes.size() - encodedAxes;
  int64_t commandCount = 1;
  for (Axis axis : llvm::ArrayRef(axes).take_front(splitAxes)) {
    std::optional<int64_t> next = checkedMulI64(commandCount, axis.count);
    if (!next || *next > 4096)
      return std::nullopt;
    commandCount = *next;
  }
  std::optional<int64_t> innerBytes =
      checkedMulI64(destInfo->cBlock, destInfo->elementBytes);
  if (!innerBytes || *innerBytes <= 0)
    return std::nullopt;
  int64_t bytesPerCommand = *innerBytes;
  for (Axis axis : llvm::ArrayRef(axes).drop_front(splitAxes)) {
    std::optional<int64_t> next = checkedMulI64(bytesPerCommand, axis.count);
    if (!next)
      return std::nullopt;
    bytesPerCommand = *next;
  }
  if (static_cast<uint64_t>(bytesPerCommand) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(*innerBytes) > std::numeric_limits<uint32_t>::max())
    return std::nullopt;

  llvm::SmallVector<MovementDescriptorPair, 4> descriptors;
  descriptors.reserve(commandCount);
  std::function<bool(size_t, int64_t, int64_t)> emit = [&](size_t axisIndex,
                                                           int64_t sourceBase,
                                                           int64_t destBase) {
    if (axisIndex != splitAxes) {
      const Axis &axis = axes[axisIndex];
      for (int64_t iteration = 0; iteration < axis.count; ++iteration) {
        __int128 source =
            static_cast<__int128>(sourceBase) +
            static_cast<__int128>(iteration) * axis.sourceStrideBytes;
        __int128 dest = static_cast<__int128>(destBase) +
                        static_cast<__int128>(iteration) * axis.destStrideBytes;
        if (source < 0 || dest < 0 ||
            source > std::numeric_limits<int64_t>::max() ||
            dest > std::numeric_limits<int64_t>::max() ||
            !emit(axisIndex + 1, static_cast<int64_t>(source),
                  static_cast<int64_t>(dest)))
          return false;
      }
      return true;
    }
    MovementDescriptorPair descriptor;
    descriptor.source.byteCount = bytesPerCommand;
    descriptor.dest.byteCount = bytesPerCommand;
    descriptor.source.innerBytes = *innerBytes;
    descriptor.dest.innerBytes = *innerBytes;
    descriptor.source.byteOffset = sourceBase;
    descriptor.dest.byteOffset = destBase;
    descriptor.source.strides.assign({0, 0, 0});
    descriptor.dest.strides.assign({0, 0, 0});
    descriptor.source.iterations.assign({1, 1, 1});
    descriptor.dest.iterations.assign({1, 1, 1});
    for (auto [index, axis] :
         llvm::enumerate(llvm::ArrayRef(axes).drop_front(splitAxes))) {
      descriptor.source.strides[index] = axis.sourceStrideBytes;
      descriptor.dest.strides[index] = axis.destStrideBytes;
      descriptor.source.iterations[index] = axis.count;
      descriptor.dest.iterations[index] = axis.count;
    }
    descriptors.push_back(std::move(descriptor));
    return true;
  };
  if (!emit(0, 0, 0))
    return std::nullopt;
  std::optional<int64_t> covered = checkedMulI64(bytesPerCommand, commandCount);
  std::optional<int64_t> expectedElements =
      getStaticPositiveElementCount(destType.getShape());
  std::optional<int64_t> expected =
      expectedElements
          ? checkedMulI64(*expectedElements, sourceInfo->elementBytes)
          : std::nullopt;
  if (!covered || !expected || *covered != *expected)
    return std::nullopt;
  return descriptors;
}

std::optional<llvm::SmallVector<MovementDescriptorPair, 4>>
getExactBlockedToTensorDescriptors(mlir::MemRefType sourceType,
                                   mlir::MemRefType destType) {
  auto reverse = getExactTensorToBlockedDescriptors(
      destType, sourceType,
      mlir::AffineMap::getMultiDimIdentityMap(sourceType.getRank(),
                                              sourceType.getContext()));
  if (!reverse)
    return std::nullopt;
  for (MovementDescriptorPair &descriptor : *reverse)
    std::swap(descriptor.source, descriptor.dest);
  return reverse;
}

std::optional<DynamicSubviewDescriptor>
getDynamicSubviewDescriptor(mlir::Value source, mlir::Operation *operation) {
  auto subview = source.getDefiningOp<mlir::memref::SubViewOp>();
  if (!subview)
    return std::nullopt;
  auto baseType = mlir::dyn_cast<mlir::MemRefType>(subview.getSourceType());
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(subview.getType());
  MemoryAttr baseMemory =
      baseType ? getWaferMemoryAttr(baseType) : MemoryAttr{};
  std::optional<WaferPhysicalTensorInfo> basePhysical =
      baseType ? computeWaferPhysicalTensorInfo(baseType) : std::nullopt;
  if (!baseType || !viewType || !baseMemory ||
      baseMemory.getLayout() != MemLayout::Tensor || !basePhysical ||
      basePhysical->elementBytes <= 0)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> baseStrides;
  int64_t baseOffset = 0;
  llvm::SmallVector<int64_t, 4> viewStrides;
  int64_t viewOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(baseType, baseStrides, baseOffset)) ||
      mlir::failed(
          mlir::getStridesAndOffset(viewType, viewStrides, viewOffset)) ||
      baseOffset == mlir::ShapedType::kDynamic ||
      baseStrides.size() != subview.getMixedOffsets().size() ||
      llvm::is_contained(baseStrides, mlir::ShapedType::kDynamic) ||
      llvm::is_contained(viewStrides, mlir::ShapedType::kDynamic))
    return std::nullopt;

  DynamicSubviewDescriptor descriptor;
  descriptor.subview = subview;
  descriptor.sourceBase = subview.getSource();
  descriptor.relativeType = mlir::MemRefType::get(
      viewType.getShape(), viewType.getElementType(),
      mlir::StridedLayoutAttr::get(operation->getContext(), /*offset=*/0,
                                   viewStrides),
      viewType.getMemorySpace());
  descriptor.offsets = subview.getMixedOffsets();
  descriptor.byteCoefficients.reserve(baseStrides.size());
  if (baseOffset >
      std::numeric_limits<int64_t>::max() / basePhysical->elementBytes)
    return std::nullopt;
  descriptor.staticByteOffset = baseOffset * basePhysical->elementBytes;
  for (auto [offset, stride] :
       llvm::zip_equal(subview.getMixedOffsets(), baseStrides)) {
    if (stride >
        std::numeric_limits<int64_t>::max() / basePhysical->elementBytes)
      return std::nullopt;
    int64_t coefficient = stride * basePhysical->elementBytes;
    descriptor.byteCoefficients.push_back(coefficient);
    std::optional<int64_t> constant = mlir::getConstantIntValue(offset);
    if (!constant)
      continue;
    __int128 next = static_cast<__int128>(descriptor.staticByteOffset) +
                    static_cast<__int128>(*constant) * coefficient;
    if (next < 0 || next > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    descriptor.staticByteOffset = static_cast<int64_t>(next);
  }
  return descriptor;
}

mlir::Value
materializeDynamicSubviewByteOffset(const DynamicSubviewDescriptor &descriptor,
                                    mlir::PatternRewriter &rewriter,
                                    mlir::Location location) {
  mlir::Value result = rewriter.create<mlir::arith::ConstantIndexOp>(
      location, descriptor.staticByteOffset);
  for (auto [offset, coefficient] :
       llvm::zip_equal(descriptor.offsets, descriptor.byteCoefficients)) {
    if (mlir::getConstantIntValue(offset))
      continue;
    auto dynamic = mlir::dyn_cast<mlir::Value>(offset);
    if (!dynamic)
      return {};
    mlir::Value scale =
        rewriter.create<mlir::arith::ConstantIndexOp>(location, coefficient);
    mlir::Value contribution =
        rewriter.create<mlir::arith::MulIOp>(location, dynamic, scale);
    result =
        rewriter.create<mlir::arith::AddIOp>(location, result, contribution);
  }
  return result;
}

namespace {} // namespace

mlir::LogicalResult failPattern(mlir::PatternRewriter &rewriter,
                                mlir::Operation *op, llvm::StringRef reason) {
  return rewriter.notifyMatchFailure(op, reason);
}

std::optional<mlir::RankedTensorType>
getLogicalTensorTypeFromMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return std::nullopt;
  return mlir::RankedTensorType::get(memrefType.getShape(),
                                     memrefType.getElementType());
}

mlir::FailureOr<mlir::Value>
createDestAlloc(mlir::Location loc, mlir::Type type,
                mlir::PatternRewriter &rewriter, mlir::Operation *op,
                TileRegionToInstrBufferRecorder *bufferRecorder) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<mlir::Value>(
        rewriter, op,
        "tile-region to instr lowering requires memref result storage");
  mlir::Value allocation =
      rewriter.create<mlir::memref::AllocOp>(loc, memrefType).getResult();
  if (bufferRecorder)
    bufferRecorder->recordScratchAllocation(op, allocation);
  return allocation;
}

mlir::FailureOr<MovementDescriptor>
getContiguousDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, "instruction descriptor requires a Wafer memref type");

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        "instruction descriptor requires static positive physical byte size");

  MovementDescriptor descriptor;
  descriptor.byteCount = info->physicalBytes;
  descriptor.innerBytes = info->physicalBytes;
  descriptor.strides.assign({0, 0, 0});
  descriptor.iterations.assign({1, 1, 1});
  return descriptor;
}

mlir::FailureOr<MovementDescriptor>
getStridedTensorDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::Type type, llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        llvm::Twine(role).concat(" requires a Wafer memref type").str());

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info)
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  if (info->bitPackedElement) {
    if (info->compactBytes <= 0 || info->physicalBytes != info->compactBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op,
          llvm::Twine(role)
              .concat(" requires contiguous bitpacked tensor")
              .str());
    MovementDescriptor descriptor;
    descriptor.byteCount = info->compactBytes;
    descriptor.innerBytes = info->compactBytes;
    descriptor.strides.assign({0, 0, 0});
    descriptor.iterations.assign({1, 1, 1});
    return descriptor;
  }

  if (info->compactBytes <= 0 || info->elementBytes <= 0)
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  llvm::SmallVector<int64_t> memrefStrides;
  int64_t memrefOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(memrefType, memrefStrides, memrefOffset)) ||
      static_cast<int64_t>(memrefStrides.size()) != memrefType.getRank())
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        llvm::Twine(role)
            .concat(" requires static strided memref layout")
            .str());

  int64_t innerElements = 1;
  llvm::SmallVector<DescriptorLoop, 3> loops;
  for (int64_t dim = memrefType.getRank() - 1; dim >= 0; --dim) {
    int64_t dimSize = memrefType.getDimSize(dim);
    int64_t dimStride = memrefStrides[dim];
    if (dimSize == mlir::ShapedType::kDynamic || dimSize <= 0 ||
        dimStride == mlir::ShapedType::kDynamic || dimStride < 0)
      return failFailureOr<MovementDescriptor>(
          rewriter, op,
          llvm::Twine(role)
              .concat(" requires static positive shape and non-negative "
                      "strides")
              .str());

    if (dimSize == 1)
      continue;

    if (loops.empty() && dimStride == innerElements) {
      std::optional<int64_t> nextInner = checkedMulI64(innerElements, dimSize);
      if (!nextInner)
        return failFailureOr<MovementDescriptor>(
            rewriter, op,
            llvm::Twine(role).concat(" descriptor inner span overflows").str());
      innerElements = *nextInner;
      continue;
    }

    std::optional<int64_t> strideBytes =
        checkedMulI64(dimStride, info->elementBytes);
    if (!strideBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op,
          llvm::Twine(role).concat(" descriptor stride overflows").str());

    if (!loops.empty()) {
      std::optional<int64_t> collapsedStride =
          checkedMulI64(loops.back().strideBytes, loops.back().iterations);
      if (collapsedStride && *collapsedStride == *strideBytes) {
        std::optional<int64_t> collapsedIterations =
            checkedMulI64(loops.back().iterations, dimSize);
        if (!collapsedIterations)
          return failFailureOr<MovementDescriptor>(
              rewriter, op,
              llvm::Twine(role)
                  .concat(" descriptor iteration overflows")
                  .str());
        loops.back().iterations = *collapsedIterations;
        continue;
      }
    }

    if (loops.size() == 3)
      return failFailureOr<MovementDescriptor>(
          rewriter, op,
          llvm::Twine(role)
              .concat(" requires at most three strided descriptor levels")
              .str());
    loops.push_back({*strideBytes, dimSize});
  }

  std::optional<int64_t> innerBytes =
      checkedMulI64(innerElements, info->elementBytes);
  if (!innerBytes)
    return failFailureOr<MovementDescriptor>(
        rewriter, op,
        llvm::Twine(role)
            .concat(" descriptor inner byte count overflows")
            .str());

  MovementDescriptor descriptor;
  descriptor.byteCount = info->compactBytes;
  descriptor.innerBytes = *innerBytes;
  descriptor.strides.assign({0, 0, 0});
  descriptor.iterations.assign({1, 1, 1});
  for (auto [index, loop] : llvm::enumerate(loops)) {
    descriptor.strides[index] = loop.strideBytes;
    descriptor.iterations[index] = loop.iterations;
  }
  return descriptor;
}

InstrRDMAOp createRDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor) {
  return rewriter.create<InstrRDMAOp>(
      loc, source, dest, descriptor.byteCount, descriptor.innerBytes,
      /*src_offset=*/mlir::IntegerAttr{},
      /*dst_offset=*/mlir::IntegerAttr{}, descriptor.strides,
      descriptor.iterations);
}

InstrWDMAOp createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor) {
  return rewriter.create<InstrWDMAOp>(
      loc, source, dest, descriptor.byteCount, descriptor.innerBytes,
      /*src_offset=*/mlir::IntegerAttr{},
      /*dst_offset=*/mlir::IntegerAttr{}, descriptor.strides,
      descriptor.iterations);
}

InstrGatherScatterOp createGatherScatter(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, const MovementDescriptor &sourceDescriptor,
    const MovementDescriptor &destDescriptor, mlir::Value dynamicSourceOffset,
    mlir::Value dynamicDestOffset) {
  mlir::IntegerAttr sourceOffset =
      sourceDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(sourceDescriptor.byteOffset);
  mlir::IntegerAttr destOffset =
      destDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(destDescriptor.byteOffset);
  return rewriter.create<InstrGatherScatterOp>(
      loc, source, dest, dynamicSourceOffset, dynamicDestOffset,
      destDescriptor.byteCount, destDescriptor.innerBytes, sourceOffset,
      destOffset, sourceDescriptor.strides, sourceDescriptor.iterations,
      destDescriptor.strides, destDescriptor.iterations, DDRResourceAttr{},
      NCCWorker::Worker0);
}

namespace {

struct ProjectedCoordinate {
  // -1 is constant, >=0 is a single projected dimension, and -2 is a
  // multi-dimension affine coordinate recovered from IndexRelation.
  int64_t iterationDim = -1;
  int64_t multiplier = 0;
  int64_t offset = 0;
  llvm::SmallVector<int64_t, 4> multipliers;
};

struct SymbolicAxis {
  int64_t iterationDim = 0;
  int64_t step = 0;
  int64_t count = 1;
};

struct DimensionChoice {
  int64_t base = 0;
  llvm::SmallVector<SymbolicAxis, 2> axes;
};

struct DescriptorAxis {
  int64_t sourceStride = 0;
  int64_t destStride = 0;
  int64_t count = 1;
};

struct DirectPeriodicDecomposition {
  bool requested = false;
  bool requiresExplicitPartition = false;
  int64_t period = 0;
};

std::optional<int64_t> checkedSignedAdd(int64_t lhs, int64_t rhs) {
  __int128 value = static_cast<__int128>(lhs) + rhs;
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

std::optional<int64_t> checkedSignedMul(int64_t lhs, int64_t rhs) {
  __int128 value = static_cast<__int128>(lhs) * rhs;
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

std::optional<llvm::SmallVector<ProjectedCoordinate, 4>>
getProjectedCoordinates(const analysis::IndexRelation &relation,
                        mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> iterationShape,
                        mlir::MemRefType endpointType) {
  std::optional<mlir::AffineMap> map = relation.getProjectedAffineMap(context);
  if (!map || map->getNumDims() != iterationShape.size() ||
      map->getNumResults() != static_cast<unsigned>(endpointType.getRank()) ||
      map->getNumSymbols() != 0)
    return std::nullopt;

  llvm::SmallVector<ProjectedCoordinate, 4> coordinates;
  coordinates.reserve(map->getNumResults());
  for (mlir::AffineExpr result : map->getResults()) {
    llvm::SmallVector<int64_t, 8> coefficients;
    if (mlir::failed(mlir::getFlattenedAffineExpr(
            result, iterationShape.size(), /*numSymbols=*/0, &coefficients)) ||
        coefficients.size() != iterationShape.size() + 1)
      return std::nullopt;

    ProjectedCoordinate coordinate;
    coordinate.offset = coefficients.back();
    coordinate.multipliers.resize(iterationShape.size(), 0);
    unsigned nonzeroMultipliers = 0;
    for (auto [dim, multiplier] :
         llvm::enumerate(llvm::ArrayRef(coefficients).drop_back())) {
      if (multiplier == 0)
        continue;
      coordinate.multipliers[dim] = multiplier;
      ++nonzeroMultipliers;
      if (nonzeroMultipliers == 1) {
        coordinate.iterationDim = dim;
        coordinate.multiplier = multiplier;
      } else {
        coordinate.iterationDim = -2;
        coordinate.multiplier = 0;
      }
    }

    int64_t minimum = coordinate.offset;
    int64_t maximum = coordinate.offset;
    for (auto [dim, multiplier] : llvm::enumerate(coordinate.multipliers)) {
      int64_t iterationSize = iterationShape[dim];
      if (iterationSize <= 0)
        return std::nullopt;
      std::optional<int64_t> delta =
          checkedSignedMul(multiplier, iterationSize - 1);
      if (!delta)
        return std::nullopt;
      std::optional<int64_t> nextMinimum =
          checkedSignedAdd(minimum, std::min<int64_t>(0, *delta));
      std::optional<int64_t> nextMaximum =
          checkedSignedAdd(maximum, std::max<int64_t>(0, *delta));
      if (!nextMinimum || !nextMaximum)
        return std::nullopt;
      minimum = *nextMinimum;
      maximum = *nextMaximum;
    }
    int64_t endpointSize = endpointType.getDimSize(coordinates.size());
    if (endpointSize <= 0 || minimum < 0 || maximum >= endpointSize)
      return std::nullopt;
    coordinates.push_back(coordinate);
  }

  return coordinates;
}

struct CanonicalPeriodicPartition {
  int64_t iterationDim = -1;
  int64_t iterationPeriod = 0;
};

std::optional<CanonicalPeriodicPartition>
getCanonicalPeriodicPartition(const ProjectedCoordinate &coordinate,
                              llvm::ArrayRef<int64_t> iterationShape,
                              int64_t logicalExtent, int64_t period) {
  if (coordinate.offset != 0 || logicalExtent <= 0 || period <= 0 ||
      logicalExtent % period != 0)
    return std::nullopt;
  struct Factor {
    int64_t iterationDim;
    int64_t multiplier;
    int64_t extent;
  };
  llvm::SmallVector<Factor, 4> factors;
  for (auto [dim, multiplier] : llvm::enumerate(coordinate.multipliers)) {
    if (multiplier == 0)
      continue;
    if (multiplier < 0 || iterationShape[dim] <= 0)
      return std::nullopt;
    factors.push_back(
        {static_cast<int64_t>(dim), multiplier, iterationShape[dim]});
  }
  llvm::sort(factors, [](const Factor &lhs, const Factor &rhs) {
    return lhs.multiplier < rhs.multiplier;
  });
  int64_t expectedMultiplier = 1;
  bool hasBlockBoundary = false;
  CanonicalPeriodicPartition partition;
  for (const Factor &factor : factors) {
    if (factor.multiplier != expectedMultiplier)
      return std::nullopt;
    std::optional<int64_t> next =
        checkedSignedMul(expectedMultiplier, factor.extent);
    if (!next)
      return std::nullopt;
    if (expectedMultiplier < period && period < *next) {
      if (partition.iterationDim >= 0 || period % expectedMultiplier != 0)
        return std::nullopt;
      partition.iterationDim = factor.iterationDim;
      partition.iterationPeriod = period / expectedMultiplier;
      if (partition.iterationPeriod <= 0 ||
          factor.extent % partition.iterationPeriod != 0)
        return std::nullopt;
      hasBlockBoundary = true;
    }
    expectedMultiplier = *next;
    hasBlockBoundary |= expectedMultiplier == period;
  }
  if (!hasBlockBoundary || expectedMultiplier != logicalExtent)
    return std::nullopt;
  return partition;
}

bool appendPhysicalPieceBoundaries(
    const ProjectedCoordinate &coordinate, int64_t iterationSize,
    unsigned logicalDim, int64_t logicalExtent,
    llvm::ArrayRef<WaferPhysicalLayoutPiece> pieces,
    llvm::SmallVectorImpl<int64_t> &boundaries) {
  if (coordinate.iterationDim < 0 || coordinate.multiplier == 0)
    return true;

  llvm::SmallVector<int64_t, 16> logicalBoundaries{0, logicalExtent};
  for (const WaferPhysicalLayoutPiece &piece : pieces) {
    if (logicalDim >= piece.logicalLowerBounds.size() ||
        logicalDim >= piece.logicalUpperBounds.size() ||
        logicalDim >= piece.logicalTilePeriods.size())
      return false;
    int64_t lower = piece.logicalLowerBounds[logicalDim];
    int64_t upper = piece.logicalUpperBounds[logicalDim];
    int64_t period = piece.logicalTilePeriods[logicalDim];
    if (lower < 0 || upper < lower || upper > logicalExtent || period < 0)
      return false;
    logicalBoundaries.push_back(lower);
    logicalBoundaries.push_back(upper);
    if (period > 0) {
      for (int64_t boundary = lower; boundary < upper;) {
        std::optional<int64_t> next = checkedSignedAdd(boundary, period);
        if (!next || *next <= boundary)
          return false;
        boundary = *next;
        if (boundary < upper)
          logicalBoundaries.push_back(boundary);
      }
    }
  }
  llvm::sort(logicalBoundaries);
  logicalBoundaries.erase(
      std::unique(logicalBoundaries.begin(), logicalBoundaries.end()),
      logicalBoundaries.end());
  if (logicalBoundaries.front() != 0 ||
      logicalBoundaries.back() != logicalExtent)
    return false;

  int64_t cursor = 0;
  while (cursor < iterationSize) {
    std::optional<int64_t> scaled =
        checkedSignedMul(coordinate.multiplier, cursor);
    std::optional<int64_t> channel =
        scaled ? checkedSignedAdd(coordinate.offset, *scaled) : std::nullopt;
    if (!channel || *channel < 0 || *channel >= logicalExtent)
      return false;

    auto categoryEnd = llvm::upper_bound(logicalBoundaries, *channel);
    if (categoryEnd == logicalBoundaries.begin() ||
        categoryEnd == logicalBoundaries.end())
      return false;
    int64_t categoryLow = *(categoryEnd - 1);
    int64_t categoryHigh = *categoryEnd - 1;

    int64_t last = cursor;
    if (coordinate.multiplier > 0) {
      last = std::min(iterationSize - 1, (categoryHigh - coordinate.offset) /
                                             coordinate.multiplier);
    } else {
      int64_t magnitude = -coordinate.multiplier;
      last = std::min(iterationSize - 1,
                      cursor + (*channel - categoryLow) / magnitude);
    }
    if (last < cursor)
      return false;
    boundaries.push_back(cursor);
    boundaries.push_back(last + 1);
    cursor = last + 1;
  }
  return true;
}

bool descriptorFieldFits(int64_t value) {
  return value >= 0 &&
         static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

} // namespace

mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
getRelationMovementDescriptors(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> iterationShape,
                               const analysis::IndexRelation &iterationToSource,
                               const analysis::IndexRelation &iterationToDest,
                               MovementEngine engine, llvm::StringRef opLabel) {
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::RelationDescriptorPlanning);
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-algorithm", "tile-region-to-instr",
      "relation-descriptor-planning", op->getName().getStringRef());
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "validate-engine-and-layout", op->getName().getStringRef());
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  bool acceptedEngine = sourceMemory && destMemory &&
                        ((engine == MovementEngine::RDMA &&
                          sourceMemory.getSpace() == MemorySpace::DDR &&
                          destMemory.getSpace() == MemorySpace::SPM) ||
                         (engine == MovementEngine::WDMA &&
                          sourceMemory.getSpace() == MemorySpace::SPM &&
                          destMemory.getSpace() == MemorySpace::DDR) ||
                         (engine == MovementEngine::GatherScatter &&
                          sourceMemory.getSpace() == MemorySpace::SPM &&
                          destMemory.getSpace() == MemorySpace::SPM));
  if (!acceptedEngine)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel).concat(" has an invalid movement engine").str());

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "computeWaferPhysicalTensorInfo", op->getName().getStringRef());
  std::optional<WaferPhysicalTensorInfo> sourceInfoStorage =
      computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfoStorage =
      computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfoStorage || !destInfoStorage ||
      sourceInfoStorage->physicalBytes <= 0 ||
      destInfoStorage->physicalBytes <= 0 ||
      sourceInfoStorage->elementBytes <= 0 ||
      sourceInfoStorage->elementBytes != destInfoStorage->elementBytes ||
      sourceInfoStorage->bitPackedElement || destInfoStorage->bitPackedElement)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements in static physical "
                    "layouts")
            .str());
  const WaferPhysicalTensorInfo &sourceInfo = *sourceInfoStorage;
  const WaferPhysicalTensorInfo &destInfo = *destInfoStorage;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "compose-physical-access-relations", op->getName().getStringRef());
  mlir::FailureOr<analysis::PhysicalAccessRelation> sourceAccess =
      analysis::PhysicalAccessRelation::create(sourceType, iterationShape,
                                               iterationToSource,
                                               /*requireInjective=*/false);
  mlir::FailureOr<analysis::PhysicalAccessRelation> destAccess =
      analysis::PhysicalAccessRelation::create(destType, iterationShape,
                                               iterationToDest,
                                               /*requireInjective=*/true);
  if (mlir::failed(sourceAccess) || mlir::failed(destAccess) ||
      !sourceAccess->getPhysicalLayoutRelation().isByteAddressable() ||
      !destAccess->getPhysicalLayoutRelation().isByteAddressable() ||
      sourceAccess->getPhysicalLayoutRelation().getElementBitWidth() !=
          destAccess->getPhysicalLayoutRelation().getElementBitWidth())
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" index relation is not an exact mapped transfer")
            .str());

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "getProjectedCoordinates", op->getName().getStringRef());
  std::optional<int64_t> elementCount =
      getStaticPositiveElementCount(iterationShape);
  std::optional<llvm::SmallVector<ProjectedCoordinate, 4>> sourceMap =
      getProjectedCoordinates(iterationToSource, rewriter.getContext(),
                              iterationShape, sourceType);
  std::optional<llvm::SmallVector<ProjectedCoordinate, 4>> destMap =
      getProjectedCoordinates(iterationToDest, rewriter.getContext(),
                              iterationShape, destType);
  if (!elementCount || !sourceMap || !destMap)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" requires a static projected-affine IndexRelation")
            .str());

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "partition-symbolic-domain", op->getName().getStringRef());
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> boundaries(
      iterationShape.size());
  llvm::SmallVector<DirectPeriodicDecomposition, 4> decompositions(
      iterationShape.size());
  for (auto [dim, size] : llvm::enumerate(iterationShape)) {
    if (size <= 0)
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op,
          llvm::Twine(opLabel)
              .concat(" requires a static positive iteration domain")
              .str());
    boundaries[dim].push_back(0);
    boundaries[dim].push_back(size);
  }

  auto collectLayoutConstraints =
      [&](const analysis::PhysicalAccessRelation &access,
          llvm::ArrayRef<ProjectedCoordinate> map) -> bool {
    llvm::ArrayRef<WaferPhysicalLayoutPiece> pieces =
        access.getPhysicalLayoutRelation().getPieces();
    llvm::ArrayRef<int64_t> logicalShape = access.getEndpointType().getShape();
    if (map.size() != logicalShape.size())
      return false;
    for (unsigned logicalDim = 0; logicalDim < logicalShape.size();
         ++logicalDim) {
      int64_t logicalExtent = logicalShape[logicalDim];
      llvm::SmallVector<int64_t, 2> positivePeriods;
      bool nonTrivialPartition = false;
      for (const WaferPhysicalLayoutPiece &piece : pieces) {
        if (logicalDim >= piece.logicalLowerBounds.size() ||
            logicalDim >= piece.logicalUpperBounds.size() ||
            logicalDim >= piece.logicalTilePeriods.size())
          return false;
        int64_t period = piece.logicalTilePeriods[logicalDim];
        nonTrivialPartition |=
            piece.logicalLowerBounds[logicalDim] != 0 ||
            piece.logicalUpperBounds[logicalDim] != logicalExtent || period > 0;
        if (period > 0)
          positivePeriods.push_back(period);
      }
      if (!nonTrivialPartition)
        continue;
      llvm::sort(positivePeriods);
      positivePeriods.erase(
          std::unique(positivePeriods.begin(), positivePeriods.end()),
          positivePeriods.end());
      if (positivePeriods.size() > 1)
        return false;

      const ProjectedCoordinate &coordinate = map[logicalDim];
      if (coordinate.iterationDim == -1)
        continue;
      if (coordinate.iterationDim == -2) {
        if (positivePeriods.size() != 1)
          return false;
        std::optional<CanonicalPeriodicPartition> partition =
            getCanonicalPeriodicPartition(coordinate, iterationShape,
                                          logicalExtent,
                                          positivePeriods.front());
        if (!partition)
          return false;
        if (partition->iterationDim >= 0) {
          int64_t iterationDim = partition->iterationDim;
          int64_t iterationPeriod = partition->iterationPeriod;
          DirectPeriodicDecomposition &decomposition =
              decompositions[iterationDim];
          if (decomposition.requested &&
              decomposition.period != iterationPeriod)
            return false;
          decomposition.requested = true;
          decomposition.period = iterationPeriod;
          for (int64_t boundary = iterationPeriod;
               boundary < iterationShape[iterationDim];) {
            boundaries[iterationDim].push_back(boundary);
            std::optional<int64_t> next =
                checkedSignedAdd(boundary, iterationPeriod);
            if (!next || *next <= boundary)
              return false;
            boundary = *next;
          }
        }
        continue;
      }

      int64_t iterationDim = coordinate.iterationDim;
      if (!appendPhysicalPieceBoundaries(
              coordinate, iterationShape[iterationDim], logicalDim,
              logicalExtent, pieces, boundaries[iterationDim]))
        return false;
      bool direct = positivePeriods.size() == 1 && coordinate.multiplier == 1 &&
                    coordinate.offset == 0 &&
                    iterationShape[iterationDim] == logicalExtent;
      DirectPeriodicDecomposition &decomposition = decompositions[iterationDim];
      if (!direct) {
        decomposition.requiresExplicitPartition = true;
        continue;
      }
      if (decomposition.requested &&
          decomposition.period != positivePeriods.front())
        return false;
      decomposition.requested = true;
      decomposition.period = positivePeriods.front();
    }
    return true;
  };
  if (!collectLayoutConstraints(*sourceAccess, *sourceMap) ||
      !collectLayoutConstraints(*destAccess, *destMap))
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" cannot partition physical-layout relation pieces")
            .str());

  llvm::SmallVector<llvm::SmallVector<DimensionChoice, 4>, 4> choices(
      iterationShape.size());
  for (int64_t dim = 0; dim < static_cast<int64_t>(iterationShape.size());
       ++dim) {
    int64_t size = iterationShape[dim];
    const DirectPeriodicDecomposition &decomposition = decompositions[dim];
    if (decomposition.requested && !decomposition.requiresExplicitPartition) {
      int64_t fullBlocks = size / decomposition.period;
      int64_t remainder = size % decomposition.period;
      if (fullBlocks > 0) {
        DimensionChoice full;
        full.axes.push_back({dim, 1, decomposition.period});
        full.axes.push_back({dim, decomposition.period, fullBlocks});
        choices[dim].push_back(std::move(full));
      }
      if (remainder > 0) {
        DimensionChoice tail;
        tail.base = fullBlocks * decomposition.period;
        tail.axes.push_back({dim, 1, remainder});
        choices[dim].push_back(std::move(tail));
      }
      continue;
    }

    llvm::sort(boundaries[dim]);
    boundaries[dim].erase(
        std::unique(boundaries[dim].begin(), boundaries[dim].end()),
        boundaries[dim].end());
    for (size_t index = 1; index < boundaries[dim].size(); ++index) {
      int64_t begin = boundaries[dim][index - 1];
      int64_t end = boundaries[dim][index];
      if (begin >= end)
        continue;
      DimensionChoice choice;
      choice.base = begin;
      choice.axes.push_back({dim, 1, end - begin});
      choices[dim].push_back(std::move(choice));
    }
    if (choices[dim].empty())
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op,
          llvm::Twine(opLabel)
              .concat(" produced an empty symbolic domain partition")
              .str());
  }

  llvm::SmallVector<MovementDescriptorPair> descriptors;
  int64_t coveredBytes = 0;
  const int64_t elementBytes = sourceInfo.elementBytes;
  llvm::SmallVector<int64_t, 4> iterationBase(iterationShape.size(), 0);
  llvm::SmallVector<SymbolicAxis, 8> symbolicAxes;

  auto getPhysicalOffset =
      [&](const analysis::PhysicalAccessRelation &access,
          llvm::ArrayRef<int64_t> iteration) -> std::optional<int64_t> {
    mlir::FailureOr<WaferPhysicalElementSpan> span =
        access.getPhysicalElementSpan(iteration);
    if (mlir::failed(span) || span->bitOffset < 0 || span->bitOffset % 8 != 0)
      return std::nullopt;
    return span->bitOffset / 8;
  };

  std::function<mlir::LogicalResult(int64_t, int64_t,
                                    llvm::SmallVector<DescriptorAxis, 8>)>
      materializeDescriptor;
  materializeDescriptor =
      [&](int64_t sourceBase, int64_t destBase,
          llvm::SmallVector<DescriptorAxis, 8> axes) -> mlir::LogicalResult {
    for (DescriptorAxis &axis : axes) {
      if (axis.sourceStride < 0 && axis.destStride < 0) {
        std::optional<int64_t> sourceAdjustment =
            checkedSignedMul(axis.sourceStride, axis.count - 1);
        std::optional<int64_t> destAdjustment =
            checkedSignedMul(axis.destStride, axis.count - 1);
        std::optional<int64_t> adjustedSource =
            sourceAdjustment ? checkedSignedAdd(sourceBase, *sourceAdjustment)
                             : std::nullopt;
        std::optional<int64_t> adjustedDest =
            destAdjustment ? checkedSignedAdd(destBase, *destAdjustment)
                           : std::nullopt;
        if (!adjustedSource || !adjustedDest)
          return mlir::failure();
        sourceBase = *adjustedSource;
        destBase = *adjustedDest;
        axis.sourceStride = -axis.sourceStride;
        axis.destStride = -axis.destStride;
      }
    }

    // Target descriptor fields are unsigned.  A loop whose two endpoints
    // advance in opposite directions cannot be represented as one loop, so
    // split that symbolic axis without discovering it element by element.
    for (size_t index = 0; index < axes.size(); ++index) {
      DescriptorAxis axis = axes[index];
      if (axis.count <= 1 || (axis.sourceStride >= 0 && axis.destStride >= 0))
        continue;
      llvm::SmallVector<DescriptorAxis, 8> remaining = axes;
      remaining.erase(remaining.begin() + index);
      for (int64_t iteration = 0; iteration < axis.count; ++iteration) {
        if (descriptors.size() >=
            kMaximumStaticMovementDescriptorMaterializations)
          return mlir::failure();
        std::optional<int64_t> sourceAdjustment =
            checkedSignedMul(axis.sourceStride, iteration);
        std::optional<int64_t> destAdjustment =
            checkedSignedMul(axis.destStride, iteration);
        std::optional<int64_t> nextSource =
            sourceAdjustment ? checkedSignedAdd(sourceBase, *sourceAdjustment)
                             : std::nullopt;
        std::optional<int64_t> nextDest =
            destAdjustment ? checkedSignedAdd(destBase, *destAdjustment)
                           : std::nullopt;
        if (!nextSource || !nextDest ||
            mlir::failed(
                materializeDescriptor(*nextSource, *nextDest, remaining)))
          return mlir::failure();
      }
      return mlir::success();
    }

    for (size_t index = 0; index < axes.size();) {
      if (axes[index].count <= 1) {
        axes.erase(axes.begin() + index);
        continue;
      }
      ++index;
    }

    auto sequentialEndpointStride = [&](const DescriptorAxis &axis) {
      if (engine == MovementEngine::RDMA)
        return axis.destStride;
      if (engine == MovementEngine::WDMA)
        return axis.sourceStride;
      return axis.destStride;
    };
    llvm::sort(axes, [&](const DescriptorAxis &lhs, const DescriptorAxis &rhs) {
      int64_t lhsPrimary = sequentialEndpointStride(lhs);
      int64_t rhsPrimary = sequentialEndpointStride(rhs);
      if (lhsPrimary != rhsPrimary)
        return lhsPrimary < rhsPrimary;
      if (lhs.sourceStride != rhs.sourceStride)
        return lhs.sourceStride < rhs.sourceStride;
      return lhs.destStride < rhs.destStride;
    });

    // RDMA exposes only source stride/iteration and WDMA only destination
    // stride/iteration in the current target ABI. The opposite endpoint is
    // therefore required to advance as one contiguous payload; when a
    // physical padding boundary breaks that property, retain separate static
    // commands instead of inventing a dynamic endpoint offset form.
    if (engine != MovementEngine::GatherScatter) {
      int64_t expectedStride = elementBytes;
      for (size_t index = 0; index < axes.size(); ++index) {
        if (sequentialEndpointStride(axes[index]) != expectedStride) {
          DescriptorAxis split = axes[index];
          llvm::SmallVector<DescriptorAxis, 8> remaining = axes;
          remaining.erase(remaining.begin() + index);
          for (int64_t iteration = 0; iteration < split.count; ++iteration) {
            if (descriptors.size() >=
                kMaximumStaticMovementDescriptorMaterializations)
              return mlir::failure();
            std::optional<int64_t> sourceAdjustment =
                checkedSignedMul(split.sourceStride, iteration);
            std::optional<int64_t> destAdjustment =
                checkedSignedMul(split.destStride, iteration);
            std::optional<int64_t> nextSource =
                sourceAdjustment
                    ? checkedSignedAdd(sourceBase, *sourceAdjustment)
                    : std::nullopt;
            std::optional<int64_t> nextDest =
                destAdjustment ? checkedSignedAdd(destBase, *destAdjustment)
                               : std::nullopt;
            if (!nextSource || !nextDest ||
                mlir::failed(
                    materializeDescriptor(*nextSource, *nextDest, remaining)))
              return mlir::failure();
          }
          return mlir::success();
        }
        std::optional<int64_t> nextExpected =
            checkedMulI64(expectedStride, axes[index].count);
        if (!nextExpected)
          return mlir::failure();
        expectedStride = *nextExpected;
      }
    }

    int64_t innerBytes = elementBytes;
    for (size_t index = 0; index < axes.size();) {
      DescriptorAxis axis = axes[index];
      if (axis.sourceStride == innerBytes && axis.destStride == innerBytes) {
        std::optional<int64_t> nextInner =
            checkedMulI64(innerBytes, axis.count);
        if (!nextInner)
          return mlir::failure();
        innerBytes = *nextInner;
        axes.erase(axes.begin() + index);
        continue;
      }
      ++index;
    }

    for (size_t index = 0; index + 1 < axes.size();) {
      DescriptorAxis &inner = axes[index];
      DescriptorAxis &outer = axes[index + 1];
      std::optional<int64_t> nextSource =
          checkedSignedMul(inner.sourceStride, inner.count);
      std::optional<int64_t> nextDest =
          checkedSignedMul(inner.destStride, inner.count);
      std::optional<int64_t> mergedCount =
          checkedMulI64(inner.count, outer.count);
      if (nextSource && nextDest && mergedCount &&
          outer.sourceStride == *nextSource && outer.destStride == *nextDest) {
        inner.count = *mergedCount;
        axes.erase(axes.begin() + index + 1);
        continue;
      }
      ++index;
    }

    if (axes.size() > 3) {
      size_t splitIndex = 0;
      for (size_t index = 1; index < axes.size(); ++index)
        if (axes[index].count < axes[splitIndex].count)
          splitIndex = index;
      DescriptorAxis split = axes[splitIndex];
      axes.erase(axes.begin() + splitIndex);
      for (int64_t iteration = 0; iteration < split.count; ++iteration) {
        if (descriptors.size() >=
            kMaximumStaticMovementDescriptorMaterializations)
          return mlir::failure();
        std::optional<int64_t> sourceAdjustment =
            checkedSignedMul(split.sourceStride, iteration);
        std::optional<int64_t> destAdjustment =
            checkedSignedMul(split.destStride, iteration);
        std::optional<int64_t> nextSource =
            sourceAdjustment ? checkedSignedAdd(sourceBase, *sourceAdjustment)
                             : std::nullopt;
        std::optional<int64_t> nextDest =
            destAdjustment ? checkedSignedAdd(destBase, *destAdjustment)
                           : std::nullopt;
        if (!nextSource || !nextDest ||
            mlir::failed(materializeDescriptor(*nextSource, *nextDest, axes)))
          return mlir::failure();
      }
      return mlir::success();
    }

    llvm::SmallVector<int64_t, 3> sourceStrides({0, 0, 0});
    llvm::SmallVector<int64_t, 3> destStrides({0, 0, 0});
    llvm::SmallVector<int64_t, 3> iterations({1, 1, 1});
    int64_t byteCount = innerBytes;
    for (auto [index, axis] : llvm::enumerate(axes)) {
      sourceStrides[index] = axis.sourceStride;
      destStrides[index] = axis.destStride;
      iterations[index] = axis.count;
      std::optional<int64_t> nextByteCount =
          checkedMulI64(byteCount, axis.count);
      if (!nextByteCount)
        return mlir::failure();
      byteCount = *nextByteCount;
    }
    if (!descriptorFieldFits(sourceBase) || !descriptorFieldFits(destBase) ||
        !descriptorFieldFits(innerBytes) || !descriptorFieldFits(byteCount) ||
        !llvm::all_of(sourceStrides, descriptorFieldFits) ||
        !llvm::all_of(destStrides, descriptorFieldFits) ||
        !llvm::all_of(iterations,
                      [](int64_t value) {
                        return value > 0 && descriptorFieldFits(value);
                      }) ||
        sourceBase > sourceInfo.physicalBytes - elementBytes ||
        destBase > destInfo.physicalBytes - elementBytes)
      return mlir::failure();

    descriptors.push_back(
        {{byteCount, innerBytes, sourceBase, sourceStrides, iterations},
         {byteCount, innerBytes, destBase, destStrides, iterations}});
    std::optional<int64_t> nextCovered =
        checkedSignedAdd(coveredBytes, byteCount);
    if (!nextCovered)
      return mlir::failure();
    coveredBytes = *nextCovered;
    return mlir::success();
  };

  std::function<mlir::LogicalResult(int64_t)> visitChoices;
  visitChoices = [&](int64_t dim) -> mlir::LogicalResult {
    if (dim == static_cast<int64_t>(choices.size())) {
      std::optional<int64_t> sourceBase =
          getPhysicalOffset(*sourceAccess, iterationBase);
      std::optional<int64_t> destBase =
          getPhysicalOffset(*destAccess, iterationBase);
      if (!sourceBase || !destBase)
        return mlir::failure();
      llvm::SmallVector<DescriptorAxis, 8> descriptorAxes;
      descriptorAxes.reserve(symbolicAxes.size());
      for (const SymbolicAxis &axis : symbolicAxes) {
        if (axis.count <= 1)
          continue;
        llvm::SmallVector<int64_t, 4> nextIteration(iterationBase);
        std::optional<int64_t> nextCoordinate =
            checkedSignedAdd(nextIteration[axis.iterationDim], axis.step);
        if (!nextCoordinate)
          return mlir::failure();
        nextIteration[axis.iterationDim] = *nextCoordinate;
        std::optional<int64_t> nextSource =
            getPhysicalOffset(*sourceAccess, nextIteration);
        std::optional<int64_t> nextDest =
            getPhysicalOffset(*destAccess, nextIteration);
        if (!nextSource || !nextDest)
          return mlir::failure();
        descriptorAxes.push_back(
            {*nextSource - *sourceBase, *nextDest - *destBase, axis.count});
      }
      return materializeDescriptor(*sourceBase, *destBase,
                                   std::move(descriptorAxes));
    }

    for (const DimensionChoice &choice : choices[dim]) {
      iterationBase[dim] = choice.base;
      size_t oldAxisCount = symbolicAxes.size();
      symbolicAxes.append(choice.axes.begin(), choice.axes.end());
      if (mlir::failed(visitChoices(dim + 1)))
        return mlir::failure();
      symbolicAxes.resize(oldAxisCount);
    }
    return mlir::success();
  };

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "materialize-descriptors", op->getName().getStringRef());
  if (mlir::failed(visitChoices(0))) {
    std::string typeSummary;
    llvm::raw_string_ostream typeStream(typeSummary);
    typeStream << " after " << descriptors.size()
               << " descriptor(s), covered_bytes=" << coveredBytes
               << ", source=";
    sourceType.print(typeStream);
    typeStream << ", destination=";
    destType.print(typeStream);
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine("movement_descriptor_materialization_limit_exceeded: ")
            .concat(opLabel)
            .concat(" IndexRelation descriptor plan is not target-encodable "
                    "within 4096 commands")
            .concat(typeStream.str())
            .str());
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "verify-coverage", op->getName().getStringRef());
  std::optional<int64_t> expectedBytes =
      checkedMulI64(*elementCount, elementBytes);
  if (!expectedBytes || descriptors.empty() || coveredBytes != *expectedBytes)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op,
        llvm::Twine(opLabel)
            .concat(" IndexRelation descriptor coverage is not exact: ")
            .concat(llvm::Twine(coveredBytes))
            .concat(" covered bytes versus ")
            .concat(expectedBytes ? llvm::Twine(*expectedBytes)
                                  : llvm::Twine("overflow"))
            .str());
  return descriptors;
}

bool MovementDescriptorCache::hasOrProveIdentityPhysicalTraversal(
    mlir::MemRefType sourceType, mlir::MemRefType destType) {
  if (!sourceType || !destType || sourceType.getShape() != destType.getShape())
    return false;
  std::pair<mlir::Type, mlir::Type> key{sourceType, destType};
  if (provenIdentityPhysicalTraversals.contains(key))
    return true;
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      mlir::failed(analysis::TransferRealizability::provePhysicalTraversal(
          sourceType, destType, destType.getShape(), *identity.get(),
          *identity.get())))
    return false;
  provenIdentityPhysicalTraversals.insert(key);
  return true;
}

mlir::FailureOr<SharedMovementDescriptorPlan>
MovementDescriptorCache::getOrCreate(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    llvm::ArrayRef<int64_t> iterationShape,
    const analysis::IndexRelation &iterationToSource,
    const analysis::IndexRelation &iterationToDest, MovementEngine engine,
    llvm::StringRef opLabel) {
  std::optional<mlir::AffineMap> sourceMap;
  std::optional<mlir::AffineMap> destMap;
  const bool hasClosedProjectedRelations =
      iterationToSource.hasTotalBoundedAffineMapConstruction() &&
      iterationToDest.hasTotalBoundedAffineMapConstruction() &&
      (sourceMap =
           iterationToSource.getProjectedAffineMap(rewriter.getContext())) &&
      (destMap = iterationToDest.getProjectedAffineMap(rewriter.getContext()));

  size_t keyHash = 0;
  if (hasClosedProjectedRelations) {
    keyHash = static_cast<size_t>(
        llvm::hash_combine(sourceType, destType, *sourceMap, *destMap,
                           static_cast<unsigned>(engine),
                           llvm::hash_combine_range(iterationShape.begin(),
                                                    iterationShape.end())));
    auto bucket = entriesByHash.find(keyHash);
    if (bucket != entriesByHash.end()) {
      for (unsigned entryIndex : bucket->second) {
        const Entry &entry = entries[entryIndex];
        if (entry.sourceType == sourceType && entry.destType == destType &&
            llvm::ArrayRef<int64_t>(entry.iterationShape) == iterationShape &&
            entry.iterationToSource == *sourceMap &&
            entry.iterationToDest == *destMap && entry.engine == engine)
          return entry.plan;
      }
    }
  }

  mlir::FailureOr<MovementDescriptorPlan> descriptors =
      getRelationMovementDescriptors(rewriter, op, sourceType, destType,
                                     iterationShape, iterationToSource,
                                     iterationToDest, engine, opLabel);
  if (mlir::failed(descriptors))
    return mlir::failure();
  SharedMovementDescriptorPlan plan =
      std::make_shared<MovementDescriptorPlan>(std::move(*descriptors));
  if (hasClosedProjectedRelations) {
    uint8_t &admissionCount = cacheAdmissionCounts[keyHash];
    if (admissionCount < 2) {
      ++admissionCount;
    } else {
      const unsigned entryIndex = entries.size();
      entries.push_back({sourceType, destType,
                         llvm::SmallVector<int64_t, 4>(iterationShape),
                         *sourceMap, *destMap, engine, plan});
      entriesByHash[keyHash].push_back(entryIndex);
    }
  }
  return plan;
}

mlir::LogicalResult emitGatherScatterDescriptorPlan(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *sourceOperation, mlir::Value source, mlir::Value dest,
    llvm::ArrayRef<MovementDescriptorPair> descriptors,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    DDRResourceAttr ddrResource, mlir::Value dynamicSourceOffset,
    mlir::Value dynamicDestOffset) {
  auto sameStructure = [](const MovementDescriptor &lhs,
                          const MovementDescriptor &rhs) {
    return lhs.byteCount == rhs.byteCount && lhs.innerBytes == rhs.innerBytes &&
           lhs.strides == rhs.strides && lhs.iterations == rhs.iterations;
  };

  auto emitOne = [&](const MovementDescriptorPair &descriptor,
                     mlir::Value induction, int64_t sourceStep,
                     int64_t destStep) {
    auto buildDynamicOffset = [&](int64_t staticOffset, int64_t offsetStep,
                                  mlir::Value base) -> mlir::Value {
      if (!base && !induction && offsetStep == 0)
        return {};
      mlir::Value result = base;
      if (!result)
        result = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      if (staticOffset != 0) {
        mlir::Value constant =
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, staticOffset);
        result = rewriter.create<mlir::arith::AddIOp>(loc, result, constant);
      }
      if (induction && offsetStep != 0) {
        mlir::Value constant =
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, offsetStep);
        mlir::Value scaled =
            rewriter.create<mlir::arith::MulIOp>(loc, induction, constant);
        result = rewriter.create<mlir::arith::AddIOp>(loc, result, scaled);
      }
      return result;
    };

    MovementDescriptor sourceDescriptor = descriptor.source;
    MovementDescriptor destDescriptor = descriptor.dest;
    mlir::Value sourceOffset = buildDynamicOffset(
        sourceDescriptor.byteOffset, sourceStep, dynamicSourceOffset);
    mlir::Value destOffset = buildDynamicOffset(destDescriptor.byteOffset,
                                                destStep, dynamicDestOffset);
    if (sourceOffset)
      sourceDescriptor.byteOffset = 0;
    if (destOffset)
      destDescriptor.byteOffset = 0;
    InstrGatherScatterOp lowered =
        createGatherScatter(rewriter, loc, source, dest, sourceDescriptor,
                            destDescriptor, sourceOffset, destOffset);
    if (ddrResource)
      lowered.setDdrResourceAttr(ddrResource);
    if (bufferRecorder && sourceOperation)
      bufferRecorder->recordLoweredOperation(sourceOperation, lowered);
  };

  for (size_t begin = 0; begin < descriptors.size();) {
    size_t count = 1;
    int64_t sourceStep = 0;
    int64_t destStep = 0;
    bool hasAffineVariation = false;
    if (begin + 1 < descriptors.size() &&
        sameStructure(descriptors[begin].source,
                      descriptors[begin + 1].source) &&
        sameStructure(descriptors[begin].dest, descriptors[begin + 1].dest)) {
      if (!llvm::SubOverflow(descriptors[begin + 1].source.byteOffset,
                             descriptors[begin].source.byteOffset,
                             sourceStep) &&
          !llvm::SubOverflow(descriptors[begin + 1].dest.byteOffset,
                             descriptors[begin].dest.byteOffset, destStep)) {
        hasAffineVariation = sourceStep != 0 || destStep != 0;
        count = 2;
        while (begin + count < descriptors.size()) {
          const MovementDescriptorPair &candidate = descriptors[begin + count];
          if (!sameStructure(descriptors[begin].source, candidate.source) ||
              !sameStructure(descriptors[begin].dest, candidate.dest))
            break;
          const int64_t relative = static_cast<int64_t>(count);
          int64_t sourceDelta = 0;
          int64_t destDelta = 0;
          int64_t expectedSource = 0;
          int64_t expectedDest = 0;
          if (llvm::MulOverflow(sourceStep, relative, sourceDelta) ||
              llvm::MulOverflow(destStep, relative, destDelta) ||
              llvm::AddOverflow(descriptors[begin].source.byteOffset,
                                sourceDelta, expectedSource) ||
              llvm::AddOverflow(descriptors[begin].dest.byteOffset, destDelta,
                                expectedDest) ||
              expectedSource != candidate.source.byteOffset ||
              expectedDest != candidate.dest.byteOffset)
            break;
          ++count;
        }
      }
    }

    if (count >= 2 && hasAffineVariation) {
      auto lower = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(count));
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, lower, upper, step);
      if (!loop.getBody()->empty() &&
          mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
        rewriter.eraseOp(&loop.getBody()->back());
      rewriter.setInsertionPointToStart(loop.getBody());
      emitOne(descriptors[begin], loop.getInductionVar(), sourceStep, destStep);
      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointAfter(loop);
      begin += count;
      continue;
    }

    emitOne(descriptors[begin], /*induction=*/{}, /*sourceStep=*/0,
            /*destStep=*/0);
    ++begin;
  }
  return mlir::success();
}

llvm::SmallVector<InstrRDMAOp, 4> createMappedRDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors) {
  llvm::SmallVector<InstrRDMAOp, 4> operations;
  for (const MovementDescriptorPair &descriptor : descriptors)
    operations.push_back(rewriter.create<InstrRDMAOp>(
        loc, source, dest, descriptor.source.byteCount,
        descriptor.source.innerBytes,
        rewriter.getI64IntegerAttr(descriptor.source.byteOffset),
        rewriter.getI64IntegerAttr(descriptor.dest.byteOffset),
        descriptor.source.strides, descriptor.source.iterations));
  return operations;
}

llvm::SmallVector<InstrWDMAOp, 4> createMappedWDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors) {
  llvm::SmallVector<InstrWDMAOp, 4> operations;
  for (const MovementDescriptorPair &descriptor : descriptors)
    operations.push_back(rewriter.create<InstrWDMAOp>(
        loc, source, dest, descriptor.dest.byteCount,
        descriptor.dest.innerBytes,
        rewriter.getI64IntegerAttr(descriptor.source.byteOffset),
        rewriter.getI64IntegerAttr(descriptor.dest.byteOffset),
        descriptor.dest.strides, descriptor.dest.iterations));
  return operations;
}

mlir::IntegerAttr getI64Attr(mlir::PatternRewriter &rewriter, int64_t value) {
  return rewriter.getI64IntegerAttr(value);
}

mlir::FailureOr<int64_t> readRequiredI64Attr(mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return failFailureOr<int64_t>(
        rewriter, op,
        llvm::Twine("batched tile.gemm lowering requires ")
            .concat(name)
            .concat(" attr")
            .str());
  return attr.getInt();
}

mlir::FailureOr<int64_t> getStaticPhysicalBytes(mlir::PatternRewriter &rewriter,
                                                mlir::Operation *op,
                                                mlir::Type type,
                                                llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<int64_t>(
        rewriter, op,
        llvm::Twine(role).concat(" must be a Wafer memref").str());
  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<int64_t>(
        rewriter, op,
        llvm::Twine(role)
            .concat(" requires static positive physical byte size")
            .str());
  return info->physicalBytes;
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
getStaticCompactStrides(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::MemRefType type) {
  llvm::SmallVector<int64_t> strides(type.getRank(), 1);
  int64_t runningStride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = runningStride;
    int64_t size = type.getDimSize(dim);
    if (size == mlir::ShapedType::kDynamic)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, "tile.reshape lowering requires static result shape");
    runningStride *= size;
  }
  return strides;
}

bool isStandardViewCompatibleLayout(MemLayout layout) {
  return layout == MemLayout::Tensor || layout == MemLayout::NTensor;
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
delinearizeIndex(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                 llvm::ArrayRef<int64_t> shape, int64_t linearIndex,
                 llvm::StringRef opLabel) {
  llvm::SmallVector<int64_t> indices(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    int64_t size = shape[dim];
    if (size == mlir::ShapedType::kDynamic || size <= 0)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op,
          llvm::Twine(opLabel).concat(" requires static positive shape").str());
    indices[dim] = linearIndex % size;
    linearIndex /= size;
  }
  if (linearIndex != 0)
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op,
        llvm::Twine(opLabel).concat(" cannot delinearize logical index").str());
  return indices;
}

bool requiresGatherScatterMaterialization(InstrDataMoveKind kind) {
  switch (kind) {
  case InstrDataMoveKind::Mirror:
  case InstrDataMoveKind::Transpose:
  case InstrDataMoveKind::Rotate90:
  case InstrDataMoveKind::Rotate180:
  case InstrDataMoveKind::Rotate270:
  case InstrDataMoveKind::Nchw2Nhwc:
  case InstrDataMoveKind::Nhwc2Nchw:
  case InstrDataMoveKind::TensorNom:
    return true;
  case InstrDataMoveKind::Pad:
  case InstrDataMoveKind::Img2Col:
    return false;
  }
  llvm_unreachable("unknown instr data move kind");
}

mlir::LogicalResult verifyStaticShapeAttrMatchesMemRef(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::MemRefType type,
    mlir::DenseI64ArrayAttr shapeAttr, llvm::StringRef role,
    llvm::StringRef opLabel) {
  if (type.getRank() != static_cast<int64_t>(shapeAttr.size()))
    return failPattern(rewriter, op,
                       llvm::Twine(opLabel)
                           .concat(" requires ")
                           .concat(role)
                           .concat("_shape rank to match the ")
                           .concat(role)
                           .concat(" memref rank")
                           .str());
  for (auto [dim, attrSize] : llvm::enumerate(shapeAttr.asArrayRef())) {
    int64_t memrefSize = type.getDimSize(dim);
    if (memrefSize == mlir::ShapedType::kDynamic || memrefSize != attrSize)
      return failPattern(rewriter, op,
                         llvm::Twine(opLabel)
                             .concat(" requires static ")
                             .concat(role)
                             .concat(" memref shape to match ")
                             .concat(role)
                             .concat("_shape")
                             .str());
  }
  return mlir::success();
}

} // namespace wafer::tile_region_to_instr
