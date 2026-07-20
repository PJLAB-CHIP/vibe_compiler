//===- MovementSupport.cpp - Tile-region movement support ---------------===//

#include "Internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

namespace wafer::tile_region_to_instr {

namespace {

struct PackedMovementDescriptor {
  MovementDescriptor source;
  MovementDescriptor dest;
};

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

namespace {

std::optional<int64_t> checkedAddI64(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::nullopt;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return std::nullopt;
  return lhs + rhs;
}

std::optional<int64_t> checkedAddScaledI64(int64_t base, int64_t stride,
                                           int64_t iteration) {
  std::optional<int64_t> scaled = checkedMulI64(stride, iteration);
  if (!scaled)
    return std::nullopt;
  return checkedAddI64(base, *scaled);
}

std::optional<int64_t>
computeDescriptorPayloadBytes(int64_t innerBytes,
                              llvm::ArrayRef<int64_t> iterations) {
  int64_t total = innerBytes;
  for (int64_t iteration : iterations) {
    std::optional<int64_t> next = checkedMulI64(total, iteration);
    if (!next)
      return std::nullopt;
    total = *next;
  }
  return total;
}

} // namespace

void setFailureReason(std::string *failureReason, llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

mlir::LogicalResult failPattern(mlir::PatternRewriter &rewriter,
                                mlir::Operation *op, std::string *failureReason,
                                llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
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

mlir::FailureOr<mlir::Value> createDestAlloc(mlir::Location loc,
                                             mlir::Type type,
                                             mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             std::string *failureReason) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<mlir::Value>(
        rewriter, op, failureReason,
        "tile-region to instr lowering requires memref result storage");
  return rewriter.create<mlir::memref::AllocOp>(loc, memrefType).getResult();
}

mlir::FailureOr<MovementDescriptor>
getContiguousDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::Type type, std::string *failureReason) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        "instruction descriptor requires a Wafer memref type");

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
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
                           mlir::Type type, std::string *failureReason,
                           llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires a Wafer memref type").str());

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  if (info->bitPackedElement) {
    if (info->compactBytes <= 0 || info->physicalBytes != info->compactBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
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
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  llvm::SmallVector<int64_t> memrefStrides;
  int64_t memrefOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(memrefType, memrefStrides, memrefOffset)) ||
      static_cast<int64_t>(memrefStrides.size()) != memrefType.getRank())
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
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
          rewriter, op, failureReason,
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
            rewriter, op, failureReason,
            llvm::Twine(role).concat(" descriptor inner span overflows").str());
      innerElements = *nextInner;
      continue;
    }

    std::optional<int64_t> strideBytes =
        checkedMulI64(dimStride, info->elementBytes);
    if (!strideBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role).concat(" descriptor stride overflows").str());

    if (!loops.empty()) {
      std::optional<int64_t> collapsedStride =
          checkedMulI64(loops.back().strideBytes, loops.back().iterations);
      if (collapsedStride && *collapsedStride == *strideBytes) {
        std::optional<int64_t> collapsedIterations =
            checkedMulI64(loops.back().iterations, dimSize);
        if (!collapsedIterations)
          return failFailureOr<MovementDescriptor>(
              rewriter, op, failureReason,
              llvm::Twine(role)
                  .concat(" descriptor iteration overflows")
                  .str());
        loops.back().iterations = *collapsedIterations;
        continue;
      }
    }

    if (loops.size() == 3)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role)
              .concat(" requires at most three strided descriptor levels")
              .str());
    loops.push_back({*strideBytes, dimSize});
  }

  std::optional<int64_t> innerBytes =
      checkedMulI64(innerElements, info->elementBytes);
  if (!innerBytes)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
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

void createRDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value source, mlir::Value dest,
                const MovementDescriptor &descriptor) {
  rewriter.create<InstrRDMAOp>(loc, source, dest, descriptor.byteCount,
                               descriptor.innerBytes, descriptor.strides,
                               descriptor.iterations);
}

void createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value source, mlir::Value dest,
                const MovementDescriptor &descriptor) {
  rewriter.create<InstrWDMAOp>(loc, source, dest, descriptor.byteCount,
                               descriptor.innerBytes, descriptor.strides,
                               descriptor.iterations);
}

void createGatherScatter(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value source, mlir::Value dest,
                         const MovementDescriptor &sourceDescriptor,
                         const MovementDescriptor &destDescriptor) {
  mlir::IntegerAttr sourceOffset =
      sourceDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(sourceDescriptor.byteOffset);
  mlir::IntegerAttr destOffset =
      destDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(destDescriptor.byteOffset);
  rewriter.create<InstrGatherScatterOp>(
      loc, source, dest, destDescriptor.byteCount, destDescriptor.innerBytes,
      sourceOffset, destOffset, sourceDescriptor.strides,
      sourceDescriptor.iterations, destDescriptor.strides,
      destDescriptor.iterations);
}

namespace {

int64_t inferPackedIteration(llvm::ArrayRef<LogicalMovementSegment> segments,
                             size_t start, int64_t blockSize,
                             llvm::SmallVectorImpl<int64_t> &sourceStrides,
                             llvm::SmallVectorImpl<int64_t> &destStrides,
                             llvm::SmallVectorImpl<int64_t> &iterations,
                             int64_t dim) {
  if (blockSize <= 0)
    return 1;
  size_t nextBlockStart = start + static_cast<size_t>(blockSize);
  if (nextBlockStart >= segments.size())
    return 1;

  const LogicalMovementSegment &base = segments[start];
  const LogicalMovementSegment &nextBase = segments[nextBlockStart];
  if (nextBase.bytes != base.bytes ||
      nextBase.sourceOffset < base.sourceOffset ||
      nextBase.destOffset < base.destOffset)
    return 1;

  sourceStrides[dim] = nextBase.sourceOffset - base.sourceOffset;
  destStrides[dim] = nextBase.destOffset - base.destOffset;

  int64_t inferred = 1;
  while (true) {
    int64_t repetition = inferred;
    std::optional<int64_t> repeatedBlock = checkedMulI64(blockSize, repetition);
    if (!repeatedBlock)
      break;
    size_t blockStart = start + static_cast<size_t>(*repeatedBlock);
    if (blockStart >= segments.size() ||
        static_cast<size_t>(blockSize) > segments.size() - blockStart)
      break;

    bool matches = true;
    for (int64_t withinBlock = 0; withinBlock < blockSize; ++withinBlock) {
      const LogicalMovementSegment &first =
          segments[start + static_cast<size_t>(withinBlock)];
      const LogicalMovementSegment &candidate =
          segments[blockStart + static_cast<size_t>(withinBlock)];
      std::optional<int64_t> expectedSource = checkedAddScaledI64(
          first.sourceOffset, sourceStrides[dim], repetition);
      std::optional<int64_t> expectedDest =
          checkedAddScaledI64(first.destOffset, destStrides[dim], repetition);
      if (!expectedSource || !expectedDest || candidate.bytes != first.bytes ||
          candidate.sourceOffset != *expectedSource ||
          candidate.destOffset != *expectedDest) {
        matches = false;
        break;
      }
    }
    if (!matches)
      break;
    iterations[dim] = inferred + 1;
    ++inferred;
  }
  if (inferred == 1) {
    sourceStrides[dim] = 0;
    destStrides[dim] = 0;
  }
  return inferred;
}

PackedMovementDescriptor
packMovementDescriptor(llvm::ArrayRef<LogicalMovementSegment> segments,
                       size_t start) {
  const LogicalMovementSegment &base = segments[start];
  llvm::SmallVector<int64_t, 3> sourceStrides({0, 0, 0});
  llvm::SmallVector<int64_t, 3> destStrides({0, 0, 0});
  llvm::SmallVector<int64_t, 3> iterations({1, 1, 1});

  inferPackedIteration(segments, start, /*blockSize=*/1, sourceStrides,
                       destStrides, iterations, /*dim=*/0);
  std::optional<int64_t> dim1Block = computeDescriptorPayloadBytes(
      /*innerBytes=*/1, llvm::ArrayRef<int64_t>(iterations).take_front(1));
  if (dim1Block)
    inferPackedIteration(segments, start, *dim1Block, sourceStrides,
                         destStrides, iterations, /*dim=*/1);
  std::optional<int64_t> dim2Block = computeDescriptorPayloadBytes(
      /*innerBytes=*/1, llvm::ArrayRef<int64_t>(iterations).take_front(2));
  if (dim2Block)
    inferPackedIteration(segments, start, *dim2Block, sourceStrides,
                         destStrides, iterations, /*dim=*/2);

  std::optional<int64_t> byteCount =
      computeDescriptorPayloadBytes(base.bytes, iterations);
  if (!byteCount) {
    sourceStrides.assign({0, 0, 0});
    destStrides.assign({0, 0, 0});
    iterations.assign({1, 1, 1});
    byteCount = base.bytes;
  }

  MovementDescriptor sourceDescriptor;
  sourceDescriptor.byteCount = *byteCount;
  sourceDescriptor.innerBytes = base.bytes;
  sourceDescriptor.byteOffset = base.sourceOffset;
  sourceDescriptor.strides = sourceStrides;
  sourceDescriptor.iterations = iterations;

  MovementDescriptor destDescriptor;
  destDescriptor.byteCount = *byteCount;
  destDescriptor.innerBytes = base.bytes;
  destDescriptor.byteOffset = base.destOffset;
  destDescriptor.strides = destStrides;
  destDescriptor.iterations = iterations;

  return {sourceDescriptor, destDescriptor};
}

} // namespace

void createGatherScatterSegments(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<LogicalMovementSegment> segments,
    bool mayReorderDisjointSegments) {
  auto countCommands = [](llvm::ArrayRef<LogicalMovementSegment> ordered) {
    uint64_t count = 0;
    for (size_t index = 0; index < ordered.size();) {
      PackedMovementDescriptor packed = packMovementDescriptor(ordered, index);
      std::optional<int64_t> descriptorSegments = computeDescriptorPayloadBytes(
          /*innerBytes=*/1, packed.source.iterations);
      if (!descriptorSegments || *descriptorSegments <= 0)
        descriptorSegments = 1;
      index += static_cast<size_t>(*descriptorSegments);
      ++count;
    }
    return count;
  };

  llvm::SmallVector<LogicalMovementSegment> sourceOrdered;
  llvm::SmallVector<LogicalMovementSegment> destOrdered;
  llvm::ArrayRef<LogicalMovementSegment> selected = segments;
  uint64_t selectedCount = countCommands(selected);
  if (mayReorderDisjointSegments && segments.size() > 1) {
    sourceOrdered.assign(segments.begin(), segments.end());
    llvm::sort(sourceOrdered, [](const LogicalMovementSegment &lhs,
                                 const LogicalMovementSegment &rhs) {
      if (lhs.sourceOffset != rhs.sourceOffset)
        return lhs.sourceOffset < rhs.sourceOffset;
      if (lhs.destOffset != rhs.destOffset)
        return lhs.destOffset < rhs.destOffset;
      return lhs.bytes < rhs.bytes;
    });
    uint64_t sourceCount = countCommands(sourceOrdered);
    if (sourceCount < selectedCount) {
      selected = sourceOrdered;
      selectedCount = sourceCount;
    }

    destOrdered.assign(segments.begin(), segments.end());
    llvm::sort(destOrdered, [](const LogicalMovementSegment &lhs,
                               const LogicalMovementSegment &rhs) {
      if (lhs.destOffset != rhs.destOffset)
        return lhs.destOffset < rhs.destOffset;
      if (lhs.sourceOffset != rhs.sourceOffset)
        return lhs.sourceOffset < rhs.sourceOffset;
      return lhs.bytes < rhs.bytes;
    });
    uint64_t destCount = countCommands(destOrdered);
    if (destCount < selectedCount)
      selected = destOrdered;
  }

  for (size_t index = 0; index < selected.size();) {
    PackedMovementDescriptor packed = packMovementDescriptor(selected, index);
    createGatherScatter(rewriter, loc, source, dest, packed.source,
                        packed.dest);

    std::optional<int64_t> descriptorSegments = computeDescriptorPayloadBytes(
        /*innerBytes=*/1, packed.source.iterations);
    if (!descriptorSegments || *descriptorSegments <= 0)
      descriptorSegments = 1;
    index += static_cast<size_t>(*descriptorSegments);
  }
}

mlir::FailureOr<uint64_t> preflightPackedMovementCommands(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::ArrayRef<LogicalMovementSegment> segments, std::string *failureReason,
    llvm::StringRef opLabel) {
  if (segments.empty())
    return failFailureOr<uint64_t>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" produced no movement segments").str());

  auto fitsTargetField = [](int64_t value) {
    return value >= 0 &&
           static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
  };
  auto descriptorFitsTarget = [&](const MovementDescriptor &descriptor) {
    if (descriptor.byteCount <= 0 || descriptor.innerBytes <= 0 ||
        !fitsTargetField(descriptor.byteCount) ||
        !fitsTargetField(descriptor.innerBytes) ||
        !fitsTargetField(descriptor.byteOffset))
      return false;
    return llvm::all_of(descriptor.strides, fitsTargetField) &&
           llvm::all_of(descriptor.iterations, [&](int64_t iteration) {
             return iteration > 0 && fitsTargetField(iteration);
           });
  };

  uint64_t commandCount = 0;
  for (size_t index = 0; index < segments.size();) {
    PackedMovementDescriptor packed = packMovementDescriptor(segments, index);
    if (!descriptorFitsTarget(packed.source) ||
        !descriptorFitsTarget(packed.dest))
      return failFailureOr<uint64_t>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" descriptor exceeds uint32 target fields")
              .str());

    std::optional<int64_t> descriptorSegments = computeDescriptorPayloadBytes(
        /*innerBytes=*/1, packed.source.iterations);
    if (!descriptorSegments || *descriptorSegments <= 0)
      descriptorSegments = 1;
    index += static_cast<size_t>(*descriptorSegments);
    if (commandCount == std::numeric_limits<uint64_t>::max())
      return failFailureOr<uint64_t>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" command count overflows").str());
    ++commandCount;
  }
  return commandCount;
}

void copyOptionalAttr(mlir::Operation *from, mlir::Operation *to,
                      llvm::StringRef name) {
  if (mlir::Attribute attr = from->getAttr(name))
    to->setAttr(name, attr);
}

mlir::IntegerAttr getI64Attr(mlir::PatternRewriter &rewriter, int64_t value) {
  return rewriter.getI64IntegerAttr(value);
}

mlir::FailureOr<int64_t> readRequiredI64Attr(mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             llvm::StringRef name,
                                             std::string *failureReason) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine("batched tile.gemm lowering requires ")
            .concat(name)
            .concat(" attr")
            .str());
  return attr.getInt();
}

mlir::FailureOr<int64_t> getStaticPhysicalBytes(mlir::PatternRewriter &rewriter,
                                                mlir::Operation *op,
                                                mlir::Type type,
                                                std::string *failureReason,
                                                llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" must be a Wafer memref").str());
  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static positive physical byte size")
            .str());
  return info->physicalBytes;
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
getStaticCompactStrides(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::MemRefType type, std::string *failureReason) {
  llvm::SmallVector<int64_t> strides(type.getRank(), 1);
  int64_t runningStride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = runningStride;
    int64_t size = type.getDimSize(dim);
    if (size == mlir::ShapedType::kDynamic)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          "tile.reshape lowering requires static result shape");
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
                 std::string *failureReason, llvm::StringRef opLabel) {
  llvm::SmallVector<int64_t> indices(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    int64_t size = shape[dim];
    if (size == mlir::ShapedType::kDynamic || size <= 0)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" requires static positive shape").str());
    indices[dim] = linearIndex % size;
    linearIndex /= size;
  }
  if (linearIndex != 0)
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" cannot delinearize logical index").str());
  return indices;
}

mlir::FailureOr<llvm::SmallVector<int64_t>> expandRankReducedSliceIndices(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> reducedShape,
    llvm::ArrayRef<int64_t> reducedIndices, std::string *failureReason,
    llvm::StringRef opLabel) {
  std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
      mlir::computeRankReductionMask(fullShape, reducedShape);
  if (!rankReductionMask)
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" cannot map rank-reduced slice").str());
  if (reducedShape.size() != reducedIndices.size())
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" has mismatched reduced indices").str());

  llvm::SmallVector<int64_t> fullIndices(fullShape.size(), 0);
  size_t reducedDim = 0;
  for (unsigned fullDim = 0; fullDim < fullShape.size(); ++fullDim) {
    if (rankReductionMask->contains(fullDim)) {
      fullIndices[fullDim] = 0;
      continue;
    }
    if (reducedDim >= reducedIndices.size())
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" has incomplete slice index").str());
    fullIndices[fullDim] = reducedIndices[reducedDim++];
  }
  if (reducedDim != reducedIndices.size())
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" has unused reduced slice index").str());
  return fullIndices;
}

mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getStaticLogicalMovementSegments(mlir::PatternRewriter &rewriter,
                                 mlir::Operation *op,
                                 mlir::MemRefType sourceType,
                                 mlir::MemRefType destType,
                                 std::string *failureReason,
                                 llvm::StringRef opLabel) {
  std::optional<WaferPhysicalTensorInfo> sourceInfoStorage =
      computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfoStorage =
      computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfoStorage || !destInfoStorage)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive byte sizes")
            .str());
  const WaferPhysicalTensorInfo &sourceInfo = *sourceInfoStorage;
  const WaferPhysicalTensorInfo &destInfo = *destInfoStorage;
  if (sourceInfo.compactBytes <= 0 || destInfo.compactBytes <= 0 ||
      sourceInfo.physicalBytes <= 0 || destInfo.physicalBytes <= 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive byte sizes")
            .str());
  if (sourceInfo.compactBytes != destInfo.compactBytes)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires equal static compact byte counts")
            .str());
  if (sourceInfo.elementBytes <= 0 ||
      sourceInfo.elementBytes != destInfo.elementBytes ||
      sourceInfo.bitPackedElement || destInfo.bitPackedElement ||
      sourceInfo.compactBytes % sourceInfo.elementBytes != 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements")
            .str());

  // Equal physical geometry can be copied as one segment, including layout
  // padding.  A reshape between compact Tensor/NTensor layouts also preserves
  // physical linear order and therefore needs no per-element enumeration.
  if (sourceInfo.layout == destInfo.layout &&
      sourceType.getShape() == destType.getShape() &&
      sourceType.getElementType() == destType.getElementType() &&
      sourceInfo.physicalBytes == destInfo.physicalBytes)
    return llvm::SmallVector<LogicalMovementSegment>{
        {0, 0, sourceInfo.physicalBytes}};
  if (isStandardViewCompatibleLayout(sourceInfo.layout) &&
      isStandardViewCompatibleLayout(destInfo.layout) &&
      sourceInfo.physicalBytes == sourceInfo.compactBytes &&
      destInfo.physicalBytes == destInfo.compactBytes)
    return llvm::SmallVector<LogicalMovementSegment>{
        {0, 0, sourceInfo.compactBytes}};

  std::optional<StaticPhysicalOffsetCalculator> sourceOffsets =
      StaticPhysicalOffsetCalculator::create(sourceType);
  std::optional<StaticPhysicalOffsetCalculator> destOffsets =
      StaticPhysicalOffsetCalculator::create(destType);
  if (!sourceOffsets || !destOffsets)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive byte sizes")
            .str());

  int64_t elementBytes = sourceInfo.elementBytes;
  int64_t elementCount = sourceInfo.compactBytes / elementBytes;

  // Reshape preserves canonical logical linear order, not per-dimension index
  // equality.  Traverse the source and destination canonical shapes with two
  // incremental odometers, then ask the physical layout helper where the same
  // logical element lives in each buffer.
  llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
  llvm::ArrayRef<int64_t> destShape = destType.getShape();
  llvm::SmallVector<int64_t, 4> sourceIndices(sourceShape.size(), 0);
  llvm::SmallVector<int64_t, 4> destIndices(destShape.size(), 0);
  llvm::SmallVector<LogicalMovementSegment> segments;
  for (int64_t linearIndex = 0; linearIndex < elementCount; ++linearIndex) {
    int64_t sourceOffset =
        sourceOffsets->getByteOffsetForValidIndices(sourceIndices);
    int64_t destOffset = destOffsets->getByteOffsetForValidIndices(destIndices);
    if (sourceOffset < 0 || destOffset < 0 ||
        sourceOffset > sourceInfo.physicalBytes - elementBytes ||
        destOffset > destInfo.physicalBytes - elementBytes)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" segment exceeds static physical byte range")
              .str());

    bool coalesced = false;
    if (!segments.empty()) {
      LogicalMovementSegment &last = segments.back();
      if (last.sourceOffset + last.bytes == sourceOffset &&
          last.destOffset + last.bytes == destOffset) {
        last.bytes += elementBytes;
        coalesced = true;
      }
    }
    if (!coalesced)
      segments.push_back({sourceOffset, destOffset, elementBytes});

    if (linearIndex + 1 != elementCount) {
      for (int64_t dim = static_cast<int64_t>(sourceShape.size()) - 1; dim >= 0;
           --dim) {
        if (++sourceIndices[dim] < sourceShape[dim])
          break;
        sourceIndices[dim] = 0;
      }
      for (int64_t dim = static_cast<int64_t>(destShape.size()) - 1; dim >= 0;
           --dim) {
        if (++destIndices[dim] < destShape[dim])
          break;
        destIndices[dim] = 0;
      }
    }
  }

  return segments;
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
    std::string *failureReason, llvm::StringRef opLabel) {
  if (type.getRank() != static_cast<int64_t>(shapeAttr.size()))
    return failPattern(rewriter, op, failureReason,
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
      return failPattern(rewriter, op, failureReason,
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

mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getPermutationDataMoveSegments(mlir::PatternRewriter &rewriter,
                               InstrTDMADataMoveOp op,
                               mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> permutation,
                               std::string *failureReason,
                               llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank() ||
      sourceType.getRank() != static_cast<int64_t>(permutation.size()))
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  for (auto [destDim, sourceDim] : llvm::enumerate(permutation)) {
    int64_t sourceSize = sourceType.getDimSize(sourceDim);
    int64_t destSize = destType.getDimSize(destDim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic || sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires permutation-compatible static shapes")
              .str());
  }

  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices,
                           llvm::SmallVectorImpl<int64_t> &sourceIndices) {
    sourceIndices.resize(sourceType.getRank(), 0);
    for (auto [destDim, sourceDim] : llvm::enumerate(permutation))
      sourceIndices[sourceDim] = destIndices[destDim];
    return mlir::success();
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices,
                        llvm::SmallVectorImpl<int64_t> &result) {
    result.assign(destIndices.begin(), destIndices.end());
    return mlir::success();
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
}

mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getMirrorDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank())
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t sourceSize = sourceType.getDimSize(dim);
    int64_t destSize = destType.getDimSize(dim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic || sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires equal static source/dest shapes")
              .str());
  }

  llvm::SmallDenseSet<int64_t, 4> mirroredAxes;
  mirroredAxes.insert(axes.begin(), axes.end());
  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices,
                           llvm::SmallVectorImpl<int64_t> &sourceIndices) {
    sourceIndices.assign(destIndices.begin(), destIndices.end());
    for (int64_t axis : mirroredAxes)
      sourceIndices[axis] = sourceType.getDimSize(axis) - 1 - destIndices[axis];
    return mlir::success();
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices,
                        llvm::SmallVectorImpl<int64_t> &result) {
    result.assign(destIndices.begin(), destIndices.end());
    return mlir::success();
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
}

mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getRotateDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType, InstrDataMoveKind kind,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank() || axes.size() != 2)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  int64_t axis0 = axes[0];
  int64_t axis1 = axes[1];
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t sourceSize = sourceType.getDimSize(dim);
    int64_t destSize = destType.getDimSize(dim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" requires static shapes").str());
    if (dim != axis0 && dim != axis1 && sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires non-rotated dimensions to match")
              .str());
  }

  int64_t sourceAxis0Size = sourceType.getDimSize(axis0);
  int64_t sourceAxis1Size = sourceType.getDimSize(axis1);
  int64_t destAxis0Size = destType.getDimSize(axis0);
  int64_t destAxis1Size = destType.getDimSize(axis1);
  if (kind == InstrDataMoveKind::Rotate180) {
    if (sourceAxis0Size != destAxis0Size || sourceAxis1Size != destAxis1Size)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires equal rotated-axis shapes")
              .str());
  } else if (sourceAxis0Size != destAxis1Size ||
             sourceAxis1Size != destAxis0Size) {
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires swapped rotated-axis shapes")
            .str());
  }

  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices,
                           llvm::SmallVectorImpl<int64_t> &sourceIndices) {
    sourceIndices.assign(destIndices.begin(), destIndices.end());
    switch (kind) {
    case InstrDataMoveKind::Rotate90:
      sourceIndices[axis0] = sourceAxis0Size - 1 - destIndices[axis1];
      sourceIndices[axis1] = destIndices[axis0];
      break;
    case InstrDataMoveKind::Rotate180:
      sourceIndices[axis0] = sourceAxis0Size - 1 - destIndices[axis0];
      sourceIndices[axis1] = sourceAxis1Size - 1 - destIndices[axis1];
      break;
    case InstrDataMoveKind::Rotate270:
      sourceIndices[axis0] = destIndices[axis1];
      sourceIndices[axis1] = sourceAxis1Size - 1 - destIndices[axis0];
      break;
    default:
      llvm_unreachable("expected rotate data_move kind");
    }
    return mlir::success();
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices,
                        llvm::SmallVectorImpl<int64_t> &result) {
    result.assign(destIndices.begin(), destIndices.end());
    return mlir::success();
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
}

} // namespace wafer::tile_region_to_instr
