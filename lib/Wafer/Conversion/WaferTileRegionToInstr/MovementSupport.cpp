//===- MovementSupport.cpp - Tile-region movement support ---------------===//

#include "Internal.h"

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>

namespace wafer::tile_region_to_instr {

namespace {

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

namespace {} // namespace

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
                               descriptor.innerBytes,
                               /*src_offset=*/mlir::IntegerAttr{},
                               /*dst_offset=*/mlir::IntegerAttr{},
                               descriptor.strides, descriptor.iterations);
}

void createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value source, mlir::Value dest,
                const MovementDescriptor &descriptor) {
  rewriter.create<InstrWDMAOp>(loc, source, dest, descriptor.byteCount,
                               descriptor.innerBytes,
                               /*src_offset=*/mlir::IntegerAttr{},
                               /*dst_offset=*/mlir::IntegerAttr{},
                               descriptor.strides, descriptor.iterations);
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

struct DirectChannelDecomposition {
  bool requested = false;
  bool blockedByGeneralMap = false;
  int64_t block = 0;
};

bool isBlockedLayout(MemLayout layout) {
  return layout == MemLayout::Cx || layout == MemLayout::NCx;
}

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

  llvm::SmallVector<int64_t, 4> zero(iterationShape.size(), 0);
  llvm::SmallVector<int64_t, 4> offsets = map->compose(zero);
  llvm::SmallVector<ProjectedCoordinate, 4> coordinates;
  coordinates.reserve(offsets.size());
  for (auto [result, offset] : llvm::enumerate(offsets)) {
    ProjectedCoordinate coordinate;
    coordinate.offset = offset;
    coordinate.multipliers.resize(iterationShape.size(), 0);
    unsigned nonzeroMultipliers = 0;
    for (unsigned dim = 0; dim < iterationShape.size(); ++dim) {
      llvm::SmallVector<int64_t, 4> unit(zero);
      unit[dim] = 1;
      llvm::SmallVector<int64_t, 4> value = map->compose(unit);
      std::optional<int64_t> multiplier =
          checkedSignedAdd(value[result], -offset);
      if (!multiplier)
        return std::nullopt;
      if (*multiplier == 0)
        continue;
      coordinate.multipliers[dim] = *multiplier;
      ++nonzeroMultipliers;
      if (nonzeroMultipliers == 1) {
        coordinate.iterationDim = dim;
        coordinate.multiplier = *multiplier;
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
    int64_t endpointSize = endpointType.getDimSize(result);
    if (endpointSize <= 0 || minimum < 0 || maximum >= endpointSize)
      return std::nullopt;
    coordinates.push_back(coordinate);
  }

  MemoryAttr memory = getWaferMemoryAttr(endpointType);
  if (memory && isBlockedLayout(memory.getLayout())) {
    llvm::SmallVector<int64_t, 4> owner(iterationShape.size(), -1);
    for (auto [coordinateIndex, coordinate] : llvm::enumerate(coordinates)) {
      for (auto [dim, multiplier] : llvm::enumerate(coordinate.multipliers)) {
        if (multiplier == 0)
          continue;
        if (owner[dim] >= 0)
          return std::nullopt;
        owner[dim] = coordinateIndex;
      }
    }
  }
  return coordinates;
}

std::optional<llvm::SmallVector<int64_t, 4>>
evaluateProjectedCoordinates(llvm::ArrayRef<ProjectedCoordinate> coordinates,
                             llvm::ArrayRef<int64_t> iteration) {
  llvm::SmallVector<int64_t, 4> result;
  result.reserve(coordinates.size());
  for (const ProjectedCoordinate &coordinate : coordinates) {
    if (coordinate.multipliers.size() != iteration.size())
      return std::nullopt;
    int64_t value = coordinate.offset;
    for (auto [index, multiplier] : llvm::enumerate(coordinate.multipliers)) {
      std::optional<int64_t> scaled =
          checkedSignedMul(multiplier, iteration[index]);
      std::optional<int64_t> next =
          scaled ? checkedSignedAdd(value, *scaled) : std::nullopt;
      if (!next)
        return std::nullopt;
      value = *next;
    }
    result.push_back(value);
  }
  return result;
}

bool isCanonicalBlockedChannelDecomposition(
    const ProjectedCoordinate &coordinate,
    llvm::ArrayRef<int64_t> iterationShape, int64_t logicalChannels,
    int64_t channelBlock) {
  if (coordinate.offset != 0 || logicalChannels <= 0 || channelBlock <= 0 ||
      logicalChannels % channelBlock != 0)
    return false;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> factors;
  for (auto [dim, multiplier] : llvm::enumerate(coordinate.multipliers)) {
    if (multiplier == 0)
      continue;
    if (multiplier < 0 || iterationShape[dim] <= 0)
      return false;
    factors.push_back({multiplier, iterationShape[dim]});
  }
  llvm::sort(factors);
  int64_t expectedMultiplier = 1;
  bool hasBlockBoundary = false;
  for (auto [multiplier, extent] : factors) {
    if (multiplier != expectedMultiplier)
      return false;
    std::optional<int64_t> next = checkedSignedMul(expectedMultiplier, extent);
    if (!next)
      return false;
    expectedMultiplier = *next;
    hasBlockBoundary |= expectedMultiplier == channelBlock;
  }
  return hasBlockBoundary && expectedMultiplier == logicalChannels;
}

bool appendBlockedCategoryBoundaries(
    const ProjectedCoordinate &coordinate, int64_t iterationSize,
    const WaferPhysicalTensorInfo &info, int64_t logicalChannels,
    llvm::SmallVectorImpl<int64_t> &boundaries) {
  if (coordinate.iterationDim < 0 || coordinate.multiplier == 0)
    return true;
  std::optional<int64_t> fullChannels =
      checkedSignedMul(info.cxBlocks, info.cBlock);
  if (!fullChannels || info.cBlock <= 0)
    return false;

  int64_t cursor = 0;
  while (cursor < iterationSize) {
    std::optional<int64_t> scaled =
        checkedSignedMul(coordinate.multiplier, cursor);
    std::optional<int64_t> channel =
        scaled ? checkedSignedAdd(coordinate.offset, *scaled) : std::nullopt;
    if (!channel || *channel < 0 || *channel >= logicalChannels)
      return false;

    int64_t categoryLow = 0;
    int64_t categoryHigh = logicalChannels - 1;
    if (*channel < *fullChannels) {
      categoryLow = (*channel / info.cBlock) * info.cBlock;
      categoryHigh =
          std::min(logicalChannels - 1, categoryLow + info.cBlock - 1);
    } else {
      categoryLow = *fullChannels;
    }

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

std::optional<CanonicalReshapeMovementRelations>
getCanonicalReshapeMovementRelations(mlir::MLIRContext *context,
                                     llvm::ArrayRef<int64_t> sourceShape,
                                     llvm::ArrayRef<int64_t> destShape) {
  if (!context)
    return std::nullopt;

  auto getPrefixProducts = [](llvm::ArrayRef<int64_t> shape)
      -> std::optional<llvm::SmallVector<int64_t, 4>> {
    llvm::SmallVector<int64_t, 4> prefixes{1};
    int64_t product = 1;
    for (int64_t extent : shape) {
      if (extent <= 0)
        return std::nullopt;
      std::optional<int64_t> next = checkedMulI64(product, extent);
      if (!next)
        return std::nullopt;
      product = *next;
      prefixes.push_back(product);
    }
    return prefixes;
  };
  std::optional<llvm::SmallVector<int64_t, 4>> sourcePrefixes =
      getPrefixProducts(sourceShape);
  std::optional<llvm::SmallVector<int64_t, 4>> destPrefixes =
      getPrefixProducts(destShape);
  if (!sourcePrefixes || !destPrefixes ||
      sourcePrefixes->back() != destPrefixes->back())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> boundaries(sourcePrefixes->begin(),
                                           sourcePrefixes->end());
  boundaries.append(destPrefixes->begin(), destPrefixes->end());
  llvm::sort(boundaries);
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  llvm::SmallVector<int64_t, 4> iterationShape;
  iterationShape.reserve(boundaries.size() - 1);
  for (size_t index = 1; index < boundaries.size(); ++index) {
    if (boundaries[index - 1] <= 0 ||
        boundaries[index] % boundaries[index - 1] != 0)
      return std::nullopt;
    iterationShape.push_back(boundaries[index] / boundaries[index - 1]);
  }

  auto buildMap =
      [&](llvm::ArrayRef<int64_t> shape,
          llvm::ArrayRef<int64_t> prefixes) -> std::optional<mlir::AffineMap> {
    llvm::SmallVector<mlir::AffineExpr, 4> results;
    results.reserve(shape.size());
    for (size_t logicalDim = 0; logicalDim < shape.size(); ++logicalDim) {
      if (shape[logicalDim] == 1) {
        results.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      auto beginIt = llvm::find(boundaries, prefixes[logicalDim]);
      auto endIt = llvm::find(boundaries, prefixes[logicalDim + 1]);
      if (beginIt == boundaries.end() || endIt == boundaries.end() ||
          beginIt >= endIt)
        return std::nullopt;
      size_t begin = std::distance(boundaries.begin(), beginIt);
      size_t end = std::distance(boundaries.begin(), endIt);
      mlir::AffineExpr expression = mlir::getAffineConstantExpr(0, context);
      for (size_t axis = begin; axis < end; ++axis)
        expression = expression * iterationShape[axis] +
                     mlir::getAffineDimExpr(axis, context);
      results.push_back(expression);
    }
    return mlir::AffineMap::get(iterationShape.size(), 0, results, context);
  };

  std::optional<mlir::AffineMap> sourceMap =
      buildMap(sourceShape, *sourcePrefixes);
  std::optional<mlir::AffineMap> destMap = buildMap(destShape, *destPrefixes);
  if (!sourceMap || !destMap)
    return std::nullopt;
  analysis::IndexRelationResult sourceRelation =
      analysis::IndexRelation::fromAffineMap(*sourceMap, iterationShape,
                                             sourceShape);
  analysis::IndexRelationResult destRelation =
      analysis::IndexRelation::fromAffineMap(*destMap, iterationShape,
                                             destShape);
  if (!sourceRelation.isExact() || !destRelation.isExact())
    return std::nullopt;
  return CanonicalReshapeMovementRelations{std::move(iterationShape),
                                           std::move(*sourceRelation.relation),
                                           std::move(*destRelation.relation)};
}

mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
getRelationMovementDescriptors(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> iterationShape,
                               const analysis::IndexRelation &iterationToSource,
                               const analysis::IndexRelation &iterationToDest,
                               MovementEngine engine,
                               std::string *failureReason,
                               llvm::StringRef opLabel) {
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
        rewriter, op, failureReason,
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
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements in static physical "
                    "layouts")
            .str());
  const WaferPhysicalTensorInfo &sourceInfo = *sourceInfoStorage;
  const WaferPhysicalTensorInfo &destInfo = *destInfoStorage;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "TransferRealizability::proveMappedTransfer",
      op->getName().getStringRef());
  if (mlir::failed(analysis::TransferRealizability::proveMappedTransfer(
          sourceType, destType, iterationShape, iterationToSource,
          iterationToDest)))
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op, failureReason,
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
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires a static projected-affine IndexRelation")
            .str());

  std::optional<StaticPhysicalOffsetCalculator> sourceOffsets =
      StaticPhysicalOffsetCalculator::create(sourceType);
  std::optional<StaticPhysicalOffsetCalculator> destOffsets =
      StaticPhysicalOffsetCalculator::create(destType);
  if (!sourceOffsets || !destOffsets)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" cannot construct static physical offset calculators")
            .str());

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "partition-symbolic-domain", op->getName().getStringRef());
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> boundaries(
      iterationShape.size());
  llvm::SmallVector<DirectChannelDecomposition, 4> decompositions(
      iterationShape.size());
  for (auto [dim, size] : llvm::enumerate(iterationShape)) {
    if (size <= 0)
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires a static positive iteration domain")
              .str());
    boundaries[dim].push_back(0);
    boundaries[dim].push_back(size);
  }

  auto collectBlockedConstraint =
      [&](mlir::MemRefType type, const WaferPhysicalTensorInfo &info,
          llvm::ArrayRef<ProjectedCoordinate> map) -> bool {
    if (!isBlockedLayout(info.layout))
      return true;
    if (type.getRank() <= 0 || info.cBlock <= 0)
      return false;
    const ProjectedCoordinate &channel = map.back();
    if (channel.iterationDim == -1)
      return true;
    if (channel.iterationDim == -2)
      return isCanonicalBlockedChannelDecomposition(
          channel, iterationShape, type.getShape().back(), info.cBlock);
    int64_t iterationDim = channel.iterationDim;
    if (!appendBlockedCategoryBoundaries(channel, iterationShape[iterationDim],
                                         info, type.getShape().back(),
                                         boundaries[iterationDim]))
      return false;
    bool direct = channel.multiplier == 1 && channel.offset == 0 &&
                  iterationShape[iterationDim] == type.getShape().back();
    DirectChannelDecomposition &decomposition = decompositions[iterationDim];
    if (!direct) {
      decomposition.blockedByGeneralMap = true;
      return true;
    }
    if (decomposition.requested && decomposition.block != info.cBlock)
      return false;
    decomposition.requested = true;
    decomposition.block = info.cBlock;
    return true;
  };
  if (!collectBlockedConstraint(sourceType, sourceInfo, *sourceMap) ||
      !collectBlockedConstraint(destType, destInfo, *destMap))
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" cannot partition blocked-layout channel pieces")
            .str());

  llvm::SmallVector<llvm::SmallVector<DimensionChoice, 4>, 4> choices(
      iterationShape.size());
  for (int64_t dim = 0; dim < static_cast<int64_t>(iterationShape.size());
       ++dim) {
    int64_t size = iterationShape[dim];
    const DirectChannelDecomposition &decomposition = decompositions[dim];
    if (decomposition.requested && !decomposition.blockedByGeneralMap) {
      int64_t fullBlocks = size / decomposition.block;
      int64_t remainder = size % decomposition.block;
      if (fullBlocks > 0) {
        DimensionChoice full;
        full.axes.push_back({dim, 1, decomposition.block});
        full.axes.push_back({dim, decomposition.block, fullBlocks});
        choices[dim].push_back(std::move(full));
      }
      if (remainder > 0) {
        DimensionChoice tail;
        tail.base = fullBlocks * decomposition.block;
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
          rewriter, op, failureReason,
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
      [&](const StaticPhysicalOffsetCalculator &calculator,
          llvm::ArrayRef<ProjectedCoordinate> map,
          llvm::ArrayRef<int64_t> iteration) -> std::optional<int64_t> {
    std::optional<llvm::SmallVector<int64_t, 4>> logical =
        evaluateProjectedCoordinates(map, iteration);
    return logical ? calculator.getByteOffset(*logical) : std::nullopt;
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
        if (descriptors.size() >= detail::kStaticTerminalOperationBudget)
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

    if (engine != MovementEngine::GatherScatter) {
      int64_t expectedStride = elementBytes;
      for (size_t index = 0; index < axes.size(); ++index) {
        if (sequentialEndpointStride(axes[index]) != expectedStride) {
          DescriptorAxis split = axes[index];
          llvm::SmallVector<DescriptorAxis, 8> remaining = axes;
          remaining.erase(remaining.begin() + index);
          for (int64_t iteration = 0; iteration < split.count; ++iteration) {
            if (descriptors.size() >= detail::kStaticTerminalOperationBudget)
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
        if (descriptors.size() >= detail::kStaticTerminalOperationBudget)
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
          getPhysicalOffset(*sourceOffsets, *sourceMap, iterationBase);
      std::optional<int64_t> destBase =
          getPhysicalOffset(*destOffsets, *destMap, iterationBase);
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
            getPhysicalOffset(*sourceOffsets, *sourceMap, nextIteration);
        std::optional<int64_t> nextDest =
            getPhysicalOffset(*destOffsets, *destMap, nextIteration);
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
  if (mlir::failed(visitChoices(0)))
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op, failureReason,
        llvm::Twine("static_terminal_budget_exceeded: ")
            .concat(opLabel)
            .concat(" IndexRelation descriptor plan is not target-encodable "
                    "within 4096 commands")
            .str());

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "lowering-algorithm-phase", "getRelationMovementDescriptors",
      "verify-coverage", op->getName().getStringRef());
  std::optional<int64_t> expectedBytes =
      checkedMulI64(*elementCount, elementBytes);
  if (!expectedBytes || descriptors.empty() || coveredBytes != *expectedBytes)
    return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" IndexRelation descriptor coverage is not exact: ")
            .concat(llvm::Twine(coveredBytes))
            .concat(" covered bytes versus ")
            .concat(expectedBytes ? llvm::Twine(*expectedBytes)
                                  : llvm::Twine("overflow"))
            .str());
  return descriptors;
}

void createGatherScatterDescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors) {
  for (const MovementDescriptorPair &descriptor : descriptors)
    createGatherScatter(rewriter, loc, source, dest, descriptor.source,
                        descriptor.dest);
}

void createMappedRDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors) {
  for (const MovementDescriptorPair &descriptor : descriptors)
    rewriter.create<InstrRDMAOp>(
        loc, source, dest, descriptor.source.byteCount,
        descriptor.source.innerBytes,
        rewriter.getI64IntegerAttr(descriptor.source.byteOffset),
        rewriter.getI64IntegerAttr(descriptor.dest.byteOffset),
        descriptor.source.strides, descriptor.source.iterations);
}

void createMappedWDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors) {
  for (const MovementDescriptorPair &descriptor : descriptors)
    rewriter.create<InstrWDMAOp>(
        loc, source, dest, descriptor.dest.byteCount,
        descriptor.dest.innerBytes,
        rewriter.getI64IntegerAttr(descriptor.source.byteOffset),
        rewriter.getI64IntegerAttr(descriptor.dest.byteOffset),
        descriptor.dest.strides, descriptor.dest.iterations);
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

} // namespace wafer::tile_region_to_instr
