//===- TransferRealizability.cpp - Exact physical transfer proofs --------===//

#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/Analysis/Tile/PhysicalAccessRelation.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include <limits>

namespace wafer::analysis {

mlir::FailureOr<int64_t> TransferRealizability::proveUnitVectorBroadcast(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &destinationToSource) {
  auto source = PhysicalLayoutRelation::create(sourceType);
  auto dest = PhysicalLayoutRelation::create(destType);
  if (mlir::failed(source) || mlir::failed(dest) ||
      !mlir::isa<mlir::FloatType>(sourceType.getElementType()))
    return mlir::failure();
  int64_t unit = source->getPhysicalElementCount();
  if (unit < 1 || unit > 64 || dest->getPhysicalElementCount() < unit)
    return mlir::failure();
  auto actual =
      destinationToSource.compose(source->getLogicalToPhysicalElementOrdinal());
  auto modulo = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(
          1, 0, mlir::getAffineDimExpr(0, sourceType.getContext()) % unit),
      {dest->getPhysicalElementCount()}, {unit});
  if (!actual.isExact() || !modulo.isExact())
    return mlir::failure();
  auto expected =
      dest->getLogicalToPhysicalElementOrdinal().compose(*modulo.get());
  if (!expected.isExact() ||
      !actual.get()->isEquivalentTo(*expected.get()).isProvenTrue())
    return mlir::failure();
  return unit;
}

mlir::FailureOr<int64_t> TransferRealizability::proveGroupedUnitVectorBroadcast(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &destinationToSource) {
  auto source = PhysicalLayoutRelation::create(sourceType);
  auto dest = PhysicalLayoutRelation::create(destType);
  if (mlir::failed(source) || mlir::failed(dest) ||
      !mlir::isa<mlir::FloatType>(sourceType.getElementType()))
    return mlir::failure();
  int64_t fullUnit = source->getPhysicalElementCount();
  int64_t full = dest->getPhysicalElementCount();
  if (fullUnit <= 64 || fullUnit % 64 || full < fullUnit || full % fullUnit ||
      full > std::numeric_limits<uint32_t>::max())
    return mlir::failure();
  int64_t group = (full / fullUnit) * 64;
  auto ordinal = mlir::getAffineDimExpr(0, sourceType.getContext());
  auto grouped = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, ordinal.floorDiv(group) * 64 + ordinal % 64),
      {full}, {fullUnit});
  auto actual =
      destinationToSource.compose(source->getLogicalToPhysicalElementOrdinal());
  if (!actual.isExact() || !grouped.isExact())
    return mlir::failure();
  auto expected =
      dest->getLogicalToPhysicalElementOrdinal().compose(*grouped.get());
  if (!expected.isExact() ||
      !actual.get()->isEquivalentTo(*expected.get()).isProvenTrue())
    return mlir::failure();
  return group;
}

// The encoding's piece/period contract proves that this entire rectangle is
// affine. Its origin and basis coefficients then determine every address;
// evaluating those coefficients is an exact proof, not element sampling.
static bool
hasPhysicalRectangleStrides(const WaferStaticPhysicalOffsetCalculator &offsets,
                            llvm::ArrayRef<WaferPhysicalLayoutPiece> pieces,
                            llvm::ArrayRef<int64_t> lower,
                            llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<int64_t> expectedByteStrides) {
  bool affine = llvm::any_of(pieces, [&](const auto &piece) {
    for (unsigned axis = 0; axis < lower.size(); ++axis) {
      int64_t begin = piece.logicalLowerBounds[axis];
      int64_t end = piece.logicalUpperBounds[axis];
      int64_t period = piece.logicalTilePeriods[axis];
      if (lower[axis] < begin || lower[axis] >= end || shape[axis] <= 0 ||
          shape[axis] > end - lower[axis])
        return false;
      if (period && (lower[axis] - begin) / period !=
                        (lower[axis] - begin + shape[axis] - 1) / period)
        return false;
    }
    return true;
  });
  if (!affine)
    return false;
  auto origin = offsets.getByteOffset(lower);
  if (!origin)
    return false;
  llvm::SmallVector<int64_t> next(lower);
  for (unsigned axis = 0; axis < lower.size(); ++axis) {
    if (shape[axis] == 1)
      continue;
    ++next[axis];
    auto step = offsets.getByteOffset(next);
    --next[axis];
    if (!step || *step - *origin != expectedByteStrides[axis])
      return false;
  }
  return true;
}

mlir::FailureOr<llvm::SmallVector<UnitVectorBroadcastSlice>>
TransferRealizability::proveUnitVectorBroadcastSlices(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    mlir::AffineMap destinationToSource) {
  wafer::support::ScopedCompileTimingSpan timing(
      "analysis", "proveUnitVectorBroadcastSlices", "total");
  auto sourceOffsets = WaferStaticPhysicalOffsetCalculator::create(sourceType);
  auto destOffsets = WaferStaticPhysicalOffsetCalculator::create(destType);
  auto info = computeWaferPhysicalTensorInfo(destType);
  if (!sourceOffsets || !destOffsets || !info ||
      !sourceType.getLayout().isIdentity() ||
      !destType.getLayout().isIdentity() || destType.getRank() < 2 ||
      sourceType.getRank() < 1 ||
      sourceType.getElementType() != destType.getElementType() ||
      !mlir::isa<mlir::FloatType>(sourceType.getElementType()) ||
      (info->layout != MemLayout::NCx && info->layout != MemLayout::Cx) ||
      info->cBlock != 64 || info->physicalElements <= 0 ||
      info->physicalElements > std::numeric_limits<uint32_t>::max() ||
      !destinationToSource || !destinationToSource.isProjectedPermutation() ||
      destinationToSource.getNumDims() != destType.getRank() ||
      destinationToSource.getNumResults() != sourceType.getRank() ||
      sourceType.getShape().back() != destType.getShape().back() ||
      destinationToSource.getResults().back() !=
          mlir::getAffineDimExpr(destType.getRank() - 1, destType.getContext()))
    return mlir::failure();

  auto sourcePieces =
      getWaferMemoryAttr(sourceType).getPhysicalLayoutPieces(sourceType);
  auto destPieces =
      getWaferMemoryAttr(destType).getPhysicalLayoutPieces(destType);
  if (mlir::failed(sourcePieces) || mlir::failed(destPieces))
    return mlir::failure();

  // Only the consecutive broadcast suffix can become one repeated unit.
  // Other axes remain explicit intervals; NCx outer slices retain bank gaps.
  int64_t suffixBegin = destType.getRank() - 1;
  int64_t minimumAxis = info->layout == MemLayout::NCx ? 1 : 0;
  int64_t repetitions = 1;
  while (suffixBegin > minimumAxis) {
    int64_t axis = suffixBegin - 1;
    if (destType.getDimSize(axis) != 1 &&
        llvm::is_contained(destinationToSource.getResults(),
                           mlir::getAffineDimExpr(axis, destType.getContext())))
      break;
    repetitions *= destType.getDimSize(axis);
    --suffixBegin;
  }
  if (repetitions <= 1)
    return mlir::failure();
  int64_t prefixCount = 1;
  for (int64_t axis = 0; axis < suffixBegin; ++axis)
    prefixCount *= destType.getDimSize(axis);
  // A compiler-work bound, not a tensor or SPM capacity limit. The existing
  // mapped movement remains available when this decomposition is too large.
  IndexRelationLimits limits;
  const int64_t blocks = info->cxBlocks + (info->tailC != 0);
  if (prefixCount <= 0 ||
      prefixCount > static_cast<int64_t>(limits.maxRectangularPieces) / blocks)
    return mlir::failure();

  llvm::SmallVector<UnitVectorBroadcastSlice> slices;
  int64_t coveredElements = 0;
  const int64_t channels = destType.getShape().back();
  for (int64_t prefix = 0; prefix < prefixCount; ++prefix) {
    llvm::SmallVector<int64_t> coordinates(destType.getRank(), 0);
    int64_t remaining = prefix;
    for (int64_t axis = suffixBegin; axis-- > 0;) {
      coordinates[axis] = remaining % destType.getDimSize(axis);
      remaining /= destType.getDimSize(axis);
    }
    for (int64_t block = 0; block < blocks; ++block) {
      coordinates.back() = block * 64;
      llvm::SmallVector<int64_t> sourceCoordinates;
      for (auto expression : destinationToSource.getResults())
        sourceCoordinates.push_back(
            coordinates[mlir::cast<mlir::AffineDimExpr>(expression)
                            .getPosition()]);
      auto src = sourceOffsets->getByteOffset(sourceCoordinates);
      auto dst = destOffsets->getByteOffset(coordinates);
      if (!src || !dst)
        return mlir::failure();
      int64_t valid = std::min(int64_t(64), channels - block * 64);
      int64_t unit = block < info->cxBlocks ? 64 : info->tailC;
      const int64_t sourceOffset = *src / info->elementBytes;
      const int64_t destOffset = *dst / info->elementBytes;
      if (sourceOffset + valid > sourceOffsets->getInfo().physicalElements ||
          destOffset + repetitions * unit > info->physicalElements)
        return mlir::failure();
      llvm::SmallVector<int64_t> localShape(destType.getShape());
      for (int64_t axis = 0; axis < suffixBegin; ++axis)
        localShape[axis] = 1;
      localShape.back() = valid;
      llvm::SmallVector<int64_t> destStrides(destType.getRank(), 0);
      destStrides.back() = info->elementBytes;
      int64_t stride = unit * info->elementBytes;
      for (int64_t axis = destType.getRank() - 1; axis-- > suffixBegin;) {
        destStrides[axis] = stride;
        stride *= destType.getDimSize(axis);
      }
      llvm::SmallVector<int64_t> sourceShape;
      for (auto expression : destinationToSource.getResults())
        sourceShape.push_back(
            localShape[mlir::cast<mlir::AffineDimExpr>(expression)
                           .getPosition()]);
      llvm::SmallVector<int64_t> sourceStrides(sourceType.getRank(), 0);
      sourceStrides.back() = info->elementBytes;
      if (!hasPhysicalRectangleStrides(*sourceOffsets, *sourcePieces,
                                       sourceCoordinates, sourceShape,
                                       sourceStrides) ||
          !hasPhysicalRectangleStrides(*destOffsets, *destPieces, coordinates,
                                       localShape, destStrides))
        return mlir::failure();
      coveredElements += repetitions * valid;
      slices.push_back(
          {sourceOffset, valid, destOffset, repetitions * unit, unit, 0});
    }
  }
  llvm::sort(slices, [](const auto &lhs, const auto &rhs) {
    return lhs.destOffset < rhs.destOffset;
  });
  llvm::SmallVector<UnitVectorBroadcastSlice> merged;
  for (const auto &slice : slices) {
    if (!merged.empty()) {
      auto &previous = merged.back();
      if (previous.destOffset + previous.destElements > slice.destOffset)
        return mlir::failure();
      if (previous.unitElements == 64 && slice.unitElements == 64 &&
          previous.sourceElements % 64 == 0 && slice.sourceElements == 64 &&
          previous.destOffset + previous.destElements == slice.destOffset &&
          previous.sourceOffset + previous.sourceElements ==
              slice.sourceOffset) {
        previous.groupElements = repetitions * 64;
        previous.destElements += slice.destElements;
        previous.sourceElements += slice.sourceElements;
        continue;
      }
    }
    merged.push_back(slice);
  }

  // The rectangles enumerate every prefix once, partition the channel axis,
  // and cover the complete broadcast suffix. Physical spans are disjoint.
  if (coveredElements != destType.getNumElements())
    return mlir::failure();
  return merged;
}

namespace {

static mlir::LogicalResult
verifyMetadataViewTypeCompatibility(mlir::MemRefType sourceType,
                                    mlir::MemRefType destType) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  llvm::SmallVector<int64_t, 4> sourceStrides;
  llvm::SmallVector<int64_t, 4> destStrides;
  int64_t sourceOffset = 0;
  int64_t destOffset = 0;
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() != destMemory.getSpace() ||
      sourceType.getElementType() != destType.getElementType() ||
      mlir::failed(
          mlir::getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      mlir::failed(
          mlir::getStridesAndOffset(destType, destStrides, destOffset)) ||
      mlir::ShapedType::isDynamic(sourceOffset) ||
      mlir::ShapedType::isDynamic(destOffset) || sourceOffset != destOffset)
    return mlir::failure();
  return mlir::success();
}

static mlir::FailureOr<
    std::pair<PhysicalAccessRelation, PhysicalAccessRelation>>
getCanonicalAccessPair(mlir::MemRefType sourceType, mlir::MemRefType destType,
                       const IndexRelation &destinationToSource,
                       bool requireInjectiveSource) {
  IndexRelationResult destinationIdentity =
      IndexRelation::identity(destType.getShape());
  if (!destinationIdentity.isExact())
    return mlir::failure();
  mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
      PhysicalAccessRelation::create(sourceType, destType.getShape(),
                                     destinationToSource,
                                     requireInjectiveSource);
  mlir::FailureOr<PhysicalAccessRelation> destAccess =
      PhysicalAccessRelation::create(destType, destType.getShape(),
                                     *destinationIdentity.get(),
                                     /*requireInjective=*/true);
  if (mlir::failed(sourceAccess) || mlir::failed(destAccess))
    return mlir::failure();
  return std::make_pair(std::move(*sourceAccess), std::move(*destAccess));
}

static mlir::LogicalResult proveByteAddressableElementTransfer(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, bool requireInjective) {
  if (sourceType.getElementType() != destType.getElementType())
    return mlir::failure();
  auto accesses =
      getCanonicalAccessPair(sourceType, destType, relation, requireInjective);
  if (mlir::failed(accesses))
    return mlir::failure();
  const PhysicalLayoutRelation &sourceLayout =
      accesses->first.getPhysicalLayoutRelation();
  const PhysicalLayoutRelation &destLayout =
      accesses->second.getPhysicalLayoutRelation();
  return mlir::success(
      sourceLayout.isByteAddressable() && destLayout.isByteAddressable() &&
      sourceLayout.getElementBitWidth() == destLayout.getElementBitWidth());
}

// Proves only relative byte geometry. Actual view-base alignment is checked
// from SSA by the shared packed-view proof before target address emission.
static bool hasWholeBytePackedRows(mlir::MemRefType type) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (!type.hasStaticShape() ||
      mlir::failed(mlir::getStridesAndOffset(type, strides, offset)))
    return false;
  int64_t inner = 1;
  bool strided = false;
  for (int64_t axis = type.getRank(); axis-- > 0;) {
    int64_t extent = type.getDimSize(axis);
    if (extent <= 0 || strides[axis] < 0 ||
        mlir::ShapedType::isDynamic(strides[axis]))
      return false;
    if (extent == 1)
      continue;
    if (!strided && strides[axis] == inner) {
      if (inner > std::numeric_limits<int64_t>::max() / extent)
        return false;
      inner *= extent;
      continue;
    }
    strided = true;
    if (strides[axis] % 8)
      return false;
  }
  return inner % 8 == 0;
}

} // namespace

mlir::LogicalResult TransferRealizability::proveMappedTransfer(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    llvm::ArrayRef<int64_t> iterationShape,
    const IndexRelation &iterationToSource,
    const IndexRelation &iterationToDest) {
  if (sourceType.getElementType() != destType.getElementType())
    return mlir::failure();
  mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
      PhysicalAccessRelation::create(sourceType, iterationShape,
                                     iterationToSource,
                                     /*requireInjective=*/false);
  mlir::FailureOr<PhysicalAccessRelation> destAccess =
      PhysicalAccessRelation::create(destType, iterationShape, iterationToDest,
                                     /*requireInjective=*/true);
  if (mlir::failed(sourceAccess) || mlir::failed(destAccess))
    return mlir::failure();
  const PhysicalLayoutRelation &sourceLayout =
      sourceAccess->getPhysicalLayoutRelation();
  const PhysicalLayoutRelation &destLayout =
      destAccess->getPhysicalLayoutRelation();
  // The general mapped-transfer proof is expressed in physical bit offsets
  // and spans, so bit-packed layouts are just as representable as
  // byte-addressable layouts. Concrete DMA/GS route proofs retain their
  // byte-addressability requirements below.
  return mlir::success(sourceLayout.getElementBitWidth() ==
                       destLayout.getElementBitWidth());
}

mlir::LogicalResult TransferRealizability::provePhysicalTraversal(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    llvm::ArrayRef<int64_t> iterationShape,
    const IndexRelation &iterationToSource,
    const IndexRelation &iterationToDest) {
  mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
      PhysicalAccessRelation::create(sourceType, iterationShape,
                                     iterationToSource,
                                     /*requireInjective=*/false);
  mlir::FailureOr<PhysicalAccessRelation> destAccess =
      PhysicalAccessRelation::create(destType, iterationShape, iterationToDest,
                                     /*requireInjective=*/true);
  if (mlir::failed(sourceAccess) || mlir::failed(destAccess))
    return mlir::failure();
  return mlir::success(
      sourceAccess->hasSamePhysicalTraversal(*destAccess).isProvenTrue());
}

mlir::LogicalResult TransferRealizability::proveMetadataView(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, bool destinationMayWrite) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "analysis", "proveMetadataView", "total");
  if (mlir::failed(verifyMetadataViewTypeCompatibility(sourceType, destType)))
    return mlir::failure();

  auto accesses = getCanonicalAccessPair(sourceType, destType, relation,
                                         destinationMayWrite);
  if (mlir::failed(accesses) ||
      accesses->first.getPhysicalFootprintBytes() !=
          accesses->second.getPhysicalFootprintBytes())
    return mlir::failure();

  wafer::support::ScopedCompileTimingSpan relationTiming(
      "analysis-phase", "proveMetadataView", "presburger-physical-equivalence");
  return mlir::success(
      accesses->first.hasSamePhysicalElementMapping(accesses->second)
          .isProvenTrue());
}

mlir::LogicalResult TransferRealizability::proveStaticReshapeMetadataView(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    bool destinationMayWrite) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "analysis", "proveStaticReshapeMetadataView", "total");
  std::optional<CanonicalReshapeRelations> projected =
      getCanonicalReshapeRelations(sourceType.getContext(),
                                   sourceType.getShape(), destType.getShape());
  if (projected) {
    if (mlir::failed(verifyMetadataViewTypeCompatibility(sourceType, destType)))
      return mlir::failure();
    mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
        PhysicalAccessRelation::create(sourceType, projected->iterationShape,
                                       projected->iterationToSource,
                                       destinationMayWrite);
    mlir::FailureOr<PhysicalAccessRelation> destAccess =
        PhysicalAccessRelation::create(destType, projected->iterationShape,
                                       projected->iterationToDest,
                                       /*requireInjective=*/true);
    if (mlir::failed(sourceAccess) || mlir::failed(destAccess) ||
        sourceAccess->getPhysicalFootprintBytes() !=
            destAccess->getPhysicalFootprintBytes())
      return mlir::failure();
    return mlir::success(
        sourceAccess->hasSamePhysicalElementMapping(*destAccess)
            .isProvenTrue());
  }
  IndexRelationResult reshape =
      IndexRelation::staticReshape(destType.getShape(), sourceType.getShape());
  if (!reshape.isExact())
    return mlir::failure();
  return proveMetadataView(sourceType, destType, *reshape.get(),
                           destinationMayWrite);
}

mlir::LogicalResult
TransferRealizability::proveCompactDma(mlir::MemRefType sourceType,
                                       mlir::MemRefType destType,
                                       const IndexRelation &relation) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() == destMemory.getSpace() ||
      sourceMemory.getLayout() != MemLayout::Tensor ||
      destMemory.getLayout() != MemLayout::Tensor ||
      sourceType.getElementType() != destType.getElementType() ||
      sourceType.getShape() != destType.getShape())
    return mlir::failure();
  IndexRelationResult identity = IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      !relation.isEquivalentTo(*identity.get()).isProvenTrue())
    return mlir::failure();
  // The engine descriptor is strided on DDR only. A dynamic base offset is
  // carried by the SSA view, but gaps between SPM rows require mapped DMA.
  auto spmType =
      sourceMemory.getSpace() == MemorySpace::SPM ? sourceType : destType;
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (!spmType.hasStaticShape() ||
      mlir::failed(mlir::getStridesAndOffset(spmType, strides, offset)))
    return mlir::failure();
  int64_t contiguousStride = 1;
  for (int64_t axis = spmType.getRank(); axis-- > 0;) {
    const int64_t extent = spmType.getDimSize(axis);
    if (extent <= 0 || (extent > 1 && strides[axis] != contiguousStride) ||
        contiguousStride > std::numeric_limits<int64_t>::max() / extent)
      return mlir::failure();
    contiguousStride *= extent;
  }
  auto sourceEncoding =
      mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
          sourceType.getMemorySpace());
  auto destEncoding =
      mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
          destType.getMemorySpace());
  if (!sourceEncoding || !destEncoding)
    return mlir::failure();
  mlir::FailureOr<int64_t> sourceBits =
      sourceEncoding.getPhysicalElementBitWidth(sourceType);
  mlir::FailureOr<int64_t> destBits =
      destEncoding.getPhysicalElementBitWidth(destType);
  if (mlir::failed(sourceBits) || mlir::failed(destBits) || *sourceBits <= 0 ||
      *sourceBits != *destBits)
    return mlir::failure();
  if (*sourceBits == 1)
    return mlir::success(mlir::succeeded(provePackedByteRows(sourceType)) &&
                         mlir::succeeded(provePackedByteRows(destType)));
  return mlir::success(*sourceBits % 8 == 0);
}

mlir::LogicalResult
TransferRealizability::provePackedByteRows(mlir::MemRefType type) {
  auto memory = getWaferMemoryAttr(type);
  return mlir::success(memory && memory.getLayout() == MemLayout::Tensor &&
                       type.getElementType().isInteger(1) &&
                       hasWholeBytePackedRows(type));
}

mlir::LogicalResult
TransferRealizability::proveMappedDma(mlir::MemRefType sourceType,
                                      mlir::MemRefType destType,
                                      const IndexRelation &relation) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() == destMemory.getSpace() ||
      sourceType.getShape() != destType.getShape())
    return mlir::failure();
  bool acceptedDirection = (sourceMemory.getSpace() == MemorySpace::DDR &&
                            destMemory.getSpace() == MemorySpace::SPM) ||
                           (sourceMemory.getSpace() == MemorySpace::SPM &&
                            destMemory.getSpace() == MemorySpace::DDR);
  if (!acceptedDirection)
    return mlir::failure();
  IndexRelationResult identity = IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      !relation.isEquivalentTo(*identity.get()).isProvenTrue())
    return mlir::failure();
  return proveByteAddressableElementTransfer(sourceType, destType, relation,
                                             /*requireInjective=*/true);
}

mlir::LogicalResult
TransferRealizability::proveGatherScatter(mlir::MemRefType sourceType,
                                          mlir::MemRefType destType,
                                          const IndexRelation &relation) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() != MemorySpace::SPM ||
      destMemory.getSpace() != MemorySpace::SPM)
    return mlir::failure();
  return proveByteAddressableElementTransfer(sourceType, destType, relation,
                                             /*requireInjective=*/false);
}

mlir::LogicalResult TransferRealizability::proveStagedMovement(
    mlir::MemRefType sourceType, mlir::MemRefType temporaryType,
    mlir::MemRefType destType, const IndexRelation &sourceToTemporary,
    const IndexRelation &temporaryToDest) {
  return mlir::success((mlir::succeeded(proveCompactDma(
                            sourceType, temporaryType, sourceToTemporary)) ||
                        mlir::succeeded(proveMappedDma(
                            sourceType, temporaryType, sourceToTemporary))) &&
                       mlir::succeeded(proveGatherScatter(
                           temporaryType, destType, temporaryToDest)));
}

} // namespace wafer::analysis
