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
