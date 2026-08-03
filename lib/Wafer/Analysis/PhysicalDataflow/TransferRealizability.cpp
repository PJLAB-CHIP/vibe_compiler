//===- TransferRealizability.cpp - Exact physical transfer proofs --------===//

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/Analysis/PhysicalDataflow/PhysicalAccessRelation.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::presburger;

namespace wafer::analysis {
namespace {

static WaferPhysicalEncodingAttrInterface getEncoding(mlir::MemRefType type) {
  return mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      type.getMemorySpace());
}

static std::optional<int64_t>
getStaticElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim) || dim < 0 ||
        llvm::MulOverflow(count, dim, count))
      return std::nullopt;
  }
  return count;
}

static mlir::LogicalResult verifyExactCoveredRelation(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, bool requireInjective) {
  if (relation.getStatus() != IndexRelationStatus::Exact ||
      relation.getDestinationRank() !=
          static_cast<unsigned>(destType.getRank()) ||
      relation.getSourceRank() != static_cast<unsigned>(sourceType.getRank()) ||
      !sourceType.hasStaticShape() || !destType.hasStaticShape() ||
      !relation.isFunctional().isProvenTrue() ||
      (requireInjective && !relation.isInjective().isProvenTrue()))
    return mlir::failure();

  IndexSetResult sourceDomain =
      IndexRelation::staticDomain(sourceType.getShape());
  IndexSetResult destDomain = IndexRelation::staticDomain(destType.getShape());
  if (!sourceDomain.isExact() || !destDomain.isExact())
    return mlir::failure();
  PresburgerSet relationDomain =
      relation.getPresburgerRelation().getDomainSet();
  PresburgerSet relationRange = relation.getPresburgerRelation().getRangeSet();
  if (!relationDomain.isEqual(*destDomain.set) ||
      !relationRange.isSubsetOf(*sourceDomain.set))
    return mlir::failure();
  return mlir::success();
}

static bool isCanonicalLinearRelation(mlir::MemRefType sourceType,
                                      mlir::MemRefType destType,
                                      const IndexRelation &relation) {
  IndexRelationResult canonicalReshape =
      IndexRelation::staticReshape(destType.getShape(), sourceType.getShape());
  return canonicalReshape.isExact() &&
         relation.isEquivalentTo(*canonicalReshape.get()).isProvenTrue();
}

static mlir::FailureOr<WaferPhysicalElementSpan>
getRepresentativeSpan(mlir::MemRefType type, bool last) {
  WaferPhysicalEncodingAttrInterface encoding = getEncoding(type);
  std::optional<int64_t> elementCount = getStaticElementCount(type.getShape());
  if (!encoding || !elementCount || *elementCount <= 0)
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> point(type.getRank(), 0);
  if (last)
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      point[dim] = type.getDimSize(dim) - 1;
  return encoding.getPhysicalElementSpan(type, point);
}

static mlir::LogicalResult
verifyByteAddressablePhysicalMaps(mlir::MemRefType sourceType,
                                  mlir::MemRefType destType) {
  for (bool last : {false, true}) {
    mlir::FailureOr<WaferPhysicalElementSpan> source =
        getRepresentativeSpan(sourceType, last);
    mlir::FailureOr<WaferPhysicalElementSpan> dest =
        getRepresentativeSpan(destType, last);
    if (mlir::failed(source) || mlir::failed(dest) ||
        source->bitLength != dest->bitLength || source->bitLength <= 0 ||
        source->bitLength % 8 != 0 || source->bitOffset % 8 != 0 ||
        dest->bitOffset % 8 != 0)
      return mlir::failure();
  }
  return mlir::success();
}

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

static mlir::LogicalResult
verifyEqualPhysicalFootprints(mlir::MemRefType sourceType,
                              mlir::MemRefType destType) {
  WaferPhysicalEncodingAttrInterface sourceEncoding = getEncoding(sourceType);
  WaferPhysicalEncodingAttrInterface destEncoding = getEncoding(destType);
  if (!sourceEncoding || !destEncoding)
    return mlir::failure();
  mlir::FailureOr<int64_t> sourceBytes =
      sourceEncoding.getPhysicalFootprintBytes(sourceType);
  mlir::FailureOr<int64_t> destBytes =
      destEncoding.getPhysicalFootprintBytes(destType);
  return mlir::success(mlir::succeeded(sourceBytes) &&
                       mlir::succeeded(destBytes) &&
                       *sourceBytes == *destBytes);
}

static bool isCompactCanonicalPhysicalMap(mlir::MemRefType type) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  WaferPhysicalEncodingAttrInterface encoding = getEncoding(type);
  std::optional<int64_t> valid = getStaticElementCount(type.getShape());
  if (!memory || !encoding || !valid || *valid <= 0 ||
      (memory.getLayout() != MemLayout::Tensor &&
       memory.getLayout() != MemLayout::NTensor))
    return false;
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      offset != 0 || strides.size() != static_cast<size_t>(type.getRank()))
    return false;
  int64_t expectedStride = 1;
  for (int64_t index = type.getRank() - 1; index >= 0; --index) {
    int64_t dim = type.getDimSize(index);
    if (dim < 0 || strides[index] != expectedStride ||
        llvm::MulOverflow(expectedStride, dim, expectedStride))
      return false;
  }
  mlir::FailureOr<WaferPhysicalElementSpan> first =
      getRepresentativeSpan(type, /*last=*/false);
  mlir::FailureOr<int64_t> footprint = encoding.getPhysicalFootprintBytes(type);
  int64_t compactBits = 0;
  if (mlir::failed(first) || mlir::failed(footprint) || first->bitOffset != 0 ||
      first->bitLength <= 0 ||
      llvm::MulOverflow(*valid, first->bitLength, compactBits))
    return false;
  int64_t compactBytes = compactBits / 8 + (compactBits % 8 != 0);
  return *footprint == compactBytes;
}

template <typename Callback>
static mlir::LogicalResult
forEachMappedElement(mlir::MemRefType sourceType, mlir::MemRefType destType,
                     const IndexRelation &relation,
                     const TransferRealizabilityLimits &limits,
                     Callback &&callback) {
  std::optional<int64_t> elementCount =
      getStaticElementCount(destType.getShape());
  if (!elementCount || *elementCount < 0 ||
      *elementCount > limits.maxEnumeratedElements)
    return mlir::failure();

  mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
      PhysicalAccessRelation::create(sourceType, destType.getShape(), relation,
                                     /*requireInjective=*/false);
  IndexRelationResult destinationIdentity =
      IndexRelation::identity(destType.getShape());
  if (mlir::failed(sourceAccess) || !destinationIdentity.isExact())
    return mlir::failure();
  mlir::FailureOr<PhysicalAccessRelation> destAccess =
      PhysicalAccessRelation::create(destType, destType.getShape(),
                                     *destinationIdentity.get(),
                                     /*requireInjective=*/true);
  if (mlir::failed(destAccess))
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> destination(destType.getRank(), 0);
  for (int64_t linear = 0; linear < *elementCount; ++linear) {
    mlir::FailureOr<WaferPhysicalElementSpan> sourceSpan =
        sourceAccess->getPhysicalElementSpan(destination);
    mlir::FailureOr<WaferPhysicalElementSpan> destSpan =
        destAccess->getPhysicalElementSpan(destination);
    if (mlir::failed(sourceSpan) || mlir::failed(destSpan) ||
        mlir::failed(callback(*sourceSpan, *destSpan)))
      return mlir::failure();

    for (int64_t dim = destType.getRank() - 1; dim >= 0; --dim) {
      if (++destination[dim] < destType.getDimSize(dim))
        break;
      destination[dim] = 0;
    }
  }
  return mlir::success();
}

template <typename Callback>
static mlir::LogicalResult forEachStaticReshapeElement(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const TransferRealizabilityLimits &limits, Callback &&callback) {
  std::optional<int64_t> sourceElements =
      getStaticElementCount(sourceType.getShape());
  std::optional<int64_t> destElements =
      getStaticElementCount(destType.getShape());
  if (!sourceElements || !destElements || *sourceElements != *destElements ||
      *destElements < 0 || *destElements > limits.maxEnumeratedElements)
    return mlir::failure();

  IndexRelationResult sourceRelation =
      IndexRelation::staticReshape(destType.getShape(), sourceType.getShape());
  IndexRelationResult destRelation =
      IndexRelation::identity(destType.getShape());
  if (!sourceRelation.isExact() || !destRelation.isExact())
    return mlir::failure();
  mlir::FailureOr<PhysicalAccessRelation> sourceAccess =
      PhysicalAccessRelation::create(sourceType, destType.getShape(),
                                     *sourceRelation.get(),
                                     /*requireInjective=*/true);
  mlir::FailureOr<PhysicalAccessRelation> destAccess =
      PhysicalAccessRelation::create(destType, destType.getShape(),
                                     *destRelation.get(),
                                     /*requireInjective=*/true);
  if (mlir::failed(sourceAccess) || mlir::failed(destAccess))
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> destination(destType.getRank(), 0);
  for (int64_t linear = 0; linear < *destElements; ++linear) {
    mlir::FailureOr<WaferPhysicalElementSpan> sourceSpan =
        sourceAccess->getPhysicalElementSpan(destination);
    mlir::FailureOr<WaferPhysicalElementSpan> destSpan =
        destAccess->getPhysicalElementSpan(destination);
    if (mlir::failed(sourceSpan) || mlir::failed(destSpan) ||
        mlir::failed(callback(*sourceSpan, *destSpan)))
      return mlir::failure();

    for (int64_t dim = destType.getRank() - 1; dim >= 0; --dim) {
      if (++destination[dim] < destType.getDimSize(dim))
        break;
      destination[dim] = 0;
    }
  }
  return mlir::success();
}

static mlir::LogicalResult proveByteAddressableElementTransfer(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, const TransferRealizabilityLimits &limits,
    bool requireInjective) {
  if (sourceType.getElementType() != destType.getElementType() ||
      mlir::failed(verifyExactCoveredRelation(sourceType, destType, relation,
                                              requireInjective)))
    return mlir::failure();
  if (isCanonicalLinearRelation(sourceType, destType, relation))
    return verifyByteAddressablePhysicalMaps(sourceType, destType);
  return forEachMappedElement(
      sourceType, destType, relation, limits,
      [](WaferPhysicalElementSpan source, WaferPhysicalElementSpan dest) {
        return mlir::success(
            source.bitLength == dest.bitLength && source.bitLength > 0 &&
            source.bitLength % 8 == 0 && source.bitOffset % 8 == 0 &&
            dest.bitOffset % 8 == 0);
      });
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

  llvm::SmallVector<int64_t, 4> first(iterationShape.size(), 0);
  llvm::SmallVector<int64_t, 4> last;
  last.reserve(iterationShape.size());
  for (int64_t dim : iterationShape) {
    if (dim <= 0)
      return mlir::failure();
    last.push_back(dim - 1);
  }
  for (llvm::ArrayRef<int64_t> point :
       {llvm::ArrayRef<int64_t>(first), llvm::ArrayRef<int64_t>(last)}) {
    mlir::FailureOr<WaferPhysicalElementSpan> sourceSpan =
        sourceAccess->getPhysicalElementSpan(point);
    mlir::FailureOr<WaferPhysicalElementSpan> destSpan =
        destAccess->getPhysicalElementSpan(point);
    if (mlir::failed(sourceSpan) || mlir::failed(destSpan) ||
        sourceSpan->bitLength != destSpan->bitLength ||
        sourceSpan->bitLength <= 0 || sourceSpan->bitLength % 8 != 0 ||
        sourceSpan->bitOffset % 8 != 0 || destSpan->bitOffset % 8 != 0)
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult TransferRealizability::proveMetadataView(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, bool destinationMayWrite,
    const TransferRealizabilityLimits &limits) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "analysis", "proveMetadataView", "total");
  if (mlir::failed(verifyMetadataViewTypeCompatibility(sourceType, destType)))
    return mlir::failure();

  {
    wafer::support::ScopedCompileTimingSpan relationTiming(
        "analysis-phase", "proveMetadataView", "exact-covered-relation");
    if (mlir::failed(verifyExactCoveredRelation(sourceType, destType, relation,
                                                destinationMayWrite)))
      return mlir::failure();
  }

  if (mlir::failed(verifyEqualPhysicalFootprints(sourceType, destType)))
    return mlir::failure();

  if (isCanonicalLinearRelation(sourceType, destType, relation) &&
      isCompactCanonicalPhysicalMap(sourceType) &&
      isCompactCanonicalPhysicalMap(destType))
    return mlir::success();

  return forEachMappedElement(
      sourceType, destType, relation, limits,
      [](WaferPhysicalElementSpan source, WaferPhysicalElementSpan dest) {
        return mlir::success(source == dest);
      });
}

mlir::LogicalResult TransferRealizability::proveStaticReshapeMetadataView(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    bool destinationMayWrite, const TransferRealizabilityLimits &limits) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "analysis", "proveStaticReshapeMetadataView", "total");
  if (mlir::failed(verifyMetadataViewTypeCompatibility(sourceType, destType)))
    return mlir::failure();

  std::optional<int64_t> sourceElements =
      getStaticElementCount(sourceType.getShape());
  std::optional<int64_t> destElements =
      getStaticElementCount(destType.getShape());
  if (!sourceElements || !destElements || *sourceElements != *destElements ||
      mlir::failed(verifyEqualPhysicalFootprints(sourceType, destType)))
    return mlir::failure();

  if (sourceType == destType)
    return mlir::success();

  // A canonical row-major reshape of equal finite element domains is a
  // bijection, so destinationMayWrite requires no additional relation query.
  (void)destinationMayWrite;
  if (isCompactCanonicalPhysicalMap(sourceType) &&
      isCompactCanonicalPhysicalMap(destType))
    return mlir::success();

  wafer::support::ScopedCompileTimingSpan elementTiming(
      "analysis-phase", "proveStaticReshapeMetadataView",
      "bounded-physical-element-proof");
  return forEachStaticReshapeElement(
      sourceType, destType, limits,
      [](WaferPhysicalElementSpan source, WaferPhysicalElementSpan dest) {
        return mlir::success(source == dest);
      });
}

mlir::LogicalResult TransferRealizability::proveCompactDma(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, const TransferRealizabilityLimits &limits) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() == destMemory.getSpace() ||
      sourceMemory.getLayout() != MemLayout::Tensor ||
      destMemory.getLayout() != MemLayout::Tensor ||
      sourceType.getShape() != destType.getShape())
    return mlir::failure();
  IndexRelationResult identity = IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      !relation.isEquivalentTo(*identity.get()).isProvenTrue())
    return mlir::failure();
  return proveByteAddressableElementTransfer(sourceType, destType, relation,
                                             limits,
                                             /*requireInjective=*/true);
}

mlir::LogicalResult TransferRealizability::proveMappedDma(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, const TransferRealizabilityLimits &limits) {
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
                                             limits,
                                             /*requireInjective=*/true);
}

mlir::LogicalResult TransferRealizability::proveGatherScatter(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, const TransferRealizabilityLimits &limits) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() != MemorySpace::SPM ||
      destMemory.getSpace() != MemorySpace::SPM)
    return mlir::failure();
  return proveByteAddressableElementTransfer(sourceType, destType, relation,
                                             limits,
                                             /*requireInjective=*/false);
}

mlir::LogicalResult TransferRealizability::proveStagedMovement(
    mlir::MemRefType sourceType, mlir::MemRefType temporaryType,
    mlir::MemRefType destType, const IndexRelation &sourceToTemporary,
    const IndexRelation &temporaryToDest,
    const TransferRealizabilityLimits &limits) {
  return mlir::success(
      mlir::succeeded(proveCompactDma(sourceType, temporaryType,
                                      sourceToTemporary, limits)) &&
      mlir::succeeded(proveGatherScatter(temporaryType, destType,
                                         temporaryToDest, limits)));
}

} // namespace wafer::analysis
