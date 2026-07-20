//===- TransferRealizability.cpp - Exact physical transfer proofs --------===//

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/ADT/DynamicAPInt.h"
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

static PresburgerSet getPointSet(llvm::ArrayRef<int64_t> point) {
  IntegerPolyhedron set(PresburgerSpace::getSetSpace(point.size()));
  for (auto [index, value] : llvm::enumerate(point))
    set.addBound(BoundType::EQ, index, value);
  return PresburgerSet(set);
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getMappedSourcePoint(const IndexRelation &relation,
                     llvm::ArrayRef<int64_t> destination) {
  PresburgerRelation restricted =
      relation.getPresburgerRelation().intersectDomain(
          getPointSet(destination));
  llvm::SmallVector<llvm::DynamicAPInt, 8> sample;
  if (!restricted.findIntegerSample(sample))
    return mlir::failure();
  unsigned destinationRank = relation.getDestinationRank();
  unsigned sourceRank = relation.getSourceRank();
  if (sample.size() < destinationRank + sourceRank)
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> source;
  source.reserve(sourceRank);
  for (unsigned index = 0; index < sourceRank; ++index)
    source.push_back(static_cast<int64_t>(sample[destinationRank + index]));
  return source;
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

static bool isCompactCanonicalPhysicalMap(mlir::MemRefType type) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  WaferPhysicalEncodingAttrInterface encoding = getEncoding(type);
  std::optional<int64_t> valid = getStaticElementCount(type.getShape());
  if (!memory || !encoding || !valid || *valid <= 0 ||
      (memory.getLayout() != MemLayout::Tensor &&
       memory.getLayout() != MemLayout::NTensor))
    return false;
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

  WaferPhysicalEncodingAttrInterface sourceEncoding = getEncoding(sourceType);
  WaferPhysicalEncodingAttrInterface destEncoding = getEncoding(destType);
  if (!sourceEncoding || !destEncoding)
    return mlir::failure();

  bool usesCanonicalLinearOrder =
      isCanonicalLinearRelation(sourceType, destType, relation);

  llvm::SmallVector<int64_t, 4> destination(destType.getRank(), 0);
  for (int64_t linear = 0; linear < *elementCount; ++linear) {
    llvm::SmallVector<int64_t, 4> source;
    if (usesCanonicalLinearOrder) {
      source.resize(sourceType.getRank(), 0);
      int64_t remaining = linear;
      for (int64_t dim = sourceType.getRank() - 1; dim >= 0; --dim) {
        source[dim] = remaining % sourceType.getDimSize(dim);
        remaining /= sourceType.getDimSize(dim);
      }
    } else {
      mlir::FailureOr<llvm::SmallVector<int64_t, 4>> mapped =
          getMappedSourcePoint(relation, destination);
      if (mlir::failed(mapped))
        return mlir::failure();
      source = std::move(*mapped);
    }
    mlir::FailureOr<WaferPhysicalElementSpan> sourceSpan =
        sourceEncoding.getPhysicalElementSpan(sourceType, source);
    mlir::FailureOr<WaferPhysicalElementSpan> destSpan =
        destEncoding.getPhysicalElementSpan(destType, destination);
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

mlir::LogicalResult TransferRealizability::proveMetadataView(
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    const IndexRelation &relation, bool destinationMayWrite,
    const TransferRealizabilityLimits &limits) {
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr destMemory = getWaferMemoryAttr(destType);
  if (!sourceMemory || !destMemory ||
      sourceMemory.getSpace() != destMemory.getSpace() ||
      sourceType.getElementType() != destType.getElementType() ||
      mlir::failed(verifyExactCoveredRelation(sourceType, destType, relation,
                                              destinationMayWrite)))
    return mlir::failure();

  mlir::FailureOr<int64_t> sourceBytes =
      getEncoding(sourceType).getPhysicalFootprintBytes(sourceType);
  mlir::FailureOr<int64_t> destBytes =
      getEncoding(destType).getPhysicalFootprintBytes(destType);
  if (mlir::failed(sourceBytes) || mlir::failed(destBytes) ||
      *sourceBytes != *destBytes)
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
