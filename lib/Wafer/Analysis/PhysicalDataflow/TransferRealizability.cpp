//===- TransferRealizability.cpp - Exact physical transfer proofs --------===//

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/Analysis/PhysicalDataflow/PhysicalAccessRelation.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

namespace wafer::analysis {
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
      sourceType.getShape() != destType.getShape())
    return mlir::failure();
  IndexRelationResult identity = IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      !relation.isEquivalentTo(*identity.get()).isProvenTrue())
    return mlir::failure();
  return proveByteAddressableElementTransfer(sourceType, destType, relation,
                                             /*requireInjective=*/true);
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
  return mlir::success(mlir::succeeded(proveCompactDma(
                           sourceType, temporaryType, sourceToTemporary)) &&
                       mlir::succeeded(proveGatherScatter(
                           temporaryType, destType, temporaryToDest)));
}

} // namespace wafer::analysis
