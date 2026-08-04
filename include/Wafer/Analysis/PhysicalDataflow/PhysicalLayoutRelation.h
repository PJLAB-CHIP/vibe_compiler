//===- PhysicalLayoutRelation.h - Encoding-owned physical map -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_PHYSICALLAYOUTRELATION_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_PHYSICALLAYOUTRELATION_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer::analysis {

/// Invocation-local normalization of an encoding's exact physical layout
/// pieces into one Presburger relation:
///
///   logical index -> physical element start bit offset
///
/// The relation contains valid logical elements only. Padding remains part of
/// the physical footprint and legal-access envelope, not a logical element.
/// This object owns no Operation/Value handles and must be rebuilt after a type
/// or encoding rewrite.
class PhysicalLayoutRelation {
public:
  static mlir::FailureOr<PhysicalLayoutRelation>
  create(mlir::MemRefType type,
         const IndexRelationLimits &limits = IndexRelationLimits());

  mlir::MemRefType getType() const { return type; }
  const IndexRelation &getLogicalToPhysicalBitOffset() const {
    return logicalToPhysicalBitOffset;
  }
  llvm::ArrayRef<WaferPhysicalLayoutPiece> getPieces() const { return pieces; }

  int64_t getElementBitWidth() const { return elementBitWidth; }
  int64_t getPhysicalFootprintBytes() const { return physicalFootprintBytes; }
  int64_t getMinimumAlignmentBytes() const { return minimumAlignmentBytes; }
  int64_t getValidElementCount() const { return validElementCount; }
  int64_t getPaddingElementCount() const { return paddingElementCount; }
  bool isByteAddressable() const { return byteAddressable; }

  mlir::FailureOr<WaferPhysicalElementSpan>
  getPhysicalElementSpan(llvm::ArrayRef<int64_t> logicalPoint) const;

private:
  PhysicalLayoutRelation(mlir::MemRefType type,
                         IndexRelation logicalToPhysicalBitOffset,
                         llvm::SmallVector<WaferPhysicalLayoutPiece, 2> pieces,
                         int64_t elementBitWidth,
                         int64_t physicalFootprintBytes,
                         int64_t minimumAlignmentBytes,
                         int64_t validElementCount, int64_t paddingElementCount,
                         bool byteAddressable)
      : type(type),
        logicalToPhysicalBitOffset(std::move(logicalToPhysicalBitOffset)),
        pieces(std::move(pieces)), elementBitWidth(elementBitWidth),
        physicalFootprintBytes(physicalFootprintBytes),
        minimumAlignmentBytes(minimumAlignmentBytes),
        validElementCount(validElementCount),
        paddingElementCount(paddingElementCount),
        byteAddressable(byteAddressable) {}

  mlir::MemRefType type;
  IndexRelation logicalToPhysicalBitOffset;
  llvm::SmallVector<WaferPhysicalLayoutPiece, 2> pieces;
  int64_t elementBitWidth = 0;
  int64_t physicalFootprintBytes = 0;
  int64_t minimumAlignmentBytes = 0;
  int64_t validElementCount = 0;
  int64_t paddingElementCount = 0;
  bool byteAddressable = false;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_PHYSICALLAYOUTRELATION_H
