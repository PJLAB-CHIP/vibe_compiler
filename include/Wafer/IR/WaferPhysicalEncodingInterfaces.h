//===- WaferPhysicalEncodingInterfaces.h - Physical encoding interface ---===//

#ifndef WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H
#define WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer {

/// A contiguous physical bit segment relative to a memref view base.
struct WaferPhysicalElementSpan {
  int64_t bitOffset = -1;
  int64_t bitLength = -1;

  friend bool operator==(const WaferPhysicalElementSpan &lhs,
                         const WaferPhysicalElementSpan &rhs) {
    return lhs.bitOffset == rhs.bitOffset && lhs.bitLength == rhs.bitLength;
  }
};

/// One exact piece of an encoding-owned logical-index to physical-bit-offset
/// relation.  Bounds are half-open and have the same rank as the queried
/// memref.  The affine map has that logical rank, no symbols, and one result:
/// the physical bit offset relative to the current memref view base.
///
/// Pieces are a typed projection of the encoding, not persistent IR or a
/// lowering plan.  Consumers compose them with their logical IndexRelation and
/// discard the derived Presburger relation after the current IR epoch.
struct WaferPhysicalLayoutPiece {
  llvm::SmallVector<int64_t, 4> logicalLowerBounds;
  llvm::SmallVector<int64_t, 4> logicalUpperBounds;
  /// Per-logical-dimension period at which a quasi-affine map changes affine
  /// stride. Zero means that the map is affine throughout this piece along
  /// that dimension. Periods describe symbolic partition geometry, not an
  /// enumerated segment list.
  llvm::SmallVector<int64_t, 4> logicalTilePeriods;
  mlir::AffineMap logicalToPhysicalBitOffset;
};

} // namespace wafer

#include "Wafer/IR/WaferPhysicalEncodingInterfaces.h.inc"

#endif // WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H
