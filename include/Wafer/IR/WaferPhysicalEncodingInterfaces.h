//===- WaferPhysicalEncodingInterfaces.h - Physical encoding interface ---===//

#ifndef WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H
#define WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

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

} // namespace wafer

#include "Wafer/IR/WaferPhysicalEncodingInterfaces.h.inc"

#endif // WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_H
