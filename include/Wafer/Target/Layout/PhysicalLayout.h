//===- PhysicalLayout.h - Pure target physical tensor layout -*- C++ -*-===//

#ifndef WAFER_TARGET_PHYSICALLAYOUT_H
#define WAFER_TARGET_PHYSICALLAYOUT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace wafer {

/// Target protocol layout identity. MLIR adapters map Wafer MemLayout to this
/// enum; host/runtime/model code never needs an MLIR context to use it.
enum class PhysicalTensorLayout : uint8_t { Tensor, NTensor, Cx, NCx };

llvm::StringRef stringifyPhysicalTensorLayout(PhysicalTensorLayout layout);

struct PhysicalTensorGeometry {
  PhysicalTensorLayout layout = PhysicalTensorLayout::Tensor;
  int64_t compactBytes = -1;
  int64_t physicalBytes = -1;
  int64_t physicalElements = -1;
  int64_t elementBytes = -1;
  int64_t cBlock = 0;
  int64_t cxBlocks = 0;
  int64_t c0 = 0;
  int64_t alignedC = -1;
  int64_t tailC = 0;
  int64_t outerElements = -1;
  int64_t hwElements = -1;
  int64_t batchElements = -1;
  int64_t bankAlignElements = -1;
  bool bitPackedElement = false;
};

/// Computes checked target geometry. `physicalElementSpan` is the exact
/// strided element span for Tensor/NTensor and is ignored for blocked layouts.
/// Dynamic dimensions are represented by negative values and return no result.
std::optional<PhysicalTensorGeometry> computePhysicalTensorGeometry(
    llvm::ArrayRef<int64_t> shape, int64_t elementBitWidth,
    bool usesInt8ChannelBlock, PhysicalTensorLayout layout,
    std::optional<int64_t> physicalElementSpan);

/// Reusable coordinate mapper over one validated physical tensor geometry.
/// Unblocked layouts consume exact element strides; blocked layouts derive the
/// target-owned Cx/NCx map from the shared geometry.
class StaticPhysicalTensorOffsetCalculator {
public:
  static std::optional<StaticPhysicalTensorOffsetCalculator>
  create(PhysicalTensorGeometry geometry, llvm::ArrayRef<int64_t> shape,
         llvm::ArrayRef<int64_t> elementStrides);

  const PhysicalTensorGeometry &getGeometry() const { return geometry; }

  std::optional<int64_t>
  getBitOffset(llvm::ArrayRef<int64_t> logicalIndices) const;

private:
  StaticPhysicalTensorOffsetCalculator(
      PhysicalTensorGeometry geometry, llvm::SmallVector<int64_t, 4> shape,
      llvm::SmallVector<int64_t, 4> bitStrides,
      llvm::SmallVector<int64_t, 4> linearStrides, int64_t fullC,
      int64_t blockStrideElements, int64_t fullBlockElements,
      int64_t elementBitWidth)
      : geometry(std::move(geometry)), shape(std::move(shape)),
        bitStrides(std::move(bitStrides)),
        linearStrides(std::move(linearStrides)), fullC(fullC),
        blockStrideElements(blockStrideElements),
        fullBlockElements(fullBlockElements), elementBitWidth(elementBitWidth) {
  }

  PhysicalTensorGeometry geometry;
  llvm::SmallVector<int64_t, 4> shape;
  llvm::SmallVector<int64_t, 4> bitStrides;
  llvm::SmallVector<int64_t, 4> linearStrides;
  int64_t fullC = 0;
  int64_t blockStrideElements = 0;
  int64_t fullBlockElements = 0;
  int64_t elementBitWidth = 0;
};

} // namespace wafer

#endif // WAFER_TARGET_PHYSICALLAYOUT_H
