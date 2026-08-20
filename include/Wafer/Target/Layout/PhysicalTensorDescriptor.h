//===- PhysicalTensorDescriptor.h - Target tensor geometry -----*- C++ -*-===//

#ifndef WAFER_TARGET_PHYSICALTENSORDESCRIPTOR_H
#define WAFER_TARGET_PHYSICALTENSORDESCRIPTOR_H

#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Layout/PhysicalLayout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer {

/// Checked physical tensor geometry. It carries no model/profile identity or
/// digest; evidence owners hash the concrete facts they actually bind.
class PhysicalTensorDescriptor {
public:
  PhysicalTensorDescriptor() = delete;

  static llvm::Expected<PhysicalTensorDescriptor>
  create(LogicalFormat format, PhysicalTensorLayout layout,
         std::vector<uint64_t> shape);

  LogicalFormat getFormat() const { return format; }
  PhysicalTensorLayout getLayout() const { return layout; }
  llvm::ArrayRef<uint64_t> getShape() const { return shape; }
  uint64_t getElementCount() const { return elementCount; }

  friend bool operator==(const PhysicalTensorDescriptor &lhs,
                         const PhysicalTensorDescriptor &rhs) {
    return lhs.format == rhs.format && lhs.layout == rhs.layout &&
           lhs.shape == rhs.shape && lhs.elementCount == rhs.elementCount;
  }
  friend bool operator!=(const PhysicalTensorDescriptor &lhs,
                         const PhysicalTensorDescriptor &rhs) {
    return !(lhs == rhs);
  }

private:
  PhysicalTensorDescriptor(LogicalFormat format, PhysicalTensorLayout layout,
                           std::vector<uint64_t> shape, uint64_t elementCount)
      : format(format), layout(layout), shape(std::move(shape)),
        elementCount(elementCount) {}

  LogicalFormat format;
  PhysicalTensorLayout layout;
  std::vector<uint64_t> shape;
  uint64_t elementCount;
};

} // namespace wafer

#endif // WAFER_TARGET_PHYSICALTENSORDESCRIPTOR_H
