//===- PhysicalTensorDescriptor.cpp - Checked physical tensor facts -----===//

#include "Wafer/Target/Layout/PhysicalTensorDescriptor.h"

#include "llvm/Support/Errc.h"

#include <limits>
#include <utility>

namespace wafer {
namespace {

bool isKnownLayout(PhysicalTensorLayout layout) {
  switch (layout) {
  case PhysicalTensorLayout::Tensor:
  case PhysicalTensorLayout::NTensor:
  case PhysicalTensorLayout::Cx:
  case PhysicalTensorLayout::NCx:
    return true;
  }
  return false;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

} // namespace

llvm::Expected<PhysicalTensorDescriptor>
PhysicalTensorDescriptor::create(LogicalFormat format,
                                 PhysicalTensorLayout layout,
                                 std::vector<uint64_t> shape) {
  if (!findLogicalFormatDescriptor(format))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "physical tensor has an unknown format");
  if (!isKnownLayout(layout))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "physical tensor has an unknown layout");
  uint64_t elementCount = 1;
  for (uint64_t dimension : shape) {
    if (dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "physical tensor dimension must fit the static IR dimension domain");
    if (!checkedMultiply(elementCount, dimension, elementCount))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "physical tensor static element count overflows uint64_t");
  }
  return PhysicalTensorDescriptor(format, layout, std::move(shape),
                                  elementCount);
}

} // namespace wafer
