//===- TargetOperation.cpp - Pure target operation protocol ------------===//

#include "Wafer/Target/TargetOperation.h"

#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <limits>

namespace wafer {
namespace {

template <typename Operation, size_t N>
llvm::Expected<Operation> parseClosedOperation(
    uint32_t opcode, const std::array<Operation, N> &operations,
    llvm::StringRef family) {
  if (opcode > std::numeric_limits<std::underlying_type_t<Operation>>::max())
    return llvm::createStringError("%s opcode is outside its storage domain",
                                   family.str().c_str());
  Operation candidate = static_cast<Operation>(opcode);
  if (!llvm::is_contained(operations, candidate))
    return llvm::createStringError("%s opcode is not registered",
                                   family.str().c_str());
  return candidate;
}

constexpr std::array kConvolutionOperations = {
    TargetConvolutionOperation::Convolution,
    TargetConvolutionOperation::DepthwiseConvolution,
    TargetConvolutionOperation::BackwardConvolution,
};
constexpr std::array kPoolingOperations = {
    TargetPoolingOperation::Average,        TargetPoolingOperation::Sum,
    TargetPoolingOperation::Maximum,        TargetPoolingOperation::IndexedMaximum,
    TargetPoolingOperation::Minimum,        TargetPoolingOperation::IndexedMinimum,
};
constexpr std::array kUnpoolingOperations = {
    TargetUnpoolingOperation::Unpool,
    TargetUnpoolingOperation::Average,
    TargetUnpoolingOperation::Mask,
};
constexpr std::array kPeripheralOperations = {
    TargetPeripheralOperation::Count,
    TargetPeripheralOperation::ArgMaximum,
    TargetPeripheralOperation::ArgMinimum,
    TargetPeripheralOperation::Factorize,
    TargetPeripheralOperation::Bilinear,
    TargetPeripheralOperation::LookupTable16,
    TargetPeripheralOperation::LookupTable32,
    TargetPeripheralOperation::Random,
    TargetPeripheralOperation::ElementMask,
};

} // namespace

llvm::Expected<TargetConvertOperation>
TargetConvertOperation::create(uint16_t opcode) {
  if (!findTargetConvertRoute(opcode))
    return llvm::createStringError("CT convert opcode is not registered");
  return TargetConvertOperation(opcode);
}

llvm::ArrayRef<TargetConvolutionOperation> getTargetConvolutionOperations() {
  return kConvolutionOperations;
}

llvm::ArrayRef<TargetPoolingOperation> getTargetPoolingOperations() {
  return kPoolingOperations;
}

llvm::ArrayRef<TargetUnpoolingOperation> getTargetUnpoolingOperations() {
  return kUnpoolingOperations;
}

llvm::ArrayRef<TargetPeripheralOperation> getTargetPeripheralOperations() {
  return kPeripheralOperations;
}

llvm::StringRef stringifyTargetConvolutionOperation(
    TargetConvolutionOperation operation) {
  switch (operation) {
  case TargetConvolutionOperation::Convolution:
    return "conv";
  case TargetConvolutionOperation::DepthwiseConvolution:
    return "depthwise_conv";
  case TargetConvolutionOperation::BackwardConvolution:
    return "backward_conv";
  }
  llvm_unreachable("unknown target convolution operation");
}

llvm::StringRef
stringifyTargetPoolingOperation(TargetPoolingOperation operation) {
  switch (operation) {
  case TargetPoolingOperation::Average:
    return "avg";
  case TargetPoolingOperation::Sum:
    return "sum";
  case TargetPoolingOperation::Maximum:
    return "max";
  case TargetPoolingOperation::IndexedMaximum:
    return "indexedmax";
  case TargetPoolingOperation::Minimum:
    return "min";
  case TargetPoolingOperation::IndexedMinimum:
    return "indexedmin";
  }
  llvm_unreachable("unknown target pooling operation");
}

llvm::StringRef
stringifyTargetUnpoolingOperation(TargetUnpoolingOperation operation) {
  switch (operation) {
  case TargetUnpoolingOperation::Unpool:
    return "unpool";
  case TargetUnpoolingOperation::Average:
    return "avg";
  case TargetUnpoolingOperation::Mask:
    return "mask";
  }
  llvm_unreachable("unknown target unpooling operation");
}

llvm::StringRef
stringifyTargetPeripheralOperation(TargetPeripheralOperation operation) {
  switch (operation) {
  case TargetPeripheralOperation::Count:
    return "count";
  case TargetPeripheralOperation::ArgMaximum:
    return "argmax";
  case TargetPeripheralOperation::ArgMinimum:
    return "argmin";
  case TargetPeripheralOperation::Factorize:
    return "factorize";
  case TargetPeripheralOperation::Bilinear:
    return "bilinear";
  case TargetPeripheralOperation::LookupTable16:
    return "lut16";
  case TargetPeripheralOperation::LookupTable32:
    return "lut32";
  case TargetPeripheralOperation::Random:
    return "rand_gen";
  case TargetPeripheralOperation::ElementMask:
    return "elem_mask";
  }
  llvm_unreachable("unknown target peripheral operation");
}

llvm::Expected<TargetConvolutionOperation>
parseTargetConvolutionOperation(uint32_t opcode) {
  return parseClosedOperation(opcode, kConvolutionOperations, "convolution");
}

llvm::Expected<TargetPoolingOperation>
parseTargetPoolingOperation(uint32_t opcode) {
  return parseClosedOperation(opcode, kPoolingOperations, "pooling");
}

llvm::Expected<TargetUnpoolingOperation>
parseTargetUnpoolingOperation(uint32_t opcode) {
  return parseClosedOperation(opcode, kUnpoolingOperations, "unpooling");
}

llvm::Expected<TargetPeripheralOperation>
parseTargetPeripheralOperation(uint32_t opcode) {
  return parseClosedOperation(opcode, kPeripheralOperations, "peripheral");
}

} // namespace wafer
