//===- TargetOperation.h - Pure target operation protocol -----*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETOPERATION_H
#define WAFER_TARGET_TARGETOPERATION_H

#include "Wafer/Target/Core/NCCCompletion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace wafer {

/// Exact CT-convert command selected by its target opcode. Construction is
/// checked against the target convert-route registry; the wrapper prevents a
/// raw integer from becoming a semantic identity in target-call descriptors.
class TargetConvertOperation {
public:
  TargetConvertOperation() = delete;

  static llvm::Expected<TargetConvertOperation> create(uint16_t opcode);
  uint16_t getOpcode() const { return opcode; }

  friend constexpr bool operator==(TargetConvertOperation lhs,
                                   TargetConvertOperation rhs) {
    return lhs.opcode == rhs.opcode;
  }
  friend constexpr bool operator!=(TargetConvertOperation lhs,
                                   TargetConvertOperation rhs) {
    return !(lhs == rhs);
  }

private:
  explicit constexpr TargetConvertOperation(uint16_t opcode) : opcode(opcode) {}
  uint16_t opcode;
};

enum class TargetConvolutionOperation : uint8_t {
  Convolution = 0,
  DepthwiseConvolution = 1,
  BackwardConvolution = 2,
};

enum class TargetPoolingOperation : uint16_t {
  Average = 115,
  Sum = 116,
  Maximum = 117,
  IndexedMaximum = 118,
  Minimum = 119,
  IndexedMinimum = 120,
};

enum class TargetUnpoolingOperation : uint16_t {
  Unpool = 121,
  Average = 122,
  Mask = 123,
};

enum class TargetPeripheralOperation : uint16_t {
  Count = 175,
  ArgMaximum = 177,
  ArgMinimum = 178,
  Factorize = 180,
  Bilinear = 182,
  LookupTable16 = 183,
  LookupTable32 = 184,
  Random = 185,
  ElementMask = 186,
};

llvm::ArrayRef<TargetConvolutionOperation> getTargetConvolutionOperations();
llvm::ArrayRef<TargetPoolingOperation> getTargetPoolingOperations();
llvm::ArrayRef<TargetUnpoolingOperation> getTargetUnpoolingOperations();
llvm::ArrayRef<TargetPeripheralOperation> getTargetPeripheralOperations();

llvm::StringRef stringifyTargetConvolutionOperation(
    TargetConvolutionOperation operation);
llvm::StringRef stringifyTargetPoolingOperation(TargetPoolingOperation operation);
llvm::StringRef
stringifyTargetUnpoolingOperation(TargetUnpoolingOperation operation);
llvm::StringRef
stringifyTargetPeripheralOperation(TargetPeripheralOperation operation);

llvm::Expected<TargetConvolutionOperation>
parseTargetConvolutionOperation(uint32_t opcode);
llvm::Expected<TargetPoolingOperation>
parseTargetPoolingOperation(uint32_t opcode);
llvm::Expected<TargetUnpoolingOperation>
parseTargetUnpoolingOperation(uint32_t opcode);
llvm::Expected<TargetPeripheralOperation>
parseTargetPeripheralOperation(uint32_t opcode);

} // namespace wafer

#endif // WAFER_TARGET_TARGETOPERATION_H
