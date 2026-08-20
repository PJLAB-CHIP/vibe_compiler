//===- TargetOperation.h - Pure target operation protocol -----*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETOPERATION_H
#define WAFER_TARGET_TARGETOPERATION_H

#include "Wafer/Target/Core/NCCCompletion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace wafer {

enum class TargetRoundingMode : uint8_t {
  NearestEven = 0,
  TowardZero = 1,
  TowardPositive = 2,
  TowardNegative = 3,
  Stochastic = 4,
};

llvm::ArrayRef<TargetRoundingMode> getTargetRoundingModes();
llvm::Expected<TargetRoundingMode> parseTargetRoundingMode(uint8_t value);
llvm::StringRef stringifyTargetRoundingMode(TargetRoundingMode mode);

class TargetConvertParameter {
public:
  enum class Kind : uint8_t { RoundingMode, ZeroPoint };
  TargetConvertParameter() = delete;
  static constexpr TargetConvertParameter
  roundingMode(TargetRoundingMode mode) {
    return TargetConvertParameter(Kind::RoundingMode,
                                  static_cast<uint32_t>(mode));
  }
  static constexpr TargetConvertParameter zeroPoint(uint32_t value) {
    return TargetConvertParameter(Kind::ZeroPoint, value);
  }
  Kind getKind() const { return kind; }
  std::optional<TargetRoundingMode> getRoundingMode() const;
  std::optional<uint32_t> getZeroPoint() const;
  friend constexpr bool operator==(TargetConvertParameter lhs,
                                   TargetConvertParameter rhs) {
    return lhs.kind == rhs.kind && lhs.payload == rhs.payload;
  }

private:
  constexpr TargetConvertParameter(Kind kind, uint32_t payload)
      : kind(kind), payload(payload) {}
  Kind kind;
  uint32_t payload;
};

enum class TargetElementwiseOperation : uint8_t {
  Abs,
  Recip,
  Square,
  Sqrt,
  Rsqrt,
  Neg,
  Max,
  Min,
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
  LogicNot,
  LogicAnd,
  LogicOr,
  LogicXor,
  Log2,
  Ln,
  Pow2,
  Exp,
  ExpLp,
  Sin,
  Cos,
  Tanh,
  Sigmoid,
  Relu,
  SatRelu,
  LeakyRelu,
  Softplus,
};

llvm::ArrayRef<TargetElementwiseOperation> getTargetElementwiseOperations();
llvm::StringRef
stringifyTargetElementwiseOperation(TargetElementwiseOperation operation);
unsigned getTargetElementwiseArity(TargetElementwiseOperation operation);
bool isTargetElementwiseRelation(TargetElementwiseOperation operation);
bool isTargetElementwiseLogic(TargetElementwiseOperation operation);

enum class TargetReduceOperation : uint8_t { Sum, Max, Min, Avg };
llvm::ArrayRef<TargetReduceOperation> getTargetReduceOperations();
llvm::StringRef stringifyTargetReduceOperation(TargetReduceOperation operation);

enum class TargetReduceDimension : uint8_t {
  Trailing0 = 0,
  Trailing1 = 1,
  Trailing2 = 2,
  Trailing3 = 3,
  Trailing2And1 = 4,
  Trailing2And1And0 = 5,
};

llvm::StringRef stringifyTargetReduceDimension(TargetReduceDimension dimension);
std::vector<size_t>
getTargetReduceLogicalDimensions(TargetReduceDimension dimension, size_t rank);

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

llvm::StringRef
stringifyTargetConvolutionOperation(TargetConvolutionOperation operation);
llvm::StringRef
stringifyTargetPoolingOperation(TargetPoolingOperation operation);
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
