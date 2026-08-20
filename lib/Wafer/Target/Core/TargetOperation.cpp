//===- TargetOperation.cpp - Pure target operation protocol ------------===//

#include "Wafer/Target/Core/TargetOperation.h"

#include "Wafer/Target/Core/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <limits>

namespace wafer {
namespace {

template <typename Operation, size_t N>
llvm::Expected<Operation>
parseClosedOperation(uint32_t opcode,
                     const std::array<Operation, N> &operations,
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
    TargetPoolingOperation::Average, TargetPoolingOperation::Sum,
    TargetPoolingOperation::Maximum, TargetPoolingOperation::IndexedMaximum,
    TargetPoolingOperation::Minimum, TargetPoolingOperation::IndexedMinimum,
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
constexpr std::array kRoundingModes = {
    TargetRoundingMode::NearestEven,    TargetRoundingMode::TowardZero,
    TargetRoundingMode::TowardPositive, TargetRoundingMode::TowardNegative,
    TargetRoundingMode::Stochastic,
};
constexpr std::array kElementwiseOperations = {
    TargetElementwiseOperation::Abs,      TargetElementwiseOperation::Recip,
    TargetElementwiseOperation::Square,   TargetElementwiseOperation::Sqrt,
    TargetElementwiseOperation::Rsqrt,    TargetElementwiseOperation::Neg,
    TargetElementwiseOperation::Max,      TargetElementwiseOperation::Min,
    TargetElementwiseOperation::Add,      TargetElementwiseOperation::Sub,
    TargetElementwiseOperation::Mul,      TargetElementwiseOperation::Div,
    TargetElementwiseOperation::Eq,       TargetElementwiseOperation::Ne,
    TargetElementwiseOperation::Ge,       TargetElementwiseOperation::Gt,
    TargetElementwiseOperation::Le,       TargetElementwiseOperation::Lt,
    TargetElementwiseOperation::LogicNot, TargetElementwiseOperation::LogicAnd,
    TargetElementwiseOperation::LogicOr,  TargetElementwiseOperation::LogicXor,
    TargetElementwiseOperation::Log2,     TargetElementwiseOperation::Ln,
    TargetElementwiseOperation::Pow2,     TargetElementwiseOperation::Exp,
    TargetElementwiseOperation::ExpLp,    TargetElementwiseOperation::Sin,
    TargetElementwiseOperation::Cos,      TargetElementwiseOperation::Tanh,
    TargetElementwiseOperation::Sigmoid,  TargetElementwiseOperation::Relu,
    TargetElementwiseOperation::SatRelu,  TargetElementwiseOperation::LeakyRelu,
    TargetElementwiseOperation::Softplus,
};
constexpr std::array kReduceOperations = {
    TargetReduceOperation::Sum,
    TargetReduceOperation::Max,
    TargetReduceOperation::Min,
    TargetReduceOperation::Avg,
};

std::vector<size_t> getReducedDimensions(TargetReduceDimension dimension,
                                         size_t rank) {
  auto trailing = [&](size_t index) -> std::optional<size_t> {
    if (index >= rank)
      return std::nullopt;
    return rank - 1 - index;
  };
  std::vector<size_t> dimensions;
  switch (dimension) {
  case TargetReduceDimension::Trailing0:
    if (auto value = trailing(0))
      dimensions.push_back(*value);
    break;
  case TargetReduceDimension::Trailing1:
    if (auto value = trailing(1))
      dimensions.push_back(*value);
    break;
  case TargetReduceDimension::Trailing2:
    if (auto value = trailing(2))
      dimensions.push_back(*value);
    break;
  case TargetReduceDimension::Trailing3:
    if (auto value = trailing(3))
      dimensions.push_back(*value);
    break;
  case TargetReduceDimension::Trailing2And1: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    if (first && second)
      dimensions = {*first, *second};
    break;
  }
  case TargetReduceDimension::Trailing2And1And0: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    std::optional<size_t> third = trailing(0);
    if (first && second && third)
      dimensions = {*first, *second, *third};
    break;
  }
  }
  return dimensions;
}

} // namespace

llvm::ArrayRef<TargetRoundingMode> getTargetRoundingModes() {
  return kRoundingModes;
}

llvm::Expected<TargetRoundingMode> parseTargetRoundingMode(uint8_t value) {
  if (value <= static_cast<uint8_t>(TargetRoundingMode::Stochastic))
    return static_cast<TargetRoundingMode>(value);
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown target rounding mode %u",
                                 static_cast<unsigned>(value));
}

llvm::StringRef stringifyTargetRoundingMode(TargetRoundingMode mode) {
  switch (mode) {
  case TargetRoundingMode::NearestEven:
    return "nearest-even";
  case TargetRoundingMode::TowardZero:
    return "toward-zero";
  case TargetRoundingMode::TowardPositive:
    return "toward-positive-infinity";
  case TargetRoundingMode::TowardNegative:
    return "toward-negative-infinity";
  case TargetRoundingMode::Stochastic:
    return "stochastic";
  }
  llvm_unreachable("unknown target rounding mode");
}

std::optional<TargetRoundingMode>
TargetConvertParameter::getRoundingMode() const {
  if (kind != Kind::RoundingMode)
    return std::nullopt;
  return static_cast<TargetRoundingMode>(payload);
}

std::optional<uint32_t> TargetConvertParameter::getZeroPoint() const {
  if (kind != Kind::ZeroPoint)
    return std::nullopt;
  return payload;
}

llvm::ArrayRef<TargetElementwiseOperation> getTargetElementwiseOperations() {
  return kElementwiseOperations;
}

llvm::StringRef
stringifyTargetElementwiseOperation(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::Abs:
    return "abs";
  case TargetElementwiseOperation::Recip:
    return "recip";
  case TargetElementwiseOperation::Square:
    return "square";
  case TargetElementwiseOperation::Sqrt:
    return "sqrt";
  case TargetElementwiseOperation::Rsqrt:
    return "rsqrt";
  case TargetElementwiseOperation::Neg:
    return "neg";
  case TargetElementwiseOperation::Max:
    return "max";
  case TargetElementwiseOperation::Min:
    return "min";
  case TargetElementwiseOperation::Add:
    return "add";
  case TargetElementwiseOperation::Sub:
    return "sub";
  case TargetElementwiseOperation::Mul:
    return "mul";
  case TargetElementwiseOperation::Div:
    return "div";
  case TargetElementwiseOperation::Eq:
    return "eq";
  case TargetElementwiseOperation::Ne:
    return "ne";
  case TargetElementwiseOperation::Ge:
    return "ge";
  case TargetElementwiseOperation::Gt:
    return "gt";
  case TargetElementwiseOperation::Le:
    return "le";
  case TargetElementwiseOperation::Lt:
    return "lt";
  case TargetElementwiseOperation::LogicNot:
    return "logic_not";
  case TargetElementwiseOperation::LogicAnd:
    return "logic_and";
  case TargetElementwiseOperation::LogicOr:
    return "logic_or";
  case TargetElementwiseOperation::LogicXor:
    return "logic_xor";
  case TargetElementwiseOperation::Log2:
    return "log2";
  case TargetElementwiseOperation::Ln:
    return "ln";
  case TargetElementwiseOperation::Pow2:
    return "pow2";
  case TargetElementwiseOperation::Exp:
    return "exp";
  case TargetElementwiseOperation::ExpLp:
    return "exp_lp";
  case TargetElementwiseOperation::Sin:
    return "sin";
  case TargetElementwiseOperation::Cos:
    return "cos";
  case TargetElementwiseOperation::Tanh:
    return "tanh";
  case TargetElementwiseOperation::Sigmoid:
    return "sigmoid";
  case TargetElementwiseOperation::Relu:
    return "relu";
  case TargetElementwiseOperation::SatRelu:
    return "satrelu";
  case TargetElementwiseOperation::LeakyRelu:
    return "leakyrelu";
  case TargetElementwiseOperation::Softplus:
    return "softplus";
  }
  llvm_unreachable("unknown target elementwise operation");
}

unsigned getTargetElementwiseArity(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::Abs:
  case TargetElementwiseOperation::Recip:
  case TargetElementwiseOperation::Square:
  case TargetElementwiseOperation::Sqrt:
  case TargetElementwiseOperation::Rsqrt:
  case TargetElementwiseOperation::Neg:
  case TargetElementwiseOperation::LogicNot:
  case TargetElementwiseOperation::Log2:
  case TargetElementwiseOperation::Ln:
  case TargetElementwiseOperation::Pow2:
  case TargetElementwiseOperation::Exp:
  case TargetElementwiseOperation::ExpLp:
  case TargetElementwiseOperation::Sin:
  case TargetElementwiseOperation::Cos:
  case TargetElementwiseOperation::Tanh:
  case TargetElementwiseOperation::Sigmoid:
  case TargetElementwiseOperation::Relu:
  case TargetElementwiseOperation::SatRelu:
  case TargetElementwiseOperation::LeakyRelu:
  case TargetElementwiseOperation::Softplus:
    return 1;
  case TargetElementwiseOperation::Max:
  case TargetElementwiseOperation::Min:
  case TargetElementwiseOperation::Add:
  case TargetElementwiseOperation::Sub:
  case TargetElementwiseOperation::Mul:
  case TargetElementwiseOperation::Div:
  case TargetElementwiseOperation::Eq:
  case TargetElementwiseOperation::Ne:
  case TargetElementwiseOperation::Ge:
  case TargetElementwiseOperation::Gt:
  case TargetElementwiseOperation::Le:
  case TargetElementwiseOperation::Lt:
  case TargetElementwiseOperation::LogicAnd:
  case TargetElementwiseOperation::LogicOr:
  case TargetElementwiseOperation::LogicXor:
    return 2;
  }
  llvm_unreachable("unknown target elementwise operation");
}

bool isTargetElementwiseRelation(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::Eq:
  case TargetElementwiseOperation::Ne:
  case TargetElementwiseOperation::Ge:
  case TargetElementwiseOperation::Gt:
  case TargetElementwiseOperation::Le:
  case TargetElementwiseOperation::Lt:
    return true;
  default:
    return false;
  }
}

bool isTargetElementwiseLogic(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::LogicNot:
  case TargetElementwiseOperation::LogicAnd:
  case TargetElementwiseOperation::LogicOr:
  case TargetElementwiseOperation::LogicXor:
    return true;
  default:
    return false;
  }
}

llvm::ArrayRef<TargetReduceOperation> getTargetReduceOperations() {
  return kReduceOperations;
}

llvm::StringRef stringifyTargetReduceOperation(TargetReduceOperation kind) {
  switch (kind) {
  case TargetReduceOperation::Sum:
    return "sum";
  case TargetReduceOperation::Max:
    return "max";
  case TargetReduceOperation::Min:
    return "min";
  case TargetReduceOperation::Avg:
    return "avg";
  }
  llvm_unreachable("unknown target reduce operation");
}

llvm::StringRef
stringifyTargetReduceDimension(TargetReduceDimension dimension) {
  switch (dimension) {
  case TargetReduceDimension::Trailing0:
    return "trailing-0";
  case TargetReduceDimension::Trailing1:
    return "trailing-1";
  case TargetReduceDimension::Trailing2:
    return "trailing-2";
  case TargetReduceDimension::Trailing3:
    return "trailing-3";
  case TargetReduceDimension::Trailing2And1:
    return "trailing-2-and-1";
  case TargetReduceDimension::Trailing2And1And0:
    return "trailing-2-and-1-and-0";
  }
  llvm_unreachable("unknown target reduce dimension");
}

std::vector<size_t>
getTargetReduceLogicalDimensions(TargetReduceDimension dimension, size_t rank) {
  return getReducedDimensions(dimension, rank);
}

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

llvm::StringRef
stringifyTargetConvolutionOperation(TargetConvolutionOperation operation) {
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
