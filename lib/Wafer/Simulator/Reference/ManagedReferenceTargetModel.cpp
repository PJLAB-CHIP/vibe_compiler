//===- ManagedReferenceTargetModel.cpp - Scalable tensor reference -------===//

#include "Wafer/Simulator/Reference/TargetNumericBackend.h"

#include "Wafer/Target/PhysicalTensor/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model {
namespace {

constexpr llvm::StringLiteral kImplementation = "native-non-nan-f16-f32-tensor";

llvm::Error referenceError(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 llvm::Twine("managed-reference tensor: ") +
                                     detail);
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

float floatFromBits(uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t floatToBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float halfFromBits(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT16_C(0x8000)) << 16;
  uint32_t exponent = (bits >> 10) & UINT16_C(0x1f);
  uint32_t fraction = bits & UINT16_C(0x03ff);
  uint32_t result = sign;
  if (exponent == 0) {
    if (fraction == 0)
      return floatFromBits(result);
    int32_t unbiasedExponent = -14;
    while ((fraction & UINT32_C(0x0400)) == 0) {
      fraction <<= 1;
      --unbiasedExponent;
    }
    fraction &= UINT32_C(0x03ff);
    result |= static_cast<uint32_t>(unbiasedExponent + 127) << 23;
    result |= fraction << 13;
    return floatFromBits(result);
  }
  if (exponent == 31) {
    result |= UINT32_C(0x7f800000) | (fraction << 13);
    return floatFromBits(result);
  }
  exponent = exponent - 15 + 127;
  result |= exponent << 23;
  result |= fraction << 13;
  return floatFromBits(result);
}

std::optional<uint16_t> halfToBits(float value) {
  const uint32_t bits = floatToBits(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & UINT32_C(0x8000));
  const uint32_t exponentBits = (bits >> 23) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponentBits == UINT32_C(0xff)) {
    if (fraction != 0)
      return std::nullopt;
    return static_cast<uint16_t>(sign | UINT16_C(0x7c00));
  }

  const int32_t exponent = static_cast<int32_t>(exponentBits) - 127;
  if (exponent > 15)
    return static_cast<uint16_t>(sign | UINT16_C(0x7c00));
  if (exponent >= -14) {
    uint32_t destinationExponent = static_cast<uint32_t>(exponent + 15);
    uint32_t destinationFraction = fraction >> 13;
    const uint32_t remainder = fraction & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) &&
         (destinationFraction & UINT32_C(1)) != 0)) {
      ++destinationFraction;
      if (destinationFraction == UINT32_C(0x0400)) {
        destinationFraction = 0;
        ++destinationExponent;
        if (destinationExponent >= UINT32_C(0x1f))
          return static_cast<uint16_t>(sign | UINT16_C(0x7c00));
      }
    }
    return static_cast<uint16_t>(sign | (destinationExponent << 10) |
                                 destinationFraction);
  }

  if (exponent < -25)
    return sign;
  const uint32_t significand = fraction | UINT32_C(0x00800000);
  const uint32_t shift = static_cast<uint32_t>(-exponent - 1);
  uint32_t destinationFraction = significand >> shift;
  const uint32_t remainderMask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder = significand & remainderMask;
  const uint32_t halfway = UINT32_C(1) << (shift - 1);
  if (remainder > halfway ||
      (remainder == halfway && (destinationFraction & UINT32_C(1)) != 0))
    ++destinationFraction;
  if (destinationFraction == UINT32_C(0x0400))
    return static_cast<uint16_t>(sign | UINT16_C(0x0400));
  return static_cast<uint16_t>(sign | destinationFraction);
}

llvm::Expected<float> decodeNonNaN(RawLogicalValue value) {
  float decoded = 0.0F;
  switch (value.format) {
  case LogicalFormat::F16:
    decoded = halfFromBits(static_cast<uint16_t>(value.bits));
    break;
  case LogicalFormat::F32:
    decoded = floatFromBits(static_cast<uint32_t>(value.bits));
    break;
  default:
    return referenceError("only F16 and F32 values are supported");
  }
  if (std::isnan(decoded))
    return referenceError("input value is outside the non-NaN domain");
  return decoded;
}

llvm::Expected<RawLogicalValue> encodeNonNaN(float value,
                                             LogicalFormat format) {
  if (std::isnan(value))
    return referenceError("result value is outside the non-NaN domain");
  if (format == LogicalFormat::F32)
    return RawLogicalValue{format, floatToBits(value)};
  if (format == LogicalFormat::F16) {
    std::optional<uint16_t> bits = halfToBits(value);
    if (!bits)
      return referenceError("result cannot be encoded in the F16 domain");
    return RawLogicalValue{format, *bits};
  }
  return referenceError("only F16 and F32 results are supported");
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackInput(const TargetModelNumericTensor &tensor) {
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(tensor.key, tensor.storage);
  if (!values)
    return referenceError(llvm::toString(values.takeError()));
  return values;
}

llvm::Error validateTensorIdentity(
    const TargetModelNumericOperands &tensors,
    llvm::ArrayRef<const PhysicalTensorDescriptor *> expectedInputs,
    const PhysicalTensorDescriptor &expectedDestination) {
  if (tensors.destinationTemplate.key != expectedDestination ||
      tensors.inputs.size() != expectedInputs.size())
    return referenceError("request tensor key does not match operation");
  for (auto [input, expected] : llvm::zip_equal(tensors.inputs, expectedInputs))
    if (input.key != *expected)
      return referenceError("request input key does not match operation");
  return llvm::Error::success();
}

llvm::Error validateRequestIdentity(const TargetModelConvertRequest &request) {
  const PhysicalTensorDescriptor *inputs[] = {&request.operation.source};
  return validateTensorIdentity(request.tensors, inputs,
                                request.operation.destination);
}

llvm::Error
validateRequestIdentity(const TargetModelElementwiseRequest &request) {
  llvm::SmallVector<const PhysicalTensorDescriptor *, 2> inputs;
  for (const PhysicalTensorDescriptor &input : request.operation.inputs)
    inputs.push_back(&input);
  return validateTensorIdentity(request.tensors, inputs,
                                request.operation.destination);
}

llvm::Error validateRequestIdentity(const TargetModelReduceRequest &request) {
  const PhysicalTensorDescriptor *inputs[] = {&request.operation.input};
  return validateTensorIdentity(request.tensors, inputs,
                                request.operation.destination);
}

llvm::Expected<float> evaluateElementwise(TargetElementwiseOperation operation,
                                          llvm::ArrayRef<float> operands) {
  switch (operation) {
  case TargetElementwiseOperation::Abs:
    return std::fabs(operands[0]);
  case TargetElementwiseOperation::Neg:
    return -operands[0];
  case TargetElementwiseOperation::Recip:
    return 1.0F / operands[0];
  case TargetElementwiseOperation::Square:
    return operands[0] * operands[0];
  case TargetElementwiseOperation::Max:
    return std::fmax(operands[0], operands[1]);
  case TargetElementwiseOperation::Min:
    return std::fmin(operands[0], operands[1]);
  case TargetElementwiseOperation::Add:
    return operands[0] + operands[1];
  case TargetElementwiseOperation::Sub:
    return operands[0] - operands[1];
  case TargetElementwiseOperation::Mul:
    return operands[0] * operands[1];
  case TargetElementwiseOperation::Div:
    return operands[0] / operands[1];
  case TargetElementwiseOperation::Relu:
    return operands[0] < 0.0F ? 0.0F : operands[0];
  case TargetElementwiseOperation::Sqrt:
    return std::sqrt(operands[0]);
  case TargetElementwiseOperation::Rsqrt:
    return 1.0F / std::sqrt(operands[0]);
  case TargetElementwiseOperation::Log2:
    return std::log2(operands[0]);
  case TargetElementwiseOperation::Ln:
    return std::log(operands[0]);
  case TargetElementwiseOperation::Pow2:
    return std::exp2(operands[0]);
  case TargetElementwiseOperation::Exp:
    return std::exp(operands[0]);
  case TargetElementwiseOperation::Sin:
    return std::sin(operands[0]);
  case TargetElementwiseOperation::Cos:
    return std::cos(operands[0]);
  case TargetElementwiseOperation::Tanh:
    return std::tanh(operands[0]);
  case TargetElementwiseOperation::Sigmoid:
    if (operands[0] >= 0.0F)
      return 1.0F / (1.0F + std::exp(-operands[0]));
    return std::exp(operands[0]) / (1.0F + std::exp(operands[0]));
  case TargetElementwiseOperation::Softplus:
    return std::fmax(operands[0], 0.0F) +
           std::log1p(std::exp(-std::fabs(operands[0])));
  case TargetElementwiseOperation::Eq:
  case TargetElementwiseOperation::Ne:
  case TargetElementwiseOperation::Ge:
  case TargetElementwiseOperation::Gt:
  case TargetElementwiseOperation::Le:
  case TargetElementwiseOperation::Lt:
  case TargetElementwiseOperation::LogicNot:
  case TargetElementwiseOperation::LogicAnd:
  case TargetElementwiseOperation::LogicOr:
  case TargetElementwiseOperation::LogicXor:
  case TargetElementwiseOperation::ExpLp:
  case TargetElementwiseOperation::SatRelu:
  case TargetElementwiseOperation::LeakyRelu:
    return referenceError("elementwise operation is not in the managed domain");
  }
  llvm_unreachable("unknown numeric elementwise operation");
}

llvm::Expected<std::vector<RawLogicalValue>>
executeElementwise(const FormalElementwiseOperation &command,
                   llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  const uint64_t outputCount = command.destination.getElementCount();
  if (inputs.size() != command.inputs.size() ||
      inputs.size() != getTargetElementwiseArity(command.operation))
    return referenceError("elementwise input arity does not match semantics");
  for (auto [input, key] : llvm::zip_equal(inputs, command.inputs)) {
    if (input.size() != outputCount ||
        key.getShape() != command.destination.getShape() ||
        key.getLayout() != command.destination.getLayout() ||
        key.getFormat() != command.destination.getFormat())
      return referenceError(
          "elementwise tensors do not have one same-shape F16/F32 domain");
  }
  const LogicalFormat format = command.destination.getFormat();
  if (format != LogicalFormat::F16 && format != LogicalFormat::F32)
    return referenceError("elementwise result is not F16 or F32");

  std::vector<RawLogicalValue> result;
  result.reserve(static_cast<size_t>(outputCount));
  llvm::SmallVector<float, 2> operands;
  for (uint64_t index = 0; index < outputCount; ++index) {
    operands.clear();
    for (const std::vector<RawLogicalValue> &input : inputs) {
      llvm::Expected<float> value = decodeNonNaN(input[index]);
      if (!value)
        return value.takeError();
      operands.push_back(*value);
    }
    llvm::Expected<float> value =
        evaluateElementwise(command.operation, operands);
    if (!value)
      return value.takeError();
    llvm::Expected<RawLogicalValue> encoded = encodeNonNaN(*value, format);
    if (!encoded)
      return encoded.takeError();
    result.push_back(*encoded);
  }
  return result;
}

llvm::Expected<std::vector<RawLogicalValue>>
executeConvert(const FormalConvertOperation &command,
               llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  if (!findTargetConvertRoute(command.opcode))
    return referenceError("convert opcode is not a target route");
  if (command.parameter &&
      command.parameter->getRoundingMode() != TargetRoundingMode::NearestEven)
    return referenceError("convert is not nearest-even");
  const LogicalFormat sourceFormat = command.source.getFormat();
  const LogicalFormat destinationFormat = command.destination.getFormat();
  if (!((sourceFormat == LogicalFormat::F16 &&
         destinationFormat == LogicalFormat::F32) ||
        (sourceFormat == LogicalFormat::F32 &&
         destinationFormat == LogicalFormat::F16)) ||
      command.source.getShape() != command.destination.getShape() ||
      command.source.getLayout() != command.destination.getLayout() ||
      inputs.size() != 1 ||
      inputs.front().size() != command.destination.getElementCount())
    return referenceError("convert is outside the F16/F32 same-shape domain");

  std::vector<RawLogicalValue> result;
  result.reserve(inputs.front().size());
  for (RawLogicalValue input : inputs.front()) {
    llvm::Expected<float> value = decodeNonNaN(input);
    if (!value)
      return value.takeError();
    llvm::Expected<RawLogicalValue> encoded =
        encodeNonNaN(*value, destinationFormat);
    if (!encoded)
      return encoded.takeError();
    result.push_back(*encoded);
  }
  return result;
}

llvm::Expected<std::vector<RawLogicalValue>>
executeReduce(const FormalReduceOperation &command,
              llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  if ((command.operation != TargetReduceOperation::Sum &&
       command.operation != TargetReduceOperation::Max &&
       command.operation != TargetReduceOperation::Min) ||
      command.input.getFormat() != LogicalFormat::F32 ||
      command.destination.getFormat() != LogicalFormat::F32 ||
      inputs.size() != 1 ||
      inputs.front().size() != command.input.getElementCount())
    return referenceError(
        "reduce is outside the native F32 sum/max/min domain");
  const llvm::ArrayRef<uint64_t> inputShape = command.input.getShape();
  const std::vector<size_t> reducedDimensions =
      getTargetReduceLogicalDimensions(command.dimension, inputShape.size());
  if (reducedDimensions.empty())
    return referenceError("reduce has no logical dimensions");

  const uint64_t outputCount = command.destination.getElementCount();
  float identity = command.operation == TargetReduceOperation::Max
                       ? -std::numeric_limits<float>::infinity()
                   : command.operation == TargetReduceOperation::Min
                       ? std::numeric_limits<float>::infinity()
                       : 0.0F;
  std::vector<float> accumulators(static_cast<size_t>(outputCount), identity);
  llvm::SmallVector<bool, 4> reduced(inputShape.size(), false);
  for (size_t dimension : reducedDimensions)
    reduced[dimension] = true;
  for (uint64_t inputIndex = 0; inputIndex < inputs.front().size();
       ++inputIndex) {
    uint64_t remaining = inputIndex;
    llvm::SmallVector<uint64_t, 4> coordinates(inputShape.size(), 0);
    for (size_t reverse = inputShape.size(); reverse > 0; --reverse) {
      const size_t dimension = reverse - 1;
      coordinates[dimension] = remaining % inputShape[dimension];
      remaining /= inputShape[dimension];
    }
    uint64_t destinationIndex = 0;
    for (size_t dimension = 0; dimension < inputShape.size(); ++dimension) {
      if (!reduced[dimension])
        destinationIndex =
            destinationIndex * inputShape[dimension] + coordinates[dimension];
    }
    if (destinationIndex >= outputCount)
      return referenceError("reduce mapped input outside the destination");
    llvm::Expected<float> value = decodeNonNaN(inputs.front()[inputIndex]);
    if (!value)
      return value.takeError();
    float &accumulator = accumulators[destinationIndex];
    if (command.operation == TargetReduceOperation::Sum) {
      accumulator += *value;
    } else if (accumulator == 0.0F && *value == 0.0F) {
      bool negative = command.operation == TargetReduceOperation::Max
                          ? std::signbit(accumulator) && std::signbit(*value)
                          : std::signbit(accumulator) || std::signbit(*value);
      accumulator = negative ? -0.0F : 0.0F;
    } else {
      accumulator = command.operation == TargetReduceOperation::Max
                        ? std::fmax(accumulator, *value)
                        : std::fmin(accumulator, *value);
    }
    if (std::isnan(accumulators[destinationIndex]))
      return referenceError("reduce accumulator left the non-NaN domain");
  }

  std::vector<RawLogicalValue> result;
  result.reserve(accumulators.size());
  for (float value : accumulators)
    result.push_back({LogicalFormat::F32, floatToBits(value)});
  return result;
}

uint64_t getScalarEvaluations(const FormalConvertOperation &operation) {
  return operation.destination.getElementCount();
}
uint64_t getScalarEvaluations(const FormalElementwiseOperation &operation) {
  return operation.destination.getElementCount();
}
uint64_t getScalarEvaluations(const FormalReduceOperation &operation) {
  return operation.input.getElementCount();
}

llvm::Error validateBudget(const TargetModelNumericOperands &tensors,
                           FormalNumericWorkBudget scalarBudget,
                           OneDNNNumericWorkBudget byteBudget,
                           uint64_t scalarEvaluations) {
  if (scalarEvaluations > scalarBudget.getMaximumScalarEvaluations())
    return referenceError("scalar evaluation budget exceeded");
  uint64_t totalBytes = tensors.destinationTemplate.storage.size();
  uint64_t logicalValues = tensors.destinationTemplate.key.getElementCount();
  for (const TargetModelNumericTensor &input : tensors.inputs) {
    if (!checkedAdd(totalBytes, input.storage.size(), totalBytes) ||
        !checkedAdd(logicalValues, input.key.getElementCount(), logicalValues))
      return referenceError("work count overflows uint64_t");
  }
  if (totalBytes > byteBudget.getMaximumTotalBytes())
    return referenceError("physical byte budget exceeded");
  uint64_t scratchBytes = 0;
  if (!checkedMultiply(logicalValues, sizeof(RawLogicalValue), scratchBytes) ||
      scratchBytes > byteBudget.getMaximumScratchpadBytes())
    return referenceError("logical scratch budget exceeded");
  return llvm::Error::success();
}

llvm::Expected<std::vector<RawLogicalValue>>
executeReferenceOperation(const FormalConvertOperation &operation,
                          llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  return executeConvert(operation, inputs);
}

llvm::Expected<std::vector<RawLogicalValue>>
executeReferenceOperation(const FormalElementwiseOperation &operation,
                          llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  return executeElementwise(operation, inputs);
}

llvm::Expected<std::vector<RawLogicalValue>>
executeReferenceOperation(const FormalReduceOperation &operation,
                          llvm::ArrayRef<std::vector<RawLogicalValue>> inputs) {
  return executeReduce(operation, inputs);
}

template <typename Request>
llvm::Expected<TargetModelManagedReferenceResult>
executeManagedReference(const Request &request,
                        FormalNumericWorkBudget scalarBudget,
                        OneDNNNumericWorkBudget byteBudget,
                        const OneDNNExecutionEnvironment &environment) {
  if (llvm::Error error = validateRequestIdentity(request))
    return std::move(error);
  const uint64_t scalarEvaluations = getScalarEvaluations(request.operation);
  if (llvm::Error error = validateBudget(request.tensors, scalarBudget,
                                         byteBudget, scalarEvaluations))
    return std::move(error);

  std::vector<std::vector<RawLogicalValue>> inputs;
  inputs.reserve(request.tensors.inputs.size());
  for (const TargetModelNumericTensor &input : request.tensors.inputs) {
    llvm::Expected<std::vector<RawLogicalValue>> values = unpackInput(input);
    if (!values)
      return values.takeError();
    inputs.push_back(std::move(*values));
  }

  llvm::Expected<std::vector<RawLogicalValue>> output =
      executeReferenceOperation(request.operation, inputs);
  if (!output)
    return output.takeError();
  if (output->size() !=
      request.tensors.destinationTemplate.key.getElementCount())
    return referenceError("kernel produced the wrong element count");
  llvm::Expected<std::vector<uint8_t>> storage =
      packPhysicalTensorLogicalValues(
          request.tensors.destinationTemplate.key, *output,
          request.tensors.destinationTemplate.storage);
  if (!storage)
    return referenceError(llvm::toString(storage.takeError()));
  return TargetModelManagedReferenceResult{
      {request.tensors.destinationTemplate.key, std::move(*storage)},
      {},
      {scalarEvaluations, environment.getDigest().str(),
       kImplementation.str()}};
}

} // namespace

llvm::Expected<TargetModelManagedReferenceResult>
ManagedReferenceTargetModelBackend::execute(
    const TargetModelConvertRequest &request,
    FormalNumericWorkBudget scalarBudget) const {
  return executeManagedReference(request, scalarBudget, budget, environment);
}

llvm::Expected<TargetModelManagedReferenceResult>
ManagedReferenceTargetModelBackend::execute(
    const TargetModelElementwiseRequest &request,
    FormalNumericWorkBudget scalarBudget) const {
  return executeManagedReference(request, scalarBudget, budget, environment);
}

llvm::Expected<TargetModelManagedReferenceResult>
ManagedReferenceTargetModelBackend::execute(
    const TargetModelReduceRequest &request,
    FormalNumericWorkBudget scalarBudget) const {
  return executeManagedReference(request, scalarBudget, budget, environment);
}

} // namespace wafer::model
