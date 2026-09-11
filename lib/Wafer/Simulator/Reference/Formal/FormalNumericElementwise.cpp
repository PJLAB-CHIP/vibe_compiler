//===- FormalNumericElementwise.cpp - Formal elementwise execution ---===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer {

using namespace formal_detail;

llvm::Expected<FormalNumericResult>
evaluateFormalElementwiseLLVM(const FormalElementwiseOperation &elementwise,
                              llvm::ArrayRef<RawLogicalValue> inputs) {
  if (llvm::Error error = validateFormalElementwiseOperation(elementwise))
    return std::move(error);
  const LogicalFormat inputFormat = elementwise.inputs.front().getFormat();
  if (inputs.size() != elementwise.inputs.size())
    return formalError(FormalNumericErrorCode::OperandCountMismatch,
                       llvm::Twine("elementwise operation expects ") +
                           llvm::Twine(elementwise.inputs.size()) +
                           " scalar operands, got " +
                           llvm::Twine(inputs.size()));

  std::vector<RawLogicalValue> canonicalInputs;
  canonicalInputs.reserve(inputs.size());
  for (auto [index, input] : llvm::enumerate(inputs)) {
    llvm::Expected<RawLogicalValue> canonical = validateOperand(
        input, inputFormat,
        llvm::Twine("elementwise operand ") + llvm::Twine(index));
    if (!canonical)
      return canonical.takeError();
    canonicalInputs.push_back(*canonical);
  }

  const TargetElementwiseOperation operation = elementwise.operation;
  if (isTargetElementwiseLogic(operation)) {
    const bool lhs = canonicalInputs[0].bits != 0;
    bool result = false;
    switch (operation) {
    case TargetElementwiseOperation::LogicNot:
      result = !lhs;
      break;
    case TargetElementwiseOperation::LogicAnd:
      result = lhs && canonicalInputs[1].bits != 0;
      break;
    case TargetElementwiseOperation::LogicOr:
      result = lhs || canonicalInputs[1].bits != 0;
      break;
    case TargetElementwiseOperation::LogicXor:
      result = lhs != (canonicalInputs[1].bits != 0);
      break;
    default:
      llvm_unreachable("non-logic operation passed the logic validator");
    }
    return finishRawResult(LogicalFormat::Bool, result, {});
  }

  std::vector<LogicalValueClassification> classifications;
  std::vector<llvm::APFloat> floatingInputs;
  classifications.reserve(canonicalInputs.size());
  floatingInputs.reserve(canonicalInputs.size());
  FormalNumericExceptionFlags flags;
  bool hasNaN = false;
  for (auto [index, input] : llvm::enumerate(canonicalInputs)) {
    llvm::Expected<LogicalValueClassification> classification = classifyOperand(
        input, llvm::Twine("elementwise operand ") + llvm::Twine(index));
    if (!classification)
      return classification.takeError();
    hasNaN |= isNaNClass(classification->valueClass);
    flags.invalid |=
        classification->valueClass == LogicalValueClass::SignalingNaN;
    classifications.push_back(*classification);
    floatingInputs.push_back(decodeFloat(input));
  }
  const LogicalFormat resultFormat = elementwise.destination.getFormat();
  const LogicalFormatDescriptor &resultDescriptor =
      *findLogicalFormatDescriptor(resultFormat);
  auto finishNaN = [&]() -> llvm::Expected<FormalNumericResult> {
    return finishRawResult(
        resultFormat, canonicalPositiveQuietNaNBits(resultDescriptor), flags);
  };

  if (isTargetElementwiseRelation(operation)) {
    bool result = false;
    if (hasNaN) {
      result = operation == TargetElementwiseOperation::Ne;
    } else {
      const llvm::APFloat::cmpResult comparison =
          floatingInputs[0].compare(floatingInputs[1]);
      switch (operation) {
      case TargetElementwiseOperation::Eq:
        result = comparison == llvm::APFloat::cmpEqual;
        break;
      case TargetElementwiseOperation::Ne:
        result = comparison != llvm::APFloat::cmpEqual;
        break;
      case TargetElementwiseOperation::Ge:
        result = comparison == llvm::APFloat::cmpGreaterThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case TargetElementwiseOperation::Gt:
        result = comparison == llvm::APFloat::cmpGreaterThan;
        break;
      case TargetElementwiseOperation::Le:
        result = comparison == llvm::APFloat::cmpLessThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case TargetElementwiseOperation::Lt:
        result = comparison == llvm::APFloat::cmpLessThan;
        break;
      default:
        llvm_unreachable("non-relation operation passed relation validator");
      }
    }
    return finishRawResult(LogicalFormat::Bool, result, flags);
  }

  if ((operation == TargetElementwiseOperation::Abs ||
       operation == TargetElementwiseOperation::Neg ||
       operation == TargetElementwiseOperation::Max ||
       operation == TargetElementwiseOperation::Min ||
       operation == TargetElementwiseOperation::Relu) &&
      hasNaN)
    return finishNaN();

  llvm::APFloat result = floatingInputs[0];
  llvm::APFloat::opStatus status = llvm::APFloat::opOK;
  switch (operation) {
  case TargetElementwiseOperation::Abs:
    result.clearSign();
    break;
  case TargetElementwiseOperation::Neg:
    result.changeSign();
    break;
  case TargetElementwiseOperation::Recip:
    result = llvm::APFloat::getOne(*getFloatSemantics(inputFormat));
    status =
        result.divide(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case TargetElementwiseOperation::Square:
    status =
        result.multiply(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case TargetElementwiseOperation::Max:
  case TargetElementwiseOperation::Min: {
    if (floatingInputs[0].isZero() && floatingInputs[1].isZero()) {
      const bool negative =
          operation == TargetElementwiseOperation::Max
              ? classifications[0].negative && classifications[1].negative
              : classifications[0].negative || classifications[1].negative;
      result =
          llvm::APFloat::getZero(*getFloatSemantics(inputFormat), negative);
      break;
    }
    const llvm::APFloat::cmpResult comparison =
        floatingInputs[0].compare(floatingInputs[1]);
    const bool chooseRhs = operation == TargetElementwiseOperation::Max
                               ? comparison == llvm::APFloat::cmpLessThan
                               : comparison == llvm::APFloat::cmpGreaterThan;
    if (chooseRhs)
      result = floatingInputs[1];
    break;
  }
  case TargetElementwiseOperation::Add:
    status = result.add(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case TargetElementwiseOperation::Sub:
    status =
        result.subtract(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case TargetElementwiseOperation::Mul:
    status =
        result.multiply(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case TargetElementwiseOperation::Relu:
    if (classifications[0].negative && !floatingInputs[0].isZero())
      result = llvm::APFloat::getZero(*getFloatSemantics(inputFormat));
    break;
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
  case TargetElementwiseOperation::Sqrt:
  case TargetElementwiseOperation::Rsqrt:
  case TargetElementwiseOperation::Log2:
  case TargetElementwiseOperation::Ln:
  case TargetElementwiseOperation::Pow2:
  case TargetElementwiseOperation::Exp:
  case TargetElementwiseOperation::ExpLp:
  case TargetElementwiseOperation::Sin:
  case TargetElementwiseOperation::Cos:
  case TargetElementwiseOperation::Tanh:
  case TargetElementwiseOperation::Sigmoid:
  case TargetElementwiseOperation::SatRelu:
  case TargetElementwiseOperation::LeakyRelu:
  case TargetElementwiseOperation::Softplus:
    llvm_unreachable("operation escaped the LLVM elementwise validator");
  }
  mergeFlags(flags, flagsFromStatus(status));

  if ((operation == TargetElementwiseOperation::Mul ||
       operation == TargetElementwiseOperation::Square) &&
      flags.inexact) {
    std::optional<ExactDyadic> lhs = decodeFiniteDyadic(canonicalInputs[0]);
    const RawLogicalValue rhsValue =
        operation == TargetElementwiseOperation::Square ? canonicalInputs[0]
                                                        : canonicalInputs[1];
    std::optional<ExactDyadic> rhs = decodeFiniteDyadic(rhsValue);
    if (lhs && rhs)
      flags.underflow |=
          isTinyAfterRNE(multiplyDyadics(*lhs, *rhs), resultDescriptor);
  }

  if (result.isNaN())
    return finishNaN();
  std::optional<uint64_t> bits = encodeFloat(result, resultFormat);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat elementwise result has an unexpected width");
  return finishRawResult(resultFormat, *bits, flags);
}

} // namespace wafer
