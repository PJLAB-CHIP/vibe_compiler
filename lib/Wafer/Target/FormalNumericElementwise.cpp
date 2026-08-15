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
evaluateFormalElementwiseLLVM(const ResolvedNumericCommand &command,
                              llvm::ArrayRef<RawLogicalValue> inputs) {
  if (llvm::Error error = validateElementwiseResolvedCommand(command))
    return std::move(error);
  const NumericCTElementwiseCommand &elementwise =
      *command.getCommandKey().getCTElementwise();
  const NumericCTElementwiseSemanticsKey &key =
      *command.getSemantics()->getCTElementwiseKey();
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
        input, key.getInputFormat(),
        llvm::Twine("elementwise operand ") + llvm::Twine(index));
    if (!canonical)
      return canonical.takeError();
    canonicalInputs.push_back(*canonical);
  }

  const NumericElementwiseOperation operation = elementwise.operation;
  if (isNumericElementwiseLogic(operation)) {
    const bool lhs = canonicalInputs[0].bits != 0;
    bool result = false;
    switch (operation) {
    case NumericElementwiseOperation::LogicNot:
      result = !lhs;
      break;
    case NumericElementwiseOperation::LogicAnd:
      result = lhs && canonicalInputs[1].bits != 0;
      break;
    case NumericElementwiseOperation::LogicOr:
      result = lhs || canonicalInputs[1].bits != 0;
      break;
    case NumericElementwiseOperation::LogicXor:
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
  const LogicalFormat resultFormat = key.getDestinationFormat();
  const LogicalFormatDescriptor &resultDescriptor =
      *findLogicalFormatDescriptor(resultFormat);
  auto finishNaN = [&]() -> llvm::Expected<FormalNumericResult> {
    return finishRawResult(
        resultFormat, canonicalPositiveQuietNaNBits(resultDescriptor), flags);
  };

  if (isNumericElementwiseRelation(operation)) {
    bool result = false;
    if (hasNaN) {
      result = operation == NumericElementwiseOperation::Ne;
    } else {
      const llvm::APFloat::cmpResult comparison =
          floatingInputs[0].compare(floatingInputs[1]);
      switch (operation) {
      case NumericElementwiseOperation::Eq:
        result = comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Ne:
        result = comparison != llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Ge:
        result = comparison == llvm::APFloat::cmpGreaterThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Gt:
        result = comparison == llvm::APFloat::cmpGreaterThan;
        break;
      case NumericElementwiseOperation::Le:
        result = comparison == llvm::APFloat::cmpLessThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Lt:
        result = comparison == llvm::APFloat::cmpLessThan;
        break;
      default:
        llvm_unreachable("non-relation operation passed relation validator");
      }
    }
    return finishRawResult(LogicalFormat::Bool, result, flags);
  }

  if ((operation == NumericElementwiseOperation::Abs ||
       operation == NumericElementwiseOperation::Neg ||
       operation == NumericElementwiseOperation::Max ||
       operation == NumericElementwiseOperation::Min ||
       operation == NumericElementwiseOperation::Relu) &&
      hasNaN)
    return finishNaN();

  llvm::APFloat result = floatingInputs[0];
  llvm::APFloat::opStatus status = llvm::APFloat::opOK;
  switch (operation) {
  case NumericElementwiseOperation::Abs:
    result.clearSign();
    break;
  case NumericElementwiseOperation::Neg:
    result.changeSign();
    break;
  case NumericElementwiseOperation::Recip:
    result = llvm::APFloat::getOne(*getFloatSemantics(key.getInputFormat()));
    status =
        result.divide(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Square:
    status =
        result.multiply(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Max:
  case NumericElementwiseOperation::Min: {
    if (floatingInputs[0].isZero() && floatingInputs[1].isZero()) {
      const bool negative =
          operation == NumericElementwiseOperation::Max
              ? classifications[0].negative && classifications[1].negative
              : classifications[0].negative || classifications[1].negative;
      result = llvm::APFloat::getZero(*getFloatSemantics(key.getInputFormat()),
                                      negative);
      break;
    }
    const llvm::APFloat::cmpResult comparison =
        floatingInputs[0].compare(floatingInputs[1]);
    const bool chooseRhs = operation == NumericElementwiseOperation::Max
                               ? comparison == llvm::APFloat::cmpLessThan
                               : comparison == llvm::APFloat::cmpGreaterThan;
    if (chooseRhs)
      result = floatingInputs[1];
    break;
  }
  case NumericElementwiseOperation::Add:
    status = result.add(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Sub:
    status =
        result.subtract(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Mul:
    status =
        result.multiply(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Div:
    status =
        result.divide(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Relu:
    if (classifications[0].negative && !floatingInputs[0].isZero())
      result = llvm::APFloat::getZero(*getFloatSemantics(key.getInputFormat()));
    break;
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    llvm_unreachable("operation escaped the LLVM elementwise validator");
  }
  mergeFlags(flags, flagsFromStatus(status));

  if ((operation == NumericElementwiseOperation::Mul ||
       operation == NumericElementwiseOperation::Square) &&
      flags.inexact) {
    std::optional<ExactDyadic> lhs = decodeFiniteDyadic(canonicalInputs[0]);
    const RawLogicalValue rhsValue =
        operation == NumericElementwiseOperation::Square ? canonicalInputs[0]
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
