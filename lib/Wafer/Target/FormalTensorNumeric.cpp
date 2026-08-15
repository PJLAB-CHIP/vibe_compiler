//===- FormalTensorNumeric.cpp - Atomic formal tensor execution ----------===//

#include "Wafer/Target/FormalTensorNumeric.h"

#include "Wafer/Target/MPFRNumeric.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace wafer {
namespace {

llvm::Error tensorError(FormalTensorNumericErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<FormalTensorNumericError>(code, detail.str());
}

void mergeFlags(FormalNumericExceptionFlags &destination,
                FormalNumericExceptionFlags source) {
  destination.invalid |= source.invalid;
  destination.divByZero |= source.divByZero;
  destination.overflow |= source.overflow;
  destination.underflow |= source.underflow;
  destination.inexact |= source.inexact;
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

struct ValidatedFormalCommand {
  llvm::SmallVector<const NumericTensorKey *, 2> inputKeys;
  const NumericTensorKey *destinationKey = nullptr;
  uint64_t scalarEvaluations = 0;
  uint64_t fusedMultiplyAdds = 0;
};

llvm::Expected<ValidatedFormalCommand>
validateFormalCommand(const ResolvedNumericCommand &command,
                      FormalNumericWorkBudget budget) {
  if (!command.isSupported() || !command.getSemantics() ||
      command.getComparatorKind() != NumericComparatorKind::RawExact)
    return tensorError(
        FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
        "the resolved command has no complete formal execution identity");

  ValidatedFormalCommand result;
  switch (command.getFamily()) {
  case NumericCommandFamily::CTConvert: {
    const NumericCTConvertCommand *convert =
        command.getCommandKey().getCTConvert();
    if (!convert ||
        command.getFormalKernelKind() != FormalKernelKind::Convert ||
        command.getFormalBackendKind() !=
            FormalNumericBackendKind::LLVMAPFloatAPInt)
      return tensorError(
          FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
          "the resolved CT convert command lost its formal kernel identity");
    result.inputKeys.push_back(&convert->source);
    result.destinationKey = &convert->destination;
    result.scalarEvaluations = convert->destination.getElementCount();
    break;
  }
  case NumericCommandFamily::CTElementwise: {
    const NumericCTElementwiseCommand *elementwise =
        command.getCommandKey().getCTElementwise();
    if (!elementwise ||
        command.getFormalKernelKind() != FormalKernelKind::Elementwise ||
        !command.getFormalBackendKind())
      return tensorError(
          FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
          "the resolved CT elementwise command lost its formal kernel "
          "identity");
    for (const NumericTensorKey &input : elementwise->inputs)
      result.inputKeys.push_back(&input);
    result.destinationKey = &elementwise->destination;
    result.scalarEvaluations = elementwise->destination.getElementCount();
    break;
  }
  case NumericCommandFamily::NEGemm: {
    const NumericNEGemmCommand *gemm = command.getCommandKey().getNEGemm();
    if (!gemm || command.getFormalKernelKind() != FormalKernelKind::Gemm ||
        command.getFormalBackendKind() !=
            FormalNumericBackendKind::LLVMAPFloatAPInt)
      return tensorError(
          FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
          "the resolved NE GEMM command lost its formal kernel identity");
    result.inputKeys.push_back(&gemm->lhs);
    result.inputKeys.push_back(&gemm->rhs);
    result.destinationKey = &gemm->destination;
    uint64_t outputCount = 0;
    if (!checkedMultiply(gemm->batchCount, gemm->m, outputCount) ||
        !checkedMultiply(outputCount, gemm->n, outputCount) ||
        outputCount != gemm->destination.getElementCount() ||
        !checkedMultiply(outputCount, gemm->k, result.fusedMultiplyAdds) ||
        !checkedAdd(result.fusedMultiplyAdds, outputCount,
                    result.scalarEvaluations))
      return tensorError(FormalTensorNumericErrorCode::WorkCountOverflow,
                         "NE GEMM formal work count is inconsistent or "
                         "overflows uint64_t");
    break;
  }
  case NumericCommandFamily::NativeCTReduce: {
    const NumericNativeCTReduceCommand *reduce =
        command.getCommandKey().getNativeCTReduce();
    if (!reduce || command.getFormalKernelKind() != FormalKernelKind::Reduce ||
        command.getFormalBackendKind() !=
            FormalNumericBackendKind::LLVMAPFloatAPInt)
      return tensorError(
          FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
          "the resolved native reduction lost its formal kernel identity");
    result.inputKeys.push_back(&reduce->input);
    result.destinationKey = &reduce->destination;
    result.scalarEvaluations = reduce->input.getElementCount();
    break;
  }
  }

  if (!result.destinationKey)
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "formal command validation found no destination tensor");
  if (result.scalarEvaluations > budget.getMaximumScalarEvaluations())
    return tensorError(
        FormalTensorNumericErrorCode::ScalarWorkBudgetExceeded,
        llvm::Twine("formal command requires ") +
            llvm::Twine(result.scalarEvaluations) +
            " scalar evaluations but the invocation budget allows " +
            llvm::Twine(budget.getMaximumScalarEvaluations()));
  if (result.fusedMultiplyAdds > budget.getMaximumFusedMultiplyAdds())
    return tensorError(
        FormalTensorNumericErrorCode::MultiplyAccumulateWorkBudgetExceeded,
        llvm::Twine("formal command requires ") +
            llvm::Twine(result.fusedMultiplyAdds) +
            " fused multiply-adds but the invocation budget allows " +
            llvm::Twine(budget.getMaximumFusedMultiplyAdds()));
  return result;
}

llvm::Error
validateInputs(const ValidatedFormalCommand &validatedCommand,
               llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs) {
  if (inputs.size() != validatedCommand.inputKeys.size())
    return tensorError(FormalTensorNumericErrorCode::InputArityMismatch,
                       llvm::Twine("expected ") +
                           llvm::Twine(validatedCommand.inputKeys.size()) +
                           " input tensors, got " + llvm::Twine(inputs.size()));

  for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
    const NumericTensorKey &key = *validatedCommand.inputKeys[inputIndex];
    llvm::ArrayRef<RawLogicalValue> values = inputs[inputIndex];
    if (values.size() != key.getElementCount())
      return tensorError(
          FormalTensorNumericErrorCode::InputElementCountMismatch,
          llvm::Twine("input ") + llvm::Twine(inputIndex) + " expects " +
              llvm::Twine(key.getElementCount()) + " logical elements, got " +
              llvm::Twine(values.size()));
    for (size_t elementIndex = 0; elementIndex < values.size();
         ++elementIndex) {
      if (values[elementIndex].format != key.getFormat())
        return tensorError(
            FormalTensorNumericErrorCode::InputFormatMismatch,
            llvm::Twine("input ") + llvm::Twine(inputIndex) + " element " +
                llvm::Twine(elementIndex) + " has format " +
                stringifyLogicalFormat(values[elementIndex].format) +
                " but the command requires " +
                stringifyLogicalFormat(key.getFormat()));
      llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
          values[elementIndex].format, values[elementIndex].bits,
          NonCanonicalEncodingPolicy::Reject);
      if (!canonical)
        return tensorError(FormalTensorNumericErrorCode::InvalidInputEncoding,
                           llvm::Twine("input ") + llvm::Twine(inputIndex) +
                               " element " + llvm::Twine(elementIndex) + ": " +
                               llvm::toString(canonical.takeError()));
    }
  }
  return llvm::Error::success();
}

std::optional<MPFRFormalOperation>
getMPFROperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Sqrt:
    return MPFRFormalOperation::Sqrt;
  case NumericElementwiseOperation::Rsqrt:
    return MPFRFormalOperation::Rsqrt;
  case NumericElementwiseOperation::Log2:
    return MPFRFormalOperation::Log2;
  case NumericElementwiseOperation::Ln:
    return MPFRFormalOperation::Ln;
  case NumericElementwiseOperation::Pow2:
    return MPFRFormalOperation::Pow2;
  case NumericElementwiseOperation::Exp:
    return MPFRFormalOperation::Exp;
  case NumericElementwiseOperation::Sin:
    return MPFRFormalOperation::Sin;
  case NumericElementwiseOperation::Cos:
    return MPFRFormalOperation::Cos;
  case NumericElementwiseOperation::Tanh:
    return MPFRFormalOperation::Tanh;
  case NumericElementwiseOperation::Sigmoid:
    return MPFRFormalOperation::Sigmoid;
  case NumericElementwiseOperation::Softplus:
    return MPFRFormalOperation::Softplus;
  default:
    return std::nullopt;
  }
}

llvm::Expected<FormalNumericResult>
evaluateElementwise(const ResolvedNumericCommand &command,
                    llvm::ArrayRef<RawLogicalValue> inputs) {
  if (command.getFormalBackendKind() ==
      FormalNumericBackendKind::LLVMAPFloatAPInt)
    return evaluateFormalElementwiseLLVM(command, inputs);
  if (command.getFormalBackendKind() != FormalNumericBackendKind::MPFR)
    return tensorError(FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
                       "elementwise command has an unknown formal backend");

  const NumericCTElementwiseCommand *elementwise =
      command.getCommandKey().getCTElementwise();
  const NumericSemanticsProfile *semantics = command.getSemantics();
  if (!elementwise || !semantics || inputs.size() != 1 ||
      semantics->getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::
              CorrectlyRoundedMathematicalResultAdaptiveMPFRFinalRNE ||
      semantics->getRoundingModePolicy() != NumericRoundingMode::NearestEven)
    return tensorError(
        FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
        "MPFR elementwise command has an incomplete adaptive-RNE policy");
  std::optional<MPFRFormalOperation> operation =
      getMPFROperation(elementwise->operation);
  if (!operation)
    return tensorError(FormalTensorNumericErrorCode::UnsupportedResolvedCommand,
                       "elementwise operation is not an MPFR formal function");
  return executeMPFRFormal({*operation, NumericRoundingMode::NearestEven,
                            elementwise->destination.getFormat(),
                            inputs.front()});
}

llvm::Expected<FormalTensorNumericResult>
executeConvert(const ResolvedNumericCommand &command,
               llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
               uint64_t outputCount) {
  FormalTensorNumericResult result;
  result.values.reserve(static_cast<size_t>(outputCount));
  for (RawLogicalValue source : inputs.front()) {
    llvm::Expected<FormalNumericResult> scalar =
        evaluateFormalConvert(command, source);
    if (!scalar)
      return scalar.takeError();
    result.values.push_back(scalar->value);
    mergeFlags(result.flags, scalar->flags);
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeElementwise(const ResolvedNumericCommand &command,
                   llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                   uint64_t outputCount) {
  FormalTensorNumericResult result;
  result.values.reserve(static_cast<size_t>(outputCount));
  llvm::SmallVector<RawLogicalValue, 2> scalarInputs;
  for (uint64_t elementIndex = 0; elementIndex < outputCount; ++elementIndex) {
    scalarInputs.clear();
    for (llvm::ArrayRef<RawLogicalValue> input : inputs)
      scalarInputs.push_back(input[static_cast<size_t>(elementIndex)]);
    llvm::Expected<FormalNumericResult> scalar =
        evaluateElementwise(command, scalarInputs);
    if (!scalar)
      return scalar.takeError();
    result.values.push_back(scalar->value);
    mergeFlags(result.flags, scalar->flags);
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeGemm(const ResolvedNumericCommand &command,
            llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
            const NumericNEGemmCommand &gemm) {
  FormalTensorNumericResult result;
  const uint64_t outputCount = gemm.destination.getElementCount();
  result.values.reserve(static_cast<size_t>(outputCount));
  llvm::ArrayRef<RawLogicalValue> lhs = inputs[0];
  llvm::ArrayRef<RawLogicalValue> rhs = inputs[1];

  for (uint64_t batch = 0; batch < gemm.batchCount; ++batch) {
    const uint64_t lhsBatchBase = batch * gemm.m * gemm.k;
    const uint64_t rhsBatchBase = batch * gemm.k * gemm.n;
    for (uint64_t m = 0; m < gemm.m; ++m) {
      for (uint64_t n = 0; n < gemm.n; ++n) {
        RawLogicalValue accumulator{LogicalFormat::F32, UINT64_C(0)};
        for (uint64_t k = 0; k < gemm.k; ++k) {
          const uint64_t lhsIndex =
              lhsBatchBase +
              (gemm.lhsOrientation == TargetGemmOrientation::Normal
                   ? m * gemm.k + k
                   : k * gemm.m + m);
          const uint64_t rhsIndex =
              rhsBatchBase +
              (gemm.rhsOrientation == TargetGemmOrientation::Normal
                   ? k * gemm.n + n
                   : n * gemm.k + k);
          llvm::Expected<FormalNumericResult> step =
              evaluateFormalGemmFusedMultiplyAdd(
                  command, lhs[static_cast<size_t>(lhsIndex)],
                  rhs[static_cast<size_t>(rhsIndex)], accumulator);
          if (!step)
            return step.takeError();
          accumulator = step->value;
          mergeFlags(result.flags, step->flags);
        }
        llvm::Expected<FormalNumericResult> destination =
            evaluateFormalGemmFinalize(command, accumulator);
        if (!destination)
          return destination.takeError();
        result.values.push_back(destination->value);
        mergeFlags(result.flags, destination->flags);
      }
    }
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeReduce(const ResolvedNumericCommand &command,
              llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
              const NumericNativeCTReduceCommand &reduce) {
  const llvm::ArrayRef<uint64_t> inputShape = reduce.input.getShape();
  const std::vector<size_t> reducedDimensions =
      getNativeCTReduceLogicalDimensions(reduce.dimension, inputShape.size());
  if (reducedDimensions.empty())
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "native reduction has no logical dimensions");

  const uint64_t outputCount = reduce.destination.getElementCount();
  FormalTensorNumericResult result;
  result.values.assign(static_cast<size_t>(outputCount),
                       RawLogicalValue{LogicalFormat::F32, UINT64_C(0)});
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
      if (reduced[dimension])
        continue;
      destinationIndex =
          destinationIndex * inputShape[dimension] + coordinates[dimension];
    }
    if (destinationIndex >= outputCount)
      return tensorError(
          FormalTensorNumericErrorCode::ResultInvariantViolation,
          "native reduction mapped an input outside the destination tensor");
    llvm::Expected<FormalNumericResult> step = evaluateFormalReduceStep(
        command, result.values[static_cast<size_t>(destinationIndex)],
        inputs.front()[static_cast<size_t>(inputIndex)]);
    if (!step)
      return step.takeError();
    result.values[static_cast<size_t>(destinationIndex)] = step->value;
    mergeFlags(result.flags, step->flags);
  }
  return result;
}

} // namespace

llvm::StringRef
stringifyFormalTensorNumericErrorCode(FormalTensorNumericErrorCode code) {
  switch (code) {
  case FormalTensorNumericErrorCode::UnsupportedResolvedCommand:
    return "unsupported-resolved-command";
  case FormalTensorNumericErrorCode::InputArityMismatch:
    return "input-arity-mismatch";
  case FormalTensorNumericErrorCode::InputElementCountMismatch:
    return "input-element-count-mismatch";
  case FormalTensorNumericErrorCode::InputFormatMismatch:
    return "input-format-mismatch";
  case FormalTensorNumericErrorCode::InvalidInputEncoding:
    return "invalid-input-encoding";
  case FormalTensorNumericErrorCode::ScalarWorkBudgetExceeded:
    return "scalar-work-budget-exceeded";
  case FormalTensorNumericErrorCode::MultiplyAccumulateWorkBudgetExceeded:
    return "multiply-accumulate-work-budget-exceeded";
  case FormalTensorNumericErrorCode::WorkCountOverflow:
    return "work-count-overflow";
  case FormalTensorNumericErrorCode::ResultInvariantViolation:
    return "result-invariant-violation";
  }
  llvm_unreachable("formal tensor numeric error code is not registered");
}

char FormalTensorNumericError::ID;

void FormalTensorNumericError::log(llvm::raw_ostream &stream) const {
  stream << "formal tensor numeric "
         << stringifyFormalTensorNumericErrorCode(code) << ": " << detail;
}

std::error_code FormalTensorNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<FormalTensorNumericResult> executeFormalTensorNumeric(
    FormalNumericExecutionContext &context,
    const ResolvedNumericCommand &command,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget) {
  llvm::Expected<ValidatedFormalCommand> validatedCommand =
      validateFormalCommand(command, budget);
  if (!validatedCommand)
    return validatedCommand.takeError();
  if (llvm::Error error = validateInputs(*validatedCommand, inputs))
    return std::move(error);

  const uint64_t outputCount =
      validatedCommand->destinationKey->getElementCount();
  if (outputCount > std::numeric_limits<size_t>::max())
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "destination element count does not fit host size_t");

  llvm::Expected<FormalTensorNumericResult> result =
      [&]() -> llvm::Expected<FormalTensorNumericResult> {
    switch (command.getFamily()) {
    case NumericCommandFamily::CTConvert:
      return executeConvert(command, inputs, outputCount);
    case NumericCommandFamily::CTElementwise:
      return executeElementwise(command, inputs, outputCount);
    case NumericCommandFamily::NEGemm:
      return executeGemm(command, inputs, *command.getCommandKey().getNEGemm());
    case NumericCommandFamily::NativeCTReduce:
      return executeReduce(command, inputs,
                           *command.getCommandKey().getNativeCTReduce());
    }
    llvm_unreachable("numeric command family is not registered");
  }();
  if (!result)
    return result.takeError();
  if (result->values.size() != outputCount)
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "formal kernel produced the wrong output element count");
  context.mergeExceptionFlags(result->flags);
  return result;
}

llvm::Expected<bool>
compareFormalTensorNumericResultsExact(const FormalTensorNumericResult &lhs,
                                       const FormalTensorNumericResult &rhs) {
  auto validateValues =
      [](llvm::ArrayRef<RawLogicalValue> values) -> llvm::Error {
    for (RawLogicalValue value : values) {
      llvm::Expected<bool> valid =
          compareFormalNumericResultsExact({value, {}}, {value, {}});
      if (!valid)
        return valid.takeError();
      if (!*valid)
        llvm_unreachable("a validated raw value must compare equal to itself");
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validateValues(lhs.values))
    return std::move(error);
  if (llvm::Error error = validateValues(rhs.values))
    return std::move(error);

  if (lhs.values.size() != rhs.values.size() || lhs.flags != rhs.flags)
    return false;
  for (size_t index = 0; index < lhs.values.size(); ++index) {
    if (lhs.values[index].format != rhs.values[index].format ||
        lhs.values[index].bits != rhs.values[index].bits)
      return false;
  }
  return true;
}

} // namespace wafer
