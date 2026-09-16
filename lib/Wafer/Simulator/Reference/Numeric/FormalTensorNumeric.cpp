//===- FormalTensorNumeric.cpp - Atomic formal tensor execution ----------===//

#include "Wafer/Simulator/Reference/FormalTensorNumeric.h"

#include "Wafer/Simulator/Reference/MPFRNumeric.h"

#include "Wafer/Simulator/Reference/Formal/FormalNumericInternal.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Parallel.h"

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

struct ValidatedFormalOperation {
  llvm::SmallVector<const PhysicalTensorDescriptor *, 2> inputKeys;
  const PhysicalTensorDescriptor *destinationKey = nullptr;
  uint64_t scalarEvaluations = 0;
  uint64_t fusedMultiplyAdds = 0;
};

llvm::Error validateBudget(const ValidatedFormalOperation &operation,
                           FormalNumericWorkBudget budget) {
  if (operation.scalarEvaluations > budget.getMaximumScalarEvaluations())
    return tensorError(
        FormalTensorNumericErrorCode::ScalarWorkBudgetExceeded,
        llvm::Twine("formal operation requires ") +
            llvm::Twine(operation.scalarEvaluations) +
            " scalar evaluations but the invocation budget allows " +
            llvm::Twine(budget.getMaximumScalarEvaluations()));
  if (operation.fusedMultiplyAdds > budget.getMaximumFusedMultiplyAdds())
    return tensorError(
        FormalTensorNumericErrorCode::MultiplyAccumulateWorkBudgetExceeded,
        llvm::Twine("formal operation requires ") +
            llvm::Twine(operation.fusedMultiplyAdds) +
            " fused multiply-adds but the invocation budget allows " +
            llvm::Twine(budget.getMaximumFusedMultiplyAdds()));
  return llvm::Error::success();
}

llvm::Expected<ValidatedFormalOperation>
validateFormalOperation(const FormalConvertOperation &operation,
                        FormalNumericWorkBudget budget) {
  ValidatedFormalOperation result;
  result.inputKeys.push_back(&operation.source);
  result.destinationKey = &operation.destination;
  result.scalarEvaluations = operation.destination.getElementCount();
  if (llvm::Error error = validateBudget(result, budget))
    return std::move(error);
  return result;
}

llvm::Expected<ValidatedFormalOperation>
validateFormalOperation(const FormalElementwiseOperation &operation,
                        FormalNumericWorkBudget budget) {
  ValidatedFormalOperation result;
  for (const PhysicalTensorDescriptor &input : operation.inputs)
    result.inputKeys.push_back(&input);
  result.destinationKey = &operation.destination;
  result.scalarEvaluations = operation.destination.getElementCount();
  if (llvm::Error error = validateBudget(result, budget))
    return std::move(error);
  return result;
}

llvm::Expected<ValidatedFormalOperation>
validateFormalOperation(const FormalGemmOperation &operation,
                        FormalNumericWorkBudget budget) {
  if (llvm::Error error = formal_detail::validateFormalGemmOperation(operation))
    return std::move(error);
  ValidatedFormalOperation result;
  result.inputKeys.push_back(&operation.lhs);
  result.inputKeys.push_back(&operation.rhs);
  if (operation.psum)
    result.inputKeys.push_back(&*operation.psum);
  result.destinationKey = &operation.destination;
  uint64_t outputCount = 0;
  if (!checkedMultiply(operation.batchCount, operation.m, outputCount) ||
      !checkedMultiply(outputCount, operation.n, outputCount) ||
      outputCount != operation.destination.getElementCount() ||
      !checkedMultiply(outputCount, operation.k, result.fusedMultiplyAdds) ||
      !checkedAdd(result.fusedMultiplyAdds, outputCount,
                  result.scalarEvaluations))
    return tensorError(FormalTensorNumericErrorCode::WorkCountOverflow,
                       "NE GEMM formal work count is inconsistent or "
                       "overflows uint64_t");
  if (operation.psum && !checkedAdd(result.scalarEvaluations, outputCount,
                                    result.scalarEvaluations))
    return tensorError(FormalTensorNumericErrorCode::WorkCountOverflow,
                       "GEMM psum work count overflows");
  if (llvm::Error error = validateBudget(result, budget))
    return std::move(error);
  return result;
}

llvm::Expected<ValidatedFormalOperation>
validateFormalOperation(const FormalReduceOperation &operation,
                        FormalNumericWorkBudget budget) {
  if (llvm::Error error =
          formal_detail::validateFormalReduceOperation(operation))
    return std::move(error);
  ValidatedFormalOperation result;
  result.inputKeys.push_back(&operation.input);
  result.destinationKey = &operation.destination;
  result.scalarEvaluations = operation.input.getElementCount();
  if (llvm::Error error = validateBudget(result, budget))
    return std::move(error);
  return result;
}

llvm::Error
validateInputs(const ValidatedFormalOperation &validatedOperation,
               llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs) {
  if (inputs.size() != validatedOperation.inputKeys.size())
    return tensorError(FormalTensorNumericErrorCode::InputArityMismatch,
                       llvm::Twine("expected ") +
                           llvm::Twine(validatedOperation.inputKeys.size()) +
                           " input tensors, got " + llvm::Twine(inputs.size()));

  for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
    const PhysicalTensorDescriptor &key =
        *validatedOperation.inputKeys[inputIndex];
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
getMPFROperation(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::Sqrt:
    return MPFRFormalOperation::Sqrt;
  case TargetElementwiseOperation::Rsqrt:
    return MPFRFormalOperation::Rsqrt;
  case TargetElementwiseOperation::Log2:
    return MPFRFormalOperation::Log2;
  case TargetElementwiseOperation::Ln:
    return MPFRFormalOperation::Ln;
  case TargetElementwiseOperation::Pow2:
    return MPFRFormalOperation::Pow2;
  case TargetElementwiseOperation::Exp:
    return MPFRFormalOperation::Exp;
  case TargetElementwiseOperation::Sin:
    return MPFRFormalOperation::Sin;
  case TargetElementwiseOperation::Cos:
    return MPFRFormalOperation::Cos;
  case TargetElementwiseOperation::Tanh:
    return MPFRFormalOperation::Tanh;
  case TargetElementwiseOperation::Sigmoid:
    return MPFRFormalOperation::Sigmoid;
  case TargetElementwiseOperation::Softplus:
    return MPFRFormalOperation::Softplus;
  default:
    return std::nullopt;
  }
}

llvm::Expected<FormalNumericResult>
evaluateElementwise(const FormalElementwiseOperation &operation,
                    llvm::ArrayRef<RawLogicalValue> inputs) {
  std::optional<MPFRFormalOperation> mpfrOperation =
      getMPFROperation(operation.operation);
  if (!mpfrOperation)
    return evaluateFormalElementwiseLLVM(operation, inputs);
  if (inputs.size() != 1)
    return tensorError(FormalTensorNumericErrorCode::InputArityMismatch,
                       "MPFR elementwise operation requires one input");
  return executeMPFRFormal({*mpfrOperation, TargetRoundingMode::NearestEven,
                            operation.destination.getFormat(), inputs.front()});
}

llvm::Expected<FormalTensorNumericResult>
executeConvert(const FormalConvertOperation &operation,
               llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
               uint64_t outputCount) {
  FormalTensorNumericResult result;
  result.values.reserve(static_cast<size_t>(outputCount));
  for (RawLogicalValue source : inputs.front()) {
    llvm::Expected<FormalNumericResult> scalar =
        evaluateFormalConvert(operation, source);
    if (!scalar)
      return scalar.takeError();
    result.values.push_back(scalar->value);
    mergeFlags(result.flags, scalar->flags);
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeElementwise(const FormalElementwiseOperation &operation,
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
        evaluateElementwise(operation, scalarInputs);
    if (!scalar)
      return scalar.takeError();
    result.values.push_back(scalar->value);
    mergeFlags(result.flags, scalar->flags);
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeGemm(const FormalGemmOperation &gemm,
            llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs, uint64_t) {
  FormalTensorNumericResult result;
  const uint64_t outputCount = gemm.destination.getElementCount();
  result.values.resize(static_cast<size_t>(outputCount));
  llvm::ArrayRef<RawLogicalValue> lhs = inputs[0];
  llvm::ArrayRef<RawLogicalValue> rhs = inputs[1];

  std::optional<FormalElementwiseOperation> accumulation;
  if (gemm.psum) {
    auto operation = createFormalElementwiseOperation(
        TargetElementwiseOperation::Add, {*gemm.psum, *gemm.psum}, *gemm.psum);
    if (!operation)
      return operation.takeError();
    accumulation = std::move(*operation);
  }
  struct ElementResult {
    FormalNumericExceptionFlags flags;
    std::optional<llvm::Error> error;
  };
  std::vector<ElementResult> elements(static_cast<size_t>(outputCount));
  auto evaluate = [&](size_t index) -> llvm::Error {
    const uint64_t n = index % gemm.n;
    const uint64_t m = (index / gemm.n) % gemm.m;
    const uint64_t batch = index / (uint64_t(gemm.m) * gemm.n);
    const uint64_t lhsBatchBase = batch * gemm.m * gemm.k;
    const uint64_t rhsBatchBase = batch * gemm.k * gemm.n;
    FormalNumericExceptionFlags flags;
    RawLogicalValue accumulator{LogicalFormat::F32, UINT64_C(0)};
    for (uint64_t k = 0; k < gemm.k; ++k) {
      const uint64_t lhsIndex =
          lhsBatchBase + (gemm.lhsOrientation == TargetGemmOrientation::Normal
                              ? m * gemm.k + k
                              : k * gemm.m + m);
      const uint64_t rhsIndex =
          rhsBatchBase + (gemm.rhsOrientation == TargetGemmOrientation::Normal
                              ? k * gemm.n + n
                              : n * gemm.k + k);
      llvm::Expected<FormalNumericResult> step =
          evaluateFormalGemmFusedMultiplyAdd(
              gemm, lhs[static_cast<size_t>(lhsIndex)],
              rhs[static_cast<size_t>(rhsIndex)], accumulator);
      if (!step)
        return step.takeError();
      accumulator = step->value;
      mergeFlags(flags, step->flags);
    }
    if (gemm.psum) {
      const auto partial = inputs[2][batch * gemm.m * gemm.n + m * gemm.n + n];
      auto sum = evaluateFormalElementwiseLLVM(
          *accumulation, llvm::ArrayRef<RawLogicalValue>{accumulator, partial});
      if (!sum)
        return sum.takeError();
      accumulator = sum->value;
      mergeFlags(flags, sum->flags);
    }
    llvm::Expected<FormalNumericResult> destination =
        evaluateFormalGemmFinalize(gemm, accumulator);
    if (!destination)
      return destination.takeError();
    result.values[index] = destination->value;
    mergeFlags(flags, destination->flags);
    elements[index].flags = flags;
    return llvm::Error::success();
  };
  auto work = [&](size_t index) {
    elements[index].error.emplace(evaluate(index));
  };
  // This threshold only amortizes host scheduling. The validated work budget
  // and numeric algorithm are identical on the serial and parallel paths.
  if (outputCount >= 128 && outputCount * gemm.k >= 65536)
    llvm::parallelFor(0, static_cast<size_t>(outputCount), work);
  else
    for (size_t index = 0; index < outputCount; ++index)
      work(index);
  llvm::Error firstError = llvm::Error::success();
  for (auto &element : elements) {
    if (*element.error) {
      if (!firstError)
        firstError = std::move(*element.error);
      else
        llvm::consumeError(std::move(*element.error));
    } else {
      mergeFlags(result.flags, element.flags);
    }
  }
  if (firstError)
    return std::move(firstError);
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeReduce(const FormalReduceOperation &reduce,
              llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
              uint64_t) {
  const llvm::ArrayRef<uint64_t> inputShape = reduce.input.getShape();
  const std::vector<size_t> reducedDimensions =
      getTargetReduceLogicalDimensions(reduce.dimension, inputShape.size());
  if (reducedDimensions.empty())
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "native reduction has no logical dimensions");

  const uint64_t outputCount = reduce.destination.getElementCount();
  FormalTensorNumericResult result;
  uint64_t identity =
      reduce.operation == TargetReduceOperation::Max   ? UINT64_C(0xff800000)
      : reduce.operation == TargetReduceOperation::Min ? UINT64_C(0x7f800000)
                                                       : UINT64_C(0);
  result.values.assign(static_cast<size_t>(outputCount),
                       RawLogicalValue{LogicalFormat::F32, identity});
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
        reduce, result.values[static_cast<size_t>(destinationIndex)],
        inputs.front()[static_cast<size_t>(inputIndex)]);
    if (!step)
      return step.takeError();
    result.values[static_cast<size_t>(destinationIndex)] = step->value;
    mergeFlags(result.flags, step->flags);
  }
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalConvertOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount);
llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalElementwiseOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount);
llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalGemmOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount);
llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalReduceOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount);

template <typename Operation>
llvm::Expected<FormalTensorNumericResult>
executeCheckedOperation(FormalNumericExecutionContext &context,
                        const Operation &operation,
                        llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                        FormalNumericWorkBudget budget) {
  llvm::Expected<ValidatedFormalOperation> validated =
      validateFormalOperation(operation, budget);
  if (!validated)
    return validated.takeError();
  if (llvm::Error error = validateInputs(*validated, inputs))
    return std::move(error);

  const uint64_t outputCount = validated->destinationKey->getElementCount();
  if (outputCount > std::numeric_limits<size_t>::max())
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "destination element count does not fit host size_t");
  llvm::Expected<FormalTensorNumericResult> result =
      executeOperation(operation, inputs, outputCount);
  if (!result)
    return result.takeError();
  if (result->values.size() != outputCount)
    return tensorError(FormalTensorNumericErrorCode::ResultInvariantViolation,
                       "formal operation produced the wrong output element "
                       "count");
  context.mergeExceptionFlags(result->flags);
  return result;
}

llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalConvertOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount) {
  return executeConvert(operation, inputs, outputCount);
}

llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalElementwiseOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount) {
  return executeElementwise(operation, inputs, outputCount);
}

llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalGemmOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount) {
  return executeGemm(operation, inputs, outputCount);
}

llvm::Expected<FormalTensorNumericResult>
executeOperation(const FormalReduceOperation &operation,
                 llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
                 uint64_t outputCount) {
  return executeReduce(operation, inputs, outputCount);
}

} // namespace

llvm::StringRef
stringifyFormalTensorNumericErrorCode(FormalTensorNumericErrorCode code) {
  switch (code) {
  case FormalTensorNumericErrorCode::UnsupportedOperation:
    return "unsupported-operation";
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
    const FormalConvertOperation &operation,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget) {
  return executeCheckedOperation(context, operation, inputs, budget);
}

llvm::Expected<FormalTensorNumericResult> executeFormalTensorNumeric(
    FormalNumericExecutionContext &context,
    const FormalElementwiseOperation &operation,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget) {
  return executeCheckedOperation(context, operation, inputs, budget);
}

llvm::Expected<FormalTensorNumericResult> executeFormalTensorNumeric(
    FormalNumericExecutionContext &context,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget) {
  return executeCheckedOperation(context, operation, inputs, budget);
}

llvm::Expected<FormalTensorNumericResult> executeFormalTensorNumeric(
    FormalNumericExecutionContext &context,
    const FormalReduceOperation &operation,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget) {
  return executeCheckedOperation(context, operation, inputs, budget);
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
