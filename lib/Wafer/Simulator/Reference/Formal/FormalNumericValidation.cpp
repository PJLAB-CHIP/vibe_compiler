//===- FormalNumericValidation.cpp - Family-local formal validation ------===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::formal_detail {
namespace {

bool isLLVMElementwiseOperation(TargetElementwiseOperation operation) {
  switch (operation) {
  case TargetElementwiseOperation::Abs:
  case TargetElementwiseOperation::Recip:
  case TargetElementwiseOperation::Square:
  case TargetElementwiseOperation::Neg:
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
  case TargetElementwiseOperation::LogicNot:
  case TargetElementwiseOperation::LogicAnd:
  case TargetElementwiseOperation::LogicOr:
  case TargetElementwiseOperation::LogicXor:
  case TargetElementwiseOperation::Relu:
    return true;
  default:
    return false;
  }
}

bool isFormalFloat(LogicalFormat format) {
  return format == LogicalFormat::F16 || format == LogicalFormat::BF16 ||
         format == LogicalFormat::F32;
}

} // namespace

llvm::Error validateFormalElementwiseOperation(
    const FormalElementwiseOperation &operation) {
  if (!isLLVMElementwiseOperation(operation.operation) ||
      operation.inputs.size() !=
          getTargetElementwiseArity(operation.operation) ||
      operation.inputs.empty())
    return formalError(FormalNumericErrorCode::UnsupportedOperation,
                       "elementwise operation is outside the formal subset");
  LogicalFormat inputFormat = operation.inputs.front().getFormat();
  if (llvm::any_of(operation.inputs, [&](const auto &input) {
        return input.getFormat() != inputFormat;
      }))
    return formalError(FormalNumericErrorCode::OperandFormatMismatch,
                       "elementwise inputs have different formats");
  const bool logic = isTargetElementwiseLogic(operation.operation);
  const bool relation = isTargetElementwiseRelation(operation.operation);
  LogicalFormat expectedDestination =
      relation ? LogicalFormat::Bool : inputFormat;
  if ((logic && inputFormat != LogicalFormat::Bool) ||
      (!logic && !isFormalFloat(inputFormat)) ||
      operation.destination.getFormat() != expectedDestination)
    return formalError(FormalNumericErrorCode::UnsupportedOperation,
                       "elementwise formats are outside the formal subset");
  return llvm::Error::success();
}

llvm::Error validateFormalGemmOperation(const FormalGemmOperation &operation) {
  if (!isFormalFloat(operation.lhs.getFormat()) ||
      operation.rhs.getFormat() != operation.lhs.getFormat() ||
      operation.destination.getFormat() != operation.lhs.getFormat())
    return formalError(FormalNumericErrorCode::UnsupportedOperation,
                       "GEMM formats are outside the formal subset");
  return llvm::Error::success();
}

llvm::Error
validateFormalReduceOperation(const FormalReduceOperation &operation) {
  if ((operation.operation != TargetReduceOperation::Sum &&
       operation.operation != TargetReduceOperation::Max &&
       operation.operation != TargetReduceOperation::Min) ||
      operation.input.getFormat() != LogicalFormat::F32 ||
      operation.destination.getFormat() != LogicalFormat::F32)
    return formalError(
        FormalNumericErrorCode::UnsupportedOperation,
        "reduction is outside the formal F32 sum/max/min subset");
  return llvm::Error::success();
}

} // namespace wafer::formal_detail
