//===- TargetModelKernel.cpp - Plain target kernel dispatch -----------===//

#include "TargetModelKernelInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>
#include <variant>

namespace wafer::model {

llvm::Expected<TargetModelCommandEffect>
executeTargetModelCommand(const compiler::TargetTransaction &transaction,
                          const InvocationMemoryRegistry &memory,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy) {
  if (llvm::Error error = validateTargetModelTransactionFields(transaction))
    return std::move(error);
  if (!llvm::is_contained(memory.getAddressPlan().getLogicalRanks(),
                          transaction.logicalRank))
    return kernel_detail::kernelError(
        TargetModelKernelErrorCode::InvalidTransactionField,
        "transaction logical rank is outside invocation");

  if (const auto *value = std::get_if<compiler::TargetStridedDMATransaction>(
          &transaction.payload))
    return kernel_detail::executeMovement(transaction, *value, memory, budget);
  if (const auto *value = std::get_if<compiler::TargetGatherScatterTransaction>(
          &transaction.payload))
    return kernel_detail::executeGatherScatter(transaction, *value, memory,
                                               budget);
  if (const auto *value =
          std::get_if<compiler::TargetMemsetTransaction>(&transaction.payload))
    return kernel_detail::executeMemset(transaction, *value, memory);
  if (const auto *value = std::get_if<compiler::TargetElementwiseTransaction>(
          &transaction.payload))
    return kernel_detail::executeElementwise(transaction, *value, memory,
                                             budget);
  if (const auto *value =
          std::get_if<compiler::TargetConvertTransaction>(&transaction.payload))
    return kernel_detail::executeConvert(transaction, *value, memory, budget);
  if (const auto *value =
          std::get_if<compiler::TargetGemmTransaction>(&transaction.payload))
    return kernel_detail::executeGemm(transaction, *value, memory, budget,
                                      policy);

  TargetModelControlAction control =
      kernel_detail::getControlAction(transaction.payload);
  if (control != TargetModelControlAction::None) {
    if (llvm::Error error = kernel_detail::validateControlAddresses(
            transaction, memory.getAddressPlan()))
      return std::move(error);
    return TargetModelCommandEffect{{}, {}, control};
  }
  return kernel_detail::kernelError(
      TargetModelKernelErrorCode::UnsupportedTransaction,
      "typed transaction is field-valid but has no evidence-backed plain "
      "functional kernel");
}

llvm::Error
commitTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                               FormalNumericExecutionContext &context,
                               TargetModelCommandEffect effect) {
  if (effect.controlAction != TargetModelControlAction::None &&
      !effect.pendingWrites.empty())
    return kernel_detail::kernelError(
        TargetModelKernelErrorCode::InvalidTransactionField,
        "control effect cannot carry immediate byte writes");
  if (llvm::Error error = memory.applyAtomically(effect.pendingWrites))
    return error;
  context.recordCommittedFlags(effect.numericFlags);
  return llvm::Error::success();
}

} // namespace wafer::model
