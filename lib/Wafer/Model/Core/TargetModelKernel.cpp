//===- TargetModelKernel.cpp - Plain target kernel dispatch -----------===//

#include "TargetModelKernelInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>
#include <variant>

namespace wafer::model {

llvm::Expected<TargetModelCommandEffect>
executeTargetModelCommand(const compiler::TargetCommand &command,
                          const InvocationMemoryRegistry &memory,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy) {
  if (llvm::Error error = validateTargetModelCommandFields(command))
    return std::move(error);
  if (!llvm::is_contained(memory.getAddressPlan().getLaunchSlots(),
                          command.launchSlotId.getValue()))
    return kernel_detail::kernelError(
        TargetModelKernelErrorCode::InvalidCommandField,
        "command launch slot is outside invocation");

  if (const auto *value =
          std::get_if<target::TargetStridedDMACommand>(&command.payload))
    return kernel_detail::executeMovement(command, *value, memory, budget);
  if (const auto *value =
          std::get_if<target::TargetGatherScatterCommand>(&command.payload))
    return kernel_detail::executeGatherScatter(command, *value, memory, budget);
  if (const auto *value =
          std::get_if<target::TargetMemsetCommand>(&command.payload))
    return kernel_detail::executeMemset(command, *value, memory);
  if (const auto *value =
          std::get_if<target::TargetElementwiseCommand>(&command.payload))
    return kernel_detail::executeElementwise(command, *value, memory, budget,
                                             policy);
  if (const auto *value =
          std::get_if<target::TargetConvertCommand>(&command.payload))
    return kernel_detail::executeConvert(command, *value, memory, budget,
                                         policy);
  if (const auto *value =
          std::get_if<target::TargetReduceCommand>(&command.payload))
    return kernel_detail::executeReduce(command, *value, memory, budget,
                                        policy);
  if (const auto *value =
          std::get_if<target::TargetGemmCommand>(&command.payload))
    return kernel_detail::executeGemm(command, *value, memory, budget, policy);

  TargetModelControlAction control =
      kernel_detail::getControlAction(command.payload);
  if (control != TargetModelControlAction::None) {
    if (llvm::Error error = kernel_detail::validateControlAddresses(
            command, memory.getAddressPlan()))
      return std::move(error);
    return TargetModelCommandEffect{{}, {}, control};
  }
  return kernel_detail::kernelError(
      TargetModelKernelErrorCode::UnsupportedCommand,
      "typed command is field-valid but has no evidence-backed plain "
      "functional kernel");
}

llvm::Error
applyTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                              FormalNumericExecutionContext &context,
                              TargetModelCommandEffect effect) {
  if (effect.controlAction != TargetModelControlAction::None &&
      (!effect.pendingWrites.empty() || !effect.pendingReads.empty()))
    return kernel_detail::kernelError(
        TargetModelKernelErrorCode::InvalidCommandField,
        "control effect cannot carry plain-command memory accesses");
  if (llvm::Error error = memory.applyAtomically(effect.pendingWrites))
    return error;
  context.mergeExceptionFlags(effect.numericFlags);
  return llvm::Error::success();
}

} // namespace wafer::model
