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
          std::get_if<target::TargetBit2FPCommand>(&command.payload))
    return kernel_detail::executeBit2FP(command, *value, memory, budget);
  if (const auto *value =
          std::get_if<target::TargetMaskMoveCommand>(&command.payload))
    return kernel_detail::executeMaskMove(command, *value, memory, budget);
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
  if (const auto *value =
          std::get_if<target::TargetPoolCommand>(&command.payload))
    return kernel_detail::executePool(command, *value, memory, budget);

  if (std::holds_alternative<target::TargetKcoreReleaseCommand>(command.payload))
    return TargetModelCommandEffect{};
  auto scalarAddressSpace = [](target::TargetScalarMemorySpace space) {
    return space == target::TargetScalarMemorySpace::SPM
               ? TargetModelAddressSpace::TileSPM : TargetModelAddressSpace::DDR;
  };
  if (const auto *mapping =
          std::get_if<target::TargetMemoryMappingCommand>(&command.payload)) {
    auto space = scalarAddressSpace(mapping->space);
    auto range = memory.getAddressPlan().resolve(
        command.launchSlotId.getValue(), space, TargetModelAccess::Read,
        mapping->address, mapping->byteCount, 1);
    if (!range)
      return range.takeError();
    TargetModelCommandEffect effect;
    effect.scalarResult = mapping->address;
    if (mapping->space == target::TargetScalarMemorySpace::DDR)
      effect.pendingReads.push_back({command.launchSlotId.getValue(), space,
                                      mapping->address, mapping->byteCount,
                                      std::nullopt});
    return effect;
  }
  if (const auto *load =
          std::get_if<target::TargetScalarLoadCommand>(&command.payload)) {
    auto space = scalarAddressSpace(load->space);
    auto bytes = memory.readSnapshot(command.launchSlotId.getValue(),
                                      space,
                                      load->address, load->byteWidth,
                                      load->byteWidth);
    if (!bytes)
      return bytes.takeError();
    TargetModelCommandEffect effect;
    for (unsigned byte = 0; byte < load->byteWidth; ++byte)
      effect.scalarResult |= uint64_t((*bytes)[byte]) << (byte * 8);
    effect.pendingReads.push_back({command.launchSlotId.getValue(),
                                    space,
                                    load->address, load->byteWidth, std::nullopt});
    return effect;
  }
  if (const auto *store =
          std::get_if<target::TargetScalarStoreCommand>(&command.payload)) {
    auto space = scalarAddressSpace(store->space);
    auto range = memory.getAddressPlan().resolve(
        command.launchSlotId.getValue(), space,
        TargetModelAccess::Write, store->address, store->byteWidth,
        store->byteWidth);
    if (!range)
      return range.takeError();
    std::vector<uint8_t> bytes(store->byteWidth);
    for (unsigned byte = 0; byte < store->byteWidth; ++byte)
      bytes[byte] = static_cast<uint8_t>(store->value >> (byte * 8));
    TargetModelCommandEffect effect;
    effect.pendingWrites.push_back({command.launchSlotId.getValue(),
                                     space,
                                     store->address, store->byteWidth,
                                     std::move(bytes), std::nullopt});
    return effect;
  }

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
