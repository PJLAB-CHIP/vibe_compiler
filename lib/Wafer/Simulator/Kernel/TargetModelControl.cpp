//===- TargetModelControl.cpp - Control and Direct DTE validation ----===//

#include "TargetModelKernelInternal.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"

#include <limits>
#include <utility>
#include <variant>

namespace wafer::model::kernel_detail {

TargetModelControlAction
getControlAction(const target::TargetCommandPayload &payload) {
  if (std::holds_alternative<target::TargetNCCJoinCommand>(payload))
    return TargetModelControlAction::NCCJoin;
  if (std::holds_alternative<target::TargetDirectDTEBeginCommand>(payload))
    return TargetModelControlAction::DirectDTEBegin;
  if (std::holds_alternative<target::TargetDirectDTESendCommand>(payload))
    return TargetModelControlAction::DirectDTESendPrepare;
  if (std::holds_alternative<target::TargetDirectDTEMultiSendCommand>(payload))
    return TargetModelControlAction::DirectDTEMultiSendPrepare;
  if (std::holds_alternative<
          target::TargetDirectDTEMultiSendDestinationCommand>(payload))
    return TargetModelControlAction::DirectDTEMultiSendDestination;
  if (std::holds_alternative<target::TargetDirectDTESendIssueCommand>(payload))
    return TargetModelControlAction::DirectDTESendIssue;
  if (std::holds_alternative<target::TargetDirectDTEReceiveCommand>(payload))
    return TargetModelControlAction::DirectDTEReceive;
  if (std::holds_alternative<target::TargetDirectDTEWaitCommand>(payload))
    return TargetModelControlAction::DirectDTEWait;
  if (std::holds_alternative<target::TargetDirectDTEFinishCommand>(payload))
    return TargetModelControlAction::DirectDTEFinish;
  return TargetModelControlAction::None;
}

llvm::Error validateControlAddresses(const compiler::TargetCommand &command,
                                     const InvocationAddressPlan &plan) {
  if (const auto *begin =
          std::get_if<target::TargetDirectDTEBeginCommand>(&command.payload)) {
    if (begin->participantCount != plan.getLaunchSlots().size())
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "Direct DTE participant count differs from invocation");
    llvm::Expected<TargetModelResolvedRange> status = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::DDR,
        TargetModelAccess::ReadWrite, begin->statusAddress,
        WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES,
        WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES);
    if (!status)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(status.takeError()));
  } else if (const auto *send = std::get_if<target::TargetDirectDTESendCommand>(
                 &command.payload)) {
    if (send->localTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteFSM >= 4)
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "Direct DTE send tile or receiver FSM is outside "
                         "the accepted target ABI domain");
    if (send->highPerformance)
      return kernelError(
          TargetModelKernelErrorCode::UnsupportedCommand,
          "Direct DTE high-performance allocation is outside the accepted "
          "normal sender profile");
    llvm::Expected<TargetModelResolvedRange> source = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
        TargetModelAccess::Read, send->source, send->byteCount, 1);
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
        TargetModelAccess::Write, send->remoteDestination, send->byteCount, 1);
    if (!source || !destination) {
      llvm::Error errors = llvm::Error::success();
      if (!source)
        errors = llvm::joinErrors(std::move(errors), source.takeError());
      if (!destination)
        errors = llvm::joinErrors(std::move(errors), destination.takeError());
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(std::move(errors)));
    }
  } else if (const auto *multi =
                 std::get_if<target::TargetDirectDTEMultiSendCommand>(
                     &command.payload)) {
    if (multi->localTile > std::numeric_limits<uint16_t>::max() ||
        multi->bytesPerDestination != 256 ||
        (multi->destinationCount != 2 && multi->destinationCount != 4 &&
         multi->destinationCount != 8 && multi->destinationCount != 15) ||
        multi->highPerformance)
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "Direct DTE multi-send prepare is outside the accepted target "
          "domain");
    uint64_t sourceBytes = multi->bytesPerDestination;
    if (multi->kind == target::TargetDirectDTEMultiSendKind::Scatter) {
      if (multi->destinationCount >
          std::numeric_limits<uint64_t>::max() / multi->bytesPerDestination)
        return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                           "Direct DTE scatter source span overflows");
      sourceBytes *= multi->destinationCount;
    }
    llvm::Expected<TargetModelResolvedRange> source = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
        TargetModelAccess::Read, multi->source, sourceBytes, 1);
    if (!source)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(source.takeError()));
  } else if (const auto *destination = std::get_if<
                 target::TargetDirectDTEMultiSendDestinationCommand>(
                 &command.payload)) {
    if (destination->event == 0 ||
        destination->remoteTile > std::numeric_limits<uint16_t>::max() ||
        destination->remoteFSM >= 4)
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "Direct DTE multi-send destination is outside the accepted target "
          "domain");
    llvm::Expected<TargetModelResolvedRange> remote = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
        TargetModelAccess::Write, destination->remoteDestination, 256, 1);
    if (!remote)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(remote.takeError()));
  } else if (const auto *receive =
                 std::get_if<target::TargetDirectDTEReceiveCommand>(
                     &command.payload)) {
    if (receive->localTile > std::numeric_limits<uint16_t>::max() ||
        receive->remoteTile > std::numeric_limits<uint16_t>::max() ||
        receive->localFSM >= 4)
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "Direct DTE receive tile or receiver FSM is outside the accepted "
          "target ABI domain");
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
        TargetModelAccess::Write, receive->destination, receive->byteCount, 1);
    if (!destination)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(destination.takeError()));
  }
  return llvm::Error::success();
}

} // namespace wafer::model::kernel_detail
