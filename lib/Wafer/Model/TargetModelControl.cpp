//===- TargetModelControl.cpp - Control and Direct DTE validation ----===//

#include "TargetModelKernelInternal.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"

#include <limits>
#include <utility>
#include <variant>

namespace wafer::model::kernel_detail {

TargetModelControlAction
getControlAction(const compiler::TargetTransactionPayload &payload) {
  if (std::holds_alternative<compiler::TargetNCCJoinTransaction>(payload))
    return TargetModelControlAction::NCCJoin;
  if (std::holds_alternative<compiler::TargetDirectDTEBeginTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEBegin;
  if (std::holds_alternative<compiler::TargetDirectDTESendTransaction>(payload))
    return TargetModelControlAction::DirectDTESendPrepare;
  if (std::holds_alternative<
          compiler::TargetDirectDTESendIssueTransaction>(payload))
    return TargetModelControlAction::DirectDTESendIssue;
  if (std::holds_alternative<compiler::TargetDirectDTEReceiveTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEReceive;
  if (std::holds_alternative<compiler::TargetDirectDTEWaitTransaction>(payload))
    return TargetModelControlAction::DirectDTEWait;
  if (std::holds_alternative<compiler::TargetDirectDTEFinishTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEFinish;
  return TargetModelControlAction::None;
}

llvm::Error
validateControlAddresses(const compiler::TargetTransaction &transaction,
                         const InvocationAddressPlan &plan) {
  if (const auto *begin =
          std::get_if<compiler::TargetDirectDTEBeginTransaction>(
              &transaction.payload)) {
    if (begin->rankCount != plan.getLogicalRanks().size())
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "Direct DTE rank_count differs from invocation");
    llvm::Expected<TargetModelResolvedRange> status =
        plan.resolve(transaction.logicalRank, TargetModelAddressSpace::CardDDR,
                     TargetModelAccess::ReadWrite, begin->statusAddress,
                     WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES,
                     WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES);
    if (!status)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(status.takeError()));
  } else if (const auto *send =
                 std::get_if<compiler::TargetDirectDTESendTransaction>(
                     &transaction.payload)) {
    if (send->localTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteFSM >= 4)
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "Direct DTE send tile or receiver FSM is outside "
                         "the accepted target ABI domain");
    if (send->highPerformance)
      return kernelError(
          TargetModelKernelErrorCode::UnsupportedTransaction,
          "Direct DTE high-performance allocation is outside the accepted "
          "normal sender profile");
    llvm::Expected<TargetModelResolvedRange> source =
        plan.resolve(transaction.logicalRank, TargetModelAddressSpace::RankSPM,
                     TargetModelAccess::Read, send->source, send->byteCount, 1);
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        transaction.logicalRank, TargetModelAddressSpace::RankSPM,
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
  } else if (const auto *receive =
                 std::get_if<compiler::TargetDirectDTEReceiveTransaction>(
                     &transaction.payload)) {
    if (receive->localTile > std::numeric_limits<uint16_t>::max() ||
        receive->remoteTile > std::numeric_limits<uint16_t>::max() ||
        receive->localFSM >= 4)
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "Direct DTE receive tile or receiver FSM is outside the accepted "
          "target ABI domain");
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        transaction.logicalRank, TargetModelAddressSpace::RankSPM,
        TargetModelAccess::Write, receive->destination, receive->byteCount, 1);
    if (!destination)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(destination.takeError()));
  }
  return llvm::Error::success();
}

} // namespace wafer::model::kernel_detail
