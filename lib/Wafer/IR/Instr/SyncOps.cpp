//===- SyncOps.cpp - Wafer sync verifier implementation ---------------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult SyncNCCJoinOp::verify() {
  llvm::ArrayRef<int64_t> participants = getParticipants();
  if (participants.empty())
    return emitOpError(
        "completion_participants_empty: NCC join requires at least one worker");

  int64_t previous = -1;
  for (int64_t worker : participants) {
    if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount))
      return emitOpError() << "completion_participant_out_of_range: worker "
                           << worker << " is outside [0, " << kNCCWorkerCount
                           << ")";
    if (worker <= previous)
      return emitOpError(
          "completion_participants_not_canonical: workers must be strictly "
          "increasing and unique");
    previous = worker;
  }
  return mlir::success();
}

NCCOperationCompletion SyncNCCJoinOp::getNCCCompletion() {
  uint32_t participantMask = 0;
  for (int64_t worker : getParticipants()) {
    if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount))
      return {NCCCompletionKind::ParticipantJoin, std::nullopt, 0};
    participantMask |= uint32_t{1} << static_cast<uint32_t>(worker);
  }
  return {NCCCompletionKind::ParticipantJoin, std::nullopt, participantMask};
}
