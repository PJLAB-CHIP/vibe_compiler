//===- ReferenceProgramProjectionDTE.cpp - DTE and sync projection ------===//

#include "ReferenceProgramProjectionInternal.h"

namespace wafer::compiler::reference_detail {

llvm::Expected<bool>
ProgramProjector::projectDTEAndSync(mlir::Operation &operation,
                                    Command &command) {
  if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(operation)) {
    if (llvm::Error error = projectDTEIssue(
            send.getBuffer(), send.getToken(), send.getPeer(), send.getBytes(),
            send.getMessage(), send.getBinding(),
            /*isSend=*/true, command))
      return error;
  } else if (auto recv = mlir::dyn_cast<wafer::InstrDTERecvOp>(operation)) {
    if (llvm::Error error = projectDTEIssue(
            recv.getBuffer(), recv.getToken(), recv.getPeer(), recv.getBytes(),
            recv.getMessage(), recv.getBinding(),
            /*isSend=*/false, command))
      return error;
  } else if (auto wait = mlir::dyn_cast<wafer::InstrDTEWaitOp>(operation)) {
    if (wait.getTokens().empty())
      return unsupported("Direct DTE wait has no tokens");
    command.kind = CommandKind::DTEWait;
    for (mlir::Value token : wait.getTokens()) {
      auto id = use(token);
      if (!id)
        return id.takeError();
      command.inputs.push_back(*id);
    }
  } else if (mlir::isa<wafer::SyncLocalFenceOp>(operation)) {
    command.kind = CommandKind::LocalFence;

  } else {
    return false;
  }
  return true;
}

llvm::Error ProgramProjector::projectDTEIssue(
    mlir::Value bufferValue, mlir::Value tokenValue, int64_t peer,
    int64_t bytes, wafer::DTEMessageAttr message,
    std::optional<wafer::DirectDTEBindingAttr> binding, bool isSend,
    Command &command) {
  auto bufferType = mlir::dyn_cast<mlir::MemRefType>(bufferValue.getType());
  auto info = bufferType ? wafer::computeWaferPhysicalTensorInfo(bufferType)
                         : std::nullopt;
  if (!bufferType || !wafer::isWaferSPMMemRefType(bufferType) || !info ||
      info->physicalBytes < 0)
    return unsupported("Direct DTE issue requires static Wafer SPM memref");
  if (peer < 0 || bytes <= 0 || bytes > info->physicalBytes)
    return unsupported("Direct DTE peer or byte range is invalid");
  if (!binding ||
      binding->getAllocationProfile() != wafer::DTEAllocationProfile::Normal ||
      binding->getCompletionProfile() !=
          wafer::DTECompletionProfile::SenderWaitReceiverFSM ||
      binding->getReceiverFsmId() < 0 || binding->getRemoteReceiverOffset() < 0)
    return unsupported("Direct DTE issue has no supported accepted binding");
  auto buffer = use(bufferValue);
  auto token = define(tokenValue);
  if (!buffer)
    return buffer.takeError();
  if (!token)
    return token.takeError();
  command.kind = isSend ? CommandKind::DTESend : CommandKind::DTERecv;
  command.source = *buffer;
  command.result = *token;
  command.peer = peer;
  command.byteCount = static_cast<uint64_t>(bytes);
  command.messageCommunication = message.getCommunicationId();
  command.messagePhase = message.getPhase();
  command.messageRound = message.getRound();
  command.messagePayloadSlice = message.getPayloadSlice();
  command.remoteReceiverOffset = binding->getRemoteReceiverOffset();
  return llvm::Error::success();
}

} // namespace wafer::compiler::reference_detail
