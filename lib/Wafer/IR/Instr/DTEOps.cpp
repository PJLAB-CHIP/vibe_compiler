//===- DTEOps.cpp - Wafer DTE instruction verifier implementation --------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>

using namespace wafer;
using namespace wafer::detail;

namespace {

template <typename Op> mlir::LogicalResult verifyBufferOffset(Op op) {
  std::optional<int64_t> offset = op.getBufferOffset();
  if (!offset)
    return mlir::success();
  auto type = mlir::dyn_cast<mlir::MemRefType>(op.getBuffer().getType());
  std::optional<WaferPhysicalTensorInfo> info =
      type ? computeWaferPhysicalTensorInfo(type) : std::nullopt;
  int64_t end = 0;
  if (*offset < 0 ||
      llvm::AddOverflow(*offset, op.getBytesAttr().getInt(), end) || !info ||
      end > info->physicalBytes)
    return op.emitOpError(
        "target_geometry_mismatch: Direct DTE buffer offset and byte count "
        "must name a static in-bounds physical range");
  return mlir::success();
}

} // namespace

mlir::LogicalResult InstrDTERecvOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  if (getBindingSelector())
    return emitOpError(
        "Direct DTE receive cannot carry a sender route selector");
  return verifyBufferOffset(*this);
}

InstrFamily InstrDTERecvOp::getInstructionFamily() { return InstrFamily::DTE; }

mlir::LogicalResult InstrDTESendOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  if (getBindingSelector()) {
    if (!getBinding() || getBinding()->getRemoteAddressMode() !=
                             DTERemoteAddressMode::SelectorTable)
      return emitOpError(
          "Direct DTE send route selector requires selector-table binding");
  } else if (getBinding() && getBinding()->getRemoteAddressMode() ==
                                 DTERemoteAddressMode::SelectorTable) {
    return emitOpError(
        "Direct DTE selector-table binding requires route selector operand");
  }
  return verifyBufferOffset(*this);
}

InstrFamily InstrDTESendOp::getInstructionFamily() { return InstrFamily::DTE; }

mlir::LogicalResult InstrDTEBroadcastOp::verify() {
  return verifyDTEMultiSend(getOperation(), getBuffer(), getSourceOffsetAttr(),
                            getPeersAttr(), getBytesAttr(), getMessagesAttr(),
                            getBindings(), getToken().getType(),
                            /*scatter=*/false);
}

InstrFamily InstrDTEBroadcastOp::getInstructionFamily() {
  return InstrFamily::DTE;
}

mlir::LogicalResult InstrDTEScatterOp::verify() {
  return verifyDTEMultiSend(getOperation(), getBuffer(), getSourceOffsetAttr(),
                            getPeersAttr(), getBytesAttr(), getMessagesAttr(),
                            getBindings(), getToken().getType(),
                            /*scatter=*/true);
}

InstrFamily InstrDTEScatterOp::getInstructionFamily() {
  return InstrFamily::DTE;
}

mlir::LogicalResult InstrDTEWaitOp::verify() {
  if (getTokens().size() > std::numeric_limits<uint32_t>::max())
    return emitOpError(
        "target_abi_narrowing: DTE wait token count must fit uint32_t");
  return verifyDTEWaitTokens(getOperation(), getTokens());
}

InstrFamily InstrDTEWaitOp::getInstructionFamily() { return InstrFamily::DTE; }
