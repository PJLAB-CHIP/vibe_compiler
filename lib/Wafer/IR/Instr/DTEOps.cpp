//===- DTEOps.cpp - Wafer DTE instruction verifier implementation --------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult verifyDTEPeer(mlir::Operation *op,
                                         mlir::IntegerAttr peerAttr) {
  return verifyLogicalRankWithinExecutionMesh(op, peerAttr.getInt(),
                                              "DTE peer logical rank");
}

} // namespace

mlir::LogicalResult InstrDTERecvOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  if (getBindingSelector())
    return emitOpError(
        "Direct DTE receive cannot carry a sender route selector");
  return verifyDTEPeer(getOperation(), getPeerAttr());
}

InstrFamily InstrDTERecvOp::getInstructionFamily() { return InstrFamily::DTE; }

mlir::LogicalResult InstrDTESendOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  if (getBindingSelector()) {
    if (!getBinding() ||
        getBinding()->getRemoteAddressMode() !=
            DTERemoteAddressMode::SelectorTable)
      return emitOpError(
          "Direct DTE send route selector requires selector-table binding");
  } else if (getBinding() &&
             getBinding()->getRemoteAddressMode() ==
                 DTERemoteAddressMode::SelectorTable) {
    return emitOpError(
        "Direct DTE selector-table binding requires route selector operand");
  }
  return verifyDTEPeer(getOperation(), getPeerAttr());
}

InstrFamily InstrDTESendOp::getInstructionFamily() { return InstrFamily::DTE; }

mlir::LogicalResult InstrDTEWaitOp::verify() {
  if (getTokens().size() > std::numeric_limits<uint32_t>::max())
    return emitOpError(
        "target_abi_narrowing: DTE wait token count must fit uint32_t");
  return verifyDTEWaitTokens(getOperation(), getTokens());
}

InstrFamily InstrDTEWaitOp::getInstructionFamily() { return InstrFamily::DTE; }
