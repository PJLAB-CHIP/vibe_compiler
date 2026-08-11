//===- CommOps.cpp - Wafer peer communication verifiers -------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult
verifyPeerTile(mlir::Operation *op, mlir::Value buffer, mlir::IntegerAttr peer,
               mlir::IntegerAttr bytes, mlir::Type tokenType) {
  if (mlir::failed(verifyDTEP2P(op, buffer, peer, bytes, tokenType)))
    return mlir::failure();
  return verifyPhysicalTileIdWithinTopology(op, peer.getInt(), "peer tile_id");
}

} // namespace

mlir::LogicalResult CommPeerSendOp::verify() {
  return verifyPeerTile(getOperation(), getBuffer(), getPeerAttr(),
                        getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommPeerRecvOp::verify() {
  return verifyPeerTile(getOperation(), getBuffer(), getPeerAttr(),
                        getBytesAttr(), getToken().getType());
}
