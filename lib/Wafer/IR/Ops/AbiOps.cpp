//===- AbiOps.cpp - Wafer ABI verifier implementation ----------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace wafer;
using namespace wafer::detail;

struct DirectDteResourceTuple {
  int64_t fsmId = -1;
  int64_t packetId = -1;
  int64_t streamId = -1;

  bool operator==(const DirectDteResourceTuple &rhs) const {
    return fsmId == rhs.fsmId && packetId == rhs.packetId &&
           streamId == rhs.streamId;
  }
};

struct ActiveDirectDteResource {
  mlir::Value token;
  DirectDteResourceTuple resource;
};

static DirectDteResourceTuple
getDirectDteResourceTuple(mlir::IntegerAttr fsmId, mlir::IntegerAttr packetId,
                          mlir::IntegerAttr streamId) {
  return {fsmId.getInt(), packetId.getInt(), streamId.getInt()};
}

static mlir::LogicalResult verifyNonNegativeDirectDteResourceAttr(
    mlir::Operation *op, mlir::IntegerAttr attr, llvm::StringRef name) {
  if (attr.getInt() < 0)
    return op->emitOpError("Direct DTE ") << name << " must be non-negative";
  return mlir::success();
}

static std::optional<ActiveDirectDteResource>
getDirectDteResourceIssue(mlir::Operation *op) {
  if (auto send = mlir::dyn_cast<AbiDteSendOp>(op))
    return ActiveDirectDteResource{
        send.getToken(),
        getDirectDteResourceTuple(send.getFsmIdAttr(), send.getPacketIdAttr(),
                                  send.getStreamIdAttr())};
  if (auto recv = mlir::dyn_cast<AbiDteRecvOp>(op))
    return ActiveDirectDteResource{
        recv.getToken(),
        getDirectDteResourceTuple(recv.getFsmIdAttr(), recv.getPacketIdAttr(),
                                  recv.getStreamIdAttr())};
  return std::nullopt;
}

static void releaseDirectDteResources(
    mlir::ValueRange tokens,
    llvm::SmallVectorImpl<ActiveDirectDteResource> &activeResources) {
  for (mlir::Value token : tokens) {
    for (auto it = activeResources.begin(); it != activeResources.end();) {
      if (it->token == token)
        it = activeResources.erase(it);
      else
        ++it;
    }
  }
}

static mlir::LogicalResult
verifyDirectDteResourceAvailability(mlir::Operation *op,
                                    DirectDteResourceTuple resource) {
  mlir::Block *block = op->getBlock();
  if (!block)
    return mlir::success();

  llvm::SmallVector<ActiveDirectDteResource, 8> activeResources;
  for (mlir::Operation &previous : *block) {
    if (&previous == op)
      break;

    if (auto wait = mlir::dyn_cast<AbiDteWaitOp>(&previous))
      releaseDirectDteResources(wait.getTokens(), activeResources);

    if (std::optional<ActiveDirectDteResource> issue =
            getDirectDteResourceIssue(&previous))
      activeResources.push_back(*issue);
  }

  for (const ActiveDirectDteResource &active : activeResources) {
    if (active.resource == resource)
      return op->emitOpError("Direct DTE resource tuple is already in use");
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyDirectDteP2P(mlir::Operation *op, mlir::Value buffer,
                   mlir::IntegerAttr peer, mlir::IntegerAttr bytes,
                   mlir::Type tokenType, mlir::IntegerAttr fsmId,
                   mlir::IntegerAttr packetId, mlir::IntegerAttr streamId) {
  if (mlir::failed(verifyCommP2P(op, buffer, peer, bytes, tokenType)))
    return mlir::failure();
  if (mlir::failed(verifyNonNegativeDirectDteResourceAttr(op, fsmId, "fsm_id")))
    return mlir::failure();
  if (mlir::failed(
          verifyNonNegativeDirectDteResourceAttr(op, packetId, "packet_id")))
    return mlir::failure();
  if (mlir::failed(
          verifyNonNegativeDirectDteResourceAttr(op, streamId, "stream_id")))
    return mlir::failure();

  return verifyDirectDteResourceAvailability(
      op, getDirectDteResourceTuple(fsmId, packetId, streamId));
}

static mlir::LogicalResult
verifyIssueOnlyWaitPolicy(mlir::Operation *op,
                          wafer::AbiWaitPolicyAttr policy) {
  if (policy.getValue() != wafer::AbiWaitPolicy::IssueOnly)
    return op->emitOpError(
        "C ABI skeleton ops must use issue_only wait policy");
  return mlir::success();
}

mlir::LogicalResult AbiRdma1DOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError(
        "ABI RDMA expects ranked tensor source and tile_buffer result");

  if (resultType.getTensorType() != sourceType)
    return emitOpError(
        "ABI RDMA result tensor type must match source tensor type");
  if (!hasTileBufferMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("ABI RDMA result must use SPM memory space");
  if (!hasTileBufferLayout(resultType, MemLayout::Tensor))
    return emitOpError("ABI RDMA result must use tensor mem_layout");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(sourceType);
  if (!expectedBytes)
    return emitOpError(
        "ABI RDMA compact tensor transfer size is not representable");
  if (getBytesAttr().getInt() != *expectedBytes)
    return emitOpError(
        "ABI RDMA byte count must match compact tensor transfer size");

  return mlir::success();
}

mlir::LogicalResult AbiWdma1DOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto destType = mlir::dyn_cast<mlir::RankedTensorType>(getDest().getType());
  if (!sourceType || !destType)
    return emitOpError(
        "ABI WDMA expects tile_buffer source and ranked tensor dest");

  if (sourceType.getTensorType() != destType)
    return emitOpError(
        "ABI WDMA source tensor type must match dest tensor type");
  if (!hasTileBufferMemorySpace(sourceType, MemorySpace::SPM))
    return emitOpError("ABI WDMA source must use SPM memory space");
  if (!hasTileBufferLayout(sourceType, MemLayout::Tensor))
    return emitOpError("ABI WDMA source must use tensor mem_layout");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(destType);
  if (!expectedBytes)
    return emitOpError(
        "ABI WDMA compact tensor transfer size is not representable");
  if (getBytesAttr().getInt() != *expectedBytes)
    return emitOpError(
        "ABI WDMA byte count must match compact tensor transfer size");

  return mlir::success();
}

mlir::LogicalResult AbiGemmOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto lhsType = mlir::dyn_cast<TileBufferType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<TileBufferType>(getRhs().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("ABI GEMM expects tile_buffer operands and result");

  for (TileBufferType type : {lhsType, rhsType, resultType}) {
    if (!hasTileBufferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("ABI GEMM tile buffers must use SPM memory space");
    if (!hasTileBufferLayout(type, MemLayout::Cx))
      return emitOpError("ABI GEMM tile buffers must use cx mem_layout");
  }

  mlir::RankedTensorType lhsTensor = getTileBufferTensorType(lhsType);
  mlir::RankedTensorType rhsTensor = getTileBufferTensorType(rhsType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultType);

  if (lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("ABI GEMM operand and result element types must match");

  if (lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
      resultTensor.getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    if (mlir::failed(verifyBatchedGemmTileContract(
            getOperation(), lhsTensor, rhsTensor, resultTensor, attrs)))
      return mlir::failure();

    if (getMAttr().getInt() != lhsTensor.getDimSize(attrs.lhsMDim) ||
        getKAttr().getInt() != lhsTensor.getDimSize(attrs.lhsContractingDim) ||
        getNAttr().getInt() != rhsTensor.getDimSize(attrs.rhsNDim))
      return emitOpError("ABI GEMM m/k/n attrs must match batched GEMM dims");
    return mlir::success();
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError(
        "ABI GEMM rank-2 form must not carry batched GEMM attrs");

  if (hasStaticMismatch(lhsTensor.getDimSize(1), rhsTensor.getDimSize(0)))
    return emitOpError("ABI GEMM lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(0), resultTensor.getDimSize(0)) ||
      hasStaticMismatch(rhsTensor.getDimSize(1), resultTensor.getDimSize(1)))
    return emitOpError("ABI GEMM result shape must be lhs M by rhs N");

  if (getMAttr().getInt() != lhsTensor.getDimSize(0) ||
      getKAttr().getInt() != lhsTensor.getDimSize(1) ||
      getNAttr().getInt() != rhsTensor.getDimSize(1))
    return emitOpError("ABI GEMM m/k/n attrs must match tile buffer shapes");

  return mlir::success();
}
mlir::LogicalResult AbiElementwiseOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType());
}

mlir::LogicalResult AbiReduceOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}

mlir::LogicalResult AbiDteRecvOp::verify() {
  return verifyDirectDteP2P(getOperation(), getBuffer(), getPeerAttr(),
                            getBytesAttr(), getToken().getType(),
                            getFsmIdAttr(), getPacketIdAttr(),
                            getStreamIdAttr());
}

mlir::LogicalResult AbiDteSendOp::verify() {
  return verifyDirectDteP2P(getOperation(), getBuffer(), getPeerAttr(),
                            getBytesAttr(), getToken().getType(),
                            getFsmIdAttr(), getPacketIdAttr(),
                            getStreamIdAttr());
}

mlir::LogicalResult AbiDteWaitOp::verify() {
  return verifyCommWaitTokens(getOperation(), getTokens());
}
