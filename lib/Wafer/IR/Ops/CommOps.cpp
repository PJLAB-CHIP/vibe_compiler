//===- CommOps.cpp - Wafer Comm verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult CommRecvOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommSendOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommWaitOp::verify() {
  return verifyCommWaitTokens(getOperation(), getTokens());
}

mlir::LogicalResult CommAllGatherOp::verify() {
  auto localType = mlir::dyn_cast<TileBufferType>(getLocalChunk().getType());
  auto gatherType = mlir::dyn_cast<TileBufferType>(getGatherBuffer().getType());
  if (!localType || !gatherType)
    return emitOpError("all_gather expects tile_buffer operands");
  if (!hasTileBufferMemorySpace(localType, MemorySpace::SPM) ||
      !hasTileBufferMemorySpace(gatherType, MemorySpace::SPM))
    return emitOpError("all_gather buffers must use SPM memory space");

  mlir::RankedTensorType localTensor = getTileBufferTensorType(localType);
  mlir::RankedTensorType gatherTensor = getTileBufferTensorType(gatherType);
  if (localTensor.getElementType() != gatherTensor.getElementType())
    return emitOpError("all_gather local and gather element types must match");

  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize <= 1)
    return emitOpError("all_gather group_size must be greater than one");
  int64_t localRank = getLocalRankAttr().getInt();
  if (localRank < 0 || localRank >= groupSize)
    return emitOpError(
        "all_gather local_rank must be within the collective group");
  int64_t bytes = getBytesAttr().getInt();
  if (bytes <= 0)
    return emitOpError("all_gather byte count must be positive");

  std::optional<int64_t> localBytes = getCompactTensorByteSize(localTensor);
  if (!localBytes)
    return emitOpError(
        "all_gather local compact byte size is not representable");
  if (*localBytes != bytes)
    return emitOpError(
        "all_gather byte count must match local compact byte size");

  int64_t expectedGatherBytes = 0;
  if (!checkedMul(bytes, groupSize, expectedGatherBytes))
    return emitOpError("all_gather total byte size is not representable");
  std::optional<int64_t> gatherBytes = getCompactTensorByteSize(gatherTensor);
  if (!gatherBytes)
    return emitOpError(
        "all_gather gather buffer compact byte size is not representable");
  if (*gatherBytes != expectedGatherBytes)
    return emitOpError(
        "all_gather gather buffer compact byte size must equal bytes times "
        "group_size");

  return mlir::success();
}

static mlir::LogicalResult verifyCommReduceCollective(
    mlir::Operation *op, llvm::StringRef collectiveName, mlir::Value input,
    mlir::Value recvBuffer, mlir::Type resultType,
    mlir::IntegerAttr localRankAttr, mlir::IntegerAttr groupSizeAttr,
    mlir::IntegerAttr bytesAttr) {
  auto inputType = mlir::dyn_cast<TileBufferType>(input.getType());
  auto recvType = mlir::dyn_cast<TileBufferType>(recvBuffer.getType());
  auto resultTileType = mlir::dyn_cast<TileBufferType>(resultType);
  if (!inputType || !recvType || !resultTileType)
    return op->emitOpError(collectiveName)
           << " expects tile_buffer operands and result";
  if (inputType != recvType || inputType != resultTileType)
    return op->emitOpError(collectiveName)
           << " input, recv buffer, and result types must match";
  if (!hasTileBufferMemorySpace(inputType, MemorySpace::SPM))
    return op->emitOpError(collectiveName)
           << " buffers must use SPM memory space";

  int64_t groupSize = groupSizeAttr.getInt();
  if (groupSize <= 1)
    return op->emitOpError(collectiveName)
           << " group_size must be greater than one";
  int64_t localRank = localRankAttr.getInt();
  if (localRank < 0 || localRank >= groupSize)
    return op->emitOpError(collectiveName)
           << " local_rank must be within the collective group";
  int64_t bytes = bytesAttr.getInt();
  if (bytes <= 0)
    return op->emitOpError(collectiveName) << " byte count must be positive";

  std::optional<int64_t> compactBytes =
      getCompactTensorByteSize(getTileBufferTensorType(inputType));
  if (!compactBytes)
    return op->emitOpError(collectiveName)
           << " compact byte size is not representable";
  if (*compactBytes != bytes)
    return op->emitOpError(collectiveName)
           << " byte count must match compact byte size";

  return mlir::success();
}

mlir::LogicalResult CommReduceScatterOp::verify() {
  return verifyCommReduceCollective(getOperation(), "reduce_scatter",
                                    getInput(), getRecvBuffer(),
                                    getResult().getType(), getLocalRankAttr(),
                                    getGroupSizeAttr(), getBytesAttr());
}

mlir::LogicalResult CommAllReduceOp::verify() {
  return verifyCommReduceCollective(getOperation(), "all_reduce", getInput(),
                                    getRecvBuffer(), getResult().getType(),
                                    getLocalRankAttr(), getGroupSizeAttr(),
                                    getBytesAttr());
}
