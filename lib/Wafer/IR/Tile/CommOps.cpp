//===- CommOps.cpp - Wafer Comm verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult CommRecvOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

void CommRecvOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto bufferType = mlir::dyn_cast<TileBufferType>(getBuffer().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            bufferType);
}

mlir::LogicalResult CommRecvOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void CommRecvOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  int64_t bytes = getBytesAttr().getInt();
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 0,
                       bytes);
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       bytes);
}

mlir::LogicalResult CommRecvOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult CommSendOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

void CommSendOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto bufferType = mlir::dyn_cast<TileBufferType>(getBuffer().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            bufferType);
}

mlir::LogicalResult CommSendOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void CommSendOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  int64_t bytes = getBytesAttr().getInt();
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       bytes);
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       bytes);
}

mlir::LogicalResult CommSendOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult CommWaitOp::verify() {
  return verifyCommWaitTokens(getOperation(), getTokens());
}

void CommWaitOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, token] : llvm::enumerate(getTokens())) {
    (void)token;
    appendResourceEffect(effects, WaferResourceKind::Communication,
                         WaferResourceAccess::Wait, WaferValueRole::Operand,
                         index, -1);
  }
}

mlir::LogicalResult CommWaitOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
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
  auto rankGroup = getRankGroupAttr().asArrayRef();
  if (static_cast<int64_t>(rankGroup.size()) != groupSize)
    return emitOpError("all_gather rank_group size must equal group_size");
  llvm::SmallSet<int64_t, 8> seenRanks;
  for (int64_t rank : rankGroup) {
    if (rank < 0)
      return emitOpError("all_gather rank_group entries must be non-negative");
    if (!seenRanks.insert(rank).second)
      return emitOpError("all_gather rank_group entries must be unique");
  }
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

void CommAllGatherOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto localType =
          mlir::dyn_cast<TileBufferType>(getLocalChunk().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            localType);
  if (auto gatherType =
          mlir::dyn_cast<TileBufferType>(getGatherBuffer().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 1,
                            gatherType);
}

mlir::LogicalResult CommAllGatherOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void CommAllGatherOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  int64_t bytes = getBytesAttr().getInt();
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       bytes);
  appendResourceEffect(
      effects, WaferResourceKind::SPM, WaferResourceAccess::Write,
      WaferValueRole::Operand, 1,
      getCompactByteSizeOrUnknown(getGatherBuffer().getType()));
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       bytes);
}

mlir::LogicalResult CommAllGatherOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

static mlir::LogicalResult verifyCommReduceCollective(
    mlir::Operation *op, llvm::StringRef collectiveName, mlir::Value input,
    mlir::Value recvBuffer, mlir::Type resultType,
    mlir::IntegerAttr localRankAttr, mlir::IntegerAttr groupSizeAttr,
    mlir::DenseI64ArrayAttr rankGroupAttr, mlir::IntegerAttr bytesAttr) {
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
  auto rankGroup = rankGroupAttr.asArrayRef();
  if (static_cast<int64_t>(rankGroup.size()) != groupSize)
    return op->emitOpError(collectiveName)
           << " rank_group size must equal group_size";
  llvm::SmallSet<int64_t, 8> seenRanks;
  for (int64_t rank : rankGroup) {
    if (rank < 0)
      return op->emitOpError(collectiveName)
             << " rank_group entries must be non-negative";
    if (!seenRanks.insert(rank).second)
      return op->emitOpError(collectiveName)
             << " rank_group entries must be unique";
  }
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
                                    getGroupSizeAttr(), getRankGroupAttr(),
                                    getBytesAttr());
}

static void collectCommReduceCollectiveLayoutRequirements(
    mlir::Value input, mlir::Value recvBuffer, mlir::Value result,
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto inputType = mlir::dyn_cast<TileBufferType>(input.getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            inputType);
  if (auto recvType = mlir::dyn_cast<TileBufferType>(recvBuffer.getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 1, recvType);
  if (auto resultType = mlir::dyn_cast<TileBufferType>(result.getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

void CommReduceScatterOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  collectCommReduceCollectiveLayoutRequirements(getInput(), getRecvBuffer(),
                                                getResult(), requirements);
}

mlir::LogicalResult CommReduceScatterOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

static void collectCommReduceCollectiveResourceEffects(
    mlir::Value input, mlir::Value recvBuffer, mlir::Value result,
    int64_t bytes, llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       bytes);
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       bytes);
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(result.getType()));
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       bytes);
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(result.getType()));
}

void CommReduceScatterOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  collectCommReduceCollectiveResourceEffects(getInput(), getRecvBuffer(),
                                             getResult(),
                                             getBytesAttr().getInt(), effects);
}

mlir::LogicalResult CommReduceScatterOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult CommAllReduceOp::verify() {
  return verifyCommReduceCollective(getOperation(), "all_reduce", getInput(),
                                    getRecvBuffer(), getResult().getType(),
                                    getLocalRankAttr(), getGroupSizeAttr(),
                                    getRankGroupAttr(),
                                    getBytesAttr());
}

void CommAllReduceOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  collectCommReduceCollectiveLayoutRequirements(getInput(), getRecvBuffer(),
                                                getResult(), requirements);
}

mlir::LogicalResult CommAllReduceOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void CommAllReduceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  collectCommReduceCollectiveResourceEffects(getInput(), getRecvBuffer(),
                                             getResult(),
                                             getBytesAttr().getInt(), effects);
}

mlir::LogicalResult CommAllReduceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
