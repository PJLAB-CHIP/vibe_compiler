//===- CommOps.cpp - Wafer Comm verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Twine.h"

#include <string>

using namespace wafer;
using namespace wafer::detail;

static mlir::LogicalResult verifyCommunicationId(mlir::Operation *op,
                                                 mlir::IntegerAttr id) {
  if (id.getInt() < 0)
    return op->emitOpError("communication_id must be non-negative");
  return mlir::success();
}

mlir::LogicalResult CommAllGatherOp::verify() {
  if (mlir::failed(
          verifyCommunicationId(getOperation(), getCommunicationIdAttr())))
    return mlir::failure();
  std::optional<mlir::RankedTensorType> localTensor =
      getLogicalTensorType(getLocalChunk().getType());
  std::optional<mlir::RankedTensorType> gatherTensor =
      getLogicalTensorType(getGatherBuffer().getType());
  if (!localTensor || !gatherTensor)
    return emitOpError("all_gather expects Wafer buffer operands");
  if (!hasWaferMemorySpace(getLocalChunk().getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(getGatherBuffer().getType(), MemorySpace::SPM))
    return emitOpError("all_gather buffers must use SPM memory space");

  if (localTensor->getElementType() != gatherTensor->getElementType())
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
  if (mlir::failed(verifyLogicalRanksWithinExecutionMesh(
          getOperation(), rankGroup, "all_gather rank_group")))
    return mlir::failure();
  int64_t bytes = getBytesAttr().getInt();
  if (bytes <= 0)
    return emitOpError("all_gather byte count must be positive");

  std::optional<int64_t> localBytes = getCompactTensorByteSize(*localTensor);
  if (!localBytes)
    return emitOpError(
        "all_gather local compact byte size is not representable");
  if (*localBytes != bytes)
    return emitOpError(
        "all_gather byte count must match local compact byte size");

  int64_t expectedGatherBytes = 0;
  if (!checkedMul(bytes, groupSize, expectedGatherBytes))
    return emitOpError("all_gather total byte size is not representable");
  std::optional<int64_t> gatherBytes = getCompactTensorByteSize(*gatherTensor);
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
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getLocalChunk().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 1,
                          getGatherBuffer().getType());
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
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(input.getType());
  std::optional<mlir::RankedTensorType> recvTensor =
      getLogicalTensorType(recvBuffer.getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(resultType);
  if (!inputTensor || !recvTensor || !resultTensor)
    return op->emitOpError(collectiveName)
           << " expects Wafer buffer operands and result";
  if (input.getType() != recvBuffer.getType() || input.getType() != resultType)
    return op->emitOpError(collectiveName)
           << " input, recv buffer, and result types must match";
  if (!hasWaferMemorySpace(input.getType(), MemorySpace::SPM))
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
  std::string rankGroupRole =
      llvm::Twine(collectiveName).concat(" rank_group").str();
  if (mlir::failed(verifyLogicalRanksWithinExecutionMesh(op, rankGroup,
                                                         rankGroupRole)))
    return mlir::failure();
  int64_t bytes = bytesAttr.getInt();
  if (bytes <= 0)
    return op->emitOpError(collectiveName) << " byte count must be positive";

  std::optional<int64_t> compactBytes = getCompactTensorByteSize(*inputTensor);
  if (!compactBytes)
    return op->emitOpError(collectiveName)
           << " compact byte size is not representable";
  if (*compactBytes != bytes)
    return op->emitOpError(collectiveName)
           << " byte count must match compact byte size";

  return mlir::success();
}

mlir::LogicalResult CommReduceScatterOp::verify() {
  mlir::Operation *op = getOperation();
  if (mlir::failed(verifyCommunicationId(op, getCommunicationIdAttr())))
    return mlir::failure();
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  std::optional<mlir::RankedTensorType> recvTensor =
      getLogicalTensorType(getRecvBuffer().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!inputTensor || !recvTensor || !resultTensor)
    return op->emitOpError("reduce_scatter")
           << " expects Wafer buffer operands and result";
  if (getRecvBuffer().getType() != getResult().getType())
    return op->emitOpError("reduce_scatter")
           << " recv buffer and result types must match";
  if (!hasWaferMemorySpace(getInput().getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(getRecvBuffer().getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(getResult().getType(), MemorySpace::SPM))
    return op->emitOpError("reduce_scatter")
           << " buffers must use SPM memory space";

  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize <= 1)
    return op->emitOpError("reduce_scatter")
           << " group_size must be greater than one";
  int64_t localRank = getLocalRankAttr().getInt();
  if (localRank < 0 || localRank >= groupSize)
    return op->emitOpError("reduce_scatter")
           << " local_rank must be within the collective group";
  auto rankGroup = getRankGroupAttr().asArrayRef();
  if (static_cast<int64_t>(rankGroup.size()) != groupSize)
    return op->emitOpError("reduce_scatter")
           << " rank_group size must equal group_size";
  llvm::SmallSet<int64_t, 8> seenRanks;
  for (int64_t rank : rankGroup) {
    if (rank < 0)
      return op->emitOpError("reduce_scatter")
             << " rank_group entries must be non-negative";
    if (!seenRanks.insert(rank).second)
      return op->emitOpError("reduce_scatter")
             << " rank_group entries must be unique";
  }
  if (mlir::failed(verifyLogicalRanksWithinExecutionMesh(
          op, rankGroup, "reduce_scatter rank_group")))
    return mlir::failure();

  int64_t axis = getAxisAttr().getInt();
  if (axis < 0 || axis >= inputTensor->getRank())
    return op->emitOpError("reduce_scatter")
           << " axis must be within the input rank";
  if (inputTensor->getRank() != resultTensor->getRank())
    return op->emitOpError("reduce_scatter")
           << " input and result ranks must match";

  for (int64_t dim = 0; dim < inputTensor->getRank(); ++dim) {
    int64_t inputDim = inputTensor->getDimSize(dim);
    int64_t resultDim = resultTensor->getDimSize(dim);
    if (inputDim == mlir::ShapedType::kDynamic ||
        resultDim == mlir::ShapedType::kDynamic)
      return op->emitOpError("reduce_scatter")
             << " buffer shapes must be static";
    if (dim == axis) {
      int64_t expectedInputDim = 0;
      if (!checkedMul(resultDim, groupSize, expectedInputDim))
        return op->emitOpError("reduce_scatter")
               << " input axis size is not representable";
      if (inputDim != expectedInputDim)
        return op->emitOpError("reduce_scatter")
               << " input axis size must equal result axis size times "
                  "group_size";
      continue;
    }
    if (inputDim != resultDim)
      return op->emitOpError("reduce_scatter")
             << " non-axis dimensions must match";
  }

  int64_t bytes = getBytesAttr().getInt();
  if (bytes <= 0)
    return op->emitOpError("reduce_scatter") << " byte count must be positive";
  std::optional<int64_t> resultBytes = getCompactTensorByteSize(*resultTensor);
  if (!resultBytes)
    return op->emitOpError("reduce_scatter")
           << " result compact byte size is not representable";
  if (*resultBytes != bytes)
    return op->emitOpError("reduce_scatter")
           << " byte count must match result compact byte size";

  return mlir::success();
}

static void collectCommReduceCollectiveLayoutRequirements(
    mlir::Value input, mlir::Value recvBuffer, mlir::Value result,
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          input.getType());
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 1,
                          recvBuffer.getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          result.getType());
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
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getBytesAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getBytesAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult CommReduceScatterOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult CommAllReduceOp::verify() {
  if (mlir::failed(
          verifyCommunicationId(getOperation(), getCommunicationIdAttr())))
    return mlir::failure();
  return verifyCommReduceCollective(getOperation(), "all_reduce", getInput(),
                                    getRecvBuffer(), getResult().getType(),
                                    getLocalRankAttr(), getGroupSizeAttr(),
                                    getRankGroupAttr(), getBytesAttr());
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
