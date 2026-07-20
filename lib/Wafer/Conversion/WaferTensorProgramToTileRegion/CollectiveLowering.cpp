//===- CollectiveLowering.cpp - Tensor collective lowering ----------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<int64_t>
TileRegionBodyEmitter::getCompactByteSize(mlir::Value buffer,
                                          llvm::StringRef subject) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
  if (!memrefType) {
    std::string reason = subject.str() + " buffer is not a memref";
    return failI64(reason);
  }
  std::optional<WaferPhysicalTensorInfo> physicalInfo =
      computeWaferPhysicalTensorInfo(memrefType);
  if (!physicalInfo || physicalInfo->compactBytes <= 0) {
    std::string reason =
        subject.str() + " compact byte size is not representable";
    return failI64(reason);
  }
  return physicalInfo->compactBytes;
}

mlir::FailureOr<int64_t> TileRegionBodyEmitter::getCommunicationId(
    const WaferLinalgExtCollectiveInfo &info, llvm::StringRef subject) {
  if (!info.hasChannelId) {
    std::string reason =
        subject.str() +
        " materialization requires channel_id for stable DTE identity";
    return failI64(reason);
  }
  return info.channelId;
}

mlir::FailureOr<SelectedCollectiveRankGroup>
TileRegionBodyEmitter::getCollectiveRankGroup(
    const WaferLinalgExtCollectiveInfo &info) {
  auto findLocalRank = [&](llvm::ArrayRef<int64_t> ranks)
      -> std::optional<SelectedCollectiveRankGroup> {
    for (auto [index, logicalRank] : llvm::enumerate(ranks)) {
      if (logicalRank != currentLogicalRank)
        continue;
      SelectedCollectiveRankGroup selected;
      selected.ranks.assign(ranks.begin(), ranks.end());
      selected.localRank = static_cast<int64_t>(index);
      return selected;
    }
    return std::nullopt;
  };

  if (!info.rankGroup.empty()) {
    if (std::optional<SelectedCollectiveRankGroup> selected =
            findLocalRank(info.rankGroup))
      return *selected;
    return failSelectedCollectiveRankGroup(
        "logical-rank is not a member of collective rank_group");
  }

  if (!info.hasRankGroups || info.rankGroupSize <= 0)
    return failSelectedCollectiveRankGroup(
        "collective materialization requires rank_group or rank_groups");
  if (info.rankGroups.size() % static_cast<size_t>(info.rankGroupSize) != 0)
    return failSelectedCollectiveRankGroup(
        "collective rank_groups are malformed");

  for (size_t offset = 0; offset < info.rankGroups.size();
       offset += static_cast<size_t>(info.rankGroupSize)) {
    llvm::ArrayRef<int64_t> ranks(info.rankGroups.data() + offset,
                                  static_cast<size_t>(info.rankGroupSize));
    if (std::optional<SelectedCollectiveRankGroup> selected =
            findLocalRank(ranks))
      return *selected;
  }
  return failSelectedCollectiveRankGroup(
      "logical-rank is not a member of collective rank_groups");
}

std::optional<ComputeReduceKind>
TileRegionBodyEmitter::inferCollectiveReduceKind(mlir::Region &combiner) {
  if (!combiner.hasOneBlock()) {
    (void)fail("collective reduction materialization requires one combiner "
               "block");
    return std::nullopt;
  }
  mlir::Block &block = combiner.front();
  if (block.getNumArguments() != 2 ||
      !mlir::isa<LinalgExtCollectiveYieldOp>(block.getTerminator())) {
    (void)fail("collective reduction materialization requires two combiner "
               "arguments and one yielded value");
    return std::nullopt;
  }
  return matchExactReductionKind(
      llvm::ArrayRef<mlir::BlockArgument>{block.getArgument(1)},
      /*redPos=*/0, block.getArgument(0),
      "collective reduction materialization", failureReason);
}

mlir::LogicalResult
TileRegionBodyEmitter::requireSingleTensorCollective(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return fail("collective materialization supports one result");
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertAllGather(
    LinalgExtCollectiveAllGatherOp op, const WaferLinalgExtCollectiveInfo &info,
    mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_gather materialization supports one input and one out");
  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(info, "all_gather");
  if (mlir::failed(communicationId))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("all_gather result is not a ranked tensor");

  mlir::FailureOr<mlir::Value> localChunk =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(localChunk))
    return mlir::failure();

  auto gatherBuffer = builder.create<mlir::memref::AllocOp>(
      op.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
  mlir::FailureOr<int64_t> bytes =
      getCompactByteSize(*localChunk, "all_gather local chunk");
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(info);
  if (mlir::failed(bytes) || mlir::failed(rankGroup))
    return mlir::failure();

  mlir::MLIRContext *context = builder.getContext();
  builder.create<CommAllGatherOp>(
      op.getLoc(), *localChunk, gatherBuffer.getResult(),
      builder.getI64IntegerAttr(rankGroup->localRank),
      builder.getI64IntegerAttr(static_cast<int64_t>(rankGroup->ranks.size())),
      mlir::DenseI64ArrayAttr::get(context, rankGroup->ranks),
      builder.getI64IntegerAttr(*bytes),
      builder.getI64IntegerAttr(*communicationId));
  record(op.getResult(0), MemLayout::Tensor, gatherBuffer.getResult());
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertReduceScatter(
    LinalgExtCollectiveReduceScatterOp op,
    const WaferLinalgExtCollectiveInfo &info, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "reduce_scatter materialization supports one input and one out");
  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(info, "reduce_scatter");
  if (mlir::failed(communicationId))
    return mlir::failure();
  std::optional<ComputeReduceKind> kind =
      inferCollectiveReduceKind(op.getCombiner());
  if (!kind)
    return mlir::failure();

  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(info);
  if (mlir::failed(input) || mlir::failed(rankGroup))
    return mlir::failure();
  if (!info.hasAxis)
    return fail("reduce_scatter materialization requires an axis");

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("reduce_scatter result must be ranked");
  auto slotType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  auto recvBuffer =
      builder.create<mlir::memref::AllocOp>(op.getLoc(), slotType);
  mlir::FailureOr<int64_t> bytes =
      getCompactByteSize(recvBuffer.getResult(), "reduce_scatter local slot");
  if (mlir::failed(bytes))
    return mlir::failure();

  auto kindAttr = ComputeReduceKindAttr::get(builder.getContext(), *kind);
  auto result = builder.create<CommReduceScatterOp>(
      op.getLoc(), slotType, kindAttr, *input, recvBuffer.getResult(),
      builder.getI64IntegerAttr(info.axis),
      builder.getI64IntegerAttr(rankGroup->localRank),
      builder.getI64IntegerAttr(static_cast<int64_t>(rankGroup->ranks.size())),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), rankGroup->ranks),
      builder.getI64IntegerAttr(*bytes),
      builder.getI64IntegerAttr(*communicationId));
  record(op.getResult(0), MemLayout::Tensor, result.getResult());
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertAllReduce(
    LinalgExtCollectiveAllReduceOp op, const WaferLinalgExtCollectiveInfo &info,
    mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_reduce materialization supports one input and one out");
  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(info, "all_reduce");
  if (mlir::failed(communicationId))
    return mlir::failure();
  std::optional<ComputeReduceKind> kind =
      inferCollectiveReduceKind(op.getCombiner());
  if (!kind)
    return mlir::failure();

  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(info);
  if (mlir::failed(input) || mlir::failed(rankGroup))
    return mlir::failure();

  auto inputType = mlir::cast<mlir::MemRefType>((*input).getType());
  auto recvBuffer =
      builder.create<mlir::memref::AllocOp>(op.getLoc(), inputType);
  mlir::FailureOr<int64_t> bytes =
      getCompactByteSize(*input, "all_reduce input");
  if (mlir::failed(bytes))
    return mlir::failure();

  auto kindAttr = ComputeReduceKindAttr::get(builder.getContext(), *kind);
  auto result = builder.create<CommAllReduceOp>(
      op.getLoc(), inputType, kindAttr, *input, recvBuffer.getResult(),
      builder.getI64IntegerAttr(rankGroup->localRank),
      builder.getI64IntegerAttr(static_cast<int64_t>(rankGroup->ranks.size())),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), rankGroup->ranks),
      builder.getI64IntegerAttr(*bytes),
      builder.getI64IntegerAttr(*communicationId));
  record(op.getResult(0), MemLayout::Tensor, result.getResult());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllToAll(LinalgExtCollectiveAllToAllOp op,
                                       const WaferLinalgExtCollectiveInfo &info,
                                       mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_to_all materialization supports one input and one out");
  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(info, "all_to_all");
  if (mlir::failed(communicationId))
    return mlir::failure();

  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(info);
  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(rankGroup) || mlir::failed(input))
    return mlir::failure();

  int64_t groupSize = static_cast<int64_t>(rankGroup->ranks.size());
  if (groupSize <= 0 || info.splitCount != groupSize)
    return fail(
        "all_to_all materialization requires split_count to match rank_group "
        "size");
  if (!info.hasSplitAxis || !info.hasConcatAxis)
    return fail("all_to_all materialization requires split and concat axes");

  auto inputTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs().front().getType());
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!inputTensorType || !resultTensorType)
    return fail("all_to_all input and result must be ranked tensors");
  if (!inputTensorType.hasStaticShape() || !resultTensorType.hasStaticShape())
    return fail("all_to_all materialization requires static shapes");

  int64_t rank = inputTensorType.getRank();
  int64_t splitAxis = info.splitAxis;
  int64_t concatAxis = info.concatAxis;
  if (splitAxis < 0 || splitAxis >= rank || concatAxis < 0 ||
      concatAxis >= rank || resultTensorType.getRank() != rank)
    return fail("all_to_all materialization has invalid axes");

  int64_t inputSplitDim = inputTensorType.getDimSize(splitAxis);
  int64_t inputConcatDim = inputTensorType.getDimSize(concatAxis);
  if (inputSplitDim <= 0 || inputConcatDim <= 0 ||
      inputSplitDim % groupSize != 0)
    return fail("all_to_all materialization requires evenly split static "
                "dimensions");

  llvm::SmallVector<int64_t, 4> slotShape(inputTensorType.getShape().begin(),
                                          inputTensorType.getShape().end());
  slotShape[splitAxis] = inputSplitDim / groupSize;
  llvm::SmallVector<int64_t, 4> resultSlotShape(
      resultTensorType.getShape().begin(), resultTensorType.getShape().end());
  resultSlotShape[concatAxis] = inputConcatDim;
  if (slotShape != resultSlotShape)
    return fail("all_to_all materialization slot shapes do not match");

  auto slotTensorType =
      mlir::RankedTensorType::get(slotShape, inputTensorType.getElementType());
  auto slotType = makeSPMMemRefType(slotTensorType, MemLayout::Tensor);
  auto resultType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  auto resultAlloc =
      builder.create<mlir::memref::AllocOp>(op.getLoc(), resultType);
  mlir::Value resultBuffer = resultAlloc.getResult();

  llvm::SmallVector<int64_t, 4> strides(rank, 1);
  auto makeSourceOffsets = [&](int64_t targetIndex) {
    llvm::SmallVector<int64_t, 4> offsets(rank, 0);
    offsets[splitAxis] = targetIndex * slotShape[splitAxis];
    return offsets;
  };
  auto makeResultOffsets = [&](int64_t sourceIndex) {
    llvm::SmallVector<int64_t, 4> offsets(rank, 0);
    offsets[concatAxis] = sourceIndex * inputConcatDim;
    return offsets;
  };
  auto arrayAttr = [&](llvm::ArrayRef<int64_t> values) {
    return mlir::DenseI64ArrayAttr::get(builder.getContext(), values);
  };

  std::optional<WaferPhysicalTensorInfo> slotPhysicalInfo =
      computeWaferPhysicalTensorInfo(slotType);
  if (!slotPhysicalInfo || slotPhysicalInfo->compactBytes <= 0)
    return fail("all_to_all slot compact byte size is not representable");
  int64_t bytes = slotPhysicalInfo->compactBytes;

  struct PendingSend {
    mlir::Value buffer;
    int64_t targetIndex = -1;
    int64_t peer = -1;
  };
  struct PendingRecv {
    mlir::Value buffer;
    int64_t sourceIndex = -1;
    int64_t peer = -1;
  };
  llvm::SmallVector<PendingSend, 4> sends;
  llvm::SmallVector<PendingRecv, 4> recvs;

  int64_t localRank = rankGroup->localRank;
  for (int64_t targetIndex = 0; targetIndex < groupSize; ++targetIndex) {
    llvm::SmallVector<int64_t, 4> sourceOffsets =
        makeSourceOffsets(targetIndex);
    auto extract = builder.create<MoveExtractSliceOp>(
        op.getLoc(), slotType, *input, arrayAttr(sourceOffsets),
        arrayAttr(slotShape), arrayAttr(strides));

    if (targetIndex == localRank) {
      llvm::SmallVector<int64_t, 4> resultOffsets =
          makeResultOffsets(localRank);
      auto insert = builder.create<MoveInsertSliceOp>(
          op.getLoc(), resultType, extract.getResult(), resultBuffer,
          arrayAttr(resultOffsets), arrayAttr(resultSlotShape),
          arrayAttr(strides));
      resultBuffer = insert.getResult();
      continue;
    }

    sends.push_back(
        {extract.getResult(), targetIndex, rankGroup->ranks[targetIndex]});
  }

  for (int64_t sourceIndex = 0; sourceIndex < groupSize; ++sourceIndex) {
    if (sourceIndex == localRank)
      continue;
    auto recvBuffer =
        builder.create<mlir::memref::AllocOp>(op.getLoc(), slotType);
    recvs.push_back(
        {recvBuffer.getResult(), sourceIndex, rankGroup->ranks[sourceIndex]});
  }

  mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
  if (!sends.empty() || !recvs.empty())
    builder.create<SyncLocalFenceOp>(op.getLoc());
  for (int64_t distance = 1; distance < groupSize; ++distance) {
    int64_t targetIndex = (localRank + distance) % groupSize;
    int64_t sourceIndex = (localRank + groupSize - distance) % groupSize;
    auto sendIt = llvm::find_if(sends, [&](const PendingSend &send) {
      return send.targetIndex == targetIndex;
    });
    auto recvIt = llvm::find_if(recvs, [&](const PendingRecv &recv) {
      return recv.sourceIndex == sourceIndex;
    });
    if (sendIt == sends.end() || recvIt == recvs.end())
      return fail("all_to_all protocol could not recover semantic peer slot");
    auto message =
        DTEMessageAttr::get(builder.getContext(), *communicationId,
                            DTEProtocolPhase::AllToAll, distance, targetIndex);
    auto dteSend = builder.create<InstrDTESendOp>(
        op.getLoc(), tokenType, sendIt->buffer,
        builder.getI64IntegerAttr(sendIt->peer),
        builder.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
    auto recvMessage =
        DTEMessageAttr::get(builder.getContext(), *communicationId,
                            DTEProtocolPhase::AllToAll, distance, localRank);
    auto dteRecv = builder.create<InstrDTERecvOp>(
        op.getLoc(), tokenType, recvIt->buffer,
        builder.getI64IntegerAttr(recvIt->peer),
        builder.getI64IntegerAttr(bytes), recvMessage, DirectDTEBindingAttr());
    llvm::SmallVector<mlir::Value, 2> roundTokens{dteSend.getToken(),
                                                  dteRecv.getToken()};
    builder.create<InstrDTEWaitOp>(op.getLoc(), roundTokens);
  }

  for (const PendingRecv &recv : recvs) {
    llvm::SmallVector<int64_t, 4> resultOffsets =
        makeResultOffsets(recv.sourceIndex);
    auto insert = builder.create<MoveInsertSliceOp>(
        op.getLoc(), resultType, recv.buffer, resultBuffer,
        arrayAttr(resultOffsets), arrayAttr(resultSlotShape),
        arrayAttr(strides));
    resultBuffer = insert.getResult();
  }

  record(op.getResult(0), MemLayout::Tensor, resultBuffer);
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertCollectivePermute(
    LinalgExtCollectiveCollectivePermuteOp op,
    const WaferLinalgExtCollectiveInfo &info, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "collective_permute materialization supports one input and one out");
  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(info, "collective_permute");
  if (mlir::failed(communicationId))
    return mlir::failure();
  if (info.sourceTargetPairs.empty() || info.sourceTargetPairs.size() % 2)
    return fail("collective_permute materialization requires source/target "
                "pairs");

  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("collective_permute result must be ranked");
  auto resultType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);

  std::optional<int64_t> sendPeer;
  std::optional<int64_t> recvPeer;
  std::optional<int64_t> sendPayloadSlice;
  std::optional<int64_t> recvPayloadSlice;
  bool localCopy = false;
  for (size_t index = 0; index < info.sourceTargetPairs.size(); index += 2) {
    int64_t source = info.sourceTargetPairs[index];
    int64_t target = info.sourceTargetPairs[index + 1];
    if (source == currentLogicalRank) {
      if (target == currentLogicalRank)
        localCopy = true;
      else {
        sendPeer = target;
        sendPayloadSlice = static_cast<int64_t>(index / 2);
      }
    }
    if (target == currentLogicalRank && source != currentLogicalRank) {
      recvPeer = source;
      recvPayloadSlice = static_cast<int64_t>(index / 2);
    }
  }

  mlir::FailureOr<int64_t> bytes =
      getCompactByteSize(*input, "collective_permute input");
  if (mlir::failed(bytes))
    return mlir::failure();

  mlir::Value resultBuffer;
  if (localCopy) {
    resultBuffer =
        builder.create<MoveCopyOp>(op.getLoc(), resultType, *input).getResult();
  } else {
    auto alloc = builder.create<mlir::memref::AllocOp>(op.getLoc(), resultType);
    mlir::Value zero = createZeroScalar(
        op.getLoc(), resultTensorType.getElementType(), builder);
    if (!zero)
      return fail("collective_permute zero-fill requires numeric element type");
    builder.create<ComputeFillOp>(op.getLoc(), alloc.getResult(), zero,
                                  /*fill_domain=*/FillDomainAttr{});
    resultBuffer = alloc.getResult();
  }

  llvm::SmallVector<mlir::Value, 2> tokens;
  mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
  if (sendPeer) {
    auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                       DTEProtocolPhase::CollectivePermute,
                                       /*round=*/0, *sendPayloadSlice);
    auto send = builder.create<InstrDTESendOp>(
        op.getLoc(), tokenType, *input, builder.getI64IntegerAttr(*sendPeer),
        builder.getI64IntegerAttr(*bytes), message, DirectDTEBindingAttr());
    tokens.push_back(send.getToken());
  }
  if (recvPeer) {
    auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                       DTEProtocolPhase::CollectivePermute,
                                       /*round=*/0, *recvPayloadSlice);
    auto recv = builder.create<InstrDTERecvOp>(
        op.getLoc(), tokenType, resultBuffer,
        builder.getI64IntegerAttr(*recvPeer), builder.getI64IntegerAttr(*bytes),
        message, DirectDTEBindingAttr());
    tokens.push_back(recv.getToken());
  }
  if (!tokens.empty())
    builder.create<InstrDTEWaitOp>(op.getLoc(), tokens);

  record(op.getResult(0), MemLayout::Tensor, resultBuffer);
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertLinalgExtCollective(
    mlir::Operation *op, const WaferLinalgExtCollectiveInfo &info,
    mlir::OpBuilder &builder) {
  switch (info.kind) {
  case WaferLinalgExtCollectiveKind::AllGather:
    return convertAllGather(mlir::cast<LinalgExtCollectiveAllGatherOp>(op),
                            info, builder);
  case WaferLinalgExtCollectiveKind::ReduceScatter:
    return convertReduceScatter(
        mlir::cast<LinalgExtCollectiveReduceScatterOp>(op), info, builder);
  case WaferLinalgExtCollectiveKind::AllReduce:
    return convertAllReduce(mlir::cast<LinalgExtCollectiveAllReduceOp>(op),
                            info, builder);
  case WaferLinalgExtCollectiveKind::AllToAll:
    return convertAllToAll(mlir::cast<LinalgExtCollectiveAllToAllOp>(op), info,
                           builder);
  case WaferLinalgExtCollectiveKind::CollectivePermute:
    return convertCollectivePermute(
        mlir::cast<LinalgExtCollectiveCollectivePermuteOp>(op), info, builder);
  }
  llvm_unreachable("unknown linalg-ext collective kind");
}

} // namespace wafer::tensor_program_to_tile_region
