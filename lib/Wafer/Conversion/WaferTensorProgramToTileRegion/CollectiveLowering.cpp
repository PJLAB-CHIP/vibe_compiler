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

mlir::FailureOr<int64_t>
TileRegionBodyEmitter::getCommunicationId(mlir::IntegerAttr channelId,
                                          llvm::StringRef subject) {
  if (!channelId) {
    std::string reason =
        subject.str() +
        " materialization requires channel_id for stable DTE identity";
    return failI64(reason);
  }
  return channelId.getInt();
}

mlir::FailureOr<SelectedCollectiveRankGroup>
TileRegionBodyEmitter::getCollectiveRankGroup(
    mlir::DenseI64ArrayAttr rankGroup, mlir::DenseIntElementsAttr rankGroups) {
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

  if (rankGroup) {
    if (std::optional<SelectedCollectiveRankGroup> selected =
            findLocalRank(rankGroup.asArrayRef()))
      return *selected;
    return failSelectedCollectiveRankGroup(
        "logical-rank is not a member of collective rank_group");
  }

  if (!rankGroups)
    return failSelectedCollectiveRankGroup(
        "collective materialization requires rank_group or rank_groups");
  auto groupsType =
      mlir::dyn_cast<mlir::RankedTensorType>(rankGroups.getType());
  if (!groupsType || groupsType.getRank() != 2 || groupsType.getDimSize(1) <= 0)
    return failSelectedCollectiveRankGroup(
        "collective rank_groups are malformed");
  int64_t rankGroupSize = groupsType.getDimSize(1);
  llvm::SmallVector<int64_t, 8> ranks;
  for (llvm::APInt value : rankGroups.getValues<llvm::APInt>())
    ranks.push_back(value.getSExtValue());
  if (ranks.size() % static_cast<size_t>(rankGroupSize) != 0)
    return failSelectedCollectiveRankGroup(
        "collective rank_groups are malformed");

  for (size_t offset = 0; offset < ranks.size();
       offset += static_cast<size_t>(rankGroupSize)) {
    llvm::ArrayRef<int64_t> group(ranks.data() + offset,
                                  static_cast<size_t>(rankGroupSize));
    if (std::optional<SelectedCollectiveRankGroup> selected =
            findLocalRank(group))
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

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::materializeCollectiveInputInResultType(
    mlir::Value input, mlir::Type resultElementType, mlir::Location loc,
    mlir::OpBuilder &builder) {
  auto inputType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  if (!inputType)
    return failValue("collective input must materialize as a memref");
  if (inputType.getElementType() == resultElementType)
    return input;

  auto convertedTensorType = mlir::RankedTensorType::get(
      inputType.getShape(), resultElementType);
  auto converted = builder.create<ComputeConvertOp>(
      loc, makeSPMMemRefType(convertedTensorType, MemLayout::Tensor), input);
  return converted.getResult();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllGather(LinalgExtCollectiveAllGatherOp op,
                                        mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_gather materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(op.getRankGroupAttr(), op.getRankGroupsAttr());
  if (mlir::failed(rankGroup))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> localChunk =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(localChunk))
    return mlir::failure();

  // A singleton collective is the identity. Keep the already resident local
  // value instead of materializing a communication request or requiring a
  // channel that can never be used.
  if (rankGroup->ranks.size() == 1) {
    record(op.getResult(0), MemLayout::Tensor, *localChunk);
    return mlir::success();
  }

  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(op.getChannelIdAttr(), "all_gather");
  if (mlir::failed(communicationId))
    return mlir::failure();
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("all_gather result is not a ranked tensor");
  auto gatherBuffer = builder.create<mlir::memref::AllocOp>(
      op.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
  mlir::FailureOr<int64_t> bytes =
      getCompactByteSize(*localChunk, "all_gather local chunk");
  if (mlir::failed(bytes))
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
    LinalgExtCollectiveReduceScatterOp op, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "reduce_scatter materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(op.getRankGroupAttr(), op.getRankGroupsAttr());
  if (mlir::failed(rankGroup))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("reduce_scatter result must be ranked");
  input = materializeCollectiveInputInResultType(
      *input, resultTensorType.getElementType(), op.getLoc(), builder);
  if (mlir::failed(input))
    return mlir::failure();

  if (rankGroup->ranks.size() == 1) {
    record(op.getResult(0), MemLayout::Tensor, *input);
    return mlir::success();
  }

  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(op.getChannelIdAttr(), "reduce_scatter");
  if (mlir::failed(communicationId))
    return mlir::failure();
  std::optional<ComputeReduceKind> kind =
      inferCollectiveReduceKind(op.getCombiner());
  if (!kind)
    return mlir::failure();
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
      op.getAxisAttr(), builder.getI64IntegerAttr(rankGroup->localRank),
      builder.getI64IntegerAttr(static_cast<int64_t>(rankGroup->ranks.size())),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), rankGroup->ranks),
      builder.getI64IntegerAttr(*bytes),
      builder.getI64IntegerAttr(*communicationId));
  record(op.getResult(0), MemLayout::Tensor, result.getResult());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllReduce(LinalgExtCollectiveAllReduceOp op,
                                        mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_reduce materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(op.getRankGroupAttr(), op.getRankGroupsAttr());
  if (mlir::failed(rankGroup))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!resultTensorType)
    return fail("all_reduce result must be ranked");
  input = materializeCollectiveInputInResultType(
      *input, resultTensorType.getElementType(), op.getLoc(), builder);
  if (mlir::failed(input))
    return mlir::failure();

  if (rankGroup->ranks.size() == 1) {
    record(op.getResult(0), MemLayout::Tensor, *input);
    return mlir::success();
  }

  mlir::FailureOr<int64_t> communicationId =
      getCommunicationId(op.getChannelIdAttr(), "all_reduce");
  if (mlir::failed(communicationId))
    return mlir::failure();
  std::optional<ComputeReduceKind> kind =
      inferCollectiveReduceKind(op.getCombiner());
  if (!kind)
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
                                       mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_to_all materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
      getCollectiveRankGroup(op.getRankGroupAttr(), op.getRankGroupsAttr());
  mlir::FailureOr<mlir::Value> input =
      getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(rankGroup) || mlir::failed(input))
    return mlir::failure();

  int64_t groupSize = static_cast<int64_t>(rankGroup->ranks.size());
  if (groupSize <= 0 || op.getSplitCount() != groupSize)
    return fail(
        "all_to_all materialization requires split_count to match rank_group "
        "size");
  auto inputTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs().front().getType());
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
  if (!inputTensorType || !resultTensorType)
    return fail("all_to_all input and result must be ranked tensors");
  if (!inputTensorType.hasStaticShape() || !resultTensorType.hasStaticShape())
    return fail("all_to_all materialization requires static shapes");

  int64_t rank = inputTensorType.getRank();
  int64_t splitAxis = op.getSplitAxis();
  int64_t concatAxis = op.getConcatAxis();
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

  std::optional<int64_t> communicationId;
  std::optional<int64_t> bytes;
  if (!sends.empty() || !recvs.empty()) {
    mlir::FailureOr<int64_t> selectedCommunicationId =
        getCommunicationId(op.getChannelIdAttr(), "all_to_all");
    if (mlir::failed(selectedCommunicationId))
      return mlir::failure();
    communicationId = *selectedCommunicationId;

    std::optional<WaferPhysicalTensorInfo> slotPhysicalInfo =
        computeWaferPhysicalTensorInfo(slotType);
    if (!slotPhysicalInfo || slotPhysicalInfo->compactBytes <= 0)
      return fail("all_to_all slot compact byte size is not representable");
    bytes = slotPhysicalInfo->compactBytes;
  }

  mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
  // Direct all-to-all uses one globally consistent cyclic permutation of
  // semantic group indices.  It is not a ring-forwarding algorithm and must
  // not inherit the bounded exact topology-ring search.
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
    auto recvMessage =
        DTEMessageAttr::get(builder.getContext(), *communicationId,
                            DTEProtocolPhase::AllToAll, distance, localRank);
    auto dteRecv = builder.create<InstrDTERecvOp>(
        op.getLoc(), tokenType, recvIt->buffer, mlir::Value(),
        builder.getI64IntegerAttr(recvIt->peer),
        builder.getI64IntegerAttr(*bytes), recvMessage, DirectDTEBindingAttr());
    auto dteSend = builder.create<InstrDTESendOp>(
        op.getLoc(), tokenType, sendIt->buffer, mlir::Value(),
        builder.getI64IntegerAttr(sendIt->peer),
        builder.getI64IntegerAttr(*bytes), message, DirectDTEBindingAttr());
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
    LinalgExtCollectiveCollectivePermuteOp op, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
    return mlir::failure();
  if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "collective_permute materialization supports one input and one out");
  llvm::ArrayRef<int64_t> sourceTargetPairs =
      op.getSourceTargetPairsAttr().asArrayRef();
  if (sourceTargetPairs.empty() || sourceTargetPairs.size() % 2)
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
  for (size_t index = 0; index < sourceTargetPairs.size(); index += 2) {
    int64_t source = sourceTargetPairs[index];
    int64_t target = sourceTargetPairs[index + 1];
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

  mlir::Value resultBuffer;
  if (localCopy) {
    resultBuffer =
        builder.create<MoveCopyOp>(op.getLoc(), resultType, *input).getResult();
  } else {
    auto alloc = builder.create<mlir::memref::AllocOp>(op.getLoc(), resultType);
    resultBuffer = alloc.getResult();
    // Collective-permute defines zero only for ranks without an incoming
    // source.  A receive owns the whole result buffer, so initializing that
    // same storage would introduce a local-fill/DTE-write race.
    if (!recvPeer) {
      mlir::Value zero = createZeroScalar(
          op.getLoc(), resultTensorType.getElementType(), builder);
      if (!zero)
        return fail(
            "collective_permute zero-fill requires numeric element type");
      builder.create<ComputeFillOp>(op.getLoc(), resultBuffer, zero,
                                    /*fill_domain=*/FillDomainAttr{});
    }
  }

  std::optional<int64_t> communicationId;
  std::optional<int64_t> bytes;
  if (sendPeer || recvPeer) {
    mlir::FailureOr<int64_t> selectedCommunicationId =
        getCommunicationId(op.getChannelIdAttr(), "collective_permute");
    mlir::FailureOr<int64_t> selectedBytes =
        getCompactByteSize(*input, "collective_permute input");
    if (mlir::failed(selectedCommunicationId) || mlir::failed(selectedBytes))
      return mlir::failure();
    communicationId = *selectedCommunicationId;
    bytes = *selectedBytes;
  }

  llvm::SmallVector<mlir::Value, 2> tokens;
  mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
  std::optional<mlir::Value> recvToken;
  if (recvPeer) {
    auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                       DTEProtocolPhase::CollectivePermute,
                                       /*round=*/0, *recvPayloadSlice);
    auto recv = builder.create<InstrDTERecvOp>(
        op.getLoc(), tokenType, resultBuffer, mlir::Value(),
        builder.getI64IntegerAttr(*recvPeer), builder.getI64IntegerAttr(*bytes),
        message, DirectDTEBindingAttr());
    recvToken = recv.getToken();
  }
  if (sendPeer) {
    auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                       DTEProtocolPhase::CollectivePermute,
                                       /*round=*/0, *sendPayloadSlice);
    auto send = builder.create<InstrDTESendOp>(
        op.getLoc(), tokenType, *input, mlir::Value(),
        builder.getI64IntegerAttr(*sendPeer),
        builder.getI64IntegerAttr(*bytes), message, DirectDTEBindingAttr());
    tokens.push_back(send.getToken());
  }
  if (recvToken)
    tokens.push_back(*recvToken);
  if (!tokens.empty())
    builder.create<InstrDTEWaitOp>(op.getLoc(), tokens);

  record(op.getResult(0), MemLayout::Tensor, resultBuffer);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertLinalgExtCollective(mlir::Operation *op,
                                                  mlir::OpBuilder &builder) {
  if (auto allGather = mlir::dyn_cast<LinalgExtCollectiveAllGatherOp>(op))
    return convertAllGather(allGather, builder);
  if (auto reduceScatter =
          mlir::dyn_cast<LinalgExtCollectiveReduceScatterOp>(op))
    return convertReduceScatter(reduceScatter, builder);
  if (auto allReduce = mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(op))
    return convertAllReduce(allReduce, builder);
  if (auto allToAll = mlir::dyn_cast<LinalgExtCollectiveAllToAllOp>(op))
    return convertAllToAll(allToAll, builder);
  if (auto permute = mlir::dyn_cast<LinalgExtCollectiveCollectivePermuteOp>(op))
    return convertCollectivePermute(permute, builder);
  return fail("unsupported linalg-ext collective " +
              op->getName().getStringRef().str());
}

} // namespace wafer::tensor_program_to_tile_region
