//===- CollectiveLowering.cpp - Tile-region collective lowering --------===//

#include "Internal.h"

#include "Wafer/Analysis/CollectiveTopologyAnalysis.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <limits>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

template <typename PeerOp, typename InstrOp>
class PeerLowering final : public mlir::OpRewritePattern<PeerOp> {
public:
  using mlir::OpRewritePattern<PeerOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(PeerOp op, mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto lowered = rewriter.create<InstrOp>(
        op.getLoc(), rewriter.getType<mlir::async::TokenType>(), op.getBuffer(),
        mlir::Value(), op.getPeerAttr(), op.getBytesAttr(), op.getMessageAttr(),
        DirectDTEBindingAttr());
    rewriter.replaceOp(op, lowered.getToken());
    return mlir::success();
  }
};

class PeerAwaitLowering final
    : public mlir::OpRewritePattern<mlir::async::AwaitOp> {
public:
  using mlir::OpRewritePattern<mlir::async::AwaitOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::async::AwaitOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::Value token = op.getOperand();
    if (!mlir::isa<mlir::async::TokenType>(token.getType()) ||
        (!token.getDefiningOp<CommPeerSendOp>() &&
         !token.getDefiningOp<CommPeerRecvOp>() &&
         !token.getDefiningOp<InstrDTESendOp>() &&
         !token.getDefiningOp<InstrDTERecvOp>()))
      return mlir::failure();
    rewriter.create<InstrDTEWaitOp>(op.getLoc(), mlir::ValueRange{token});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static mlir::FailureOr<int64_t>
inferAllGatherAxis(mlir::PatternRewriter &rewriter, CommAllGatherOp op,
                   mlir::MemRefType localType, mlir::MemRefType gatherType,
                   std::string *failureReason) {
  if (localType.getRank() != gatherType.getRank())
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires equal-rank local and gather "
        "buffers");
  if (localType.getRank() == 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires a non-scalar gather buffer");
  if (localType.getElementType() != gatherType.getElementType())
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires matching element types");
  MemoryAttr localMemory = wafer::getWaferMemoryAttr(localType);
  MemoryAttr gatherMemory = wafer::getWaferMemoryAttr(gatherType);
  if (!localMemory || !gatherMemory ||
      localMemory.getSpace() != MemorySpace::SPM ||
      gatherMemory.getSpace() != MemorySpace::SPM ||
      localMemory.getLayout() != gatherMemory.getLayout() ||
      !isStandardViewCompatibleLayout(localMemory.getLayout()))
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires tensor or ntensor SPM layouts");

  int64_t groupSize = op.getGroupSizeAttr().getInt();
  int64_t axis = -1;
  for (int64_t dim = 0; dim < localType.getRank(); ++dim) {
    int64_t localDim = localType.getDimSize(dim);
    int64_t gatherDim = gatherType.getDimSize(dim);
    if (localDim == mlir::ShapedType::kDynamic ||
        gatherDim == mlir::ShapedType::kDynamic)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather lowering requires static buffer shapes");

    std::optional<int64_t> expectedGatherDim =
        checkedMulI64(localDim, groupSize);
    if (!expectedGatherDim)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather lowering axis size overflows");

    if (gatherDim == *expectedGatherDim) {
      if (axis >= 0)
        return failFailureOr<int64_t>(
            rewriter, op, failureReason,
            "tile.all_gather lowering requires a unique gather axis");
      axis = dim;
      continue;
    }

    if (gatherDim != localDim)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather gather buffer shape must match local shape except "
          "on the gathered axis");
  }

  if (axis < 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering could not infer gather axis");
  return axis;
}

static mlir::FailureOr<mlir::Value>
createAxisSlotView(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Operation *op, mlir::MemRefType slotType,
                   mlir::Value fullBuffer, int64_t axis, int64_t slot,
                   std::string *failureReason, llvm::StringRef opLabel) {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
  offsets.reserve(slotType.getRank());
  sizes.reserve(slotType.getRank());
  strides.reserve(slotType.getRank());

  for (int64_t dim = 0; dim < slotType.getRank(); ++dim) {
    int64_t size = slotType.getDimSize(dim);
    if (size == mlir::ShapedType::kDynamic)
      return failFailureOr<mlir::Value>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" slot view requires static local shape")
              .str());
    int64_t offset = dim == axis ? slot * size : 0;
    offsets.push_back(rewriter.getIndexAttr(offset));
    sizes.push_back(rewriter.getIndexAttr(size));
    strides.push_back(rewriter.getIndexAttr(1));
  }

  return rewriter
      .create<mlir::memref::SubViewOp>(loc, fullBuffer, offsets, sizes, strides)
      .getResult();
}

static mlir::LogicalResult
createContiguousSPMCopy(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Operation *op, mlir::Value source,
                        mlir::Value dest, std::string *failureReason,
                        llvm::StringRef role) {
  mlir::FailureOr<MovementDescriptor> sourceDescriptor =
      getContiguousDescriptor(rewriter, op, source.getType(), failureReason);
  mlir::FailureOr<MovementDescriptor> destDescriptor =
      getContiguousDescriptor(rewriter, op, dest.getType(), failureReason);
  if (mlir::failed(sourceDescriptor) || mlir::failed(destDescriptor))
    return mlir::failure();
  if (sourceDescriptor->byteCount != destDescriptor->byteCount)
    return failPattern(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires equal source and destination byte counts")
            .str());

  createGatherScatter(rewriter, loc, source, dest, *sourceDescriptor,
                      *destDescriptor);
  return mlir::success();
}

static mlir::LogicalResult
createLogicalSPMCopy(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Operation *op, mlir::Value source, mlir::Value dest,
                     std::string *failureReason, llvm::StringRef role) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(dest.getType());
  if (!sourceType || !destType)
    return failPattern(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires memref operands").str());
  mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
      getStaticLogicalMovementSegments(rewriter, op, sourceType, destType,
                                       failureReason, role);
  if (mlir::failed(segments))
    return mlir::failure();
  createGatherScatterSegments(rewriter, loc, source, dest, *segments);
  return mlir::success();
}

static mlir::MemRefType getContiguousSPMBufferType(mlir::MemRefType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               type.getMemorySpace());
}

static bool canReceiveDirectlyIntoSlot(mlir::Value slot, int64_t bytes) {
  auto slotType = mlir::dyn_cast<mlir::MemRefType>(slot.getType());
  if (!slotType)
    return false;
  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(slotType);
  return info && info->compactBytes == bytes && info->physicalBytes == bytes;
}

static mlir::FailureOr<analysis::CollectiveRingOrder>
getCollectiveRingOrder(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                       llvm::ArrayRef<int64_t> rankGroup,
                       std::string *failureReason, llvm::StringRef opLabel) {
  mlir::FailureOr<analysis::CollectiveRingOrder> order =
      analysis::buildMinimumHopCollectiveRingOrder(op, rankGroup);
  if (mlir::failed(order))
    return failFailureOr<analysis::CollectiveRingOrder>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires a topology-derived bounded rank order")
            .str());
  return order;
}

static mlir::FailureOr<int64_t>
getRingPosition(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                const analysis::CollectiveRingOrder &order,
                int64_t localGroupIndex, std::string *failureReason,
                llvm::StringRef opLabel) {
  auto position = llvm::find(order.groupIndices, localGroupIndex);
  if (position == order.groupIndices.end())
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" rank order does not contain the local rank")
            .str());
  return static_cast<int64_t>(position - order.groupIndices.begin());
}

struct RingChunking {
  int64_t axis = -1;
  int64_t bytes = 0;
  mlir::MemRefType type;
};

static std::optional<RingChunking> inferContiguousRingChunking(
    mlir::MemRefType inputType, int64_t groupSize, int64_t fullBytes,
    std::optional<int64_t> requiredAxis = std::nullopt) {
  if (inputType.getRank() == 0 || !inputType.hasStaticShape())
    return std::nullopt;

  std::optional<WaferPhysicalTensorInfo> inputInfo =
      wafer::computeWaferPhysicalTensorInfo(inputType);
  if (!inputInfo || inputInfo->elementBytes <= 0 ||
      inputInfo->bitPackedElement || inputInfo->compactBytes != fullBytes ||
      inputInfo->physicalBytes != fullBytes)
    return std::nullopt;

  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(inputType, strides, offset)) ||
      static_cast<int64_t>(strides.size()) != inputType.getRank())
    return std::nullopt;
  (void)offset;

  for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
    if (requiredAxis && axis != *requiredAxis)
      continue;
    int64_t axisSize = inputType.getDimSize(axis);
    if (axisSize <= 0 || axisSize % groupSize != 0)
      continue;

    llvm::SmallVector<int64_t> chunkShape(inputType.getShape());
    chunkShape[axis] /= groupSize;

    // A DTE message and the local reduction both consume one typed,
    // physically-contiguous chunk.  Prove compact row-major strides for this
    // particular view; size-one dimensions do not constrain their stride.
    int64_t expectedStride = 1;
    bool isContiguous = true;
    for (int64_t dim = inputType.getRank() - 1; dim >= 0; --dim) {
      int64_t size = chunkShape[dim];
      int64_t stride = strides[dim];
      if (size <= 0 || stride == mlir::ShapedType::kDynamic || stride < 0) {
        isContiguous = false;
        break;
      }
      if (size > 1 && stride != expectedStride) {
        isContiguous = false;
        break;
      }
      if (size > 0 &&
          expectedStride > std::numeric_limits<int64_t>::max() / size) {
        isContiguous = false;
        break;
      }
      expectedStride *= size;
    }
    if (!isContiguous || expectedStride <= 0 ||
        expectedStride >
            std::numeric_limits<int64_t>::max() / inputInfo->elementBytes)
      continue;

    int64_t chunkBytes = expectedStride * inputInfo->elementBytes;
    if (chunkBytes <= 0 || chunkBytes > fullBytes ||
        chunkBytes > std::numeric_limits<int64_t>::max() / groupSize ||
        chunkBytes * groupSize != fullBytes)
      continue;

    RingChunking chunking;
    chunking.axis = axis;
    chunking.bytes = chunkBytes;
    chunking.type = mlir::MemRefType::get(
        chunkShape, inputType.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, inputType.getMemorySpace());
    return chunking;
  }

  return std::nullopt;
}

static bool supportsRingElementType(mlir::MemRefType type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type.getElementType());
}

class AllGatherLowering : public mlir::OpRewritePattern<CommAllGatherOp> {
public:
  AllGatherLowering(mlir::MLIRContext *context, std::string *failureReason,
                    AllGatherSchedule schedule)
      : mlir::OpRewritePattern<CommAllGatherOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommAllGatherOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto localType =
        mlir::dyn_cast<mlir::MemRefType>(op.getLocalChunk().getType());
    auto gatherType =
        mlir::dyn_cast<mlir::MemRefType>(op.getGatherBuffer().getType());
    if (!localType || !gatherType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_gather lowering requires memref buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_gather lowering requires valid rank facts");
    mlir::FailureOr<int64_t> axis =
        inferAllGatherAxis(rewriter, op, localType, gatherType, failureReason);
    if (mlir::failed(axis))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> slots(groupSize);
    auto getSlot = [&](int64_t slot) -> mlir::FailureOr<mlir::Value> {
      if (slots[slot])
        return slots[slot];
      mlir::FailureOr<mlir::Value> view = createAxisSlotView(
          rewriter, op.getLoc(), op, localType, op.getGatherBuffer(), *axis,
          slot, failureReason, "tile.all_gather");
      if (mlir::failed(view))
        return mlir::failure();
      slots[slot] = *view;
      return slots[slot];
    };

    mlir::FailureOr<mlir::Value> localSlot = getSlot(localRank);
    if (mlir::failed(localSlot))
      return mlir::failure();

    mlir::MemRefType commSlotType = getContiguousSPMBufferType(localType);
    auto localCommSlot =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
    if (mlir::failed(
            createLogicalSPMCopy(rewriter, op.getLoc(), op, op.getLocalChunk(),
                                 localCommSlot.getResult(), failureReason,
                                 "tile.all_gather local contiguous copy")))
      return mlir::failure();
    if (mlir::failed(createLogicalSPMCopy(
            rewriter, op.getLoc(), op, localCommSlot.getResult(), *localSlot,
            failureReason, "tile.all_gather local slot copy")))
      return mlir::failure();

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == AllGatherSchedule::Direct) {
      // Direct exchange is defined over semantic group indices.  Every rank
      // uses the same cyclic round, independently of topology search limits.
      for (int64_t distance = 1; distance < groupSize; ++distance) {
        int64_t sendPeerIndex = (localRank + distance) % groupSize;
        int64_t recvPeerIndex = (localRank + groupSize - distance) % groupSize;
        mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvPeerIndex);
        if (mlir::failed(recvSlot))
          return mlir::failure();

        mlir::Value recvBuffer = *recvSlot;
        if (!canReceiveDirectlyIntoSlot(recvBuffer, bytes))
          recvBuffer =
              rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType)
                  .getResult();
        auto sendMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, localRank);
        auto recvMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, recvPeerIndex);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(), recvBuffer,
            mlir::Value(), rewriter.getI64IntegerAttr(rankGroup[recvPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), recvMessage,
            DirectDTEBindingAttr());
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            localCommSlot.getResult(), mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[sendPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), sendMessage,
            DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                                 recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
        if (recvBuffer != *recvSlot &&
            mlir::failed(createLogicalSPMCopy(
                rewriter, op.getLoc(), op, recvBuffer, *recvSlot, failureReason,
                "tile.all_gather received slot copy")))
          return mlir::failure();
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::FailureOr<analysis::CollectiveRingOrder> ringOrder =
        getCollectiveRingOrder(rewriter, op, rankGroup, failureReason,
                               "tile.all_gather ring lowering");
    if (mlir::failed(ringOrder))
      return mlir::failure();
    mlir::FailureOr<int64_t> ringPosition =
        getRingPosition(rewriter, op, *ringOrder, localRank, failureReason,
                        "tile.all_gather ring lowering");
    if (mlir::failed(ringPosition))
      return mlir::failure();
    llvm::ArrayRef<int64_t> orderedGroupIndices = ringOrder->groupIndices;
    int64_t nextPeerIndex =
        orderedGroupIndices[(*ringPosition + 1) % groupSize];
    int64_t prevPeerIndex =
        orderedGroupIndices[(*ringPosition + groupSize - 1) % groupSize];
    int64_t nextPeer = rankGroup[nextPeerIndex];
    int64_t prevPeer = rankGroup[prevPeerIndex];
    mlir::Value sendSlot = localCommSlot.getResult();
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t sendPayloadSlice =
          orderedGroupIndices[(*ringPosition + groupSize - step) % groupSize];
      int64_t recvSlotIndex =
          orderedGroupIndices[(*ringPosition + groupSize - step - 1) %
                              groupSize];
      mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvSlotIndex);
      if (mlir::failed(recvSlot))
        return mlir::failure();

      auto recvCommSlot =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, recvSlotIndex);
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          recvCommSlot.getResult(), mlir::Value(),
          rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), sendSlot,
          mlir::Value(), rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      if (mlir::failed(createLogicalSPMCopy(
              rewriter, op.getLoc(), op, recvCommSlot.getResult(), *recvSlot,
              failureReason, "tile.all_gather received slot copy")))
        return mlir::failure();
      sendSlot = recvCommSlot.getResult();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  std::string *failureReason;
  AllGatherSchedule schedule;
};

class ReduceScatterLowering
    : public mlir::OpRewritePattern<CommReduceScatterOp> {
public:
  ReduceScatterLowering(mlir::MLIRContext *context, std::string *failureReason,
                        ReduceScatterSchedule schedule)
      : mlir::OpRewritePattern<CommReduceScatterOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommReduceScatterOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto recvType =
        mlir::dyn_cast<mlir::MemRefType>(op.getRecvBuffer().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !recvType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires memref "
                         "buffers");
    if (recvType != resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires matching "
                         "recv/result buffer types");

    MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
    MemoryAttr recvMemory = wafer::getWaferMemoryAttr(recvType);
    MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    if (!inputMemory || !recvMemory || !resultMemory ||
        inputMemory.getSpace() != MemorySpace::SPM ||
        recvMemory.getSpace() != MemorySpace::SPM ||
        resultMemory.getSpace() != MemorySpace::SPM ||
        inputMemory.getLayout() != MemLayout::Tensor ||
        recvMemory.getLayout() != MemLayout::Tensor ||
        resultMemory.getLayout() != MemLayout::Tensor)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires tensor SPM "
                         "buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires valid rank "
                         "facts");
    int64_t axis = op.getAxisAttr().getInt();
    if (axis < 0 || axis >= inputType.getRank() ||
        inputType.getRank() != resultType.getRank())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires valid axis");
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      int64_t inputDim = inputType.getDimSize(dim);
      int64_t resultDim = resultType.getDimSize(dim);
      if (inputDim == mlir::ShapedType::kDynamic ||
          resultDim == mlir::ShapedType::kDynamic)
        return failPattern(rewriter, op, failureReason,
                           "tile.reduce_scatter lowering requires static "
                           "buffer shapes");
      if (dim == axis) {
        std::optional<int64_t> expectedInputDim =
            checkedMulI64(resultDim, groupSize);
        if (!expectedInputDim || inputDim != *expectedInputDim)
          return failPattern(rewriter, op, failureReason,
                             "tile.reduce_scatter input axis size must equal "
                             "result axis size times group_size");
        continue;
      }
      if (inputDim != resultDim)
        return failPattern(rewriter, op, failureReason,
                           "tile.reduce_scatter non-axis dimensions must "
                           "match");
    }

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.reduce_scatter");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == ReduceScatterSchedule::Ring) {
      if (!supportsRingElementType(inputType))
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce_scatter ring requires integer or floating-point "
            "elements");
      mlir::FailureOr<analysis::CollectiveRingOrder> ringOrder =
          getCollectiveRingOrder(rewriter, op, rankGroup, failureReason,
                                 "tile.reduce_scatter ring lowering");
      if (mlir::failed(ringOrder))
        return mlir::failure();
      mlir::FailureOr<int64_t> ringPosition =
          getRingPosition(rewriter, op, *ringOrder, localRank, failureReason,
                          "tile.reduce_scatter ring lowering");
      if (mlir::failed(ringPosition))
        return mlir::failure();
      llvm::ArrayRef<int64_t> orderedGroupIndices = ringOrder->groupIndices;

      std::optional<int64_t> fullBytes = checkedMulI64(bytes, groupSize);
      std::optional<RingChunking> chunking;
      if (fullBytes)
        chunking =
            inferContiguousRingChunking(inputType, groupSize, *fullBytes, axis);
      if (!chunking || chunking->axis != axis || chunking->bytes != bytes ||
          chunking->type != resultType)
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce_scatter ring requires a contiguous axis partition "
            "matching the result slot");

      mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
          op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
      if (mlir::failed(accumulator))
        return mlir::failure();

      llvm::SmallVector<mlir::Value> inputSlots(groupSize);
      auto getInputSlot = [&](int64_t slot) -> mlir::FailureOr<mlir::Value> {
        if (inputSlots[slot])
          return inputSlots[slot];
        mlir::FailureOr<mlir::Value> view = createAxisSlotView(
            rewriter, op.getLoc(), op, resultType, op.getInput(), axis, slot,
            failureReason, "tile.reduce_scatter ring");
        if (mlir::failed(view))
          return mlir::failure();
        inputSlots[slot] = *view;
        return inputSlots[slot];
      };

      int64_t nextPeerIndex =
          orderedGroupIndices[(*ringPosition + 1) % groupSize];
      int64_t prevPeerIndex =
          orderedGroupIndices[(*ringPosition + groupSize - 1) % groupSize];
      int64_t nextPeer = rankGroup[nextPeerIndex];
      int64_t prevPeer = rankGroup[prevPeerIndex];

      mlir::Value sendBuffer;
      for (int64_t step = 0; step < groupSize - 1; ++step) {
        int64_t sendPayloadSlice =
            orderedGroupIndices[(*ringPosition + groupSize - step - 1) %
                                groupSize];
        int64_t recvPayloadSlice =
            orderedGroupIndices[(*ringPosition + groupSize - step - 2) %
                                groupSize];
        if (step == 0) {
          mlir::FailureOr<mlir::Value> firstSendSlot =
              getInputSlot(sendPayloadSlice);
          if (mlir::failed(firstSendSlot))
            return mlir::failure();
          sendBuffer = *firstSendSlot;
        } else {
          sendBuffer = *accumulator;
        }
        mlir::FailureOr<mlir::Value> localInputSlot =
            getInputSlot(recvPayloadSlice);
        if (mlir::failed(localInputSlot))
          return mlir::failure();

        auto sendMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::ReduceScatterRing, step, sendPayloadSlice);
        auto recvMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::ReduceScatterRing, step, recvPayloadSlice);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(), mlir::Value(),
            rewriter.getI64IntegerAttr(prevPeer),
            rewriter.getI64IntegerAttr(bytes), recvMessage,
            DirectDTEBindingAttr());
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(), sendBuffer,
            mlir::Value(), rewriter.getI64IntegerAttr(nextPeer),
            rewriter.getI64IntegerAttr(bytes), sendMessage,
            DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                                 recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

        llvm::SmallVector<mlir::Value, 2> inputs{*localInputSlot,
                                                 op.getRecvBuffer()};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator,
                                            getDefaultNCCWorkerAttr(rewriter));
      }

      rewriter.replaceOp(op, *accumulator);
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value> inputSlots(groupSize);
    auto getInputSlot = [&](int64_t slot) -> mlir::FailureOr<mlir::Value> {
      if (inputSlots[slot])
        return inputSlots[slot];
      mlir::FailureOr<mlir::Value> view = createAxisSlotView(
          rewriter, op.getLoc(), op, resultType, op.getInput(), axis, slot,
          failureReason, "tile.reduce_scatter");
      if (mlir::failed(view))
        return mlir::failure();
      inputSlots[slot] = *view;
      return inputSlots[slot];
    };

    mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(accumulator))
      return mlir::failure();

    // Every destination consumes source-rank contributions in rank_group
    // order.  Source s sends its destination-specific slice to all other
    // ranks during round s; its own destination uses the local slice.  This
    // implements StableHLO reduce_scatter as ordered all_reduce followed by
    // split, while remaining independent of the bounded topology-ring search.
    for (int64_t sourceIndex = 0; sourceIndex < groupSize; ++sourceIndex) {
      mlir::Value contribution;
      if (sourceIndex == localRank) {
        for (int64_t targetIndex = 0; targetIndex < groupSize; ++targetIndex) {
          mlir::FailureOr<mlir::Value> sourceSlot = getInputSlot(targetIndex);
          if (mlir::failed(sourceSlot))
            return mlir::failure();
          if (targetIndex == localRank) {
            contribution = *sourceSlot;
            continue;
          }
          auto message = DTEMessageAttr::get(
              rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
              DTEProtocolPhase::ReduceScatterDirect, sourceIndex, targetIndex);
          auto send = rewriter.create<InstrDTESendOp>(
              op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
              *sourceSlot, mlir::Value(),
              rewriter.getI64IntegerAttr(rankGroup[targetIndex]),
              rewriter.getI64IntegerAttr(bytes), message,
              DirectDTEBindingAttr());
          // The normal Direct-DTE allocation profile permits one live sender
          // per rank block.  Each target slot is a view of the same gathered
          // input root, so complete one send before issuing the next instead
          // of relying on disjoint subview ranges to bypass root isolation.
          llvm::SmallVector<mlir::Value, 1> sendTokens{send.getToken()};
          rewriter.create<InstrDTEWaitOp>(op.getLoc(), sendTokens);
        }
      } else {
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::ReduceScatterDirect, sourceIndex, localRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(), mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[sourceIndex]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> recvTokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), recvTokens);
        contribution = op.getRecvBuffer();
      }

      if (sourceIndex == 0) {
        if (mlir::failed(createContiguousSPMCopy(
                rewriter, op.getLoc(), op, contribution, *accumulator,
                failureReason, "tile.reduce_scatter accumulator init")))
          return mlir::failure();
      } else {
        llvm::SmallVector<mlir::Value, 2> inputs{*accumulator, contribution};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator,
                                            getDefaultNCCWorkerAttr(rewriter));
      }
    }

    rewriter.replaceOp(op, *accumulator);
    return mlir::success();
  }

private:
  std::string *failureReason;
  ReduceScatterSchedule schedule;
};

class AllReduceLowering : public mlir::OpRewritePattern<CommAllReduceOp> {
public:
  AllReduceLowering(mlir::MLIRContext *context, std::string *failureReason,
                    AllReduceSchedule schedule)
      : mlir::OpRewritePattern<CommAllReduceOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommAllReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto recvType =
        mlir::dyn_cast<mlir::MemRefType>(op.getRecvBuffer().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !recvType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires memref buffers");
    if (inputType != recvType || inputType != resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires matching buffer "
                         "types");

    MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
    if (!inputMemory || inputMemory.getSpace() != MemorySpace::SPM ||
        inputMemory.getLayout() != MemLayout::Tensor)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires tensor SPM "
                         "buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires valid rank facts");

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.all_reduce");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    int64_t bytes = op.getBytesAttr().getInt();
    std::optional<RingChunking> chunking =
        inferContiguousRingChunking(inputType, groupSize, bytes);
    bool ringElementTypeSupported = supportsRingElementType(inputType);
    std::optional<analysis::CollectiveRingOrder> ringOrder;
    if (schedule != AllReduceSchedule::Tree && ringElementTypeSupported &&
        chunking) {
      mlir::FailureOr<analysis::CollectiveRingOrder> candidateOrder =
          analysis::buildMinimumHopCollectiveRingOrder(op, rankGroup);
      if (mlir::succeeded(candidateOrder))
        ringOrder = std::move(*candidateOrder);
    }
    bool useTree = schedule == AllReduceSchedule::Tree ||
                   (schedule == AllReduceSchedule::Auto &&
                    (!ringElementTypeSupported || !chunking || !ringOrder));
    if (useTree) {
      mlir::FailureOr<analysis::CollectiveTree> tree =
          analysis::buildMinimumHopCollectiveTree(op, rankGroup);
      if (mlir::failed(tree))
        return failPattern(
            rewriter, op, failureReason,
            "tile.all_reduce tree requires topology-derived rank edges");
      mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
          op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
      if (mlir::failed(accumulator))
        return mlir::failure();
      llvm::ArrayRef<int64_t> children =
          tree->childGroupIndices[static_cast<size_t>(localRank)];
      if (children.size() > 2)
        return failPattern(
            rewriter, op, failureReason,
            "tile.all_reduce ordered tree has more than two children");
      std::optional<int64_t> leftChild;
      std::optional<int64_t> rightChild;
      for (int64_t childRank : children) {
        if (childRank < localRank) {
          if (leftChild)
            return failPattern(
                rewriter, op, failureReason,
                "tile.all_reduce ordered tree has multiple left children");
          leftChild = childRank;
        } else if (childRank > localRank) {
          if (rightChild)
            return failPattern(
                rewriter, op, failureReason,
                "tile.all_reduce ordered tree has multiple right children");
          rightChild = childRank;
        } else {
          return failPattern(
              rewriter, op, failureReason,
              "tile.all_reduce ordered tree contains a self child");
        }
      }

      // StableHLO requires the reduction tree's inorder traversal to match
      // rank_group.  Build this node as
      //   left-subtree result, local operand, right-subtree result.
      if (leftChild) {
        int64_t childRank = *leftChild;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeReduce,
            tree->depths[static_cast<size_t>(childRank)], childRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(), mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

        llvm::SmallVector<mlir::Value, 2> inputs{op.getRecvBuffer(),
                                                 op.getInput()};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator,
                                            getDefaultNCCWorkerAttr(rewriter));
      } else {
        if (mlir::failed(createContiguousSPMCopy(
                rewriter, op.getLoc(), op, op.getInput(), *accumulator,
                failureReason, "tile.all_reduce accumulator init")))
          return mlir::failure();
      }

      if (rightChild) {
        int64_t childRank = *rightChild;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeReduce,
            tree->depths[static_cast<size_t>(childRank)], childRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(), mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

        llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                                 op.getRecvBuffer()};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator,
                                            getDefaultNCCWorkerAttr(rewriter));
      }
      if (localRank != tree->rootGroupIndex) {
        int64_t parentRank =
            tree->parentGroupIndices[static_cast<size_t>(localRank)];
        if (parentRank < 0)
          return failPattern(
              rewriter, op, failureReason,
              "tile.all_reduce tree is missing a non-root parent");
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeReduce,
            tree->depths[static_cast<size_t>(localRank)], localRank);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            *accumulator, mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[parentRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      }

      if (localRank != tree->rootGroupIndex) {
        int64_t parentRank =
            tree->parentGroupIndices[static_cast<size_t>(localRank)];
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeBroadcast,
            tree->depths[static_cast<size_t>(localRank)], localRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            *accumulator, mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[parentRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      }
      for (int64_t childRank : children) {
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeBroadcast,
            tree->depths[static_cast<size_t>(childRank)], childRank);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            *accumulator, mlir::Value(),
            rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      }
      rewriter.replaceOp(op, *accumulator);
      return mlir::success();
    }

    if (!ringElementTypeSupported)
      return failPattern(
          rewriter, op, failureReason,
          "tile.all_reduce ring requires integer or floating-point elements");
    if (!chunking)
      return failPattern(
          rewriter, op, failureReason,
          "tile.all_reduce ring requires an evenly divisible contiguous "
          "axis");
    if (!ringOrder)
      return failPattern(
          rewriter, op, failureReason,
          "tile.all_reduce ring requires a topology-derived bounded rank "
          "order");
    mlir::FailureOr<int64_t> ringPosition =
        getRingPosition(rewriter, op, *ringOrder, localRank, failureReason,
                        "tile.all_reduce ring");
    if (mlir::failed(ringPosition))
      return mlir::failure();
    llvm::ArrayRef<int64_t> orderedGroupIndices = ringOrder->groupIndices;

    mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(accumulator))
      return mlir::failure();
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, op.getInput(), *accumulator,
            failureReason, "tile.all_reduce accumulator init")))
      return mlir::failure();
    llvm::SmallVector<mlir::Value> accumulatorChunks(groupSize);
    llvm::SmallVector<mlir::Value> recvChunks(groupSize);
    auto getChunk = [&](mlir::Value fullBuffer,
                        llvm::SmallVectorImpl<mlir::Value> &chunks,
                        int64_t chunk) -> mlir::FailureOr<mlir::Value> {
      if (chunks[chunk])
        return chunks[chunk];
      mlir::FailureOr<mlir::Value> view = createAxisSlotView(
          rewriter, op.getLoc(), op, chunking->type, fullBuffer, chunking->axis,
          chunk, failureReason, "tile.all_reduce ring");
      if (mlir::failed(view))
        return mlir::failure();
      chunks[chunk] = *view;
      return chunks[chunk];
    };

    int64_t nextPeerIndex =
        orderedGroupIndices[(*ringPosition + 1) % groupSize];
    int64_t prevPeerIndex =
        orderedGroupIndices[(*ringPosition + groupSize - 1) % groupSize];
    int64_t nextPeer = rankGroup[nextPeerIndex];
    int64_t prevPeer = rankGroup[prevPeerIndex];

    // Reduce-scatter.  The accumulator starts with this rank's full input.
    // Each round forwards one partial chunk and reduces the predecessor's
    // contribution into the next chunk.  The wait completes DTE before local
    // compute. Same-worker issue order carries the local dependency until the
    // chunk crosses into the DTE domain.
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t sendPayloadSlice =
          orderedGroupIndices[(*ringPosition + groupSize - step) % groupSize];
      int64_t recvPayloadSlice =
          orderedGroupIndices[(*ringPosition + groupSize - step - 1) %
                              groupSize];
      mlir::FailureOr<mlir::Value> sendChunk =
          getChunk(*accumulator, accumulatorChunks, sendPayloadSlice);
      mlir::FailureOr<mlir::Value> recvChunk =
          getChunk(op.getRecvBuffer(), recvChunks, recvPayloadSlice);
      mlir::FailureOr<mlir::Value> accumulatorChunk =
          getChunk(*accumulator, accumulatorChunks, recvPayloadSlice);
      if (mlir::failed(sendChunk) || mlir::failed(recvChunk) ||
          mlir::failed(accumulatorChunk))
        return mlir::failure();

      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, recvPayloadSlice);
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *recvChunk,
          mlir::Value(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(chunking->bytes), recvMessage,
          DirectDTEBindingAttr());
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *sendChunk,
          mlir::Value(), rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(chunking->bytes), sendMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      llvm::SmallVector<mlir::Value, 2> inputs{*accumulatorChunk, *recvChunk};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, *accumulatorChunk,
                                          getDefaultNCCWorkerAttr(rewriter));
    }

    // All-gather.  Reduce-scatter leaves rank r owning reduced chunk r + 1
    // (mod P) for the chosen ring orientation.  Each round forwards the most
    // recently acquired complete chunk and receives the next one directly
    // into its final accumulator slot.  Continue round numbering across both
    // phases so DTE message identity remains globally unique.
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t sendPayloadSlice =
          orderedGroupIndices[(*ringPosition + 1 + groupSize - step) %
                              groupSize];
      int64_t recvPayloadSlice =
          orderedGroupIndices[(*ringPosition + groupSize - step) % groupSize];
      mlir::FailureOr<mlir::Value> sendChunk =
          getChunk(*accumulator, accumulatorChunks, sendPayloadSlice);
      mlir::FailureOr<mlir::Value> recvChunk =
          getChunk(op.getRecvBuffer(), recvChunks, recvPayloadSlice);
      mlir::FailureOr<mlir::Value> accumulatorChunk =
          getChunk(*accumulator, accumulatorChunks, recvPayloadSlice);
      if (mlir::failed(sendChunk) || mlir::failed(recvChunk) ||
          mlir::failed(accumulatorChunk))
        return mlir::failure();

      int64_t round = groupSize - 1 + step;
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, round, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, round, recvPayloadSlice);
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *recvChunk,
          mlir::Value(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(chunking->bytes), recvMessage,
          DirectDTEBindingAttr());
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *sendChunk,
          mlir::Value(), rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(chunking->bytes), sendMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      // Direct DTE requires both issue roots to remain isolated until the
      // joint wait. Receive into the dedicated communication buffer, then
      // publish the completed chunk into its final accumulator slot.
      if (mlir::failed(createContiguousSPMCopy(
              rewriter, op.getLoc(), op, *recvChunk, *accumulatorChunk,
              failureReason, "tile.all_reduce ring all-gather receive copy")))
        return mlir::failure();
    }

    rewriter.replaceOp(op, *accumulator);
    return mlir::success();
  }

private:
  std::string *failureReason;
  AllReduceSchedule schedule;
};

} // namespace

void wafer::tile_region_to_instr::populatePeerLoweringPatterns(
    mlir::RewritePatternSet &patterns, std::string *failureReason) {
  (void)failureReason;
  patterns.add<PeerLowering<CommPeerSendOp, InstrDTESendOp>,
               PeerLowering<CommPeerRecvOp, InstrDTERecvOp>, PeerAwaitLowering>(
      patterns.getContext());
}

void wafer::tile_region_to_instr::populateCollectiveLoweringPatterns(
    mlir::RewritePatternSet &patterns, const TileRegionToInstrOptions &options,
    std::string *failureReason) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<AllGatherLowering>(context, failureReason,
                                  options.allGatherSchedule);
  patterns.add<ReduceScatterLowering>(context, failureReason,
                                      options.reduceScatterSchedule);
  patterns.add<AllReduceLowering>(context, failureReason,
                                  options.allReduceSchedule);
}
