//===- CollectiveLowering.cpp - Tile-region collective lowering --------===//

#include "Internal.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

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
static int64_t getHighestTreeMask(int64_t groupSize) {
  int64_t mask = 1;
  while (mask < groupSize)
    mask <<= 1;
  return mask >> 1;
}

static int64_t getLowestSetBit(int64_t value) { return value & -value; }

class AllGatherLowering : public mlir::OpRewritePattern<CommAllGatherOp> {
public:
  AllGatherLowering(mlir::MLIRContext *context, std::string *failureReason,
                    AllGatherSchedule schedule)
      : mlir::OpRewritePattern<CommAllGatherOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommAllGatherOp op,
                  mlir::PatternRewriter &rewriter) const final {
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
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == AllGatherSchedule::Direct) {
      for (int64_t distance = 1; distance < groupSize; ++distance) {
        int64_t sendPeerIndex = (localRank + distance) % groupSize;
        int64_t recvPeerIndex = (localRank + groupSize - distance) % groupSize;
        mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvPeerIndex);
        if (mlir::failed(recvSlot))
          return mlir::failure();

        auto recvCommSlot =
            rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
        auto sendMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, localRank);
        auto recvMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, recvPeerIndex);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            localCommSlot.getResult(),
            rewriter.getI64IntegerAttr(rankGroup[sendPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), sendMessage,
            DirectDTEBindingAttr());
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            recvCommSlot.getResult(),
            rewriter.getI64IntegerAttr(rankGroup[recvPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), recvMessage,
            DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                                 recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
        if (mlir::failed(createLogicalSPMCopy(
                rewriter, op.getLoc(), op, recvCommSlot.getResult(), *recvSlot,
                failureReason, "tile.all_gather received slot copy")))
          return mlir::failure();
      }

      // DTE wait completes each receive, but the following local copy into the
      // gathered result is a separate movement-engine issue.  Complete all
      // such providers before a resident consumer can read gatherBuffer.
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      rewriter.eraseOp(op);
      return mlir::success();
    }

    int64_t nextPeer = rankGroup[(localRank + 1) % groupSize];
    int64_t prevPeer = rankGroup[(localRank + groupSize - 1) % groupSize];
    mlir::Value sendSlot = localCommSlot.getResult();
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t recvSlotIndex = (localRank + groupSize - step - 1) % groupSize;
      mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvSlotIndex);
      if (mlir::failed(recvSlot))
        return mlir::failure();

      auto recvCommSlot =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
      int64_t sendPayloadSlice = (localRank + groupSize - step) % groupSize;
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, recvSlotIndex);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), sendSlot,
          rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          recvCommSlot.getResult(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
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

    rewriter.create<SyncLocalFenceOp>(op.getLoc());
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
          failureReason, "tile.reduce_scatter");
      if (mlir::failed(view))
        return mlir::failure();
      inputSlots[slot] = *view;
      return inputSlots[slot];
    };

    mlir::FailureOr<mlir::Value> localSlot = getInputSlot(localRank);
    if (mlir::failed(localSlot))
      return mlir::failure();
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, *localSlot, *accumulator, failureReason,
            "tile.reduce_scatter accumulator init")))
      return mlir::failure();
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    switch (schedule) {
    case ReduceScatterSchedule::Direct:
      break;
    }

    int64_t bytes = op.getBytesAttr().getInt();
    for (int64_t distance = 1; distance < groupSize; ++distance) {
      int64_t sendSlotIndex = (localRank + distance) % groupSize;
      int64_t recvRankIndex = (localRank + groupSize - distance) % groupSize;
      mlir::FailureOr<mlir::Value> sendSlot = getInputSlot(sendSlotIndex);
      if (mlir::failed(sendSlot))
        return mlir::failure();

      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::ReduceScatterDirect, distance, sendSlotIndex);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::ReduceScatterDirect, distance, localRank);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *sendSlot,
          rewriter.getI64IntegerAttr(rankGroup[sendSlotIndex]),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          op.getRecvBuffer(),
          rewriter.getI64IntegerAttr(rankGroup[recvRankIndex]),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                               op.getRecvBuffer()};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, *accumulator);
      // Reduction order is an execution dependency, not just block order.
      // This also completes the final accumulator before a resident consumer.
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
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

    mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(accumulator))
      return mlir::failure();

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == AllReduceSchedule::Tree) {
      if (mlir::failed(createContiguousSPMCopy(
              rewriter, op.getLoc(), op, op.getInput(), *accumulator,
              failureReason, "tile.all_reduce accumulator init")))
        return mlir::failure();
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      for (int64_t mask = 1; mask < groupSize; mask <<= 1) {
        if ((localRank & mask) != 0) {
          int64_t parentRank = localRank ^ mask;
          auto message = DTEMessageAttr::get(
              rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
              DTEProtocolPhase::AllReduceTreeReduce, mask, localRank);
          auto send = rewriter.create<InstrDTESendOp>(
              op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
              *accumulator, rewriter.getI64IntegerAttr(rankGroup[parentRank]),
              rewriter.getI64IntegerAttr(bytes), message,
              DirectDTEBindingAttr());
          llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
          rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
          break;
        }

        int64_t childRank = localRank | mask;
        if (childRank >= groupSize)
          continue;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeReduce, mask, childRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(),
            rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

        llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                                 op.getRecvBuffer()};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator);
        rewriter.create<SyncLocalFenceOp>(op.getLoc());
      }

      bool hasFinalResult = localRank == 0;
      int64_t receiveMask = localRank == 0 ? 0 : getLowestSetBit(localRank);
      for (int64_t mask = getHighestTreeMask(groupSize); mask >= 1;
           mask >>= 1) {
        if (!hasFinalResult && receiveMask == mask) {
          int64_t parentRank = localRank ^ mask;
          auto message = DTEMessageAttr::get(
              rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
              DTEProtocolPhase::AllReduceTreeBroadcast, mask, localRank);
          auto recv = rewriter.create<InstrDTERecvOp>(
              op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
              *accumulator, rewriter.getI64IntegerAttr(rankGroup[parentRank]),
              rewriter.getI64IntegerAttr(bytes), message,
              DirectDTEBindingAttr());
          llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
          rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
          hasFinalResult = true;
          continue;
        }

        if (!hasFinalResult || (localRank & mask) != 0)
          continue;
        int64_t childRank = localRank | mask;
        if (childRank >= groupSize || getLowestSetBit(childRank) != mask)
          continue;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeBroadcast, mask, childRank);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            *accumulator, rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      }

      rewriter.replaceOp(op, *accumulator);
      return mlir::success();
    }

    auto forwardBuffer =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), inputType);
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, op.getInput(), *accumulator,
            failureReason, "tile.all_reduce accumulator init")))
      return mlir::failure();
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, op.getInput(), forwardBuffer.getResult(),
            failureReason, "tile.all_reduce forward init")))
      return mlir::failure();
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    int64_t nextPeer = rankGroup[(localRank + 1) % groupSize];
    int64_t prevPeer = rankGroup[(localRank + groupSize - 1) % groupSize];
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t sendPayloadSlice = (localRank + groupSize - step) % groupSize;
      int64_t recvPayloadSlice = (localRank + groupSize - step - 1) % groupSize;
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, recvPayloadSlice);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          forwardBuffer.getResult(), rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          op.getRecvBuffer(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                               op.getRecvBuffer()};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, *accumulator);

      if (step + 1 == groupSize - 1)
        continue;
      if (mlir::failed(createContiguousSPMCopy(
              rewriter, op.getLoc(), op, op.getRecvBuffer(),
              forwardBuffer.getResult(), failureReason,
              "tile.all_reduce forward copy")))
        return mlir::failure();
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
    }

    // The final reduction writes the value returned by the collective.  A
    // following resident consumer (for example the tensor-parallel residual
    // add) may read that same accumulator immediately, so the last local
    // elementwise issue needs an explicit completion boundary just like the
    // intermediate rounds.  DTE wait only completes the send/recv tokens; it
    // does not complete the local reduction engine.
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    rewriter.replaceOp(op, *accumulator);
    return mlir::success();
  }

private:
  std::string *failureReason;
  AllReduceSchedule schedule;
};

} // namespace

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
