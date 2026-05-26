//===- LowerRingCollectives.cpp - Expand collectives to p2p rings --------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace wafer {
namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

static std::optional<int64_t>
getPhysicalTileId(int64_t cardY, int64_t cardX, int64_t tileY, int64_t tileX,
                  int64_t cardXCount, int64_t tileYCount, int64_t tileXCount) {
  int64_t cardBase = 0;
  if (!checkedMul(cardY, cardXCount, cardBase))
    return std::nullopt;
  int64_t cardIndex = 0;
  if (!checkedAdd(cardBase, cardX, cardIndex))
    return std::nullopt;

  int64_t tileBase = 0;
  if (!checkedMul(cardIndex, tileYCount, tileBase))
    return std::nullopt;
  int64_t tileRow = 0;
  if (!checkedAdd(tileBase, tileY, tileRow))
    return std::nullopt;

  int64_t tileIdBase = 0;
  if (!checkedMul(tileRow, tileXCount, tileIdBase))
    return std::nullopt;
  int64_t tileId = 0;
  if (!checkedAdd(tileIdBase, tileX, tileId))
    return std::nullopt;
  return tileId;
}

static int64_t wrapRank(int64_t rank, int64_t groupSize) {
  int64_t wrapped = rank % groupSize;
  if (wrapped < 0)
    wrapped += groupSize;
  return wrapped;
}

static mlir::LogicalResult
collectPhysicalTileIds(mlir::ModuleOp module, mlir::Operation *anchor,
                       int64_t groupSize,
                       llvm::SmallVectorImpl<int64_t> &physicalTileIds) {
  llvm::SmallVector<wafer::PlacementMapOp, 1> placements;
  module.walk([&](wafer::PlacementMapOp placement) {
    placements.push_back(placement);
  });
  if (placements.size() != 1)
    return anchor->emitOpError(
        "ring all-gather lowering requires exactly one placement map");

  wafer::PlacementMapOp placement = placements.front();
  int64_t logicalRankCount = placement.getLogicalRankCountAttr().getInt();
  if (logicalRankCount < groupSize)
    return anchor->emitOpError(
        "ring all-gather group_size exceeds placement logical rank count");

  llvm::ArrayRef<int64_t> coords =
      placement.getPhysicalTileCoordsAttr().asArrayRef();
  int64_t cardXCount = placement.getCardXCountAttr().getInt();
  int64_t tileYCount = placement.getTileYCountAttr().getInt();
  int64_t tileXCount = placement.getTileXCountAttr().getInt();

  physicalTileIds.clear();
  physicalTileIds.reserve(groupSize);
  for (int64_t rank = 0; rank < groupSize; ++rank) {
    int64_t base = rank * 4;
    std::optional<int64_t> tileId =
        getPhysicalTileId(coords[base], coords[base + 1], coords[base + 2],
                          coords[base + 3], cardXCount, tileYCount, tileXCount);
    if (!tileId)
      return anchor->emitOpError(
          "ring all-gather placement tile id is not representable");
    physicalTileIds.push_back(*tileId);
  }
  return mlir::success();
}

static mlir::LogicalResult lowerAllGather(mlir::ModuleOp module,
                                          wafer::CommAllGatherOp allGather) {
  int64_t groupSize = allGather.getGroupSizeAttr().getInt();
  int64_t localRank = allGather.getLocalRankAttr().getInt();

  llvm::SmallVector<int64_t, 8> physicalTileIds;
  if (mlir::failed(collectPhysicalTileIds(module, allGather.getOperation(),
                                          groupSize, physicalTileIds)))
    return mlir::failure();

  int64_t nextPeer = physicalTileIds[wrapRank(localRank + 1, groupSize)];
  int64_t previousPeer = physicalTileIds[wrapRank(localRank - 1, groupSize)];

  mlir::OpBuilder builder(allGather);
  mlir::Type tokenType = mlir::async::TokenType::get(allGather.getContext());
  mlir::IntegerAttr bytesAttr = allGather.getBytesAttr();

  for (int64_t step = 0; step < groupSize - 1; ++step) {
    int64_t sendSlot = wrapRank(localRank - step, groupSize);
    int64_t recvSlot = wrapRank(localRank - step - 1, groupSize);
    mlir::Value sendBuffer =
        step == 0 ? allGather.getLocalChunk() : allGather.getGatherBuffer();

    auto send = builder.create<wafer::CommSendOp>(
        allGather.getLoc(), tokenType, sendBuffer,
        builder.getI64IntegerAttr(nextPeer), bytesAttr);
    send->setAttr(wafer::kWaferCommSlotAttrName,
                  builder.getI64IntegerAttr(sendSlot));

    auto recv = builder.create<wafer::CommRecvOp>(
        allGather.getLoc(), tokenType, allGather.getGatherBuffer(),
        builder.getI64IntegerAttr(previousPeer), bytesAttr);
    recv->setAttr(wafer::kWaferCommSlotAttrName,
                  builder.getI64IntegerAttr(recvSlot));

    llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(), recv.getToken()};
    builder.create<wafer::CommWaitOp>(allGather.getLoc(), tokens);
  }

  allGather.erase();
  return mlir::success();
}

static wafer::ComputeElementwiseKind
getElementwiseReduceKind(wafer::ComputeReduceKind kind) {
  switch (kind) {
  case wafer::ComputeReduceKind::Sum:
    return wafer::ComputeElementwiseKind::Add;
  case wafer::ComputeReduceKind::Max:
    return wafer::ComputeElementwiseKind::Max;
  case wafer::ComputeReduceKind::Min:
    return wafer::ComputeElementwiseKind::Min;
  }
  llvm_unreachable("unknown reduce kind");
}

static mlir::LogicalResult lowerReduceCollective(
    mlir::ModuleOp module, mlir::Operation *op, mlir::Value input,
    mlir::Value recvBuffer, mlir::Value result, wafer::ComputeReduceKind kind,
    mlir::IntegerAttr localRankAttr, mlir::IntegerAttr groupSizeAttr,
    mlir::IntegerAttr bytesAttr) {
  int64_t groupSize = groupSizeAttr.getInt();
  int64_t localRank = localRankAttr.getInt();

  llvm::SmallVector<int64_t, 8> physicalTileIds;
  if (mlir::failed(
          collectPhysicalTileIds(module, op, groupSize, physicalTileIds)))
    return mlir::failure();

  int64_t nextPeer = physicalTileIds[wrapRank(localRank + 1, groupSize)];
  int64_t previousPeer = physicalTileIds[wrapRank(localRank - 1, groupSize)];

  mlir::OpBuilder builder(op);
  mlir::Type tokenType = mlir::async::TokenType::get(op->getContext());
  auto elementwiseKind = wafer::ComputeElementwiseKindAttr::get(
      op->getContext(), getElementwiseReduceKind(kind));
  mlir::Value accumulator = input;

  for (int64_t step = 0; step < groupSize - 1; ++step) {
    auto send = builder.create<wafer::CommSendOp>(
        op->getLoc(), tokenType, accumulator,
        builder.getI64IntegerAttr(nextPeer), bytesAttr);
    send->setAttr(wafer::kWaferCommSlotAttrName, localRankAttr);

    auto recv = builder.create<wafer::CommRecvOp>(
        op->getLoc(), tokenType, recvBuffer,
        builder.getI64IntegerAttr(previousPeer), bytesAttr);
    recv->setAttr(wafer::kWaferCommSlotAttrName, localRankAttr);

    llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(), recv.getToken()};
    builder.create<wafer::CommWaitOp>(op->getLoc(), tokens);

    llvm::SmallVector<mlir::Value, 2> inputs{accumulator, recvBuffer};
    auto reduce = builder.create<wafer::ComputeElementwiseOp>(
        op->getLoc(), result.getType(), elementwiseKind, inputs);
    accumulator = reduce.getResult();
  }

  result.replaceAllUsesWith(accumulator);
  op->erase();
  return mlir::success();
}

struct LowerRingAllGatherPass
    : public mlir::PassWrapper<LowerRingAllGatherPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRingAllGatherPass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-ring-all-gather";
  }

  llvm::StringRef getDescription() const final {
    return "lower wafer.comm.all_gather to explicit unicast ring p2p steps";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::async::AsyncDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<wafer::CommAllGatherOp, 4> allGathers;
    module.walk([&](wafer::CommAllGatherOp allGather) {
      allGathers.push_back(allGather);
    });

    for (wafer::CommAllGatherOp allGather : allGathers) {
      if (mlir::failed(lowerAllGather(module, allGather))) {
        signalPassFailure();
        return;
      }
    }
  }
};

struct LowerRingReduceCollectivesPass
    : public mlir::PassWrapper<LowerRingReduceCollectivesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRingReduceCollectivesPass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-ring-reduce-collectives";
  }

  llvm::StringRef getDescription() const final {
    return "lower wafer.comm reduce collectives to explicit unicast ring steps";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::async::AsyncDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::Operation *, 4> collectives;
    module.walk([&](mlir::Operation *op) {
      if (mlir::isa<wafer::CommReduceScatterOp, wafer::CommAllReduceOp>(op))
        collectives.push_back(op);
    });

    for (mlir::Operation *op : collectives) {
      mlir::LogicalResult result = mlir::success();
      if (auto reduceScatter = mlir::dyn_cast<wafer::CommReduceScatterOp>(op)) {
        result = lowerReduceCollective(
            module, op, reduceScatter.getInput(), reduceScatter.getRecvBuffer(),
            reduceScatter.getResult(), reduceScatter.getKindAttr().getValue(),
            reduceScatter.getLocalRankAttr(), reduceScatter.getGroupSizeAttr(),
            reduceScatter.getBytesAttr());
      } else if (auto allReduce = mlir::dyn_cast<wafer::CommAllReduceOp>(op)) {
        result = lowerReduceCollective(
            module, op, allReduce.getInput(), allReduce.getRecvBuffer(),
            allReduce.getResult(), allReduce.getKindAttr().getValue(),
            allReduce.getLocalRankAttr(), allReduce.getGroupSizeAttr(),
            allReduce.getBytesAttr());
      }

      if (mlir::failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerRingAllGatherPass() {
  return std::make_unique<LowerRingAllGatherPass>();
}

std::unique_ptr<mlir::Pass> createLowerRingReduceCollectivesPass() {
  return std::make_unique<LowerRingReduceCollectivesPass>();
}

} // namespace wafer
