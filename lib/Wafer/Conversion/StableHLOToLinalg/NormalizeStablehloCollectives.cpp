//===- NormalizeStablehloCollectives.cpp - StableHLO collective handoff ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
#define GEN_PASS_DEF_NORMALIZESTABLEHLOCOLLECTIVESPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

#ifdef WAFER_ENABLE_STABLEHLO
static mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType type) {
  return builder
      .create<mlir::tensor::EmptyOp>(loc, type.getShape(),
                                     type.getElementType())
      .getResult();
}

static std::optional<llvm::SmallVector<int64_t>>
getSingleReplicaGroup(mlir::DenseIntElementsAttr replicaGroups) {
  auto groupsType = mlir::cast<mlir::RankedTensorType>(replicaGroups.getType());
  if (groupsType.getRank() == 2 && groupsType.getDimSize(0) != 1)
    return std::nullopt;

  llvm::SmallVector<int64_t> ranks;
  for (llvm::APInt value : replicaGroups.getValues<llvm::APInt>()) {
    if (!value.isSignedIntN(63))
      return std::nullopt;
    ranks.push_back(value.getSExtValue());
  }
  return ranks;
}

static mlir::DenseI64ArrayAttr getRankGroupAttr(mlir::OpBuilder &builder,
                                                llvm::ArrayRef<int64_t> ranks) {
  return mlir::DenseI64ArrayAttr::get(builder.getContext(), ranks);
}

static mlir::IntegerAttr
getChannelIdAttr(mlir::OpBuilder &builder,
                 mlir::stablehlo::ChannelHandleAttr channelHandle) {
  if (!channelHandle)
    return {};
  return builder.getI64IntegerAttr(channelHandle.getHandle());
}

static mlir::BoolAttr getUseGlobalDeviceIdsAttr(mlir::OpBuilder &builder,
                                                bool useGlobalDeviceIds) {
  if (!useGlobalDeviceIds)
    return {};
  return builder.getBoolAttr(true);
}

static std::optional<llvm::SmallVector<mlir::Value>>
createDestinationTensors(mlir::OpBuilder &builder, mlir::Operation *op) {
  llvm::SmallVector<mlir::Value> outs;
  for (mlir::Type resultType : op->getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape())
      return std::nullopt;
    outs.push_back(createEmptyTensor(builder, op->getLoc(), tensorType));
  }
  return outs;
}

static void replaceAndErase(mlir::Operation *oldOp, mlir::Operation *newOp) {
  for (auto [oldResult, newResult] :
       llvm::zip(oldOp->getResults(), newOp->getResults()))
    oldResult.replaceAllUsesWith(newResult);
  oldOp->erase();
}

static bool areMappedBlockArguments(mlir::Value lhs, mlir::Value rhs,
                                    mlir::BlockArgument sourceLhs,
                                    mlir::BlockArgument sourceRhs) {
  return (lhs == sourceLhs && rhs == sourceRhs) ||
         (lhs == sourceRhs && rhs == sourceLhs);
}

static mlir::Value buildScalarAdd(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs) {
  mlir::Type type = lhs.getType();
  if (mlir::isa<mlir::FloatType>(type))
    return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  if (mlir::isa<mlir::IntegerType>(type))
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
  return {};
}

static mlir::Value buildScalarMax(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs) {
  mlir::Type type = lhs.getType();
  if (mlir::isa<mlir::FloatType>(type))
    return builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
  if (mlir::isa<mlir::IntegerType>(type))
    return builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
  return {};
}

static mlir::Value buildScalarMin(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs) {
  mlir::Type type = lhs.getType();
  if (mlir::isa<mlir::FloatType>(type))
    return builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs);
  if (mlir::isa<mlir::IntegerType>(type))
    return builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs);
  return {};
}

static mlir::Value convertReturnedCombinerValue(
    mlir::OpBuilder &builder, mlir::Value returnedValue,
    mlir::BlockArgument sourceLhs, mlir::BlockArgument sourceRhs,
    const llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) {
  if (auto add = returnedValue.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (!areMappedBlockArguments(add.getLhs(), add.getRhs(), sourceLhs,
                                 sourceRhs))
      return {};
    return buildScalarAdd(builder, add.getLoc(), valueMap.lookup(add.getLhs()),
                          valueMap.lookup(add.getRhs()));
  }
  if (auto max = returnedValue.getDefiningOp<mlir::stablehlo::MaxOp>()) {
    if (!areMappedBlockArguments(max.getLhs(), max.getRhs(), sourceLhs,
                                 sourceRhs))
      return {};
    return buildScalarMax(builder, max.getLoc(), valueMap.lookup(max.getLhs()),
                          valueMap.lookup(max.getRhs()));
  }
  if (auto min = returnedValue.getDefiningOp<mlir::stablehlo::MinOp>()) {
    if (!areMappedBlockArguments(min.getLhs(), min.getRhs(), sourceLhs,
                                 sourceRhs))
      return {};
    return buildScalarMin(builder, min.getLoc(), valueMap.lookup(min.getLhs()),
                          valueMap.lookup(min.getRhs()));
  }
  return {};
}

static mlir::LogicalResult convertCombinerRegion(mlir::Region &sourceRegion,
                                                 mlir::Region &destRegion,
                                                 mlir::ValueRange inputs,
                                                 mlir::ResultRange results) {
  if (!sourceRegion.hasOneBlock())
    return mlir::failure();
  mlir::Block &sourceBlock = sourceRegion.front();
  if (sourceBlock.getNumArguments() != inputs.size() * 2)
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(sourceBlock.getNumArguments());
  for (mlir::Value input : inputs) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return mlir::failure();
    argTypes.push_back(inputType.getElementType());
  }
  for (mlir::Value input : inputs) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    argTypes.push_back(inputType.getElementType());
  }

  auto sourceReturn =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(sourceBlock.getTerminator());
  if (!sourceReturn || sourceReturn.getResults().size() != results.size())
    return mlir::failure();

  destRegion.push_back(new mlir::Block);
  mlir::Block &destBlock = destRegion.front();
  mlir::Location argLoc = sourceRegion.getParentOp()->getLoc();
  for (mlir::Type argType : argTypes)
    destBlock.addArgument(argType, argLoc);

  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  for (auto [sourceArg, destArg] :
       llvm::zip(sourceBlock.getArguments(), destBlock.getArguments()))
    valueMap[sourceArg] = destArg;

  mlir::OpBuilder nestedBuilder(&destBlock, destBlock.end());
  llvm::SmallVector<mlir::Value> yielded;
  yielded.reserve(sourceReturn.getResults().size());
  for (auto [index, returnedValue] :
       llvm::enumerate(sourceReturn.getResults())) {
    mlir::Value converted = convertReturnedCombinerValue(
        nestedBuilder, returnedValue, sourceBlock.getArgument(index),
        sourceBlock.getArgument(index + inputs.size()), valueMap);
    if (!converted)
      return mlir::failure();
    yielded.push_back(converted);
  }
  nestedBuilder.create<LinalgExtCollectiveYieldOp>(sourceReturn.getLoc(), yielded);
  return mlir::success();
}

static void eraseUnusedDestinationTensors(llvm::ArrayRef<mlir::Value> outs) {
  for (mlir::Value out : outs) {
    if (!out.use_empty())
      continue;
    if (auto empty = out.getDefiningOp<mlir::tensor::EmptyOp>())
      empty.erase();
  }
}

static bool lowerAllGather(mlir::stablehlo::AllGatherOp allGather) {
  auto rankGroup = getSingleReplicaGroup(allGather.getReplicaGroups());
  if (!rankGroup)
    return false;

  mlir::OpBuilder builder(allGather);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, allGather);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveAllGatherOp>(
      allGather.getLoc(), allGather->getResultTypes(), allGather.getOperands(),
      *outs, allGather.getAllGatherDimAttr(),
      getRankGroupAttr(builder, *rankGroup),
      getChannelIdAttr(builder, allGather.getChannelHandleAttr()),
      getUseGlobalDeviceIdsAttr(builder, allGather.getUseGlobalDeviceIds()));
  replaceAndErase(allGather, lowered);
  return true;
}

static bool lowerAllReduce(mlir::stablehlo::AllReduceOp allReduce) {
  auto rankGroup = getSingleReplicaGroup(allReduce.getReplicaGroups());
  if (!rankGroup)
    return false;

  mlir::OpBuilder builder(allReduce);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, allReduce);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveAllReduceOp>(
      allReduce.getLoc(), allReduce->getResultTypes(), allReduce.getOperands(),
      *outs, getRankGroupAttr(builder, *rankGroup),
      getChannelIdAttr(builder, allReduce.getChannelHandleAttr()),
      getUseGlobalDeviceIdsAttr(builder, allReduce.getUseGlobalDeviceIds()));
  if (mlir::failed(convertCombinerRegion(
          allReduce.getComputation(), lowered.getCombiner(),
          allReduce.getOperands(), lowered.getResults()))) {
    lowered.erase();
    eraseUnusedDestinationTensors(*outs);
    return false;
  }
  replaceAndErase(allReduce, lowered);
  return true;
}

static bool lowerReduceScatter(mlir::stablehlo::ReduceScatterOp reduceScatter) {
  auto rankGroup = getSingleReplicaGroup(reduceScatter.getReplicaGroups());
  if (!rankGroup)
    return false;

  mlir::OpBuilder builder(reduceScatter);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, reduceScatter);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveReduceScatterOp>(
      reduceScatter.getLoc(), reduceScatter->getResultTypes(),
      reduceScatter->getOperands(), *outs,
      reduceScatter.getScatterDimensionAttr(),
      getRankGroupAttr(builder, *rankGroup),
      getChannelIdAttr(builder, reduceScatter.getChannelHandleAttr()),
      getUseGlobalDeviceIdsAttr(builder,
                                reduceScatter.getUseGlobalDeviceIds()));
  if (mlir::failed(convertCombinerRegion(
          reduceScatter.getComputation(), lowered.getCombiner(),
          reduceScatter->getOperands(), lowered.getResults()))) {
    lowered.erase();
    eraseUnusedDestinationTensors(*outs);
    return false;
  }
  replaceAndErase(reduceScatter, lowered);
  return true;
}

static bool lowerAllToAll(mlir::stablehlo::AllToAllOp allToAll) {
  auto rankGroup = getSingleReplicaGroup(allToAll.getReplicaGroups());
  if (!rankGroup)
    return false;

  mlir::OpBuilder builder(allToAll);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, allToAll);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveAllToAllOp>(
      allToAll.getLoc(), allToAll->getResultTypes(), allToAll.getOperands(),
      *outs, allToAll.getSplitDimensionAttr(),
      allToAll.getConcatDimensionAttr(), allToAll.getSplitCountAttr(),
      getRankGroupAttr(builder, *rankGroup),
      getChannelIdAttr(builder, allToAll.getChannelHandleAttr()),
      mlir::BoolAttr());
  replaceAndErase(allToAll, lowered);
  return true;
}

static std::optional<llvm::SmallVector<int64_t>>
getSourceTargetPairs(mlir::DenseIntElementsAttr sourceTargetPairs) {
  auto pairsType =
      mlir::cast<mlir::RankedTensorType>(sourceTargetPairs.getType());
  if (pairsType.getRank() != 2 || pairsType.getDimSize(1) != 2)
    return std::nullopt;

  llvm::SmallVector<int64_t> pairs;
  for (llvm::APInt value : sourceTargetPairs.getValues<llvm::APInt>()) {
    if (!value.isSignedIntN(63))
      return std::nullopt;
    pairs.push_back(value.getSExtValue());
  }
  return pairs;
}

static bool
lowerCollectivePermute(mlir::stablehlo::CollectivePermuteOp collectivePermute) {
  std::optional<llvm::SmallVector<int64_t>> sourceTargetPairs =
      getSourceTargetPairs(collectivePermute.getSourceTargetPairs());
  if (!sourceTargetPairs)
    return false;

  mlir::OpBuilder builder(collectivePermute);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, collectivePermute);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveCollectivePermuteOp>(
      collectivePermute.getLoc(), collectivePermute->getResultTypes(),
      collectivePermute->getOperands(), *outs,
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *sourceTargetPairs),
      getChannelIdAttr(builder, collectivePermute.getChannelHandleAttr()));
  replaceAndErase(collectivePermute, lowered);
  return true;
}
#endif

struct NormalizeStablehloCollectivesPass
    : public impl::NormalizeStablehloCollectivesPassBase<
          NormalizeStablehloCollectivesPass> {
  using impl::NormalizeStablehloCollectivesPassBase<
      NormalizeStablehloCollectivesPass>::NormalizeStablehloCollectivesPassBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    impl::NormalizeStablehloCollectivesPassBase<
        NormalizeStablehloCollectivesPass>::getDependentDialects(registry);
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::Operation *> collectives;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::stablehlo::AllGatherOp, mlir::stablehlo::AllReduceOp,
                    mlir::stablehlo::ReduceScatterOp,
                    mlir::stablehlo::AllToAllOp,
                    mlir::stablehlo::CollectivePermuteOp>(op))
        collectives.push_back(op);
    });

    for (mlir::Operation *op : collectives) {
      bool lowered = false;
      if (auto allGather = mlir::dyn_cast<mlir::stablehlo::AllGatherOp>(op)) {
        lowered = lowerAllGather(allGather);
      } else if (auto allReduce =
                     mlir::dyn_cast<mlir::stablehlo::AllReduceOp>(op)) {
        lowered = lowerAllReduce(allReduce);
      } else if (auto reduceScatter =
                     mlir::dyn_cast<mlir::stablehlo::ReduceScatterOp>(op)) {
        lowered = lowerReduceScatter(reduceScatter);
      } else if (auto allToAll =
                     mlir::dyn_cast<mlir::stablehlo::AllToAllOp>(op)) {
        lowered = lowerAllToAll(allToAll);
      } else if (auto collectivePermute =
                     mlir::dyn_cast<mlir::stablehlo::CollectivePermuteOp>(op)) {
        lowered = lowerCollectivePermute(collectivePermute);
      }

      if (!lowered) {
        op->emitError("failed to normalize StableHLO collective to "
                      "wafer_linalg_ext collective handoff");
        signalPassFailure();
        return;
      }
    }
#endif
  }
};

} // namespace

} // namespace wafer
