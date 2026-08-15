//===- NormalizeStablehloCollectives.cpp - StableHLO collective rewrite ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
#define GEN_PASS_DEF_NORMALIZESTABLEHLOCOLLECTIVESPASS
#define GEN_PASS_DEF_FOLDDEFAULTSTABLEHLOEXECUTIONIDSPASS
#define GEN_PASS_DEF_FOLDCONSTANTINTEGERTENSORCASTSPASS
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

struct ReplicaGroups {
  mlir::DenseIntElementsAttr attr;
  llvm::SmallVector<int64_t> flattened;
  int64_t groupCount = 0;
  int64_t groupSize = 0;
};

static std::optional<ReplicaGroups>
getReplicaGroups(mlir::DenseIntElementsAttr replicaGroups) {
  auto groupsType = mlir::cast<mlir::RankedTensorType>(replicaGroups.getType());
  if (groupsType.getRank() != 2 || groupsType.getDimSize(0) <= 0 ||
      groupsType.getDimSize(1) <= 0)
    return std::nullopt;

  ReplicaGroups groups;
  groups.attr = replicaGroups;
  groups.groupCount = groupsType.getDimSize(0);
  groups.groupSize = groupsType.getDimSize(1);
  for (llvm::APInt value : replicaGroups.getValues<llvm::APInt>()) {
    if (!value.isSignedIntN(63))
      return std::nullopt;
    groups.flattened.push_back(value.getSExtValue());
  }
  return groups;
}

static mlir::DenseI64ArrayAttr
getPartitionGroupAttr(mlir::OpBuilder &builder, const ReplicaGroups &groups) {
  if (groups.groupCount != 1)
    return {};
  return mlir::DenseI64ArrayAttr::get(builder.getContext(), groups.flattened);
}

static mlir::DenseIntElementsAttr
getPartitionGroupsAttr(const ReplicaGroups &groups) {
  if (groups.groupCount == 1)
    return {};
  return groups.attr;
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
  if (inputs.size() != results.size() ||
      sourceBlock.getNumArguments() != inputs.size() * 2)
    return mlir::failure();

  llvm::SmallVector<mlir::Type> resultElementTypes;
  resultElementTypes.reserve(results.size());
  for (auto [index, result] : llvm::enumerate(results)) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!resultType)
      return mlir::failure();
    mlir::Type elementType = resultType.getElementType();
    if (mlir::getElementTypeOrSelf(sourceBlock.getArgument(index).getType()) !=
            elementType ||
        mlir::getElementTypeOrSelf(
            sourceBlock.getArgument(index + inputs.size()).getType()) !=
            elementType)
      return mlir::failure();
    resultElementTypes.push_back(elementType);
  }

  auto sourceReturn =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(sourceBlock.getTerminator());
  if (!sourceReturn || sourceReturn.getResults().size() != results.size())
    return mlir::failure();

  destRegion.push_back(new mlir::Block);
  mlir::Block &destBlock = destRegion.front();
  mlir::Location argLoc = sourceRegion.getParentOp()->getLoc();
  for (unsigned copy = 0; copy < 2; ++copy)
    for (mlir::Type elementType : resultElementTypes)
      destBlock.addArgument(elementType, argLoc);

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
  nestedBuilder.create<LinalgExtCollectiveYieldOp>(sourceReturn.getLoc(),
                                                   yielded);
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
  std::optional<ReplicaGroups> replicaGroups =
      getReplicaGroups(allGather.getReplicaGroups());
  if (!replicaGroups)
    return false;

  mlir::OpBuilder builder(allGather);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, allGather);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveAllGatherOp>(
      allGather.getLoc(), allGather->getResultTypes(), allGather.getOperands(),
      *outs, allGather.getAllGatherDimAttr(),
      getPartitionGroupAttr(builder, *replicaGroups),
      getPartitionGroupsAttr(*replicaGroups),
      getChannelIdAttr(builder, allGather.getChannelHandleAttr()),
      getUseGlobalDeviceIdsAttr(builder, allGather.getUseGlobalDeviceIds()));
  replaceAndErase(allGather, lowered);
  return true;
}

static bool lowerAllReduce(mlir::stablehlo::AllReduceOp allReduce) {
  std::optional<ReplicaGroups> replicaGroups =
      getReplicaGroups(allReduce.getReplicaGroups());
  if (!replicaGroups)
    return false;

  mlir::OpBuilder builder(allReduce);
  std::optional<llvm::SmallVector<mlir::Value>> outs =
      createDestinationTensors(builder, allReduce);
  if (!outs)
    return false;

  auto lowered = builder.create<LinalgExtCollectiveAllReduceOp>(
      allReduce.getLoc(), allReduce->getResultTypes(), allReduce.getOperands(),
      *outs, getPartitionGroupAttr(builder, *replicaGroups),
      getPartitionGroupsAttr(*replicaGroups),
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
  std::optional<ReplicaGroups> replicaGroups =
      getReplicaGroups(reduceScatter.getReplicaGroups());
  if (!replicaGroups)
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
      getPartitionGroupAttr(builder, *replicaGroups),
      getPartitionGroupsAttr(*replicaGroups),
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
  std::optional<ReplicaGroups> replicaGroups =
      getReplicaGroups(allToAll.getReplicaGroups());
  if (!replicaGroups)
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
      getPartitionGroupAttr(builder, *replicaGroups),
      getPartitionGroupsAttr(*replicaGroups),
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

static bool lowerReplicaOrPartitionId(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() != 0)
    return false;
  auto intType = mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
  if (!intType)
    return false;

  mlir::OpBuilder builder(op);
  auto value = mlir::DenseElementsAttr::get(resultType,
                                            llvm::APInt(intType.getWidth(), 0));
  auto constant =
      builder.create<mlir::arith::ConstantOp>(op->getLoc(), resultType, value);
  op->getResult(0).replaceAllUsesWith(constant.getResult());
  op->erase();
  return true;
}

static bool
lowerConstantIntegerTensorCast(mlir::UnrealizedConversionCastOp cast) {
  if (cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return false;

  auto constant =
      cast.getInputs().front().getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto sourceAttr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(constant.getValue());
  if (!sourceAttr)
    return false;

  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(cast.getResult(0).getType());
  if (!sourceType || !resultType ||
      sourceType.getShape() != resultType.getShape())
    return false;

  auto sourceIntType =
      mlir::dyn_cast<mlir::IntegerType>(sourceType.getElementType());
  auto resultIntType =
      mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
  if (!sourceIntType || !resultIntType ||
      sourceIntType.getWidth() != resultIntType.getWidth())
    return false;

  llvm::SmallVector<llvm::APInt> values;
  values.reserve(sourceAttr.getNumElements());
  for (llvm::APInt value : sourceAttr.getValues<llvm::APInt>())
    values.push_back(value);

  mlir::OpBuilder builder(cast);
  auto resultAttr = mlir::DenseElementsAttr::get(resultType, values);
  auto replacement = builder.create<mlir::arith::ConstantOp>(
      cast.getLoc(), resultType, resultAttr);
  cast.getResult(0).replaceAllUsesWith(replacement.getResult());
  cast.erase();
  return true;
}

enum class StablehloNormalizationAction {
  Collective,
  DefaultExecutionId,
  ConstantIntegerTensorCast,
};

static bool isSelectedOperation(mlir::Operation *op,
                                StablehloNormalizationAction action) {
  switch (action) {
  case StablehloNormalizationAction::Collective:
    return mlir::isa<mlir::stablehlo::AllGatherOp, mlir::stablehlo::AllReduceOp,
                     mlir::stablehlo::ReduceScatterOp,
                     mlir::stablehlo::AllToAllOp,
                     mlir::stablehlo::CollectivePermuteOp>(op);
  case StablehloNormalizationAction::DefaultExecutionId:
    return mlir::isa<mlir::stablehlo::PartitionIdOp,
                     mlir::stablehlo::ReplicaIdOp>(op);
  case StablehloNormalizationAction::ConstantIntegerTensorCast:
    return mlir::isa<mlir::UnrealizedConversionCastOp>(op);
  }
  llvm_unreachable("unknown StableHLO normalization action");
}

static bool
requiresStablehloNormalizationTransaction(mlir::ModuleOp module,
                                          StablehloNormalizationAction action) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (isSelectedOperation(op, action)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

static mlir::LogicalResult
normalizeStablehloModuleInPlace(mlir::ModuleOp module,
                                StablehloNormalizationAction action) {
  llvm::SmallVector<mlir::Operation *> opsToNormalize;
  module.walk([&](mlir::Operation *op) {
    if (isSelectedOperation(op, action))
      opsToNormalize.push_back(op);
  });

  for (mlir::Operation *op : opsToNormalize) {
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
    } else if (mlir::isa<mlir::stablehlo::PartitionIdOp,
                         mlir::stablehlo::ReplicaIdOp>(op)) {
      lowered = lowerReplicaOrPartitionId(op);
    } else if (auto cast =
                   mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      lowered = lowerConstantIntegerTensorCast(cast);
    }

    if (!lowered)
      return op->emitError(
          "failed to normalize residual StableHLO op before Wafer "
          "structured tensor-program scheduling");
  }

  return mlir::success();
}

static mlir::LogicalResult
runAtomicStablehloNormalization(mlir::ModuleOp module,
                                StablehloNormalizationAction action) {
  if (!requiresStablehloNormalizationTransaction(module, action))
    return mlir::success();
  mlir::OwningOpRef<mlir::ModuleOp> transaction =
      mlir::cast<mlir::ModuleOp>(module->clone());
  if (mlir::failed(normalizeStablehloModuleInPlace(*transaction, action)) ||
      mlir::failed(mlir::verify(*transaction)))
    return mlir::failure();
  module.getBodyRegion().takeBody(transaction->getBodyRegion());
  return mlir::success();
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
    if (mlir::failed(runAtomicStablehloNormalization(
            getOperation(), StablehloNormalizationAction::Collective)))
      signalPassFailure();
#endif
  }
};

struct FoldDefaultStablehloExecutionIdsPass
    : public impl::FoldDefaultStablehloExecutionIdsPassBase<
          FoldDefaultStablehloExecutionIdsPass> {
  using impl::FoldDefaultStablehloExecutionIdsPassBase<
      FoldDefaultStablehloExecutionIdsPass>::
      FoldDefaultStablehloExecutionIdsPassBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    impl::FoldDefaultStablehloExecutionIdsPassBase<
        FoldDefaultStablehloExecutionIdsPass>::getDependentDialects(registry);
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    if (mlir::failed(runAtomicStablehloNormalization(
            getOperation(), StablehloNormalizationAction::DefaultExecutionId)))
      signalPassFailure();
#endif
  }
};

struct FoldConstantIntegerTensorCastsPass
    : public impl::FoldConstantIntegerTensorCastsPassBase<
          FoldConstantIntegerTensorCastsPass> {
  using impl::FoldConstantIntegerTensorCastsPassBase<
      FoldConstantIntegerTensorCastsPass>::
      FoldConstantIntegerTensorCastsPassBase;

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    if (mlir::failed(runAtomicStablehloNormalization(
            getOperation(),
            StablehloNormalizationAction::ConstantIntegerTensorCast)))
      signalPassFailure();
#endif
  }
};

} // namespace

} // namespace wafer
