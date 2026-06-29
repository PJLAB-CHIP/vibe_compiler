//===- NormalizeStablehloCollectives.cpp - StableHLO collective handoff ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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

static mlir::DenseI64ArrayAttr getRankGroupAttr(mlir::OpBuilder &builder,
                                                const ReplicaGroups &groups) {
  if (groups.groupCount != 1)
    return {};
  return mlir::DenseI64ArrayAttr::get(builder.getContext(), groups.flattened);
}

static mlir::DenseIntElementsAttr
getRankGroupsAttr(const ReplicaGroups &groups) {
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
      getRankGroupAttr(builder, *replicaGroups),
      getRankGroupsAttr(*replicaGroups),
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
      *outs, getRankGroupAttr(builder, *replicaGroups),
      getRankGroupsAttr(*replicaGroups),
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
      getRankGroupAttr(builder, *replicaGroups),
      getRankGroupsAttr(*replicaGroups),
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
      getRankGroupAttr(builder, *replicaGroups),
      getRankGroupsAttr(*replicaGroups),
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

static bool allStatic(llvm::ArrayRef<int64_t> values) {
  return llvm::all_of(values, [](int64_t value) {
    return !mlir::ShapedType::isDynamic(value);
  });
}

static int64_t getLinearIndex(llvm::ArrayRef<int64_t> shape,
                              llvm::ArrayRef<int64_t> indices) {
  int64_t linear = 0;
  for (auto [dim, index] : llvm::enumerate(indices))
    linear = linear * shape[dim] + index;
  return linear;
}

static void delinearizeIndex(int64_t linear, llvm::ArrayRef<int64_t> shape,
                             llvm::SmallVectorImpl<int64_t> &indices) {
  indices.assign(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    int64_t size = shape[dim];
    indices[dim] = linear % size;
    linear /= size;
  }
}

static mlir::DenseElementsAttr getDenseConstantAttr(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return {};
  return mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
}

static bool replaceWithDenseConstant(mlir::Operation *op, mlir::Value result,
                                     mlir::DenseElementsAttr attr) {
  mlir::OpBuilder builder(op);
  auto constant = builder.create<mlir::arith::ConstantOp>(
      op->getLoc(), result.getType(), attr);
  result.replaceAllUsesWith(constant.getResult());
  op->erase();
  return true;
}

static bool foldConstantTensorExtractSlice(mlir::tensor::ExtractSliceOp slice) {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(slice.getSourceType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getRank() != resultType.getRank())
    return false;
  if (!allStatic(slice.getStaticOffsets()) ||
      !allStatic(slice.getStaticSizes()) ||
      !allStatic(slice.getStaticStrides()))
    return false;

  mlir::DenseElementsAttr sourceAttr = getDenseConstantAttr(slice.getSource());
  if (!sourceAttr)
    return false;

  llvm::SmallVector<mlir::Attribute> sourceValues;
  for (mlir::Attribute value : sourceAttr.getValues<mlir::Attribute>())
    sourceValues.push_back(value);

  llvm::SmallVector<mlir::Attribute> resultValues;
  resultValues.reserve(resultType.getNumElements());
  llvm::SmallVector<int64_t> resultIndices;
  llvm::SmallVector<int64_t> sourceIndices(sourceType.getRank(), 0);
  llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  for (int64_t linear = 0; linear < resultType.getNumElements(); ++linear) {
    delinearizeIndex(linear, resultShape, resultIndices);
    for (auto [dim, index] : llvm::enumerate(resultIndices))
      sourceIndices[dim] = offsets[dim] + index * strides[dim];
    int64_t sourceLinear = getLinearIndex(sourceShape, sourceIndices);
    if (sourceLinear < 0 ||
        sourceLinear >= static_cast<int64_t>(sourceValues.size()))
      return false;
    resultValues.push_back(sourceValues[sourceLinear]);
  }

  auto resultAttr = mlir::DenseElementsAttr::get(resultType, resultValues);
  return replaceWithDenseConstant(slice.getOperation(), slice.getResult(),
                                  resultAttr);
}

static bool foldConstantTensorReshape(mlir::Operation *op, mlir::Value source,
                                      mlir::Value result) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getNumElements() != resultType.getNumElements())
    return false;

  mlir::DenseElementsAttr sourceAttr = getDenseConstantAttr(source);
  if (!sourceAttr)
    return false;

  llvm::SmallVector<mlir::Attribute> values;
  for (mlir::Attribute value : sourceAttr.getValues<mlir::Attribute>())
    values.push_back(value);
  auto resultAttr = mlir::DenseElementsAttr::get(resultType, values);
  return replaceWithDenseConstant(op, result, resultAttr);
}

static bool foldConstantTensorExtract(mlir::tensor::ExtractOp extract) {
  mlir::DenseElementsAttr sourceAttr =
      getDenseConstantAttr(extract.getTensor());
  if (!sourceAttr)
    return false;
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(extract.getTensor().getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      extract.getIndices().size() != static_cast<size_t>(sourceType.getRank()))
    return false;

  llvm::SmallVector<int64_t> indices;
  indices.reserve(extract.getIndices().size());
  for (mlir::Value index : extract.getIndices()) {
    auto constant = index.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constant)
      return false;
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (!attr)
      return false;
    int64_t constantIndex = attr.getInt();
    indices.push_back(constantIndex);
  }

  llvm::SmallVector<mlir::Attribute> sourceValues;
  for (mlir::Attribute value : sourceAttr.getValues<mlir::Attribute>())
    sourceValues.push_back(value);
  int64_t sourceLinear = getLinearIndex(sourceType.getShape(), indices);
  if (sourceLinear < 0 ||
      sourceLinear >= static_cast<int64_t>(sourceValues.size()))
    return false;
  auto value = mlir::dyn_cast<mlir::TypedAttr>(sourceValues[sourceLinear]);
  if (!value)
    return false;

  mlir::OpBuilder builder(extract);
  auto constant =
      builder.create<mlir::arith::ConstantOp>(extract.getLoc(), value);
  extract.getResult().replaceAllUsesWith(constant.getResult());
  extract.erase();
  return true;
}

static std::optional<int64_t>
getConstantLinearIndex(mlir::AffineMap map, llvm::ArrayRef<int64_t> indices,
                       llvm::ArrayRef<int64_t> shape) {
  if (map.getNumSymbols() != 0 || map.getNumResults() != shape.size())
    return std::nullopt;
  if (shape.empty())
    return int64_t{0};

  llvm::SmallVector<int64_t> operandIndices;
  operandIndices.reserve(shape.size());
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() >= indices.size())
      return std::nullopt;
    operandIndices.push_back(indices[dimExpr.getPosition()]);
  }
  return getLinearIndex(shape, operandIndices);
}

static std::optional<mlir::Attribute>
getDenseElementAt(mlir::DenseElementsAttr attr, mlir::AffineMap map,
                  llvm::ArrayRef<int64_t> resultIndices) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  std::optional<int64_t> linear =
      getConstantLinearIndex(map, resultIndices, type.getShape());
  if (!linear)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> values;
  for (mlir::Attribute value : attr.getValues<mlir::Attribute>())
    values.push_back(value);
  if (*linear < 0 || *linear >= static_cast<int64_t>(values.size()))
    return std::nullopt;
  return values[*linear];
}

static std::optional<mlir::Attribute>
evaluateConstantScalarOp(mlir::Operation *op,
                         llvm::DenseMap<mlir::Value, mlir::Attribute> &values) {
  auto lookup = [&](mlir::Value value) -> std::optional<mlir::Attribute> {
    auto it = values.find(value);
    if (it == values.end())
      return std::nullopt;
    return it->second;
  };

  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(op)) {
    std::optional<mlir::Attribute> lhs = lookup(add.getLhs());
    std::optional<mlir::Attribute> rhs = lookup(add.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    auto lhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*lhs);
    auto rhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*rhs);
    if (!lhsAttr || !rhsAttr || lhsAttr.getType() != rhsAttr.getType())
      return std::nullopt;
    return mlir::IntegerAttr::get(lhsAttr.getType(),
                                  lhsAttr.getValue() + rhsAttr.getValue());
  }

  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
    std::optional<mlir::Attribute> lhs = lookup(cmp.getLhs());
    std::optional<mlir::Attribute> rhs = lookup(cmp.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    auto lhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*lhs);
    auto rhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*rhs);
    if (!lhsAttr || !rhsAttr)
      return std::nullopt;

    llvm::APInt lhsValue = lhsAttr.getValue();
    llvm::APInt rhsValue = rhsAttr.getValue();
    bool result = false;
    switch (cmp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhsValue == rhsValue;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhsValue != rhsValue;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhsValue.slt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhsValue.sle(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhsValue.sgt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhsValue.sge(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = lhsValue.ult(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = lhsValue.ule(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = lhsValue.ugt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = lhsValue.uge(rhsValue);
      break;
    }
    return mlir::IntegerAttr::get(op->getResult(0).getType(), result ? 1 : 0);
  }

  return std::nullopt;
}

static bool foldConstantLinalgGeneric(mlir::linalg::GenericOp generic) {
  if (generic->getNumResults() != 1 || generic.getNumDpsInits() != 1)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](auto iteratorType) {
        return iteratorType == mlir::utils::IteratorType::parallel;
      }))
    return false;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;

  llvm::SmallVector<mlir::AffineMap> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.size() !=
      generic.getNumDpsInputs() + generic.getNumDpsInits())
    return false;

  llvm::SmallVector<std::optional<mlir::DenseElementsAttr>> operandAttrs;
  operandAttrs.reserve(generic.getNumDpsInputs() + generic.getNumDpsInits());
  for (mlir::Value input : generic.getDpsInputs()) {
    mlir::DenseElementsAttr attr = getDenseConstantAttr(input);
    if (!attr)
      return false;
    operandAttrs.push_back(attr);
  }
  for (auto [index, init] : llvm::enumerate(generic.getDpsInits())) {
    mlir::DenseElementsAttr attr = getDenseConstantAttr(init);
    if (!attr && !generic.getBody()
                      ->getArgument(generic.getNumDpsInputs() + index)
                      .use_empty())
      return false;
    if (attr)
      operandAttrs.push_back(attr);
    else
      operandAttrs.push_back(std::nullopt);
  }

  auto yield =
      mlir::dyn_cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  llvm::SmallVector<mlir::Attribute> resultValues;
  resultValues.reserve(resultType.getNumElements());
  llvm::SmallVector<int64_t> resultIndices;
  for (int64_t linear = 0; linear < resultType.getNumElements(); ++linear) {
    delinearizeIndex(linear, resultType.getShape(), resultIndices);

    llvm::DenseMap<mlir::Value, mlir::Attribute> scalarValues;
    for (auto [index, attr] : llvm::enumerate(operandAttrs)) {
      if (!attr)
        continue;
      std::optional<mlir::Attribute> value =
          getDenseElementAt(*attr, indexingMaps[index], resultIndices);
      if (!value)
        return false;
      scalarValues[generic.getBody()->getArgument(index)] = *value;
    }

    for (mlir::Operation &op : generic.getBody()->without_terminator()) {
      if (op.getNumResults() != 1)
        return false;
      std::optional<mlir::Attribute> value =
          evaluateConstantScalarOp(&op, scalarValues);
      if (!value)
        return false;
      scalarValues[op.getResult(0)] = *value;
    }

    auto it = scalarValues.find(yield.getValues().front());
    if (it == scalarValues.end())
      return false;
    resultValues.push_back(it->second);
  }

  auto resultAttr = mlir::DenseElementsAttr::get(resultType, resultValues);
  return replaceWithDenseConstant(generic.getOperation(), generic.getResult(0),
                                  resultAttr);
}

static bool foldConstantTensorOp(mlir::Operation *op) {
  if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
    return foldConstantTensorExtractSlice(slice);
  if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op))
    return foldConstantTensorReshape(op, collapse.getSrc(),
                                     collapse.getResult());
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op))
    return foldConstantTensorReshape(op, expand.getSrc(), expand.getResult());
  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op))
    return foldConstantTensorExtract(extract);
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
    return foldConstantLinalgGeneric(generic);
  return false;
}

static void foldConstantTensorOps(mlir::Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> ops;
    root->walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::CollapseShapeOp,
                    mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractOp,
                    mlir::linalg::GenericOp>(op))
        ops.push_back(op);
    });
    for (mlir::Operation *op : ops)
      changed |= foldConstantTensorOp(op);
  }
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
    llvm::SmallVector<mlir::Operation *> opsToNormalize;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<
              mlir::stablehlo::AllGatherOp, mlir::stablehlo::AllReduceOp,
              mlir::stablehlo::ReduceScatterOp, mlir::stablehlo::AllToAllOp,
              mlir::stablehlo::CollectivePermuteOp,
              mlir::stablehlo::PartitionIdOp, mlir::stablehlo::ReplicaIdOp>(
              op) ||
          mlir::isa<mlir::UnrealizedConversionCastOp>(op))
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

      if (!lowered) {
        op->emitError("failed to normalize residual StableHLO op before "
                      "Wafer group lowering");
        signalPassFailure();
        return;
      }
    }

    foldConstantTensorOps(getOperation());
#endif
  }
};

} // namespace

} // namespace wafer
