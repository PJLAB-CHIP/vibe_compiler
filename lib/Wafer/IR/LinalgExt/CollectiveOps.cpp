//===- LinalgExtCollectiveOps.cpp - Wafer linalg-ext collective verifier
//--------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace wafer;
using namespace wafer::detail;

namespace {

constexpr llvm::StringRef kOperandSegmentSizesAttr = "operandSegmentSizes";

bool isAllowedAttr(mlir::NamedAttribute attr,
                   llvm::ArrayRef<llvm::StringRef> allowedAttrs) {
  llvm::StringRef name = attr.getName().getValue();
  return name == kOperandSegmentSizesAttr ||
         llvm::is_contained(allowedAttrs, name);
}

mlir::LogicalResult
verifyAllowedAttrs(mlir::Operation *op,
                   llvm::ArrayRef<llvm::StringRef> allowedAttrs) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedAttr(attr, allowedAttrs) &&
        !wafer::detail::isExternalDiscardableAttribute(op, attr))
      return op->emitOpError("does not accept attribute '")
             << attr.getName().getValue()
             << "'; linalg-ext collectives must not carry physical endpoint "
                "mapping, "
                "SPM, DTE, byte schedule, or runtime metadata";
  }
  return mlir::success();
}

mlir::LogicalResult
verifyPartitionGroup(mlir::Operation *op,
                     mlir::DenseI64ArrayAttr partitionGroupAttr) {
  auto partitionGroup = partitionGroupAttr.asArrayRef();
  if (partitionGroup.empty())
    return op->emitOpError("partition_group must not be empty");

  llvm::SmallSet<int64_t, 8> seen;
  for (int64_t partitionId : partitionGroup) {
    if (partitionId < 0)
      return op->emitOpError("partition_group entries must be non-negative");
    if (!seen.insert(partitionId).second)
      return op->emitOpError("partition_group entries must be unique");
  }

  return mlir::success();
}

mlir::LogicalResult
verifyPartitionGroups(mlir::Operation *op,
                      mlir::DenseIntElementsAttr partitionGroupsAttr) {
  auto groupsType =
      mlir::dyn_cast<mlir::RankedTensorType>(partitionGroupsAttr.getType());
  if (!groupsType || groupsType.getRank() != 2)
    return op->emitOpError(
        "partition_groups must be a rank-2 i64 dense elements attr");
  if (groupsType.getDimSize(0) <= 0 || groupsType.getDimSize(1) <= 0)
    return op->emitOpError("partition_groups dimensions must be positive");

  llvm::SmallSet<int64_t, 16> seen;
  llvm::SmallVector<int64_t, 16> partitionIds;
  for (llvm::APInt value : partitionGroupsAttr.getValues<llvm::APInt>()) {
    if (!value.isSignedIntN(63))
      return op->emitOpError("partition_groups entries must fit in int64");
    int64_t partitionId = value.getSExtValue();
    if (partitionId < 0)
      return op->emitOpError("partition_groups entries must be non-negative");
    if (!seen.insert(partitionId).second)
      return op->emitOpError("partition_groups entries must be unique");
    partitionIds.push_back(partitionId);
  }

  return mlir::success();
}

mlir::LogicalResult verifyCollectivePartitionGroups(
    mlir::Operation *op, mlir::DenseI64ArrayAttr partitionGroupAttr,
    mlir::DenseIntElementsAttr partitionGroupsAttr) {
  if (partitionGroupAttr && partitionGroupsAttr)
    return op->emitOpError(
        "must specify only one of partition_group or partition_groups");
  if (!partitionGroupAttr && !partitionGroupsAttr)
    return op->emitOpError("requires partition_group or partition_groups");
  if (partitionGroupAttr)
    return verifyPartitionGroup(op, partitionGroupAttr);
  return verifyPartitionGroups(op, partitionGroupsAttr);
}

int64_t getCollectivePartitionGroupSize(
    mlir::DenseI64ArrayAttr partitionGroupAttr,
    mlir::DenseIntElementsAttr partitionGroupsAttr) {
  if (partitionGroupAttr)
    return static_cast<int64_t>(partitionGroupAttr.asArrayRef().size());
  auto groupsType =
      mlir::cast<mlir::RankedTensorType>(partitionGroupsAttr.getType());
  return groupsType.getDimSize(1);
}

mlir::LogicalResult verifyChannelAttrs(mlir::Operation *op,
                                       mlir::IntegerAttr channelIdAttr,
                                       mlir::BoolAttr useGlobalDeviceIdsAttr) {
  if (!channelIdAttr)
    return mlir::success();

  int64_t channelId = channelIdAttr.getInt();
  if (channelId < 0)
    return op->emitOpError("channel_id must be non-negative");
  if (useGlobalDeviceIdsAttr && useGlobalDeviceIdsAttr.getValue() &&
      channelId <= 0)
    return op->emitOpError(
        "use_global_device_ids requires a positive channel_id");
  return mlir::success();
}

bool isSupportedCollectiveInputPromotion(mlir::Type inputElementType,
                                         mlir::Type resultElementType) {
  if (inputElementType == resultElementType)
    return true;

  auto inputFloat = mlir::dyn_cast<mlir::FloatType>(inputElementType);
  auto resultFloat = mlir::dyn_cast<mlir::FloatType>(resultElementType);
  if (!inputFloat || !resultFloat)
    return false;
  auto isSupportedStorageType = [](mlir::Type type) {
    return mlir::isa<mlir::Float16Type, mlir::BFloat16Type, mlir::Float32Type>(
        type);
  };
  return isSupportedStorageType(inputElementType) &&
         isSupportedStorageType(resultElementType) &&
         inputFloat.getWidth() <= resultFloat.getWidth();
}

mlir::LogicalResult verifySingleDestinationStyleShape(
    mlir::Operation *op, mlir::OperandRange inputs, mlir::OperandRange outs,
    mlir::ResultRange results, bool allowPromotedInputs = false) {
  if (inputs.empty())
    return op->emitOpError(
        "linalg-ext collective must have at least one input");
  if (inputs.size() != outs.size() || inputs.size() != results.size())
    return op->emitOpError("linalg-ext collective must have matching input, "
                           "out, and result counts");

  for (auto [index, values] :
       llvm::enumerate(llvm::zip(inputs, outs, results))) {
    mlir::Value input = std::get<0>(values);
    mlir::Value out = std::get<1>(values);
    mlir::OpResult result = std::get<2>(values);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(out.getType());
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!inputType || !outType || !resultType)
      return op->emitOpError(
          "linalg-ext collective operands and results must be ranked tensors");
    if (outType != resultType)
      return op->emitOpError("out type must match tied result type at index ")
             << index;
    if (inputType.getElementType() != outType.getElementType()) {
      if (!allowPromotedInputs ||
          !isSupportedCollectiveInputPromotion(inputType.getElementType(),
                                               outType.getElementType()))
        return op->emitOpError(
                   "input element type must match the out element type or use "
                   "a supported floating promotion at index ")
               << index;
    }
  }

  return mlir::success();
}

mlir::FailureOr<int64_t> verifyAxis(mlir::Operation *op,
                                    mlir::IntegerAttr axisAttr,
                                    mlir::RankedTensorType tensorType) {
  int64_t axis = axisAttr.getInt();
  if (axis < 0 || axis >= tensorType.getRank())
    return op->emitOpError("collective axis must be within tensor rank");
  return axis;
}

bool isStaticDim(int64_t dim) { return dim != mlir::ShapedType::kDynamic; }

mlir::LogicalResult
verifySameRankAndNonAxisDims(mlir::Operation *op,
                             mlir::RankedTensorType inputType,
                             mlir::RankedTensorType outputType, int64_t axis) {
  if (inputType.getRank() != outputType.getRank())
    return op->emitOpError("collective input and result ranks must match");

  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (dim == axis)
      continue;
    if (hasStaticMismatch(inputType.getDimSize(dim),
                          outputType.getDimSize(dim)))
      return op->emitOpError(
          "collective non-axis dimensions must match input shape");
  }
  return mlir::success();
}

mlir::LogicalResult verifyAllGatherLikeShape(mlir::Operation *op,
                                             mlir::OperandRange inputs,
                                             mlir::ResultRange results,
                                             mlir::IntegerAttr axisAttr,
                                             int64_t groupSize) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    auto axisOr = verifyAxis(op, axisAttr, inputType);
    if (mlir::failed(axisOr))
      return mlir::failure();
    int64_t axis = *axisOr;
    if (mlir::failed(
            verifySameRankAndNonAxisDims(op, inputType, resultType, axis)))
      return mlir::failure();

    int64_t inputDim = inputType.getDimSize(axis);
    int64_t resultDim = resultType.getDimSize(axis);
    if (isStaticDim(inputDim) && isStaticDim(resultDim)) {
      int64_t expected = 0;
      if (!checkedMul(inputDim, groupSize, expected))
        return op->emitOpError(
            "all_gather result dimension is not representable");
      if (resultDim != expected)
        return op->emitOpError(
            "all_gather result dimension along axis must equal input "
            "dimension times partition_group size");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyReduceScatterLikeShape(mlir::Operation *op,
                                                 mlir::OperandRange inputs,
                                                 mlir::ResultRange results,
                                                 mlir::IntegerAttr axisAttr,
                                                 int64_t groupSize) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    auto axisOr = verifyAxis(op, axisAttr, inputType);
    if (mlir::failed(axisOr))
      return mlir::failure();
    int64_t axis = *axisOr;
    if (mlir::failed(
            verifySameRankAndNonAxisDims(op, inputType, resultType, axis)))
      return mlir::failure();

    int64_t inputDim = inputType.getDimSize(axis);
    int64_t resultDim = resultType.getDimSize(axis);
    if (isStaticDim(inputDim) && isStaticDim(resultDim)) {
      int64_t expectedInput = 0;
      if (!checkedMul(resultDim, groupSize, expectedInput))
        return op->emitOpError(
            "reduce_scatter input dimension is not representable");
      if (inputDim != expectedInput)
        return op->emitOpError(
            "reduce_scatter input dimension along axis must equal result "
            "dimension times partition_group size");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyAllReduceLikeShape(mlir::Operation *op,
                                             mlir::OperandRange inputs,
                                             mlir::ResultRange results) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    if (inputType.getShape() != resultType.getShape())
      return op->emitOpError("all_reduce input and result shapes must match");
  }
  return mlir::success();
}

mlir::LogicalResult
verifyAllToAllShape(mlir::Operation *op, mlir::OperandRange inputs,
                    mlir::ResultRange results, mlir::IntegerAttr splitAxisAttr,
                    mlir::IntegerAttr concatAxisAttr, int64_t groupSize) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    auto splitAxisOr = verifyAxis(op, splitAxisAttr, inputType);
    if (mlir::failed(splitAxisOr))
      return mlir::failure();
    auto concatAxisOr = verifyAxis(op, concatAxisAttr, inputType);
    if (mlir::failed(concatAxisOr))
      return mlir::failure();
    int64_t splitAxis = *splitAxisOr;
    int64_t concatAxis = *concatAxisOr;
    if (inputType.getRank() != resultType.getRank())
      return op->emitOpError("all_to_all input and result ranks must match");

    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      if (dim == splitAxis || dim == concatAxis)
        continue;
      if (hasStaticMismatch(inputType.getDimSize(dim),
                            resultType.getDimSize(dim)))
        return op->emitOpError(
            "all_to_all non-collective dimensions must match input shape");
    }

    int64_t inputSplit = inputType.getDimSize(splitAxis);
    int64_t resultSplit = resultType.getDimSize(splitAxis);
    if (isStaticDim(inputSplit) && isStaticDim(resultSplit)) {
      int64_t expectedInputSplit = 0;
      if (!checkedMul(resultSplit, groupSize, expectedInputSplit))
        return op->emitOpError(
            "all_to_all split dimension is not representable");
      if (inputSplit != expectedInputSplit)
        return op->emitOpError(
            "all_to_all input split dimension must equal result split "
            "dimension times partition_group size");
    }

    int64_t inputConcat = inputType.getDimSize(concatAxis);
    int64_t resultConcat = resultType.getDimSize(concatAxis);
    if (isStaticDim(inputConcat) && isStaticDim(resultConcat)) {
      int64_t expectedResultConcat = 0;
      if (!checkedMul(inputConcat, groupSize, expectedResultConcat))
        return op->emitOpError(
            "all_to_all concat dimension is not representable");
      if (resultConcat != expectedResultConcat)
        return op->emitOpError(
            "all_to_all result concat dimension must equal input concat "
            "dimension times partition_group size");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyCollectivePermuteShape(mlir::Operation *op,
                                                 mlir::OperandRange inputs,
                                                 mlir::ResultRange results) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    if (input.getType() != result.getType())
      return op->emitOpError(
          "collective_permute input and result types must match");
  }
  return mlir::success();
}

mlir::LogicalResult
verifySourceTargetPairs(mlir::Operation *op,
                        mlir::DenseI64ArrayAttr sourceTargetPairsAttr) {
  auto pairs = sourceTargetPairsAttr.asArrayRef();
  if (pairs.empty() || pairs.size() % 2 != 0)
    return op->emitOpError(
        "source_target_pairs must contain source/target pairs");

  llvm::SmallSet<int64_t, 8> seenSources;
  llvm::SmallSet<int64_t, 8> seenTargets;
  llvm::SmallVector<int64_t, 8> partitionIds;
  for (size_t index = 0; index < pairs.size(); index += 2) {
    int64_t source = pairs[index];
    int64_t target = pairs[index + 1];
    if (source < 0 || target < 0)
      return op->emitOpError(
          "source_target_pairs entries must be non-negative");
    if (!seenSources.insert(source).second)
      return op->emitOpError("source partition IDs must be unique");
    if (!seenTargets.insert(target).second)
      return op->emitOpError("target partition IDs must be unique");
    partitionIds.push_back(source);
    partitionIds.push_back(target);
  }
  return mlir::success();
}

mlir::LogicalResult verifyPartitionGroupsAgainstExecutionMesh(
    mlir::Operation *op, mlir::DenseI64ArrayAttr partitionGroup,
    mlir::DenseIntElementsAttr partitionGroups) {
  llvm::SmallVector<int64_t, 16> partitionIds;
  if (partitionGroup) {
    partitionIds.append(partitionGroup.asArrayRef().begin(),
                        partitionGroup.asArrayRef().end());
  } else if (partitionGroups) {
    for (llvm::APInt value : partitionGroups.getValues<llvm::APInt>())
      partitionIds.push_back(value.getSExtValue());
  }
  return verifyPartitionIdsWithinExecutionMesh(
      op, partitionIds, "linalg-ext collective partition group");
}

mlir::OpFoldResult getTensorDim(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value tensor, int64_t dim) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(tensor.getType());
  int64_t staticDim = tensorType.getDimSize(dim);
  if (!mlir::ShapedType::isDynamic(staticDim))
    return builder.getIndexAttr(staticDim);
  return builder.create<mlir::tensor::DimOp>(loc, tensor, dim).getResult();
}

llvm::SmallVector<mlir::Range>
getLinalgExtCollectiveIterationDomain(mlir::Operation *op,
                                      mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Range> domain;
  if (op->getNumResults() == 0)
    return domain;

  mlir::Value result = op->getResult(0);
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!resultType)
    return domain;

  mlir::Location loc = op->getLoc();
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    domain.push_back({builder.getIndexAttr(0),
                      getTensorDim(builder, loc, result, dim),
                      builder.getIndexAttr(1)});
  }
  return domain;
}

llvm::SmallVector<mlir::utils::IteratorType>
getLinalgExtCollectiveLoopIteratorTypes(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return {};
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return {};
  return llvm::SmallVector<mlir::utils::IteratorType>(
      resultType.getRank(), mlir::utils::IteratorType::parallel);
}

bool containsAxis(llvm::ArrayRef<int64_t> axes, int64_t axis) {
  return llvm::is_contained(axes, axis);
}

void appendUniqueAxis(llvm::SmallVectorImpl<int64_t> &axes, int64_t axis) {
  if (!containsAxis(axes, axis))
    axes.push_back(axis);
}

bool isStaticFullDimTile(mlir::RankedTensorType tensorType, int64_t dim,
                         mlir::OpFoldResult offset, mlir::OpFoldResult size) {
  std::optional<int64_t> staticOffset = mlir::getConstantIntValue(offset);
  std::optional<int64_t> staticSize = mlir::getConstantIntValue(size);
  int64_t dimSize = tensorType.getDimSize(dim);
  return staticOffset && *staticOffset == 0 && staticSize &&
         !mlir::ShapedType::isDynamic(dimSize) && *staticSize == dimSize;
}

mlir::LogicalResult requireFullStaticResultAxes(
    mlir::Operation *op, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, llvm::ArrayRef<int64_t> axes) {
  if (op->getNumResults() == 0)
    return mlir::failure();
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType ||
      offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank()))
    return mlir::failure();

  for (int64_t axis : axes) {
    if (axis < 0 || axis >= resultType.getRank())
      return mlir::failure();
    if (!isStaticFullDimTile(resultType, axis, offsets[axis], sizes[axis]))
      return mlir::failure();
  }
  return mlir::success();
}

void mapTileWithFullAxes(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, llvm::ArrayRef<int64_t> fullAxes,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedSizes) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  mappedOffsets.clear();
  mappedSizes.clear();
  for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
    if (containsAxis(fullAxes, dim)) {
      mappedOffsets.push_back(builder.getIndexAttr(0));
      mappedSizes.push_back(getTensorDim(builder, loc, value, dim));
      continue;
    }
    mappedOffsets.push_back(offsets[dim]);
    mappedSizes.push_back(sizes[dim]);
  }
}

using TensorTileMappingFn = llvm::function_ref<mlir::LogicalResult(
    mlir::Value value, unsigned valueIndex, bool isOut,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedSizes)>;

mlir::FailureOr<mlir::TilingResult>
buildTiledCollectiveClone(mlir::Operation *op, mlir::OpBuilder &builder,
                          mlir::OperandRange inputs, mlir::OperandRange outs,
                          TensorTileMappingFn mapTile) {
  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Value> tiledOperands;
  llvm::SmallVector<mlir::Type> tiledResultTypes;
  llvm::SmallVector<mlir::Operation *> generatedSlices;

  auto addSlicedOperand = [&](mlir::Value value, unsigned index,
                              bool isOut) -> mlir::LogicalResult {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!tensorType)
      return mlir::failure();

    llvm::SmallVector<mlir::OpFoldResult> mappedOffsets;
    llvm::SmallVector<mlir::OpFoldResult> mappedSizes;
    if (mlir::failed(mapTile(value, index, isOut, mappedOffsets, mappedSizes)))
      return mlir::failure();
    if (mappedOffsets.size() != static_cast<size_t>(tensorType.getRank()) ||
        mappedSizes.size() != static_cast<size_t>(tensorType.getRank()))
      return mlir::failure();

    llvm::SmallVector<mlir::OpFoldResult> strides(tensorType.getRank(),
                                                  builder.getIndexAttr(1));
    auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
        loc, value, mappedOffsets, mappedSizes, strides);
    tiledOperands.push_back(slice.getResult());
    generatedSlices.push_back(slice.getOperation());
    if (isOut)
      tiledResultTypes.push_back(slice.getResult().getType());
    return mlir::success();
  };

  for (auto [index, input] : llvm::enumerate(inputs)) {
    if (mlir::failed(
            addSlicedOperand(input, static_cast<unsigned>(index), false)))
      return mlir::failure();
  }
  for (auto [index, out] : llvm::enumerate(outs)) {
    if (mlir::failed(addSlicedOperand(out, static_cast<unsigned>(index), true)))
      return mlir::failure();
  }

  mlir::OperationState state(loc, op->getName().getStringRef());
  state.addOperands(tiledOperands);
  state.addTypes(tiledResultTypes);
  state.addAttributes(op->getAttrs());
  mlir::IRMapping mapper;
  for (mlir::Region &region : op->getRegions()) {
    mlir::Region *clonedRegion = state.addRegion();
    region.cloneInto(clonedRegion, mapper);
  }

  mlir::Operation *tiledOp = builder.create(state);
  return mlir::TilingResult{
      {tiledOp},
      llvm::SmallVector<mlir::Value>(tiledOp->getResults()),
      generatedSlices};
}

mlir::FailureOr<mlir::TilingResult>
buildIdentityTiledCollective(mlir::Operation *op, mlir::OpBuilder &builder,
                             mlir::OperandRange inputs, mlir::OperandRange outs,
                             llvm::ArrayRef<mlir::OpFoldResult> offsets,
                             llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  return buildTiledCollectiveClone(
      op, builder, inputs, outs,
      [&](mlir::Value value, unsigned, bool,
          llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedOffsets,
          llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedSizes)
          -> mlir::LogicalResult {
        auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
        if (offsets.size() != static_cast<size_t>(tensorType.getRank()) ||
            sizes.size() != static_cast<size_t>(tensorType.getRank()))
          return mlir::failure();
        mappedOffsets.assign(offsets.begin(), offsets.end());
        mappedSizes.assign(sizes.begin(), sizes.end());
        return mlir::success();
      });
}

mlir::FailureOr<mlir::TilingResult> buildAxisConservativeTiledCollective(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::OperandRange inputs,
    mlir::OperandRange outs, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::ArrayRef<int64_t> fullAxes) {
  if (mlir::failed(requireFullStaticResultAxes(op, offsets, sizes, fullAxes)))
    return mlir::failure();

  return buildTiledCollectiveClone(
      op, builder, inputs, outs,
      [&](mlir::Value value, unsigned, bool isOut,
          llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedOffsets,
          llvm::SmallVectorImpl<mlir::OpFoldResult> &mappedSizes)
          -> mlir::LogicalResult {
        if (isOut) {
          mappedOffsets.assign(offsets.begin(), offsets.end());
          mappedSizes.assign(sizes.begin(), sizes.end());
          return mlir::success();
        }
        mapTileWithFullAxes(builder, op->getLoc(), value, offsets, sizes,
                            fullAxes, mappedOffsets, mappedSizes);
        return mlir::success();
      });
}

mlir::LogicalResult getIdentityResultTilePosition(
    mlir::Operation *op, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  if (resultNumber >= op->getNumResults())
    return mlir::failure();
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      op->getResult(resultNumber).getType());
  if (!resultType ||
      offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank()))
    return mlir::failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return mlir::success();
}

mlir::LogicalResult getAxisConservativeResultTilePosition(
    mlir::Operation *op, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, llvm::ArrayRef<int64_t> fullAxes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  if (mlir::failed(requireFullStaticResultAxes(op, offsets, sizes, fullAxes)))
    return mlir::failure();
  return getIdentityResultTilePosition(op, resultNumber, offsets, sizes,
                                       resultOffsets, resultSizes);
}

mlir::LogicalResult verifyCombinerRegion(mlir::Operation *op,
                                         mlir::Region &combiner,
                                         mlir::OperandRange inputs,
                                         mlir::ResultRange results) {
  if (!combiner.hasOneBlock())
    return op->emitOpError("combiner must contain one block");

  mlir::Block &block = combiner.front();
  size_t expectedArgs = inputs.size() * 2;
  if (block.getNumArguments() != expectedArgs)
    return op->emitOpError("combiner must have two scalar arguments per input");

  for (auto [index, result] : llvm::enumerate(results)) {
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    mlir::Type elementType = resultType.getElementType();
    if (block.getArgument(index).getType() != elementType ||
        block.getArgument(index + inputs.size()).getType() != elementType)
      return op->emitOpError(
          "combiner argument types must match result element types");
  }

  auto yield =
      mlir::dyn_cast<LinalgExtCollectiveYieldOp>(block.getTerminator());
  if (!yield)
    return op->emitOpError(
        "combiner must terminate with wafer.linalg_ext.collective.yield");

  if (yield.getValues().size() != results.size()) {
    if (results.size() == 1)
      return op->emitOpError("combiner must yield exactly one scalar value");
    return op->emitOpError(
        "combiner must yield one scalar value per collective result");
  }

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), results))) {
    mlir::Value yielded = std::get<0>(yieldedAndResult);
    auto resultType = mlir::cast<mlir::RankedTensorType>(
        std::get<1>(yieldedAndResult).getType());
    if (yielded.getType() != resultType.getElementType())
      return op->emitOpError("combiner yield type ")
             << yielded.getType() << " must match result element type at index "
             << index;
  }

  return mlir::success();
}

} // namespace

mlir::LogicalResult LinalgExtCollectiveAllGatherOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(), {"axis", "partition_group", "partition_groups",
                           "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyCollectivePartitionGroups(
          getOperation(), getPartitionGroupAttr(), getPartitionGroupsAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyAllGatherLikeShape(
      getOperation(), getInputs(), getResults(), getAxisAttr(),
      getCollectivePartitionGroupSize(getPartitionGroupAttr(),
                                      getPartitionGroupsAttr()));
}

WaferLinalgExtCollectiveKind
LinalgExtCollectiveAllGatherOp::getCollectiveKind() {
  return WaferLinalgExtCollectiveKind::AllGather;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtCollectiveAllGatherOp::getLoopIteratorTypes() {
  return getLinalgExtCollectiveLoopIteratorTypes(getOperation());
}

llvm::SmallVector<mlir::Range>
LinalgExtCollectiveAllGatherOp::getIterationDomain(mlir::OpBuilder &builder) {
  return getLinalgExtCollectiveIterationDomain(getOperation(), builder);
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtCollectiveAllGatherOp::getTiledImplementation(
    mlir::OpBuilder &builder, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  llvm::SmallVector<int64_t, 1> fullAxes{static_cast<int64_t>(getAxis())};
  return buildAxisConservativeTiledCollective(getOperation(), builder,
                                              getInputs(), getOuts(), offsets,
                                              sizes, fullAxes);
}

mlir::LogicalResult LinalgExtCollectiveAllGatherOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  llvm::SmallVector<int64_t, 1> fullAxes{static_cast<int64_t>(getAxis())};
  return getAxisConservativeResultTilePosition(getOperation(), resultNumber,
                                               offsets, sizes, fullAxes,
                                               resultOffsets, resultSizes);
}

mlir::LogicalResult LinalgExtCollectiveReduceScatterOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(), {"axis", "partition_group", "partition_groups",
                           "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults(),
          /*allowPromotedInputs=*/true)))
    return mlir::failure();
  if (mlir::failed(verifyCollectivePartitionGroups(
          getOperation(), getPartitionGroupAttr(), getPartitionGroupsAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyReduceScatterLikeShape(
      getOperation(), getInputs(), getResults(), getAxisAttr(),
      getCollectivePartitionGroupSize(getPartitionGroupAttr(),
                                      getPartitionGroupsAttr()));
}

mlir::LogicalResult LinalgExtCollectiveReduceScatterOp::verifyRegions() {
  return verifyCombinerRegion(getOperation(), getCombiner(), getInputs(),
                              getResults());
}

WaferLinalgExtCollectiveKind
LinalgExtCollectiveReduceScatterOp::getCollectiveKind() {
  return WaferLinalgExtCollectiveKind::ReduceScatter;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtCollectiveReduceScatterOp::getLoopIteratorTypes() {
  return getLinalgExtCollectiveLoopIteratorTypes(getOperation());
}

llvm::SmallVector<mlir::Range>
LinalgExtCollectiveReduceScatterOp::getIterationDomain(
    mlir::OpBuilder &builder) {
  return getLinalgExtCollectiveIterationDomain(getOperation(), builder);
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtCollectiveReduceScatterOp::getTiledImplementation(
    mlir::OpBuilder &builder, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  llvm::SmallVector<int64_t, 1> fullAxes{static_cast<int64_t>(getAxis())};
  return buildAxisConservativeTiledCollective(getOperation(), builder,
                                              getInputs(), getOuts(), offsets,
                                              sizes, fullAxes);
}

mlir::LogicalResult LinalgExtCollectiveReduceScatterOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  llvm::SmallVector<int64_t, 1> fullAxes{static_cast<int64_t>(getAxis())};
  return getAxisConservativeResultTilePosition(getOperation(), resultNumber,
                                               offsets, sizes, fullAxes,
                                               resultOffsets, resultSizes);
}

mlir::LogicalResult LinalgExtCollectiveAllReduceOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(getOperation(),
                                      {"partition_group", "partition_groups",
                                       "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults(),
          /*allowPromotedInputs=*/true)))
    return mlir::failure();
  if (mlir::failed(verifyCollectivePartitionGroups(
          getOperation(), getPartitionGroupAttr(), getPartitionGroupsAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyAllReduceLikeShape(getOperation(), getInputs(), getResults());
}

mlir::LogicalResult LinalgExtCollectiveAllReduceOp::verifyRegions() {
  return verifyCombinerRegion(getOperation(), getCombiner(), getInputs(),
                              getResults());
}

WaferLinalgExtCollectiveKind
LinalgExtCollectiveAllReduceOp::getCollectiveKind() {
  return WaferLinalgExtCollectiveKind::AllReduce;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtCollectiveAllReduceOp::getLoopIteratorTypes() {
  return getLinalgExtCollectiveLoopIteratorTypes(getOperation());
}

llvm::SmallVector<mlir::Range>
LinalgExtCollectiveAllReduceOp::getIterationDomain(mlir::OpBuilder &builder) {
  return getLinalgExtCollectiveIterationDomain(getOperation(), builder);
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtCollectiveAllReduceOp::getTiledImplementation(
    mlir::OpBuilder &builder, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  return buildIdentityTiledCollective(getOperation(), builder, getInputs(),
                                      getOuts(), offsets, sizes);
}

mlir::LogicalResult LinalgExtCollectiveAllReduceOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  return getIdentityResultTilePosition(getOperation(), resultNumber, offsets,
                                       sizes, resultOffsets, resultSizes);
}

mlir::LogicalResult LinalgExtCollectiveAllToAllOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(),
          {"split_axis", "concat_axis", "split_count", "partition_group",
           "partition_groups", "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyCollectivePartitionGroups(
          getOperation(), getPartitionGroupAttr(), getPartitionGroupsAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  if (getSplitCountAttr().getInt() <= 0)
    return emitOpError("all_to_all split_count must be positive");
  return verifyAllToAllShape(getOperation(), getInputs(), getResults(),
                             getSplitAxisAttr(), getConcatAxisAttr(),
                             getSplitCountAttr().getInt());
}

WaferLinalgExtCollectiveKind
LinalgExtCollectiveAllToAllOp::getCollectiveKind() {
  return WaferLinalgExtCollectiveKind::AllToAll;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtCollectiveAllToAllOp::getLoopIteratorTypes() {
  return getLinalgExtCollectiveLoopIteratorTypes(getOperation());
}

llvm::SmallVector<mlir::Range>
LinalgExtCollectiveAllToAllOp::getIterationDomain(mlir::OpBuilder &builder) {
  return getLinalgExtCollectiveIterationDomain(getOperation(), builder);
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtCollectiveAllToAllOp::getTiledImplementation(
    mlir::OpBuilder &builder, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  llvm::SmallVector<int64_t, 2> fullAxes;
  appendUniqueAxis(fullAxes, static_cast<int64_t>(getSplitAxis()));
  appendUniqueAxis(fullAxes, static_cast<int64_t>(getConcatAxis()));
  return buildAxisConservativeTiledCollective(getOperation(), builder,
                                              getInputs(), getOuts(), offsets,
                                              sizes, fullAxes);
}

mlir::LogicalResult LinalgExtCollectiveAllToAllOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  llvm::SmallVector<int64_t, 2> fullAxes;
  appendUniqueAxis(fullAxes, static_cast<int64_t>(getSplitAxis()));
  appendUniqueAxis(fullAxes, static_cast<int64_t>(getConcatAxis()));
  return getAxisConservativeResultTilePosition(getOperation(), resultNumber,
                                               offsets, sizes, fullAxes,
                                               resultOffsets, resultSizes);
}

mlir::LogicalResult LinalgExtCollectiveCollectivePermuteOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(getOperation(),
                                      {"source_target_pairs", "channel_id"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(
          verifySourceTargetPairs(getOperation(), getSourceTargetPairsAttr())))
    return mlir::failure();
  if (mlir::failed(
          verifyChannelAttrs(getOperation(), getChannelIdAttr(), nullptr)))
    return mlir::failure();
  return verifyCollectivePermuteShape(getOperation(), getInputs(),
                                      getResults());
}

WaferLinalgExtCollectiveKind
LinalgExtCollectiveCollectivePermuteOp::getCollectiveKind() {
  return WaferLinalgExtCollectiveKind::CollectivePermute;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtCollectiveCollectivePermuteOp::getLoopIteratorTypes() {
  return getLinalgExtCollectiveLoopIteratorTypes(getOperation());
}

llvm::SmallVector<mlir::Range>
LinalgExtCollectiveCollectivePermuteOp::getIterationDomain(
    mlir::OpBuilder &builder) {
  return getLinalgExtCollectiveIterationDomain(getOperation(), builder);
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtCollectiveCollectivePermuteOp::getTiledImplementation(
    mlir::OpBuilder &builder, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  return buildIdentityTiledCollective(getOperation(), builder, getInputs(),
                                      getOuts(), offsets, sizes);
}

mlir::LogicalResult
LinalgExtCollectiveCollectivePermuteOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  return getIdentityResultTilePosition(getOperation(), resultNumber, offsets,
                                       sizes, resultOffsets, resultSizes);
}

mlir::LogicalResult
wafer::verifyLinalgExtCollectiveExecutionMesh(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](mlir::Operation *operation) {
    mlir::LogicalResult valid = mlir::success();
    if (auto allGather =
            mlir::dyn_cast<LinalgExtCollectiveAllGatherOp>(operation))
      valid = verifyPartitionGroupsAgainstExecutionMesh(
          operation, allGather.getPartitionGroupAttr(),
          allGather.getPartitionGroupsAttr());
    else if (auto reduceScatter =
                 mlir::dyn_cast<LinalgExtCollectiveReduceScatterOp>(operation))
      valid = verifyPartitionGroupsAgainstExecutionMesh(
          operation, reduceScatter.getPartitionGroupAttr(),
          reduceScatter.getPartitionGroupsAttr());
    else if (auto allReduce =
                 mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(operation))
      valid = verifyPartitionGroupsAgainstExecutionMesh(
          operation, allReduce.getPartitionGroupAttr(),
          allReduce.getPartitionGroupsAttr());
    else if (auto allToAll =
                 mlir::dyn_cast<LinalgExtCollectiveAllToAllOp>(operation))
      valid = verifyPartitionGroupsAgainstExecutionMesh(
          operation, allToAll.getPartitionGroupAttr(),
          allToAll.getPartitionGroupsAttr());
    else if (auto permute =
                 mlir::dyn_cast<LinalgExtCollectiveCollectivePermuteOp>(
                     operation)) {
      llvm::SmallVector<int64_t, 16> partitionIds;
      llvm::ArrayRef<int64_t> pairs =
          permute.getSourceTargetPairsAttr().asArrayRef();
      partitionIds.append(pairs.begin(), pairs.end());
      valid = verifyPartitionIdsWithinExecutionMesh(operation, partitionIds,
                                                    "source_target_pairs");
    }
    return mlir::failed(valid) ? mlir::WalkResult::interrupt()
                               : mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}
