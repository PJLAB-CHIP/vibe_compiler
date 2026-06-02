//===- TensorCollectiveOps.cpp - Wafer tensor collective verifier --------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

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
    if (!isAllowedAttr(attr, allowedAttrs))
      return op->emitOpError("does not accept attribute '")
             << attr.getName()
             << "'; tensor collectives must not carry physical placement, "
                "SPM, DTE, byte schedule, or runtime metadata";
  }
  return mlir::success();
}

mlir::LogicalResult verifyRankGroup(mlir::Operation *op,
                                    mlir::DenseI64ArrayAttr rankGroupAttr) {
  auto rankGroup = rankGroupAttr.asArrayRef();
  if (rankGroup.empty())
    return op->emitOpError("rank_group must not be empty");

  llvm::SmallSet<int64_t, 8> seen;
  for (int64_t rank : rankGroup) {
    if (rank < 0)
      return op->emitOpError("rank_group entries must be non-negative");
    if (!seen.insert(rank).second)
      return op->emitOpError("rank_group entries must be unique");
  }

  return mlir::success();
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

mlir::LogicalResult verifySingleDestinationStyleShape(
    mlir::Operation *op, mlir::OperandRange inputs, mlir::OperandRange outs,
    mlir::ResultRange results) {
  if (inputs.empty())
    return op->emitOpError("tensor collective must have at least one input");
  if (inputs.size() != outs.size() || inputs.size() != results.size())
    return op->emitOpError(
        "tensor collective must have matching input, out, and result counts");

  for (auto [index, values] : llvm::enumerate(llvm::zip(inputs, outs, results))) {
    mlir::Value input = std::get<0>(values);
    mlir::Value out = std::get<1>(values);
    mlir::OpResult result = std::get<2>(values);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(out.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!inputType || !outType || !resultType)
      return op->emitOpError(
          "tensor collective operands and results must be ranked tensors");
    if (outType != resultType)
      return op->emitOpError("out type must match tied result type at index ")
             << index;
    if (inputType.getElementType() != outType.getElementType())
      return op->emitOpError(
                 "input and out element types must match at index ")
             << index;
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

mlir::LogicalResult verifySameRankAndNonAxisDims(
    mlir::Operation *op, mlir::RankedTensorType inputType,
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
            "dimension times rank_group size");
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
            "dimension times rank_group size");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyAllReduceLikeShape(mlir::Operation *op,
                                             mlir::OperandRange inputs,
                                             mlir::ResultRange results) {
  for (auto [input, result] : llvm::zip(inputs, results)) {
    if (input.getType() != result.getType())
      return op->emitOpError("all_reduce input and result types must match");
  }
  return mlir::success();
}

mlir::LogicalResult verifyAllToAllShape(mlir::Operation *op,
                                        mlir::OperandRange inputs,
                                        mlir::ResultRange results,
                                        mlir::IntegerAttr splitAxisAttr,
                                        mlir::IntegerAttr concatAxisAttr,
                                        int64_t groupSize) {
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
            "dimension times rank_group size");
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
            "dimension times rank_group size");
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
  for (size_t index = 0; index < pairs.size(); index += 2) {
    int64_t source = pairs[index];
    int64_t target = pairs[index + 1];
    if (source < 0 || target < 0)
      return op->emitOpError(
          "source_target_pairs entries must be non-negative");
    if (!seenSources.insert(source).second)
      return op->emitOpError("source ranks must be unique");
    if (!seenTargets.insert(target).second)
      return op->emitOpError("target ranks must be unique");
  }
  return mlir::success();
}

void collectTensorCollectiveTilingDemand(
    mlir::OperandRange inputs, mlir::OperandRange outs,
    mlir::ResultRange results,
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  for (auto [index, input] : llvm::enumerate(inputs))
    demands.push_back({WaferTilingDemandKind::Input,
                       static_cast<unsigned>(index), input.getType()});
  for (auto [index, out] : llvm::enumerate(outs))
    demands.push_back({WaferTilingDemandKind::Output,
                       static_cast<unsigned>(index), out.getType()});
  for (auto [index, result] : llvm::enumerate(results))
    demands.push_back({WaferTilingDemandKind::Result,
                       static_cast<unsigned>(index), result.getType()});
}

mlir::LogicalResult verifyTensorCollectiveTilingContract(
    mlir::Operation *op, mlir::OperandRange inputs, mlir::OperandRange outs,
    mlir::ResultRange results) {
  llvm::SmallVector<WaferTilingDemand, 8> demands;
  collectTensorCollectiveTilingDemand(inputs, outs, results, demands);
  if (demands.empty())
    return op->emitOpError("tiling interface must expose collective tensors");
  for (const WaferTilingDemand &demand : demands) {
    if (!mlir::isa<mlir::RankedTensorType>(demand.type))
      return op->emitOpError(
          "tiling interface must expose only ranked tensors");
  }
  return mlir::success();
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

  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    mlir::Type elementType = inputType.getElementType();
    if (block.getArgument(index).getType() != elementType ||
        block.getArgument(index + inputs.size()).getType() != elementType)
      return op->emitOpError(
          "combiner argument types must match input element types");
  }

  auto yield = mlir::dyn_cast<TensorCollectiveYieldOp>(block.getTerminator());
  if (!yield)
    return op->emitOpError(
        "combiner must terminate with wafer.tensor_collective.yield");

  if (yield.getValues().size() != results.size()) {
    if (results.size() == 1)
      return op->emitOpError(
          "combiner must yield exactly one scalar value");
    return op->emitOpError(
        "combiner must yield one scalar value per collective result");
  }

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), results))) {
    mlir::Value yielded = std::get<0>(yieldedAndResult);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(std::get<1>(yieldedAndResult)
                                               .getType());
    if (yielded.getType() != resultType.getElementType())
      return op->emitOpError("combiner yield type ")
             << yielded.getType()
             << " must match result element type at index " << index;
  }

  return mlir::success();
}

} // namespace

mlir::LogicalResult TensorCollectiveAllGatherOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(), {"axis", "rank_group", "channel_id",
                           "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyRankGroup(getOperation(), getRankGroupAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyAllGatherLikeShape(getOperation(), getInputs(), getResults(),
                                  getAxisAttr(),
                                  getRankGroupAttr().asArrayRef().size());
}

void TensorCollectiveAllGatherOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  collectTensorCollectiveTilingDemand(getInputs(), getOuts(), getResults(),
                                      demands);
}

mlir::LogicalResult
TensorCollectiveAllGatherOp::verifyWaferTilingContract() {
  return verifyTensorCollectiveTilingContract(getOperation(), getInputs(),
                                             getOuts(), getResults());
}

mlir::LogicalResult TensorCollectiveReduceScatterOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(), {"axis", "rank_group", "channel_id",
                           "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyRankGroup(getOperation(), getRankGroupAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyReduceScatterLikeShape(getOperation(), getInputs(), getResults(),
                                      getAxisAttr(),
                                      getRankGroupAttr().asArrayRef().size());
}

mlir::LogicalResult TensorCollectiveReduceScatterOp::verifyRegions() {
  return verifyCombinerRegion(getOperation(), getCombiner(), getInputs(),
                              getResults());
}

void TensorCollectiveReduceScatterOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  collectTensorCollectiveTilingDemand(getInputs(), getOuts(), getResults(),
                                      demands);
}

mlir::LogicalResult
TensorCollectiveReduceScatterOp::verifyWaferTilingContract() {
  return verifyTensorCollectiveTilingContract(getOperation(), getInputs(),
                                             getOuts(), getResults());
}

mlir::LogicalResult TensorCollectiveAllReduceOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(),
          {"rank_group", "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyRankGroup(getOperation(), getRankGroupAttr())))
    return mlir::failure();
  if (mlir::failed(verifyChannelAttrs(getOperation(), getChannelIdAttr(),
                                      getUseGlobalDeviceIdsAttr())))
    return mlir::failure();
  return verifyAllReduceLikeShape(getOperation(), getInputs(), getResults());
}

mlir::LogicalResult TensorCollectiveAllReduceOp::verifyRegions() {
  return verifyCombinerRegion(getOperation(), getCombiner(), getInputs(),
                              getResults());
}

void TensorCollectiveAllReduceOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  collectTensorCollectiveTilingDemand(getInputs(), getOuts(), getResults(),
                                      demands);
}

mlir::LogicalResult TensorCollectiveAllReduceOp::verifyWaferTilingContract() {
  return verifyTensorCollectiveTilingContract(getOperation(), getInputs(),
                                             getOuts(), getResults());
}

mlir::LogicalResult TensorCollectiveAllToAllOp::verify() {
  if (mlir::failed(verifyAllowedAttrs(
          getOperation(), {"split_axis", "concat_axis", "split_count", "rank_group",
                           "channel_id", "use_global_device_ids"})))
    return mlir::failure();
  if (mlir::failed(verifySingleDestinationStyleShape(
          getOperation(), getInputs(), getOuts(), getResults())))
    return mlir::failure();
  if (mlir::failed(verifyRankGroup(getOperation(), getRankGroupAttr())))
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

void TensorCollectiveAllToAllOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  collectTensorCollectiveTilingDemand(getInputs(), getOuts(), getResults(),
                                      demands);
}

mlir::LogicalResult TensorCollectiveAllToAllOp::verifyWaferTilingContract() {
  return verifyTensorCollectiveTilingContract(getOperation(), getInputs(),
                                             getOuts(), getResults());
}

mlir::LogicalResult TensorCollectiveCollectivePermuteOp::verify() {
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

void TensorCollectiveCollectivePermuteOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  collectTensorCollectiveTilingDemand(getInputs(), getOuts(), getResults(),
                                      demands);
}

mlir::LogicalResult
TensorCollectiveCollectivePermuteOp::verifyWaferTilingContract() {
  return verifyTensorCollectiveTilingContract(getOperation(), getInputs(),
                                             getOuts(), getResults());
}
