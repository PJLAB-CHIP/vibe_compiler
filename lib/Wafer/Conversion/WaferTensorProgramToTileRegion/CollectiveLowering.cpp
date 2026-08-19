//===- CollectiveLowering.cpp - Card collective lowering ------------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<SelectedCollectivePartitionGroup>
TileRegionBodyEmitter::getCollectivePartitionGroup(
    mlir::DenseI64ArrayAttr partitionGroup,
    mlir::DenseIntElementsAttr partitionGroups) {
  auto findLocalPartition = [&](llvm::ArrayRef<int64_t> partitionIds)
      -> std::optional<SelectedCollectivePartitionGroup> {
    for (auto [index, logicalPartition] : llvm::enumerate(partitionIds)) {
      if (logicalPartition != currentLogicalPartition)
        continue;
      SelectedCollectivePartitionGroup selected;
      selected.partitionIds.assign(partitionIds.begin(), partitionIds.end());
      selected.localParticipantIndex = static_cast<int64_t>(index);
      return selected;
    }
    return std::nullopt;
  };

  if (partitionGroup) {
    if (std::optional<SelectedCollectivePartitionGroup> selected =
            findLocalPartition(partitionGroup.asArrayRef()))
      return *selected;
    return failSelectedCollectivePartitionGroup(
        "logical card partition is not a member of collective "
        "partition_group");
  }

  if (!partitionGroups)
    return failSelectedCollectivePartitionGroup(
        "collective materialization requires partition_group or "
        "partition_groups");
  auto groupsType =
      mlir::dyn_cast<mlir::RankedTensorType>(partitionGroups.getType());
  if (!groupsType || groupsType.getRank() != 2 || groupsType.getDimSize(1) <= 0)
    return failSelectedCollectivePartitionGroup(
        "collective partition_groups are malformed");
  int64_t partitionGroupSize = groupsType.getDimSize(1);
  llvm::SmallVector<int64_t, 8> partitionIds;
  for (llvm::APInt value : partitionGroups.getValues<llvm::APInt>())
    partitionIds.push_back(value.getSExtValue());
  if (partitionIds.size() % static_cast<size_t>(partitionGroupSize) != 0)
    return failSelectedCollectivePartitionGroup(
        "collective partition_groups are malformed");

  for (size_t offset = 0; offset < partitionIds.size();
       offset += static_cast<size_t>(partitionGroupSize)) {
    llvm::ArrayRef<int64_t> group(partitionIds.data() + offset,
                                  static_cast<size_t>(partitionGroupSize));
    if (std::optional<SelectedCollectivePartitionGroup> selected =
            findLocalPartition(group))
      return *selected;
  }
  return failSelectedCollectivePartitionGroup(
      "logical card partition is not a member of collective "
      "partition_groups");
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

  auto convertedTensorType =
      mlir::RankedTensorType::get(inputType.getShape(), resultElementType);
  auto converted = builder.create<ComputeConvertOp>(
      loc, makeSPMMemRefType(convertedTensorType, MemLayout::Tensor), input);
  recordStructuredComputeOperation(converted);
  return converted.getResult();
}

static bool
isSingleton(const mlir::FailureOr<SelectedCollectivePartitionGroup> &group) {
  return mlir::succeeded(group) && group->partitionIds.size() == 1;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllGather(LinalgExtCollectiveAllGatherOp op,
                                        mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())) ||
      op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_gather materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectivePartitionGroup> group =
      getCollectivePartitionGroup(op.getPartitionGroupAttr(),
                                  op.getPartitionGroupsAttr());
  if (mlir::failed(group))
    return mlir::failure();
  if (!isSingleton(group))
    return fail("non-singleton card-level all_gather lowering is unsupported; "
                "Tile communication must be selected by card "
                "dataflow synthesis");
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  record(op.getResult(0), MemLayout::Tensor, *input);
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertReduceScatter(
    LinalgExtCollectiveReduceScatterOp op, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())) ||
      op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "reduce_scatter materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectivePartitionGroup> group =
      getCollectivePartitionGroup(op.getPartitionGroupAttr(),
                                  op.getPartitionGroupsAttr());
  if (mlir::failed(group))
    return mlir::failure();
  if (!isSingleton(group))
    return fail(
        "non-singleton card-level reduce_scatter lowering is unsupported; "
        "Tile communication must be selected by card dataflow "
        "synthesis");
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType(0));
  if (!resultType)
    return fail("reduce_scatter result must be ranked");
  input = materializeCollectiveInputInResultType(
      *input, resultType.getElementType(), op.getLoc(), builder);
  if (mlir::failed(input))
    return mlir::failure();
  record(op.getResult(0), MemLayout::Tensor, *input);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllReduce(LinalgExtCollectiveAllReduceOp op,
                                        mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())) ||
      op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_reduce materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectivePartitionGroup> group =
      getCollectivePartitionGroup(op.getPartitionGroupAttr(),
                                  op.getPartitionGroupsAttr());
  if (mlir::failed(group))
    return mlir::failure();
  if (!isSingleton(group))
    return fail("non-singleton card-level all_reduce lowering is unsupported; "
                "Tile communication must be selected by card "
                "dataflow synthesis");
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType(0));
  if (!resultType)
    return fail("all_reduce result must be ranked");
  input = materializeCollectiveInputInResultType(
      *input, resultType.getElementType(), op.getLoc(), builder);
  if (mlir::failed(input))
    return mlir::failure();
  record(op.getResult(0), MemLayout::Tensor, *input);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertAllToAll(LinalgExtCollectiveAllToAllOp op,
                                       mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())) ||
      op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail("all_to_all materialization supports one input and one out");
  mlir::FailureOr<SelectedCollectivePartitionGroup> group =
      getCollectivePartitionGroup(op.getPartitionGroupAttr(),
                                  op.getPartitionGroupsAttr());
  if (mlir::failed(group))
    return mlir::failure();
  if (!isSingleton(group))
    return fail("non-singleton card-level all_to_all lowering is unsupported; "
                "Tile communication must be selected by card "
                "dataflow synthesis");
  if (op.getSplitCount() != 1)
    return fail("singleton all_to_all requires split_count = 1");
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  record(op.getResult(0), MemLayout::Tensor, *input);
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertCollectivePermute(
    LinalgExtCollectiveCollectivePermuteOp op, mlir::OpBuilder &builder) {
  if (mlir::failed(requireSingleTensorCollective(op.getOperation())) ||
      op.getInputs().size() != 1 || op.getOuts().size() != 1)
    return fail(
        "collective_permute materialization supports one input and one out");
  llvm::ArrayRef<int64_t> pairs = op.getSourceTargetPairsAttr().asArrayRef();
  if (pairs.size() != 2 || pairs[0] != currentLogicalPartition ||
      pairs[1] != currentLogicalPartition)
    return fail(
        "non-singleton card-level collective_permute lowering is unsupported; "
        "Tile communication must be selected by card dataflow "
        "synthesis");
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getInputs().front(), MemLayout::Tensor, builder);
  if (mlir::failed(input))
    return mlir::failure();
  record(op.getResult(0), MemLayout::Tensor, *input);
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
