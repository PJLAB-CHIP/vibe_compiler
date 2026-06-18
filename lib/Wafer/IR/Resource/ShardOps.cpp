//===- ShardOps.cpp - Wafer shard binding verifier implementation --------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult ShardBindingOp::verify() {
  mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError("must be nested under a module");

  mlir::func::FuncOp funcOp =
      moduleOp.lookupSymbol<mlir::func::FuncOp>(getKernelAttr().getValue());
  if (!funcOp)
    return emitOpError("references unknown function symbol @")
           << getKernelAttr().getValue();

  int64_t argumentIndex = getArgumentIndexAttr().getInt();
  if (argumentIndex < 0 ||
      argumentIndex >=
          static_cast<int64_t>(funcOp.getFunctionType().getNumInputs()))
    return emitOpError("argument_index is outside the referenced function "
                       "argument list");

  mlir::Type argumentType = funcOp.getFunctionType().getInput(argumentIndex);
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(argumentType);
  if (!tensorType || !tensorType.hasStaticShape())
    return emitOpError("referenced function argument must be a static ranked "
                       "tensor");

  int64_t logicalRankCount = getLogicalRankCountAttr().getInt();
  if (logicalRankCount <= 0)
    return emitOpError("logical rank count must be positive");

  llvm::ArrayRef<int64_t> globalShape = getGlobalShapeAttr().asArrayRef();
  llvm::ArrayRef<int64_t> localShape = getLocalShapeAttr().asArrayRef();
  int64_t tensorRank = tensorType.getRank();
  if (static_cast<int64_t>(globalShape.size()) != tensorRank)
    return emitOpError("global_shape rank must match referenced argument rank");
  if (static_cast<int64_t>(localShape.size()) != tensorRank)
    return emitOpError("local_shape rank must match referenced argument rank");

  for (auto [dim, expected] : llvm::enumerate(tensorType.getShape())) {
    if (expected < 0)
      return emitOpError("referenced function argument must have static "
                         "dimensions");
    if (localShape[dim] != expected)
      return emitOpError("local_shape must match referenced argument type");
    if (globalShape[dim] < 0 || localShape[dim] < 0)
      return emitOpError("global_shape and local_shape dimensions must be "
                         "non-negative");
  }

  llvm::ArrayRef<int64_t> shardRanks = getShardRanksAttr().asArrayRef();
  if (static_cast<int64_t>(shardRanks.size()) != logicalRankCount)
    return emitOpError("shard_ranks must contain one entry per logical rank");

  int64_t expectedSliceEntries = 0;
  if (!checkedMul(logicalRankCount, tensorRank, expectedSliceEntries))
    return emitOpError("shard slice arrays are too large to verify");

  llvm::ArrayRef<int64_t> shardOffsets = getShardOffsetsAttr().asArrayRef();
  llvm::ArrayRef<int64_t> shardSizes = getShardSizesAttr().asArrayRef();
  llvm::ArrayRef<int64_t> shardStrides = getShardStridesAttr().asArrayRef();
  if (static_cast<int64_t>(shardOffsets.size()) != expectedSliceEntries ||
      static_cast<int64_t>(shardSizes.size()) != expectedSliceEntries ||
      static_cast<int64_t>(shardStrides.size()) != expectedSliceEntries)
    return emitOpError("shard_offsets, shard_sizes and shard_strides must "
                       "contain one slice tuple per logical rank");

  llvm::DenseSet<int64_t> usedRanks;
  for (int64_t ordinal = 0; ordinal < logicalRankCount; ++ordinal) {
    int64_t rank = shardRanks[ordinal];
    if (rank < 0 || rank >= logicalRankCount)
      return emitOpError("shard logical rank is out of range");
    if (!usedRanks.insert(rank).second)
      return emitOpError("maps multiple shard slices to logical rank ") << rank;

    int64_t base = ordinal * tensorRank;
    for (int64_t dim = 0; dim < tensorRank; ++dim) {
      int64_t offset = shardOffsets[base + dim];
      int64_t size = shardSizes[base + dim];
      int64_t stride = shardStrides[base + dim];
      if (offset < 0 || size < 0)
        return emitOpError("shard offsets and sizes must be non-negative");
      if (stride <= 0)
        return emitOpError("shard strides must be positive");
      if (offset > globalShape[dim] || size > globalShape[dim] - offset)
        return emitOpError("shard slice for logical rank ")
               << rank << " exceeds global shape";
      if (size > localShape[dim])
        return emitOpError("shard slice for logical rank ")
               << rank << " exceeds local shape";
    }
  }

  bool duplicateBinding = false;
  moduleOp.walk([&](ShardBindingOp binding) {
    if (binding.getOperation() == getOperation())
      return;
    if (binding.getKernelAttr() == getKernelAttr() &&
        binding.getArgumentIndexAttr().getInt() == argumentIndex)
      duplicateBinding = true;
  });
  if (duplicateBinding)
    return emitOpError("duplicates shard binding for referenced function "
                       "argument");

  bool shardRankMismatch = false;
  int64_t existingShardRankCount = 0;
  moduleOp.walk([&](ShardBindingOp binding) {
    if (binding.getOperation() == getOperation())
      return;
    existingShardRankCount = binding.getLogicalRankCountAttr().getInt();
    if (existingShardRankCount != logicalRankCount)
      shardRankMismatch = true;
  });
  if (shardRankMismatch)
    return emitOpError("shard binding logical rank count ")
           << logicalRankCount
           << " does not match existing shard binding logical rank count "
           << existingShardRankCount;

  bool placementRankMismatch = false;
  int64_t placementRankCount = 0;
  moduleOp.walk([&](PlacementMapOp placementMap) {
    placementRankCount = placementMap.getLogicalRankCountAttr().getInt();
    if (placementRankCount != logicalRankCount)
      placementRankMismatch = true;
  });
  if (placementRankMismatch)
    return emitOpError("shard binding logical rank count ")
           << logicalRankCount
           << " does not match placement map logical rank count "
           << placementRankCount;

  return mlir::success();
}
