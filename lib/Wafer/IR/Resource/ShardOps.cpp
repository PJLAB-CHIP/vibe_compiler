//===- ShardOps.cpp - Wafer shard binding verifier implementation --------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

static mlir::FailureOr<int64_t>
getExecutionMeshRankCount(ShardBindingOp binding, mlir::ModuleOp moduleOp) {
  ExecutionMeshOp meshOp =
      moduleOp.lookupSymbol<ExecutionMeshOp>(
          binding.getExecutionMeshAttr().getValue());
  if (!meshOp) {
    binding.emitOpError("references unknown execution mesh symbol @")
        << binding.getExecutionMeshAttr().getValue();
    return mlir::failure();
  }

  int64_t rankCount = 1;
  for (int64_t dim : meshOp.getShapeAttr().asArrayRef()) {
    int64_t next = 0;
    if (dim <= 0 || !checkedMul(rankCount, dim, next)) {
      binding.emitOpError("references execution mesh with invalid rank count");
      return mlir::failure();
    }
    rankCount = next;
  }
  return rankCount;
}

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

  mlir::FailureOr<int64_t> executionMeshRankCount =
      getExecutionMeshRankCount(*this, moduleOp);
  if (mlir::failed(executionMeshRankCount))
    return mlir::failure();
  int64_t rankCount = *executionMeshRankCount;

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
  if (static_cast<int64_t>(shardRanks.size()) != rankCount)
    return emitOpError(
        "shard_ranks must contain one entry per execution mesh rank");

  int64_t expectedSliceEntries = 0;
  if (!checkedMul(rankCount, tensorRank, expectedSliceEntries))
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
  for (int64_t ordinal = 0; ordinal < rankCount; ++ordinal) {
    int64_t rank = shardRanks[ordinal];
    if (rank < 0 || rank >= rankCount)
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

  return mlir::success();
}
