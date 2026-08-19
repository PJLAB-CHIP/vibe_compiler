//===- SingleRootTileRegionInternal.h - Region construction internals -*- C++
//-*-===//

#pragma once

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"

namespace wafer::tensor_program_to_tile_region {

struct RootFragment {
  mlir::func::FuncOp function;
  StructuredMaterializationRelations relations;
};

mlir::FailureOr<mlir::func::FuncOp> buildRootFunction(
    mlir::Block &destination, mlir::Operation *sourceRoot,
    uint32_t structuredNodeId,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount);

void retainLiveOperationNodes(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes);

mlir::FailureOr<RootFragment> materializeRootFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const StructuredNodeIterationShard &shard, std::string *failureReason);

mlir::FailureOr<RootFragment> materializeReductionMergeFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    uint32_t structuredNodeId,
    llvm::ArrayRef<const StructuredNodeIterationShard *> contributionShards,
    llvm::ArrayRef<mlir::func::FuncOp> contributionFunctions,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
