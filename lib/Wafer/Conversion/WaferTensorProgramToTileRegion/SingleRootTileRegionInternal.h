//===- SingleRootTileRegionInternal.h - Region construction internals -*- C++
//-*-===//

#pragma once

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
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

mlir::FailureOr<mlir::func::FuncOp> buildCoupledRootFunction(
    mlir::Block &destination, llvm::ArrayRef<mlir::Operation *> sourceRoots,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    llvm::ArrayRef<uint32_t> coupledNodeIds,
    llvm::ArrayRef<uint32_t> recomputedNodeIds, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount);

void retainLiveOperationNodes(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes);

mlir::LogicalResult
bindFullResultsToOutputDestinations(mlir::func::FuncOp function,
                                    unsigned functionalArgumentCount,
                                    std::string *failureReason);

mlir::FailureOr<RootFragment> materializeRootFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const StructuredNodeIterationShard &shard,
    const StructuredNodeTemporalTile *temporal,
    const StructuredNodePhysicalRepresentation *representation,
    const StructuredNodeComputeImplementation *implementation,
    std::string *failureReason);

mlir::FailureOr<RootFragment> materializeCoupledRootFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const StructuredNodeShardGroup &group, std::string *failureReason);

mlir::FailureOr<RootFragment> materializeReductionMergeFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    uint32_t structuredNodeId,
    llvm::ArrayRef<const StructuredNodeIterationShard *> contributionShards,
    llvm::ArrayRef<mlir::func::FuncOp> contributionFunctions,
    const StructuredNodePhysicalRepresentation *representation,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
