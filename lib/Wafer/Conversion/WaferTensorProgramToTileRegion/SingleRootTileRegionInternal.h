//===- SingleRootTileRegionInternal.h - Region construction internals -*- C++
//-*-===//

#pragma once

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"

#include <tuple>

namespace wafer::tensor_program_to_tile_region {

enum class RootValueKind : uint8_t { SourceArgument, StructuredResult };

struct RootValueKey {
  RootValueKind kind = RootValueKind::SourceArgument;
  uint32_t owner = 0;
  unsigned resultIndex = 0;

  friend bool operator<(const RootValueKey &lhs, const RootValueKey &rhs) {
    return std::tie(lhs.kind, lhs.owner, lhs.resultIndex) <
           std::tie(rhs.kind, rhs.owner, rhs.resultIndex);
  }
};

struct RootFragment {
  mlir::func::FuncOp function;
  StructuredMaterializationRelations relations;
  llvm::SmallVector<RootValueKey, 8> boundaries;
  llvm::SmallVector<RootValueKey, 4> results;
};

mlir::FailureOr<mlir::func::FuncOp> buildRootFunction(
    mlir::Block &destination, mlir::Operation *sourceRoot,
    uint32_t structuredNodeId,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount,
    llvm::SmallVectorImpl<RootValueKey> &boundaries,
    llvm::SmallVectorImpl<RootValueKey> &results);

mlir::FailureOr<mlir::func::FuncOp> buildCoupledRootFunction(
    mlir::Block &destination, llvm::ArrayRef<mlir::Operation *> sourceRoots,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    llvm::ArrayRef<uint32_t> coupledNodeIds,
    llvm::ArrayRef<uint32_t> recomputedNodeIds,
    llvm::ArrayRef<uint32_t> independentlyMaterializedNodeIds,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount,
    llvm::SmallVectorImpl<RootValueKey> &boundaries,
    llvm::SmallVectorImpl<RootValueKey> &results);

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
    const compiler::detail::ReductionGroupId &group,
    llvm::ArrayRef<const StructuredNodeIterationShard *> contributionShards,
    llvm::ArrayRef<mlir::func::FuncOp> contributionFunctions,
    const StructuredNodePhysicalRepresentation *representation,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
