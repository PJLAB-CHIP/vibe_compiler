//===- StructuredMaterializationRelations.h - Current IR ownership -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H
#define WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer {

struct StructuredOperationBufferRelation {
  uint32_t structuredNodeId = 0;
  mlir::Value buffer;
};

struct StructuredOperationResultBufferRelation {
  uint32_t structuredNodeId = 0;
  unsigned resultIndex = 0;
  mlir::Value buffer;
};

struct StructuredOperationEmissionRelation {
  uint32_t structuredNodeId = 0;
  mlir::Operation *operation = nullptr;
};

struct SpatialOutputBufferRelation {
  unsigned outputIndex = 0;
  mlir::Value buffer;
};

/// One observable TensorProgram result piece that already exists as a current
/// structural tensor endpoint. Layout/bufferization retargets this relation to
/// the corresponding actual destination before it is converted into the
/// buffer-level output relation above.
struct StructuredOutputRelation {
  unsigned outputIndex = 0;
  mlir::Value endpoint;
};

struct DDRBufferRelation {
  int64_t resourceId = -1;
  mlir::Value buffer;
};

struct DDRTransferRelation {
  uint32_t producerNodeId = 0;
  unsigned producerResult = 0;
  TileId producerTile{0};
  TileId consumerTile{0};
  int64_t resourceId = -1;
};

/// One selected external Region binding whose two current tensor endpoints
/// already exist in the candidate IR. Same-Tile endpoints remain directly
/// connected by SSA; cross-Tile endpoints cannot be joined by SSA because
/// TileModule is IsolatedFromAbove. Movement consumes this exact relation after
/// layout/bufferization. No route, buffer, storage, event or completion fact is
/// represented here.
struct StructuredBoundaryRelation {
  mlir::Value sourceEndpoint;
  mlir::Value destinationEndpoint;
};

struct PartialReductionContributionBufferRelation {
  uint32_t structuredNodeId = 0;
  compiler::detail::ReductionGroupId group;
  unsigned resultIndex = 0;
  TileId sourceTile{0};
  mlir::Value buffer;
};

struct PartialReductionMergeInputBufferRelation {
  uint32_t structuredNodeId = 0;
  compiler::detail::ReductionGroupId group;
  unsigned resultIndex = 0;
  TileId sourceTile{0};
  mlir::Value buffer;
};

/// Actual operation/buffer ownership recorded during one candidate rewrite.
/// Every pointer/value is valid only for the owning current IR epoch.
struct StructuredMaterializationRelations {
  llvm::SmallVector<StructuredOperationEmissionRelation, 16> operationEmissions;
  llvm::SmallVector<StructuredOperationResultBufferRelation, 16>
      operationResultBuffers;
  llvm::SmallVector<StructuredOperationBufferRelation, 16> operandBuffers;
  llvm::SmallVector<StructuredOperationBufferRelation, 16> scratchBuffers;
  llvm::SmallVector<SpatialOutputBufferRelation, 4> outputBuffers;
  llvm::SmallVector<StructuredOutputRelation, 8> structuralOutputs;
  llvm::SmallVector<DDRBufferRelation, 8> ddrBuffers;
  llvm::SmallVector<DDRTransferRelation, 8> ddrTransfers;
  llvm::SmallVector<StructuredBoundaryRelation, 8> boundaryRelations;
  llvm::SmallVector<PartialReductionContributionBufferRelation, 8>
      partialReductionContributions;
  llvm::SmallVector<PartialReductionMergeInputBufferRelation, 8>
      partialReductionMergeInputs;
};

} // namespace wafer

#endif // WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H
