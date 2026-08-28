//===- StructuredMaterializationRelations.h - Current IR ownership -*- C++
//-*-===//

#ifndef WAFER_ANALYSIS_STRUCTURED_STRUCTUREDMATERIALIZATIONRELATIONS_H
#define WAFER_ANALYSIS_STRUCTURED_STRUCTUREDMATERIALIZATIONRELATIONS_H

#include "Wafer/Analysis/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer {

struct StructuredOperationNodeMapping {
  mlir::Operation *operation = nullptr;
  uint32_t structuredNodeId = 0;
};

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

struct CardDDRBufferRelation {
  int64_t resourceId = -1;
  mlir::Value buffer;
};

struct CardDDRTransferRelation {
  uint32_t producerNodeId = 0;
  unsigned producerResult = 0;
  TileId producerTile{0};
  TileId consumerTile{0};
  int64_t resourceId = -1;
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
  llvm::SmallVector<CardDDRBufferRelation, 8> cardDDRBuffers;
  llvm::SmallVector<CardDDRTransferRelation, 8> cardDDRTransfers;
  llvm::SmallVector<PartialReductionContributionBufferRelation, 8>
      partialReductionContributions;
  llvm::SmallVector<PartialReductionMergeInputBufferRelation, 8>
      partialReductionMergeInputs;
};

} // namespace wafer

#endif // WAFER_ANALYSIS_STRUCTURED_STRUCTUREDMATERIALIZATIONRELATIONS_H
