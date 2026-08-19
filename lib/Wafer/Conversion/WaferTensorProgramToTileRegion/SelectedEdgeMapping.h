//===- SelectedEdgeMapping.h - Map selected edges to candidate -*- C++ -*-===//
#pragma once

#include "SelectedEdgeMaterialization.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "mlir/IR/IRMapping.h"

#include <vector>

namespace wafer::tensor_program_to_tile_region {

struct SelectedEdgeProgramMapping {
  llvm::SmallVector<StructuredOpTemporalTile, 16> operationTemporalTiles;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  std::vector<analysis::ConsumerInputDemand> consumerInputDemands;
  llvm::SmallVector<MappedStrategy, 16> strategies;
  bool independentDDRStages = false;
};

mlir::FailureOr<SelectedEdgeProgramMapping> mapSelectedEdgesToCandidate(
    mlir::Block &sourceBody, mlir::IRMapping &cloneMapping,
    TensorProgramScope candidateScope, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
