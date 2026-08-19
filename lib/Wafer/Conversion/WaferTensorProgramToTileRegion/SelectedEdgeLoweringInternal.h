//===- SelectedEdgeLoweringInternal.h - Private edge lowering state -------===//

#ifndef WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_SELECTEDEDGELOWERINGINTERNAL_H
#define WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_SELECTEDEDGELOWERINGINTERNAL_H

#include "ConsumerInputReconstruction.h"
#include "SelectedEdgeMapping.h"

namespace wafer::tensor_program_to_tile_region {

struct SelectedEdgeLoweringState {
  SelectedEdgeLoweringState(
      mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
      llvm::ArrayRef<SpatialOutputShard> outputShards, TileId currentTile,
      int64_t currentLogicalPartition, bool requireOneStructuredRootPerRegion,
      llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
      mlir::OwningOpRef<mlir::ModuleOp> candidate,
      mlir::func::FuncOp candidateFunction, SelectedEdgeProgramMapping mapping,
      std::string *failureReason)
      : sourceModule(sourceModule),
        functionalArgumentCount(functionalArgumentCount),
        outputShards(outputShards), currentTile(currentTile),
        currentLogicalPartition(currentLogicalPartition),
        requireOneStructuredRootPerRegion(requireOneStructuredRootPerRegion),
        sourceOperationNodes(sourceOperationNodes),
        candidate(std::move(candidate)),
        scope(candidateFunction, functionalArgumentCount),
        mapping(std::move(mapping)), failureReason(failureReason) {}

  mlir::ModuleOp sourceModule;
  unsigned functionalArgumentCount;
  llvm::ArrayRef<SpatialOutputShard> outputShards;
  TileId currentTile;
  int64_t currentLogicalPartition;
  bool requireOneStructuredRootPerRegion;
  llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes;
  mlir::OwningOpRef<mlir::ModuleOp> candidate;
  TensorProgramScope scope;
  SelectedEdgeProgramMapping mapping;
  llvm::SmallVector<CandidatePeerEndpoint, 8> endpoints;
  llvm::SmallVector<MaterializedSource, 8> materializedSources;
  llvm::SmallVector<MaterializedRegionCutSpill, 8> regionCutSpills;
  llvm::SmallVector<CandidateSelectedDDRStage, 8> selectedDDRStages;
  llvm::DenseSet<mlir::Operation *> preservedOperations;
  llvm::SmallVector<ProducerValue, 8> producerValues;
  std::string *failureReason;
};

mlir::LogicalResult reportSelectedEdgeFailure(std::string *failureReason,
                                              llvm::StringRef message);

mlir::LogicalResult
materializeSelectedDirectActions(SelectedEdgeLoweringState &state);
mlir::LogicalResult
materializeSelectedPeerReceives(SelectedEdgeLoweringState &state);
mlir::LogicalResult
verifySelectedReceiveOwners(SelectedEdgeLoweringState &state);
bool selectedValueReachesDDRStage(SelectedEdgeLoweringState &state,
                                  mlir::Operation *source);
mlir::LogicalResult
materializeSelectedSourceStages(SelectedEdgeLoweringState &state);
mlir::LogicalResult
materializeSelectedConsumerStages(SelectedEdgeLoweringState &state);
mlir::LogicalResult
requireLiveSelectedReceives(SelectedEdgeLoweringState &state,
                            llvm::StringRef stage);
mlir::LogicalResult
materializeSelectedPeerSends(SelectedEdgeLoweringState &state);
mlir::LogicalResult
materializeSelectedOutputs(SelectedEdgeLoweringState &state);
mlir::LogicalResult
verifySelectedReceiveExecutionSinks(SelectedEdgeLoweringState &state);
mlir::LogicalResult finishSelectedEdgeLowering(
    SelectedEdgeLoweringState &state,
    mlir::OwningOpRef<mlir::ModuleOp> &resultModule,
    StructuredMaterializationRelations *materializationRelations);

} // namespace wafer::tensor_program_to_tile_region

#endif // WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_SELECTEDEDGELOWERINGINTERNAL_H
