//===- SelectedEdgeMaterialization.h - Apply selected edge actions -*- C++
//-*-===//
#pragma once

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

namespace wafer::tensor_program_to_tile_region {

struct MappedStrategy {
  SpatialEdgeStrategy strategy;
  mlir::Operation *producer = nullptr;
  mlir::Operation *consumer = nullptr;
  mlir::Operation *sourceProducer = nullptr;
  mlir::Operation *sourceConsumer = nullptr;
  uint64_t consumerScheduleOrdinal = 0;
  bool requiresConsumerInputReconstruction = false;
};

struct MaterializedSource {
  mlir::Operation *producer = nullptr;
  unsigned result = 0;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  mlir::Value value;
};

struct MaterializedRegionCutSpill {
  mlir::Operation *producer = nullptr;
  unsigned result = 0;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  mlir::Value value;
  mlir::Value buffer;
};

mlir::FailureOr<mlir::Value> getOrMaterializeSource(
    TensorProgramScope scope, mlir::Operation *producer,
    unsigned producerResult, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes);

void eraseDeadExcept(TensorProgramScope scope,
                     const llvm::DenseSet<mlir::Operation *> &preserved);

bool isInBackwardClosure(mlir::Value value, mlir::Operation *needle,
                         llvm::DenseSet<mlir::Value> &visited);

bool isInSelectedOutputClosure(TensorProgramScope scope,
                               llvm::ArrayRef<SpatialOutputShard> outputShards,
                               mlir::Operation *operation);

bool hasSplitSelectedOutputTraversal(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    mlir::Operation *operation);

mlir::Value createExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes);

mlir::Value insertExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source, mlir::Value destination,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes);

std::optional<uint32_t> findStructuredNodeId(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes);

mlir::LogicalResult materializeLocalShardResidency(MappedStrategy &mapped,
                                                   std::string *failureReason);

mlir::LogicalResult materializeSpill(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::SmallVectorImpl<MaterializedRegionCutSpill> &regionCutSpills,
    llvm::SmallVectorImpl<CandidateSelectedDDRStage> &selectedDDRStages,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes);

mlir::LogicalResult materializeRecompute(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
