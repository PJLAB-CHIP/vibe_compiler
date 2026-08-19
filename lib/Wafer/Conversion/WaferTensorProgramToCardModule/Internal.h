//===- Internal.h - Tensor program to CardModule internals -------*- C++
//-*-===//

#pragma once

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

namespace wafer::tensor_program_to_card_module {

inline mlir::LogicalResult failCardModule(std::string *failureReason,
                                          llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

template <typename T>
inline mlir::FailureOr<T> failCardModuleValue(std::string *failureReason,
                                              llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

/// Validated source facts shared by all mappings over one source IR epoch.
struct TileMaterializationSourcePreparation {
  mlir::ModuleOp sourceModule;
  mlir::func::FuncOp sourceProgram;
  CardId cardId{0};
  unsigned sourceArgumentCount = 0;
  llvm::SmallVector<TileId, 16> availableTiles;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> outputDomains;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
};

/// Mapping-local input consumed by final CardModule construction.
struct TileMaterializationPreparation {
  mlir::ModuleOp sourceModule;
  mlir::func::FuncOp sourceProgram;
  unsigned sourceArgumentCount = 0;
  llvm::SmallVector<TileId, 16> availableTiles;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> outputDomains;
  llvm::SmallVector<const OutputTileMapping *, 4> outputMappings;
  llvm::SmallVector<StructuredOpTemporalTile, 16> operationTemporalTiles;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  llvm::SmallVector<SpatialEdgeStrategy, 16> edgeStrategies;
  llvm::SmallVector<SpatialEdgeMaterializationFacts, 16> edgeFacts;
  std::vector<analysis::ConsumerInputDemand> consumerInputDemands;
};

bool isCardSharedDeclaration(mlir::Operation &operation);
bool relationsBelongTo(mlir::Operation *root,
                       const StructuredMaterializationRelations &relations);
mlir::FailureOr<mlir::func::FuncOp>
getSourceTensorProgram(mlir::ModuleOp module, std::string *failureReason);
mlir::LogicalResult verifyLogicalMesh(mlir::ModuleOp module,
                                      std::string *failureReason);
mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>
getStaticOutputDomains(mlir::func::FuncOp program, std::string *failureReason);
mlir::FailureOr<mlir::presburger::PresburgerSet>
getExactStrategyDemand(const SpatialEdgeStrategy &strategy,
                       std::string *failureReason);
mlir::FailureOr<mlir::func::FuncOp>
takeLoweredTensorProgram(mlir::ModuleOp shardModule,
                         std::string *failureReason);
mlir::FailureOr<mlir::func::FuncOp>
createNoWorkEntry(mlir::func::FuncOp sourceProgram, std::string *failureReason);
mlir::LogicalResult removeTileOutputDestinations(
    mlir::func::FuncOp entry, unsigned sourceArgumentCount,
    StructuredMaterializationRelations &relations, std::string *failureReason,
    uint64_t boundaryArgumentCount = 0);
mlir::LogicalResult verifyDirectSourceMembers(mlir::ModuleOp sourceModule,
                                              mlir::func::FuncOp sourceProgram,
                                              std::string *failureReason);
void cloneModuleFacts(mlir::ModuleOp sourceModule,
                      mlir::ModuleOp destinationModule);

mlir::FailureOr<TileMaterializationSourcePreparation>
prepareTileMaterializationSource(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    std::string *failureReason);
mlir::FailureOr<TileMaterializationPreparation>
prepareTileMaterialization(const TileMaterializationSourcePreparation &source,
                           const TileMapping &mapping,
                           std::string *failureReason);

void collectTileOutputShards(
    const TileMaterializationPreparation &preparation, TileId tileId,
    llvm::SmallVectorImpl<SpatialOutputShard> &tileShards,
    llvm::SmallVectorImpl<int64_t> &coveredShardExtents);
mlir::FailureOr<mlir::func::FuncOp> lowerTileEntry(
    const TileMaterializationPreparation &preparation, CardId cardId,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards, bool materializeTile,
    std::string *failureReason,
    StructuredMaterializationRelations *tileRelations,
    std::optional<llvm::ArrayRef<SpatialEdgeStrategy>> narrowedEdgeStrategies =
        std::nullopt);
mlir::FailureOr<mlir::func::FuncOp>
lowerBaselineTileEntry(const TileMaterializationPreparation &preparation,
                       CardId cardId, TileId tileId, const TileMapping &mapping,
                       llvm::ArrayRef<SpatialOutputShard> tileShards,
                       StructuredMaterializationRelations &tileRelations,
                       std::string *failureReason);

mlir::LogicalResult lowerPreparedTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    const TileMaterializationPreparation &preparation,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    StructuredMaterializationRelations *materializationRelations,
    CardModuleMaterializationStatistics *statistics);

} // namespace wafer::tensor_program_to_card_module
