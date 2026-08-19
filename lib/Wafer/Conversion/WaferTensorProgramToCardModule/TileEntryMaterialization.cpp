//===- TileEntryMaterialization.cpp - Per-Tile entry construction ---===//

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>

namespace wafer::tensor_program_to_card_module {

void collectTileOutputShards(
    const TileMaterializationPreparation &preparation, TileId tileId,
    llvm::SmallVectorImpl<SpatialOutputShard> &tileShards,
    llvm::SmallVectorImpl<int64_t> &coveredShardExtents) {
  for (auto [outputIndex, output] :
       llvm::enumerate(preparation.outputMappings)) {
    auto active = llvm::find(output->activeTileIds, tileId);
    if (active == output->activeTileIds.end())
      continue;
    const size_t activeOrdinal = static_cast<size_t>(
        std::distance(output->activeTileIds.begin(), active));
    const size_t activeShardCount = output->activeTileIds.size();
    const llvm::SmallVector<int64_t, 4> &domain =
        preparation.outputDomains[outputIndex];
    if (!output->shardDimension) {
      SpatialOutputShard shard;
      shard.outputIndex = static_cast<unsigned>(outputIndex);
      shard.offsets.assign(domain.size(), 0);
      shard.sizes = domain;
      shard.temporalTileSizes = output->temporalTileSizes;
      ++coveredShardExtents[outputIndex];
      tileShards.push_back(std::move(shard));
      continue;
    }
    const unsigned shardDimension = *output->shardDimension;
    const int64_t shardExtent = domain[shardDimension];
    const int64_t baseShardSize =
        shardExtent / static_cast<int64_t>(activeShardCount);
    const int64_t largerShardCount =
        shardExtent % static_cast<int64_t>(activeShardCount);
    const int64_t size = baseShardSize + (static_cast<int64_t>(activeOrdinal) <
                                          largerShardCount);
    const int64_t offset =
        static_cast<int64_t>(activeOrdinal) * baseShardSize +
        std::min<int64_t>(static_cast<int64_t>(activeOrdinal),
                          largerShardCount);
    SpatialOutputShard shard;
    shard.outputIndex = static_cast<unsigned>(outputIndex);
    shard.offsets.assign(domain.size(), 0);
    shard.sizes = domain;
    shard.offsets[shardDimension] = offset;
    shard.sizes[shardDimension] = size;
    shard.temporalTileSizes.reserve(domain.size());
    for (auto [temporalSize, shardSize] :
         llvm::zip_equal(output->temporalTileSizes, shard.sizes))
      shard.temporalTileSizes.push_back(std::min(temporalSize, shardSize));
    coveredShardExtents[outputIndex] += size;
    tileShards.push_back(std::move(shard));
  }
}

static mlir::FailureOr<mlir::func::FuncOp> lowerTileEntryFromSource(
    mlir::ModuleOp sourceModule, mlir::func::FuncOp sourceProgram,
    unsigned sourceArgumentCount,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards, bool materializeTile,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    std::string *failureReason,
    StructuredMaterializationRelations *tileRelations) {
  mlir::func::FuncOp entry;
  const bool hasEdgeAction =
      materializeTile &&
      llvm::any_of(edgeStrategies, [&](const SpatialEdgeStrategy &strategy) {
        return isSpatialEdgeStrategyIncidentOnTile(strategy, tileId);
      });
  if (hasEdgeAction) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialEdgeStrategiesToTileRegionModule(
            sourceModule, sourceArgumentCount, tileShards, tileId,
            mapping.materializationMode, edgeStrategies, loweredShard,
            failureReason,
            /*currentLogicalPartition=*/0, operationTemporalTiles,
            operationNodes, tileRelations, edgeFacts, operandDemands,
            mapping.materializationMode ==
                SpatialDataflowMaterializationMode::IndependentDDRStages)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else if (materializeTile && !tileShards.empty()) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialOutputShardsToTileRegionModule(
            sourceModule, sourceArgumentCount, tileShards, loweredShard,
            failureReason,
            /*currentLogicalPartition=*/0, operationTemporalTiles,
            operationNodes, tileRelations,
            mapping.materializationMode ==
                SpatialDataflowMaterializationMode::IndependentDDRStages)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else {
    mlir::FailureOr<mlir::func::FuncOp> noWork =
        createNoWorkEntry(sourceProgram, failureReason);
    if (mlir::failed(noWork))
      return mlir::failure();
    entry = *noWork;
  }
  return entry;
}

/// Lowers one Tile entry from the validated source epoch. The returned
/// function is detached from any module; the caller attaches it to its final
/// IR scope before `removeTileOutputDestinations` runs, because peer
/// endpoint verification resolves physical identities through the enclosing
/// module topology.
mlir::FailureOr<mlir::func::FuncOp> lowerTileEntry(
    const TileMaterializationPreparation &preparation, CardId cardId,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards, bool materializeTile,
    std::string *failureReason,
    StructuredMaterializationRelations *tileRelations,
    std::optional<llvm::ArrayRef<SpatialEdgeStrategy>> narrowedEdgeStrategies) {
  const llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies =
      narrowedEdgeStrategies
          ? *narrowedEdgeStrategies
          : llvm::ArrayRef<SpatialEdgeStrategy>(preparation.edgeStrategies);
  return lowerTileEntryFromSource(
      preparation.sourceModule, preparation.sourceProgram,
      preparation.sourceArgumentCount, preparation.operationTemporalTiles,
      preparation.operationNodes, tileId, mapping, tileShards, materializeTile,
      edgeStrategies,
      narrowedEdgeStrategies ? llvm::ArrayRef<SpatialEdgeMaterializationFacts>{}
                             : llvm::ArrayRef<SpatialEdgeMaterializationFacts>(
                                   preparation.edgeFacts),
      narrowedEdgeStrategies ? llvm::ArrayRef<analysis::ConsumerInputDemand>{}
                             : llvm::ArrayRef<analysis::ConsumerInputDemand>(
                                   preparation.consumerInputDemands),
      failureReason, tileRelations);
}
/// Checks the baseline root-scope postcondition on the final Tile entry.
static mlir::LogicalResult verifyOneStructuredRootPerRegion(
    mlir::func::FuncOp entry, TileId tileId,
    const StructuredMaterializationRelations &tileRelations,
    std::string *failureReason) {
  bool contractViolated = false;
  unsigned violatingRegion = 0;
  unsigned regionOrdinal = 0;
  llvm::SmallVector<uint32_t, 4> violatingRoots;
  llvm::SmallVector<uint32_t, 4> roots;
  entry.walk([&](TileRegionOp region) {
    if (contractViolated || region->getParentOfType<TileRegionOp>())
      return;
    const unsigned currentRegion = regionOrdinal++;
    roots.clear();
    for (const StructuredOperationEmissionRelation &relation :
         tileRelations.operationEmissions) {
      mlir::Operation *operation = relation.operation;
      if (operation && operation->getParentOfType<TileRegionOp>() == region &&
          !llvm::is_contained(roots, relation.structuredNodeId))
        roots.push_back(relation.structuredNodeId);
    }
    if (roots.empty())
      return;
    if (roots.size() != 1) {
      contractViolated = true;
      violatingRegion = currentRegion;
      violatingRoots = roots;
    }
  });
  if (contractViolated) {
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "carrier-boundary materialization on Tile "
               << tileId.getValue() << " produced compute region "
               << violatingRegion << " with " << violatingRoots.size()
               << " structured roots";
    if (!violatingRoots.empty()) {
      diagnostic << " [";
      llvm::interleaveComma(violatingRoots, diagnostic);
      diagnostic << ']';
    }
    diagnostic << "; exactly one is required; entry carries "
               << tileRelations.operationEmissions.size()
               << " structured compute relations";
    return failCardModule(failureReason, diagnostic.str());
  }
  return mlir::success();
}

/// Lowers one Tile entry from the grouped demand and carrier plan, then checks
/// that every compute region owns exactly one structured root.
mlir::FailureOr<mlir::func::FuncOp>
lowerBaselineTileEntry(const TileMaterializationPreparation &preparation,
                       CardId cardId, TileId tileId, const TileMapping &mapping,
                       llvm::ArrayRef<SpatialOutputShard> tileShards,
                       StructuredMaterializationRelations &tileRelations,
                       std::string *failureReason) {
  mlir::FailureOr<mlir::func::FuncOp> entry =
      lowerTileEntry(preparation, cardId, tileId, mapping, tileShards,
                     /*materializeTile=*/true, failureReason, &tileRelations);
  if (mlir::failed(entry))
    return entry;
  if (mlir::failed(verifyOneStructuredRootPerRegion(
          *entry, tileId, tileRelations, failureReason)))
    return mlir::failure();
  return entry;
}

} // namespace wafer::tensor_program_to_card_module
