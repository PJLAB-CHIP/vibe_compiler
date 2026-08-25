//===- TileEntryMaterialization.cpp - Per-Tile entry construction ---===//

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>

namespace wafer::tensor_program_to_card_module {

static llvm::StringRef stringifyEdgeAction(SpatialEdgeAction action) {
  switch (action) {
  case SpatialEdgeAction::RecursiveProducerTiling:
    return "recursive-producer-tiling";
  case SpatialEdgeAction::LocalShardResidency:
    return "local-shard-residency";
  case SpatialEdgeAction::PeerFragments:
    return "peer-fragments";
  case SpatialEdgeAction::SpillReload:
    return "spill-reload";
  case SpatialEdgeAction::Recompute:
    return "recompute";
  case SpatialEdgeAction::RegionCut:
    return "region-cut";
  case SpatialEdgeAction::CardDDRTransfer:
    return "card-ddr-transfer";
  }
  llvm_unreachable("unknown spatial edge action");
}

void collectTileOutputShards(
    const TileMaterializationPreparation &preparation, TileId tileId,
    llvm::SmallVectorImpl<SpatialOutputShard> &tileShards,
    llvm::SmallVectorImpl<int64_t> &coveredShardExtents) {
  for (auto [outputIndex, output] :
       llvm::enumerate(preparation.outputMappings)) {
    auto selected =
        llvm::find_if(output->shards, [tileId](const OutputTileShard &shard) {
          return shard.tile == tileId;
        });
    if (selected == output->shards.end())
      continue;
    const llvm::SmallVector<int64_t, 4> &domain =
        preparation.outputDomains[outputIndex];
    SpatialOutputShard shard;
    shard.outputIndex = static_cast<unsigned>(outputIndex);
    shard.offsets = selected->offsets;
    shard.sizes = selected->sizes;
    shard.temporalTileSizes.reserve(domain.size());
    for (auto [temporalSize, shardSize] :
         llvm::zip_equal(output->temporalTileSizes, shard.sizes))
      shard.temporalTileSizes.push_back(std::min(temporalSize, shardSize));
    int64_t elements = 1;
    for (int64_t size : shard.sizes)
      elements *= size;
    coveredShardExtents[outputIndex] += elements;
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
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
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
      narrowedEdgeStrategies ? llvm::ArrayRef<analysis::DependencyDemand>{}
                             : llvm::ArrayRef<analysis::DependencyDemand>(
                                   preparation.consumerInputDemands),
      failureReason, tileRelations);
}
/// Checks the baseline root-scope postcondition on the final Tile entry.
static mlir::LogicalResult verifyOneStructuredRootPerRegion(
    mlir::func::FuncOp entry, TileId tileId,
    const StructuredMaterializationRelations &tileRelations,
    llvm::ArrayRef<StructuredNodeRootGroup> operationRootGroups,
    std::string *failureReason) {
  llvm::DenseMap<uint32_t, uint32_t> rootGroupByNode;
  for (const StructuredNodeRootGroup &relation : operationRootGroups)
    if (!rootGroupByNode
             .try_emplace(relation.structuredNodeId, relation.rootGroupId)
             .second)
      return failCardModule(
          failureReason,
          "card structured node/root-group relation is duplicated");
  bool contractViolated = false;
  bool missingRootGroup = false;
  unsigned violatingRegion = 0;
  unsigned regionOrdinal = 0;
  llvm::SmallVector<uint32_t, 4> violatingNodes;
  llvm::SmallVector<uint32_t, 4> violatingRootGroups;
  llvm::SmallVector<std::string, 4> violatingOperations;
  llvm::SmallVector<uint32_t, 4> nodes;
  llvm::SmallVector<uint32_t, 4> rootGroups;
  llvm::SmallVector<std::string, 4> operations;
  entry.walk([&](TileRegionOp region) {
    if (contractViolated || missingRootGroup ||
        region->getParentOfType<TileRegionOp>())
      return;
    const unsigned currentRegion = regionOrdinal++;
    nodes.clear();
    rootGroups.clear();
    operations.clear();
    for (const StructuredOperationEmissionRelation &relation :
         tileRelations.operationEmissions) {
      mlir::Operation *operation = relation.operation;
      if (operation && operation->getParentOfType<TileRegionOp>() == region &&
          !llvm::is_contained(nodes, relation.structuredNodeId)) {
        nodes.push_back(relation.structuredNodeId);
        auto root = rootGroupByNode.find(relation.structuredNodeId);
        if (root == rootGroupByNode.end()) {
          missingRootGroup = true;
          return;
        }
        if (!llvm::is_contained(rootGroups, root->second))
          rootGroups.push_back(root->second);
        operations.push_back(operation->getName().getStringRef().str());
      }
    }
    if (nodes.empty())
      return;
    if (rootGroups.size() != 1) {
      contractViolated = true;
      violatingRegion = currentRegion;
      violatingNodes = nodes;
      violatingRootGroups = rootGroups;
      violatingOperations = operations;
    }
  });
  if (missingRootGroup)
    return failCardModule(
        failureReason,
        "materialized structured operation has no current root group");
  if (contractViolated) {
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "carrier-boundary materialization on Tile "
               << tileId.getValue() << " produced compute region "
               << violatingRegion << " with " << violatingRootGroups.size()
               << " semantic roots";
    if (!violatingNodes.empty()) {
      diagnostic << "; structured_nodes=[";
      llvm::interleaveComma(violatingNodes, diagnostic);
      diagnostic << "]; root_groups=[";
      llvm::interleaveComma(violatingRootGroups, diagnostic);
      diagnostic << "]; operations=[";
      llvm::interleaveComma(violatingOperations, diagnostic);
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
          *entry, tileId, tileRelations, preparation.operationRootGroups,
          failureReason))) {
    if (failureReason) {
      llvm::raw_string_ostream diagnostic(*failureReason);
      diagnostic << "; incident_edge_actions=[";
      bool first = true;
      for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
        if (!isSpatialEdgeStrategyIncidentOnTile(strategy, tileId))
          continue;
        if (!first)
          diagnostic << ';';
        first = false;
        diagnostic << stringifyEdgeAction(strategy.action) << ':'
                   << strategy.producer->getName() << "->"
                   << strategy.consumer->getName()
                   << ":destination=" << strategy.destinationTile.getValue()
                   << ":fragments=" << strategy.fragments.size();
      }
      diagnostic << ']';
    }
    return mlir::failure();
  }
  return entry;
}

} // namespace wafer::tensor_program_to_card_module
