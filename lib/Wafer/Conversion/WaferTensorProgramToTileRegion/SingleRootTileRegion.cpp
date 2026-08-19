//===- SingleRootTileRegion.cpp - Selected shard Card assembly --------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"

#include "Internal.h"
#include "SingleRootTileRegionInternal.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

bool isDirectSymbolDeclaration(mlir::Operation &operation) {
  auto symbol = mlir::dyn_cast<mlir::SymbolOpInterface>(&operation);
  return symbol &&
         llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

void cloneModuleTopology(mlir::ModuleOp source, mlir::ModuleOp destination) {
  mlir::OpBuilder builder(destination.getBodyRegion());
  mlir::IRMapping mapping;
  for (mlir::Operation &operation : source.getBody()->without_terminator())
    if (mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      builder.clone(operation, mapping);
}

void appendRelations(StructuredMaterializationRelations &destination,
                     StructuredMaterializationRelations source) {
  destination.operationEmissions.append(source.operationEmissions.begin(),
                                        source.operationEmissions.end());
  destination.operationResultBuffers.append(
      source.operationResultBuffers.begin(),
      source.operationResultBuffers.end());
  destination.operandBuffers.append(source.operandBuffers.begin(),
                                    source.operandBuffers.end());
  destination.outputBuffers.append(source.outputBuffers.begin(),
                                   source.outputBuffers.end());
}

} // namespace
} // namespace wafer::tensor_program_to_tile_region

mlir::LogicalResult wafer::lowerStructuredNodeShardsToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeIterationShard> shards,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason) {
  using namespace tensor_program_to_tile_region;
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || availableTiles.empty() || operationNodes.empty())
    return failResult(
        failureReason,
        "single-root CardModule requires source, Tiles and structured roots");
  llvm::DenseSet<mlir::Operation *> seenOperations;
  llvm::DenseSet<uint32_t> seenNodeIds;
  for (const StructuredOperationNodeMapping &mapping : operationNodes)
    if (!mapping.operation ||
        !sourceModule->isProperAncestor(mapping.operation) ||
        !seenOperations.insert(mapping.operation).second ||
        !seenNodeIds.insert(mapping.structuredNodeId).second)
      return failResult(
          failureReason,
          "structured root mapping is null, duplicated, or outside source");

  llvm::SmallVector<TileId, 16> sortedTiles(availableTiles.begin(),
                                            availableTiles.end());
  llvm::sort(sortedTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (std::adjacent_find(sortedTiles.begin(), sortedTiles.end()) !=
      sortedTiles.end())
    return failResult(failureReason,
                      "single-root CardModule has duplicate available Tiles");

  std::set<std::pair<int64_t, uint32_t>> seenShards;
  std::map<uint32_t, StructuredNodeIterationShardRole> nodeRoles;
  std::map<uint32_t, TileId> reductionMergeTiles;
  llvm::SmallVector<const StructuredNodeIterationShard *, 32> orderedShards;
  for (const StructuredNodeIterationShard &shard : shards) {
    if (!llvm::is_contained(sortedTiles, shard.tile) ||
        !seenShards.emplace(shard.tile.getValue(), shard.structuredNodeId)
             .second)
      return failResult(
          failureReason,
          "node shards contain an unavailable Tile or duplicate node/Tile");
    auto [role, inserted] =
        nodeRoles.try_emplace(shard.structuredNodeId, shard.role);
    if (!inserted && role->second != shard.role)
      return failResult(
          failureReason,
          "one structured node mixes complete and partial shards");
    if (shard.role ==
        StructuredNodeIterationShardRole::PartialReductionContribution) {
      if (!shard.reductionMergeTile ||
          !llvm::is_contained(sortedTiles, *shard.reductionMergeTile))
        return failResult(
            failureReason,
            "partial-reduction shard has no available merge Tile");
      auto [merge, mergeInserted] = reductionMergeTiles.try_emplace(
          shard.structuredNodeId, *shard.reductionMergeTile);
      if (!mergeInserted && merge->second != *shard.reductionMergeTile)
        return failResult(failureReason,
                          "one partial reduction selects several merge Tiles");
    } else if (shard.reductionMergeTile) {
      return failResult(failureReason,
                        "complete node shard unexpectedly has a merge Tile");
    }
    orderedShards.push_back(&shard);
  }
  llvm::sort(orderedShards, [](const auto *lhs, const auto *rhs) {
    return std::tuple(lhs->tile.getValue(), lhs->structuredNodeId) <
           std::tuple(rhs->tile.getValue(), rhs->structuredNodeId);
  });

  mlir::OwningOpRef<mlir::ModuleOp> result =
      mlir::ModuleOp::create(sourceModule.getLoc());
  result->getOperation()->setAttrs(sourceModule->getAttrDictionary());
  cloneModuleTopology(sourceModule, *result);
  mlir::OpBuilder moduleBuilder(result->getBodyRegion());
  auto card = moduleBuilder.create<CardModuleOp>(
      sourceModule.getLoc(),
      moduleBuilder.getI64IntegerAttr(cardId.getValue()));
  card.getBody().push_back(new mlir::Block());
  mlir::Block &cardBody = card.getBody().front();
  mlir::OpBuilder cardBuilder(&cardBody, cardBody.end());
  mlir::IRMapping declarationMapping;
  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator())
    if (isDirectSymbolDeclaration(operation) &&
        !mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      cardBuilder.clone(operation, declarationMapping);

  llvm::DenseMap<int64_t, TileModuleOp> tiles;
  for (TileId tileId : sortedTiles) {
    auto tile = cardBuilder.create<TileModuleOp>(
        sourceModule.getLoc(),
        cardBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    tiles.try_emplace(tileId.getValue(), tile);
  }

  StructuredMaterializationRelations relations;
  struct PartialGroup {
    TileId mergeTile{0};
    llvm::SmallVector<const StructuredNodeIterationShard *, 8> shards;
    llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  };
  std::map<uint32_t, PartialGroup> partialGroups;
  for (const StructuredNodeIterationShard *shard : orderedShards) {
    TileModuleOp tile = tiles.lookup(shard->tile.getValue());
    if (!tile)
      return failResult(failureReason,
                        "single-root fragment lost its Tile owner");
    mlir::FailureOr<RootFragment> fragment =
        materializeRootFragment(tile, operationNodes, *shard, failureReason);
    if (mlir::failed(fragment))
      return mlir::failure();
    if (shard->role ==
        StructuredNodeIterationShardRole::PartialReductionContribution) {
      PartialGroup &group = partialGroups[shard->structuredNodeId];
      group.mergeTile = *shard->reductionMergeTile;
      group.shards.push_back(shard);
      group.functions.push_back(fragment->function);
    }
    appendRelations(relations, std::move(fragment->relations));
  }

  for (auto &[node, group] : partialGroups) {
    if (group.shards.size() < 2)
      return failResult(
          failureReason,
          "partial reduction requires at least two contribution shards");
    TileModuleOp tile = tiles.lookup(group.mergeTile.getValue());
    if (!tile)
      return failResult(failureReason,
                        "partial merge lost its selected Tile owner");
    mlir::FailureOr<RootFragment> merge = materializeReductionMergeFragment(
        tile, operationNodes, node, group.shards, group.functions,
        failureReason);
    if (mlir::failed(merge))
      return mlir::failure();
    appendRelations(relations, std::move(merge->relations));
  }

  if (mlir::failed(mlir::verify(*result)))
    return failResult(
        failureReason,
        "single-root CardModule is not verifier-legal after assembly");
  if (materializationRelations)
    *materializationRelations = std::move(relations);
  cardModule = std::move(result);
  return mlir::success();
}
