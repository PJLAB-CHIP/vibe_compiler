//===- SingleRootTileRegion.cpp - Selected shard Card assembly --------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"

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
  destination.partialReductionContributions.append(
      source.partialReductionContributions.begin(),
      source.partialReductionContributions.end());
  destination.partialReductionMergeInputs.append(
      source.partialReductionMergeInputs.begin(),
      source.partialReductionMergeInputs.end());
}

} // namespace
} // namespace wafer::tensor_program_to_tile_region

mlir::LogicalResult wafer::lowerStructuredNodeGroupsToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeShardGroup> groups,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason) {
  using namespace tensor_program_to_tile_region;
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || availableTiles.empty() || operationNodes.empty())
    return failResult(
        failureReason,
        "node-group CardModule requires source, Tiles and structured roots");
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
                      "node-group CardModule has duplicate available Tiles");

  std::set<std::pair<int64_t, uint32_t>> seenShards;
  std::map<uint32_t, StructuredNodeIterationShardRole> nodeRoles;
  std::map<uint32_t, TileId> reductionMergeTiles;
  std::map<uint32_t, StructuredComputeImplementation> nodeImplementations;
  std::map<uint32_t, std::pair<llvm::SmallVector<int64_t, 4>,
                               llvm::SmallVector<uint32_t, 4>>>
      nodeTemporalTiles;
  llvm::SmallVector<const StructuredNodeShardGroup *, 32> orderedGroups;
  for (const StructuredNodeShardGroup &group : groups) {
    if (group.shards.empty())
      return failResult(failureReason, "node-shard group is empty");
    if (!group.temporalTiles.empty() &&
        group.temporalTiles.size() != group.shards.size())
      return failResult(
          failureReason,
          "selected temporal group does not cover every node shard");
    if (!group.representations.empty() &&
        group.representations.size() != group.shards.size())
      return failResult(
          failureReason,
          "selected representation group does not cover every node shard");
    if (!group.implementations.empty() &&
        group.implementations.size() != group.shards.size())
      return failResult(
          failureReason,
          "selected implementation group does not cover every node shard");
    const TileId groupTile = group.shards.front().tile;
    uint32_t previousNode = 0;
    bool firstNode = true;
    for (auto [index, shard] : llvm::enumerate(group.shards)) {
      if (shard.tile != groupTile ||
          !llvm::is_contained(sortedTiles, shard.tile) ||
          (!firstNode && shard.structuredNodeId <= previousNode) ||
          !seenShards.emplace(shard.tile.getValue(), shard.structuredNodeId)
               .second)
        return failResult(
            failureReason,
            "node group has mixed Tiles, unstable order, or duplicate shard");
      firstNode = false;
      previousNode = shard.structuredNodeId;
      auto [role, inserted] =
          nodeRoles.try_emplace(shard.structuredNodeId, shard.role);
      if (!inserted && role->second != shard.role)
        return failResult(
            failureReason,
            "one structured node mixes complete and partial shards");
      if (shard.role ==
          StructuredNodeIterationShardRole::PartialReductionContribution) {
        if (group.shards.size() != 1 || !shard.reductionMergeTile ||
            !llvm::is_contained(sortedTiles, *shard.reductionMergeTile))
          return failResult(
              failureReason,
              "partial-reduction contribution must be one singleton group");
        auto [merge, mergeInserted] = reductionMergeTiles.try_emplace(
            shard.structuredNodeId, *shard.reductionMergeTile);
        if (!mergeInserted && merge->second != *shard.reductionMergeTile)
          return failResult(
              failureReason,
              "one partial reduction selects several merge Tiles");
      } else if (shard.reductionMergeTile) {
        return failResult(failureReason,
                          "complete node shard unexpectedly has a merge Tile");
      }
      if (!group.temporalTiles.empty()) {
        const StructuredNodeTemporalTile &temporal = group.temporalTiles[index];
        llvm::SmallVector<uint32_t, 4> sortedOrder = temporal.waveLoopOrder;
        llvm::sort(sortedOrder);
        if (temporal.structuredNodeId != shard.structuredNodeId ||
            temporal.iteratorTileSizes.size() != shard.sizes.size() ||
            llvm::any_of(temporal.iteratorTileSizes,
                         [](int64_t size) { return size <= 0; }) ||
            std::adjacent_find(sortedOrder.begin(), sortedOrder.end()) !=
                sortedOrder.end() ||
            llvm::any_of(sortedOrder, [&](uint32_t dimension) {
              return dimension >= temporal.iteratorTileSizes.size();
            }))
          return failResult(failureReason,
                            "selected temporal node assignment is malformed");
        auto [stored, temporalInserted] = nodeTemporalTiles.try_emplace(
            shard.structuredNodeId,
            std::make_pair(temporal.iteratorTileSizes, temporal.waveLoopOrder));
        if (!temporalInserted &&
            (stored->second.first != temporal.iteratorTileSizes ||
             stored->second.second != temporal.waveLoopOrder))
          return failResult(
              failureReason,
              "one structured node has inconsistent temporal assignments");
      }
      if (!group.representations.empty() &&
          group.representations[index].structuredNodeId !=
              shard.structuredNodeId)
        return failResult(
            failureReason,
            "selected representation node identity is inconsistent");
      if (!group.implementations.empty()) {
        const StructuredNodeComputeImplementation &implementation =
            group.implementations[index];
        if (implementation.structuredNodeId != shard.structuredNodeId)
          return failResult(
              failureReason,
              "selected implementation node identity is inconsistent");
        auto [stored, inserted] = nodeImplementations.try_emplace(
            shard.structuredNodeId, implementation.implementation);
        if (!inserted && stored->second != implementation.implementation)
          return failResult(
              failureReason,
              "one structured node has inconsistent implementations");
      }
    }
    orderedGroups.push_back(&group);
  }
  llvm::sort(orderedGroups, [](const auto *lhs, const auto *rhs) {
    return std::tuple(lhs->shards.front().tile.getValue(),
                      lhs->shards.front().structuredNodeId) <
           std::tuple(rhs->shards.front().tile.getValue(),
                      rhs->shards.front().structuredNodeId);
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
    const StructuredNodePhysicalRepresentation *mergeRepresentation = nullptr;
  };
  std::map<uint32_t, PartialGroup> partialGroups;
  for (const StructuredNodeShardGroup *group : orderedGroups) {
    const StructuredNodeIterationShard &firstShard = group->shards.front();
    TileModuleOp tile = tiles.lookup(firstShard.tile.getValue());
    if (!tile)
      return failResult(failureReason,
                        "node-group fragment lost its Tile owner");
    mlir::FailureOr<RootFragment> fragment =
        group->shards.size() == 1 && group->recomputedProducerNodes.empty()
            ? materializeRootFragment(tile, operationNodes, firstShard,
                                      group->temporalTiles.empty()
                                          ? nullptr
                                          : &group->temporalTiles.front(),
                                      group->representations.empty()
                                          ? nullptr
                                          : &group->representations.front(),
                                      group->implementations.empty()
                                          ? nullptr
                                          : &group->implementations.front(),
                                      failureReason)
            : materializeCoupledRootFragment(tile, operationNodes, *group,
                                             failureReason);
    if (mlir::failed(fragment))
      return mlir::failure();
    if (firstShard.role ==
        StructuredNodeIterationShardRole::PartialReductionContribution) {
      PartialGroup &partial = partialGroups[firstShard.structuredNodeId];
      partial.mergeTile = *firstShard.reductionMergeTile;
      partial.shards.push_back(&firstShard);
      partial.functions.push_back(fragment->function);
      if (!group->representations.empty() &&
          firstShard.tile == *firstShard.reductionMergeTile)
        partial.mergeRepresentation = &group->representations.front();
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
        group.mergeRepresentation, failureReason);
    if (mlir::failed(merge))
      return mlir::failure();
    appendRelations(relations, std::move(merge->relations));
  }

  if (mlir::failed(mlir::verify(*result)))
    return failResult(failureReason,
                      "node-group CardModule is not verifier-legal");
  if (materializationRelations)
    *materializationRelations = std::move(relations);
  cardModule = std::move(result);
  return mlir::success();
}

mlir::LogicalResult wafer::lowerStructuredNodeShardsToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeIterationShard> shards,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason) {
  llvm::SmallVector<StructuredNodeShardGroup, 32> groups;
  groups.reserve(shards.size());
  for (const StructuredNodeIterationShard &shard : shards)
    groups.push_back(StructuredNodeShardGroup{{shard}});
  return lowerStructuredNodeGroupsToCardModule(
      sourceModule, cardId, availableTiles, operationNodes, groups, cardModule,
      materializationRelations, failureReason);
}
