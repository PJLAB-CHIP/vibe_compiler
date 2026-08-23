//===- SingleRootTileRegion.cpp - Selected shard Card assembly --------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"

#include "Internal.h"
#include "SingleRootTileRegionInternal.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

using compiler::detail::ReductionGroupId;
using compiler::detail::ReductionGroupPlacement;

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

struct TileEntryStage {
  mlir::func::FuncOp function;
  llvm::SmallVector<RootValueKey, 8> boundaries;
  llvm::SmallVector<RootValueKey, 4> results;
};

mlir::Value createEmptyTensor(mlir::Type type, mlir::Location loc,
                              mlir::OpBuilder &builder) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return {};
  return builder
      .create<mlir::tensor::EmptyOp>(loc, tensor.getShape(),
                                     tensor.getElementType())
      .getResult();
}

mlir::Value createDDRTensorDestination(mlir::Type type, mlir::Location loc,
                                       mlir::OpBuilder &builder) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return {};
  auto bufferType = mlir::MemRefType::get(
      tensor.getShape(), tensor.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(builder.getContext(), MemorySpace::DDR,
                      MemLayout::Tensor));
  auto buffer = builder.create<mlir::memref::AllocOp>(loc, bufferType);
  return builder
      .create<mlir::bufferization::ToTensorOp>(loc, buffer.getResult(),
                                               /*restrict=*/true,
                                               /*writable=*/true)
      .getResult();
}

mlir::LogicalResult composeTileEntry(
    TileModuleOp tile, mlir::func::FuncOp sourceProgram,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    llvm::ArrayRef<TileEntryStage> stages, std::string *failureReason) {
  if (!tile || !sourceProgram || sourceProgram.isExternal() ||
      !sourceProgram.getBody().hasOneBlock())
    return failResult(failureReason,
                      "Tile entry composition requires one source function");
  mlir::OpBuilder builder(&tile.getBody().front(),
                          tile.getBody().front().begin());
  auto entry = builder.create<mlir::func::FuncOp>(
      sourceProgram.getLoc(), sourceProgram.getSymName(),
      sourceProgram.getFunctionType());
  mlir::Block *body = entry.addEntryBlock();
  builder.setInsertionPointToStart(body);
  std::map<RootValueKey, mlir::Value> values;
  std::map<RootValueKey, llvm::SmallVector<mlir::Value, 2>> destinations;
  for (auto [index, argument] : llvm::enumerate(body->getArguments()))
    values[{RootValueKind::SourceArgument, static_cast<uint32_t>(index), 0}] =
        argument;

  llvm::SmallVector<const TileEntryStage *, 8> pending;
  for (const TileEntryStage &stage : stages)
    pending.push_back(&stage);
  while (!pending.empty()) {
    auto ready = llvm::find_if(pending, [&](const TileEntryStage *stage) {
      return llvm::all_of(stage->boundaries, [&](const RootValueKey &key) {
        return key.kind == RootValueKind::SourceArgument ||
               values.count(key) != 0;
      });
    });
    if (ready == pending.end())
      ready = pending.begin();
    const TileEntryStage &stage = **ready;
    mlir::func::FuncOp function = stage.function;
    pending.erase(ready);
    if (!function || function.getNumResults() != stage.results.size() ||
        function.getNumArguments() < stage.boundaries.size())
      return failResult(failureReason,
                        "Tile entry stage relation is structurally invalid");
    llvm::SmallVector<mlir::Value, 8> operands;
    for (auto [index, key] : llvm::enumerate(stage.boundaries)) {
      auto value = values.find(key);
      mlir::Value operand =
          value == values.end()
              ? (key.kind == RootValueKind::StructuredResult
                     ? createDDRTensorDestination(
                           function.getArgumentTypes()[index],
                           function.getLoc(), builder)
                     : createEmptyTensor(function.getArgumentTypes()[index],
                                         function.getLoc(), builder))
              : value->second;
      if (!operand || operand.getType() != function.getArgumentTypes()[index])
        return failResult(failureReason,
                          "Tile entry cannot resolve one stage boundary");
      operands.push_back(operand);
    }
    for (unsigned index = stage.boundaries.size();
         index < function.getNumArguments(); ++index) {
      // A stage result crosses this private fragment boundary. The first
      // movement assignment is an explicit DDR cut, so its full logical
      // destination must not become a full-size Tile-local SPM allocation.
      // Actual wave/shard buffers remain inside the moved TileRegion body.
      mlir::Value destination = createDDRTensorDestination(
          function.getArgumentTypes()[index], function.getLoc(), builder);
      if (!destination)
        return failResult(failureReason,
                          "Tile entry stage destination is not static tensor");
      operands.push_back(destination);
    }
    for (const RootValueKey &key : stage.results)
      for (size_t operandIndex = stage.boundaries.size();
           operandIndex < operands.size(); ++operandIndex)
        destinations[key].push_back(operands[operandIndex]);
    if (!function.getBody().hasOneBlock())
      return failResult(failureReason,
                        "Tile entry stage function is not one block");
    mlir::Block &stageBody = function.getBody().front();
    if (stageBody.getNumArguments() != operands.size())
      return failResult(failureReason,
                        "Tile entry stage operand count changed");
    for (auto [argument, operand] :
         llvm::zip_equal(stageBody.getArguments(), operands))
      argument.replaceAllUsesWith(operand);
    auto stageReturn =
        mlir::dyn_cast<mlir::func::ReturnOp>(stageBody.getTerminator());
    if (!stageReturn || stageReturn.getNumOperands() != stage.results.size())
      return failResult(failureReason,
                        "Tile entry stage has no exact functional return");
    llvm::SmallVector<mlir::Value, 4> stageResults(
        stageReturn.getOperands().begin(), stageReturn.getOperands().end());
    llvm::SmallVector<mlir::Operation *, 16> stageOperations;
    for (mlir::Operation &operation : stageBody.without_terminator())
      stageOperations.push_back(&operation);
    for (mlir::Operation *operation : stageOperations)
      operation->moveBefore(body, body->end());
    stageReturn.erase();
    function.erase();
    builder.setInsertionPointToEnd(body);
    for (auto [key, value] : llvm::zip_equal(stage.results, stageResults))
      values[key] = value;
  }

  auto sourceReturn = mlir::cast<mlir::func::ReturnOp>(
      sourceProgram.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value, 4> outputs;
  for (auto [resultIndex, sourceValue, resultType] : llvm::enumerate(
           sourceReturn.getOperands(), sourceProgram.getResultTypes())) {
    mlir::Value output;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(sourceValue)) {
      auto owner =
          llvm::find_if(sourceOperationNodes,
                        [&](const StructuredOperationNodeMapping &candidate) {
                          return candidate.operation == result.getOwner();
                        });
      if (owner != sourceOperationNodes.end()) {
        auto found =
            values.find({RootValueKind::StructuredResult,
                         owner->structuredNodeId, result.getResultNumber()});
        if (found != values.end())
          output = found->second;
        if ((!output || output.getType() != resultType)) {
          auto destination = destinations.find({RootValueKind::StructuredResult,
                                                owner->structuredNodeId,
                                                result.getResultNumber()});
          if (destination != destinations.end())
            for (mlir::Value candidate : destination->second)
              if (candidate.getType() == resultType) {
                output = candidate;
                break;
              }
        }
      }
    }
    if (!output || output.getType() != resultType)
      output = createDDRTensorDestination(resultType, sourceProgram.getLoc(),
                                          builder);
    if (!output || output.getType() != resultType)
      return failResult(failureReason,
                        "Tile entry cannot construct one program output");
    outputs.push_back(output);
    (void)resultIndex;
  }
  builder.create<mlir::func::ReturnOp>(sourceProgram.getLoc(), outputs);
  return mlir::verify(entry);
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
  destination.scratchBuffers.append(source.scratchBuffers.begin(),
                                    source.scratchBuffers.end());
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

mlir::FailureOr<wafer::PreparedRootWorkLeaf>
wafer::prepareRootWorkLeaf(uint32_t structuredNodeId,
                           const analysis::RootRegionWork &work,
                           std::string *failureReason) {
  using namespace tensor_program_to_tile_region;
  auto failLeaf =
      [&](llvm::StringRef detail) -> mlir::FailureOr<PreparedRootWorkLeaf> {
    setFailureReason(failureReason, detail);
    return mlir::failure();
  };
  if (!work.rootOperation)
    return failLeaf("structured root leaf has no current root operation");
  PreparedRootWorkLeaf prepared;
  for (const analysis::ReductionMergeRequirement &merge : work.merges) {
    if (merge.group.root != work.id.root || merge.mergeTile != work.id.tile)
      return failLeaf("structured root merge does not match its leaf");
    prepared.merges.push_back({merge.group, merge.mergeTile});
  }
  llvm::sort(prepared.merges,
             [](const compiler::detail::ReductionGroupPlacement &lhs,
                const compiler::detail::ReductionGroupPlacement &rhs) {
               return lhs.group < rhs.group;
             });
  if (std::adjacent_find(
          prepared.merges.begin(), prepared.merges.end(),
          [](const compiler::detail::ReductionGroupPlacement &lhs,
             const compiler::detail::ReductionGroupPlacement &rhs) {
            return lhs.group == rhs.group;
          }) != prepared.merges.end())
    return failLeaf("structured root leaf has duplicate merge groups");
  if (work.execution.empty()) {
    if (work.contributions.empty() && !prepared.merges.empty())
      return prepared;
    return failLeaf("structured root leaf has no execution piece");
  }
  if (work.execution.size() != 1)
    return failLeaf(
        "structured root leaf requires one execution piece per Tile");
  const analysis::RootExecutionWork &execution = work.execution.front();
  if (execution.shard.root != work.id.root)
    return failLeaf("structured root leaf execution belongs to another root");

  StructuredNodeIterationShard result;
  result.structuredNodeId = structuredNodeId;
  result.tile = work.id.tile;
  for (const compiler::detail::IteratorInterval &interval :
       execution.iterationDomain) {
    if (interval.size <= 0)
      return failLeaf("structured root leaf has an empty iterator interval");
    result.offsets.push_back(interval.offset);
    result.sizes.push_back(interval.size);
  }
  for (const analysis::RootContributionWork &contribution :
       work.contributions) {
    if (contribution.contribution.shard != execution.shard ||
        contribution.contribution.tile != work.id.tile)
      return failLeaf("structured root contribution does not match its leaf");
    compiler::detail::ReductionGroupPlacement placement{contribution.group,
                                                        contribution.mergeTile};
    if (!llvm::any_of(result.reductionGroups, [&](const auto &existing) {
          return existing.group == placement.group &&
                 existing.mergeTile == placement.mergeTile;
        }))
      result.reductionGroups.push_back(std::move(placement));
  }
  llvm::sort(result.reductionGroups,
             [](const compiler::detail::ReductionGroupPlacement &lhs,
                const compiler::detail::ReductionGroupPlacement &rhs) {
               return lhs.group < rhs.group;
             });
  prepared.execution = std::move(result);
  return prepared;
}

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
  mlir::func::FuncOp sourceProgram =
      operationNodes.front().operation
          ? operationNodes.front()
                .operation->getParentOfType<mlir::func::FuncOp>()
          : mlir::func::FuncOp{};
  for (const StructuredOperationNodeMapping &mapping : operationNodes)
    if (!mapping.operation ||
        !sourceModule->isProperAncestor(mapping.operation) ||
        mapping.operation->getParentOfType<mlir::func::FuncOp>() !=
            sourceProgram ||
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
  std::map<uint32_t, bool> nodePartialStates;
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
      const bool partial = !shard.reductionGroups.empty();
      auto [state, inserted] =
          nodePartialStates.try_emplace(shard.structuredNodeId, partial);
      if (!inserted && state->second != partial)
        return failResult(
            failureReason,
            "one structured node mixes complete and partial shards");
      if (partial) {
        if (group.shards.size() != 1 || shard.reductionGroups.size() != 1 ||
            !llvm::is_contained(sortedTiles,
                                shard.reductionGroups.front().mergeTile))
          return failResult(
              failureReason,
              "current partial contribution apply requires one output group");
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
  std::map<int64_t, llvm::SmallVector<TileEntryStage, 8>> entryStages;
  struct PartialGroup {
    uint32_t node = 0;
    ReductionGroupId group;
    TileId mergeTile{0};
    llvm::SmallVector<const StructuredNodeIterationShard *, 8> shards;
    llvm::SmallVector<mlir::func::FuncOp, 8> functions;
    const StructuredNodePhysicalRepresentation *mergeRepresentation = nullptr;
  };
  std::map<ReductionGroupId, PartialGroup> partialGroups;
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
    entryStages[firstShard.tile.getValue()].push_back(
        {fragment->function, fragment->boundaries, fragment->results});
    if (!firstShard.reductionGroups.empty()) {
      const ReductionGroupPlacement &placement =
          firstShard.reductionGroups.front();
      PartialGroup &partial = partialGroups[placement.group];
      partial.node = firstShard.structuredNodeId;
      partial.group = placement.group;
      partial.mergeTile = placement.mergeTile;
      partial.shards.push_back(&firstShard);
      partial.functions.push_back(fragment->function);
      if (!group->representations.empty() &&
          firstShard.tile == placement.mergeTile)
        partial.mergeRepresentation = &group->representations.front();
    }
    appendRelations(relations, std::move(fragment->relations));
  }

  for (auto &[groupId, group] : partialGroups) {
    (void)groupId;
    if (group.shards.size() < 2)
      return failResult(
          failureReason,
          "partial reduction requires at least two contribution shards");
    TileModuleOp tile = tiles.lookup(group.mergeTile.getValue());
    if (!tile)
      return failResult(failureReason,
                        "partial merge lost its selected Tile owner");
    mlir::FailureOr<RootFragment> merge = materializeReductionMergeFragment(
        tile, operationNodes, group.node, group.group, group.shards,
        group.functions, group.mergeRepresentation, failureReason);
    if (mlir::failed(merge))
      return mlir::failure();
    entryStages[group.mergeTile.getValue()].push_back(
        {merge->function, merge->boundaries, merge->results});
    appendRelations(relations, std::move(merge->relations));
  }

  for (TileId tileId : sortedTiles) {
    TileModuleOp tile = tiles.lookup(tileId.getValue());
    if (!tile || mlir::failed(composeTileEntry(
                     tile, sourceProgram, operationNodes,
                     entryStages[tileId.getValue()], failureReason)))
      return mlir::failure();
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
