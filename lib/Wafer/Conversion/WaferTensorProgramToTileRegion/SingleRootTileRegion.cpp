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
#include <vector>

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
  llvm::SmallVector<StructuredNodeExternalSource, 4> externalSources;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 2>, 4> outputBuffers;
};

mlir::FailureOr<TileEntryStage>
makeTileEntryStage(const RootFragment &fragment,
                   llvm::ArrayRef<StructuredNodeExternalSource> externalSources,
                   std::string *failureReason) {
  TileEntryStage stage{fragment.function,
                       fragment.boundaries,
                       fragment.results,
                       llvm::SmallVector<StructuredNodeExternalSource, 4>(
                           externalSources.begin(), externalSources.end()),
                       {}};
  stage.outputBuffers.resize(stage.results.size());
  for (const SpatialOutputBufferRelation &relation :
       fragment.relations.outputBuffers) {
    if (relation.outputIndex >= stage.outputBuffers.size()) {
      setFailureReason(failureReason,
                       "root fragment has an out-of-range output relation");
      return mlir::failure();
    }
    auto &buffers = stage.outputBuffers[relation.outputIndex];
    if (!llvm::is_contained(buffers, relation.buffer))
      buffers.push_back(relation.buffer);
  }
  return stage;
}

struct CardDDRStageResource {
  RootValueKey value;
  int64_t resourceId = -1;
  mlir::RankedTensorType tensorType;
  llvm::SmallVector<TileId, 4> producers;
  llvm::SmallVector<TileId, 4> consumers;
};

const CardDDRStageResource *
findCardDDRStageResource(llvm::ArrayRef<CardDDRStageResource> resources,
                         const RootValueKey &value, TileId tile,
                         bool producer) {
  auto resource = llvm::find_if(resources, [&](const auto &candidate) {
    return !(candidate.value < value) && !(value < candidate.value) &&
           (producer ? llvm::is_contained(candidate.producers, tile)
                     : llvm::is_contained(candidate.consumers, tile));
  });
  return resource == resources.end() ? nullptr : &*resource;
}

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
    TileModuleOp tile, TileId tileId, mlir::func::FuncOp sourceProgram,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    llvm::ArrayRef<TileEntryStage> stages,
    llvm::ArrayRef<CardDDRStageResource> cardDDRResources,
    StructuredMaterializationRelations &relations, std::string *failureReason) {
  if (!tile || !sourceProgram || sourceProgram.isExternal() ||
      !sourceProgram.getBody().hasOneBlock())
    return failResult(failureReason,
                      "Tile entry composition requires one source function");
  mlir::OpBuilder builder(&tile.getBody().front(),
                          tile.getBody().front().begin());
  llvm::SmallVector<mlir::Type, 16> entryInputTypes(
      sourceProgram.getArgumentTypes().begin(),
      sourceProgram.getArgumentTypes().end());
  for (const CardDDRStageResource &resource : cardDDRResources)
    entryInputTypes.push_back(resource.tensorType);
  auto entry = builder.create<mlir::func::FuncOp>(
      sourceProgram.getLoc(), sourceProgram.getSymName(),
      builder.getFunctionType(entryInputTypes, sourceProgram.getResultTypes()));
  mlir::Block *body = entry.addEntryBlock();
  builder.setInsertionPointToStart(body);
  std::map<RootValueKey, mlir::Value> values;
  std::map<RootValueKey, llvm::SmallVector<mlir::Value, 2>> destinations;
  for (auto [index, argument] : llvm::enumerate(
           body->getArguments().take_front(sourceProgram.getNumArguments())))
    values[{RootValueKind::SourceArgument, static_cast<uint32_t>(index), 0}] =
        argument;
  std::map<int64_t, mlir::Value> currentCardDDRValues;
  for (auto [resourceIndex, resource] : llvm::enumerate(cardDDRResources)) {
    const unsigned argumentIndex =
        sourceProgram.getNumArguments() + resourceIndex;
    mlir::BlockArgument argument = body->getArgument(argumentIndex);
    CardDDRAccess access = CardDDRAccess::None;
    const bool writes = llvm::is_contained(resource.producers, tileId);
    const bool reads = llvm::is_contained(resource.consumers, tileId);
    if (writes && reads)
      access = CardDDRAccess::ReadWrite;
    else if (writes)
      access = CardDDRAccess::Write;
    else if (reads)
      access = CardDDRAccess::Read;
    std::string symbol =
        (llvm::Twine("card_ddr_") + llvm::Twine(resource.resourceId)).str();
    entry.setArgAttr(argumentIndex, kWaferCardDDRBindingAttrName,
                     CardDDRBindingAttr::get(entry.getContext(),
                                             mlir::FlatSymbolRefAttr::get(
                                                 entry.getContext(), symbol),
                                             resource.resourceId, access));
    currentCardDDRValues.emplace(resource.resourceId, argument);
    relations.cardDDRBuffers.push_back({resource.resourceId, argument});
  }

  llvm::SmallVector<const TileEntryStage *, 8> pending;
  for (const TileEntryStage &stage : stages)
    pending.push_back(&stage);
  while (!pending.empty()) {
    auto ready = llvm::find_if(pending, [&](const TileEntryStage *stage) {
      return llvm::all_of(stage->boundaries, [&](const RootValueKey &key) {
        bool hasSelectedOwner = false;
        bool allSelectedOwnersAreLocal = true;
        for (const StructuredNodeExternalSource &source :
             stage->externalSources)
          if (source.producerNodeId == key.owner &&
              source.producerResult == key.resultIndex) {
            hasSelectedOwner = true;
            allSelectedOwnersAreLocal &= source.producerTile == tileId;
          }
        const bool requiresCardDDR =
            hasSelectedOwner && !allSelectedOwnersAreLocal;
        const CardDDRStageResource *resource = findCardDDRStageResource(
            cardDDRResources, key, tileId, /*producer=*/false);
        return key.kind == RootValueKind::SourceArgument ||
               (!requiresCardDDR && values.count(key) != 0) || resource;
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
      bool hasSelectedOwner = false;
      bool allSelectedOwnersAreLocal = true;
      for (const StructuredNodeExternalSource &source : stage.externalSources)
        if (source.producerNodeId == key.owner &&
            source.producerResult == key.resultIndex) {
          hasSelectedOwner = true;
          allSelectedOwnersAreLocal &= source.producerTile == tileId;
        }
      const bool requiresCardDDR =
          hasSelectedOwner && !allSelectedOwnersAreLocal;
      auto value = values.find(key);
      mlir::Value operand = requiresCardDDR || value == values.end()
                                ? mlir::Value{}
                                : value->second;
      if (!operand) {
        const CardDDRStageResource *resource =
            findCardDDRStageResource(cardDDRResources, key, tileId,
                                     /*producer=*/false);
        auto current = resource
                           ? currentCardDDRValues.find(resource->resourceId)
                           : currentCardDDRValues.end();
        if (resource && current != currentCardDDRValues.end() &&
            llvm::is_contained(resource->consumers, tileId))
          operand = current->second;
      }
      if (!operand && key.kind == RootValueKind::SourceArgument)
        operand = createEmptyTensor(function.getArgumentTypes()[index],
                                    function.getLoc(), builder);
      if (!operand || operand.getType() != function.getArgumentTypes()[index]) {
        if (failureReason) {
          llvm::raw_string_ostream diagnostic(*failureReason);
          diagnostic << "Tile entry cannot resolve one stage boundary; tile="
                     << tileId.getValue()
                     << ",function=" << function.getSymName()
                     << ",argument=" << index
                     << ",key-kind=" << static_cast<unsigned>(key.kind)
                     << ",key-owner=" << key.owner
                     << ",key-result=" << key.resultIndex
                     << ",expected=" << function.getArgumentTypes()[index]
                     << ",actual=";
          if (operand)
            diagnostic << operand.getType();
          else
            diagnostic << "missing";
          const CardDDRStageResource *resource =
              findCardDDRStageResource(cardDDRResources, key, tileId,
                                       /*producer=*/false);
          diagnostic << ",card-ddr-resource=";
          if (resource)
            diagnostic << resource->resourceId;
          else
            diagnostic << "none";
        }
        return mlir::failure();
      }
      operands.push_back(operand);
    }
    if (function.getNumArguments() !=
        stage.boundaries.size() + stage.results.size())
      return failResult(failureReason,
                        "Tile entry stage destination arity changed");
    for (auto [resultIndex, key] : llvm::enumerate(stage.results)) {
      const unsigned index = stage.boundaries.size() + resultIndex;
      // A stage result crosses this private fragment boundary. The first
      // movement assignment is an explicit DDR cut, so its full logical
      // destination must not become a full-size Tile-local SPM allocation.
      // Actual wave/shard buffers remain inside the moved TileRegion body.
      mlir::Value destination;
      const CardDDRStageResource *resource =
          findCardDDRStageResource(cardDDRResources, key, tileId,
                                   /*producer=*/true);
      auto current = resource ? currentCardDDRValues.find(resource->resourceId)
                              : currentCardDDRValues.end();
      if (resource && llvm::is_contained(resource->producers, tileId) &&
          current != currentCardDDRValues.end())
        destination = current->second;
      else
        destination = createDDRTensorDestination(
            function.getArgumentTypes()[index], function.getLoc(), builder);
      if (!destination)
        return failResult(failureReason,
                          "Tile entry stage destination is not static tensor");
      operands.push_back(destination);
      destinations[key].push_back(destination);
    }
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
    for (auto [key, value] : llvm::zip_equal(stage.results, stageResults)) {
      values[key] = value;
      const CardDDRStageResource *resource = findCardDDRStageResource(
          cardDDRResources, key, tileId, /*producer=*/true);
      if (resource && llvm::is_contained(resource->producers, tileId))
        currentCardDDRValues[resource->resourceId] = value;
    }
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
  destination.cardDDRBuffers.append(source.cardDDRBuffers.begin(),
                                    source.cardDDRBuffers.end());
  destination.cardDDRTransfers.append(source.cardDDRTransfers.begin(),
                                      source.cardDDRTransfers.end());
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
  llvm::SmallVector<const StructuredNodeShardGroup *, 32> orderedGroups;
  for (const StructuredNodeShardGroup &group : groups) {
    if (group.shards.empty())
      return failResult(failureReason, "node-shard group is empty");
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
    llvm::SmallVector<uint32_t, 4> independentlyMaterialized(
        group.independentlyMaterializedNodes.begin(),
        group.independentlyMaterializedNodes.end());
    llvm::sort(independentlyMaterialized);
    if (std::adjacent_find(independentlyMaterialized.begin(),
                           independentlyMaterialized.end()) !=
            independentlyMaterialized.end() ||
        llvm::any_of(independentlyMaterialized, [&](uint32_t node) {
          return llvm::none_of(group.shards,
                               [&](const auto &shard) {
                                 return shard.structuredNodeId == node;
                               }) &&
                 !llvm::is_contained(group.recomputedProducerNodes, node);
        }))
      return failResult(
          failureReason,
          "independent group materialization names a missing or duplicate "
          "scheduled node");
    llvm::SmallVector<uint32_t, 4> recomputedShardNodes;
    for (const StructuredNodeIterationShard &shard :
         group.recomputedProducerShards) {
      if (shard.tile != group.shards.front().tile ||
          !llvm::is_contained(group.recomputedProducerNodes,
                              shard.structuredNodeId) ||
          shard.offsets.size() != shard.sizes.size() ||
          llvm::any_of(shard.sizes, [](int64_t size) { return size <= 0; }))
        return failResult(
            failureReason,
            "recomputed producer shard is malformed or outside its group");
      recomputedShardNodes.push_back(shard.structuredNodeId);
    }
    llvm::sort(recomputedShardNodes);
    llvm::SmallVector<uint32_t, 4> recomputedNodes(
        group.recomputedProducerNodes.begin(),
        group.recomputedProducerNodes.end());
    llvm::sort(recomputedNodes);
    if (std::adjacent_find(recomputedShardNodes.begin(),
                           recomputedShardNodes.end()) !=
            recomputedShardNodes.end() ||
        llvm::any_of(recomputedShardNodes,
                     [&](uint32_t node) {
                       return !llvm::is_contained(recomputedNodes, node);
                     }) ||
        llvm::any_of(independentlyMaterialized, [&](uint32_t node) {
          return !llvm::any_of(group.shards, [&](const auto &shard) {
            return shard.structuredNodeId == node;
          }) && !llvm::is_contained(recomputedShardNodes, node);
        }))
      return failResult(
          failureReason,
          "recomputed producer work is duplicated, unknown, or missing for "
          "an independent replica");
    const TileId groupTile = group.shards.front().tile;
    std::map<uint32_t, const StructuredNodeTemporalTile *> temporalByNode;
    for (const StructuredNodeTemporalTile &temporal : group.temporalTiles) {
      llvm::SmallVector<uint32_t, 4> sortedOrder = temporal.waveLoopOrder;
      llvm::sort(sortedOrder);
      auto shard = llvm::find_if(group.shards, [&](const auto &candidate) {
        return candidate.structuredNodeId == temporal.structuredNodeId;
      });
      if (shard == group.shards.end() ||
          !temporalByNode.try_emplace(temporal.structuredNodeId, &temporal)
               .second ||
          temporal.iteratorTileSizes.size() != shard->sizes.size() ||
          llvm::any_of(temporal.iteratorTileSizes,
                       [](int64_t size) { return size <= 0; }) ||
          std::adjacent_find(sortedOrder.begin(), sortedOrder.end()) !=
              sortedOrder.end() ||
          llvm::any_of(sortedOrder, [&](uint32_t dimension) {
            return dimension >= temporal.iteratorTileSizes.size();
          }))
        return failResult(
            failureReason,
            "selected top-level temporal assignment is malformed");
    }
    std::set<std::tuple<uint32_t, uint32_t, unsigned, unsigned,
                        std::vector<int64_t>, std::vector<int64_t>>>
        nestedKeys;
    std::set<uint32_t> nestedProducerNodes;
    for (const StructuredNodeNestedTemporalTile &nested :
         group.nestedTemporalTiles) {
      auto producer = llvm::find_if(group.shards, [&](const auto &candidate) {
        return candidate.structuredNodeId == nested.producerNodeId;
      });
      auto parent = llvm::find_if(group.shards, [&](const auto &candidate) {
        return candidate.structuredNodeId == nested.parentNodeId;
      });
      llvm::SmallVector<uint32_t, 4> sortedOrder = nested.waveLoopOrder;
      llvm::sort(sortedOrder);
      if (producer == group.shards.end() || parent == group.shards.end() ||
          nested.producerNodeId == nested.parentNodeId ||
          nested.requestedResultExtents.empty() ||
          nested.producerIterationExtents.size() !=
              nested.iteratorTileSizes.size() ||
          llvm::any_of(nested.requestedResultExtents,
                       [](int64_t extent) { return extent <= 0; }) ||
          llvm::any_of(llvm::zip_equal(nested.producerIterationExtents,
                                       nested.iteratorTileSizes),
                       [](auto values) {
                         auto [extent, tile] = values;
                         return extent <= 0 || tile <= 0 || tile > extent;
                       }) ||
          std::adjacent_find(sortedOrder.begin(), sortedOrder.end()) !=
              sortedOrder.end() ||
          llvm::any_of(sortedOrder,
                       [&](uint32_t dimension) {
                         return dimension >= nested.iteratorTileSizes.size();
                       }) ||
          !nestedKeys
               .emplace(
                   nested.producerNodeId, nested.parentNodeId,
                   nested.producerResult, nested.parentOperand,
                   std::vector<int64_t>(nested.requestedResultExtents.begin(),
                                        nested.requestedResultExtents.end()),
                   std::vector<int64_t>(nested.producerIterationExtents.begin(),
                                        nested.producerIterationExtents.end()))
               .second)
        return failResult(failureReason,
                          "selected nested temporal assignment is malformed");
      nestedProducerNodes.insert(nested.producerNodeId);
    }
    std::set<std::tuple<uint32_t, uint32_t, unsigned, unsigned>> localUseKeys;
    for (const StructuredNodeLocalUse &use : group.localUses)
      if (use.producerNodeId == use.consumerNodeId ||
          (!llvm::any_of(group.shards,
                         [&](const auto &shard) {
                           return shard.structuredNodeId == use.producerNodeId;
                         }) &&
           !llvm::is_contained(group.recomputedProducerNodes,
                               use.producerNodeId)) ||
          !llvm::any_of(group.shards,
                        [&](const auto &shard) {
                          return shard.structuredNodeId == use.consumerNodeId;
                        }) ||
          !localUseKeys
               .emplace(use.producerNodeId, use.consumerNodeId,
                        use.producerResult, use.consumerOperand)
               .second)
        return failResult(
            failureReason,
            "selected local use is self-referential, missing, or duplicated");
    if ((!group.temporalTiles.empty() || !group.nestedTemporalTiles.empty()) &&
        llvm::any_of(group.shards, [&](const auto &shard) {
          const bool topLevel = temporalByNode.count(shard.structuredNodeId);
          const bool nested = nestedProducerNodes.count(shard.structuredNodeId);
          return topLevel == nested;
        }))
      return failResult(
          failureReason,
          "selected temporal group must classify every node exactly once");
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
    if (mlir::failed(fragment)) {
      if (!failureReason || failureReason->empty())
        setFailureReason(failureReason,
                         "selected node-group fragment materialization failed");
      return mlir::failure();
    }
    mlir::FailureOr<TileEntryStage> stage =
        makeTileEntryStage(*fragment, group->externalSources, failureReason);
    if (mlir::failed(stage))
      return mlir::failure();
    entryStages[firstShard.tile.getValue()].push_back(std::move(*stage));
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
    mlir::FailureOr<TileEntryStage> stage =
        makeTileEntryStage(*merge, /*externalSources=*/{}, failureReason);
    if (mlir::failed(stage))
      return mlir::failure();
    entryStages[group.mergeTile.getValue()].push_back(std::move(*stage));
    appendRelations(relations, std::move(merge->relations));
  }

  struct StageEndpoint {
    TileId tile{0};
    mlir::Type type;
    llvm::SmallVector<TileId, 4> selectedProducers;
  };
  std::map<RootValueKey, std::vector<StageEndpoint>> producers;
  std::map<RootValueKey, std::vector<StageEndpoint>> consumers;
  for (const auto &[tileValue, stages] : entryStages) {
    TileId tileId(tileValue);
    for (const TileEntryStage &stage : stages) {
      mlir::func::FuncOp function = stage.function;
      if (!function || function.getNumResults() != stage.results.size() ||
          function.getNumArguments() < stage.boundaries.size())
        return failResult(failureReason,
                          "node-group stage boundary inventory is malformed");
      for (auto [index, key] : llvm::enumerate(stage.results))
        producers[key].push_back(
            {tileId, function.getResultTypes()[index], {}});
      for (auto [index, key] : llvm::enumerate(stage.boundaries))
        if (key.kind == RootValueKind::StructuredResult) {
          StageEndpoint endpoint{
              tileId, function.getArgumentTypes()[index], {}};
          for (const StructuredNodeExternalSource &source :
               stage.externalSources)
            if (source.producerNodeId == key.owner &&
                source.producerResult == key.resultIndex &&
                !llvm::is_contained(endpoint.selectedProducers,
                                    source.producerTile))
              endpoint.selectedProducers.push_back(source.producerTile);
          llvm::sort(endpoint.selectedProducers, [](TileId lhs, TileId rhs) {
            return lhs.getValue() < rhs.getValue();
          });
          consumers[key].push_back(std::move(endpoint));
        }
    }
  }
  std::map<RootValueKey, CardDDRStageResource> resourcesByValue;
  for (const auto &[value, uses] : consumers) {
    auto definitions = producers.find(value);
    if (definitions == producers.end()) {
      if (failureReason) {
        llvm::raw_string_ostream diagnostic(*failureReason);
        diagnostic << "structured stage boundary has no selected producer; "
                      "node="
                   << value.owner << ",result=" << value.resultIndex
                   << ",consumer-tiles=[";
        for (auto [index, use] : llvm::enumerate(uses))
          diagnostic << (index ? "," : "") << use.tile.getValue();
        diagnostic << ']';
      }
      return mlir::failure();
    }
    for (const StageEndpoint &use : uses) {
      llvm::SmallVector<const StageEndpoint *, 4> selectedProducers;
      if (!use.selectedProducers.empty()) {
        for (TileId selectedTile : use.selectedProducers) {
          auto selected = llvm::find_if(definitions->second,
                                        [&](const StageEndpoint &producer) {
                                          return producer.tile == selectedTile;
                                        });
          if (selected == definitions->second.end() ||
              llvm::count_if(definitions->second,
                             [&](const StageEndpoint &producer) {
                               return producer.tile == selectedTile;
                             }) != 1) {
            selectedProducers.clear();
            break;
          }
          selectedProducers.push_back(&*selected);
        }
      } else {
        auto local = llvm::find_if(definitions->second,
                                   [&](const StageEndpoint &producer) {
                                     return producer.tile == use.tile;
                                   });
        if (local != definitions->second.end())
          continue;
        if (definitions->second.size() == 1)
          selectedProducers.push_back(&definitions->second.front());
      }
      if (selectedProducers.empty()) {
        if (failureReason) {
          llvm::raw_string_ostream diagnostic(*failureReason);
          diagnostic << "cross-Tile structured boundary has no unique "
                        "selected owner; node="
                     << value.owner << ",result=" << value.resultIndex
                     << ",consumer-tile=" << use.tile.getValue()
                     << ",producer-tiles=[";
          for (auto [index, producer] : llvm::enumerate(definitions->second))
            diagnostic << (index ? "," : "") << producer.tile.getValue();
          diagnostic << ']';
        }
        return mlir::failure();
      }
      if (selectedProducers.size() == 1 &&
          selectedProducers.front()->tile == use.tile)
        continue;
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
          selectedProducers.front()->type);
      if (!tensorType || !tensorType.hasStaticShape() ||
          use.type != selectedProducers.front()->type ||
          llvm::any_of(selectedProducers, [&](const StageEndpoint *producer) {
            return producer->type != selectedProducers.front()->type;
          }))
        return failResult(
            failureReason,
            "cross-Tile structured boundary has inconsistent tensor types");
      auto [resource, inserted] = resourcesByValue.try_emplace(value);
      if (inserted) {
        resource->second.value = value;
        resource->second.tensorType = tensorType;
      } else if (resource->second.tensorType != tensorType) {
        return failResult(
            failureReason,
            "cross-Tile structured boundary changed its selected type");
      }
      for (const StageEndpoint *producer : selectedProducers)
        if (!llvm::is_contained(resource->second.producers, producer->tile))
          resource->second.producers.push_back(producer->tile);
      if (!llvm::is_contained(resource->second.consumers, use.tile))
        resource->second.consumers.push_back(use.tile);
    }
  }
  std::vector<CardDDRStageResource> cardDDRResources;
  cardDDRResources.reserve(resourcesByValue.size());
  int64_t nextCardDDRResource = 0;
  for (auto &[value, resource] : resourcesByValue) {
    (void)value;
    llvm::sort(resource.producers, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    llvm::sort(resource.consumers, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    resource.resourceId = nextCardDDRResource++;
    cardDDRResources.push_back(resource);
  }
  for (const CardDDRStageResource &resource : cardDDRResources) {
    for (TileId producer : resource.producers)
      for (TileId consumer : resource.consumers)
        relations.cardDDRTransfers.push_back(
            {resource.value.owner, resource.value.resultIndex, producer,
             consumer, resource.resourceId});
    auto memrefType = mlir::MemRefType::get(
        resource.tensorType.getShape(), resource.tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(sourceModule.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
    std::string symbol =
        (llvm::Twine("card_ddr_") + llvm::Twine(resource.resourceId)).str();
    auto declaration = cardBuilder.create<mlir::memref::GlobalOp>(
        sourceModule.getLoc(), symbol, cardBuilder.getStringAttr("private"),
        memrefType, /*initial_value=*/mlir::Attribute{}, /*constant=*/false,
        /*alignment=*/mlir::IntegerAttr{});
    declaration->setAttr(kWaferCardDDRResourceAttrName,
                         CardDDRResourceAttr::get(sourceModule.getContext(),
                                                  resource.resourceId));
  }

  auto sourceReturn = mlir::cast<mlir::func::ReturnOp>(
      sourceProgram.getBody().front().getTerminator());
  for (auto [outputIndex, sourceValue] :
       llvm::enumerate(sourceReturn.getOperands())) {
    auto result = mlir::dyn_cast<mlir::OpResult>(sourceValue);
    auto owner =
        result ? llvm::find_if(
                     operationNodes,
                     [&](const StructuredOperationNodeMapping &candidate) {
                       return candidate.operation == result.getOwner();
                     })
               : operationNodes.end();
    if (!result || owner == operationNodes.end())
      continue;
    RootValueKey outputKey{RootValueKind::StructuredResult,
                           owner->structuredNodeId, result.getResultNumber()};
    for (const auto &[tileValue, stages] : entryStages) {
      (void)tileValue;
      llvm::SmallVector<mlir::Value, 4> outputBuffers;
      for (const TileEntryStage &stage : stages)
        for (auto [resultIndex, key] : llvm::enumerate(stage.results)) {
          const bool sameKey = !(key < outputKey) && !(outputKey < key);
          if (!sameKey || resultIndex >= stage.outputBuffers.size())
            continue;
          for (mlir::Value buffer : stage.outputBuffers[resultIndex])
            if (buffer && !llvm::is_contained(outputBuffers, buffer))
              outputBuffers.push_back(buffer);
        }
      for (mlir::Value outputBuffer : outputBuffers)
        relations.outputBuffers.push_back(
            {static_cast<unsigned>(outputIndex), outputBuffer});
    }
  }

  for (TileId tileId : sortedTiles) {
    TileModuleOp tile = tiles.lookup(tileId.getValue());
    if (!tile || mlir::failed(composeTileEntry(
                     tile, tileId, sourceProgram, operationNodes,
                     entryStages[tileId.getValue()], cardDDRResources,
                     relations, failureReason)))
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
