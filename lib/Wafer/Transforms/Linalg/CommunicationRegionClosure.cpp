//===- CommunicationRegionClosure.cpp - Close communication scopes -----===//

#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>

namespace wafer {
namespace {

static mlir::LogicalResult fail(SpatialRegionMaterializationFailure *failure,
                                SpatialRegionMaterializationFailureKind kind,
                                llvm::StringRef detail) {
  if (failure) {
    failure->kind = kind;
    failure->detail = detail.str();
  }
  return mlir::failure();
}

static TileRegionOp getRegionOwner(mlir::Value value) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    return mlir::dyn_cast<TileRegionOp>(result.getOwner());
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner()
             ? mlir::dyn_cast_or_null<TileRegionOp>(
                   argument.getOwner()->getParentOp())
             : TileRegionOp{};
}

static uint64_t getTileId(TileRegionOp region) {
  return static_cast<uint64_t>(
      region->getParentOfType<TileModuleOp>().getTileIdAttr().getInt());
}

static mlir::Operation *getOperation(TileRegionOp operation) {
  return operation.getOperation();
}

struct RelationInfo {
  unsigned index = 0;
  TileRegionOp sourceRegion;
  TileRegionOp destinationRegion;
  uint64_t sourceTile = 0;
  uint64_t destinationTile = 0;
};

struct PayloadGroup {
  mlir::Value source;
  TileRegionOp sourceRegion;
  uint64_t sourceTile = 0;
  llvm::SmallVector<unsigned, 8> relations;
  llvm::SmallVector<uint64_t, 16> participants;
  llvm::SmallVector<mlir::Operation *, 16> regions;
};

struct ExchangeComponent {
  llvm::SmallVector<unsigned, 16> groups;
  llvm::SmallVector<uint64_t, 16> participants;
};

static bool sameOrBefore(TileRegionOp lhs, TileRegionOp rhs) {
  return lhs == rhs ||
         (lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs));
}

static llvm::SmallVector<PayloadGroup, 16>
buildPayloadGroups(llvm::ArrayRef<RelationInfo> relationInfo,
                   llvm::ArrayRef<StructuredBoundaryRelation> relations) {
  llvm::SmallVector<PayloadGroup, 16> groups;
  for (const RelationInfo &info : relationInfo) {
    mlir::Value source = relations[info.index].sourceEndpoint;
    auto found = llvm::find_if(groups, [&](const PayloadGroup &group) {
      return group.source == source;
    });
    if (found == groups.end()) {
      PayloadGroup group;
      group.source = source;
      group.sourceRegion = info.sourceRegion;
      group.sourceTile = info.sourceTile;
      group.relations.push_back(info.index);
      group.participants.push_back(info.sourceTile);
      group.participants.push_back(info.destinationTile);
      group.regions.push_back(getOperation(info.sourceRegion));
      group.regions.push_back(getOperation(info.destinationRegion));
      groups.push_back(std::move(group));
      continue;
    }
    found->relations.push_back(info.index);
    if (!llvm::is_contained(found->participants, info.destinationTile))
      found->participants.push_back(info.destinationTile);
    if (!llvm::is_contained(found->regions,
                            getOperation(info.destinationRegion)))
      found->regions.push_back(getOperation(info.destinationRegion));
  }
  for (PayloadGroup &group : groups)
    llvm::sort(group.participants);
  return groups;
}

static llvm::SmallVector<ExchangeComponent, 8>
buildExchangeComponents(llvm::ArrayRef<PayloadGroup> groups,
                        llvm::ArrayRef<RelationInfo> relationInfo) {
  llvm::SmallVector<unsigned, 16> parents(groups.size());
  for (auto [index, parent] : llvm::enumerate(parents))
    parent = static_cast<unsigned>(index);
  std::function<unsigned(unsigned)> find = [&](unsigned index) {
    if (parents[index] != index)
      parents[index] = find(parents[index]);
    return parents[index];
  };
  auto unite = [&](unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  };
  auto isOrderedTwoTileExchange = [&](unsigned lhs, unsigned rhs) {
    const PayloadGroup &left = groups[lhs];
    const PayloadGroup &right = groups[rhs];
    if (left.participants.size() != 2 ||
        left.participants != right.participants || left.relations.size() != 1 ||
        right.relations.size() != 1 || left.sourceTile == right.sourceTile)
      return false;
    const RelationInfo &leftRelation = relationInfo[left.relations.front()];
    const RelationInfo &rightRelation = relationInfo[right.relations.front()];
    return leftRelation.destinationTile == right.sourceTile &&
           rightRelation.destinationTile == left.sourceTile &&
           sameOrBefore(leftRelation.sourceRegion,
                        rightRelation.destinationRegion) &&
           sameOrBefore(rightRelation.sourceRegion,
                        leftRelation.destinationRegion);
  };
  for (unsigned lhs = 0; lhs < groups.size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < groups.size(); ++rhs) {
      bool sharesRegion =
          llvm::any_of(groups[lhs].regions, [&](mlir::Operation *region) {
            return llvm::is_contained(groups[rhs].regions, region);
          });
      if (sharesRegion || isOrderedTwoTileExchange(lhs, rhs))
        unite(lhs, rhs);
    }

  llvm::SmallVector<ExchangeComponent, 8> components;
  llvm::DenseMap<unsigned, unsigned> componentByRoot;
  for (unsigned group = 0; group < groups.size(); ++group) {
    unsigned root = find(group);
    auto found = componentByRoot.find(root);
    if (found == componentByRoot.end()) {
      componentByRoot.try_emplace(root,
                                  static_cast<unsigned>(components.size()));
      components.emplace_back();
      found = componentByRoot.find(root);
    }
    ExchangeComponent &component = components[found->second];
    component.groups.push_back(group);
    for (uint64_t participant : groups[group].participants)
      if (!llvm::is_contained(component.participants, participant))
        component.participants.push_back(participant);
  }
  for (ExchangeComponent &component : components)
    llvm::sort(component.participants);
  return components;
}

static std::optional<unsigned>
getCompleteExchangeLaneCount(const ExchangeComponent &component,
                             llvm::ArrayRef<PayloadGroup> groups,
                             llvm::ArrayRef<RelationInfo> relationInfo) {
  if (component.participants.size() < 2)
    return std::nullopt;
  llvm::DenseMap<uint64_t, unsigned> sourceCounts;
  for (unsigned groupIndex : component.groups) {
    const PayloadGroup &group = groups[groupIndex];
    if (group.participants != component.participants ||
        group.relations.size() + 1 != component.participants.size())
      return std::nullopt;
    ++sourceCounts[group.sourceTile];
    llvm::DenseSet<uint64_t> destinations;
    for (unsigned relationIndex : group.relations) {
      const RelationInfo &relation = relationInfo[relationIndex];
      if (relation.sourceTile != group.sourceTile ||
          !destinations.insert(relation.destinationTile).second)
        return std::nullopt;
    }
    for (uint64_t participant : component.participants)
      if (participant != group.sourceTile &&
          !destinations.contains(participant))
        return std::nullopt;
  }
  unsigned lanes = 0;
  for (uint64_t participant : component.participants) {
    unsigned count = sourceCounts.lookup(participant);
    if (count == 0 || (lanes != 0 && count != lanes))
      return std::nullopt;
    lanes = count;
  }
  return lanes;
}

struct Position {
  unsigned region = 0;
  unsigned operation = 0;
};

static bool operator<(const Position &lhs, const Position &rhs) {
  return std::tie(lhs.region, lhs.operation) <
         std::tie(rhs.region, rhs.operation);
}

static std::optional<unsigned> getOperationIndex(mlir::Operation *operation,
                                                 mlir::Block &block) {
  mlir::Operation *anchor = operation;
  while (anchor && anchor->getBlock() != &block)
    anchor = anchor->getParentOp();
  if (!anchor)
    return std::nullopt;
  unsigned index = 1;
  for (mlir::Operation &candidate : block) {
    if (&candidate == anchor)
      return index;
    ++index;
  }
  return std::nullopt;
}

static std::optional<Position>
getSourcePosition(const StructuredBoundaryRelation &relation,
                  unsigned regionIndex) {
  auto result = mlir::dyn_cast<mlir::OpResult>(relation.sourceEndpoint);
  TileRegionOp region = getRegionOwner(relation.sourceEndpoint);
  if (!result || !region || result.getResultNumber() >= region.getNumResults())
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
  if (!yield)
    return std::nullopt;
  mlir::Value yielded = yield.getValues()[result.getResultNumber()];
  if (mlir::isa<mlir::BlockArgument>(yielded))
    return Position{regionIndex, 0};
  std::optional<unsigned> operation =
      getOperationIndex(yielded.getDefiningOp(), region.getBody().front());
  return operation ? std::optional<Position>(Position{regionIndex, *operation})
                   : std::nullopt;
}

static std::optional<Position>
getDestinationPosition(const StructuredBoundaryRelation &relation,
                       unsigned regionIndex) {
  auto argument =
      mlir::dyn_cast<mlir::BlockArgument>(relation.destinationEndpoint);
  TileRegionOp region = getRegionOwner(relation.destinationEndpoint);
  if (!argument || !region)
    return std::nullopt;
  unsigned first = std::numeric_limits<unsigned>::max();
  for (mlir::Operation *user : argument.getUsers()) {
    std::optional<unsigned> operation =
        getOperationIndex(user, region.getBody().front());
    if (!operation)
      return std::nullopt;
    first = std::min(first, *operation);
  }
  if (first == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return Position{regionIndex, first};
}

static bool hasCommonCut(const ExchangeComponent &component,
                         llvm::ArrayRef<PayloadGroup> groups,
                         llvm::ArrayRef<RelationInfo> relationInfo,
                         llvm::ArrayRef<StructuredBoundaryRelation> relations) {
  for (uint64_t tile : component.participants) {
    llvm::SmallVector<TileRegionOp, 8> tileRegions;
    for (unsigned groupIndex : component.groups)
      for (unsigned relationIndex : groups[groupIndex].relations) {
        const RelationInfo &info = relationInfo[relationIndex];
        if (info.sourceTile == tile &&
            !llvm::is_contained(tileRegions, info.sourceRegion))
          tileRegions.push_back(info.sourceRegion);
        if (info.destinationTile == tile &&
            !llvm::is_contained(tileRegions, info.destinationRegion))
          tileRegions.push_back(info.destinationRegion);
      }
    if (tileRegions.empty())
      return false;
    llvm::sort(tileRegions, [](TileRegionOp lhs, TileRegionOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    llvm::DenseMap<mlir::Operation *, unsigned> regionIndices;
    for (auto [index, region] : llvm::enumerate(tileRegions))
      regionIndices.try_emplace(region.getOperation(),
                                static_cast<unsigned>(index));

    std::optional<Position> lastProducer;
    std::optional<Position> firstConsumer;
    for (unsigned groupIndex : component.groups)
      for (unsigned relationIndex : groups[groupIndex].relations) {
        const RelationInfo &info = relationInfo[relationIndex];
        if (info.sourceTile == tile) {
          auto position = getSourcePosition(
              relations[relationIndex],
              regionIndices.lookup(getOperation(info.sourceRegion)));
          if (!position)
            return false;
          if (!lastProducer || *lastProducer < *position)
            lastProducer = position;
        }
        if (info.destinationTile == tile) {
          auto position = getDestinationPosition(
              relations[relationIndex],
              regionIndices.lookup(getOperation(info.destinationRegion)));
          if (!position)
            return false;
          if (!firstConsumer || *position < *firstConsumer)
            firstConsumer = position;
        }
      }
    if (!lastProducer || !firstConsumer || !(*lastProducer < *firstConsumer))
      return false;
  }
  return true;
}

struct TileMergeSet {
  unsigned component = 0;
  TileModuleOp tile;
  mlir::func::FuncOp function;
  llvm::SmallVector<TileRegionOp, 16> regions;
};

struct MergeBuilder {
  explicit MergeBuilder(mlir::MLIRContext *context) : builder(context) {
    body.push_back(new mlir::Block());
    block = &body.front();
    builder.setInsertionPointToEnd(block);
  }

  mlir::FailureOr<mlir::Value> mapExternal(mlir::Value value) {
    if (mlir::Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (definition && selectedRegions.contains(definition))
      return mlir::failure();
    if (definition && definition->getBlock() == parentBlock &&
        insertionAnchor->isBeforeInBlock(definition)) {
      if (!mlir::isMemoryEffectFree(definition) ||
          mlir::isa<TileRegionOp>(definition))
        return mlir::failure();
      for (mlir::Value operand : definition->getOperands()) {
        auto mapped = mapExternal(operand);
        if (mlir::failed(mapped))
          return mlir::failure();
        mapping.map(operand, *mapped);
      }
      mlir::Operation *cloned = builder.clone(*definition, mapping);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(definition->getResults(), cloned->getResults()))
        mapping.map(oldResult, newResult);
      return mapping.lookup(value);
    }
    auto found = inputArguments.find(value);
    if (found != inputArguments.end())
      return found->second;
    inputs.push_back(value);
    mlir::BlockArgument argument =
        block->addArgument(value.getType(), value.getLoc());
    inputArguments.try_emplace(value, argument);
    mapping.map(value, argument);
    return argument;
  }

  mlir::Region body;
  mlir::Block *block = nullptr;
  mlir::OpBuilder builder;
  mlir::IRMapping mapping;
  mlir::Block *parentBlock = nullptr;
  mlir::Operation *insertionAnchor = nullptr;
  llvm::DenseSet<mlir::Operation *> selectedRegions;
  llvm::SmallVector<mlir::Value, 16> inputs;
  llvm::DenseMap<mlir::Value, mlir::BlockArgument> inputArguments;
  llvm::DenseMap<mlir::Value, mlir::Value> argumentReplacements;
  llvm::DenseMap<mlir::Value, unsigned> resultIndices;
  llvm::SmallVector<mlir::Value, 16> yields;
};

static mlir::LogicalResult buildMergedBody(TileMergeSet &set,
                                           MergeBuilder &merged) {
  merged.parentBlock = &set.function.getBody().front();
  merged.insertionAnchor = set.regions.front().getOperation();
  for (TileRegionOp region : set.regions)
    merged.selectedRegions.insert(region.getOperation());

  for (TileRegionOp region : set.regions) {
    mlir::Block &sourceBlock = region.getBody().front();
    for (auto [input, argument] :
         llvm::zip_equal(region.getInputs(), sourceBlock.getArguments())) {
      auto mapped = merged.mapExternal(input);
      if (mlir::failed(mapped))
        return mlir::failure();
      merged.mapping.map(argument, *mapped);
      merged.argumentReplacements.try_emplace(argument, *mapped);
    }
    for (mlir::Operation &operation : sourceBlock.without_terminator()) {
      mlir::Operation *cloned = merged.builder.clone(operation, merged.mapping);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(operation.getResults(), cloned->getResults()))
        merged.mapping.map(oldResult, newResult);
    }
    auto yield = mlir::cast<TileYieldOp>(sourceBlock.getTerminator());
    for (auto [oldResult, yielded] :
         llvm::zip_equal(region.getResults(), yield.getValues())) {
      mlir::Value mapped = merged.mapping.lookupOrDefault(yielded);
      merged.mapping.map(oldResult, mapped);
      merged.resultIndices.try_emplace(oldResult, merged.yields.size());
      merged.yields.push_back(mapped);
    }
  }
  merged.builder.create<TileYieldOp>(set.regions.front().getLoc(),
                                     merged.yields);
  return mlir::success();
}

static bool canMergeTileRegions(TileMergeSet &set) {
  if (set.regions.size() < 2)
    return true;
  MergeBuilder merged(set.function.getContext());
  return mlir::succeeded(buildMergedBody(set, merged));
}

static mlir::FailureOr<TileRegionOp>
mergeTileRegions(TileMergeSet &set,
                 llvm::DenseMap<mlir::Value, mlir::Value> &replacements) {
  if (set.regions.size() < 2)
    return set.regions.front();
  MergeBuilder merged(set.function.getContext());
  if (mlir::failed(buildMergedBody(set, merged)))
    return mlir::failure();
  llvm::SmallVector<mlir::Type, 16> resultTypes;
  llvm::transform(merged.yields, std::back_inserter(resultTypes),
                  [](mlir::Value value) { return value.getType(); });
  mlir::OpBuilder parentBuilder(set.regions.front());
  auto replacement = parentBuilder.create<TileRegionOp>(
      set.regions.front().getLoc(), resultTypes, merged.inputs);
  replacement.getBody().takeBody(merged.body);

  for (const auto &[argument, mapped] : merged.argumentReplacements)
    replacements.try_emplace(argument, mapped);

  for (TileRegionOp region : set.regions)
    for (mlir::Value result : region.getResults()) {
      mlir::Value replacementResult =
          replacement.getResult(merged.resultIndices.lookup(result));
      replacements[result] = replacementResult;
      result.replaceAllUsesWith(replacementResult);
    }
  for (TileRegionOp region : llvm::reverse(set.regions))
    region.erase();
  return replacement;
}

} // namespace

mlir::LogicalResult closeCrossTileCommunicationRegions(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    CommunicationRegionClosureStatistics *statistics,
    SpatialRegionMaterializationFailure *failure) {
  if (failure)
    *failure = {};
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyStructuralTileRegions(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module.getOperation(), relations)))
    return fail(failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "communication closure requires current structural Tile IR");
  if (relations.boundaryRelations.empty())
    return mlir::success();

  llvm::SmallVector<RelationInfo, 32> relationInfo;
  for (auto [index, relation] : llvm::enumerate(relations.boundaryRelations)) {
    TileRegionOp source = getRegionOwner(relation.sourceEndpoint);
    TileRegionOp destination = getRegionOwner(relation.destinationEndpoint);
    if (!source || !destination ||
        source->getParentOfType<TileModuleOp>() ==
            destination->getParentOfType<TileModuleOp>())
      return fail(failure,
                  SpatialRegionMaterializationFailureKind::BrokenContract,
                  "communication closure relation has invalid endpoints");
    relationInfo.push_back(RelationInfo{static_cast<unsigned>(index), source,
                                        destination, getTileId(source),
                                        getTileId(destination)});
  }
  llvm::SmallVector<PayloadGroup, 16> groups =
      buildPayloadGroups(relationInfo, relations.boundaryRelations);
  llvm::SmallVector<ExchangeComponent, 8> components =
      buildExchangeComponents(groups, relationInfo);

  llvm::SmallVector<TileMergeSet, 16> sets;
  llvm::SmallVector<unsigned, 8> candidateComponents;
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    if (!getCompleteExchangeLaneCount(component, groups, relationInfo) ||
        !hasCommonCut(component, groups, relationInfo,
                      relations.boundaryRelations))
      continue;
    candidateComponents.push_back(static_cast<unsigned>(componentIndex));
    for (uint64_t tileId : component.participants) {
      TileMergeSet set;
      set.component = static_cast<unsigned>(componentIndex);
      for (unsigned groupIndex : component.groups)
        for (unsigned relationIndex : groups[groupIndex].relations) {
          const RelationInfo &info = relationInfo[relationIndex];
          for (TileRegionOp region :
               {info.sourceRegion, info.destinationRegion}) {
            if (getTileId(region) != tileId ||
                llvm::is_contained(set.regions, region))
              continue;
            set.regions.push_back(region);
          }
        }
      llvm::sort(set.regions, [](TileRegionOp lhs, TileRegionOp rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
      if (set.regions.empty())
        return fail(failure,
                    SpatialRegionMaterializationFailureKind::BrokenContract,
                    "closed exchange has no participating TileRegion");
      set.tile = set.regions.front()->getParentOfType<TileModuleOp>();
      set.function = set.regions.front()->getParentOfType<mlir::func::FuncOp>();
      if (!set.tile || !set.function ||
          llvm::any_of(set.regions, [&](TileRegionOp region) {
            return region->getParentOfType<TileModuleOp>() != set.tile ||
                   region->getParentOfType<mlir::func::FuncOp>() !=
                       set.function ||
                   !region.getBody().hasOneBlock();
          }))
        return fail(failure,
                    SpatialRegionMaterializationFailureKind::Unsupported,
                    "one exchange scope spans unsupported Region owners");
      sets.push_back(std::move(set));
    }
  }

  llvm::DenseSet<unsigned> closableComponents;
  for (unsigned component : candidateComponents) {
    bool closable = true;
    for (TileMergeSet &set : sets)
      if (set.component == component && !canMergeTileRegions(set)) {
        closable = false;
        break;
      }
    if (!closable)
      continue;
    closableComponents.insert(component);
    if (statistics)
      ++statistics->closedExchangeComponents;
  }

  llvm::DenseMap<mlir::Value, mlir::Value> replacements;
  for (TileMergeSet &set : sets) {
    if (!closableComponents.contains(set.component))
      continue;
    if (set.regions.size() < 2)
      continue;
    uint64_t regionCount = set.regions.size();
    if (mlir::failed(mergeTileRegions(set, replacements)))
      return fail(failure,
                  SpatialRegionMaterializationFailureKind::CompilerFailure,
                  "preflighted exchange scope failed during current IR "
                  "rewrite");
    if (statistics) {
      ++statistics->mergedTileScopes;
      statistics->mergedRegions += regionCount;
    }
  }

  auto retarget = [&](mlir::Value &value) {
    if (auto found = replacements.find(value); found != replacements.end())
      value = found->second;
  };
  for (StructuredBoundaryRelation &relation : relations.boundaryRelations) {
    retarget(relation.sourceEndpoint);
    retarget(relation.destinationEndpoint);
  }
  for (StructuredOutputRelation &output : relations.structuralOutputs)
    retarget(output.endpoint);

  if (mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyStructuralTileRegions(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module.getOperation(), relations)))
    return fail(failure,
                SpatialRegionMaterializationFailureKind::CompilerFailure,
                "communication closure produced invalid current Tile IR");
  return mlir::success();
}

} // namespace wafer
