//===- CardEdgeCarriers.cpp - Selected dependency carriers ------------===//

#include "Wafer/Planning/PhysicalDataflow/CardEdgeCarriers.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
getRectangles(const analysis::ExactIndexSet &set) {
  mlir::FailureOr<analysis::ExactIndexSet> normalized =
      analysis::normalizeFiniteExactIndexSet(set);
  if (mlir::failed(normalized))
    return mlir::failure();
  return llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>(
      normalized->getBoxes().begin(), normalized->getBoxes().end());
}

const analysis::SourceDemand *
findSource(const analysis::DestinationDemand &destination,
           const SemanticRootKey &producer, uint32_t result) {
  auto source = llvm::find_if(
      destination.sources, [&](const analysis::SourceDemand &candidate) {
        const auto *structured =
            std::get_if<analysis::StructuredResultSource>(&candidate.source);
        return structured && structured->root == producer &&
               structured->result == result;
      });
  return source == destination.sources.end() ? nullptr : &*source;
}

const analysis::FinalResultOwner *
findOwner(const analysis::ExactDemandProof &proof,
          const analysis::StructuredResultSource &source,
          const analysis::OwnerIntersection &intersection) {
  auto owner = llvm::find_if(
      proof.finalOwners, [&](const analysis::FinalResultOwner &candidate) {
        return candidate.root == source.root &&
               candidate.result == source.result &&
               candidate.tile == intersection.tile &&
               candidate.shard == intersection.ownerShard &&
               candidate.reductionGroup == intersection.reductionGroup;
      });
  return owner == proof.finalOwners.end() ? nullptr : &*owner;
}

bool updateBounds(SpatialEdgeStrategy &strategy,
                  const analysis::StaticRectangularIndexSet &piece) {
  if (strategy.producerOffsets.empty()) {
    strategy.producerOffsets = piece.offsets;
    strategy.producerSizes = piece.sizes;
    return true;
  }
  if (piece.offsets.size() != strategy.producerOffsets.size())
    return false;
  for (size_t dimension = 0; dimension < piece.offsets.size(); ++dimension) {
    const int64_t currentLimit =
        strategy.producerOffsets[dimension] + strategy.producerSizes[dimension];
    const int64_t pieceLimit =
        piece.offsets[dimension] + piece.sizes[dimension];
    const int64_t offset =
        std::min(strategy.producerOffsets[dimension], piece.offsets[dimension]);
    const int64_t limit = std::max(currentLimit, pieceLimit);
    strategy.producerOffsets[dimension] = offset;
    strategy.producerSizes[dimension] = limit - offset;
  }
  return true;
}

mlir::LogicalResult appendCarrier(
    TileMapping &mapping, const StructuredDAGEdge &edge,
    const StructuredDAGNode &producer, const StructuredDAGNode &consumer,
    const SemanticRootKey &producerRoot, const SemanticRootKey &consumerRoot,
    const analysis::DependencyDemand &dependency,
    const analysis::ExactDemandProof &proof, const MovementPlan &movement,
    llvm::ArrayRef<CoupledComponentResultMapping> components,
    llvm::ArrayRef<StructuredOperationRootMapping> roots,
    const std::map<MovementActionId, int64_t> &cardDDRIds,
    std::map<int64_t, mlir::RankedTensorType> &usedCardDDRResources,
    std::string *failureReason) {
  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producer.operation->getResult(edge.producerResult).getType());
  if (!producerType || !producerType.hasStaticShape() ||
      !producerType.getElementType().isIntOrFloat()) {
    if (failureReason)
      *failureReason = "producer result is not a static numeric tensor";
    return mlir::failure();
  }
  const unsigned elementBits =
      producerType.getElementType().getIntOrFloatBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return mlir::failure();
  int64_t payloadSlice = 0;
  for (const analysis::DestinationDemand &destination :
       dependency.perDestination) {
    const analysis::SourceDemand *source =
        findSource(destination, producerRoot, edge.producerResult);
    if (!source || source->requiredDomain.isEmpty())
      continue;
    mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
        demandPieces = getRectangles(source->requiredDomain);
    if (mlir::failed(demandPieces) || demandPieces->empty()) {
      if (failureReason)
        *failureReason = "producer demand has no finite rectangle union";
      return mlir::failure();
    }

    SpatialEdgeStrategy strategy;
    strategy.producer = producer.operation;
    strategy.producerResult = edge.producerResult;
    strategy.consumer = consumer.operation;
    strategy.consumerOperand = edge.consumerOperand;
    strategy.destinationTile = destination.destinationTile;
    strategy.sourceTile = destination.destinationTile;
    strategy.action = SpatialEdgeAction::PeerFragments;
    strategy.fragmentsDefineProducerDemand = demandPieces->size() > 1;
    for (const analysis::StaticRectangularIndexSet &piece : *demandPieces)
      if (!updateBounds(strategy, piece))
        return mlir::failure();

    for (const analysis::OwnerIntersection &intersection :
         source->eligibleFinalOwners) {
      const auto *structured =
          std::get_if<analysis::StructuredResultSource>(&source->source);
      const analysis::FinalResultOwner *owner =
          structured ? findOwner(proof, *structured, intersection) : nullptr;
      if (!owner)
        return mlir::failure();
      mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
          ownedPieces = getRectangles(intersection.domain);
      if (mlir::failed(ownedPieces))
        return mlir::failure();
      for (const analysis::StaticRectangularIndexSet &piece : *ownedPieces) {
        const bool resident = intersection.tile == destination.destinationTile;
        uint64_t elements = 1;
        for (int64_t size : piece.sizes)
          elements = saturatingMultiply(elements, size);
        const uint64_t bytes =
            resident ? 0 : saturatingMultiply(elements, elementBits / 8);
        if (!resident &&
            (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()))
          return mlir::failure();
        SpatialEdgeFragment fragment{resident
                                         ? SpatialEdgeFragmentKind::Resident
                                         : SpatialEdgeFragmentKind::Peer,
                                     piece.offsets,
                                     piece.sizes,
                                     intersection.tile,
                                     bytes,
                                     resident ? 0 : edge.id,
                                     resident ? 0 : payloadSlice++};
        fragment.ownerShard = intersection.ownerShard;
        fragment.reductionGroup = intersection.reductionGroup;
        strategy.fragments.push_back(std::move(fragment));
      }
    }
    if (strategy.fragments.empty())
      return mlir::failure();
    auto consumerOwner = llvm::find_if(
        proof.finalOwners, [&](const analysis::FinalResultOwner &owner) {
          return owner.root == consumerRoot && owner.result == 0 &&
                 owner.shard && *owner.shard == destination.destinationShard;
        });
    if (consumerOwner == proof.finalOwners.end()) {
      if (failureReason)
        *failureReason = "candidate consumer result has no final owner";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
        consumerPieces = getRectangles(consumerOwner->domain);
    if (mlir::failed(consumerPieces) || consumerPieces->size() != 1) {
      if (failureReason)
        *failureReason =
            "candidate consumer result has no finite final-owner rectangle";
      return mlir::failure();
    }
    strategy.consumerOffsets = consumerPieces->front().offsets;
    strategy.consumerSizes = consumerPieces->front().sizes;
    auto selectExactSourceTile = [&](TileId selectedTile) {
      llvm::SmallVector<SpatialEdgeFragment, 4> selectedFragments;
      llvm::SmallVector<analysis::StaticRectangularIndexSet, 4> selectedBoxes;
      for (const SpatialEdgeFragment &fragment : strategy.fragments) {
        if (fragment.sourceTile != selectedTile)
          continue;
        selectedFragments.push_back(fragment);
        selectedBoxes.push_back({fragment.offsets, fragment.sizes});
      }
      auto orderBox = [](const analysis::StaticRectangularIndexSet &lhs,
                         const analysis::StaticRectangularIndexSet &rhs) {
        return std::tie(lhs.offsets, lhs.sizes) <
               std::tie(rhs.offsets, rhs.sizes);
      };
      llvm::sort(selectedBoxes, orderBox);
      llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> requiredBoxes =
          *demandPieces;
      llvm::sort(requiredBoxes, orderBox);
      if (selectedBoxes.size() != requiredBoxes.size() ||
          !llvm::equal(selectedBoxes, requiredBoxes,
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.offsets == rhs.offsets &&
                                lhs.sizes == rhs.sizes;
                       }))
        return false;
      strategy.sourceTile = selectedTile;
      strategy.fragments = std::move(selectedFragments);
      strategy.fragmentsDefineProducerDemand = strategy.fragments.size() > 1;
      return true;
    };
    bool movementResolved = false;
    const bool requiresCrossTileCarrier =
        llvm::any_of(strategy.fragments, [](const SpatialEdgeFragment &piece) {
          return piece.kind == SpatialEdgeFragmentKind::Peer;
        });
    if (!movementResolved) {
      auto component = llvm::find_if(
          components, [&](const CoupledComponentResultMapping &candidate) {
            return candidate.operation == producer.operation &&
                   candidate.result == edge.producerResult;
          });
      if (component != components.end() && requiresCrossTileCarrier) {
        TileId componentTile =
            std::visit([](const auto &source) { return source.work.tile; },
                       component->component.execution.source);
        if (!selectExactSourceTile(componentTile)) {
          if (failureReason)
            *failureReason =
                "coupled component edge is not exactly covered by its "
                "selected source Tile";
          return mlir::failure();
        }
        llvm::SmallVector<const ReductionGatherPlan *, 2> gathers;
        for (const ReductionGatherPlan &candidate : movement.reductionGathers) {
          const auto *value =
              std::get_if<CoupledComponentValueId>(&candidate.id.value);
          TileId mergeTile =
              std::visit([](const auto &source) { return source.work.tile; },
                         candidate.mergeExecution.source);
          if (value && *value == component->component &&
              mergeTile == strategy.destinationTile)
            gathers.push_back(&candidate);
        }
        if (gathers.size() != 1) {
          if (failureReason)
            *failureReason = "coupled component edge has no unique selected "
                             "reduction-gather action";
          return mlir::failure();
        }
        MovementActionId action(gathers.front()->id);
        const bool selectedPeer =
            llvm::any_of(movement.peerGraphs, [&](const auto &graph) {
              return llvm::is_contained(graph.actions, action);
            });
        strategy.action = selectedPeer ? SpatialEdgeAction::PeerFragments
                                       : SpatialEdgeAction::CardDDRTransfer;
        if (!selectedPeer) {
          auto resourceId = cardDDRIds.find(action);
          if (resourceId == cardDDRIds.end()) {
            if (failureReason)
              *failureReason = "coupled DDR action has no unique card-shared "
                               "resource";
            return mlir::failure();
          }
          auto [resource, inserted] = usedCardDDRResources.try_emplace(
              resourceId->second, producerType);
          if (!inserted && resource->second != producerType) {
            if (failureReason)
              *failureReason =
                  "coupled card DDR resource has inconsistent tensor types";
            return mlir::failure();
          }
          strategy.cardDDRResource = resourceId->second;
          for (SpatialEdgeFragment &fragment : strategy.fragments) {
            fragment.kind = SpatialEdgeFragmentKind::CardDDR;
            fragment.communicationId = 0;
            fragment.payloadSlice = 0;
          }
        }
        movementResolved = true;
      }
    }
    if (!movementResolved) {
      llvm::SmallVector<const DDRBoundaryTransferPlan *, 2> transfers;
      auto producerMovementRoot = llvm::find_if(
          roots, [&](const StructuredOperationRootMapping &mapping) {
            return mapping.operation == producer.operation;
          });
      auto consumerMovementRoot = llvm::find_if(
          roots, [&](const StructuredOperationRootMapping &mapping) {
            return mapping.operation == consumer.operation;
          });
      const SemanticRootKey &selectedProducerRoot =
          producerMovementRoot == roots.end() ? producerRoot
                                              : producerMovementRoot->root;
      const SemanticRootKey &selectedConsumerRoot =
          consumerMovementRoot == roots.end() ? consumerRoot
                                              : consumerMovementRoot->root;
      unsigned selectedConsumerOperand = edge.consumerOperand;
      if (consumerMovementRoot != roots.end()) {
        auto semanticOperand =
            llvm::find_if(consumerMovementRoot->semanticOperandIndices,
                          [&](const std::pair<unsigned, unsigned> &mapping) {
                            return mapping.first == edge.consumerOperand;
                          });
        if (semanticOperand !=
            consumerMovementRoot->semanticOperandIndices.end())
          selectedConsumerOperand = semanticOperand->second;
      }
      LogicalShardId selectedDestinationShard = destination.destinationShard;
      selectedDestinationShard.root = selectedConsumerRoot;
      if (consumerMovementRoot != roots.end() &&
          consumerMovementRoot->execution)
        if (const auto *required = std::get_if<RequiredRootExecution>(
                &consumerMovementRoot->execution->source))
          selectedDestinationShard = required->shard;
      for (const DDRBoundaryTransferPlan &candidate : movement.ddrTransfers) {
        const auto *result =
            std::get_if<ExecutionResultValueId>(&candidate.source.logicalValue);
        const auto *execution =
            result ? std::get_if<ExecutionInstanceId>(&result->execution)
                   : nullptr;
        std::optional<SemanticRootKey> sourceRoot;
        if (execution)
          sourceRoot =
              std::visit([](const auto &source) { return source.work.root; },
                         execution->source);
        const BoundaryRegionValueId &destinationId = candidate.id.destination;
        if (sourceRoot && *sourceRoot == selectedProducerRoot &&
            result->result == edge.producerResult &&
            destinationId.work.root == selectedConsumerRoot &&
            destinationId.work.tile == destination.destinationTile &&
            destinationId.fragment.use.operand == selectedConsumerOperand &&
            destinationId.fragment.use.destinationShard ==
                selectedDestinationShard)
          transfers.push_back(&candidate);
      }
      if (transfers.empty()) {
        if (!requiresCrossTileCarrier) {
          mapping.edgeStrategies.push_back(std::move(strategy));
          continue;
        }
        if (failureReason) {
          llvm::raw_string_ostream diagnostic(*failureReason);
          diagnostic << "selected cross-Tile edge has no MovementPlan "
                        "boundary action; producer="
                     << producer.operation->getName()
                     << ",producer-result=" << edge.producerResult
                     << ",consumer=" << consumer.operation->getName()
                     << ",consumer-operand=" << edge.consumerOperand
                     << ",destination-tile="
                     << strategy.destinationTile.getValue()
                     << ",eligible-source-tiles=[";
          llvm::interleaveComma(strategy.fragments, diagnostic,
                                [&](const SpatialEdgeFragment &fragment) {
                                  diagnostic << fragment.sourceTile.getValue();
                                });
          diagnostic << "],semantic-consumer-operand="
                     << selectedConsumerOperand;
        }
        return mlir::failure();
      }
      std::vector<const PeerTransferGraphPlan *> orderedPeerGraphs;
      for (const PeerTransferGraphPlan &graph : movement.peerGraphs)
        orderedPeerGraphs.push_back(&graph);
      llvm::sort(orderedPeerGraphs, [](const auto *lhs, const auto *rhs) {
        return lhs->actions < rhs->actions;
      });
      std::map<MovementActionId, llvm::SmallVector<SpatialEdgeFragment *, 4>>
          fragmentsByAction;
      bool hasPeer = false;
      bool hasCardDDR = false;
      for (SpatialEdgeFragment &fragment : strategy.fragments) {
        std::optional<LogicalShardId> selectedOwnerShard = fragment.ownerShard;
        std::optional<ReductionGroupId> selectedReductionGroup =
            fragment.reductionGroup;
        if (producerMovementRoot != roots.end() &&
            producerMovementRoot->execution) {
          if (const auto *required = std::get_if<RequiredRootExecution>(
                  &producerMovementRoot->execution->source)) {
            selectedOwnerShard = required->shard;
            selectedReductionGroup.reset();
          }
          if (const auto *required = std::get_if<RequiredMergeExecution>(
                  &producerMovementRoot->execution->source)) {
            selectedReductionGroup = required->group;
            selectedOwnerShard.reset();
          }
        } else {
          if (selectedOwnerShard)
            selectedOwnerShard->root = selectedProducerRoot;
          if (selectedReductionGroup)
            selectedReductionGroup->root = selectedProducerRoot;
        }
        llvm::SmallVector<const DDRBoundaryTransferPlan *, 2> matches;
        for (const DDRBoundaryTransferPlan *transfer : transfers) {
          const auto *result = std::get_if<ExecutionResultValueId>(
              &transfer->source.logicalValue);
          const auto *execution =
              result ? std::get_if<ExecutionInstanceId>(&result->execution)
                     : nullptr;
          std::optional<TileId> sourceTile =
              execution
                  ? std::optional<TileId>(std::visit(
                        [](const auto &source) { return source.work.tile; },
                        execution->source))
                  : std::nullopt;
          const DemandFragmentId &selected = transfer->id.destination.fragment;
          std::optional<LogicalShardId> candidateOwnerShard =
              selected.ownerShard;
          std::optional<ReductionGroupId> candidateReductionGroup =
              selected.reductionGroup;
          std::optional<TileId> candidateOwnerTile = selected.ownerTile;
          if (execution) {
            if (const auto *required =
                    std::get_if<RequiredRootExecution>(&execution->source)) {
              candidateOwnerShard = required->shard;
              candidateReductionGroup.reset();
            }
            if (const auto *required =
                    std::get_if<RequiredMergeExecution>(&execution->source)) {
              candidateReductionGroup = required->group;
              candidateOwnerShard.reset();
            }
            candidateOwnerTile = sourceTile;
          }
          if (sourceTile && *sourceTile == fragment.sourceTile &&
              candidateOwnerTile ==
                  std::optional<TileId>(fragment.sourceTile) &&
              candidateOwnerShard == selectedOwnerShard &&
              candidateReductionGroup == selectedReductionGroup)
            matches.push_back(transfer);
        }
        if (matches.size() != 1) {
          if (failureReason) {
            llvm::raw_string_ostream diagnostic(*failureReason);
            diagnostic << "selected edge fragment has no unique MovementPlan "
                          "owner action; producer="
                       << producer.operation->getName()
                       << ",consumer=" << consumer.operation->getName()
                       << ",source-tile=" << fragment.sourceTile.getValue()
                       << ",matches=" << matches.size() << ",owner-shard=";
            if (selectedOwnerShard) {
              diagnostic << '[';
              llvm::interleaveComma(selectedOwnerShard->coordinate, diagnostic);
              diagnostic << ']';
            } else {
              diagnostic << "none";
            }
            diagnostic << ",candidate-owners=[";
            bool first = true;
            for (const DDRBoundaryTransferPlan *transfer : transfers) {
              if (!first)
                diagnostic << ';';
              first = false;
              const DemandFragmentId &candidate =
                  transfer->id.destination.fragment;
              diagnostic << "tile=";
              if (candidate.ownerTile)
                diagnostic << candidate.ownerTile->getValue();
              else
                diagnostic << "none";
              diagnostic << ",shard=";
              if (candidate.ownerShard) {
                diagnostic << '[';
                llvm::interleaveComma(candidate.ownerShard->coordinate,
                                      diagnostic);
                diagnostic << ']';
              } else {
                diagnostic << "none";
              }
              const auto *result = std::get_if<ExecutionResultValueId>(
                  &transfer->source.logicalValue);
              const auto *execution =
                  result ? std::get_if<ExecutionInstanceId>(&result->execution)
                         : nullptr;
              diagnostic << ",source=";
              if (execution) {
                std::visit(
                    [&](const auto &source) {
                      using T = std::decay_t<decltype(source)>;
                      if constexpr (std::is_same_v<T, RequiredRootExecution>) {
                        diagnostic << "shard:[";
                        llvm::interleaveComma(source.shard.coordinate,
                                              diagnostic);
                        diagnostic << ']';
                      } else {
                        diagnostic << "group:" << source.group.resultGroup
                                   << ":[";
                        llvm::interleaveComma(source.group.parallelCoordinate,
                                              diagnostic);
                        diagnostic << ']';
                      }
                    },
                    execution->source);
              } else {
                diagnostic << "none";
              }
            }
            diagnostic << "],selected-group=";
            if (selectedReductionGroup)
              diagnostic << selectedReductionGroup->resultGroup;
            else
              diagnostic << "none";
          }
          return mlir::failure();
        }
        MovementActionId action(matches.front()->id);
        fragmentsByAction[action].push_back(&fragment);
        auto graph = llvm::find_if(
            orderedPeerGraphs, [&](const PeerTransferGraphPlan *candidate) {
              return llvm::is_contained(candidate->actions, action);
            });
        if (graph != orderedPeerGraphs.end()) {
          if (fragment.sourceTile == strategy.destinationTile) {
            if (failureReason)
              *failureReason =
                  "selected peer action has identical physical endpoints";
            return mlir::failure();
          }
          fragment.kind = SpatialEdgeFragmentKind::Peer;
          fragment.communicationId =
              static_cast<int64_t>(graph - orderedPeerGraphs.begin());
          fragment.cardDDRResource.reset();
          hasPeer = true;
          continue;
        }
        if (fragment.sourceTile == strategy.destinationTile) {
          fragment.kind = SpatialEdgeFragmentKind::Resident;
          fragment.bytes = 0;
          fragment.communicationId = 0;
          fragment.payloadSlice = 0;
          fragment.cardDDRResource.reset();
          continue;
        }
        auto resourceId = cardDDRIds.find(action);
        if (resourceId == cardDDRIds.end()) {
          if (failureReason)
            *failureReason = "boundary DDR action has no unique card-shared "
                             "resource";
          return mlir::failure();
        }
        auto [resource, inserted] =
            usedCardDDRResources.try_emplace(resourceId->second, producerType);
        if (!inserted && resource->second != producerType) {
          if (failureReason)
            *failureReason =
                "boundary card DDR resource has inconsistent tensor types";
          return mlir::failure();
        }
        fragment.kind = SpatialEdgeFragmentKind::CardDDR;
        uint64_t elements = 1;
        for (int64_t size : fragment.sizes)
          elements = saturatingMultiply(elements, static_cast<uint64_t>(size));
        fragment.bytes = saturatingMultiply(
            elements, static_cast<uint64_t>(elementBits / 8));
        fragment.communicationId = 0;
        fragment.payloadSlice = 0;
        fragment.cardDDRResource = resourceId->second;
        hasCardDDR = true;
      }
      for (auto &[action, fragments] : fragmentsByAction) {
        (void)action;
        llvm::sort(fragments, [](const auto *lhs, const auto *rhs) {
          return std::tie(lhs->offsets, lhs->sizes) <
                 std::tie(rhs->offsets, rhs->sizes);
        });
        for (auto [payloadSlice, fragment] : llvm::enumerate(fragments))
          if (fragment->kind == SpatialEdgeFragmentKind::Peer)
            fragment->payloadSlice = static_cast<int64_t>(payloadSlice);
      }
      if (!hasCardDDR && !hasPeer) {
        if (strategy.fragments.size() == 1 &&
            !strategy.fragmentsDefineProducerDemand &&
            destination.reconstruction.steps.empty()) {
          strategy.action = SpatialEdgeAction::RegionCut;
          strategy.sourceTile = strategy.destinationTile;
          strategy.fragments.clear();
        } else {
          strategy.action = SpatialEdgeAction::PeerFragments;
        }
        strategy.cardDDRResource.reset();
      } else if (hasCardDDR && !hasPeer) {
        std::set<int64_t> resources;
        for (const SpatialEdgeFragment &fragment : strategy.fragments)
          if (fragment.cardDDRResource)
            resources.insert(*fragment.cardDDRResource);
        const bool uniformCardDDR =
            llvm::all_of(strategy.fragments, [](const auto &fragment) {
              return fragment.kind == SpatialEdgeFragmentKind::CardDDR;
            });
        if (resources.size() == 1 && uniformCardDDR) {
          strategy.action = SpatialEdgeAction::CardDDRTransfer;
          strategy.cardDDRResource = *resources.begin();
        } else {
          strategy.action = SpatialEdgeAction::PeerFragments;
          strategy.cardDDRResource.reset();
        }
      } else {
        strategy.action = SpatialEdgeAction::PeerFragments;
        strategy.cardDDRResource.reset();
      }
      movementResolved = true;
    }
    mapping.edgeStrategies.push_back(std::move(strategy));
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult
addCardEdgeCarriers(TileMapping &mapping, const SpatialAssignment &spatial,
                    const analysis::ExactDemandProof &demand,
                    const StructuredDAGAnalysis &dag,
                    const MovementPlan &movement,
                    llvm::ArrayRef<CoupledComponentResultMapping> components,
                    llvm::ArrayRef<StructuredOperationRootMapping> roots,
                    std::string *failureReason) {
  TileMapping prepared = mapping;
  std::set<std::pair<mlir::Operation *, unsigned>> componentResults;
  for (const CoupledComponentResultMapping &component : components)
    if (!component.operation ||
        !componentResults.insert({component.operation, component.result})
             .second) {
      if (failureReason)
        *failureReason =
            "coupled component result mapping is null or duplicated";
      return mlir::failure();
    }
  std::set<mlir::Operation *> rootOperations;
  for (const StructuredOperationRootMapping &root : roots)
    if (!root.operation || !rootOperations.insert(root.operation).second) {
      if (failureReason)
        *failureReason = "structured operation root mapping is null or "
                         "duplicated";
      return mlir::failure();
    }
  std::set<MovementActionId> peerActions;
  for (const PeerTransferGraphPlan &graph : movement.peerGraphs)
    peerActions.insert(graph.actions.begin(), graph.actions.end());
  std::vector<MovementActionId> cardDDRActions;
  for (const DDRBoundaryTransferPlan &transfer : movement.ddrTransfers) {
    const auto *result =
        std::get_if<ExecutionResultValueId>(&transfer.source.logicalValue);
    const auto *execution =
        result ? std::get_if<ExecutionInstanceId>(&result->execution) : nullptr;
    std::optional<TileId> sourceTile =
        execution ? std::optional<TileId>(std::visit(
                        [](const auto &source) { return source.work.tile; },
                        execution->source))
                  : std::nullopt;
    if (!peerActions.count(MovementActionId(transfer.id)) && sourceTile &&
        *sourceTile != transfer.id.destination.work.tile)
      cardDDRActions.emplace_back(transfer.id);
  }
  for (const ReductionGatherPlan &gather : movement.reductionGathers)
    if (!peerActions.count(MovementActionId(gather.id)))
      cardDDRActions.emplace_back(gather.id);
  llvm::sort(cardDDRActions);
  if (std::adjacent_find(cardDDRActions.begin(), cardDDRActions.end()) !=
      cardDDRActions.end()) {
    if (failureReason)
      *failureReason = "selected movement has a duplicate card DDR action";
    return mlir::failure();
  }
  std::map<MovementActionId, int64_t> cardDDRIds;
  for (auto [resourceId, action] : llvm::enumerate(cardDDRActions))
    cardDDRIds.emplace(action, static_cast<int64_t>(resourceId));
  std::map<int64_t, mlir::RankedTensorType> usedCardDDRResources;
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  mapping.edgeStrategies.clear();
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    const StructuredDAGNode *producer = dag.getNode(edge.producer);
    const StructuredDAGNode *consumer = dag.getNode(edge.consumer);
    const SemanticRootKey *producerRoot = view->getRoot(edge.producer);
    const SemanticRootKey *consumerRoot = view->getRoot(edge.consumer);
    const analysis::DependencyDemand *dependency =
        view->getDependency(edge.consumer, edge.consumerOperand);
    if (!producer || !consumer || !producerRoot || !consumerRoot ||
        !dependency ||
        mlir::failed(appendCarrier(
            prepared, edge, *producer, *consumer, *producerRoot, *consumerRoot,
            *dependency, demand, movement, components, roots, cardDDRIds,
            usedCardDDRResources, failureReason)))
      return mlir::failure();
  }
  prepared.cardDDRResources.clear();
  for (const auto &[resourceId, tensorType] : usedCardDDRResources)
    prepared.cardDDRResources.push_back({resourceId, tensorType});
  mapping = std::move(prepared);
  return mlir::success();
}

} // namespace wafer::compiler::detail
