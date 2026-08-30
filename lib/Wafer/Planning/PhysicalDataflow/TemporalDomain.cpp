//===- TemporalDomain.cpp - Live-operation temporal choices -----------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <set>

namespace wafer::compiler::detail {

struct TemporalDomain::Completion {
  TemporalSuccessorKind kind = TemporalSuccessorKind::CompilerBug;
  std::optional<TemporalChoice> choice;
  std::string detail;
};

namespace {

TemporalDomainResult failed(TemporalDomainFailureKind kind,
                            llvm::StringRef detail) {
  return {{}, TemporalDomainFailure{kind, detail.str()}};
}

bool isAcyclic(unsigned rank,
               llvm::ArrayRef<TemporalPrecedenceEdge> precedence) {
  llvm::SmallVector<unsigned, 8> indegree(rank, 0);
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 8> successors(rank);
  for (const TemporalPrecedenceEdge &edge : precedence) {
    ++indegree[edge.after];
    successors[edge.before].push_back(edge.after);
  }
  llvm::SmallVector<uint32_t, 8> ready;
  for (auto [iterator, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready.push_back(static_cast<uint32_t>(iterator));
  unsigned visited = 0;
  while (!ready.empty()) {
    const uint32_t current = ready.pop_back_val();
    ++visited;
    for (uint32_t next : successors[current])
      if (--indegree[next] == 0)
        ready.push_back(next);
  }
  return visited == rank;
}

llvm::SmallVector<uint8_t, 64>
getReachability(unsigned rank,
                llvm::ArrayRef<TemporalPrecedenceEdge> precedence) {
  llvm::SmallVector<uint8_t, 64> result(rank * rank, 0);
  for (const TemporalPrecedenceEdge &edge : precedence)
    result[edge.before * rank + edge.after] = 1;
  for (unsigned via = 0; via < rank; ++via)
    for (unsigned before = 0; before < rank; ++before)
      if (result[before * rank + via])
        for (unsigned after = 0; after < rank; ++after)
          result[before * rank + after] |= result[via * rank + after];
  return result;
}

llvm::SmallVector<uint32_t, 4>
getActiveIterators(const TemporalScopeDescriptor &scope,
                   llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<uint32_t, 4> active;
  for (auto [iterator, extent, size] :
       llvm::enumerate(scope.iterationExtents, sizes))
    if (size < extent)
      active.push_back(static_cast<uint32_t>(iterator));
  return active;
}

bool isTopologicalOrder(const TemporalScopeDescriptor &scope,
                        llvm::ArrayRef<uint32_t> active,
                        llvm::ArrayRef<uint32_t> order) {
  if (active.size() != order.size())
    return false;
  llvm::SmallVector<uint32_t, 4> sortedOrder(order.begin(), order.end());
  llvm::SmallVector<uint32_t, 4> sortedActive(active.begin(), active.end());
  llvm::sort(sortedOrder);
  llvm::sort(sortedActive);
  if (sortedOrder != sortedActive)
    return false;
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  llvm::SmallVector<unsigned, 8> position(rank, rank);
  for (auto [ordinal, iterator] : llvm::enumerate(order))
    position[iterator] = ordinal;
  for (uint32_t before : active)
    for (uint32_t after : active)
      if (reachability[before * rank + after] &&
          position[before] >= position[after])
        return false;
  return true;
}

bool containsScopeChoice(const TemporalScopeDescriptor &descriptor,
                         const TemporalScopeChoice &choice) {
  if (descriptor.operation != choice.operation ||
      choice.iteratorTileSizes.size() != descriptor.iterationExtents.size())
    return false;
  for (auto [size, extent, capability] :
       llvm::zip_equal(choice.iteratorTileSizes, descriptor.iterationExtents,
                       descriptor.iteratorCapabilities))
    if (size <= 0 || size > extent ||
        (capability == IteratorTilingCapability::FullExtentOnly &&
         size != extent))
      return false;
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(descriptor, choice.iteratorTileSizes);
  return isTopologicalOrder(descriptor, active, choice.loopOrder);
}

std::optional<llvm::SmallVector<uint32_t, 4>>
getFirstTopologicalOrder(const TemporalScopeDescriptor &scope,
                         llvm::ArrayRef<uint32_t> active,
                         llvm::ArrayRef<uint32_t> prefix = {}) {
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  llvm::SmallVector<uint32_t, 4> order(prefix.begin(), prefix.end());
  std::set<uint32_t> selected(prefix.begin(), prefix.end());
  if (selected.size() != prefix.size())
    return std::nullopt;
  while (order.size() != active.size()) {
    std::optional<uint32_t> next;
    for (uint32_t candidate : active) {
      if (selected.count(candidate))
        continue;
      bool available = true;
      for (uint32_t predecessor : active)
        if (!selected.count(predecessor) &&
            reachability[predecessor * rank + candidate]) {
          available = false;
          break;
        }
      if (available) {
        next = candidate;
        break;
      }
    }
    if (!next)
      return std::nullopt;
    selected.insert(*next);
    order.push_back(*next);
  }
  return order;
}

std::optional<llvm::SmallVector<uint32_t, 4>>
getNextTopologicalOrder(const TemporalScopeDescriptor &scope,
                        llvm::ArrayRef<uint32_t> active,
                        llvm::ArrayRef<uint32_t> current) {
  if (!isTopologicalOrder(scope, active, current))
    return std::nullopt;
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  for (size_t reverse = 0; reverse < current.size(); ++reverse) {
    const size_t position = current.size() - reverse - 1;
    llvm::SmallVector<uint32_t, 4> prefix(current.begin(),
                                          current.begin() + position);
    std::set<uint32_t> selected(prefix.begin(), prefix.end());
    for (uint32_t candidate : active) {
      if (candidate <= current[position] || selected.count(candidate))
        continue;
      bool available = true;
      for (uint32_t predecessor : active)
        if (!selected.count(predecessor) && predecessor != candidate &&
            reachability[predecessor * rank + candidate]) {
          available = false;
          break;
        }
      if (!available)
        continue;
      prefix.push_back(candidate);
      if (auto completed = getFirstTopologicalOrder(scope, active, prefix))
        return completed;
      prefix.pop_back();
    }
  }
  return std::nullopt;
}

bool isTemporalCandidate(mlir::Operation *operation) {
  if (mlir::isa<LinalgExtOnlineAttentionOp>(operation))
    return true;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  return linalg && linalg.hasPureTensorSemantics() &&
         operation->getNumResults() != 0 &&
         !mlir::isa<mlir::linalg::FillOp>(operation);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticIterationExtents(mlir::Operation *operation) {
  llvm::SmallVector<int64_t, 4> extents;
  if (auto online = mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operation)) {
    llvm::append_range(extents, online.getStaticLoopRanges());
  } else if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation)) {
    llvm::append_range(extents, linalg.getStaticLoopRanges());
  } else {
    return mlir::failure();
  }
  if (extents.empty() || llvm::any_of(extents, [](int64_t extent) {
        return extent <= 0 || mlir::ShapedType::isDynamic(extent);
      }))
    return mlir::failure();
  return extents;
}

mlir::FailureOr<llvm::SmallVector<IteratorTilingCapability, 4>>
getIteratorCapabilities(mlir::Operation *operation, unsigned rank) {
  llvm::SmallVector<IteratorTilingCapability, 4> capabilities(
      rank, IteratorTilingCapability::Tileable);
  if (auto online = mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operation)) {
    mlir::FailureOr<AttentionIterationRoles> roles = online.getIterationRoles();
    if (mlir::failed(roles))
      return mlir::failure();
    for (unsigned dimension : roles->queryKeyReduction) {
      if (dimension >= rank)
        return mlir::failure();
      capabilities[dimension] = IteratorTilingCapability::FullExtentOnly;
    }
  }
  return capabilities;
}

mlir::FailureOr<TemporalScopeDescriptor>
buildDescriptor(mlir::Operation *operation) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling)
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> extents =
      getStaticIterationExtents(operation);
  if (mlir::failed(extents) ||
      tiling.getLoopIteratorTypes().size() != extents->size())
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<IteratorTilingCapability, 4>> capabilities =
      getIteratorCapabilities(operation, extents->size());
  if (mlir::failed(capabilities))
    return mlir::failure();
  TemporalScopeDescriptor descriptor;
  descriptor.operation = operation;
  descriptor.iterationExtents = std::move(*extents);
  descriptor.iteratorCapabilities = std::move(*capabilities);
  return descriptor;
}

mlir::FailureOr<mlir::AffineMap> getProducerResultMap(mlir::OpResult result) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(result.getOwner());
  if (!linalg || result.getResultNumber() >= linalg.getNumDpsInits())
    return mlir::failure();
  return linalg.getMatchingIndexingMap(
      linalg.getDpsInitOperand(result.getResultNumber()));
}

mlir::FailureOr<mlir::AffineMap>
getConsumerOperandMap(mlir::OpOperand &operand) {
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operand.getOwner()))
    return linalg.getMatchingIndexingMap(&operand);
  if (auto online =
          mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operand.getOwner())) {
    llvm::SmallVector<mlir::AffineMap, 8> maps = online.getIndexingMapsArray();
    if (operand.getOperandNumber() >= maps.size())
      return mlir::failure();
    return maps[operand.getOperandNumber()];
  }
  return mlir::failure();
}

llvm::SmallBitVector getUsedDimensions(mlir::AffineMap map) {
  llvm::SmallBitVector result(map.getNumDims(), false);
  for (mlir::AffineExpr expression : map.getResults())
    if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression))
      result.set(dimension.getPosition());
  return result;
}

std::optional<mlir::OpOperand *> getOnlyOperationUse(mlir::Operation *op) {
  mlir::OpOperand *onlyUse = nullptr;
  for (mlir::OpResult result : op->getResults())
    for (mlir::OpOperand &use : result.getUses()) {
      if (onlyUse)
        return std::nullopt;
      onlyUse = &use;
    }
  return onlyUse ? std::optional<mlir::OpOperand *>(onlyUse) : std::nullopt;
}

} // namespace

TemporalFusionQueryResult
queryTemporalProducerFusion(mlir::OpResult producer,
                            mlir::OpOperand &consumerOperand) {
  mlir::Operation *producerOperation = producer.getOwner();
  mlir::Operation *consumerOperation = consumerOperand.getOwner();
  if (!producerOperation || !consumerOperation ||
      consumerOperand.get() != producer)
    return {TemporalFusionQueryKind::BrokenContract,
            "producer result and consumer operand do not form a current SSA "
            "edge"};
  if (mlir::isa<LinalgExtOnlineAttentionOp>(producerOperation))
    return {TemporalFusionQueryKind::NonUnique,
            "online attention state remains an independent traversal root"};
  if (producerOperation->getBlock() != consumerOperation->getBlock() ||
      producerOperation->getParentOfType<TileRegionOp>() !=
          consumerOperation->getParentOfType<TileRegionOp>() ||
      !producerOperation->isBeforeInBlock(consumerOperation))
    return {TemporalFusionQueryKind::NonUnique,
            "producer and consumer are not one ordered same-Region edge"};
  std::optional<mlir::OpOperand *> onlyUse =
      getOnlyOperationUse(producerOperation);
  if (!onlyUse || *onlyUse != &consumerOperand)
    return {TemporalFusionQueryKind::NonUnique,
            "producer has multiple current uses or results"};
  if (!mlir::isMemoryEffectFree(producerOperation))
    return {TemporalFusionQueryKind::NonUnique,
            "effectful producer cannot be implicitly replicated"};
  auto consumerDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumerOperation);
  if (consumerDps && consumerDps.isDpsInit(&consumerOperand))
    return {TemporalFusionQueryKind::NonUnique,
            "destination producer remains outside reduction traversal"};
  if (!isTemporalCandidate(producerOperation) ||
      !isTemporalCandidate(consumerOperation))
    return {TemporalFusionQueryKind::Unsupported,
            "producer or consumer has no supported static temporal contract"};

  mlir::FailureOr<TemporalScopeDescriptor> producerDescriptor =
      buildDescriptor(producerOperation);
  mlir::FailureOr<TemporalScopeDescriptor> consumerDescriptor =
      buildDescriptor(consumerOperation);
  mlir::FailureOr<mlir::AffineMap> producerMap = getProducerResultMap(producer);
  mlir::FailureOr<mlir::AffineMap> consumerMap =
      getConsumerOperandMap(consumerOperand);
  if (mlir::failed(producerDescriptor) || mlir::failed(consumerDescriptor) ||
      mlir::failed(producerMap) || mlir::failed(consumerMap))
    return {TemporalFusionQueryKind::Unsupported,
            "current interfaces cannot expose an exact tile relation"};
  if (producerMap->getNumSymbols() != 0 || consumerMap->getNumSymbols() != 0 ||
      !producerMap->isProjectedPermutation() ||
      !consumerMap->isProjectedPermutation())
    return {TemporalFusionQueryKind::Unsupported,
            "fusion requires symbol-free projected-permutation maps"};
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(producer.getType());
  auto consumerType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumerOperand.get().getType());
  if (!producerType || !consumerType)
    return {TemporalFusionQueryKind::Unsupported,
            "fusion requires ranked tensor endpoints"};
  if (producerMap->getNumDims() !=
          producerDescriptor->iterationExtents.size() ||
      consumerMap->getNumDims() !=
          consumerDescriptor->iterationExtents.size() ||
      producerMap->getNumResults() != producerType.getRank() ||
      consumerMap->getNumResults() != consumerType.getRank())
    return {TemporalFusionQueryKind::BrokenContract,
            "fusion indexing map ranks do not match current tensor types"};

  llvm::SmallBitVector consumerDimensions = getUsedDimensions(*consumerMap);
  for (auto [dimension, capability] :
       llvm::enumerate(consumerDescriptor->iteratorCapabilities))
    if (capability == IteratorTilingCapability::Tileable &&
        !consumerDimensions.test(dimension))
      return {TemporalFusionQueryKind::NonUnique,
              "consumer tiling can request the same producer tile more than "
              "once"};

  llvm::SmallBitVector producerDimensions = getUsedDimensions(*producerMap);
  auto producerTiling = mlir::cast<mlir::TilingInterface>(producerOperation);
  llvm::SmallVector<mlir::utils::IteratorType, 4> producerIterators =
      producerTiling.getLoopIteratorTypes();
  for (unsigned dimension = 0; dimension < producerDimensions.size();
       ++dimension)
    if (!producerDimensions.test(dimension) &&
        producerDescriptor->iterationExtents[dimension] != 1 &&
        producerIterators[dimension] != mlir::utils::IteratorType::reduction)
      return {TemporalFusionQueryKind::NonUnique,
              "producer result does not uniquely determine a parallel "
              "iterator"};
  return {TemporalFusionQueryKind::ExactDerived, {}};
}

TemporalScopeChoice
TemporalDomain::getFirstScopeChoice(const TemporalScopeDescriptor &scope) {
  TemporalScopeChoice result;
  result.operation = scope.operation;
  result.iteratorTileSizes = scope.iterationExtents;
  return result;
}

bool TemporalDomain::advanceScopeChoice(const TemporalScopeDescriptor &scope,
                                        TemporalScopeChoice &choice) {
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(scope, choice.iteratorTileSizes);
  if (auto next = getNextTopologicalOrder(scope, active, choice.loopOrder)) {
    choice.loopOrder = std::move(*next);
    return true;
  }
  for (size_t reverse = 0; reverse < scope.iterationExtents.size(); ++reverse) {
    const size_t dimension = scope.iterationExtents.size() - reverse - 1;
    if (scope.iteratorCapabilities[dimension] ==
            IteratorTilingCapability::FullExtentOnly ||
        choice.iteratorTileSizes[dimension] <= 1)
      continue;
    --choice.iteratorTileSizes[dimension];
    for (size_t reset = dimension + 1; reset < scope.iterationExtents.size();
         ++reset)
      choice.iteratorTileSizes[reset] = scope.iterationExtents[reset];
    active = getActiveIterators(scope, choice.iteratorTileSizes);
    auto first = getFirstTopologicalOrder(scope, active);
    if (!first)
      return false;
    choice.loopOrder = std::move(*first);
    return true;
  }
  return false;
}

TemporalDomain::Completion TemporalDomain::completeChoice(
    llvm::ArrayRef<TemporalScopeChoice> prefix) const {
  Completion completion;
  completion.kind = TemporalSuccessorKind::Choice;
  if (prefix.size() > scopes.size()) {
    completion.kind = TemporalSuccessorKind::CompilerBug;
    completion.detail = "temporal prefix has an unexpected extra scope";
    return completion;
  }
  completion.choice.emplace();
  for (auto [index, descriptor] : llvm::enumerate(scopes)) {
    if (index < prefix.size()) {
      if (!containsScopeChoice(descriptor, prefix[index])) {
        completion.kind = TemporalSuccessorKind::CompilerBug;
        completion.detail =
            "temporal prefix does not match its live scope descriptor";
        completion.choice.reset();
        return completion;
      }
      completion.choice->scopes.push_back(prefix[index]);
    } else {
      completion.choice->scopes.push_back(getFirstScopeChoice(descriptor));
    }
  }
  return completion;
}

TemporalSuccessor TemporalDomain::getFirstChoice() const {
  return completePrefix(TemporalChoice{});
}

TemporalSuccessor
TemporalDomain::completePrefix(const TemporalChoice &prefix) const {
  Completion completed = completeChoice(prefix.scopes);
  if (completed.kind != TemporalSuccessorKind::Choice || !completed.choice)
    return {completed.kind, {}, {}, std::move(completed.detail)};
  TemporalCursor cursor;
  cursor.choice = *completed.choice;
  return {TemporalSuccessorKind::Choice, std::move(completed.choice),
          std::move(cursor)};
}

TemporalSuccessor
TemporalDomain::getNextChoice(const TemporalCursor &cursor) const {
  Completion current = completeChoice(cursor.choice.scopes);
  if (current.kind != TemporalSuccessorKind::Choice || !current.choice ||
      !(*current.choice == cursor.choice))
    return {TemporalSuccessorKind::CompilerBug,
            {},
            {},
            "temporal cursor is stale for the current operation domain"};
  for (size_t reverse = 0; reverse < scopes.size(); ++reverse) {
    const size_t index = scopes.size() - reverse - 1;
    TemporalChoice prefix;
    prefix.scopes.assign(cursor.choice.scopes.begin(),
                         cursor.choice.scopes.begin() + index + 1);
    if (!advanceScopeChoice(scopes[index], prefix.scopes.back()))
      continue;
    Completion next = completeChoice(prefix.scopes);
    if (next.kind != TemporalSuccessorKind::Choice || !next.choice)
      return {next.kind, {}, {}, std::move(next.detail)};
    TemporalCursor nextCursor;
    nextCursor.choice = *next.choice;
    return {TemporalSuccessorKind::Choice, std::move(next.choice),
            std::move(nextCursor)};
  }
  return {TemporalSuccessorKind::End};
}

bool TemporalDomain::contains(const TemporalChoice &choice) const {
  Completion completed = completeChoice(choice.scopes);
  return completed.kind == TemporalSuccessorKind::Choice && completed.choice &&
         *completed.choice == choice;
}

TemporalDomainResult buildTemporalDomain(TileRegionOp region) {
  if (!region || mlir::failed(mlir::verify(region)))
    return failed(TemporalDomainFailureKind::BrokenContract,
                  "temporal domain requires a verifier-valid TileRegion");
  mlir::Block &body = region.getBody().front();
  llvm::SmallVector<mlir::Operation *, 16> candidates;
  for (mlir::Operation &operation : body.without_terminator())
    if (isTemporalCandidate(&operation))
      candidates.push_back(&operation);

  llvm::SmallPtrSet<mlir::Operation *, 16> candidateSet(candidates.begin(),
                                                        candidates.end());
  llvm::SmallPtrSet<mlir::Operation *, 16> derivedProducers;
  for (mlir::Operation *producer : candidates) {
    if (mlir::isa<LinalgExtOnlineAttentionOp>(producer))
      continue;
    std::optional<mlir::OpOperand *> onlyUse = getOnlyOperationUse(producer);
    if (!onlyUse || !candidateSet.contains((*onlyUse)->getOwner()))
      continue;
    auto result = mlir::dyn_cast<mlir::OpResult>((*onlyUse)->get());
    if (!result || result.getOwner() != producer)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal producer use is not owned by its current result");
    TemporalFusionQueryResult query =
        queryTemporalProducerFusion(result, **onlyUse);
    if (query.kind == TemporalFusionQueryKind::BrokenContract)
      return failed(TemporalDomainFailureKind::BrokenContract, query.detail);
    if (query.kind == TemporalFusionQueryKind::ExactDerived)
      derivedProducers.insert(producer);
  }

  std::vector<TemporalScopeDescriptor> scopes;
  for (mlir::Operation *operation : candidates) {
    if (derivedProducers.contains(operation))
      continue;
    mlir::FailureOr<TemporalScopeDescriptor> descriptor =
        buildDescriptor(operation);
    if (mlir::failed(descriptor))
      continue;
    scopes.push_back(std::move(*descriptor));
  }
  return {TemporalDomain(region, std::move(scopes)), {}};
}

mlir::FailureOr<TemporalIntervalChildren>
splitTemporalSizeInterval(TemporalSizeInterval interval,
                          std::optional<int64_t> proposal,
                          std::string *failureReason) {
  if (interval.lower <= 0 || interval.upper < interval.lower ||
      (proposal &&
       (*proposal < interval.lower || *proposal > interval.upper))) {
    if (failureReason)
      *failureReason = "temporal size interval or proposal is invalid";
    return mlir::failure();
  }
  const int64_t point = proposal.value_or(static_cast<int64_t>(
      static_cast<__int128>(interval.lower) +
      (static_cast<__int128>(interval.upper) - interval.lower) / 2));
  TemporalIntervalChildren result;
  result.singleton = {point, point};
  if (point < interval.upper)
    result.above = TemporalSizeInterval{point + 1, interval.upper};
  if (point > interval.lower)
    result.below = TemporalSizeInterval{interval.lower, point - 1};
  return result;
}

mlir::FailureOr<llvm::SmallVector<uint32_t, 4>>
buildFirstTemporalLoopOrder(llvm::ArrayRef<int64_t> iteratorExtents,
                            llvm::ArrayRef<int64_t> iteratorTileSizes,
                            llvm::ArrayRef<TemporalPrecedenceEdge> precedence,
                            std::string *failureReason) {
  if (iteratorExtents.size() != iteratorTileSizes.size() ||
      llvm::any_of(llvm::zip_equal(iteratorExtents, iteratorTileSizes),
                   [](auto values) {
                     auto [extent, size] = values;
                     return extent <= 0 || size <= 0 || size > extent;
                   })) {
    if (failureReason)
      *failureReason = "temporal size vector is outside its iterator domain";
    return mlir::failure();
  }
  TemporalScopeDescriptor descriptor;
  descriptor.iterationExtents.assign(iteratorExtents.begin(),
                                     iteratorExtents.end());
  descriptor.precedence.assign(precedence.begin(), precedence.end());
  llvm::sort(descriptor.precedence);
  for (const TemporalPrecedenceEdge &edge : descriptor.precedence)
    if (edge.before >= iteratorExtents.size() ||
        edge.after >= iteratorExtents.size() || edge.before == edge.after) {
      if (failureReason)
        *failureReason = "temporal precedence edge is outside iterator rank";
      return mlir::failure();
    }
  if (!isAcyclic(iteratorExtents.size(), descriptor.precedence)) {
    if (failureReason)
      *failureReason = "temporal precedence graph has a cycle";
    return mlir::failure();
  }
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(descriptor, iteratorTileSizes);
  auto order = getFirstTopologicalOrder(descriptor, active);
  if (!order) {
    if (failureReason)
      *failureReason = "temporal precedence graph has no linear extension";
    return mlir::failure();
  }
  return std::move(*order);
}

} // namespace wafer::compiler::detail
