//===- TemporalDomain.cpp - Live-operation temporal choices -----------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
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
  for (uint32_t dimension : descriptor.exactReshapeDimensions)
    if (dimension >= choice.iteratorTileSizes.size() ||
        (choice.iteratorTileSizes[dimension] !=
             descriptor.iterationExtents[dimension] &&
         choice.iteratorTileSizes[dimension] <=
             descriptor.iterationExtents[dimension] / 2))
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
  if (!operation || operation->getNumResults() == 0)
    return false;
  // Reject unsupported shape/type boundaries before querying external tiling
  // models. A dynamic Tensor domain query may otherwise need to create IR.
  if (!llvm::all_of(operation->getResultTypes(),
                    [](mlir::Type type) {
                      auto tensor =
                          mlir::dyn_cast<mlir::RankedTensorType>(type);
                      return tensor && tensor.hasStaticShape();
                    }) ||
      llvm::any_of(operation->getOperandTypes(), [](mlir::Type type) {
        if (mlir::isa<mlir::BaseMemRefType>(type))
          return true;
        auto tensor = mlir::dyn_cast<mlir::TensorType>(type);
        return tensor && (!tensor.hasRank() || !tensor.hasStaticShape());
      }))
    return false;
  if (!mlir::isa<mlir::TilingInterface>(operation) ||
      !mlir::isMemoryEffectFree(operation) ||
      mlir::isa<WaferLinalgExtCollectiveOpInterface>(operation) ||
      mlir::isa<mlir::linalg::FillOp>(operation))
    return false;
  if (auto pad = mlir::dyn_cast<mlir::tensor::PadOp>(operation))
    if (!pad.getConstantPaddingValue())
      return false;
  return true;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticIterationExtents(mlir::Operation *operation) {
  llvm::SmallVector<int64_t, 4> extents;
  if (auto online = mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operation)) {
    llvm::append_range(extents, online.getStaticLoopRanges());
  } else if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation)) {
    llvm::append_range(extents, linalg.getStaticLoopRanges());
  } else {
    auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
    if (!tiling)
      return mlir::failure();
    mlir::OpBuilder builder(operation->getContext());
    llvm::SmallVector<mlir::Range> ranges = tiling.getIterationDomain(builder);
    for (const mlir::Range &range : ranges) {
      std::optional<int64_t> offset = mlir::getConstantIntValue(range.offset);
      std::optional<int64_t> size = mlir::getConstantIntValue(range.size);
      std::optional<int64_t> stride = mlir::getConstantIntValue(range.stride);
      if (!offset || *offset != 0 || !size || !stride || *stride != 1)
        return mlir::failure();
      extents.push_back(*size);
    }
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
  if (auto coupled =
          mlir::dyn_cast<WaferCoupledReductionOpInterface>(operation)) {
    auto description = coupled.getCoupledReductionDescription();
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
    if (!tiling)
      return mlir::failure();
    auto iterators = tiling.getLoopIteratorTypes();
    if (iterators.size() != rank)
      return mlir::failure();
    for (unsigned dimension = 0; dimension < rank; ++dimension)
      if (iterators[dimension] == mlir::utils::IteratorType::parallel &&
          description.hasReplicatedComponent(dimension))
        capabilities[dimension] = IteratorTilingCapability::FullExtentOnly;
  }
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
  if (auto pad = mlir::dyn_cast<mlir::tensor::PadOp>(operation)) {
    llvm::SmallVector<mlir::OpFoldResult, 4> lowPadding = pad.getMixedLowPad();
    llvm::SmallVector<mlir::OpFoldResult, 4> highPadding =
        pad.getMixedHighPad();
    if (lowPadding.size() != rank || highPadding.size() != rank)
      return mlir::failure();
    for (auto [dimension, low, high] :
         llvm::enumerate(lowPadding, highPadding)) {
      std::optional<int64_t> lowValue = mlir::getConstantIntValue(low);
      std::optional<int64_t> highValue = mlir::getConstantIntValue(high);
      if (!lowValue || !highValue)
        return mlir::failure();
      if (*lowValue != 0 || *highValue != 0)
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

bool isViewTransparentKind(TensorIndexingTransformKind kind) {
  return kind == TensorIndexingTransformKind::Cast ||
         kind == TensorIndexingTransformKind::ExtractSlice ||
         kind == TensorIndexingTransformKind::ExpandShape ||
         kind == TensorIndexingTransformKind::CollapseShape;
}

bool matchOffsetDimension(mlir::AffineExpr expression,
                          std::optional<unsigned> &dimension, int64_t &offset) {
  offset = 0;
  if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
    dimension = dim.getPosition();
    return true;
  }
  if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
    dimension.reset();
    offset = constant.getValue();
    return true;
  }
  auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression);
  if (!binary || binary.getKind() != mlir::AffineExprKind::Add)
    return false;
  auto lhsDim = mlir::dyn_cast<mlir::AffineDimExpr>(binary.getLHS());
  auto rhsDim = mlir::dyn_cast<mlir::AffineDimExpr>(binary.getRHS());
  auto lhsConstant = mlir::dyn_cast<mlir::AffineConstantExpr>(binary.getLHS());
  auto rhsConstant = mlir::dyn_cast<mlir::AffineConstantExpr>(binary.getRHS());
  if (lhsDim && rhsConstant) {
    dimension = lhsDim.getPosition();
    offset = rhsConstant.getValue();
    return true;
  }
  if (rhsDim && lhsConstant) {
    dimension = rhsDim.getPosition();
    offset = lhsConstant.getValue();
    return true;
  }
  return false;
}

bool isDenseOffsetProjection(
    mlir::AffineMap map,
    llvm::SmallVectorImpl<TemporalViewDimensionMapping> *mappings = nullptr) {
  if (!map || map.getNumSymbols() != 0)
    return false;
  llvm::SmallBitVector seenDimensions(map.getNumDims());
  for (mlir::AffineExpr expression : map.getResults()) {
    std::optional<unsigned> dimension;
    int64_t offset = 0;
    if (!matchOffsetDimension(expression, dimension, offset))
      return false;
    if (dimension &&
        (*dimension >= map.getNumDims() || !seenDimensions.test(*dimension)))
      seenDimensions.set(*dimension);
    else if (dimension)
      return false;
    if (mappings)
      mappings->push_back(
          {dimension ? static_cast<int32_t>(*dimension) : -1, offset});
  }
  return true;
}

} // namespace

static TemporalFusionQueryResult
relationFailure(analysis::IndexRelationStatus status, llvm::StringRef detail) {
  auto kind = status == analysis::IndexRelationStatus::Invalid
                  ? TemporalFusionQueryKind::BrokenContract
              : status == analysis::IndexRelationStatus::ResourceExhausted
                  ? TemporalFusionQueryKind::Indeterminate
                  : TemporalFusionQueryKind::Unsupported;
  return {kind, detail.str()};
}

static analysis::IndexRelationQueryResult
queryProducerParallelFiber(mlir::OpResult producer,
                           const TemporalScopeDescriptor &descriptor,
                           const analysis::IndexRelationLimits &limits) {
  auto outputMap = analysis::getStructuredResultMap(producer);
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(producer.getType());
  if (mlir::failed(outputMap) || !type)
    return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
            "producer does not expose a result indexing relation"};
  auto iterators = mlir::cast<mlir::TilingInterface>(producer.getOwner())
                       .getLoopIteratorTypes();
  llvm::SmallVector<mlir::AffineExpr, 6> parallel;
  llvm::SmallVector<int64_t, 6> shape;
  for (auto [axis, kind] : llvm::enumerate(iterators)) {
    if (kind == mlir::utils::IteratorType::reduction)
      continue;
    parallel.push_back(mlir::getAffineDimExpr(axis, producer.getContext()));
    shape.push_back(descriptor.iterationExtents[axis]);
  }
  auto fiber = analysis::IndexRelation::fromCommonIterationDomain(
      *outputMap, type.getShape(),
      mlir::AffineMap::get(iterators.size(), 0, parallel,
                           producer.getContext()),
      shape, descriptor.iterationExtents, limits);
  if (!fiber.isExact())
    return {fiber.status, std::nullopt, fiber.reason};
  return fiber.get()->isFunctional(limits);
}

static analysis::IndexRelationQueryResult
queryConsumerRequestUniqueness(mlir::OpOperand &operand,
                               const TemporalScopeDescriptor &descriptor,
                               const analysis::IndexRelationLimits &limits) {
  auto relation = analysis::deriveIterationOperandRelation(
      operand, descriptor.iterationExtents, limits);
  if (!relation.isExact())
    return {relation.status, std::nullopt, relation.reason};
  auto type = mlir::cast<mlir::RankedTensorType>(operand.get().getType());
  llvm::SmallVector<int64_t, 4> sizes = descriptor.iterationExtents;
  for (auto [axis, capability] :
       llvm::enumerate(descriptor.iteratorCapabilities))
    if (capability == IteratorTilingCapability::Tileable)
      sizes[axis] = 1;
  auto image = relation.get()->getRectangularTileImage(
      operand.getOwner()->getContext(), descriptor.iterationExtents,
      type.getShape(), sizes, limits);
  if (!image.isExact())
    return {image.status, std::nullopt, image.reason};
  for (uint32_t axis : image.image->invariantDimensions)
    if (descriptor.iteratorCapabilities[axis] ==
            IteratorTilingCapability::Tileable &&
        descriptor.iterationExtents[axis] > 1)
      return {analysis::IndexRelationStatus::Exact, false,
              "operand demand is invariant over a consumer iterator"};
  return {analysis::IndexRelationStatus::Exact,
          image.image->distinctTilesDisjoint,
          {}};
}

namespace {

std::optional<mlir::AffineMap>
getViewProjection(const analysis::IndexRelation &relation,
                  llvm::ArrayRef<int64_t> shape, mlir::MLIRContext *context) {
  auto map = relation.getProjectedAffineMap(context);
  if (!map)
    return std::nullopt;
  llvm::SmallVector<mlir::AffineExpr, 6> dimensions;
  for (size_t axis = 0; axis < shape.size(); ++axis)
    dimensions.push_back(shape[axis] == 1
                             ? mlir::getAffineConstantExpr(0, context)
                             : mlir::getAffineDimExpr(axis, context));
  return mlir::simplifyAffineMap(
      map->replaceDimsAndSymbols(dimensions, {}, shape.size(), 0));
}

void appendDimensions(mlir::AffineExpr expression,
                      llvm::SmallVectorImpl<uint32_t> &dimensions) {
  expression.walk([&](mlir::AffineExpr nested) {
    if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(nested))
      if (!llvm::is_contained(dimensions, dim.getPosition()))
        dimensions.push_back(dim.getPosition());
  });
  llvm::sort(dimensions);
}

// This preflight describes local representation, after the SSA relation has
// been built. It neither discovers uses nor gives direct/view/shared their own
// legality rules. The projected result requirement belongs to the pinned
// producer TilingInterface helper.
TemporalFusionQueryResult
describeFusionUse(mlir::OpResult producer, TemporalFusionUse &use,
                  const TemporalScopeDescriptor &descriptor,
                  const analysis::IndexRelationLimits &limits) {
  auto producerType = mlir::cast<mlir::RankedTensorType>(producer.getType());
  auto viewType =
      mlir::cast<mlir::RankedTensorType>(use.consumerValue.getType());
  auto consumerMap = analysis::getStructuredOperandMap(*use.operand);
  auto projection = getViewProjection(*use.viewToProducer, viewType.getShape(),
                                      producer.getContext());
  bool localProjection =
      projection &&
      isDenseOffsetProjection(*projection, &use.producerDimensions);
  auto accessProjection =
      getViewProjection(*use.iterationToProducer, descriptor.iterationExtents,
                        producer.getContext());
  auto pack = mlir::dyn_cast<mlir::tensor::PackOp>(use.operand->getOwner());
  if (pack && use.operand->getOperandNumber() == 0 && localProjection) {
    // The pack source relation is one-to-many but injective: a source point
    // belongs to exactly one outer iteration. Its interface emits these slices.
    auto unique = use.iterationToProducer->isInjective(limits);
    if (!unique.isProvenTrue())
      return relationFailure(unique.status, unique.reason);
    use.representation = TemporalTileRepresentation::ResultSlice;
  } else if (localProjection && accessProjection &&
             isDenseOffsetProjection(*accessProjection) &&
             queryConsumerRequestUniqueness(*use.operand, descriptor, limits)
                 .isProvenTrue()) {
    use.representation = TemporalTileRepresentation::ResultSlice;
  } else if (localProjection) {
    auto full = use.iterationToProducer->getRectangularTileImage(
        producer.getContext(), descriptor.iterationExtents,
        producerType.getShape(), descriptor.iterationExtents, limits);
    if (!full.isExact())
      return relationFailure(full.status, full.reason);
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
        use.operand->getOwner());
    if (!dps ||
        dps.getNumDpsInits() != use.operand->getOwner()->getNumResults())
      return {
          TemporalFusionQueryKind::Unsupported,
          "common-loop generator requires complete DPS result destinations"};
    use.representation = TemporalTileRepresentation::RectangularImage;
  } else if (use.viewToProducer->hasCanonicalRowMajorReshapeConstruction()) {
    auto reassociation =
        mlir::getReassociationIndicesForReshape(producerType, viewType);
    if (!reassociation || mlir::failed(consumerMap))
      return {TemporalFusionQueryKind::Unsupported,
              "local reshape has no supported reassociation generator"};
    use.representation = TemporalTileRepresentation::ReshapePieces;
    llvm::SmallVector<unsigned, 4> fullViewDimensions;
    if (producerType.getRank() > viewType.getRank()) {
      for (auto [axis, group] : llvm::enumerate(*reassociation))
        if (group.size() > 1)
          fullViewDimensions.push_back(axis);
    } else {
      for (auto group : *reassociation)
        if (group.size() > 1)
          llvm::append_range(fullViewDimensions, group);
    }
    for (unsigned dimension : fullViewDimensions)
      appendDimensions(consumerMap->getResult(dimension),
                       use.reshapeDimensions);
    if (!use.viewToProducer->isInjective(limits).isProvenTrue())
      return {TemporalFusionQueryKind::NonUnique,
              "local reshape cannot share distinct producer requests"};
  } else {
    return {TemporalFusionQueryKind::Unsupported,
            "exact view relation has no supported local tile representation"};
  }

  llvm::SmallVector<unsigned, 4> fullResultAxes;
  if (auto pad = mlir::dyn_cast<mlir::tensor::PadOp>(producer.getOwner())) {
    for (auto [axis, low, high] :
         llvm::enumerate(pad.getMixedLowPad(), pad.getMixedHighPad())) {
      auto l = mlir::getConstantIntValue(low),
           h = mlir::getConstantIntValue(high);
      if (!l || !h)
        return {TemporalFusionQueryKind::Unsupported, "padding must be static"};
      if (*l || *h)
        fullResultAxes.push_back(axis);
    }
  }
  if (auto packed = mlir::dyn_cast<mlir::tensor::PackOp>(producer.getOwner()))
    for (unsigned axis = packed.getSourceRank(); axis < producerType.getRank();
         ++axis)
      fullResultAxes.push_back(axis);
  for (unsigned axis : fullResultAxes) {
    if (!accessProjection)
      return {TemporalFusionQueryKind::Unsupported,
              "result-tile generator cannot express the required full axis"};
    appendDimensions(accessProjection->getResult(axis),
                     use.fullExtentDimensions);
  }
  return {TemporalFusionQueryKind::ExactDerived, {}};
}

} // namespace

TemporalFusionQueryResult
queryTemporalFusion(mlir::OpResult producer,
                    const analysis::IndexRelationLimits &limits) {
  if (!producer)
    return {TemporalFusionQueryKind::BrokenContract,
            "fusion has no current producer result"};
  auto *operation = producer.getOwner();
  if (operation->getNumResults() != 1)
    return {
        TemporalFusionQueryKind::NonUnique,
        "multi-result producer requires whole-result state materialization"};
  if (!isTemporalCandidate(operation))
    return {TemporalFusionQueryKind::Unsupported,
            "producer has no static pure tensor tiling contract"};
  auto producerDescriptor = buildDescriptor(operation);
  if (mlir::failed(producerDescriptor))
    return {TemporalFusionQueryKind::Unsupported,
            "producer iteration domain is unavailable"};
  bool tensorResultTiling = mlir::isa<mlir::tensor::PadOp, mlir::tensor::PackOp,
                                      mlir::tensor::UnPackOp>(operation);
  if (!tensorResultTiling) {
    auto resultMap = analysis::getStructuredResultMap(producer);
    if (mlir::failed(resultMap) || !resultMap->isProjectedPermutation())
      return {TemporalFusionQueryKind::Unsupported,
              "producer result-tile interface requires a projected result map"};
    auto fiber =
        queryProducerParallelFiber(producer, *producerDescriptor, limits);
    if (fiber.status != analysis::IndexRelationStatus::Exact)
      return relationFailure(fiber.status, fiber.reason);
    if (!fiber.isProvenTrue())
      return {
          TemporalFusionQueryKind::NonUnique,
          "producer result does not determine its parallel iteration fiber"};
  }
  auto producerType = mlir::cast<mlir::RankedTensorType>(producer.getType());
  auto identity =
      analysis::IndexRelation::identity(producerType.getShape(), limits);
  if (!identity.isExact())
    return relationFailure(identity.status, identity.reason);
  struct Path {
    mlir::Value value;
    analysis::IndexRelation relation;
  };
  llvm::SmallVector<Path, 8> pending;
  pending.push_back({producer, std::move(*identity.get())});
  llvm::SmallPtrSet<mlir::Operation *, 16> visitedViews;
  TemporalFusion fusion;
  fusion.producer = producer;
  uint64_t work = 0;
  for (size_t index = 0; index < pending.size(); ++index) {
    // Copy before appending to pending; its storage may move.
    Path path = pending[index];
    llvm::SmallVector<mlir::OpOperand *, 8> uses;
    for (auto &use : path.value.getUses())
      uses.push_back(&use);
    if (uses.empty())
      return {TemporalFusionQueryKind::NonUnique,
              "producer path has no terminal consumer"};
    if (llvm::any_of(uses, [&](auto *use) {
          return use->getOwner()->getBlock() != operation->getBlock();
        }))
      return {TemporalFusionQueryKind::NonUnique,
              "fusion use is outside the producer block"};
    llvm::sort(uses, [](auto *lhs, auto *rhs) {
      if (lhs->getOwner() == rhs->getOwner())
        return lhs->getOperandNumber() < rhs->getOperandNumber();
      return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
    });
    for (auto *use : uses) {
      if (++work > limits.maxRectangularPieces)
        return {TemporalFusionQueryKind::Indeterminate,
                "fusion use traversal exceeded its work budget"};
      auto *owner = use->getOwner();
      if (owner->getBlock() != operation->getBlock() ||
          owner->getParentOfType<TileRegionOp>() !=
              operation->getParentOfType<TileRegionOp>() ||
          !operation->isBeforeInBlock(owner))
        return {TemporalFusionQueryKind::NonUnique,
                "fusion use is outside the ordered producer block"};
      if (isTemporalCandidate(owner)) {
        auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(owner);
        if (dps && dps.isDpsInit(use))
          return {TemporalFusionQueryKind::NonUnique,
                  "fusion use reads a DPS destination"};
        auto descriptor = buildDescriptor(owner);
        if (mlir::failed(descriptor))
          return {TemporalFusionQueryKind::Unsupported,
                  "consumer iteration domain is unavailable"};
        auto access = analysis::deriveIterationOperandRelation(
            *use, descriptor->iterationExtents, limits);
        if (!access.isExact())
          return relationFailure(access.status, access.reason);
        auto relation = access.get()->compose(path.relation, limits);
        if (!relation.isExact())
          return relationFailure(relation.status, relation.reason);
        TemporalFusionUse terminal;
        terminal.operand = use;
        terminal.consumerValue = path.value;
        terminal.iterationShape = descriptor->iterationExtents;
        terminal.iterationToProducer = std::move(*relation.get());
        terminal.viewToProducer = path.relation;
        auto generation =
            describeFusionUse(producer, terminal, *descriptor, limits);
        if (generation.kind != TemporalFusionQueryKind::ExactDerived)
          return generation;
        fusion.uses.push_back(std::move(terminal));
        continue;
      }
      if (!mlir::isMemoryEffectFree(owner) || owner->getNumResults() != 1)
        return {TemporalFusionQueryKind::NonUnique,
                "producer has an observable or effectful unfused use"};
      auto step =
          analysis::deriveTensorResultIndexing(owner->getResult(0), limits);
      if (!step.isExact()) {
        auto status =
            step.status == analysis::TensorResultIndexingStatus::BrokenContract
                ? analysis::IndexRelationStatus::Invalid
            : step.status ==
                    analysis::TensorResultIndexingStatus::ResourceExhausted
                ? analysis::IndexRelationStatus::ResourceExhausted
                : analysis::IndexRelationStatus::Unsupported;
        return relationFailure(status, step.detail);
      }
      if (!isViewTransparentKind(step.indexing->kind) ||
          step.indexing->operands.size() != 1 ||
          step.indexing->operands.front().role !=
              TensorIndexingOperandRole::Source ||
          step.indexing->operands.front().operand != use->getOperandNumber())
        return {TemporalFusionQueryKind::NonUnique,
                "producer use is not a transparent source relation"};
      if (!visitedViews.insert(owner).second)
        continue;
      auto next = step.indexing->operands.front().resultToOperand.compose(
          path.relation, limits);
      if (!next.isExact())
        return relationFailure(next.status, next.reason);
      pending.push_back({owner->getResult(0), std::move(*next.get())});
    }
  }
  if (fusion.uses.empty())
    return {TemporalFusionQueryKind::NonUnique,
            "producer has no live terminal consumer"};
  llvm::sort(fusion.uses, [](const auto &lhs, const auto &rhs) {
    if (lhs.operand->getOwner() == rhs.operand->getOwner())
      return lhs.operand->getOperandNumber() < rhs.operand->getOperandNumber();
    return lhs.operand->getOwner()->isBeforeInBlock(rhs.operand->getOwner());
  });
  const auto &first = fusion.uses.front();
  for (const auto &use : llvm::drop_begin(fusion.uses)) {
    if (use.iterationShape != first.iterationShape)
      return {TemporalFusionQueryKind::NonUnique,
              "shared consumers have no common iteration grid"};
    auto equal = use.iterationToProducer->isEquivalentTo(
        *first.iterationToProducer, limits);
    if (equal.status != analysis::IndexRelationStatus::Exact)
      return relationFailure(equal.status, equal.reason);
    if (!equal.isProvenTrue())
      return {TemporalFusionQueryKind::NonUnique,
              "terminal uses require different producer regions"};
    if (use.consumerValue != first.consumerValue &&
        (use.representation == TemporalTileRepresentation::ReshapePieces ||
         first.representation == TemporalTileRepresentation::ReshapePieces))
      return {TemporalFusionQueryKind::Unsupported,
              "piece assembly generator requires a common current view value"};
  }
  auto *lastConsumer = fusion.uses.back().operand->getOwner();
  llvm::SmallPtrSet<mlir::Operation *, 8> checkedConsumers;
  for (const auto &use : fusion.uses) {
    auto *consumer = use.operand->getOwner();
    if (consumer == lastConsumer || !checkedConsumers.insert(consumer).second)
      continue;
    for (auto result : consumer->getResults())
      for (auto &read : result.getUses()) {
        if (++work > limits.maxRectangularPieces)
          return {TemporalFusionQueryKind::Indeterminate,
                  "fusion dominance query exceeded its work budget"};
        auto *user = read.getOwner();
        while (user && user->getBlock() != operation->getBlock())
          user = user->getParentOp();
        if (!user || !lastConsumer->isBeforeInBlock(user))
          return {TemporalFusionQueryKind::Unsupported,
                  "common-loop generator cannot dominate an earlier consumer "
                  "result use"};
      }
  }
  // Distinct view values with identical producer demand use one rectangular
  // producer tile and reconstruct each local representation from that value.
  if (llvm::any_of(fusion.uses, [&](const auto &use) {
        return use.consumerValue != first.consumerValue;
      }))
    for (auto &use : fusion.uses)
      use.representation = TemporalTileRepresentation::RectangularImage;
  return {TemporalFusionQueryKind::ExactDerived, {}, std::move(fusion)};
}

analysis::RectangularTileImageResult
queryTemporalFusionTile(const TemporalFusion &fusion,
                        const TemporalFusionUse &use,
                        const TemporalScopeChoice &choice) {
  if (!fusion.producer || !use.operand || !use.iterationToProducer ||
      choice.operation != use.operand->getOwner())
    return {analysis::IndexRelationStatus::Invalid, std::nullopt,
            "operand tile query does not match the current consumer"};
  auto type = mlir::cast<mlir::RankedTensorType>(fusion.producer.getType());
  auto image = use.iterationToProducer->getRectangularTileImage(
      fusion.producer.getContext(), use.iterationShape, type.getShape(),
      choice.iteratorTileSizes);
  if (!image.isExact() || choice.loopOrder.empty())
    return image;

  // Pinned Linalg makeTiledShapes uses map(offsets) and map(sizes-1)+1.
  // Check that this generator convention equals the proved operand bounds;
  // relation expressibility alone cannot authorize its use. View composition
  // is outside this check: it has its own actual local view materializer.
  auto map = analysis::getStructuredOperandMap(*use.operand);
  auto operandType =
      mlir::cast<mlir::RankedTensorType>(use.operand->get().getType());
  auto access = analysis::deriveIterationOperandRelation(*use.operand,
                                                         use.iterationShape);
  if (mlir::failed(map) || !access.isExact())
    return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
            "consumer does not expose its tiling access contract"};
  auto bounds = access.get()->getRectangularTileImage(
      fusion.producer.getContext(), use.iterationShape, operandType.getShape(),
      choice.iteratorTileSizes);
  if (!bounds.isExact())
    return bounds;
  unsigned rank = use.iterationShape.size();
  auto *context = fusion.producer.getContext();
  llvm::SmallVector<mlir::AffineExpr, 6> lastPositions;
  for (unsigned axis = 0; axis < rank; ++axis)
    lastPositions.push_back(mlir::getAffineDimExpr(axis, context) - 1);
  auto last = mlir::AffineMap::get(rank, 0, lastPositions, context);
  llvm::SmallVector<mlir::AffineExpr, 6> sizes, starts;
  auto closedSizes = map->compose(last);
  for (unsigned axis = 0; axis < map->getNumResults(); ++axis) {
    if (mlir::isa<mlir::AffineConstantExpr>(
            mlir::simplifyAffineExpr(map->getResult(axis), rank, 0))) {
      starts.push_back(mlir::getAffineConstantExpr(0, context));
      sizes.push_back(
          mlir::getAffineConstantExpr(operandType.getDimSize(axis), context));
    } else {
      starts.push_back(map->getResult(axis));
      sizes.push_back(closedSizes.getResult(axis) + 1);
    }
  }
  auto offsets = mlir::AffineMap::get(2 * rank, 0, starts, context);
  auto sizeMap = mlir::AffineMap::get(rank, 0, sizes, context);
  if (mlir::simplifyAffineMap(offsets) !=
          mlir::simplifyAffineMap(bounds.image->offsetMap) ||
      mlir::simplifyAffineMap(sizeMap) !=
          mlir::simplifyAffineMap(bounds.image->sizeMap))
    return {
        analysis::IndexRelationStatus::Unsupported, std::nullopt,
        "consumer tiling interface cannot generate the proved access bounds"};
  return image;
}

TemporalConcatQueryResult
queryTemporalConcatAssembly(mlir::OpOperand &consumerOperand) {
  TemporalConcatQueryResult result;
  result.consumerOperand = &consumerOperand;
  result.assembledValue = consumerOperand.get();
  auto finalInsert =
      consumerOperand.get().getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!finalInsert)
    return result;
  mlir::Operation *consumer = consumerOperand.getOwner();
  auto consumerDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumer);
  if (!consumer || !isTemporalCandidate(consumer) ||
      (consumerDps && consumerDps.isDpsInit(&consumerOperand))) {
    result.kind = TemporalConcatQueryKind::NonUnique;
    result.detail = "insert assembly is not one temporal data operand";
    return result;
  }
  auto assembledType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumerOperand.get().getType());
  if (!assembledType || !assembledType.hasStaticShape()) {
    result.kind = TemporalConcatQueryKind::Unsupported;
    result.detail = "insert assembly requires a static ranked tensor";
    return result;
  }

  mlir::Value current = consumerOperand.get();
  llvm::SmallVector<TemporalConcatSegment, 4> reverseSegments;
  const analysis::IndexRelationLimits relationLimits;
  while (auto insert = current.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
    if (reverseSegments.size() >= relationLimits.maxRectangularPieces) {
      result.kind = TemporalConcatQueryKind::ResourceExhausted;
      result.detail = "insert assembly exceeded its segment work bound";
      return result;
    }
    if (!insert->hasOneUse() || !insert.hasUnitStride() ||
        insert.getResultType() != assembledType) {
      result.kind = TemporalConcatQueryKind::NonUnique;
      result.detail =
          "insert assembly chain is multi-use, strided, or type-inconsistent";
      return result;
    }
    llvm::SmallVector<int64_t, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    for (mlir::OpFoldResult offset : insert.getMixedOffsets()) {
      std::optional<int64_t> value = mlir::getConstantIntValue(offset);
      if (!value) {
        result.kind = TemporalConcatQueryKind::Unsupported;
        result.detail = "insert assembly requires static offsets";
        return result;
      }
      offsets.push_back(*value);
    }
    for (mlir::OpFoldResult size : insert.getMixedSizes()) {
      std::optional<int64_t> value = mlir::getConstantIntValue(size);
      if (!value) {
        result.kind = TemporalConcatQueryKind::Unsupported;
        result.detail = "insert assembly requires static sizes";
        return result;
      }
      sizes.push_back(*value);
    }
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        offsets.size() != static_cast<size_t>(assembledType.getRank()) ||
        sizes.size() != static_cast<size_t>(assembledType.getRank()) ||
        llvm::ArrayRef<int64_t>(sizes) != sourceType.getShape()) {
      result.kind = TemporalConcatQueryKind::BrokenContract;
      result.detail = "insert assembly source does not match its rectangle";
      return result;
    }
    TemporalConcatSegment segment;
    segment.source = insert.getSource();
    segment.offsets = std::move(offsets);
    segment.sizes = std::move(sizes);
    if (auto sourceResult = mlir::dyn_cast<mlir::OpResult>(segment.source)) {
      mlir::Operation *sourceOwner = sourceResult.getOwner();
      std::optional<mlir::OpOperand *> sourceUse =
          getOnlyOperationUse(sourceOwner);
      if (sourceUse && (*sourceUse)->getOwner() == insert.getOperation() &&
          (*sourceUse)->getOperandNumber() == 0 &&
          isTemporalCandidate(sourceOwner) &&
          !mlir::isa<mlir::tensor::PadOp>(sourceOwner) &&
          !mlir::isa<LinalgExtOnlineAttentionOp>(sourceOwner) &&
          sourceOwner->getBlock() == consumer->getBlock() &&
          sourceOwner->getParentOfType<TileRegionOp>() ==
              consumer->getParentOfType<TileRegionOp>() &&
          sourceOwner->isBeforeInBlock(consumer))
        segment.derivedProducer = sourceResult;
    }
    reverseSegments.push_back(std::move(segment));
    current = insert.getDest();
  }
  if (!current.getDefiningOp<mlir::tensor::EmptyOp>() ||
      reverseSegments.size() < 2) {
    result.kind = TemporalConcatQueryKind::NotConcat;
    result.detail.clear();
    return result;
  }
  llvm::reverse(reverseSegments);

  int64_t fullVolume = 1;
  for (int64_t extent : assembledType.getShape()) {
    int64_t next = 0;
    if (extent <= 0 || llvm::MulOverflow(fullVolume, extent, next)) {
      result.kind = TemporalConcatQueryKind::ResourceExhausted;
      result.detail = "insert assembly volume is not representable";
      return result;
    }
    fullVolume = next;
  }
  int64_t coveredVolume = 0;
  for (auto [index, segment] : llvm::enumerate(reverseSegments)) {
    int64_t volume = 1;
    for (auto [offset, size, extent] : llvm::zip_equal(
             segment.offsets, segment.sizes, assembledType.getShape())) {
      int64_t end = 0;
      int64_t next = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, end) ||
          end > extent || llvm::MulOverflow(volume, size, next)) {
        result.kind = TemporalConcatQueryKind::BrokenContract;
        result.detail = "insert assembly rectangle is invalid";
        return result;
      }
      volume = next;
    }
    int64_t nextCovered = 0;
    if (llvm::AddOverflow(coveredVolume, volume, nextCovered)) {
      result.kind = TemporalConcatQueryKind::ResourceExhausted;
      result.detail = "insert assembly covered volume is not representable";
      return result;
    }
    coveredVolume = nextCovered;
    for (size_t previous = 0; previous < index; ++previous) {
      bool overlaps = true;
      for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
           llvm::zip_equal(segment.offsets, segment.sizes,
                           reverseSegments[previous].offsets,
                           reverseSegments[previous].sizes))
        overlaps &=
            lhsOffset < rhsOffset + rhsSize && rhsOffset < lhsOffset + lhsSize;
      if (overlaps) {
        result.kind = TemporalConcatQueryKind::Unsupported;
        result.detail = "insert assembly rectangles overlap";
        return result;
      }
    }
  }
  if (coveredVolume != fullVolume) {
    result.kind = TemporalConcatQueryKind::Unsupported;
    result.detail = "insert assembly does not exactly cover its result";
    return result;
  }
  result.kind = TemporalConcatQueryKind::Exact;
  result.segments = std::move(reverseSegments);
  return result;
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
    TemporalTraversalKind kind,
    llvm::ArrayRef<TemporalScopeChoice> prefix) const {
  Completion completion;
  completion.kind = TemporalSuccessorKind::Choice;
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      getScopeDescriptors(kind);
  if (prefix.size() > descriptors.size()) {
    completion.kind = TemporalSuccessorKind::CompilerBug;
    completion.detail = "temporal prefix has an unexpected extra scope";
    return completion;
  }
  completion.choice.emplace();
  completion.choice->kind = kind;
  for (auto [index, descriptor] : llvm::enumerate(descriptors)) {
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

TemporalSuccessor TemporalDomain::getFirstIndependentChoice() const {
  TemporalChoice prefix;
  prefix.kind = TemporalTraversalKind::Independent;
  return completePrefix(prefix);
}

TemporalSuccessor
TemporalDomain::completePrefix(const TemporalChoice &prefix) const {
  Completion completed = completeChoice(prefix.kind, prefix.scopes);
  if (completed.kind != TemporalSuccessorKind::Choice || !completed.choice)
    return {completed.kind, {}, {}, std::move(completed.detail)};
  TemporalCursor cursor;
  cursor.choice = *completed.choice;
  return {TemporalSuccessorKind::Choice, std::move(completed.choice),
          std::move(cursor)};
}

TemporalSuccessor
TemporalDomain::getNextChoice(const TemporalCursor &cursor) const {
  Completion current = completeChoice(cursor.choice.kind, cursor.choice.scopes);
  if (current.kind != TemporalSuccessorKind::Choice || !current.choice ||
      !(*current.choice == cursor.choice))
    return {TemporalSuccessorKind::CompilerBug,
            {},
            {},
            "temporal cursor is stale for the current operation domain"};
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      getScopeDescriptors(cursor.choice.kind);
  TemporalChoice currentChoice = cursor.choice;
  while (true) {
    std::optional<TemporalChoice> nextChoice;
    for (size_t reverse = 0; reverse < descriptors.size(); ++reverse) {
      const size_t index = descriptors.size() - reverse - 1;
      TemporalChoice prefix;
      prefix.kind = cursor.choice.kind;
      prefix.scopes.assign(currentChoice.scopes.begin(),
                           currentChoice.scopes.begin() + index + 1);
      bool advanced = false;
      while (advanceScopeChoice(descriptors[index], prefix.scopes.back())) {
        if (containsScopeChoice(descriptors[index], prefix.scopes.back())) {
          advanced = true;
          break;
        }
      }
      if (!advanced)
        continue;
      Completion next = completeChoice(prefix.kind, prefix.scopes);
      if (next.kind != TemporalSuccessorKind::Choice || !next.choice)
        return {next.kind, {}, {}, std::move(next.detail)};
      nextChoice = std::move(*next.choice);
      break;
    }
    if (!nextChoice)
      break;
    if (nextChoice->kind == TemporalTraversalKind::Joint &&
        !isJointChoiceCompatible(*nextChoice)) {
      currentChoice = std::move(*nextChoice);
      continue;
    }
    TemporalCursor nextCursor;
    nextCursor.choice = *nextChoice;
    return {TemporalSuccessorKind::Choice, std::move(nextChoice),
            std::move(nextCursor)};
  }
  if (cursor.choice.kind == TemporalTraversalKind::Joint)
    return getFirstIndependentChoice();
  return {TemporalSuccessorKind::End};
}

bool TemporalDomain::isJointChoiceCompatible(
    const TemporalChoice &choice) const {
  if (choice.kind != TemporalTraversalKind::Joint)
    return true;
  llvm::DenseMap<mlir::Operation *, const TemporalScopeChoice *> byOperation;
  for (const TemporalScopeChoice &scope : choice.scopes)
    byOperation.try_emplace(scope.operation, &scope);
  llvm::DenseMap<mlir::Operation *, const TemporalFusion *> byProducer;
  for (const auto &fusion : fusions)
    byProducer.try_emplace(fusion.producer.getOwner(), &fusion);
  auto derivedRequestsAgree = [&](const TemporalFusion &fusion) {
    struct Demand {
      mlir::Operation *consumer;
      llvm::SmallVector<int64_t, 4> shape;
      analysis::IndexRelation relation;
    };
    llvm::SmallVector<Demand, 8> pending, roots;
    for (const auto &use : fusion.uses)
      pending.push_back({use.operand->getOwner(), use.iterationShape,
                         *use.iterationToProducer});
    const analysis::IndexRelationLimits limits;
    for (size_t index = 0; index < pending.size(); ++index) {
      if (pending.size() > limits.maxRectangularPieces)
        return false;
      Demand demand = pending[index];
      if (byOperation.count(demand.consumer)) {
        roots.push_back(std::move(demand));
        continue;
      }
      auto downstream = byProducer.find(demand.consumer);
      if (downstream == byProducer.end())
        return false;
      auto result = downstream->second->producer;
      auto resultMap = analysis::getStructuredResultMap(result);
      auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
      if (mlir::failed(resultMap))
        return false;
      auto toResult = analysis::IndexRelation::fromAffineMap(
          *resultMap, demand.shape, resultType.getShape());
      if (!toResult.isExact())
        return false;
      auto toIteration = toResult.get()->inverse();
      if (!toIteration.isExact())
        return false;
      for (const auto &use : downstream->second->uses) {
        auto mapped = use.iterationToProducer->compose(*toIteration.get());
        if (!mapped.isExact())
          return false;
        auto complete = mapped.get()->compose(demand.relation);
        if (!complete.isExact())
          return false;
        pending.push_back({use.operand->getOwner(), use.iterationShape,
                           std::move(*complete.get())});
      }
    }
    if (roots.empty())
      return false;
    if (llvm::all_of(roots, [&](const auto &root) {
          return byOperation.lookup(root.consumer)->loopOrder.empty();
        }))
      return true; // Full-extent choices do not transform the current IR.
    const auto &first = roots.front();
    const auto &selected = *byOperation.lookup(first.consumer);
    for (const auto &root : roots) {
      const auto &other = *byOperation.lookup(root.consumer);
      if (root.shape != first.shape ||
          other.iteratorTileSizes != selected.iteratorTileSizes ||
          other.loopOrder != selected.loopOrder ||
          !root.relation.isEquivalentTo(first.relation).isProvenTrue())
        return false;
    }
    auto type = mlir::cast<mlir::RankedTensorType>(fusion.producer.getType());
    auto image = first.relation.getRectangularTileImage(
        region->getContext(), first.shape, type.getShape(),
        selected.iteratorTileSizes);
    if (!image.isExact() || !image.image->distinctTilesDisjoint)
      return false;
    // The late slice helper has no hoist operation. Invariant requests need
    // the explicit common traversal described by the normal selected-use path.
    for (auto axis : selected.loopOrder)
      if (llvm::is_contained(image.image->invariantDimensions, axis))
        return false;
    return true;
  };
  for (const auto &fusion : fusions) {
    const TemporalScopeChoice *first = nullptr;
    unsigned foundChoices = 0;
    for (const auto &use : fusion.uses) {
      auto found = byOperation.find(use.operand->getOwner());
      if (found == byOperation.end())
        continue;
      ++foundChoices;
      const auto &selected = *found->second;
      if (first && (selected.iteratorTileSizes != first->iteratorTileSizes ||
                    selected.loopOrder != first->loopOrder))
        return false;
      first = &selected;
      if (selected.loopOrder.empty())
        continue;
      if (auto pack = mlir::dyn_cast<mlir::tensor::PackOp>(selected.operation);
          pack && use.representation == TemporalTileRepresentation::ResultSlice)
        continue;
      if (use.representation == TemporalTileRepresentation::ReshapePieces) {
        auto relation = analysis::deriveIterationOperandRelation(
            *use.operand, use.iterationShape);
        auto type =
            mlir::cast<mlir::RankedTensorType>(use.consumerValue.getType());
        if (!relation.isExact())
          return false;
        auto image = relation.get()->getRectangularTileImage(
            region->getContext(), use.iterationShape, type.getShape(),
            selected.iteratorTileSizes);
        if (!image.isExact() || !image.image->distinctTilesDisjoint)
          return false;
        for (auto axis : selected.loopOrder)
          if (llvm::is_contained(image.image->invariantDimensions, axis))
            return false;
        continue;
      }
      auto image = queryTemporalFusionTile(fusion, use, selected);
      if (!image.isExact() || !image.image->distinctTilesDisjoint)
        return false;
      bool invariant = false;
      for (auto axis : selected.loopOrder) {
        if (llvm::is_contained(image.image->invariantDimensions, axis))
          invariant = true;
        else if (invariant)
          return false;
      }
    }
    if (foundChoices && foundChoices != fusion.uses.size())
      return false;
    if (!foundChoices && fusion.uses.size() > 1 &&
        !derivedRequestsAgree(fusion))
      return false;
  }
  return true;
}

std::optional<TemporalChoice>
TemporalDomain::getCoupledStateProposal(const TemporalChoice &choice,
                                        const TemporalChoice *anchor) const {
  if (!contains(choice) ||
      (anchor && (anchor->kind != choice.kind || !contains(*anchor))))
    return std::nullopt;
  auto proposal = choice;
  auto descriptors = getScopeDescriptors(choice.kind);
  llvm::DenseMap<mlir::Operation *, size_t> positions;
  for (auto [index, scope] : llvm::enumerate(proposal.scopes))
    positions.try_emplace(scope.operation, index);
  for (auto [producerIndex, producerScope] : llvm::enumerate(proposal.scopes)) {
    auto *producer = producerScope.operation;
    if (producer->getNumResults() < 2)
      continue;
    mlir::Operation *soleConsumer = nullptr;
    bool unique = true;
    for (auto *user : producer->getUsers()) {
      unique &= !soleConsumer || soleConsumer == user;
      soleConsumer = user;
    }
    auto consumer =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(soleConsumer);
    if (!unique || !consumer || !consumer.hasPureTensorSemantics() ||
        consumer.getNumReductionLoops() || !positions.count(consumer))
      continue;
    size_t consumerIndex = positions.lookup(consumer);
    auto &consumerScope = proposal.scopes[consumerIndex];
    // Capacity repair couples only a changed producer/consumer pair. The
    // immutable parent point keeps unrelated traversals out of that direction.
    if (anchor &&
        producerScope.iteratorTileSizes ==
            anchor->scopes[producerIndex].iteratorTileSizes &&
        consumerScope.iteratorTileSizes ==
            anchor->scopes[consumerIndex].iteratorTileSizes)
      continue;
    const auto &p = descriptors[producerIndex];
    const auto &c = descriptors[consumerIndex];
    llvm::SmallVector<int64_t> producerToConsumer(p.iterationExtents.size(),
                                                  -1);
    llvm::SmallVector<int64_t> consumerToProducer(c.iterationExtents.size(),
                                                  -1);
    llvm::SmallBitVector common(c.iterationExtents.size(), true);
    llvm::SmallBitVector seen(producer->getNumResults());
    bool exact = true;
    for (mlir::OpOperand *input : consumer.getDpsInputOperands()) {
      auto result = mlir::dyn_cast<mlir::OpResult>(input->get());
      if (!result || result.getOwner() != producer)
        continue;
      auto from = analysis::getStructuredResultMap(result);
      auto to = consumer.getMatchingIndexingMap(input);
      if (mlir::failed(from) || !from->isProjectedPermutation() ||
          !to.isProjectedPermutation() ||
          from->getNumResults() != to.getNumResults()) {
        exact = false;
        break;
      }
      seen.set(result.getResultNumber());
      llvm::SmallBitVector dimensions(c.iterationExtents.size());
      for (auto [source, destination] :
           llvm::zip_equal(from->getResults(), to.getResults())) {
        unsigned a = mlir::cast<mlir::AffineDimExpr>(source).getPosition();
        unsigned b = mlir::cast<mlir::AffineDimExpr>(destination).getPosition();
        if (a >= p.iterationExtents.size() || b >= c.iterationExtents.size() ||
            p.iterationExtents[a] != c.iterationExtents[b] ||
            (producerToConsumer[a] >= 0 && producerToConsumer[a] != b) ||
            (consumerToProducer[b] >= 0 && consumerToProducer[b] != a)) {
          exact = false;
          break;
        }
        producerToConsumer[a] = b;
        consumerToProducer[b] = a;
        dimensions.set(b);
      }
      common &= dimensions;
    }
    // An unobserved state result still constrains the common traversal, but
    // need not become an artificial consumer operand (e.g. a retained max).
    for (auto result : producer->getResults()) {
      if (!exact || seen.test(result.getResultNumber()))
        continue;
      auto map = analysis::getStructuredResultMap(result);
      if (mlir::failed(map) || !map->isProjectedPermutation()) {
        exact = false;
        break;
      }
      llvm::SmallBitVector dimensions(c.iterationExtents.size());
      for (auto expression : map->getResults()) {
        unsigned axis =
            mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
        if (axis >= producerToConsumer.size()) {
          exact = false;
          break;
        }
        if (producerToConsumer[axis] >= 0)
          dimensions.set(producerToConsumer[axis]);
      }
      common &= dimensions;
    }
    if (!exact || seen.count() < 2)
      continue;
    for (auto [dimension, source] : llvm::enumerate(consumerToProducer)) {
      int64_t size = c.iterationExtents[dimension];
      if (common.test(dimension) && source >= 0 &&
          p.iteratorCapabilities[source] ==
              IteratorTilingCapability::Tileable &&
          c.iteratorCapabilities[dimension] ==
              IteratorTilingCapability::Tileable)
        size = std::min(producerScope.iteratorTileSizes[source],
                        consumerScope.iteratorTileSizes[dimension]);
      consumerScope.iteratorTileSizes[dimension] = size;
      if (source >= 0)
        producerScope.iteratorTileSizes[source] = size;
    }
  }
  for (auto [scope, descriptor] :
       llvm::zip_equal(proposal.scopes, descriptors)) {
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    if (mlir::failed(order))
      return std::nullopt;
    scope.loopOrder = std::move(*order);
  }
  if (proposal == choice || !contains(proposal))
    return std::nullopt;
  return proposal;
}

bool TemporalDomain::contains(const TemporalChoice &choice) const {
  Completion completed = completeChoice(choice.kind, choice.scopes);
  return completed.kind == TemporalSuccessorKind::Choice && completed.choice &&
         *completed.choice == choice && isJointChoiceCompatible(choice);
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

  std::vector<TemporalScopeDescriptor> independentScopes;
  for (mlir::Operation *operation : candidates) {
    mlir::FailureOr<TemporalScopeDescriptor> descriptor =
        buildDescriptor(operation);
    if (mlir::succeeded(descriptor))
      independentScopes.push_back(std::move(*descriptor));
  }

  std::vector<TemporalFusion> fusionCandidates;
  for (auto *producer : candidates) {
    if (producer->getNumResults() != 1)
      continue;
    auto query = queryTemporalFusion(producer->getResult(0));
    if (query.kind == TemporalFusionQueryKind::BrokenContract)
      return failed(TemporalDomainFailureKind::BrokenContract, query.detail);
    if (query.isExact())
      fusionCandidates.push_back(std::move(*query.fusion));
  }
  llvm::SmallPtrSet<mlir::Operation *, 16> groupRoots;
  for (const auto &fusion : fusionCandidates) {
    if (fusion.uses.size() < 2)
      continue;
    for (const auto &use : fusion.uses) {
      auto *consumer = use.operand->getOwner();
      auto descriptor = buildDescriptor(consumer);
      if (mlir::failed(descriptor))
        continue;
      auto iterators =
          mlir::cast<mlir::TilingInterface>(consumer).getLoopIteratorTypes();
      for (auto [axis, kind] : llvm::enumerate(iterators))
        if (kind == mlir::utils::IteratorType::reduction &&
            descriptor->iteratorCapabilities[axis] ==
                IteratorTilingCapability::Tileable &&
            descriptor->iterationExtents[axis] > 1)
          groupRoots.insert(consumer);
    }
  }
  llvm::SmallPtrSet<mlir::Operation *, 16> derivedProducers;
  for (const auto &fusion : fusionCandidates)
    if (!groupRoots.contains(fusion.producer.getOwner()) &&
        llvm::all_of(fusion.uses, [](const auto &use) {
          return use.representation !=
                 TemporalTileRepresentation::RectangularImage;
        }))
      derivedProducers.insert(fusion.producer.getOwner());
  std::vector<TemporalFusion> fusions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      forcedFullExtentDimensions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      generalReshapeDimensions;
  for (auto &fusion : fusionCandidates) {
    auto *producer = fusion.producer.getOwner();
    if (groupRoots.contains(producer))
      continue;
    // A parameterized demand requires its consumer's selected traversal.
    // Preserve the existing independent alternative when that traversal is
    // itself fully derived into another root.
    if (llvm::any_of(fusion.uses,
                     [](const auto &use) {
                       return use.representation ==
                              TemporalTileRepresentation::RectangularImage;
                     }) &&
        llvm::any_of(fusion.uses, [&](const auto &use) {
          return derivedProducers.contains(use.operand->getOwner());
        }))
      continue;
    derivedProducers.insert(producer);
    for (const auto &use : fusion.uses) {
      llvm::append_range(forcedFullExtentDimensions[use.operand->getOwner()],
                         use.fullExtentDimensions);
      llvm::append_range(generalReshapeDimensions[use.operand->getOwner()],
                         use.reshapeDimensions);
    }
    fusions.push_back(std::move(fusion));
  }
  for (auto *consumer : candidates)
    for (auto &operand : consumer->getOpOperands()) {
      auto concat = queryTemporalConcatAssembly(operand);
      if (concat.kind == TemporalConcatQueryKind::BrokenContract)
        return failed(TemporalDomainFailureKind::BrokenContract, concat.detail);
      if (concat.isExact())
        for (const auto &segment : concat.segments)
          if (segment.derivedProducer &&
              !groupRoots.contains(segment.derivedProducer->getOwner()))
            derivedProducers.insert(segment.derivedProducer->getOwner());
    }

  std::vector<TemporalScopeDescriptor> jointScopes;
  for (mlir::Operation *operation : candidates) {
    mlir::FailureOr<TemporalScopeDescriptor> descriptor =
        buildDescriptor(operation);
    if (mlir::failed(descriptor))
      continue;
    if (derivedProducers.contains(operation)) {
      auto tiling = mlir::cast<mlir::TilingInterface>(operation);
      auto iterators = tiling.getLoopIteratorTypes();
      bool hasFreeReduction = false;
      for (auto [axis, iterator] : llvm::enumerate(iterators)) {
        if (iterator != mlir::utils::IteratorType::reduction)
          descriptor->iteratorCapabilities[axis] =
              IteratorTilingCapability::FullExtentOnly;
        else if (descriptor->iteratorCapabilities[axis] ==
                     IteratorTilingCapability::Tileable &&
                 descriptor->iterationExtents[axis] > 1)
          hasFreeReduction = true;
      }
      if (!hasFreeReduction)
        continue;
      descriptor->role = TemporalScopeRole::FusedReduction;
    }
    auto forced = forcedFullExtentDimensions.find(operation);
    if (forced != forcedFullExtentDimensions.end())
      for (uint32_t dimension : forced->second) {
        if (dimension >= descriptor->iteratorCapabilities.size())
          return failed(TemporalDomainFailureKind::BrokenContract,
                        "derived full-extent dimension is outside its scope");
        descriptor->iteratorCapabilities[dimension] =
            IteratorTilingCapability::FullExtentOnly;
      }
    auto reshape = generalReshapeDimensions.find(operation);
    if (reshape != generalReshapeDimensions.end()) {
      llvm::sort(reshape->second);
      reshape->second.erase(
          std::unique(reshape->second.begin(), reshape->second.end()),
          reshape->second.end());
      for (uint32_t dimension : reshape->second) {
        if (dimension >= descriptor->iteratorCapabilities.size())
          return failed(TemporalDomainFailureKind::BrokenContract,
                        "reshape dimension is outside its temporal scope");
        descriptor->exactReshapeDimensions.push_back(dimension);
      }
    }
    jointScopes.push_back(std::move(*descriptor));
  }
  return {TemporalDomain(region, std::move(jointScopes),
                         std::move(independentScopes), std::move(fusions)),
          {}};
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

mlir::FailureOr<TemporalDomain>
remapTemporalDomain(const TemporalDomain &source, TileRegionOp mappedRegion,
                    const mlir::IRMapping &mapping,
                    std::string *failureReason) {
  auto fail = [&](llvm::StringRef reason)
      -> mlir::FailureOr<TemporalDomain> {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::failure();
  };
  if (!mappedRegion)
    return fail("temporal domain remap has no mapped TileRegion");

  auto mapOperation = [&](mlir::Operation *operation) -> mlir::Operation * {
    return operation ? mapping.lookupOrNull(operation) : nullptr;
  };
  auto mapValue = [&](mlir::Value value) -> mlir::Value {
    return value ? mapping.lookupOrNull(value) : mlir::Value{};
  };
  auto mapResult = [&](mlir::OpResult result) -> mlir::OpResult {
    mlir::Value mapped = mapValue(result);
    return mapped ? mlir::dyn_cast<mlir::OpResult>(mapped) : mlir::OpResult{};
  };
  auto mapOperand = [&](mlir::OpOperand *operand) -> mlir::OpOperand * {
    if (!operand)
      return nullptr;
    mlir::Operation *owner = mapOperation(operand->getOwner());
    if (!owner || operand->getOperandNumber() >= owner->getNumOperands())
      return nullptr;
    return &owner->getOpOperand(operand->getOperandNumber());
  };
  auto remapScopes = [&](llvm::ArrayRef<TemporalScopeDescriptor> scopes,
                         std::vector<TemporalScopeDescriptor> &out)
      -> bool {
    for (const TemporalScopeDescriptor &scope : scopes) {
      mlir::Operation *operation = mapOperation(scope.operation);
      if (!operation)
        return false;
      TemporalScopeDescriptor mapped{operation,
                                     scope.iterationExtents,
                                     scope.iteratorCapabilities,
                                     scope.precedence,
                                     scope.exactReshapeDimensions,
                                     scope.role};
      out.push_back(std::move(mapped));
    }
    return true;
  };

  std::vector<TemporalScopeDescriptor> jointScopes;
  std::vector<TemporalScopeDescriptor> independentScopes;
  if (!remapScopes(source.jointScopes, jointScopes) ||
      !remapScopes(source.independentScopes, independentScopes))
    return fail("temporal domain remap lost a scope operation");

  std::vector<TemporalFusion> fusions;
  for (const auto &fusion : source.fusions) {
    TemporalFusion mapped;
    mapped.producer = mapResult(fusion.producer);
    if (!mapped.producer)
      return fail("temporal domain remap lost a producer");
    for (const auto &use : fusion.uses) {
      auto copy = use;
      copy.operand = mapOperand(use.operand);
      copy.consumerValue = mapValue(use.consumerValue);
      if (!copy.operand || !copy.consumerValue)
        return fail("temporal domain remap lost a terminal use");
      mapped.uses.push_back(std::move(copy));
    }
    fusions.push_back(std::move(mapped));
  }
  return TemporalDomain(mappedRegion, std::move(jointScopes),
                        std::move(independentScopes), std::move(fusions));
}

} // namespace wafer::compiler::detail
