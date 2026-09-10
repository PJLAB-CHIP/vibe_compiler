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

std::optional<mlir::AffineMap>
getUnitReshapeProjection(const analysis::TensorResultIndexing &indexing) {
  if (indexing.operands.size() != 1 ||
      !llvm::is_contained({TensorIndexingTransformKind::ExpandShape,
                           TensorIndexingTransformKind::CollapseShape,
                           TensorIndexingTransformKind::Cast},
                          indexing.kind))
    return std::nullopt;
  mlir::Operation *operation = indexing.result.getOwner();
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(indexing.result.getType());
  auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(indexing.operands.front().operand).getType());
  if (!resultType || !operandType || !resultType.hasStaticShape() ||
      !operandType.hasStaticShape())
    return std::nullopt;
  llvm::SmallVector<unsigned, 4> resultNonUnit;
  llvm::SmallVector<unsigned, 4> operandNonUnit;
  for (auto [dimension, extent] : llvm::enumerate(resultType.getShape()))
    if (extent != 1)
      resultNonUnit.push_back(dimension);
  for (auto [dimension, extent] : llvm::enumerate(operandType.getShape()))
    if (extent != 1)
      operandNonUnit.push_back(dimension);
  if (resultNonUnit.size() != operandNonUnit.size())
    return std::nullopt;
  llvm::SmallVector<mlir::AffineExpr, 4> expressions;
  size_t nonUnit = 0;
  for (int64_t extent : operandType.getShape()) {
    if (extent == 1) {
      expressions.push_back(
          mlir::getAffineConstantExpr(0, operation->getContext()));
      continue;
    }
    if (nonUnit >= resultNonUnit.size() ||
        resultType.getDimSize(resultNonUnit[nonUnit]) != extent)
      return std::nullopt;
    expressions.push_back(mlir::getAffineDimExpr(resultNonUnit[nonUnit],
                                                 operation->getContext()));
    ++nonUnit;
  }
  return mlir::AffineMap::get(resultType.getRank(), 0, expressions,
                              operation->getContext());
}

TemporalFusionPathResult pathFailure(TemporalFusionQueryKind kind,
                                     mlir::OpResult producer,
                                     llvm::StringRef detail) {
  TemporalFusionPathResult result;
  result.kind = kind;
  result.producer = producer;
  result.detail = detail.str();
  return result;
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
                           const TemporalScopeDescriptor &descriptor) {
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
      shape, descriptor.iterationExtents);
  if (!fiber.isExact())
    return {fiber.status, std::nullopt, fiber.reason};
  return fiber.get()->isFunctional();
}

static analysis::IndexRelationQueryResult
queryConsumerRequestUniqueness(mlir::OpOperand &operand,
                               const TemporalScopeDescriptor &descriptor) {
  auto relation = analysis::deriveIterationOperandRelation(
      operand, descriptor.iterationExtents);
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
      type.getShape(), sizes);
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

static TemporalFusionQueryResult
queryTemporalProducerFusionImpl(mlir::OpResult producer,
                                mlir::OpOperand &consumerOperand,
                                bool requireUniqueUse) {
  mlir::Operation *producerOperation = producer.getOwner();
  mlir::Operation *consumerOperation = consumerOperand.getOwner();
  if (!producerOperation || !consumerOperation ||
      consumerOperand.get() != producer)
    return {TemporalFusionQueryKind::BrokenContract,
            "producer result and consumer operand do not form a current SSA "
            "edge"};
  if (producerOperation->getNumResults() > 1)
    return {TemporalFusionQueryKind::NonUnique,
            "multi-result producer requires whole-result fusion"};
  if (producerOperation->getBlock() != consumerOperation->getBlock() ||
      producerOperation->getParentOfType<TileRegionOp>() !=
          consumerOperation->getParentOfType<TileRegionOp>() ||
      !producerOperation->isBeforeInBlock(consumerOperation))
    return {TemporalFusionQueryKind::NonUnique,
            "producer and consumer are not one ordered same-Region edge"};
  if (requireUniqueUse) {
    std::optional<mlir::OpOperand *> onlyUse =
        getOnlyOperationUse(producerOperation);
    if (!onlyUse || *onlyUse != &consumerOperand)
      return {TemporalFusionQueryKind::NonUnique,
              "producer has multiple current uses or results"};
  }
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
  mlir::FailureOr<mlir::AffineMap> producerMap =
      analysis::getStructuredResultMap(producer);
  mlir::FailureOr<mlir::AffineMap> consumerMap =
      analysis::getStructuredOperandMap(consumerOperand);
  const bool producerUsesResultTiling =
      mlir::isa<mlir::tensor::PadOp, mlir::tensor::PackOp,
                mlir::tensor::UnPackOp>(producerOperation);
  auto packConsumer = mlir::dyn_cast<mlir::tensor::PackOp>(consumerOperation);
  const bool consumerUsesPackSource =
      packConsumer && consumerOperand.getOperandNumber() == 0;
  if (mlir::failed(producerDescriptor) || mlir::failed(consumerDescriptor) ||
      (!producerUsesResultTiling && mlir::failed(producerMap)) ||
      (!consumerUsesPackSource && mlir::failed(consumerMap)))
    return {TemporalFusionQueryKind::Unsupported,
            "current interfaces cannot expose an exact tile relation"};
  if ((!producerUsesResultTiling && (producerMap->getNumSymbols() != 0 ||
                                     !producerMap->isProjectedPermutation())) ||
      (!consumerUsesPackSource && (consumerMap->getNumSymbols() != 0 ||
                                   !consumerMap->isProjectedPermutation())))
    return {TemporalFusionQueryKind::Unsupported,
            "fusion requires a projected producer map and the selected "
            "symbol-free consumer map class"};
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(producer.getType());
  auto consumerType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumerOperand.get().getType());
  if (!producerType || !consumerType)
    return {TemporalFusionQueryKind::Unsupported,
            "fusion requires ranked tensor endpoints"};
  if ((!producerUsesResultTiling &&
       (producerMap->getNumDims() !=
            producerDescriptor->iterationExtents.size() ||
        producerMap->getNumResults() != producerType.getRank())) ||
      (!consumerUsesPackSource &&
       (consumerMap->getNumDims() !=
            consumerDescriptor->iterationExtents.size() ||
        consumerMap->getNumResults() != consumerType.getRank())) ||
      (consumerUsesPackSource &&
       packConsumer.getSource().getType() != consumerOperand.get().getType()))
    return {TemporalFusionQueryKind::BrokenContract,
            "fusion indexing map ranks do not match current tensor types"};

  if (!consumerUsesPackSource) {
    auto unique =
        queryConsumerRequestUniqueness(consumerOperand, *consumerDescriptor);
    if (unique.status != analysis::IndexRelationStatus::Exact)
      return relationFailure(unique.status, unique.reason);
    if (!unique.isProvenTrue())
      return {TemporalFusionQueryKind::NonUnique,
              "consumer requests share producer result positions"};
  }
  if (!producerUsesResultTiling) {
    auto fiber = queryProducerParallelFiber(producer, *producerDescriptor);
    if (fiber.status != analysis::IndexRelationStatus::Exact)
      return relationFailure(fiber.status, fiber.reason);
    if (!fiber.isProvenTrue())
      return {
          TemporalFusionQueryKind::NonUnique,
          "producer result does not determine its parallel iteration fiber"};
  }
  return {TemporalFusionQueryKind::ExactDerived, {}};
}

TemporalFusionQueryResult
queryTemporalProducerFusion(mlir::OpResult producer,
                            mlir::OpOperand &consumerOperand) {
  return queryTemporalProducerFusionImpl(producer, consumerOperand,
                                         /*requireUniqueUse=*/true);
}

namespace {

std::optional<TemporalOperandFusion>
queryOperandFusion(mlir::OpResult producer) {
  auto *operation = producer.getOwner();
  if (operation->getNumResults() != 1 || !isTemporalCandidate(operation) ||
      !mlir::isMemoryEffectFree(operation))
    return std::nullopt;
  auto descriptor = buildDescriptor(operation);
  auto outputMap = analysis::getStructuredResultMap(producer);
  // This is the pinned producer generator's result-tile contract. Access
  // relations themselves are not restricted to projected permutations.
  if (mlir::failed(descriptor) || mlir::failed(outputMap) ||
      !outputMap->isProjectedPermutation() ||
      !queryProducerParallelFiber(producer, *descriptor).isProvenTrue())
    return std::nullopt;
  auto onlyUse = getOnlyOperationUse(operation);
  if (!onlyUse)
    return std::nullopt;
  auto *use = *onlyUse;
  while (!isTemporalCandidate(use->getOwner())) {
    auto *view = use->getOwner();
    if (view->getBlock() != operation->getBlock() ||
        view->getNumResults() != 1 || !mlir::isMemoryEffectFree(view))
      return std::nullopt;
    auto indexing = analysis::deriveTensorResultIndexing(view->getResult(0));
    if (!indexing.isExact() || !isViewTransparentKind(indexing.indexing->kind))
      return std::nullopt;
    auto next = getOnlyOperationUse(view);
    if (!next)
      return std::nullopt;
    use = *next;
  }
  auto *consumer = use->getOwner();
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumer);
  if (!mlir::isa<mlir::linalg::LinalgOp>(consumer) ||
      consumer->getBlock() != operation->getBlock() ||
      consumer->getParentOfType<TileRegionOp>() !=
          operation->getParentOfType<TileRegionOp>() ||
      !operation->isBeforeInBlock(consumer) || (dps && dps.isDpsInit(use)))
    return std::nullopt;
  auto consumerDescriptor = buildDescriptor(consumer);
  if (mlir::failed(consumerDescriptor))
    return std::nullopt;
  auto relation = analysis::deriveIterationProducerRelation(
      *use, consumerDescriptor->iterationExtents, producer);
  if (!relation.isExact())
    return std::nullopt;
  auto type = mlir::cast<mlir::RankedTensorType>(producer.getType());
  auto full = relation.get()->getRectangularTileImage(
      producer.getContext(), consumerDescriptor->iterationExtents,
      type.getShape(), consumerDescriptor->iterationExtents);
  if (!full.isExact())
    return std::nullopt;
  return TemporalOperandFusion{producer, use->get(), use,
                               std::move(*relation.get()),
                               consumerDescriptor->iterationExtents};
}

bool isOperandChoiceCompatible(const TemporalOperandFusion &fusion,
                               const TemporalScopeChoice &choice) {
  auto image = queryTemporalOperandTile(fusion, choice);
  if (!image.isExact() || !image.image->distinctTilesDisjoint)
    return false;
  bool reachedInvariant = false;
  for (uint32_t axis : choice.loopOrder) {
    if (llvm::is_contained(image.image->invariantDimensions, axis))
      reachedInvariant = true;
    else if (reachedInvariant)
      return false;
  }
  return true;
}

} // namespace

analysis::RectangularTileImageResult
queryTemporalOperandTile(const TemporalOperandFusion &fusion,
                         const TemporalScopeChoice &choice) {
  if (!fusion.producer || !fusion.consumerOperand ||
      choice.operation != fusion.consumerOperand->getOwner())
    return {analysis::IndexRelationStatus::Invalid, std::nullopt,
            "operand tile query does not match the current consumer"};
  auto type = mlir::cast<mlir::RankedTensorType>(fusion.producer.getType());
  auto image = fusion.iterationToProducer.getRectangularTileImage(
      fusion.producer.getContext(), fusion.iterationShape, type.getShape(),
      choice.iteratorTileSizes);
  if (!image.isExact() || choice.loopOrder.empty())
    return image;

  // Pinned Linalg makeTiledShapes uses map(offsets) and map(sizes-1)+1.
  // Check that this generator convention equals the proved operand bounds;
  // relation expressibility alone cannot authorize its use. View composition
  // is outside this check: it has its own actual local view materializer.
  auto map = analysis::getStructuredOperandMap(*fusion.consumerOperand);
  auto operandType = mlir::cast<mlir::RankedTensorType>(
      fusion.consumerOperand->get().getType());
  auto access = analysis::deriveIterationOperandRelation(
      *fusion.consumerOperand, fusion.iterationShape);
  if (mlir::failed(map) || !access.isExact())
    return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
            "consumer does not expose its tiling access contract"};
  auto bounds = access.get()->getRectangularTileImage(
      fusion.producer.getContext(), fusion.iterationShape,
      operandType.getShape(), choice.iteratorTileSizes);
  if (!bounds.isExact())
    return bounds;
  unsigned rank = fusion.iterationShape.size();
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

TemporalFusionPathResult
queryTemporalProducerFusionPath(mlir::OpResult producer) {
  mlir::Operation *producerOperation = producer ? producer.getOwner() : nullptr;
  if (!producerOperation)
    return pathFailure(TemporalFusionQueryKind::BrokenContract, producer,
                       "temporal fusion path has no current producer");
  if (producerOperation->getNumResults() > 1)
    return pathFailure(TemporalFusionQueryKind::NonUnique, producer,
                       "multi-result producer requires whole-result fusion");
  std::optional<mlir::OpOperand *> onlyUse =
      getOnlyOperationUse(producerOperation);
  if (!onlyUse)
    return pathFailure(TemporalFusionQueryKind::NonUnique, producer,
                       "producer has multiple current uses or results");
  mlir::OpOperand *nextUse = *onlyUse;
  if (isTemporalCandidate(nextUse->getOwner())) {
    TemporalFusionQueryResult direct =
        queryTemporalProducerFusion(producer, *nextUse);
    TemporalFusionPathResult result;
    result.kind = direct.kind;
    result.producer = producer;
    result.consumerOperand =
        direct.kind == TemporalFusionQueryKind::ExactDerived ? nextUse
                                                             : nullptr;
    if (direct.kind == TemporalFusionQueryKind::ExactDerived) {
      if (auto pad = mlir::dyn_cast<mlir::tensor::PadOp>(producerOperation)) {
        mlir::FailureOr<mlir::AffineMap> consumerMap =
            analysis::getStructuredOperandMap(*nextUse);
        if (mlir::failed(consumerMap))
          return pathFailure(
              TemporalFusionQueryKind::Unsupported, producer,
              "pad consumer does not expose an exact operand map");
        llvm::SmallVector<mlir::OpFoldResult, 4> lowPadding =
            pad.getMixedLowPad();
        llvm::SmallVector<mlir::OpFoldResult, 4> highPadding =
            pad.getMixedHighPad();
        for (auto [dimension, low, high] :
             llvm::enumerate(lowPadding, highPadding)) {
          std::optional<int64_t> lowValue = mlir::getConstantIntValue(low);
          std::optional<int64_t> highValue = mlir::getConstantIntValue(high);
          if (!lowValue || !highValue)
            return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                               "pad fusion requires static padding");
          if (*lowValue == 0 && *highValue == 0)
            continue;
          auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(
              consumerMap->getResult(dimension));
          if (!iterator)
            return pathFailure(
                TemporalFusionQueryKind::Unsupported, producer,
                "padded result axis is not one consumer iterator");
          result.forcedFullExtentConsumerDimensions.push_back(
              iterator.getPosition());
        }
        llvm::sort(result.forcedFullExtentConsumerDimensions);
        result.forcedFullExtentConsumerDimensions.erase(
            std::unique(result.forcedFullExtentConsumerDimensions.begin(),
                        result.forcedFullExtentConsumerDimensions.end()),
            result.forcedFullExtentConsumerDimensions.end());
      }
    }
    result.detail = std::move(direct.detail);
    return result;
  }

  std::optional<mlir::AffineMap> projectedViewToProducer;
  std::optional<analysis::IndexRelation> exactViewToProducer;
  bool affinePath = true;
  bool crossedSupportOperation = false;
  mlir::Operation *currentOperation = nextUse->getOwner();
  while (currentOperation && !isTemporalCandidate(currentOperation)) {
    if (currentOperation->getBlock() != producerOperation->getBlock() ||
        currentOperation->getParentOfType<TileRegionOp>() !=
            producerOperation->getParentOfType<TileRegionOp>() ||
        !producerOperation->isBeforeInBlock(currentOperation) ||
        !mlir::isMemoryEffectFree(currentOperation) ||
        currentOperation->getNumResults() != 1)
      return pathFailure(
          TemporalFusionQueryKind::NonUnique, producer,
          "tensor support path is not one ordered pure same-Region chain");
    mlir::OpResult supportResult = currentOperation->getResult(0);
    analysis::TensorResultIndexingResult indexing =
        analysis::deriveTensorResultIndexing(supportResult);
    if (!indexing.isExact()) {
      TemporalFusionQueryKind kind =
          indexing.status ==
                  analysis::TensorResultIndexingStatus::ResourceExhausted
              ? TemporalFusionQueryKind::Indeterminate
          : indexing.status ==
                  analysis::TensorResultIndexingStatus::BrokenContract
              ? TemporalFusionQueryKind::BrokenContract
              : TemporalFusionQueryKind::Unsupported;
      return pathFailure(kind, producer, indexing.detail);
    }
    if (!isViewTransparentKind(indexing.indexing->kind) ||
        indexing.indexing->operands.size() != 1 ||
        indexing.indexing->operands.front().role !=
            TensorIndexingOperandRole::Source ||
        indexing.indexing->operands.front().operand !=
            nextUse->getOperandNumber())
      return pathFailure(
          TemporalFusionQueryKind::Unsupported, producer,
          "tensor support path is not one transparent unary source relation");
    const analysis::IndexRelation &step =
        indexing.indexing->operands.front().resultToOperand;
    if (!exactViewToProducer) {
      exactViewToProducer = step;
    } else {
      analysis::IndexRelationResult composed =
          step.compose(*exactViewToProducer);
      if (!composed.isExact()) {
        TemporalFusionQueryKind kind =
            composed.status == analysis::IndexRelationStatus::ResourceExhausted
                ? TemporalFusionQueryKind::Indeterminate
            : composed.status == analysis::IndexRelationStatus::Invalid
                ? TemporalFusionQueryKind::BrokenContract
                : TemporalFusionQueryKind::Unsupported;
        return pathFailure(kind, producer, composed.reason);
      }
      exactViewToProducer = std::move(*composed.get());
    }
    std::optional<mlir::AffineMap> projectedStep =
        step.getProjectedAffineMap(producer.getContext());
    if (!projectedStep)
      projectedStep = getUnitReshapeProjection(*indexing.indexing);
    if (!projectedStep) {
      affinePath = false;
      projectedViewToProducer.reset();
    } else if (affinePath) {
      if (!projectedViewToProducer) {
        projectedViewToProducer = projectedStep;
      } else {
        projectedViewToProducer = mlir::simplifyAffineMap(
            projectedViewToProducer->compose(*projectedStep));
      }
    }
    crossedSupportOperation = true;
    onlyUse = getOnlyOperationUse(currentOperation);
    if (!onlyUse)
      return pathFailure(
          TemporalFusionQueryKind::NonUnique, producer,
          "tensor support result has multiple current uses or results");
    nextUse = *onlyUse;
    currentOperation = nextUse->getOwner();
  }
  if (!currentOperation || (!projectedViewToProducer && !exactViewToProducer) ||
      !crossedSupportOperation)
    return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                       "tensor support path has no temporal consumer");

  mlir::Operation *consumerOperation = nextUse->getOwner();
  if (consumerOperation->getBlock() != producerOperation->getBlock() ||
      consumerOperation->getParentOfType<TileRegionOp>() !=
          producerOperation->getParentOfType<TileRegionOp>() ||
      !producerOperation->isBeforeInBlock(consumerOperation))
    return pathFailure(TemporalFusionQueryKind::NonUnique, producer,
                       "view consumer is not ordered in the producer Region");
  auto consumerDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumerOperation);
  if (consumerDps && consumerDps.isDpsInit(nextUse))
    return pathFailure(
        TemporalFusionQueryKind::NonUnique, producer,
        "destination producer remains outside reduction traversal");
  if (!isTemporalCandidate(producerOperation) ||
      !isTemporalCandidate(consumerOperation))
    return pathFailure(
        TemporalFusionQueryKind::Unsupported, producer,
        "producer or view consumer lacks a static temporal contract");

  mlir::FailureOr<TemporalScopeDescriptor> producerDescriptor =
      buildDescriptor(producerOperation);
  mlir::FailureOr<TemporalScopeDescriptor> consumerDescriptor =
      buildDescriptor(consumerOperation);
  mlir::FailureOr<mlir::AffineMap> producerMap =
      analysis::getStructuredResultMap(producer);
  mlir::FailureOr<mlir::AffineMap> consumerMap =
      analysis::getStructuredOperandMap(*nextUse);
  if (mlir::failed(producerDescriptor) || mlir::failed(consumerDescriptor) ||
      mlir::failed(producerMap) || mlir::failed(consumerMap))
    return pathFailure(
        TemporalFusionQueryKind::Unsupported, producer,
        "view fusion interfaces cannot expose exact producer/consumer maps");
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(producer.getType());
  auto viewType =
      mlir::dyn_cast<mlir::RankedTensorType>(nextUse->get().getType());
  if (!producerType || !viewType || producerMap->getNumSymbols() != 0 ||
      consumerMap->getNumSymbols() != 0 ||
      !producerMap->isProjectedPermutation() ||
      !consumerMap->isProjectedPermutation() ||
      producerMap->getNumDims() !=
          producerDescriptor->iterationExtents.size() ||
      consumerMap->getNumDims() !=
          consumerDescriptor->iterationExtents.size() ||
      producerMap->getNumResults() != producerType.getRank() ||
      consumerMap->getNumResults() != viewType.getRank())
    return pathFailure(
        TemporalFusionQueryKind::Unsupported, producer,
        "view fusion requires supported static producer/consumer maps");

  llvm::SmallVector<TemporalViewDimensionMapping, 4> producerTileDimensions;
  bool generalReshape = false;
  if (projectedViewToProducer &&
      projectedViewToProducer->getNumDims() == viewType.getRank() &&
      projectedViewToProducer->getNumResults() == producerType.getRank() &&
      isDenseOffsetProjection(*projectedViewToProducer,
                              &producerTileDimensions)) {
    if (!exactViewToProducer ||
        !exactViewToProducer->isInjective().isProvenTrue())
      return pathFailure(TemporalFusionQueryKind::NonUnique, producer,
                         "view requests share producer positions");
  } else if (exactViewToProducer &&
             exactViewToProducer->hasCanonicalRowMajorReshapeConstruction()) {
    generalReshape = true;
  } else {
    std::string mapText;
    llvm::raw_string_ostream stream(mapText);
    if (projectedViewToProducer)
      stream << *projectedViewToProducer;
    else
      stream << "<non-affine>";
    stream.flush();
    return pathFailure(
        TemporalFusionQueryKind::Unsupported, producer,
        "view chain relation has no exact tile materialization: " + mapText);
  }

  auto unique = queryConsumerRequestUniqueness(*nextUse, *consumerDescriptor);
  auto fiber = queryProducerParallelFiber(producer, *producerDescriptor);
  for (const auto &query : {unique, fiber}) {
    if (query.status != analysis::IndexRelationStatus::Exact) {
      auto failure = relationFailure(query.status, query.reason);
      return pathFailure(failure.kind, producer, failure.detail);
    }
    if (!query.isProvenTrue())
      return pathFailure(
          TemporalFusionQueryKind::NonUnique, producer,
          "view demand does not determine independent producer work");
  }
  TemporalFusionPathResult result;
  result.kind = TemporalFusionQueryKind::ExactDerived;
  result.producer = producer;
  result.consumerOperand = nextUse;
  result.viewTransparent = true;
  result.producerDimensions = std::move(producerTileDimensions);
  if (generalReshape) {
    std::optional<llvm::SmallVector<mlir::ReassociationIndices>> reassociation =
        mlir::getReassociationIndicesForReshape(producerType, viewType);
    if (!reassociation)
      return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                         "general reshape chain is not one standard static "
                         "reassociation");
    llvm::SmallVector<unsigned, 4> fullViewDimensions;
    if (producerType.getRank() > viewType.getRank()) {
      for (auto [viewDimension, producerDimensions] :
           llvm::enumerate(*reassociation))
        if (producerDimensions.size() > 1)
          fullViewDimensions.push_back(viewDimension);
    } else {
      for (llvm::ArrayRef<int64_t> viewDimensions : *reassociation)
        if (viewDimensions.size() > 1)
          for (int64_t viewDimension : viewDimensions)
            fullViewDimensions.push_back(viewDimension);
    }
    for (unsigned viewDimension : fullViewDimensions) {
      if (viewDimension >= consumerMap->getNumResults())
        return pathFailure(
            TemporalFusionQueryKind::BrokenContract, producer,
            "reshape view dimension is outside the consumer map");
      auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(
          consumerMap->getResult(viewDimension));
      if (!iterator)
        return pathFailure(
            TemporalFusionQueryKind::Unsupported, producer,
            "reshape view dimension is not one consumer iterator");
      result.generalReshapeConsumerDimensions.push_back(iterator.getPosition());
    }
    llvm::sort(result.generalReshapeConsumerDimensions);
    result.generalReshapeConsumerDimensions.erase(
        std::unique(result.generalReshapeConsumerDimensions.begin(),
                    result.generalReshapeConsumerDimensions.end()),
        result.generalReshapeConsumerDimensions.end());
    result.consumerViewToProducer = std::move(exactViewToProducer);
  }
  return result;
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
  for (const TemporalJointProducerGroup &group : jointProducerGroups) {
    llvm::SmallVector<mlir::Operation *, 8> consumers;
    for (mlir::OpOperand *operand : group.consumerOperands)
      if (operand && !llvm::is_contained(consumers, operand->getOwner()))
        consumers.push_back(operand->getOwner());
    llvm::SmallVector<const TemporalScopeChoice *, 8> selected;
    for (mlir::Operation *consumer : consumers)
      if (auto found = byOperation.find(consumer); found != byOperation.end())
        selected.push_back(found->second);
    if (!selected.empty() && selected.size() != consumers.size())
      return false;
    if (selected.size() < 2)
      continue;
    for (const TemporalScopeChoice *scope : llvm::drop_begin(selected))
      if (scope->iteratorTileSizes != selected.front()->iteratorTileSizes ||
          scope->loopOrder != selected.front()->loopOrder)
        return false;
  }
  for (const TemporalOperandFusion &fusion : operandFusions) {
    if (!fusion.consumerOperand)
      return false;
    auto found = byOperation.find(fusion.consumerOperand->getOwner());
    if (found != byOperation.end() &&
        !isOperandChoiceCompatible(fusion, *found->second))
      return false;
  }
  return true;
}

bool TemporalDomain::contains(const TemporalChoice &choice) const {
  Completion completed = completeChoice(choice.kind, choice.scopes);
  return completed.kind == TemporalSuccessorKind::Choice && completed.choice &&
         *completed.choice == choice && isJointChoiceCompatible(choice);
}

static std::optional<TemporalJointProducerGroup>
queryAllUseDirectFusionGroup(mlir::Operation *producerOperation) {
  if (!producerOperation || producerOperation->getNumResults() != 1 ||
      !isTemporalCandidate(producerOperation) ||
      !mlir::isa<mlir::linalg::LinalgOp>(producerOperation) ||
      !mlir::isMemoryEffectFree(producerOperation))
    return std::nullopt;
  auto producer =
      mlir::dyn_cast<mlir::OpResult>(producerOperation->getResult(0));
  if (!producer || llvm::range_size(producer.getUses()) < 2)
    return std::nullopt;
  mlir::FailureOr<mlir::AffineMap> producerMap =
      analysis::getStructuredResultMap(producer);
  mlir::FailureOr<TemporalScopeDescriptor> producerDescriptor =
      buildDescriptor(producerOperation);
  if (mlir::failed(producerMap) || mlir::failed(producerDescriptor))
    return std::nullopt;

  TemporalJointProducerGroup group;
  group.producer = producer;
  group.consumerValue = producer;
  std::optional<analysis::IndexRelation> commonRelation;
  std::optional<TemporalScopeDescriptor> commonDescriptor;
  for (mlir::OpOperand &use : producer.getUses()) {
    if (!mlir::isa<mlir::linalg::LinalgOp>(use.getOwner()))
      return std::nullopt;
    TemporalFusionQueryResult query = queryTemporalProducerFusionImpl(
        producer, use, /*requireUniqueUse=*/false);
    mlir::FailureOr<mlir::AffineMap> consumerMap =
        analysis::getStructuredOperandMap(use);
    mlir::FailureOr<TemporalScopeDescriptor> consumerDescriptor =
        buildDescriptor(use.getOwner());
    if (query.kind != TemporalFusionQueryKind::ExactDerived ||
        mlir::failed(consumerMap) || mlir::failed(consumerDescriptor))
      return std::nullopt;
    auto relation = analysis::deriveIterationProducerRelation(
        use, consumerDescriptor->iterationExtents, producer);
    if (!relation.isExact())
      return std::nullopt;
    if (!commonRelation) {
      commonRelation = std::move(*relation.get());
      commonDescriptor = *consumerDescriptor;
    } else if (consumerDescriptor->iterationExtents !=
                   commonDescriptor->iterationExtents ||
               consumerDescriptor->iteratorCapabilities !=
                   commonDescriptor->iteratorCapabilities ||
               !relation.get()
                    ->isEquivalentTo(*commonRelation)
                    .isProvenTrue()) {
      return std::nullopt;
    }
    group.consumerOperands.push_back(&use);
  }
  llvm::sort(group.consumerOperands,
             [](mlir::OpOperand *left, mlir::OpOperand *right) {
               if (left->getOwner() == right->getOwner())
                 return left->getOperandNumber() < right->getOperandNumber();
               return left->getOwner()->isBeforeInBlock(right->getOwner());
             });
  return group;
}

static std::optional<TemporalJointProducerGroup>
queryAllUseViewFusionGroup(mlir::Operation *producerOperation) {
  if (!producerOperation || producerOperation->getNumResults() != 1 ||
      !isTemporalCandidate(producerOperation) ||
      !mlir::isa<mlir::linalg::LinalgOp>(producerOperation) ||
      !mlir::isMemoryEffectFree(producerOperation))
    return std::nullopt;
  auto producer =
      mlir::dyn_cast<mlir::OpResult>(producerOperation->getResult(0));
  std::optional<mlir::OpOperand *> producerUse =
      getOnlyOperationUse(producerOperation);
  if (!producer || !producerUse ||
      isTemporalCandidate((*producerUse)->getOwner()))
    return std::nullopt;

  std::optional<mlir::AffineMap> projectedViewToProducer;
  std::optional<analysis::IndexRelation> exactViewToProducer;
  bool affinePath = true;
  mlir::OpOperand *sourceUse = *producerUse;
  mlir::Value finalView;
  llvm::SmallVector<mlir::OpOperand *, 8> finalUses;
  while (mlir::Operation *support = sourceUse->getOwner()) {
    if (isTemporalCandidate(support) ||
        support->getBlock() != producerOperation->getBlock() ||
        support->getParentOfType<TileRegionOp>() !=
            producerOperation->getParentOfType<TileRegionOp>() ||
        !producerOperation->isBeforeInBlock(support) ||
        !mlir::isMemoryEffectFree(support) || support->getNumResults() != 1)
      return std::nullopt;
    mlir::OpResult supportResult = support->getResult(0);
    analysis::TensorResultIndexingResult indexing =
        analysis::deriveTensorResultIndexing(supportResult);
    if (!indexing.isExact() ||
        !isViewTransparentKind(indexing.indexing->kind) ||
        indexing.indexing->operands.size() != 1 ||
        indexing.indexing->operands.front().role !=
            TensorIndexingOperandRole::Source ||
        indexing.indexing->operands.front().operand !=
            sourceUse->getOperandNumber())
      return std::nullopt;
    const analysis::IndexRelation &step =
        indexing.indexing->operands.front().resultToOperand;
    if (!exactViewToProducer) {
      exactViewToProducer = step;
    } else {
      analysis::IndexRelationResult composed =
          step.compose(*exactViewToProducer);
      if (!composed.isExact())
        return std::nullopt;
      exactViewToProducer = std::move(*composed.get());
    }
    std::optional<mlir::AffineMap> projectedStep =
        step.getProjectedAffineMap(producerOperation->getContext());
    if (!projectedStep)
      projectedStep = getUnitReshapeProjection(*indexing.indexing);
    if (!projectedStep) {
      affinePath = false;
      projectedViewToProducer.reset();
    } else if (affinePath) {
      projectedViewToProducer =
          projectedViewToProducer
              ? mlir::simplifyAffineMap(
                    projectedViewToProducer->compose(*projectedStep))
              : projectedStep;
    }

    finalUses.clear();
    for (mlir::OpOperand &use : supportResult.getUses())
      finalUses.push_back(&use);
    if (finalUses.size() >= 2) {
      finalView = supportResult;
      break;
    }
    if (finalUses.size() != 1 ||
        isTemporalCandidate(finalUses.front()->getOwner()))
      return std::nullopt;
    sourceUse = finalUses.front();
  }
  if (!finalView || finalUses.size() < 2 ||
      (!projectedViewToProducer && !exactViewToProducer))
    return std::nullopt;

  mlir::FailureOr<TemporalScopeDescriptor> producerDescriptor =
      buildDescriptor(producerOperation);
  mlir::FailureOr<mlir::AffineMap> producerMap =
      analysis::getStructuredResultMap(producer);
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(producer.getType());
  auto viewType = mlir::dyn_cast<mlir::RankedTensorType>(finalView.getType());
  if (mlir::failed(producerDescriptor) || mlir::failed(producerMap) ||
      !producerType || !viewType || producerMap->getNumSymbols() != 0 ||
      !producerMap->isProjectedPermutation() ||
      producerMap->getNumDims() !=
          producerDescriptor->iterationExtents.size() ||
      producerMap->getNumResults() != producerType.getRank())
    return std::nullopt;
  if (!queryProducerParallelFiber(producer, *producerDescriptor).isProvenTrue())
    return std::nullopt;

  TemporalJointProducerGroup group;
  group.producer = producer;
  group.consumerValue = finalView;
  bool generalReshape = false;
  if (projectedViewToProducer &&
      projectedViewToProducer->getNumDims() == viewType.getRank() &&
      projectedViewToProducer->getNumResults() == producerType.getRank() &&
      isDenseOffsetProjection(*projectedViewToProducer,
                              &group.producerDimensions)) {
    if (!exactViewToProducer ||
        !exactViewToProducer->isInjective().isProvenTrue())
      return std::nullopt;
  } else if (exactViewToProducer &&
             exactViewToProducer->hasCanonicalRowMajorReshapeConstruction()) {
    generalReshape = true;
    group.consumerViewToProducer = exactViewToProducer;
  } else {
    return std::nullopt;
  }

  std::optional<mlir::AffineMap> commonConsumerMap;
  std::optional<analysis::IndexRelation> commonConsumerRelation;
  std::optional<TemporalScopeDescriptor> commonConsumerDescriptor;
  for (mlir::OpOperand *use : finalUses) {
    mlir::Operation *consumer = use->getOwner();
    auto consumerDps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumer);
    mlir::FailureOr<mlir::AffineMap> consumerMap =
        analysis::getStructuredOperandMap(*use);
    mlir::FailureOr<TemporalScopeDescriptor> consumerDescriptor =
        buildDescriptor(consumer);
    if (!mlir::isa<mlir::linalg::LinalgOp>(consumer) ||
        consumer->getBlock() != producerOperation->getBlock() ||
        consumer->getParentOfType<TileRegionOp>() !=
            producerOperation->getParentOfType<TileRegionOp>() ||
        !producerOperation->isBeforeInBlock(consumer) ||
        (consumerDps && consumerDps.isDpsInit(use)) ||
        mlir::failed(consumerMap) || mlir::failed(consumerDescriptor) ||
        consumerMap->getNumSymbols() != 0 ||
        !consumerMap->isProjectedPermutation() ||
        consumerMap->getNumDims() !=
            consumerDescriptor->iterationExtents.size() ||
        consumerMap->getNumResults() != viewType.getRank())
      return std::nullopt;
    if (!queryConsumerRequestUniqueness(*use, *consumerDescriptor)
             .isProvenTrue())
      return std::nullopt;
    auto relation = analysis::deriveIterationProducerRelation(
        *use, consumerDescriptor->iterationExtents, producer);
    if (!relation.isExact())
      return std::nullopt;
    if (!commonConsumerRelation) {
      commonConsumerMap = *consumerMap;
      commonConsumerRelation = std::move(*relation.get());
      commonConsumerDescriptor = *consumerDescriptor;
    } else if (consumerDescriptor->iterationExtents !=
                   commonConsumerDescriptor->iterationExtents ||
               consumerDescriptor->iteratorCapabilities !=
                   commonConsumerDescriptor->iteratorCapabilities ||
               !relation.get()
                    ->isEquivalentTo(*commonConsumerRelation)
                    .isProvenTrue()) {
      return std::nullopt;
    }
    group.consumerOperands.push_back(use);
  }

  if (generalReshape) {
    auto reassociation =
        mlir::getReassociationIndicesForReshape(producerType, viewType);
    if (!reassociation || !commonConsumerMap)
      return std::nullopt;
    llvm::SmallVector<unsigned, 4> fullViewDimensions;
    if (producerType.getRank() > viewType.getRank()) {
      for (auto [viewDimension, producerDimensions] :
           llvm::enumerate(*reassociation))
        if (producerDimensions.size() > 1)
          fullViewDimensions.push_back(viewDimension);
    } else {
      for (llvm::ArrayRef<int64_t> viewDimensions : *reassociation)
        if (viewDimensions.size() > 1)
          for (int64_t viewDimension : viewDimensions)
            fullViewDimensions.push_back(viewDimension);
    }
    for (unsigned viewDimension : fullViewDimensions) {
      if (viewDimension >= commonConsumerMap->getNumResults())
        return std::nullopt;
      auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(
          commonConsumerMap->getResult(viewDimension));
      if (!iterator)
        return std::nullopt;
      group.generalReshapeConsumerDimensions.push_back(iterator.getPosition());
    }
    llvm::sort(group.generalReshapeConsumerDimensions);
    group.generalReshapeConsumerDimensions.erase(
        std::unique(group.generalReshapeConsumerDimensions.begin(),
                    group.generalReshapeConsumerDimensions.end()),
        group.generalReshapeConsumerDimensions.end());
  }
  llvm::sort(group.consumerOperands,
             [](mlir::OpOperand *left, mlir::OpOperand *right) {
               if (left->getOwner() == right->getOwner())
                 return left->getOperandNumber() < right->getOperandNumber();
               return left->getOwner()->isBeforeInBlock(right->getOwner());
             });
  return group;
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

  llvm::SmallPtrSet<mlir::Operation *, 16> derivedProducers;
  std::vector<TemporalJointProducerGroup> jointProducerGroups;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      forcedFullExtentDimensions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      generalReshapeDimensions;
  for (mlir::Operation *producer : candidates) {
    if (std::optional<TemporalJointProducerGroup> group =
            queryAllUseDirectFusionGroup(producer)) {
      derivedProducers.insert(producer);
      jointProducerGroups.push_back(std::move(*group));
      continue;
    }
    if (std::optional<TemporalJointProducerGroup> group =
            queryAllUseViewFusionGroup(producer)) {
      derivedProducers.insert(producer);
      for (mlir::OpOperand *operand : group->consumerOperands)
        llvm::append_range(generalReshapeDimensions[operand->getOwner()],
                           group->generalReshapeConsumerDimensions);
      jointProducerGroups.push_back(std::move(*group));
    }
  }
  // A common reduction loop owns its roots' inner choices. Pure parallel
  // consumers can still be derived together into a downstream traversal.
  llvm::SmallPtrSet<mlir::Operation *, 16> groupRoots;
  for (const auto &group : jointProducerGroups)
    for (auto *operand : group.consumerOperands) {
      auto *operation = operand->getOwner();
      auto descriptor = buildDescriptor(operation);
      if (mlir::failed(descriptor))
        continue;
      auto iterators =
          mlir::cast<mlir::TilingInterface>(operation).getLoopIteratorTypes();
      for (auto [axis, iterator] : llvm::enumerate(iterators))
        if (iterator == mlir::utils::IteratorType::reduction &&
            descriptor->iteratorCapabilities[axis] ==
                IteratorTilingCapability::Tileable &&
            descriptor->iterationExtents[axis] > 1)
          groupRoots.insert(operation);
    }
  for (mlir::Operation *producer : candidates) {
    if (groupRoots.contains(producer) || producer->getNumResults() > 1)
      continue;
    std::optional<mlir::OpOperand *> onlyUse = getOnlyOperationUse(producer);
    if (!onlyUse)
      continue;
    auto result = mlir::dyn_cast<mlir::OpResult>((*onlyUse)->get());
    if (!result || result.getOwner() != producer)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal producer use is not owned by its current result");
    TemporalFusionPathResult query = queryTemporalProducerFusionPath(result);
    if (query.kind == TemporalFusionQueryKind::BrokenContract)
      return failed(TemporalDomainFailureKind::BrokenContract, query.detail);
    if (query.kind == TemporalFusionQueryKind::ExactDerived) {
      derivedProducers.insert(producer);
      if (query.consumerOperand)
        llvm::append_range(
            forcedFullExtentDimensions[query.consumerOperand->getOwner()],
            query.forcedFullExtentConsumerDimensions);
      if (query.consumerOperand)
        llvm::append_range(
            generalReshapeDimensions[query.consumerOperand->getOwner()],
            query.generalReshapeConsumerDimensions);
    }
  }
  for (mlir::Operation *consumer : candidates) {
    for (mlir::OpOperand &operand : consumer->getOpOperands()) {
      TemporalConcatQueryResult concat = queryTemporalConcatAssembly(operand);
      if (concat.kind == TemporalConcatQueryKind::BrokenContract)
        return failed(TemporalDomainFailureKind::BrokenContract, concat.detail);
      if (!concat.isExact())
        continue;
      for (const TemporalConcatSegment &segment : concat.segments)
        if (segment.derivedProducer &&
            !groupRoots.contains(segment.derivedProducer->getOwner()))
          derivedProducers.insert(segment.derivedProducer->getOwner());
    }
  }

  std::vector<TemporalOperandFusion> operandFusions;
  for (mlir::Operation *producerOperation : candidates) {
    if (derivedProducers.contains(producerOperation) ||
        groupRoots.contains(producerOperation) ||
        producerOperation->getNumResults() != 1)
      continue;
    auto fusion = queryOperandFusion(producerOperation->getResult(0));
    if (!fusion ||
        derivedProducers.contains(fusion->consumerOperand->getOwner()))
      continue;
    derivedProducers.insert(producerOperation);
    operandFusions.push_back(std::move(*fusion));
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
  return {TemporalDomain(
              region, std::move(jointScopes), std::move(independentScopes),
              std::move(jointProducerGroups), std::move(operandFusions)),
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

  std::vector<TemporalJointProducerGroup> producerGroups;
  producerGroups.reserve(source.jointProducerGroups.size());
  for (const TemporalJointProducerGroup &group : source.jointProducerGroups) {
    mlir::OpResult producer = mapResult(group.producer);
    mlir::Value consumerValue = mapValue(group.consumerValue);
    if (!producer || !consumerValue)
      return fail("temporal domain remap lost a producer value");
    TemporalJointProducerGroup mapped;
    mapped.producer = producer;
    mapped.consumerValue = consumerValue;
    mapped.producerDimensions = group.producerDimensions;
    mapped.consumerViewToProducer = group.consumerViewToProducer;
    mapped.generalReshapeConsumerDimensions =
        group.generalReshapeConsumerDimensions;
    for (mlir::OpOperand *operand : group.consumerOperands) {
      mlir::OpOperand *mappedOperand = mapOperand(operand);
      if (!mappedOperand)
        return fail("temporal domain remap lost a consumer operand");
      mapped.consumerOperands.push_back(mappedOperand);
    }
    producerGroups.push_back(std::move(mapped));
  }

  std::vector<TemporalOperandFusion> operandFusions;
  for (const auto &fusion : source.operandFusions) {
    auto producer = mapResult(fusion.producer);
    auto value = mapValue(fusion.consumerValue);
    auto *operand = mapOperand(fusion.consumerOperand);
    if (!producer || !value || !operand)
      return fail("temporal domain remap lost an operand fusion");
    operandFusions.push_back({producer, value, operand,
                              fusion.iterationToProducer,
                              fusion.iterationShape});
  }

  return TemporalDomain(mappedRegion, std::move(jointScopes),
                        std::move(independentScopes), std::move(producerGroups),
                        std::move(operandFusions));
}

} // namespace wafer::compiler::detail
