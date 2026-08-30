//===- TemporalDomain.cpp - Live-operation temporal choices -----------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  if (linalg)
    return linalg.hasPureTensorSemantics() && operation->getNumResults() != 0 &&
           !mlir::isa<mlir::linalg::FillOp>(operation);
  if (auto pad = mlir::dyn_cast_or_null<mlir::tensor::PadOp>(operation))
    return operation->getNumResults() != 0 &&
           mlir::isMemoryEffectFree(operation) &&
           static_cast<bool>(pad.getConstantPaddingValue()) &&
           mlir::isa<mlir::TilingInterface>(operation);
  return operation && operation->getNumResults() != 0 &&
         mlir::isMemoryEffectFree(operation) &&
         mlir::isa<mlir::tensor::PackOp, mlir::tensor::UnPackOp>(operation) &&
         mlir::isa<mlir::TilingInterface>(operation);
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
            "fusion requires symbol-free projected-permutation maps"};
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
    llvm::SmallBitVector consumerDimensions = getUsedDimensions(*consumerMap);
    for (auto [dimension, capability] :
         llvm::enumerate(consumerDescriptor->iteratorCapabilities))
      if (capability == IteratorTilingCapability::Tileable &&
          !consumerDimensions.test(dimension))
        return {TemporalFusionQueryKind::NonUnique,
                "consumer tiling can request the same producer tile more "
                "than once"};
  }

  if (!producerUsesResultTiling) {
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
  }
  return {TemporalFusionQueryKind::ExactDerived, {}};
}

TemporalFusionPathResult
queryTemporalProducerFusionPath(mlir::OpResult producer) {
  mlir::Operation *producerOperation = producer ? producer.getOwner() : nullptr;
  if (!producerOperation)
    return pathFailure(TemporalFusionQueryKind::BrokenContract, producer,
                       "temporal fusion path has no current producer");
  if (mlir::isa<LinalgExtOnlineAttentionOp>(producerOperation))
    return pathFailure(
        TemporalFusionQueryKind::NonUnique, producer,
        "online attention state remains an independent traversal root");
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
            getConsumerOperandMap(*nextUse);
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
    std::optional<mlir::AffineMap> projectedStep =
        step.getProjectedAffineMap(producer.getContext());
    if (!projectedStep)
      projectedStep = getUnitReshapeProjection(*indexing.indexing);
    if (!projectedStep)
      return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                         (llvm::Twine("view chain step ") +
                          currentOperation->getName().getStringRef() +
                          " has no dense affine tile map")
                             .str());
    if (!projectedViewToProducer) {
      projectedViewToProducer = projectedStep;
    } else {
      projectedViewToProducer = mlir::simplifyAffineMap(
          projectedViewToProducer->compose(*projectedStep));
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
  if (!currentOperation || !projectedViewToProducer || !crossedSupportOperation)
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
  mlir::FailureOr<mlir::AffineMap> producerMap = getProducerResultMap(producer);
  mlir::FailureOr<mlir::AffineMap> consumerMap =
      getConsumerOperandMap(*nextUse);
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

  std::optional<mlir::AffineMap> viewMap = projectedViewToProducer;
  llvm::SmallVector<TemporalViewDimensionMapping, 4> producerTileDimensions;
  if (!viewMap)
    return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                       "view chain is exact but has no affine tile projection");
  if (viewMap->getNumDims() != viewType.getRank() ||
      viewMap->getNumResults() != producerType.getRank() ||
      !isDenseOffsetProjection(*viewMap, &producerTileDimensions)) {
    std::string mapText;
    llvm::raw_string_ostream stream(mapText);
    stream << *viewMap;
    stream.flush();
    return pathFailure(TemporalFusionQueryKind::Unsupported, producer,
                       "view chain affine projection is not one dense tile: " +
                           mapText);
  }
  llvm::SmallBitVector usedViewDimensions(viewType.getRank());
  for (const TemporalViewDimensionMapping &mapping : producerTileDimensions)
    if (mapping.viewDimension >= 0)
      usedViewDimensions.set(static_cast<unsigned>(mapping.viewDimension));
  for (auto [dimension, extent] : llvm::enumerate(viewType.getShape()))
    if (!usedViewDimensions.test(dimension) && extent != 1)
      return pathFailure(TemporalFusionQueryKind::NonUnique, producer,
                         "view chain projects a non-unit consumer coordinate");

  llvm::SmallBitVector consumerDimensions = getUsedDimensions(*consumerMap);
  for (auto [dimension, capability] :
       llvm::enumerate(consumerDescriptor->iteratorCapabilities))
    if (capability == IteratorTilingCapability::Tileable &&
        !consumerDimensions.test(dimension))
      return pathFailure(
          TemporalFusionQueryKind::NonUnique, producer,
          "consumer tiling can request the same view producer tile repeatedly");
  llvm::SmallBitVector producerDimensions = getUsedDimensions(*producerMap);
  auto producerTiling = mlir::cast<mlir::TilingInterface>(producerOperation);
  llvm::SmallVector<mlir::utils::IteratorType, 4> producerIterators =
      producerTiling.getLoopIteratorTypes();
  for (unsigned dimension = 0; dimension < producerDimensions.size();
       ++dimension)
    if (!producerDimensions.test(dimension) &&
        producerDescriptor->iterationExtents[dimension] != 1 &&
        producerIterators[dimension] != mlir::utils::IteratorType::reduction)
      return pathFailure(
          TemporalFusionQueryKind::NonUnique, producer,
          "view producer result does not determine one parallel iterator");

  TemporalFusionPathResult result;
  result.kind = TemporalFusionQueryKind::ExactDerived;
  result.producer = producer;
  result.consumerOperand = nextUse;
  result.viewTransparent = true;
  result.producerDimensions = std::move(producerTileDimensions);
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

  llvm::SmallPtrSet<mlir::Operation *, 16> derivedProducers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      forcedFullExtentDimensions;
  for (mlir::Operation *producer : candidates) {
    if (mlir::isa<LinalgExtOnlineAttentionOp>(producer))
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
        if (segment.derivedProducer)
          derivedProducers.insert(segment.derivedProducer->getOwner());
    }
  }

  std::vector<TemporalScopeDescriptor> scopes;
  for (mlir::Operation *operation : candidates) {
    if (derivedProducers.contains(operation))
      continue;
    mlir::FailureOr<TemporalScopeDescriptor> descriptor =
        buildDescriptor(operation);
    if (mlir::failed(descriptor))
      continue;
    auto forced = forcedFullExtentDimensions.find(operation);
    if (forced != forcedFullExtentDimensions.end())
      for (uint32_t dimension : forced->second) {
        if (dimension >= descriptor->iteratorCapabilities.size())
          return failed(TemporalDomainFailureKind::BrokenContract,
                        "derived full-extent dimension is outside its scope");
        descriptor->iteratorCapabilities[dimension] =
            IteratorTilingCapability::FullExtentOnly;
      }
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
