//===- AttentionOps.cpp - Wafer structured attention semantics ---------===//

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

#include <limits>
#include <optional>

using namespace wafer;

namespace {

mlir::FailureOr<llvm::SmallVector<unsigned, 4>>
getProjectedDimensions(mlir::AffineMap map, int64_t shapedRank) {
  if (!map || map.getNumSymbols() != 0 ||
      map.getNumResults() != static_cast<unsigned>(shapedRank) ||
      !map.isProjectedPermutation())
    return mlir::failure();

  llvm::SmallVector<unsigned, 4> dimensions;
  dimensions.reserve(map.getNumResults());
  llvm::SmallBitVector seen(map.getNumDims(), false);
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || seen.test(dimension.getPosition()))
      return mlir::failure();
    seen.set(dimension.getPosition());
    dimensions.push_back(dimension.getPosition());
  }
  return dimensions;
}

llvm::SmallBitVector asDimensionSet(unsigned rank,
                                    llvm::ArrayRef<unsigned> dimensions) {
  llvm::SmallBitVector set(rank, false);
  for (unsigned dimension : dimensions)
    set.set(dimension);
  return set;
}

bool contains(llvm::ArrayRef<unsigned> dimensions, unsigned dimension) {
  return llvm::is_contained(dimensions, dimension);
}

mlir::FailureOr<AttentionIterationRoles>
inferAttentionIterationRoles(llvm::ArrayRef<mlir::AffineMap> maps,
                             bool hasMask) {
  const size_t expectedMapCount = hasMask ? 6 : 5;
  if (maps.size() != expectedMapCount || maps.empty())
    return mlir::failure();
  const unsigned rank = maps.front().getNumDims();
  if (rank == 0 || llvm::any_of(maps, [rank](mlir::AffineMap map) {
        return !map || map.getNumDims() != rank || map.getNumSymbols() != 0;
      }))
    return mlir::failure();

  const unsigned outputMapIndex = hasMask ? 5 : 4;
  mlir::FailureOr<llvm::SmallVector<unsigned, 4>> query =
      getProjectedDimensions(maps[0], maps[0].getNumResults());
  mlir::FailureOr<llvm::SmallVector<unsigned, 4>> key =
      getProjectedDimensions(maps[1], maps[1].getNumResults());
  mlir::FailureOr<llvm::SmallVector<unsigned, 4>> value =
      getProjectedDimensions(maps[2], maps[2].getNumResults());
  mlir::FailureOr<llvm::SmallVector<unsigned, 4>> scale =
      getProjectedDimensions(maps[3], maps[3].getNumResults());
  mlir::FailureOr<llvm::SmallVector<unsigned, 4>> output =
      getProjectedDimensions(maps[outputMapIndex],
                             maps[outputMapIndex].getNumResults());
  if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value) ||
      mlir::failed(scale) || mlir::failed(output) || !scale->empty())
    return mlir::failure();

  llvm::SmallBitVector querySet = asDimensionSet(rank, *query);
  llvm::SmallBitVector keySet = asDimensionSet(rank, *key);
  llvm::SmallBitVector valueSet = asDimensionSet(rank, *value);
  llvm::SmallBitVector outputSet = asDimensionSet(rank, *output);

  AttentionIterationRoles roles;
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    const bool inQuery = querySet.test(dimension);
    const bool inKey = keySet.test(dimension);
    const bool inValue = valueSet.test(dimension);
    const bool inOutput = outputSet.test(dimension);
    if (inQuery && inKey && inValue && inOutput) {
      roles.batch.push_back(dimension);
    } else if (inQuery && !inKey && !inValue && inOutput) {
      roles.query.push_back(dimension);
    } else if (inQuery && inKey && !inValue && !inOutput) {
      roles.queryKeyReduction.push_back(dimension);
    } else if (!inQuery && inKey && inValue && !inOutput) {
      roles.keyValueReduction.push_back(dimension);
    } else if (!inQuery && !inKey && inValue && inOutput) {
      roles.valueOutput.push_back(dimension);
    } else {
      return mlir::failure();
    }
  }

  if (roles.query.empty() || roles.queryKeyReduction.empty() ||
      roles.keyValueReduction.empty() || roles.valueOutput.empty())
    return mlir::failure();

  if (hasMask) {
    mlir::FailureOr<llvm::SmallVector<unsigned, 4>> mask =
        getProjectedDimensions(maps[4], maps[4].getNumResults());
    if (mlir::failed(mask))
      return mlir::failure();
    for (unsigned dimension : *mask) {
      if (!contains(roles.batch, dimension) &&
          !contains(roles.query, dimension) &&
          !contains(roles.keyValueReduction, dimension))
        return mlir::failure();
    }
  }
  return roles;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
getStaticIterationExtents(llvm::ArrayRef<mlir::Value> values,
                          llvm::ArrayRef<mlir::AffineMap> maps,
                          int64_t iterationRank, bool requireStatic) {
  if (values.size() != maps.size() || iterationRank <= 0)
    return mlir::failure();
  llvm::SmallVector<int64_t, 8> extents(iterationRank, -1);
  for (auto [value, map] : llvm::zip_equal(values, maps)) {
    auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
    if (!shapedType || !shapedType.hasRank() ||
        (requireStatic && !shapedType.hasStaticShape()) ||
        map.getNumResults() != static_cast<unsigned>(shapedType.getRank()))
      return mlir::failure();
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      const int64_t extent = shapedType.getDimSize(axis);
      if (!dimension || extent == 0)
        return mlir::failure();
      if (mlir::ShapedType::isDynamic(extent))
        continue;
      int64_t &known = extents[dimension.getPosition()];
      if (known != -1 && known != extent)
        return mlir::failure();
      known = extent;
    }
  }
  if (requireStatic && llvm::is_contained(extents, int64_t{-1}))
    return mlir::failure();
  return extents;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
getStaticIterationExtents(LinalgExtAttentionOp op) {
  llvm::SmallVector<mlir::Value, 5> values{op.getQuery(), op.getKey(),
                                           op.getValue()};
  llvm::SmallVector<mlir::AffineMap, 5> maps{op.getQueryMap(), op.getKeyMap(),
                                             op.getValueMap()};
  if (op.getMask()) {
    values.push_back(op.getMask());
    maps.push_back(*op.getMaskMap());
  }
  values.push_back(op.getOutput());
  maps.push_back(op.getOutputMap());
  return getStaticIterationExtents(values, maps, op.getIterationDomainRank(),
                                   /*requireStatic=*/true);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
getStaticIterationExtents(LinalgExtOnlineAttentionOp op,
                          bool requireStatic = true) {
  llvm::SmallVector<mlir::Value, 7> values{op.getQuery(), op.getKey(),
                                           op.getValue()};
  llvm::SmallVector<mlir::AffineMap, 7> maps{op.getQueryMap(), op.getKeyMap(),
                                             op.getValueMap()};
  if (op.getMask()) {
    values.push_back(op.getMask());
    maps.push_back(*op.getMaskMap());
  }
  values.append({op.getAccumulator(), op.getMaximum(), op.getSum()});
  maps.append({op.getAccumulatorMap(), op.getMaximumMap(), op.getSumMap()});
  return getStaticIterationExtents(values, maps, op.getIterationDomainRank(),
                                   requireStatic);
}

mlir::FailureOr<AttentionIterationRoles>
inferAttentionIterationRoles(LinalgExtOnlineAttentionOp op) {
  llvm::SmallVector<mlir::AffineMap, 6> semanticMaps{
      op.getQueryMap(), op.getKeyMap(), op.getValueMap(), op.getScaleMap()};
  if (op.getMask())
    semanticMaps.push_back(*op.getMaskMap());
  semanticMaps.push_back(op.getAccumulatorMap());
  return inferAttentionIterationRoles(semanticMaps,
                                      static_cast<bool>(op.getMask()));
}

mlir::FailureOr<mlir::Value>
buildSlice(mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
           mlir::AffineMap map, llvm::ArrayRef<mlir::OpFoldResult> offsets,
           llvm::ArrayRef<mlir::OpFoldResult> sizes,
           llvm::SmallVectorImpl<mlir::Operation *> &generatedSlices) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank() ||
      map.getNumResults() != static_cast<unsigned>(shapedType.getRank()))
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult, 4> mappedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mappedSizes;
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= offsets.size() ||
        dimension.getPosition() >= sizes.size())
      return mlir::failure();
    mappedOffsets.push_back(offsets[dimension.getPosition()]);
    mappedSizes.push_back(sizes[dimension.getPosition()]);
  }
  llvm::SmallVector<mlir::OpFoldResult, 4> strides(shapedType.getRank(),
                                                   builder.getIndexAttr(1));
  if (mlir::isa<mlir::RankedTensorType>(value.getType())) {
    auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
        location, value, mappedOffsets, mappedSizes, strides);
    generatedSlices.push_back(slice.getOperation());
    return slice.getResult();
  }
  if (mlir::isa<mlir::MemRefType>(value.getType())) {
    auto slice = builder.create<mlir::memref::SubViewOp>(
        location, value, mappedOffsets, mappedSizes, strides);
    generatedSlices.push_back(slice.getOperation());
    return slice.getResult();
  }
  return mlir::failure();
}

bool isFullDimensionTile(int64_t extent, mlir::OpFoldResult offset,
                         mlir::OpFoldResult size) {
  std::optional<int64_t> constantOffset = mlir::getConstantIntValue(offset);
  std::optional<int64_t> constantSize = mlir::getConstantIntValue(size);
  return constantOffset && *constantOffset == 0 && constantSize &&
         *constantSize == extent;
}

} // namespace

llvm::SmallVector<mlir::AffineMap, 6>
LinalgExtAttentionOp::getIndexingMapsArray() {
  llvm::SmallVector<mlir::AffineMap, 6> maps;
  maps.reserve(getIndexingMaps().size());
  for (mlir::Attribute attribute : getIndexingMaps())
    maps.push_back(mlir::cast<mlir::AffineMapAttr>(attribute).getValue());
  return maps;
}

llvm::SmallVector<int64_t> LinalgExtAttentionOp::getStaticLoopRanges() {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(extents))
    return llvm::SmallVector<int64_t>(getIterationDomainRank(),
                                      mlir::ShapedType::kDynamic);
  return llvm::SmallVector<int64_t>(extents->begin(), extents->end());
}

mlir::FailureOr<AttentionIterationRoles>
LinalgExtAttentionOp::getIterationRoles() {
  return inferAttentionIterationRoles(getIndexingMapsArray(),
                                      static_cast<bool>(getMask()));
}

mlir::LogicalResult LinalgExtAttentionOp::verify() {
  const size_t expectedMapCount = getMask() ? 6 : 5;
  if (getIndexingMaps().size() != expectedMapCount)
    return emitOpError("requires one indexing map for each operand role");
  for (mlir::Attribute attribute : getIndexingMaps()) {
    if (!mlir::isa<mlir::AffineMapAttr>(attribute))
      return emitOpError("indexing_maps must contain only affine maps");
  }

  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  if (mlir::failed(roles))
    return emitOpError(
        "indexing maps must form complete B/M/K1/K2/N attention roles");
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(extents))
    return emitOpError(
        "requires positive static and mutually consistent iterator extents");

  auto queryType = mlir::cast<mlir::ShapedType>(getQuery().getType());
  auto keyType = mlir::cast<mlir::ShapedType>(getKey().getType());
  auto valueType = mlir::cast<mlir::ShapedType>(getValue().getType());
  auto outputType = mlir::cast<mlir::ShapedType>(getOutput().getType());
  mlir::Type elementType = queryType.getElementType();
  if (!mlir::isa<mlir::FloatType>(elementType) ||
      keyType.getElementType() != elementType ||
      valueType.getElementType() != elementType ||
      outputType.getElementType() != elementType ||
      !mlir::isa<mlir::FloatType>(getScale().getType()) ||
      (getMask() &&
       !mlir::isa<mlir::FloatType>(
           mlir::cast<mlir::ShapedType>(getMask().getType()).getElementType())))
    return emitOpError(
        "query, key, value, and output must share one floating storage type; "
        "scale and mask must also be floating");

  const bool tensorSemantics =
      mlir::isa<mlir::RankedTensorType>(getQuery().getType());
  for (mlir::Value shaped : llvm::SmallVector<mlir::Value, 5>{
           getQuery(), getKey(), getValue(), getOutput()}) {
    if (tensorSemantics != mlir::isa<mlir::RankedTensorType>(shaped.getType()))
      return emitOpError("cannot mix tensor and buffer semantics");
  }
  if (getMask() &&
      tensorSemantics != mlir::isa<mlir::RankedTensorType>(getMask().getType()))
    return emitOpError("cannot mix tensor and buffer semantics");
  if (tensorSemantics) {
    if (getNumResults() != 1 || getResult(0).getType() != getOutput().getType())
      return emitOpError(
          "tensor form requires one result tied to the output type");
  } else if (getNumResults() != 0) {
    return emitOpError("buffer form must not return a tensor result");
  }

  if (getAlgorithm() == AttentionAlgorithm::FlashDecoding) {
    int64_t keyValueElements = 1;
    for (unsigned dimension : roles->keyValueReduction) {
      int64_t extent = (*extents)[dimension];
      if (keyValueElements > std::numeric_limits<int64_t>::max() / extent)
        return emitOpError("K2 domain size is not representable");
      keyValueElements *= extent;
    }
    if (keyValueElements < 2)
      return emitOpError(
          "flash_decoding requires at least two nonempty K2 pieces");
  }
  return mlir::success();
}

void LinalgExtAttentionOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (mlir::isa<mlir::RankedTensorType>(getQuery().getType()))
    return;
  for (unsigned index = 0; index + 1 < getNumOperands(); ++index) {
    mlir::OpOperand &operand = getOperation()->getOpOperand(index);
    if (mlir::isa<mlir::MemRefType>(operand.get().getType()))
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &operand,
                           mlir::SideEffects::DefaultResource::get());
  }
  mlir::OpOperand &output = getOperation()->getOpOperand(getNumOperands() - 1);
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &output,
                       mlir::SideEffects::DefaultResource::get());
}

llvm::SmallVector<mlir::Range>
LinalgExtAttentionOp::getIterationDomain(mlir::OpBuilder &builder) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(extents))
    return {};
  llvm::SmallVector<mlir::Range> domain;
  domain.reserve(extents->size());
  for (int64_t extent : *extents)
    domain.push_back({builder.getIndexAttr(0), builder.getIndexAttr(extent),
                      builder.getIndexAttr(1)});
  return domain;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtAttentionOp::getLoopIteratorTypes() {
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  if (mlir::failed(roles))
    return {};
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      getIterationDomainRank(), mlir::utils::IteratorType::parallel);
  for (unsigned dimension : roles->queryKeyReduction)
    iteratorTypes[dimension] = mlir::utils::IteratorType::reduction;
  for (unsigned dimension : roles->keyValueReduction)
    iteratorTypes[dimension] = mlir::utils::IteratorType::reduction;
  return iteratorTypes;
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtAttentionOp::getTiledImplementation(
    mlir::OpBuilder &builder, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  if (offsets.size() != static_cast<size_t>(getIterationDomainRank()) ||
      sizes.size() != static_cast<size_t>(getIterationDomainRank()))
    return mlir::failure();
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(roles) || mlir::failed(extents))
    return mlir::failure();
  for (unsigned dimension : llvm::concat<const unsigned>(
           roles->queryKeyReduction, roles->keyValueReduction)) {
    if (!isFullDimensionTile((*extents)[dimension], offsets[dimension],
                             sizes[dimension]))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *> slices;
  llvm::SmallVector<mlir::Value, 6> operands;
  for (auto [value, map] : llvm::zip_equal(
           llvm::SmallVector<mlir::Value, 3>{getQuery(), getKey(), getValue()},
           llvm::SmallVector<mlir::AffineMap, 3>{getQueryMap(), getKeyMap(),
                                                 getValueMap()})) {
    mlir::FailureOr<mlir::Value> slice =
        buildSlice(builder, getLoc(), value, map, offsets, sizes, slices);
    if (mlir::failed(slice))
      return mlir::failure();
    operands.push_back(*slice);
  }
  operands.push_back(getScale());
  if (getMask()) {
    mlir::FailureOr<mlir::Value> mask = buildSlice(
        builder, getLoc(), getMask(), *getMaskMap(), offsets, sizes, slices);
    if (mlir::failed(mask))
      return mlir::failure();
    operands.push_back(*mask);
  }
  mlir::FailureOr<mlir::Value> output = buildSlice(
      builder, getLoc(), getOutput(), getOutputMap(), offsets, sizes, slices);
  if (mlir::failed(output))
    return mlir::failure();
  operands.push_back(*output);

  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (mlir::isa<mlir::RankedTensorType>(output->getType()))
    resultTypes.push_back(output->getType());
  mlir::Operation *tiled =
      mlir::clone(builder, getOperation(), resultTypes, operands);
  return mlir::TilingResult{
      {tiled}, llvm::SmallVector<mlir::Value>(tiled->getResults()), slices};
}

mlir::LogicalResult LinalgExtAttentionOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  if (resultNumber != 0 ||
      offsets.size() != static_cast<size_t>(getIterationDomainRank()) ||
      sizes.size() != static_cast<size_t>(getIterationDomainRank()))
    return mlir::failure();
  resultOffsets.clear();
  resultSizes.clear();
  for (mlir::AffineExpr expression : getOutputMap().getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension)
      return mlir::failure();
    resultOffsets.push_back(offsets[dimension.getPosition()]);
    resultSizes.push_back(sizes[dimension.getPosition()]);
  }
  return mlir::success();
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtAttentionOp::generateResultTileValue(
    mlir::OpBuilder &builder, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  if (resultNumber != 0 || offsets.size() != getOutputMap().getNumResults() ||
      sizes.size() != getOutputMap().getNumResults())
    return mlir::failure();
  llvm::SmallVector<mlir::Range> domain = getIterationDomain(builder);
  if (domain.size() != static_cast<size_t>(getIterationDomainRank()))
    return mlir::failure();
  llvm::SmallVector<mlir::OpFoldResult> iterationOffsets(
      domain.size(), builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> iterationSizes;
  iterationSizes.reserve(domain.size());
  for (mlir::Range range : domain)
    iterationSizes.push_back(range.size);
  for (auto [axis, expression] : llvm::enumerate(getOutputMap().getResults())) {
    unsigned dimension =
        mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
    iterationOffsets[dimension] = offsets[axis];
    iterationSizes[dimension] = sizes[axis];
  }
  return getTiledImplementation(builder, iterationOffsets, iterationSizes);
}

mlir::LogicalResult LinalgExtAttentionOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.clear();
  if (getNumResults() == 0)
    return mlir::success();
  auto outputType = mlir::cast<mlir::ShapedType>(getOutput().getType());
  llvm::SmallVector<mlir::OpFoldResult, 4> dimensions;
  dimensions.reserve(outputType.getRank());
  for (int64_t dimension = 0; dimension < outputType.getRank(); ++dimension) {
    int64_t extent = outputType.getDimSize(dimension);
    if (!mlir::ShapedType::isDynamic(extent)) {
      dimensions.push_back(builder.getIndexAttr(extent));
    } else if (mlir::isa<mlir::RankedTensorType>(getOutput().getType())) {
      dimensions.push_back(
          builder.create<mlir::tensor::DimOp>(getLoc(), getOutput(), dimension)
              .getResult());
    } else {
      dimensions.push_back(
          builder.create<mlir::memref::DimOp>(getLoc(), getOutput(), dimension)
              .getResult());
    }
  }
  reifiedReturnShapes.push_back(std::move(dimensions));
  return mlir::success();
}

CoupledReductionDescription
LinalgExtAttentionOp::getCoupledReductionDescription() {
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  assert(mlir::succeeded(roles) && "verifier-valid attention roles expected");
  llvm::SmallVector<mlir::AffineExpr, 4> rowExpressions;
  for (mlir::AffineExpr expression : getOutputMap().getResults()) {
    unsigned dimension =
        mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
    if (!contains(roles->valueOutput, dimension))
      rowExpressions.push_back(expression);
  }
  mlir::AffineMap rowMap = mlir::AffineMap::get(getIterationDomainRank(), 0,
                                                rowExpressions, getContext());

  CoupledReductionDescription description;
  description.reductionIterators = roles->keyValueReduction;
  mlir::Type stateElementType = getScale().getType();
  mlir::Type accumulatorElementType =
      mlir::cast<mlir::ShapedType>(getOutput().getType()).getElementType();
  description.components.push_back(
      {CoupledReductionComponentKind::Maximum, rowMap, stateElementType});
  description.components.push_back(
      {CoupledReductionComponentKind::Sum, rowMap, stateElementType});
  description.components.push_back({CoupledReductionComponentKind::Accumulator,
                                    getOutputMap(), accumulatorElementType});
  return description;
}

mlir::MutableOperandRange LinalgExtOnlineAttentionOp::getDpsInitsMutable() {
  return mlir::MutableOperandRange(getOperation(), getMask() ? 5 : 4, 3);
}

llvm::SmallVector<mlir::AffineMap, 8>
LinalgExtOnlineAttentionOp::getIndexingMapsArray() {
  llvm::SmallVector<mlir::AffineMap, 8> maps;
  maps.reserve(getIndexingMaps().size());
  for (mlir::Attribute attribute : getIndexingMaps())
    maps.push_back(mlir::cast<mlir::AffineMapAttr>(attribute).getValue());
  return maps;
}

llvm::SmallVector<int64_t> LinalgExtOnlineAttentionOp::getStaticLoopRanges() {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this, /*requireStatic=*/false);
  if (mlir::failed(extents))
    return llvm::SmallVector<int64_t>(getIterationDomainRank(),
                                      mlir::ShapedType::kDynamic);
  return llvm::SmallVector<int64_t>(extents->begin(), extents->end());
}

mlir::FailureOr<AttentionIterationRoles>
LinalgExtOnlineAttentionOp::getIterationRoles() {
  return inferAttentionIterationRoles(*this);
}

mlir::LogicalResult LinalgExtOnlineAttentionOp::verify() {
  const size_t expectedMapCount = getMask() ? 8 : 7;
  if (getIndexingMaps().size() != expectedMapCount)
    return emitOpError("requires one indexing map for each operand role");
  for (mlir::Attribute attribute : getIndexingMaps())
    if (!mlir::isa<mlir::AffineMapAttr>(attribute))
      return emitOpError("indexing_maps must contain only affine maps");

  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  if (mlir::failed(roles))
    return emitOpError(
        "indexing maps must form complete B/M/K1/K2/N attention roles");
  if (mlir::failed(getStaticIterationExtents(*this, /*requireStatic=*/false)))
    return emitOpError(
        "requires positive and mutually consistent known iterator extents");

  llvm::SmallVector<mlir::AffineExpr, 4> rowExpressions;
  for (mlir::AffineExpr expression : getAccumulatorMap().getResults()) {
    unsigned dimension =
        mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
    if (!contains(roles->valueOutput, dimension))
      rowExpressions.push_back(expression);
  }
  mlir::AffineMap expectedRowMap = mlir::AffineMap::get(
      getIterationDomainRank(), 0, rowExpressions, getContext());
  if (getMaximumMap() != expectedRowMap || getSumMap() != expectedRowMap)
    return emitOpError(
        "maximum and sum maps must equal the accumulator row map");

  mlir::Type storageElementType = getQuery().getType().getElementType();
  if (!mlir::isa<mlir::FloatType>(storageElementType) ||
      getKey().getType().getElementType() != storageElementType ||
      getValue().getType().getElementType() != storageElementType ||
      getAccumulator().getType().getElementType() != storageElementType ||
      !mlir::isa<mlir::FloatType>(getScale().getType()) ||
      getMaximum().getType().getElementType() != getScale().getType() ||
      getSum().getType().getElementType() != getScale().getType() ||
      (getMask() &&
       !mlir::isa<mlir::FloatType>(getMask().getType().getElementType())))
    return emitOpError(
        "query, key, value, and accumulator must share one floating storage "
        "type; maximum and sum must use the scale type; mask must be floating");

  if (getUpdatedAccumulator().getType() != getAccumulator().getType() ||
      getUpdatedMaximum().getType() != getMaximum().getType() ||
      getUpdatedSum().getType() != getSum().getType())
    return emitOpError(
        "each result type must equal its destination state type");
  return mlir::success();
}

void LinalgExtOnlineAttentionOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  (void)effects;
}

llvm::SmallVector<mlir::Range>
LinalgExtOnlineAttentionOp::getIterationDomain(mlir::OpBuilder &builder) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(extents))
    return {};
  llvm::SmallVector<mlir::Range> domain;
  domain.reserve(extents->size());
  for (int64_t extent : *extents)
    domain.push_back({builder.getIndexAttr(0), builder.getIndexAttr(extent),
                      builder.getIndexAttr(1)});
  return domain;
}

llvm::SmallVector<mlir::utils::IteratorType>
LinalgExtOnlineAttentionOp::getLoopIteratorTypes() {
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  if (mlir::failed(roles))
    return {};
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      getIterationDomainRank(), mlir::utils::IteratorType::parallel);
  for (unsigned dimension : roles->queryKeyReduction)
    iteratorTypes[dimension] = mlir::utils::IteratorType::reduction;
  for (unsigned dimension : roles->keyValueReduction)
    iteratorTypes[dimension] = mlir::utils::IteratorType::reduction;
  return iteratorTypes;
}

mlir::FailureOr<mlir::TilingResult>
LinalgExtOnlineAttentionOp::getTiledImplementation(
    mlir::OpBuilder &builder, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  if (offsets.size() != static_cast<size_t>(getIterationDomainRank()) ||
      sizes.size() != static_cast<size_t>(getIterationDomainRank()))
    return mlir::failure();
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> extents =
      getStaticIterationExtents(*this);
  if (mlir::failed(roles) || mlir::failed(extents))
    return mlir::failure();
  for (unsigned dimension : roles->queryKeyReduction)
    if (!isFullDimensionTile((*extents)[dimension], offsets[dimension],
                             sizes[dimension]))
      return mlir::failure();

  llvm::SmallVector<mlir::Operation *> slices;
  llvm::SmallVector<mlir::Value, 8> operands;
  auto appendSlice = [&](mlir::Value value,
                         mlir::AffineMap map) -> mlir::LogicalResult {
    mlir::FailureOr<mlir::Value> slice =
        buildSlice(builder, getLoc(), value, map, offsets, sizes, slices);
    if (mlir::failed(slice))
      return mlir::failure();
    operands.push_back(*slice);
    return mlir::success();
  };
  if (mlir::failed(appendSlice(getQuery(), getQueryMap())) ||
      mlir::failed(appendSlice(getKey(), getKeyMap())) ||
      mlir::failed(appendSlice(getValue(), getValueMap())))
    return mlir::failure();
  operands.push_back(getScale());
  if (getMask() && mlir::failed(appendSlice(getMask(), *getMaskMap())))
    return mlir::failure();
  if (mlir::failed(appendSlice(getAccumulator(), getAccumulatorMap())) ||
      mlir::failed(appendSlice(getMaximum(), getMaximumMap())) ||
      mlir::failed(appendSlice(getSum(), getSumMap())))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 3> resultTypes{
      operands[operands.size() - 3].getType(),
      operands[operands.size() - 2].getType(),
      operands[operands.size() - 1].getType()};
  mlir::Operation *tiled =
      mlir::clone(builder, getOperation(), resultTypes, operands);
  return mlir::TilingResult{
      {tiled}, llvm::SmallVector<mlir::Value>(tiled->getResults()), slices};
}

mlir::LogicalResult LinalgExtOnlineAttentionOp::getResultTilePosition(
    mlir::OpBuilder &builder, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::SmallVector<mlir::OpFoldResult> &resultOffsets,
    llvm::SmallVector<mlir::OpFoldResult> &resultSizes) {
  (void)builder;
  if (offsets.size() != static_cast<size_t>(getIterationDomainRank()) ||
      sizes.size() != static_cast<size_t>(getIterationDomainRank()))
    return mlir::failure();
  mlir::AffineMap resultMap;
  switch (resultNumber) {
  case 0:
    resultMap = getAccumulatorMap();
    break;
  case 1:
    resultMap = getMaximumMap();
    break;
  case 2:
    resultMap = getSumMap();
    break;
  default:
    return mlir::failure();
  }
  resultOffsets.clear();
  resultSizes.clear();
  for (mlir::AffineExpr expression : resultMap.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension)
      return mlir::failure();
    resultOffsets.push_back(offsets[dimension.getPosition()]);
    resultSizes.push_back(sizes[dimension.getPosition()]);
  }
  return mlir::success();
}

mlir::LogicalResult LinalgExtOnlineAttentionOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.clear();
  for (mlir::Value destination : {getAccumulator(), getMaximum(), getSum()}) {
    auto type = mlir::cast<mlir::RankedTensorType>(destination.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> dimensions;
    dimensions.reserve(type.getRank());
    for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
      int64_t extent = type.getDimSize(dimension);
      if (mlir::ShapedType::isDynamic(extent)) {
        dimensions.push_back(
            builder
                .create<mlir::tensor::DimOp>(getLoc(), destination, dimension)
                .getResult());
      } else {
        dimensions.push_back(builder.getIndexAttr(extent));
      }
    }
    reifiedReturnShapes.push_back(std::move(dimensions));
  }
  return mlir::success();
}

CoupledReductionDescription
LinalgExtOnlineAttentionOp::getCoupledReductionDescription() {
  mlir::FailureOr<AttentionIterationRoles> roles = getIterationRoles();
  assert(mlir::succeeded(roles) &&
         "verifier-valid online attention roles expected");
  CoupledReductionDescription description;
  description.reductionIterators = roles->keyValueReduction;
  description.components.push_back({CoupledReductionComponentKind::Maximum,
                                    getMaximumMap(),
                                    getMaximum().getType().getElementType()});
  description.components.push_back({CoupledReductionComponentKind::Sum,
                                    getSumMap(),
                                    getSum().getType().getElementType()});
  description.components.push_back(
      {CoupledReductionComponentKind::Accumulator, getAccumulatorMap(),
       getAccumulator().getType().getElementType()});
  return description;
}
