//===- EdgeDomain.cpp - Selected edge logical domains ------------===//

#include "EdgeDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

mlir::LogicalResult validateEdge(mlir::Operation *producer,
                                 unsigned producerResult,
                                 mlir::Operation *consumer,
                                 unsigned consumerOperand, mlir::Block &body,
                                 std::string *failureReason,
                                 bool dependencyAlreadyValidated) {
  if (!producer || !consumer || producer->getBlock() != &body ||
      consumer->getBlock() != &body ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands())
    return failResult(failureReason,
                      "edge strategy must name one current-SSA dependency");
  if (!dependencyAlreadyValidated &&
      mlir::failed(traceProducerToConsumerChain(
          producer, producerResult, consumer, consumerOperand, failureReason)))
    return mlir::failure();
  if (producerResult != 0 || producer->getNumResults() != 1 ||
      consumer->getNumResults() != 1)
    return failResult(
        failureReason,
        "edge strategy requires one tiled producer and consumer result");
  return mlir::success();
}

mlir::LogicalResult validateStaticDomain(mlir::RankedTensorType type,
                                         llvm::ArrayRef<int64_t> offsets,
                                         llvm::ArrayRef<int64_t> sizes,
                                         std::string *failureReason,
                                         llvm::StringRef subject) {
  if (!type || !type.hasStaticShape() ||
      offsets.size() != static_cast<size_t>(type.getRank()) ||
      sizes.size() != offsets.size() ||
      llvm::any_of(llvm::zip_equal(offsets, sizes, type.getShape()),
                   [](auto values) {
                     auto [offset, size, extent] = values;
                     return offset < 0 || size <= 0 || extent <= 0 ||
                            offset > extent - size;
                   }))
    return failResult(failureReason,
                      (subject + " has an invalid static domain").str());
  return mlir::success();
}

mlir::LogicalResult
validateDemandIndexRelation(mlir::Operation *consumer, unsigned consumerOperand,
                            llvm::ArrayRef<int64_t> consumerOffsets,
                            llvm::ArrayRef<int64_t> consumerSizes,
                            llvm::ArrayRef<int64_t> producerOffsets,
                            llvm::ArrayRef<int64_t> producerSizes,
                            std::string *failureReason) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(consumer);
  auto consumerResultType = consumer && consumer->getNumResults() == 1
                                ? mlir::dyn_cast<mlir::RankedTensorType>(
                                      consumer->getResult(0).getType())
                                : mlir::RankedTensorType{};
  auto producerType = consumer && consumerOperand < consumer->getNumOperands()
                          ? mlir::dyn_cast<mlir::RankedTensorType>(
                                consumer->getOperand(consumerOperand).getType())
                          : mlir::RankedTensorType{};
  if (!linalg || !consumerResultType || !consumerResultType.hasStaticShape() ||
      !producerType || !producerType.hasStaticShape())
    return failResult(
        failureReason,
        "edge strategy requires one static Linalg index relation");
  llvm::SmallVector<mlir::AffineMap, 4> maps = linalg.getIndexingMapsArray();
  const unsigned resultMapIndex = linalg.getNumDpsInputs();
  llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
  if (consumerOperand >= maps.size() || resultMapIndex >= maps.size() ||
      llvm::any_of(loopShape, [](int64_t extent) { return extent <= 0; }))
    return failResult(
        failureReason,
        "consumer result and producer demand have no exact index relation");

  analysis::IndexRelationResult relation =
      analysis::IndexRelation::fromCommonIterationDomain(
          maps[resultMapIndex], consumerResultType.getShape(),
          maps[consumerOperand], producerType.getShape(), loopShape);
  if (!relation.isExact())
    return failResult(
        failureReason,
        "consumer result and producer demand have no exact index relation");
  analysis::StaticRectangularIndexSetResult exactDemand =
      relation.get()->getExactStaticRectangularImage(consumerOffsets,
                                                     consumerSizes);
  if (!exactDemand.isExact() ||
      exactDemand.domain->offsets != producerOffsets ||
      exactDemand.domain->sizes != producerSizes)
    return failResult(
        failureReason,
        "selected producer domain differs from the exact relation image");
  return mlir::success();
}

mlir::LogicalResult
mapOperandTileToResultTile(mlir::Operation *operation, unsigned operandNumber,
                           llvm::ArrayRef<int64_t> operandTileOffsets,
                           llvm::ArrayRef<int64_t> operandTileSizes,
                           llvm::SmallVectorImpl<int64_t> &resultTileOffsets,
                           llvm::SmallVectorImpl<int64_t> &resultTileSizes,
                           std::string *failureReason) {
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
  mlir::OpBuilder builder(operation);
  llvm::SmallVector<mlir::OpFoldResult, 4> operandOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> operandSizes;
  for (int64_t value : operandTileOffsets)
    operandOffsets.push_back(builder.getIndexAttr(value));
  for (int64_t value : operandTileSizes)
    operandSizes.push_back(builder.getIndexAttr(value));
  llvm::SmallVector<mlir::OpFoldResult, 4> iterationOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> iterationSizes;
  llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultSizes;
  if (tiling &&
      mlir::succeeded(tiling.getIterationDomainTileFromOperandTile(
          builder, operandNumber, operandOffsets, operandSizes,
          iterationOffsets, iterationSizes)) &&
      mlir::succeeded(tiling.getResultTilePosition(
          builder, /*resultNumber=*/0, iterationOffsets, iterationSizes,
          resultOffsets, resultSizes))) {
    for (mlir::OpFoldResult offset : resultOffsets) {
      std::optional<int64_t> value = mlir::getConstantIntValue(offset);
      if (!value)
        return failResult(failureReason,
                          "derived edge demand has a dynamic result offset");
      resultTileOffsets.push_back(*value);
    }
    for (mlir::OpFoldResult size : resultSizes) {
      std::optional<int64_t> value = mlir::getConstantIntValue(size);
      if (!value)
        return failResult(failureReason,
                          "derived edge demand has a dynamic result size");
      resultTileSizes.push_back(*value);
    }
    return mlir::success();
  }

  if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(operation)) {
    if (operandNumber != 0 ||
        operandTileOffsets.size() != slice.getStaticOffsets().size() ||
        operandTileSizes.size() != operandTileOffsets.size() ||
        llvm::any_of(slice.getStaticStrides(),
                     [](int64_t stride) { return stride != 1; }) ||
        llvm::any_of(slice.getStaticOffsets(),
                     [](int64_t offset) {
                       return mlir::ShapedType::isDynamic(offset);
                     }) ||
        llvm::any_of(slice.getStaticSizes(), [](int64_t size) {
          return mlir::ShapedType::isDynamic(size);
        }))
      return failResult(
          failureReason,
          "extract_slice tensor transform relation is not static unit "
          "stride");
    auto dropped = slice.getDroppedDims();
    for (auto [dimension, values] : llvm::enumerate(llvm::zip_equal(
             operandTileOffsets, operandTileSizes, slice.getStaticOffsets(),
             slice.getStaticSizes()))) {
      auto [tileOffset, tileSize, sliceOffset, sliceSize] = values;
      const int64_t begin = std::max(tileOffset, sliceOffset);
      const int64_t end =
          std::min(tileOffset + tileSize, sliceOffset + sliceSize);
      if (begin >= end)
        return failResult(
            failureReason,
            "extract_slice producer shard does not intersect its result");
      if (dropped.test(dimension)) {
        if (begin != sliceOffset || end - begin != 1)
          return failResult(
              failureReason,
              "extract_slice rank-reduced dimension is not exact");
        continue;
      }
      resultTileOffsets.push_back(begin - sliceOffset);
      resultTileSizes.push_back(end - begin);
    }
    return mlir::success();
  }

  if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(operation)) {
    if (operandTileOffsets.size() != operandTileSizes.size() ||
        llvm::any_of(insert.getStaticStrides(),
                     [](int64_t stride) { return stride != 1; }) ||
        llvm::any_of(insert.getStaticOffsets(),
                     [](int64_t offset) {
                       return mlir::ShapedType::isDynamic(offset);
                     }) ||
        llvm::any_of(insert.getStaticSizes(), [](int64_t size) {
          return mlir::ShapedType::isDynamic(size);
        }))
      return failResult(
          failureReason,
          "insert_slice tensor transform relation is not static unit "
          "stride");
    if (operation->getOperand(operandNumber) == insert.getSource()) {
      auto dropped = insert.getDroppedDims();
      size_t sourceDimension = 0;
      for (size_t resultDimension = 0;
           resultDimension < insert.getStaticOffsets().size();
           ++resultDimension) {
        const int64_t insertionOffset =
            insert.getStaticOffsets()[resultDimension];
        if (dropped.test(resultDimension)) {
          resultTileOffsets.push_back(insertionOffset);
          resultTileSizes.push_back(1);
          continue;
        }
        if (sourceDimension >= operandTileOffsets.size())
          return failResult(failureReason,
                            "insert_slice source shard rank is inconsistent");
        resultTileOffsets.push_back(insertionOffset +
                                    operandTileOffsets[sourceDimension]);
        resultTileSizes.push_back(operandTileSizes[sourceDimension]);
        ++sourceDimension;
      }
      if (sourceDimension != operandTileOffsets.size())
        return failResult(failureReason,
                          "insert_slice source shard rank is inconsistent");
      return mlir::success();
    }
    if (operation->getOperand(operandNumber) == insert.getDest()) {
      if (operandTileOffsets.size() != insert.getStaticOffsets().size())
        return failResult(
            failureReason,
            "insert_slice destination shard rank is inconsistent");
      bool overlapsInsertion = true;
      for (size_t dimension = 0; dimension < operandTileOffsets.size();
           ++dimension) {
        const int64_t tileBegin = operandTileOffsets[dimension];
        const int64_t tileEnd = tileBegin + operandTileSizes[dimension];
        const int64_t insertBegin = insert.getStaticOffsets()[dimension];
        const int64_t insertEnd =
            insertBegin + insert.getStaticSizes()[dimension];
        overlapsInsertion &= tileBegin < insertEnd && insertBegin < tileEnd;
      }
      if (overlapsInsertion)
        return failResult(
            failureReason,
            "insert_slice destination contribution is not one exact "
            "rectangle after overwrite");
      resultTileOffsets.assign(operandTileOffsets.begin(),
                               operandTileOffsets.end());
      resultTileSizes.assign(operandTileSizes.begin(), operandTileSizes.end());
      return mlir::success();
    }
    return failResult(failureReason, "insert_slice producer-to-consumer tensor "
                                     "chain names an unknown operand");
  }

  // Expand/collapse shape preserve row-major linear element order. A
  // contiguous rectangular producer shard therefore has one exact
  // rectangular image whenever the same linear interval can be expressed as
  // a rectangle in the result shape. This covers head/hidden reshapes without
  // inventing an axis from numeric equality.
  if (!mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
          operation) ||
      operation->getNumResults() != 1 ||
      operandNumber >= operation->getNumOperands())
    return failResult(failureReason, (llvm::Twine("tensor transform ") +
                                      operation->getName().getStringRef() +
                                      " has no exact tile index relation")
                                         .str());
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(operandNumber).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      operandTileOffsets.size() != static_cast<size_t>(sourceType.getRank()) ||
      operandTileSizes.size() != operandTileOffsets.size())
    return failResult(
        failureReason,
        "reshape tensor transform relation is not statically ranked");

  auto getGroupInterval = [&](mlir::RankedTensorType type,
                              mlir::ReassociationIndices group,
                              uint64_t &linearBegin, uint64_t &linearSize) {
    linearBegin = 0;
    linearSize = 1;
    uint64_t stride = 1;
    std::optional<size_t> firstVarying;
    for (auto [position, dimension] : llvm::enumerate(group)) {
      if (dimension < 0 || dimension >= type.getRank())
        return false;
      const int64_t extent = type.getDimSize(dimension);
      const int64_t offset = operandTileOffsets[dimension];
      const int64_t size = operandTileSizes[dimension];
      if (extent <= 0 || offset < 0 || size <= 0 || offset > extent - size)
        return false;
      if (size > 1 && !firstVarying)
        firstVarying = position;
    }
    if (firstVarying)
      for (size_t position = *firstVarying + 1; position < group.size();
           ++position) {
        const int64_t dimension = group[position];
        if (operandTileOffsets[dimension] != 0 ||
            operandTileSizes[dimension] != type.getDimSize(dimension))
          return false;
      }
    for (auto iterator = group.rbegin(); iterator != group.rend(); ++iterator) {
      const int64_t dimension = *iterator;
      linearBegin +=
          static_cast<uint64_t>(operandTileOffsets[dimension]) * stride;
      linearSize *= static_cast<uint64_t>(operandTileSizes[dimension]);
      stride *= static_cast<uint64_t>(type.getDimSize(dimension));
    }
    return true;
  };
  auto appendIntervalRectangle = [&](uint64_t linearBegin, uint64_t linearSize,
                                     mlir::ReassociationIndices group) {
    llvm::SmallVector<int64_t, 4> coordinates(group.size(), 0);
    uint64_t remainder = linearBegin;
    for (int64_t position = static_cast<int64_t>(group.size()) - 1;
         position >= 0; --position) {
      const int64_t dimension = group[position];
      if (dimension < 0 || dimension >= resultType.getRank())
        return false;
      const uint64_t extent =
          static_cast<uint64_t>(resultType.getDimSize(dimension));
      coordinates[position] = static_cast<int64_t>(remainder % extent);
      remainder /= extent;
    }
    if (remainder != 0)
      return false;
    for (int64_t pivot = static_cast<int64_t>(group.size()) - 1; pivot >= 0;
         --pivot) {
      uint64_t trailing = 1;
      bool aligned = true;
      for (size_t position = pivot + 1; position < group.size(); ++position) {
        if (coordinates[position] != 0) {
          aligned = false;
          break;
        }
        trailing *=
            static_cast<uint64_t>(resultType.getDimSize(group[position]));
      }
      if (!aligned || linearSize % trailing != 0)
        continue;
      const uint64_t pivotSize = linearSize / trailing;
      const uint64_t pivotExtent =
          static_cast<uint64_t>(resultType.getDimSize(group[pivot]));
      if (pivotSize == 0 ||
          pivotSize > pivotExtent - static_cast<uint64_t>(coordinates[pivot]))
        continue;
      for (size_t position = 0; position < group.size(); ++position) {
        resultTileOffsets.push_back(coordinates[position]);
        resultTileSizes.push_back(position < static_cast<size_t>(pivot) ? 1
                                  : position == static_cast<size_t>(pivot)
                                      ? static_cast<int64_t>(pivotSize)
                                      : resultType.getDimSize(group[position]));
      }
      return true;
    }
    return false;
  };

  if (auto collapse =
          mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation)) {
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
        collapse.getReassociationIndices();
    if (reassociation.size() != static_cast<size_t>(resultType.getRank()))
      return failResult(failureReason,
                        "collapse_shape reassociation rank is inconsistent");
    for (mlir::ReassociationIndices group : reassociation) {
      uint64_t begin = 0;
      uint64_t size = 0;
      if (!getGroupInterval(sourceType, group, begin, size))
        return failResult(
            failureReason,
            "collapse_shape source group is not one exact interval");
      resultTileOffsets.push_back(static_cast<int64_t>(begin));
      resultTileSizes.push_back(static_cast<int64_t>(size));
    }
    return mlir::success();
  }

  auto expand = mlir::cast<mlir::tensor::ExpandShapeOp>(operation);
  llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
      expand.getReassociationIndices();
  if (reassociation.size() != static_cast<size_t>(sourceType.getRank()))
    return failResult(failureReason,
                      "expand_shape reassociation rank is inconsistent");
  for (auto [sourceDimension, group] : llvm::enumerate(reassociation))
    if (!appendIntervalRectangle(
            static_cast<uint64_t>(operandTileOffsets[sourceDimension]),
            static_cast<uint64_t>(operandTileSizes[sourceDimension]), group))
      return failResult(failureReason,
                        "expand_shape interval has no exact result rectangle");
  return mlir::success();
}

mlir::LogicalResult deriveConsumerDomainFromProducerDemand(
    mlir::Operation *producer, unsigned producerResult,
    mlir::Operation *consumer, unsigned consumerOperand,
    llvm::ArrayRef<int64_t> producerOffsets,
    llvm::ArrayRef<int64_t> producerSizes,
    llvm::SmallVectorImpl<int64_t> &consumerOffsets,
    llvm::SmallVectorImpl<int64_t> &consumerSizes, std::string *failureReason) {
  if (!consumerOffsets.empty() || !consumerSizes.empty())
    return failResult(
        failureReason,
        "edge consumer demand must provide both offsets and sizes or neither");
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
      producerToConsumerChain = traceProducerToConsumerChain(
          producer, producerResult, consumer, consumerOperand, failureReason);
  if (mlir::failed(producerToConsumerChain))
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> currentOffsets(producerOffsets.begin(),
                                               producerOffsets.end());
  llvm::SmallVector<int64_t, 4> currentSizes(producerSizes.begin(),
                                             producerSizes.end());
  mlir::Value currentValue = producer->getResult(producerResult);
  for (mlir::Operation *support : *producerToConsumerChain) {
    std::optional<unsigned> sourceOperand;
    for (auto indexed : llvm::enumerate(support->getOperands()))
      if (indexed.value() == currentValue) {
        sourceOperand = static_cast<unsigned>(indexed.index());
        break;
      }
    if (!sourceOperand || support->getNumResults() != 1)
      return failResult(failureReason,
                        (llvm::Twine("support op ") +
                         support->getName().getStringRef() +
                         " lost its exact tensor source or single result")
                            .str());
    llvm::SmallVector<int64_t, 4> nextOffsets;
    llvm::SmallVector<int64_t, 4> nextSizes;
    if (mlir::failed(mapOperandTileToResultTile(
            support, *sourceOperand, currentOffsets, currentSizes, nextOffsets,
            nextSizes, failureReason)))
      return mlir::failure();
    currentOffsets = std::move(nextOffsets);
    currentSizes = std::move(nextSizes);
    currentValue = support->getResult(0);
  }

  if (mlir::failed(mapOperandTileToResultTile(
          consumer, consumerOperand, currentOffsets, currentSizes,
          consumerOffsets, consumerSizes, failureReason)))
    return mlir::failure();
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
