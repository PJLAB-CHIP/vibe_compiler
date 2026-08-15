//===- DependentDataflow.cpp - Selected edge-action lowering --------===//

#include "Internal.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <functional>
#include <limits>
#include <optional>
#include <tuple>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
wafer::deriveUnaryPureSupportChain(mlir::Operation *producer,
                                   unsigned producerResult,
                                   mlir::Operation *consumer,
                                   unsigned consumerOperand,
                                   std::string *failureReason) {
  auto failChain = [&](llvm::StringRef message)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> {
    setFailureReason(failureReason, message);
    return mlir::failure();
  };
  if (!producer || !consumer || producer->getBlock() != consumer->getBlock() ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands())
    return failChain(
        "edge strategy does not name one in-block structured dependency");

  mlir::Value source = producer->getResult(producerResult);
  mlir::Value current = consumer->getOperand(consumerOperand);
  llvm::DenseMap<mlir::Value, bool> sourceDependencyMemo;
  std::function<bool(mlir::Value)> dependsOnSource =
      [&](mlir::Value value) -> bool {
    if (value == source)
      return true;
    auto found = sourceDependencyMemo.find(value);
    if (found != sourceDependencyMemo.end())
      return found->second;
    mlir::Operation *operation = value.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        (mlir::isa<mlir::TilingInterface>(operation) &&
         mlir::isa<mlir::DestinationStyleOpInterface>(operation))) {
      sourceDependencyMemo[value] = false;
      return false;
    }
    // Break malformed cycles conservatively while descending.
    sourceDependencyMemo[value] = false;
    const bool result =
        llvm::any_of(operation->getOperands(), [&](mlir::Value operand) {
          return mlir::isa<mlir::RankedTensorType>(operand.getType()) &&
                 dependsOnSource(operand);
        });
    sourceDependencyMemo[value] = result;
    return result;
  };
  llvm::SmallVector<mlir::Operation *, 4> reverseChain;
  llvm::DenseSet<mlir::Value> visited;
  while (current != source) {
    if (!current || !visited.insert(current).second)
      return failChain("structured dependency support chain is cyclic");
    mlir::Operation *operation = current.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        operation == producer || !mlir::isMemoryEffectFree(operation))
      return failChain(
          "structured dependency is not a pure in-block support chain");
    if (mlir::isa<mlir::TilingInterface>(operation) &&
        mlir::isa<mlir::DestinationStyleOpInterface>(operation))
      return failChain(
          "structured dependency crosses another scheduled operation");

    mlir::Value tensorSource;
    for (mlir::Value operand : operation->getOperands()) {
      if (!mlir::isa<mlir::RankedTensorType>(operand.getType()))
        continue;
      if (dependsOnSource(operand)) {
        if (tensorSource)
          return failChain((llvm::Twine("structured dependency support op ") +
                            operation->getName().getStringRef() +
                            " has multiple paths from one producer")
                               .str());
        tensorSource = operand;
        continue;
      }
    }
    if (!tensorSource)
      return failChain(
          "structured dependency support operation has no tensor source");
    reverseChain.push_back(operation);
    current = tensorSource;
  }
  std::reverse(reverseChain.begin(), reverseChain.end());
  return reverseChain;
}

namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

struct MappedStrategy {
  SpatialEdgeStrategy strategy;
  mlir::Operation *producer = nullptr;
  mlir::Operation *consumer = nullptr;
  mlir::Operation *supportTemplateProducer = nullptr;
  mlir::Operation *supportTemplateConsumer = nullptr;
  mlir::Value supportTemplateOperand;
  uint64_t consumerScheduleOrdinal = 0;
  bool hasSupportPath = false;
};

struct MaterializedSource {
  mlir::Operation *producer = nullptr;
  unsigned result = 0;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  mlir::Value value;
};

struct MaterializedRegionCutSpill {
  mlir::Operation *producer = nullptr;
  unsigned result = 0;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  mlir::Value value;
  mlir::Value buffer;
};

mlir::LogicalResult validateEdge(mlir::Operation *producer,
                                 unsigned producerResult,
                                 mlir::Operation *consumer,
                                 unsigned consumerOperand, mlir::Block &body,
                                 std::string *failureReason) {
  if (!producer || !consumer || producer->getBlock() != &body ||
      consumer->getBlock() != &body ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands())
    return failResult(failureReason,
                      "edge strategy must name one current-SSA dependency");
  if (mlir::failed(deriveUnaryPureSupportChain(
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
      return failResult(failureReason,
                        "extract_slice support relation is not static unit "
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
      return failResult(failureReason,
                        "insert_slice support relation is not static unit "
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
    return failResult(failureReason,
                      "insert_slice support path names an unknown operand");
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
    return failResult(failureReason, (llvm::Twine("support operation ") +
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
    return failResult(failureReason,
                      "reshape support relation is not statically ranked");

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
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> supportChain =
      deriveUnaryPureSupportChain(producer, producerResult, consumer,
                                  consumerOperand, failureReason);
  if (mlir::failed(supportChain))
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> currentOffsets(producerOffsets.begin(),
                                               producerOffsets.end());
  llvm::SmallVector<int64_t, 4> currentSizes(producerSizes.begin(),
                                             producerSizes.end());
  mlir::Value currentValue = producer->getResult(producerResult);
  for (mlir::Operation *support : *supportChain) {
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

std::optional<uint64_t> getElementBytes(mlir::Type type) {
  unsigned bits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    bits = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    bits = floating.getWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

std::optional<uint64_t> getDomainElements(llvm::ArrayRef<int64_t> sizes) {
  uint64_t result = 1;
  for (int64_t size : sizes) {
    if (size <= 0 || result > std::numeric_limits<uint64_t>::max() /
                                  static_cast<uint64_t>(size))
      return std::nullopt;
    result *= static_cast<uint64_t>(size);
  }
  return result;
}

bool sameEdge(const MappedStrategy &lhs, const MappedStrategy &rhs) {
  return lhs.producer == rhs.producer &&
         lhs.strategy.producerResult == rhs.strategy.producerResult &&
         lhs.consumer == rhs.consumer &&
         lhs.strategy.consumerOperand == rhs.strategy.consumerOperand;
}

bool isContained(llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
                 llvm::ArrayRef<int64_t> containerOffsets,
                 llvm::ArrayRef<int64_t> containerSizes) {
  if (offsets.size() != sizes.size() ||
      offsets.size() != containerOffsets.size() ||
      offsets.size() != containerSizes.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(offsets, sizes, containerOffsets, containerSizes),
      [](auto values) {
        auto [offset, size, containerOffset, containerSize] = values;
        return offset >= 0 && containerOffset >= 0 && size > 0 &&
               containerSize > 0 && size <= containerSize &&
               offset >= containerOffset &&
               offset - containerOffset <= containerSize - size;
      });
}

bool overlaps(llvm::ArrayRef<int64_t> lhsOffsets,
              llvm::ArrayRef<int64_t> lhsSizes,
              llvm::ArrayRef<int64_t> rhsOffsets,
              llvm::ArrayRef<int64_t> rhsSizes) {
  if (lhsOffsets.size() != lhsSizes.size() ||
      lhsOffsets.size() != rhsOffsets.size() ||
      lhsOffsets.size() != rhsSizes.size())
    return true;
  return llvm::all_of(
      llvm::zip_equal(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes),
      [](auto values) {
        auto [lhsOffset, lhsSize, rhsOffset, rhsSize] = values;
        if (lhsOffset < 0 || rhsOffset < 0 || lhsSize <= 0 || rhsSize <= 0)
          return false;
        if (lhsOffset <= rhsOffset)
          return lhsSize > rhsOffset - lhsOffset;
        return rhsSize > lhsOffset - rhsOffset;
      });
}

bool isFullStaticResultDomain(mlir::Operation *operation, unsigned resultNumber,
                              llvm::ArrayRef<int64_t> offsets,
                              llvm::ArrayRef<int64_t> sizes) {
  if (!operation || resultNumber >= operation->getNumResults())
    return false;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  return type && type.hasStaticShape() &&
         offsets.size() == static_cast<size_t>(type.getRank()) &&
         sizes.size() == offsets.size() &&
         llvm::all_of(llvm::zip_equal(offsets, sizes, type.getShape()),
                      [](auto values) {
                        auto [offset, size, extent] = values;
                        return offset == 0 && size == extent;
                      });
}

bool isOneFullTemporalWave(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (selected == operationTemporalTiles.end())
    return true;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return false;
  llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
  return selected->iteratorTileSizes.size() == ranges.size() &&
         llvm::all_of(llvm::zip_equal(selected->iteratorTileSizes, ranges),
                      [](auto values) {
                        auto [tile, extent] = values;
                        return tile > 0 && extent > 0 && tile >= extent;
                      });
}

bool isOneFullTemporalWaveClosure(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  if (!operation)
    return false;
  llvm::DenseSet<mlir::Operation *> visited;
  llvm::SmallVector<mlir::Operation *, 16> worklist{operation};
  while (!worklist.empty()) {
    mlir::Operation *current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;
    if (mlir::isa<mlir::linalg::LinalgOp>(current) &&
        !isOneFullTemporalWave(current, operationTemporalTiles))
      return false;
    for (mlir::Value operand : current->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      if (definition && definition->getBlock() == operation->getBlock())
        worklist.push_back(definition);
    }
  }
  return true;
}

bool isOneFullTemporalWaveForResultDemand(
    mlir::Operation *operation, unsigned resultNumber,
    llvm::ArrayRef<int64_t> resultSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (selected == operationTemporalTiles.end())
    return true;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg || resultNumber >= operation->getNumResults())
    return false;
  llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
  mlir::AffineMap resultMap =
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber));
  if (!resultMap || resultMap.getNumResults() != resultSizes.size() ||
      selected->iteratorTileSizes.size() != ranges.size())
    return false;

  llvm::SmallVector<int64_t, 4> required(ranges.begin(), ranges.end());
  for (auto [resultDimension, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension || loopDimension.getPosition() >= required.size() ||
        resultSizes[resultDimension] <= 0)
      return false;
    required[loopDimension.getPosition()] = resultSizes[resultDimension];
  }
  return llvm::all_of(llvm::zip_equal(selected->iteratorTileSizes, required),
                      [](auto values) {
                        auto [tile, extent] = values;
                        return tile > 0 && extent > 0 && tile >= extent;
                      });
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>> getResultTemporalTileSizes(
    mlir::Operation *operation, unsigned resultNumber,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason) {
  auto resultType = operation && resultNumber < operation->getNumResults()
                        ? mlir::dyn_cast<mlir::RankedTensorType>(
                              operation->getResult(resultNumber).getType())
                        : mlir::RankedTensorType{};
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (!resultType || !resultType.hasStaticShape() || !linalg ||
      selected == operationTemporalTiles.end())
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "peer fragment temporal projection requires one selected static "
        "structured producer");
  mlir::AffineMap resultMap =
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber));
  if (!resultMap ||
      resultMap.getNumResults() != static_cast<unsigned>(resultType.getRank()))
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "peer fragment temporal projection lacks its exact result map");
  llvm::SmallVector<int64_t, 4> resultTiles;
  resultTiles.reserve(resultType.getRank());
  for (auto [resultDimension, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension ||
        loopDimension.getPosition() >= selected->iteratorTileSizes.size())
      return fail<llvm::SmallVector<int64_t, 4>>(
          failureReason,
          "peer fragment temporal projection is not a projected iterator "
          "domain");
    const int64_t tile =
        selected->iteratorTileSizes[loopDimension.getPosition()];
    const int64_t extent = resultType.getDimSize(resultDimension);
    if (tile <= 0 || extent <= 0)
      return fail<llvm::SmallVector<int64_t, 4>>(
          failureReason,
          "peer fragment temporal projection has a nonpositive extent");
    resultTiles.push_back(std::min(tile, extent));
  }
  return resultTiles;
}

mlir::LogicalResult splitIndependentPeerFragmentsAtTemporalWaves(
    llvm::MutableArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason) {
  llvm::DenseMap<int64_t, int64_t> nextPayloadSlice;
  for (SpatialEdgeStrategy &strategy : edgeStrategies) {
    if (strategy.action != SpatialEdgeAction::PeerFragments)
      continue;
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> temporalTiles =
        getResultTemporalTileSizes(strategy.producer, strategy.producerResult,
                                   operationTemporalTiles, failureReason);
    if (mlir::failed(temporalTiles))
      return mlir::failure();
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        strategy.producer->getResult(strategy.producerResult).getType());
    const unsigned elementBits =
        producerType.getElementType().isIntOrFloat()
            ? producerType.getElementType().getIntOrFloatBitWidth()
            : 0;
    if (elementBits == 0 || elementBits % 8 != 0)
      return failResult(
          failureReason,
          "peer fragment temporal projection requires a byte-addressable "
          "element type");

    llvm::SmallVector<SpatialEdgeFragment, 16> splitFragments;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.offsets.size() != temporalTiles->size() ||
          fragment.sizes.size() != temporalTiles->size())
        return failResult(
            failureReason,
            "peer fragment temporal projection has an inconsistent rank");
      if (!isContained(fragment.offsets, fragment.sizes,
                       strategy.producerOffsets, strategy.producerSizes))
        return failResult(
            failureReason,
            "dependent fragment extends outside its consumer demand");
      std::optional<size_t> splitDimension;
      for (size_t dimension = 0; dimension < temporalTiles->size(); ++dimension)
        if ((*temporalTiles)[dimension] < fragment.sizes[dimension]) {
          splitDimension = dimension;
          break;
        }
      llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>, 4>, 4>
          dimensionSegments(temporalTiles->size());
      for (size_t dimension = 0; dimension < temporalTiles->size();
           ++dimension) {
        const int64_t begin = fragment.offsets[dimension];
        const int64_t size = fragment.sizes[dimension];
        const int64_t tile =
            splitDimension == dimension ? (*temporalTiles)[dimension] : size;
        if (begin < 0 || size <= 0 || tile <= 0)
          return failResult(
              failureReason,
              "peer fragment temporal projection has an invalid domain");
        if (begin > std::numeric_limits<int64_t>::max() - size)
          return failResult(
              failureReason,
              "dependent fragment extends outside its consumer demand");
        const int64_t end = begin + size;
        for (int64_t offset = begin; offset < end;) {
          const int64_t nextGrid = ((offset / tile) + 1) * tile;
          const int64_t next = std::min(end, nextGrid);
          dimensionSegments[dimension].push_back({offset, next - offset});
          offset = next;
        }
      }

      llvm::SmallVector<int64_t, 4> offsets(temporalTiles->size());
      llvm::SmallVector<int64_t, 4> sizes(temporalTiles->size());
      std::function<mlir::LogicalResult(size_t)> appendDimension =
          [&](size_t dimension) -> mlir::LogicalResult {
        if (dimension != dimensionSegments.size()) {
          for (auto [offset, size] : dimensionSegments[dimension]) {
            offsets[dimension] = offset;
            sizes[dimension] = size;
            if (mlir::failed(appendDimension(dimension + 1)))
              return mlir::failure();
          }
          return mlir::success();
        }
        uint64_t elements = 1;
        for (int64_t size : sizes) {
          if (elements > std::numeric_limits<uint64_t>::max() /
                             static_cast<uint64_t>(size))
            return failResult(failureReason,
                              "peer temporal fragment byte count overflows");
          elements *= static_cast<uint64_t>(size);
        }
        const uint64_t bytes = elements * (elementBits / 8);
        if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max())
          return failResult(
              failureReason,
              "peer temporal fragment exceeds the target payload range");
        SpatialEdgeFragment split = fragment;
        split.offsets = offsets;
        split.sizes = sizes;
        split.bytes = split.kind == SpatialEdgeFragmentKind::Peer ? bytes : 0;
        if (split.kind == SpatialEdgeFragmentKind::Peer)
          split.payloadSlice = nextPayloadSlice[split.communicationId]++;
        splitFragments.push_back(std::move(split));
        return mlir::success();
      };
      if (mlir::failed(appendDimension(0)))
        return mlir::failure();
    }
    strategy.fragments = std::move(splitFragments);
  }
  return mlir::success();
}

mlir::Value createExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes);

mlir::FailureOr<mlir::Value> getOrMaterializeSource(
    TensorProgramScope scope, mlir::Operation *producer,
    unsigned producerResult, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto existing = llvm::find_if(materialized, [&](const auto &candidate) {
    return candidate.producer == producer &&
           candidate.result == producerResult &&
           llvm::equal(candidate.offsets, offsets) &&
           llvm::equal(candidate.sizes, sizes);
  });
  if (existing != materialized.end())
    return existing->value;
  auto containing = llvm::find_if(materialized, [&](const auto &candidate) {
    return candidate.producer == producer &&
           candidate.result == producerResult &&
           isContained(offsets, sizes, candidate.offsets, candidate.sizes);
  });
  if (containing != materialized.end()) {
    mlir::Operation *definition = containing->value.getDefiningOp();
    if (!definition || definition->getBlock() != &scope.getBody())
      return fail<mlir::Value>(
          failureReason, "materialized producer domain is not available in the "
                         "tensor-program body");
    llvm::SmallVector<int64_t, 4> relativeOffsets;
    relativeOffsets.reserve(offsets.size());
    for (auto [offset, containerOffset] :
         llvm::zip_equal(offsets, containing->offsets))
      relativeOffsets.push_back(offset - containerOffset);
    mlir::OpBuilder builder(definition);
    builder.setInsertionPointAfter(definition);
    mlir::Value slice = createExactSlice(
        builder, producer->getLoc(), containing->value, relativeOffsets, sizes);
    preserved.insert(slice.getDefiningOp());
    materialized.push_back(MaterializedSource{producer, producerResult,
                                              llvm::to_vector(offsets),
                                              llvm::to_vector(sizes), slice});
    return slice;
  }
  // A full-domain, one-wave op is already the exact actual traversal.  Keep
  // that SSA result in place instead of cloning it through the generic tile
  // materializer: query-local peer endpoints refer to its current operands,
  // and replacing them with untracked cloned tensor.empty values would sever
  // the selected receive/send relation before TileRegion conversion.
  if (isFullStaticResultDomain(producer, producerResult, offsets, sizes) &&
      isOneFullTemporalWaveClosure(producer, operationTemporalTiles)) {
    preserved.insert(producer);
    mlir::Value value = producer->getResult(producerResult);
    materialized.push_back(MaterializedSource{producer, producerResult,
                                              llvm::to_vector(offsets),
                                              llvm::to_vector(sizes), value});
    return value;
  }
  mlir::FailureOr<mlir::Value> tiled = materializeCandidateRootTileValue(
      scope, producer, /*outputIndex=*/0, offsets, sizes,
      operationTemporalTiles, failureReason, operationNodes);
  if (mlir::failed(tiled))
    return mlir::failure();
  mlir::Operation *defining = tiled->getDefiningOp();
  if (!defining)
    return fail<mlir::Value>(
        failureReason,
        "selected producer tile did not materialize an operation result");
  preserved.insert(defining);
  materialized.push_back(MaterializedSource{producer, producerResult,
                                            llvm::to_vector(offsets),
                                            llvm::to_vector(sizes), *tiled});
  return *tiled;
}

void eraseDeadExcept(TensorProgramScope scope,
                     const llvm::DenseSet<mlir::Operation *> &preserved) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 32> operations;
    for (mlir::Operation &operation : scope.getBody().without_terminator())
      operations.push_back(&operation);
    for (mlir::Operation *operation : llvm::reverse(operations)) {
      if (preserved.contains(operation) || !mlir::isOpTriviallyDead(operation))
        continue;
      operation->erase();
      changed = true;
    }
  }
}

bool isInBackwardClosure(mlir::Value value, mlir::Operation *needle,
                         llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (definition == needle)
    return true;
  if (llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
        return isInBackwardClosure(operand, needle, visited);
      }))
    return true;

  // Region-bearing SSA definitions such as scf.for/scf.if derive each op
  // result from the corresponding region terminator operand. Lexically
  // captured values used only by a steady-state loop body are not operands of
  // the parent op, so following only definition operands loses a real dataflow
  // dependency. Follow the yielded value for this exact result number; the
  // visited set closes loop-carried cycles.
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;
  const unsigned resultNumber = result.getResultNumber();
  for (mlir::Region &region : definition->getRegions())
    for (mlir::Block &block : region) {
      mlir::Operation *terminator = block.getTerminator();
      if (terminator && resultNumber < terminator->getNumOperands() &&
          isInBackwardClosure(terminator->getOperand(resultNumber), needle,
                              visited))
        return true;
    }
  return false;
}

bool isInSelectedOutputClosure(TensorProgramScope scope,
                               llvm::ArrayRef<SpatialOutputShard> outputShards,
                               mlir::Operation *operation) {
  mlir::func::ReturnOp returnOp = scope.getReturn();
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= returnOp.getNumOperands())
      continue;
    llvm::DenseSet<mlir::Value> visited;
    if (isInBackwardClosure(returnOp.getOperand(shard.outputIndex), operation,
                            visited))
      return true;
  }
  return false;
}

/// Returns whether an ordinary selected output traversal already supplies a
/// repeated consumer-driven wave containing `operation`.  In that case a
/// coupled producer must stay on the consumer's recursive tiling path: eagerly
/// assembling the producer's whole spatial shard would turn one fused wave
/// pipeline into two sequential traversals and erase the selected temporal
/// relationship between the edge endpoints.
bool hasSplitSelectedOutputTraversal(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    mlir::Operation *operation) {
  mlir::func::ReturnOp returnOp = scope.getReturn();
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= returnOp.getNumOperands() ||
        shard.temporalTileSizes == shard.sizes)
      continue;
    llvm::DenseSet<mlir::Value> visited;
    if (isInBackwardClosure(returnOp.getOperand(shard.outputIndex), operation,
                            visited))
      return true;
  }
  return false;
}

mlir::Value createExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sliceType = mlir::RankedTensorType::get(
      sizes, sourceType.getElementType(), sourceType.getEncoding());
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder
      .create<mlir::tensor::ExtractSliceOp>(loc, sliceType, source,
                                            mixedOffsets, mixedSizes, strides)
      .getResult();
}

mlir::Value insertExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source, mlir::Value destination,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder
      .create<mlir::tensor::InsertSliceOp>(loc, source, destination,
                                           mixedOffsets, mixedSizes, strides)
      .getResult();
}

mlir::LogicalResult materializeLocalShardResidency(MappedStrategy &mapped,
                                                   std::string *failureReason) {
  SpatialEdgeStrategy &strategy = mapped.strategy;
  auto producerType = mlir::cast<mlir::RankedTensorType>(
      mapped.producer->getResult(strategy.producerResult).getType());
  mlir::OpBuilder builder(mapped.consumer);
  mlir::Value slice =
      createExactSlice(builder, mapped.producer->getLoc(),
                       mapped.producer->getResult(strategy.producerResult),
                       strategy.producerOffsets, strategy.producerSizes);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      mapped.consumer->getLoc(), producerType.getShape(),
      producerType.getElementType(), mlir::ValueRange{},
      producerType.getEncoding());
  mlir::Value staged = insertExactSlice(
      builder, mapped.consumer->getLoc(), slice, empty.getResult(),
      strategy.producerOffsets, strategy.producerSizes);
  mapped.consumer->setOperand(strategy.consumerOperand, staged);
  return mlir::success();
}

mlir::LogicalResult materializeSpill(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::SmallVectorImpl<MaterializedRegionCutSpill>
        &materializedRegionCutSpills,
    llvm::SmallVectorImpl<CandidateSelectedDDRStage> &selectedDDRStages,
    bool &createdRegionCut,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto wireStoredProducerToConsumer = [&](mlir::Value storedProducer) {
    if (mapped.hasSupportPath) {
      // A support DAG may join several independently spilled structured
      // producers (for example tensor.insert_slice assembly). Rebind the
      // complete consumer operand once every selected producer spill exists;
      // doing it edge-by-edge would leave the other producer's SPM value in
      // the cloned support operation.
      return mlir::success();
    }
    mapped.consumer->setOperand(mapped.strategy.consumerOperand,
                                storedProducer);
    return mlir::success();
  };

  SpatialEdgeStrategy &strategy = mapped.strategy;
  createdRegionCut = false;
  if (strategy.action == SpatialEdgeAction::RegionCut) {
    auto existing =
        llvm::find_if(materializedRegionCutSpills,
                      [&](const MaterializedRegionCutSpill &spill) {
                        return spill.producer == mapped.producer &&
                               spill.result == strategy.producerResult &&
                               spill.offsets == strategy.producerOffsets &&
                               spill.sizes == strategy.producerSizes;
                      });
    if (existing != materializedRegionCutSpills.end()) {
      return wireStoredProducerToConsumer(existing->value);
    }
  }
  auto producerType = mlir::cast<mlir::RankedTensorType>(
      mapped.producer->getResult(strategy.producerResult).getType());
  mlir::Location loc = mapped.consumer->getLoc();
  auto ddrType = mlir::MemRefType::get(
      producerType.getShape(), producerType.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                      MemLayout::Tensor));
  mlir::OpBuilder builder(mapped.consumer);
  auto allocation = builder.create<mlir::memref::AllocOp>(loc, ddrType);
  // This allocation is the selected edge action itself, not an incidental
  // source-program side effect. Keep it explicit when a Tile owns only a
  // subset of observable outputs; the partial-output purity check can then
  // distinguish selected storage from unknown source effects.
  preserved.insert(allocation.getOperation());
  auto destination = builder.create<mlir::bufferization::ToTensorOp>(
      loc, allocation.getResult(), /*restrict=*/true, /*writable=*/true);
  preserved.insert(destination.getOperation());
  mlir::FailureOr<mlir::Value> slice = getOrMaterializeSource(
      scope, mapped.producer, strategy.producerResult, strategy.producerOffsets,
      strategy.producerSizes, materialized, preserved, operationTemporalTiles,
      failureReason, operationNodes);
  if (mlir::failed(slice))
    return mlir::failure();
  mlir::Value stored =
      insertExactSlice(builder, loc, *slice, destination.getResult(),
                       strategy.producerOffsets, strategy.producerSizes);
  if (strategy.action == SpatialEdgeAction::RegionCut) {
    // The functional insert result still exposes its producer slice to the
    // generic tensor tiler. If it is wired directly to the consumer, temporal
    // consumer materialization can clone the insert (and therefore the spill)
    // into every consumer wave, silently composing two supposedly independent
    // op stages. Keep the exact store alive, then reopen the same compiler-
    // owned DDR allocation as a read-only SSA boundary. TileRegion lowering
    // sees the store before the new view and materializes later consumer loads
    // without a tensor-level producer path to fuse across the cut.
    preserved.insert(stored.getDefiningOp());
    auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
        loc, allocation.getResult(), /*restrict=*/false,
        /*writable=*/false);
    preserved.insert(sealed.getOperation());
    mlir::Value sealedSlice =
        createExactSlice(builder, loc, sealed.getResult(),
                         strategy.producerOffsets, strategy.producerSizes);
    preserved.insert(sealedSlice.getDefiningOp());

    // One independently executed producer may fan out to several later op
    // stages or peer sends. They must all read the stored DDR value;
    // retaining a cached SPM traversal for a later fanout would keep that SPM
    // value live across this cut. Preserve only the current store's backward
    // slice, and redirect every already-materialized external fanout plus the
    // exact source cache to the compact read-only DDR view.
    llvm::DenseSet<mlir::Value> visitedValues;
    llvm::DenseSet<mlir::Operation *> storeDependencies;
    llvm::SmallVector<mlir::Value, 16> worklist{stored};
    while (!worklist.empty()) {
      mlir::Value value = worklist.pop_back_val();
      if (!value || !visitedValues.insert(value).second)
        continue;
      mlir::Operation *definition = value.getDefiningOp();
      if (!definition || !storeDependencies.insert(definition).second)
        continue;
      worklist.append(definition->operand_begin(), definition->operand_end());
    }

    // A previously materialized fanout (most notably a resident contribution
    // to a peer-fragment DDR assembly) can lexically precede this deferred
    // RegionCut. Replacing that use with the fresh sealed view without moving
    // the selected store would create a use-before-definition and, more
    // importantly, put the fanout on the wrong side of the explicit op-stage
    // boundary. Place the complete static DDR store/reopen chain immediately
    // before the earliest such use. The original producer slice already
    // dominates every one of its uses, so this preserves SSA and makes the
    // concrete operation order match the selected baseline schedule.
    auto getTopLevelInScope = [&](mlir::Operation *operation) {
      while (operation && operation->getBlock() != &scope.getBody())
        operation = operation->getParentOp();
      return operation;
    };
    mlir::Operation *earliestExternalFanout = nullptr;
    for (mlir::OpOperand &use : slice->getUses()) {
      if (storeDependencies.contains(use.getOwner()))
        continue;
      mlir::Operation *root = getTopLevelInScope(use.getOwner());
      if (!root)
        continue;
      if (!earliestExternalFanout ||
          root->isBeforeInBlock(earliestExternalFanout))
        earliestExternalFanout = root;
    }
    mlir::Operation *sealedSliceDefinition = sealedSlice.getDefiningOp();
    if (earliestExternalFanout &&
        sealedSliceDefinition->getBlock() ==
            earliestExternalFanout->getBlock() &&
        !sealedSliceDefinition->isBeforeInBlock(earliestExternalFanout)) {
      allocation->moveBefore(earliestExternalFanout);
      destination->moveAfter(allocation);
      stored.getDefiningOp()->moveAfter(destination);
      sealed->moveAfter(stored.getDefiningOp());
      sealedSliceDefinition->moveAfter(sealed);
    }
    for (mlir::OpOperand &use : llvm::make_early_inc_range(slice->getUses()))
      if (!storeDependencies.contains(use.getOwner()))
        use.set(sealedSlice);
    for (MaterializedSource &cached : materialized) {
      if (cached.producer != mapped.producer ||
          cached.result != strategy.producerResult ||
          !isContained(cached.offsets, cached.sizes, strategy.producerOffsets,
                       strategy.producerSizes))
        continue;
      if (cached.offsets == strategy.producerOffsets &&
          cached.sizes == strategy.producerSizes) {
        cached.value = sealedSlice;
        continue;
      }
      cached.value = createExactSlice(builder, loc, sealed.getResult(),
                                      cached.offsets, cached.sizes);
      preserved.insert(cached.value.getDefiningOp());
    }
    stored = sealed.getResult();
    materializedRegionCutSpills.push_back(MaterializedRegionCutSpill{
        mapped.producer, strategy.producerResult, strategy.producerOffsets,
        strategy.producerSizes, stored, allocation.getResult()});
    selectedDDRStages.push_back(
        CandidateSelectedDDRStage{allocation.getResult(), &mapped.strategy});
    createdRegionCut = true;
  }
  if (mlir::failed(wireStoredProducerToConsumer(stored)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult materializeRecompute(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  if (!mlir::isMemoryEffectFree(mapped.producer))
    return failResult(failureReason,
                      "recompute requires a pure current-SSA producer");
  // The producer op-wave remains an actual scheduled obligation. Materialize
  // its selected domain first, then wire an independent cloned producer into
  // the consumer traversal. Across a CardModule this is the concrete
  // distinction between retention and recomputation, even when a focused
  // single-Tile test places both obligations on one endpoint.
  mlir::FailureOr<mlir::Value> original = getOrMaterializeSource(
      scope, mapped.producer, mapped.strategy.producerResult,
      mapped.strategy.producerOffsets, mapped.strategy.producerSizes,
      materialized, preserved, operationTemporalTiles, failureReason,
      &operationNodes);
  if (mlir::failed(original))
    return mlir::failure();
  mlir::OpBuilder builder(mapped.consumer);
  mlir::IRMapping mapping;
  mlir::Operation *clone = builder.clone(*mapped.producer, mapping);
  if (!clone || clone->getNumResults() != mapped.producer->getNumResults())
    return failResult(failureReason,
                      "recompute could not clone the exact producer");
  auto sourceNode = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &entry) {
        return entry.operation == mapped.producer;
      });
  if (sourceNode != operationNodes.end())
    operationNodes.push_back({clone, sourceNode->structuredNodeId});
  mapped.consumer->setOperand(mapped.strategy.consumerOperand,
                              clone->getResult(mapped.strategy.producerResult));
  return mlir::success();
}

mlir::Value getViewRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
        value.getDefiningOp());
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

[[maybe_unused]] bool
isPeerEndpointForStrategy(mlir::Operation *operation,
                          const SpatialEdgeStrategy &strategy) {
  auto matches = [&](int64_t peer, uint64_t bytes, DTEMessageAttr message,
                     bool receive) {
    return llvm::any_of(
        strategy.fragments, [&](const SpatialEdgeFragment &fragment) {
          if (fragment.kind != SpatialEdgeFragmentKind::Peer ||
              fragment.bytes != bytes ||
              fragment.communicationId != message.getCommunicationId() ||
              fragment.payloadSlice != message.getPayloadSlice())
            return false;
          return receive ? fragment.sourceTile.getValue() == peer
                         : strategy.destinationTile.getValue() == peer;
        });
  };
  if (auto receive = mlir::dyn_cast<CommPeerRecvOp>(operation))
    return matches(receive.getPeer(), receive.getBytes(), receive.getMessage(),
                   /*receive=*/true);
  if (auto send = mlir::dyn_cast<CommPeerSendOp>(operation))
    return matches(send.getPeer(), send.getBytes(), send.getMessage(),
                   /*receive=*/false);
  return false;
}

mlir::LogicalResult rebindSelectedReceiveEndpoints(
    llvm::MutableArrayRef<CandidatePeerEndpoint> endpoints,
    llvm::ArrayRef<MappedStrategy> mappedStrategies,
    std::string *failureReason) {
  for (CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
        !endpoint.selectedFragment)
      continue;
    llvm::SmallVector<mlir::bufferization::ToTensorOp, 2> matches;
    if (endpoint.carrierBuffer)
      for (mlir::Operation *user : endpoint.carrierBuffer.getUsers())
        if (auto toTensor =
                mlir::dyn_cast<mlir::bufferization::ToTensorOp>(user);
            toTensor && toTensor.getMemref() == endpoint.carrierBuffer &&
            !toTensor.getResult().use_empty())
          matches.push_back(toTensor);
    if (matches.size() != 1 || matches.front().getResult().use_empty()) {
      std::string detail;
      llvm::raw_string_ostream diagnostic(detail);
      diagnostic << "selected receive fragment did not survive as one exact "
                    "consumer value (matches="
                 << matches.size();
      if (matches.size() == 1 && matches.front().getResult().use_empty())
        diagnostic << ", unused";
      diagnostic << ", peer=" << endpoint.peer.getValue()
                 << ", communication_id=" << endpoint.communicationId
                 << ", payload_slice=" << endpoint.payloadSlice;
      if (endpoint.selectedFragment) {
        diagnostic << ", offsets=[";
        llvm::interleaveComma(endpoint.selectedFragment->offsets, diagnostic);
        diagnostic << "], sizes=[";
        llvm::interleaveComma(endpoint.selectedFragment->sizes, diagnostic);
        diagnostic << ']';
      }
      for (const MappedStrategy &mapped : mappedStrategies) {
        auto fragment =
            llvm::find_if(mapped.strategy.fragments,
                          [&](const SpatialEdgeFragment &candidate) {
                            return &candidate == endpoint.selectedFragment;
                          });
        if (fragment == mapped.strategy.fragments.end())
          continue;
        mlir::Operation *producer = mapped.supportTemplateProducer;
        mlir::Operation *consumer = mapped.supportTemplateConsumer;
        if (!producer || !consumer)
          break;
        diagnostic
            << ", edge=" << producer->getName() << ':'
            << producer->getResult(mapped.strategy.producerResult).getType();
        diagnostic
            << " -> " << consumer->getName() << " operand "
            << mapped.strategy.consumerOperand << ':'
            << consumer->getOperand(mapped.strategy.consumerOperand).getType();
        diagnostic << ", support=" << mapped.hasSupportPath
                   << ", producer_demand_offsets=[";
        llvm::interleaveComma(mapped.strategy.producerOffsets, diagnostic);
        diagnostic << "], producer_demand_sizes=[";
        llvm::interleaveComma(mapped.strategy.producerSizes, diagnostic);
        diagnostic << "], consumer_offsets=[";
        llvm::interleaveComma(mapped.strategy.consumerOffsets, diagnostic);
        diagnostic << "], consumer_sizes=[";
        llvm::interleaveComma(mapped.strategy.consumerSizes, diagnostic);
        diagnostic << ']';
        break;
      }
      diagnostic << ')';
      return failResult(failureReason, diagnostic.str());
    }
    endpoint.value = matches.front().getResult();
  }
  return mlir::success();
}

/// Split the one ordinary TileRegion at the store/reload associated with a
/// selected RegionCut.  The prefix yields only compiler-owned DDR; the suffix
/// reloads it through a new region argument.  No SPM value is allowed to cross.
mlir::LogicalResult
splitAtRegionCut(mlir::memref::AllocOp spillAllocation,
                 const SpatialEdgeStrategy *marker,
                 llvm::ArrayRef<mlir::Value> selectedDDRStageBuffers,
                 StructuredMaterializationRelations *materializationRelations,
                 std::string *failureReason) {
  TileRegionOp region = spillAllocation
                            ? spillAllocation->getParentOfType<TileRegionOp>()
                            : TileRegionOp{};
  if (!region || !spillAllocation || !region.getBody().hasOneBlock() ||
      !marker || !isWaferDDRMemRefType(spillAllocation.getType()))
    return failResult(failureReason,
                      "region cut did not materialize its compiler-owned DDR");

  mlir::Block &body = region.getBody().front();
  auto getTopLevelOperation = [&](mlir::Operation *operation) {
    while (operation && operation->getBlock() != &body)
      operation = operation->getParentOp();
    return operation;
  };

  llvm::SmallVector<mlir::Operation *, 4> storeRoots;
  llvm::SmallVector<mlir::Operation *, 4> reloadRoots;
  region.walk([&](StorageStoreOp candidate) {
    if (getViewRoot(candidate.getDest()) != spillAllocation.getResult())
      return;
    mlir::Operation *root = getTopLevelOperation(candidate);
    if (root && !llvm::is_contained(storeRoots, root))
      storeRoots.push_back(root);
  });
  region.walk([&](StorageLoadOp candidate) {
    if (getViewRoot(candidate.getSource()) != spillAllocation.getResult())
      return;
    mlir::Operation *root = getTopLevelOperation(candidate);
    if (root && !llvm::is_contained(reloadRoots, root))
      reloadRoots.push_back(root);
  });
  llvm::sort(storeRoots, [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  mlir::Operation *lastStoreRoot =
      storeRoots.empty() ? nullptr : storeRoots.back();
  const bool hasReloadAfterStore =
      lastStoreRoot && llvm::any_of(reloadRoots, [&](mlir::Operation *reload) {
        return reload != lastStoreRoot &&
               lastStoreRoot->isBeforeInBlock(reload);
      });
  if (!lastStoreRoot || !hasReloadAfterStore) {
    std::string allocationDetails;
    llvm::raw_string_ostream details(allocationDetails);
    size_t allocationStores = 0;
    size_t allocationLoads = 0;
    region.walk([&](StorageStoreOp candidate) {
      if (getViewRoot(candidate.getDest()) == spillAllocation.getResult())
        ++allocationStores;
    });
    region.walk([&](StorageLoadOp candidate) {
      if (getViewRoot(candidate.getSource()) == spillAllocation.getResult())
        ++allocationLoads;
    });
    details << "stores=" << allocationStores << "/loads=" << allocationLoads
            << "/type=" << spillAllocation.getType();
    std::string storeDetails;
    llvm::raw_string_ostream stores(storeDetails);
    size_t totalStores = 0;
    region.walk([&](StorageStoreOp candidate) {
      ++totalStores;
      if (totalStores > 12)
        return;
      mlir::Value root = getViewRoot(candidate.getDest());
      stores << (totalStores == 1 ? "" : ",")
             << "dest=" << candidate.getDest().getType()
             << "/root=" << root.getType();
    });
    std::string orderDetails;
    llvm::raw_string_ostream order(orderDetails);
    for (auto [index, operation] :
         llvm::enumerate(region.getBody().front().without_terminator())) {
      bool storesStage = false;
      bool loadsStage = false;
      operation.walk([&](StorageStoreOp store) {
        storesStage |=
            getViewRoot(store.getDest()) == spillAllocation.getResult();
      });
      operation.walk([&](StorageLoadOp load) {
        loadsStage |=
            getViewRoot(load.getSource()) == spillAllocation.getResult();
      });
      if (storesStage || loadsStage)
        order << (orderDetails.empty() ? "" : ",") << index << ':'
              << operation.getName() << "/store=" << storesStage
              << "/load=" << loadsStage;
    }
    return failResult(
        failureReason,
        "region cut requires an ordered exact store/reload "
        "interval (store_roots=" +
            std::to_string(storeRoots.size()) + ", reload_roots=" +
            std::to_string(reloadRoots.size()) + ", marked_allocations=[" +
            details.str() + "], total_stores=" + std::to_string(totalStores) +
            ", stores=[" + stores.str() + "], order=[" + order.str() + "])");
  }

  llvm::DenseSet<mlir::Operation *> prefix;
  std::function<mlir::LogicalResult(mlir::Operation *)> collectPrefix =
      [&](mlir::Operation *operation) -> mlir::LogicalResult {
    if (!operation || operation == spillAllocation.getOperation() ||
        operation->getBlock() != &body || mlir::isa<TileYieldOp>(operation) ||
        !prefix.insert(operation).second)
      return mlir::success();
    mlir::WalkResult dependencyResult =
        operation->walk([&](mlir::Operation *nested) -> mlir::WalkResult {
          for (mlir::Value operand : nested->getOperands()) {
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand);
                argument && argument.getOwner() == &body)
              continue;
            mlir::Operation *definition = operand.getDefiningOp();
            mlir::Operation *root = getTopLevelOperation(definition);
            if (root && root != operation && mlir::failed(collectPrefix(root)))
              return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (dependencyResult.wasInterrupted())
      return mlir::failure();
    return mlir::success();
  };
  llvm::SmallVector<mlir::Value, 4> storeSourceRoots;
  for (mlir::Operation *storeRoot : storeRoots) {
    if (mlir::failed(collectPrefix(storeRoot)))
      return mlir::failure();
    storeRoot->walk([&](StorageStoreOp store) {
      mlir::Value sourceRoot = getViewRoot(store.getSource());
      if (!llvm::is_contained(storeSourceRoots, sourceRoot))
        storeSourceRoots.push_back(sourceRoot);
    });
  }

  // Peer endpoints access existing SPM allocations, so their writes/reads are
  // not represented by SSA edges from a later StorageStore. Find the last
  // endpoint owned by this exact stage, then preserve the complete preceding
  // transport order. Moving only the owned receives ahead of intervening
  // sends changes the per-Tile wait order and can manufacture a card
  // DTE cycle even though every receive buffer is otherwise initialized.
  mlir::Operation *lastOwnedPeerActionRoot = nullptr;
  auto ownsPeerEndpoint = [&](mlir::Operation *endpoint, mlir::Value buffer) {
    if (isPeerEndpointForStrategy(endpoint, *marker))
      return true;
    return llvm::is_contained(storeSourceRoots, getViewRoot(buffer));
  };
  auto visitPeerEndpoint = [&](mlir::Operation *endpoint, mlir::Value buffer) {
    if (!ownsPeerEndpoint(endpoint, buffer))
      return;
    mlir::Operation *root = getTopLevelOperation(endpoint);
    if (root && (!lastOwnedPeerActionRoot ||
                 lastOwnedPeerActionRoot->isBeforeInBlock(root)))
      lastOwnedPeerActionRoot = root;
  };
  region.walk([&](CommPeerRecvOp receive) {
    visitPeerEndpoint(receive.getOperation(), receive.getBuffer());
  });
  region.walk([&](CommPeerSendOp send) {
    visitPeerEndpoint(send.getOperation(), send.getBuffer());
  });
  auto collectTransportOrderThrough =
      [&](mlir::Operation *lastAction) -> mlir::LogicalResult {
    llvm::SmallVector<mlir::Operation *, 16> orderedPeerActions;
    auto collectOrderedPeerEndpoint = [&](mlir::Operation *endpoint,
                                          mlir::Value token) {
      mlir::Operation *root = getTopLevelOperation(endpoint);
      if (!root || (root != lastAction && !root->isBeforeInBlock(lastAction)))
        return;
      if (!llvm::is_contained(orderedPeerActions, root))
        orderedPeerActions.push_back(root);
      for (mlir::Operation *user : token.getUsers()) {
        mlir::Operation *completionRoot = getTopLevelOperation(user);
        if (completionRoot &&
            !llvm::is_contained(orderedPeerActions, completionRoot))
          orderedPeerActions.push_back(completionRoot);
      }
    };
    region.walk([&](CommPeerRecvOp receive) {
      collectOrderedPeerEndpoint(receive.getOperation(), receive.getToken());
    });
    region.walk([&](CommPeerSendOp send) {
      collectOrderedPeerEndpoint(send.getOperation(), send.getToken());
    });
    for (mlir::Operation *action : orderedPeerActions)
      if (mlir::failed(collectPrefix(action)))
        return mlir::failure();
    return mlir::success();
  };
  if (lastOwnedPeerActionRoot &&
      mlir::failed(collectTransportOrderThrough(lastOwnedPeerActionRoot)))
    return mlir::failure();

  // A send may observe a selected SPM result before the same result is sealed
  // into compiler-owned DDR. Pulling the send's backward closure into the
  // prefix must also pull that exact sealing store; otherwise the SPM value
  // would acquire a suffix use and illegally cross the TileRegion boundary.
  // Close only StorageStore effects, not arbitrary forward consumers.
  // Conversely, a sealing store can pull a producer into the prefix while a
  // later send still observes it. Grow one small effect fixed point across
  // stores, sends and all earlier transport actions; ordinary compute users
  // remain outside this closure.
  bool addedEffect = true;
  while (addedEffect) {
    addedEffect = false;

    // A StorageLoad initializes an existing SPM allocation and therefore has
    // no SSA result connecting it to a later compute or send. If a prefix
    // operation reads that buffer, the initializing load belongs to the same
    // side of the cut. This is the memory-effect counterpart of collectPrefix:
    // without it the allocation can be cloned into the prefix while its load
    // remains as an unread operation in the suffix.
    llvm::DenseSet<mlir::Value> prefixReadBuffers;
    auto recordSPMRead = [&](mlir::Value value) {
      mlir::Value root = getViewRoot(value);
      if (root && isWaferSPMMemRefType(root.getType()))
        prefixReadBuffers.insert(root);
    };
    for (mlir::Operation *operation : prefix)
      operation->walk([&](mlir::Operation *nested) {
        if (mlir::isa<StorageLoadOp, CommPeerRecvOp, mlir::memref::DeallocOp>(
                nested))
          return;
        if (auto store = mlir::dyn_cast<StorageStoreOp>(nested)) {
          recordSPMRead(store.getSource());
          return;
        }
        if (auto send = mlir::dyn_cast<CommPeerSendOp>(nested)) {
          recordSPMRead(send.getBuffer());
          return;
        }
        for (mlir::Value operand : nested->getOperands())
          recordSPMRead(operand);
      });
    llvm::SmallVector<mlir::Operation *, 4> initializingLoads;
    region.walk([&](StorageLoadOp load) {
      mlir::Operation *loadRoot = getTopLevelOperation(load);
      if (loadRoot && !prefix.contains(loadRoot) &&
          prefixReadBuffers.contains(getViewRoot(load.getDest())) &&
          !llvm::is_contained(initializingLoads, loadRoot))
        initializingLoads.push_back(loadRoot);
    });
    for (mlir::Operation *loadRoot : initializingLoads) {
      if (mlir::failed(collectPrefix(loadRoot)))
        return mlir::failure();
      addedEffect = true;
    }

    llvm::SmallVector<mlir::Operation *, 4> sealingStores;
    region.walk([&](StorageStoreOp store) {
      mlir::Operation *storeRoot = getTopLevelOperation(store);
      mlir::Operation *sourceRoot =
          getTopLevelOperation(store.getSource().getDefiningOp());
      if (storeRoot && sourceRoot && !prefix.contains(storeRoot) &&
          prefix.contains(sourceRoot) &&
          !llvm::is_contained(sealingStores, storeRoot))
        sealingStores.push_back(storeRoot);
    });
    for (mlir::Operation *storeRoot : sealingStores) {
      if (mlir::failed(collectPrefix(storeRoot)))
        return mlir::failure();
      addedEffect = true;
    }

    mlir::Operation *expandedTransportRoot = lastOwnedPeerActionRoot;
    region.walk([&](CommPeerSendOp send) {
      mlir::Operation *sendRoot = getTopLevelOperation(send);
      mlir::Operation *bufferRoot =
          getTopLevelOperation(getViewRoot(send.getBuffer()).getDefiningOp());
      if (!sendRoot || !bufferRoot || prefix.contains(sendRoot) ||
          !prefix.contains(bufferRoot))
        return;
      if (!expandedTransportRoot ||
          expandedTransportRoot->isBeforeInBlock(sendRoot))
        expandedTransportRoot = sendRoot;
    });
    if (expandedTransportRoot != lastOwnedPeerActionRoot) {
      lastOwnedPeerActionRoot = expandedTransportRoot;
      if (mlir::failed(collectTransportOrderThrough(lastOwnedPeerActionRoot)))
        return mlir::failure();
      addedEffect = true;
    }
  }

  // A compiler-owned residency release belongs to the same side of the cut
  // as the value it releases.  Temporal materialization emits memref.dealloc
  // after the value's last tensor-level observation; when RegionCut replaces
  // a later consumer with store/reload, that last observation becomes the
  // prefix spill store.  Move the marker with the prefix instead of treating
  // it as shaped data that must cross the DDR boundary.
  llvm::SmallVector<mlir::memref::DeallocOp, 4> prefixReleases;
  for (mlir::Operation *operation : prefix)
    for (mlir::Value result : operation->getResults())
      for (mlir::Operation *user : result.getUsers())
        if (auto release = mlir::dyn_cast<mlir::memref::DeallocOp>(user);
            release && release.getMemref() == result &&
            release->getBlock() == &body)
          prefixReleases.push_back(release);
  prefix.insert(prefixReleases.begin(), prefixReleases.end());

  // Source-only peer producers are sealed in compiler-owned DDR even though
  // they do not own a destination-side edge marker on this Tile. A later
  // selected cut may need that same allocation on both sides (for example,
  // one prefix peer send and one suffix consumer/send). Duplicating the alloc
  // would create two unrelated DDR objects; carrying an SPM value would be
  // illegal. Hoist only unmarked static DDR allocations that are proven used
  // on both sides and pass the one object explicitly to both regions.
  llvm::SmallVector<mlir::memref::AllocOp, 4> eligibleSharedDDRAllocations;
  llvm::DenseMap<mlir::Value, size_t> eligibleSharedDDRIndices;
  llvm::SmallVector<std::pair<bool, bool>, 4> sharedDDRUses;
  llvm::SmallVector<mlir::memref::AllocOp, 4> sharedDDRAllocations;
  llvm::DenseSet<mlir::Operation *> sharedDDRAllocationOps;
  for (mlir::Operation &operation : body.without_terminator()) {
    auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation);
    if (!allocation || allocation == spillAllocation ||
        !isWaferDDRMemRefType(allocation.getType()) ||
        !allocation.getDynamicSizes().empty() ||
        !allocation.getSymbolOperands().empty() ||
        llvm::is_contained(selectedDDRStageBuffers, allocation.getResult()))
      continue;
    eligibleSharedDDRIndices.try_emplace(allocation.getResult(),
                                         eligibleSharedDDRAllocations.size());
    eligibleSharedDDRAllocations.push_back(allocation);
    sharedDDRUses.push_back({false, false});
  }
  // Classify every eligible allocation in one region walk. Scanning the
  // complete region separately for every allocation made an independent
  // op-stage chain cubic in practice once this split was repeated for every
  // selected edge.
  region.walk([&](mlir::Operation *user) {
    mlir::Operation *root = getTopLevelOperation(user);
    if (!root)
      return;
    for (mlir::Value operand : user->getOperands()) {
      auto found = eligibleSharedDDRIndices.find(getViewRoot(operand));
      if (found == eligibleSharedDDRIndices.end())
        continue;
      auto &uses = sharedDDRUses[found->second];
      if (prefix.contains(root))
        uses.first = true;
      else
        uses.second = true;
    }
  });
  for (auto [index, allocation] :
       llvm::enumerate(eligibleSharedDDRAllocations)) {
    if (sharedDDRUses[index].first && sharedDDRUses[index].second) {
      sharedDDRAllocations.push_back(allocation);
      sharedDDRAllocationOps.insert(allocation.getOperation());
    }
  }

  auto isBoundaryView = [&](mlir::Operation *operation) {
    auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(operation);
    if (!view)
      return false;
    mlir::Value root = getViewRoot(view.getViewSource());
    if (root == spillAllocation.getResult())
      return true;
    if (auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
        allocation &&
        sharedDDRAllocationOps.contains(allocation.getOperation()))
      return true;
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(root);
    return argument && argument.getOwner() == &body;
  };

  // A temporal loop or another prefix operation may produce additional DDR
  // state that a later independent op stage consumes. Such values are legal
  // TileRegion boundaries and must be explicit region results/inputs. Tensor
  // and SPM values remain forbidden: admitting only typed DDR memrefs keeps
  // the selected op boundary concrete without turning this into an opaque
  // cross-region value channel.
  llvm::SmallVector<mlir::Value, 8> carriedDDRValues;
  llvm::DenseSet<mlir::Value> carriedDDRValueSet;
  for (mlir::Operation &operation : body.without_terminator()) {
    if (!prefix.contains(&operation))
      continue;
    for (mlir::Value result : operation.getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        mlir::Operation *useRoot = getTopLevelOperation(use.getOwner());
        if (useRoot && prefix.contains(useRoot))
          continue;
        // Scalar constants, static allocations, and views rooted in a region
        // input or the selected spill are region-local resources, not
        // produced SSA data. Reconstruct such views on each side of the cut;
        // an additional produced DDR value crosses as an explicit typed
        // result/input. No tensor or SPM value may bypass the selected
        // spill/reload boundary.
        if (mlir::isa<mlir::arith::ConstantOp, mlir::memref::AllocOp>(
                &operation) ||
            isBoundaryView(&operation))
          continue;
        if (isWaferDDRMemRefType(result.getType())) {
          if (carriedDDRValueSet.insert(result).second)
            carriedDDRValues.push_back(result);
          break;
        }
        std::string detail;
        llvm::raw_string_ostream stream(detail);
        mlir::Value root = getViewRoot(result);
        stream << "region cut has a producer value from " << operation.getName()
               << " used by " << use.getOwner()->getName();
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root))
          stream << " (view root is block argument " << argument.getArgNumber()
                 << ')';
        else if (mlir::Operation *rootDefinition = root.getDefiningOp())
          stream << " (view root is " << rootDefinition->getName() << ')';
        stream << " that bypasses the DDR boundary; operation=";
        operation.print(stream, mlir::OpPrintingFlags().skipRegions());
        stream << "; spill=" << spillAllocation.getType()
               << "; last_store_root=" << lastStoreRoot->getName();
        return failResult(failureReason, stream.str());
      }
    }
  }

  mlir::OpBuilder outer(region);
  auto externalSpill = outer.create<mlir::memref::AllocOp>(
      spillAllocation.getLoc(), spillAllocation.getType());
  llvm::SmallVector<mlir::Value, 4> externalSharedDDR;
  for (mlir::memref::AllocOp allocation : sharedDDRAllocations)
    externalSharedDDR.push_back(
        outer
            .create<mlir::memref::AllocOp>(allocation.getLoc(),
                                           allocation.getType())
            .getResult());

  llvm::SmallVector<mlir::Value, 8> prefixInputs(region.getInputs().begin(),
                                                 region.getInputs().end());
  prefixInputs.push_back(externalSpill.getResult());
  prefixInputs.append(externalSharedDDR);
  llvm::SmallVector<mlir::Type, 8> prefixResultTypes{spillAllocation.getType()};
  for (mlir::Value value : carriedDDRValues)
    prefixResultTypes.push_back(value.getType());
  auto prefixRegion = outer.create<TileRegionOp>(
      region.getLoc(), prefixResultTypes, prefixInputs);
  prefixRegion.getBody().push_back(new mlir::Block());
  mlir::Block &prefixBody = prefixRegion.getBody().front();
  for (mlir::Value input : prefixRegion.getInputs())
    prefixBody.addArgument(input.getType(), input.getLoc());

  llvm::SmallVector<mlir::Value, 8> suffixInputs(region.getInputs().begin(),
                                                 region.getInputs().end());
  suffixInputs.append(prefixRegion.getResults().begin(),
                      prefixRegion.getResults().end());
  suffixInputs.append(externalSharedDDR);
  auto suffixRegion = outer.create<TileRegionOp>(
      region.getLoc(), region.getResultTypes(), suffixInputs);
  suffixRegion.getBody().push_back(new mlir::Block());
  mlir::Block &suffixBody = suffixRegion.getBody().front();
  for (mlir::Value input : suffixRegion.getInputs())
    suffixBody.addArgument(input.getType(), input.getLoc());

  mlir::IRMapping prefixMapping;
  mlir::IRMapping suffixMapping;
  for (auto [original, replacement] : llvm::zip_equal(
           body.getArguments(),
           prefixBody.getArguments().take_front(body.getNumArguments())))
    prefixMapping.map(original, replacement);
  for (auto [original, replacement] : llvm::zip_equal(
           body.getArguments(),
           suffixBody.getArguments().take_front(body.getNumArguments())))
    suffixMapping.map(original, replacement);
  prefixMapping.map(spillAllocation.getResult(),
                    prefixBody.getArgument(body.getNumArguments()));
  suffixMapping.map(spillAllocation.getResult(),
                    suffixBody.getArgument(body.getNumArguments()));
  for (auto [index, value] : llvm::enumerate(carriedDDRValues))
    suffixMapping.map(
        value, suffixBody.getArgument(body.getNumArguments() + 1 + index));
  for (auto [index, allocation] : llvm::enumerate(sharedDDRAllocations)) {
    prefixMapping.map(
        allocation.getResult(),
        prefixBody.getArgument(body.getNumArguments() + 1 + index));
    suffixMapping.map(allocation.getResult(),
                      suffixBody.getArgument(body.getNumArguments() +
                                             prefixRegion.getNumResults() +
                                             index));
  }

  // Discover the small set of prefix-owned resources that the suffix must
  // duplicate before moving either side. The prior implementation cloned
  // every prefix and suffix operation for every selected cut, making a chain
  // of independent DDR stages quadratic in the remaining IR size. Moving
  // each ordinary operation preserves its exact SSA body and makes only true
  // two-sided resources pay a clone cost.
  mlir::OpBuilder suffixBuilder = mlir::OpBuilder::atBlockEnd(&suffixBody);
  std::function<mlir::LogicalResult(mlir::Operation *)> cloneResource =
      [&](mlir::Operation *operation) -> mlir::LogicalResult {
    if (operation->getNumResults() == 0)
      return failResult(failureReason,
                        "region cut suffix resource has no result");
    if (llvm::all_of(operation->getResults(), [&](mlir::Value result) {
          return suffixMapping.contains(result);
        }))
      return mlir::success();
    auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation);
    const bool boundaryView = isBoundaryView(operation);
    if (!mlir::isa<mlir::arith::ConstantOp>(operation) && !boundaryView &&
        (!allocation || !allocation.getDynamicSizes().empty() ||
         !allocation.getSymbolOperands().empty()))
      return failResult(failureReason,
                        "region cut suffix depends on an uncut producer");
    if (boundaryView) {
      for (mlir::Value operand : operation->getOperands()) {
        if (suffixMapping.contains(operand) ||
            mlir::isa<mlir::BlockArgument>(operand))
          continue;
        mlir::Operation *definition = operand.getDefiningOp();
        if (!definition || !prefix.contains(definition) ||
            mlir::failed(cloneResource(definition)))
          return failResult(
              failureReason,
              "region input view has an uncloneable suffix dependency");
      }
    }
    suffixBuilder.clone(*operation, suffixMapping);
    return mlir::success();
  };

  llvm::SmallVector<mlir::Operation *, 64> bodyOperations;
  for (mlir::Operation &operation : body.without_terminator())
    bodyOperations.push_back(&operation);
  auto oldYield = mlir::cast<TileYieldOp>(body.getTerminator());
  llvm::SmallVector<mlir::Value, 4> oldYieldValues(oldYield.getValues());
  mlir::Location oldYieldLoc = oldYield.getLoc();

  for (mlir::Operation *operation : bodyOperations) {
    if (prefix.contains(operation) ||
        operation == spillAllocation.getOperation() ||
        sharedDDRAllocationOps.contains(operation))
      continue;
    // Region-bearing suffix operations may lexically capture a boundary view
    // without listing it as a parent-op operand (for example a subview used
    // only inside scf.for). Discover and clone only those boundary resources
    // before moving the parent; otherwise its nested region retains a use of
    // the soon-to-be erased original TileRegion value.
    mlir::WalkResult resourceResult =
        operation->walk([&](mlir::Operation *nested) -> mlir::WalkResult {
          for (mlir::Value operand : nested->getOperands()) {
            mlir::Operation *definition = operand.getDefiningOp();
            if (definition && prefix.contains(definition) &&
                !suffixMapping.contains(operand) &&
                mlir::failed(cloneResource(definition)))
              return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (resourceResult.wasInterrupted())
      return mlir::failure();
  }

  auto remapOperationTree = [](mlir::Operation *operation,
                               mlir::IRMapping &mapping) {
    operation->walk([&](mlir::Operation *nested) {
      for (mlir::OpOperand &operand : nested->getOpOperands())
        if (mlir::Value replacement = mapping.lookupOrNull(operand.get()))
          operand.set(replacement);
    });
  };
  for (mlir::Operation *operation : bodyOperations) {
    if (operation == spillAllocation.getOperation() ||
        sharedDDRAllocationOps.contains(operation))
      continue;
    if (prefix.contains(operation)) {
      remapOperationTree(operation, prefixMapping);
      operation->moveBefore(&prefixBody, prefixBody.end());
      continue;
    }
    remapOperationTree(operation, suffixMapping);
    operation->moveBefore(&suffixBody, suffixBody.end());
  }

  mlir::OpBuilder prefixBuilder = mlir::OpBuilder::atBlockEnd(&prefixBody);
  llvm::SmallVector<mlir::Value, 8> prefixYields{
      prefixBody.getArgument(body.getNumArguments())};
  prefixYields.append(carriedDDRValues.begin(), carriedDDRValues.end());
  prefixBuilder.create<TileYieldOp>(region.getLoc(), prefixYields);

  llvm::SmallVector<mlir::Value, 4> suffixYields;
  for (mlir::Value value : oldYieldValues) {
    mlir::Value mapped = suffixMapping.lookupOrDefault(value);
    mlir::Operation *definition = mapped.getDefiningOp();
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(mapped);
    if ((!definition || definition->getBlock() != &suffixBody) &&
        (!argument || argument.getOwner() != &suffixBody))
      return failResult(
          failureReason,
          "region cut could not map an observable result into its suffix");
    suffixYields.push_back(mapped);
  }
  suffixBuilder.create<TileYieldOp>(oldYieldLoc, suffixYields);

  mlir::Operation *externalUseOwner = nullptr;
  mlir::Operation *externalUseDefinition = nullptr;
  region.walk([&](mlir::Operation *operation) {
    if (operation == region.getOperation() || externalUseOwner)
      return;
    for (mlir::Value result : operation->getResults())
      for (mlir::Operation *user : result.getUsers())
        if (user->getParentOfType<TileRegionOp>() != region) {
          externalUseDefinition = operation;
          externalUseOwner = user;
          return;
        }
  });
  if (externalUseOwner) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    stream << "region cut cannot erase an internal value used outside its "
              "TileRegion; definition="
           << externalUseDefinition->getName()
           << ", user=" << externalUseOwner->getName() << "; definition_ir=";
    externalUseDefinition->print(stream, mlir::OpPrintingFlags().skipRegions());
    stream << "; user_ir=";
    externalUseOwner->print(stream, mlir::OpPrintingFlags().skipRegions());
    return failResult(failureReason, stream.str());
  }

  if (materializationRelations) {
    auto retargetValue = [&](mlir::Value value) -> mlir::Value {
      if (value == spillAllocation.getResult())
        return externalSpill.getResult();
      for (auto [index, allocation] : llvm::enumerate(sharedDDRAllocations))
        if (value == allocation.getResult())
          return externalSharedDDR[index];
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
          argument && argument.getOwner() == &body)
        return suffixMapping.lookupOrDefault(value);
      if (auto result = mlir::dyn_cast<mlir::OpResult>(value);
          result && result.getOwner() == region.getOperation() &&
          result.getResultNumber() < suffixRegion.getNumResults())
        return suffixRegion.getResult(result.getResultNumber());
      return value;
    };
    auto retarget = [&](auto &entries) {
      for (auto &relation : entries)
        relation.buffer = retargetValue(relation.buffer);
    };
    retarget(materializationRelations->operationResultBuffers);
    retarget(materializationRelations->operandBuffers);
    retarget(materializationRelations->outputBuffers);
  }

  region.replaceAllUsesWith(suffixRegion.getResults());
  region.erase();
  return mlir::success();
}

void eraseUnreadDirectPrivateLoads(mlir::Operation *operation) {
  llvm::SmallVector<StorageLoadOp, 8> unreadLoads;
  operation->walk([&](StorageLoadOp load) {
    auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && allocation.getResult().hasOneUse())
      unreadLoads.push_back(load);
  });
  for (StorageLoadOp load : unreadLoads) {
    auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
    load.erase();
    if (allocation.getResult().use_empty())
      allocation.erase();
  }
}

} // namespace

mlir::LogicalResult wafer::deriveSpatialEdgeConsumerResultDomain(
    const SpatialEdgeStrategy &strategy,
    llvm::SmallVectorImpl<int64_t> &consumerOffsets,
    llvm::SmallVectorImpl<int64_t> &consumerSizes, std::string *failureReason) {
  consumerOffsets.clear();
  consumerSizes.clear();
  return deriveConsumerDomainFromProducerDemand(
      strategy.producer, strategy.producerResult, strategy.consumer,
      strategy.consumerOperand, strategy.producerOffsets,
      strategy.producerSizes, consumerOffsets, consumerSizes, failureReason);
}

bool wafer::isSpatialEdgeStrategyIncidentOnTile(
    const SpatialEdgeStrategy &strategy, TileId tile) {
  if (strategy.destinationTile == tile)
    return true;
  if (strategy.action != SpatialEdgeAction::PeerFragments)
    return false;
  return llvm::any_of(strategy.fragments, [&](const SpatialEdgeFragment &item) {
    return item.kind == SpatialEdgeFragmentKind::Peer &&
           item.sourceTile == tile;
  });
}

mlir::LogicalResult wafer::lowerSpatialEdgeStrategiesToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || currentLogicalPartition < 0)
    return failResult(failureReason,
                      "edge-action lowering requires a source module and "
                      "logical card partition");

  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone(cloneMapping));
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(*candidate);
  mlir::func::FuncOp supportTemplateFunction =
      findSingleStandaloneTensorProgram(sourceModule);
  if (!function || mlir::failed(verifyTensorProgramScope(
                       function, functionalArgumentCount, failureReason)))
    return mlir::failure();
  if (!supportTemplateFunction || supportTemplateFunction.isExternal() ||
      !supportTemplateFunction.getBody().hasOneBlock())
    return failResult(
        failureReason,
        "edge-action lowering requires one pristine support template body");
  TensorProgramScope scope(function, functionalArgumentCount);
  mlir::Block &supportTemplateBody = supportTemplateFunction.getBody().front();

  llvm::DenseSet<mlir::Operation *> seenTemporalOperations;
  llvm::SmallVector<StructuredOpTemporalTile, 16> mappedTemporalTiles;
  for (const StructuredOpTemporalTile &tile : operationTemporalTiles) {
    if (!tile.operation ||
        !seenTemporalOperations.insert(tile.operation).second)
      return failResult(
          failureReason,
          "structured temporal mapping contains a null or duplicate operation");
    mlir::Operation *mapped = cloneMapping.lookupOrNull(tile.operation);
    if (!mapped)
      return failResult(
          failureReason,
          "structured temporal mapping operation is outside source module");
    mappedTemporalTiles.push_back(
        StructuredOpTemporalTile{mapped, tile.iteratorTileSizes});
  }

  llvm::DenseSet<mlir::Operation *> seenNodeOperations;
  llvm::DenseSet<uint32_t> seenNodeIds;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> mappedOperationNodes;
  mappedOperationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !seenNodeOperations.insert(node.operation).second ||
        !seenNodeIds.insert(node.structuredNodeId).second)
      return failResult(
          failureReason,
          "structured operation-node mapping contains a null or duplicate "
          "entry");
    mlir::Operation *mapped = cloneMapping.lookupOrNull(node.operation);
    if (!mapped)
      return failResult(failureReason,
                        "structured operation-node mapping is outside source "
                        "module");
    mappedOperationNodes.push_back({mapped, node.structuredNodeId});
  }

  const bool independentDDRStages =
      materializationMode ==
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  llvm::SmallVector<SpatialEdgeStrategy, 16> normalizedEdgeStrategies(
      edgeStrategies.begin(), edgeStrategies.end());
  if (independentDDRStages &&
      mlir::failed(splitIndependentPeerFragmentsAtTemporalWaves(
          normalizedEdgeStrategies, operationTemporalTiles, failureReason)))
    return mlir::failure();

  llvm::SmallVector<MappedStrategy, 16> mappedStrategies;
  mappedStrategies.reserve(normalizedEdgeStrategies.size());
  for (const SpatialEdgeStrategy &strategy : normalizedEdgeStrategies) {
    if (!isSpatialEdgeStrategyIncidentOnTile(strategy, currentTile))
      continue;
    MappedStrategy mapped;
    mapped.strategy = strategy;
    mapped.supportTemplateProducer = strategy.producer;
    mapped.supportTemplateConsumer = strategy.consumer;
    if (strategy.consumer &&
        strategy.consumerOperand < strategy.consumer->getNumOperands())
      mapped.supportTemplateOperand =
          strategy.consumer->getOperand(strategy.consumerOperand);
    mapped.producer = cloneMapping.lookupOrNull(strategy.producer);
    mapped.consumer = cloneMapping.lookupOrNull(strategy.consumer);
    auto consumerPosition =
        llvm::find_if(supportTemplateBody, [&](mlir::Operation &operation) {
          return &operation == mapped.supportTemplateConsumer;
        });
    if (consumerPosition == supportTemplateBody.end())
      return failResult(
          failureReason,
          "edge strategy consumer is outside the pristine program body");
    mapped.consumerScheduleOrdinal = static_cast<uint64_t>(
        std::distance(supportTemplateBody.begin(), consumerPosition));
    mapped.strategy.producer = mapped.producer;
    mapped.strategy.consumer = mapped.consumer;
    if (mlir::failed(validateEdge(mapped.producer, strategy.producerResult,
                                  mapped.consumer, strategy.consumerOperand,
                                  scope.getBody(), failureReason)))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> supportChain =
        deriveUnaryPureSupportChain(mapped.producer, strategy.producerResult,
                                    mapped.consumer, strategy.consumerOperand,
                                    failureReason);
    if (mlir::failed(supportChain))
      return mlir::failure();
    mapped.hasSupportPath = !supportChain->empty();
    if (!supportChain->empty() &&
        strategy.action != SpatialEdgeAction::RegionCut &&
        strategy.action != SpatialEdgeAction::PeerFragments)
      return failResult(
          failureReason,
          "pure support-chain dependencies require RegionCut or exact peer "
          "fragments");
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.producer->getResult(strategy.producerResult).getType());
    auto consumerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.consumer->getResult(0).getType());
    if (strategy.consumerOffsets.empty() || strategy.consumerSizes.empty()) {
      if (!strategy.consumerOffsets.empty() ||
          !strategy.consumerSizes.empty() ||
          mlir::failed(deriveConsumerDomainFromProducerDemand(
              mapped.producer, strategy.producerResult, mapped.consumer,
              strategy.consumerOperand, strategy.producerOffsets,
              strategy.producerSizes, mapped.strategy.consumerOffsets,
              mapped.strategy.consumerSizes, failureReason)))
        return mlir::failure();
    }
    const SpatialEdgeStrategy &validated = mapped.strategy;
    if (mlir::failed(validateStaticDomain(
            producerType, validated.producerOffsets, validated.producerSizes,
            failureReason, "edge producer demand")) ||
        mlir::failed(validateStaticDomain(
            consumerType, validated.consumerOffsets, validated.consumerSizes,
            failureReason, "edge consumer demand")) ||
        (supportChain->empty() &&
         mlir::failed(validateDemandIndexRelation(
             mapped.consumer, validated.consumerOperand,
             validated.consumerOffsets, validated.consumerSizes,
             validated.producerOffsets, validated.producerSizes,
             failureReason))))
      return mlir::failure();
    if (llvm::any_of(mappedStrategies, [&](const MappedStrategy &other) {
          return sameEdge(mapped, other) && mapped.strategy.destinationTile ==
                                                other.strategy.destinationTile;
        }))
      return failResult(failureReason,
                        "edge mapping duplicates one destination strategy");

    if (strategy.bufferCount == 0 || strategy.bufferCount > 3)
      return failResult(failureReason,
                        "edge strategy buffer count is outside [1, 3]");
    if (strategy.action == SpatialEdgeAction::LocalPhysicalConversion &&
        (!strategy.hasLayoutAssignment ||
         strategy.producerLayout == strategy.consumerLayout))
      return failResult(
          failureReason,
          "local physical conversion requires distinct typed layouts");
    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        !strategy.fragments.empty())
      return failResult(failureReason,
                        "non-peer edge strategy cannot carry fragments");
    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        strategy.sourceTile != strategy.destinationTile)
      return failResult(
          failureReason,
          "local edge strategy requires one Tile placement");
    mappedStrategies.push_back(std::move(mapped));
  }
  if (mappedStrategies.empty())
    return failResult(failureReason,
                      "edge-action lowering has no action on this Tile");
  if (independentDDRStages &&
      !llvm::all_of(mappedStrategies, [](const MappedStrategy &mapped) {
        return mapped.strategy.bufferCount == 1 &&
               (mapped.strategy.action == SpatialEdgeAction::RegionCut ||
                mapped.strategy.action == SpatialEdgeAction::PeerFragments);
      }))
    return failResult(
        failureReason,
        "independent DDR stages require single-buffer RegionCut or exact "
        "cross-Tile fragment actions");

  llvm::SmallVector<CandidatePeerEndpoint, 8> endpoints;
  llvm::SmallVector<MaterializedSource, 8> materialized;
  llvm::SmallVector<MaterializedRegionCutSpill, 8> materializedRegionCutSpills;
  llvm::SmallVector<CandidateSelectedDDRStage, 8> selectedDDRStages;
  llvm::DenseSet<mlir::Operation *> preserved;
  llvm::SmallVector<const SpatialEdgeStrategy *, 2> regionCuts;

  for (MappedStrategy &mapped : mappedStrategies) {
    SpatialEdgeStrategy &strategy = mapped.strategy;
    if (strategy.destinationTile != currentTile)
      continue;
    switch (strategy.action) {
    case SpatialEdgeAction::CoupledFusion: {
      // A full-domain consumer does not manufacture an extract_slice, so the
      // ordinary producer-fusion walk has no trigger.  When the common state
      // selected a finer iterator traversal for this producer, materialize
      // that exact traversal explicitly and keep its assembled SSA value in
      // the same region.  Otherwise the selected temporal dimension would be
      // silently replaced by the untiled source operation.
      if (!hasSplitSelectedOutputTraversal(scope, outputShards,
                                           mapped.consumer) &&
          isOneFullTemporalWave(mapped.consumer, mappedTemporalTiles) &&
          !isOneFullTemporalWaveForResultDemand(
              mapped.producer, strategy.producerResult, strategy.producerSizes,
              mappedTemporalTiles)) {
        mlir::FailureOr<mlir::Value> value = getOrMaterializeSource(
            scope, mapped.producer, strategy.producerResult,
            strategy.producerOffsets, strategy.producerSizes, materialized,
            preserved, mappedTemporalTiles, failureReason,
            &mappedOperationNodes);
        if (mlir::failed(value))
          return mlir::failure();
        auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        if (!fullType)
          return failResult(failureReason,
                            "coupled producer result is not a ranked tensor");
        if (isFullStaticResultDomain(mapped.producer, strategy.producerResult,
                                     strategy.producerOffsets,
                                     strategy.producerSizes) &&
            value->getType() == fullType) {
          // A full selected window is already in the consumer's global index
          // space.  Re-embedding it through tensor.empty/insert_slice creates
          // a second full-size physical allocation without changing any
          // index relation.
          mapped.consumer->setOperand(strategy.consumerOperand, *value);
        } else {
          // `getOrMaterializeSource` returns the compact selected producer
          // window.  The consumer still owns global indexing maps, so expose
          // that window at its global result offsets before consumer tiling.
          mlir::OpBuilder builder(mapped.consumer);
          auto empty = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), fullType.getShape(),
              fullType.getElementType(), mlir::ValueRange{},
              fullType.getEncoding());
          mlir::Value embedded = insertExactSlice(
              builder, mapped.consumer->getLoc(), *value, empty.getResult(),
              strategy.producerOffsets, strategy.producerSizes);
          mapped.consumer->setOperand(strategy.consumerOperand, embedded);
        }
      }
      break;
    }
    case SpatialEdgeAction::LocalShardResidency:
      if (mlir::failed(materializeLocalShardResidency(mapped, failureReason)))
        return mlir::failure();
      break;
    case SpatialEdgeAction::SpillReload: {
      if (independentDDRStages)
        break;
      bool ignoredCreatedRegionCut = false;
      if (mlir::failed(
              materializeSpill(scope, mapped, materialized, preserved,
                               materializedRegionCutSpills, selectedDDRStages,
                               ignoredCreatedRegionCut, mappedTemporalTiles,
                               failureReason, &mappedOperationNodes)))
        return mlir::failure();
    } break;
    case SpatialEdgeAction::RegionCut: {
      if (independentDDRStages)
        break;
      bool createdRegionCut = false;
      if (mlir::failed(materializeSpill(
              scope, mapped, materialized, preserved,
              materializedRegionCutSpills, selectedDDRStages, createdRegionCut,
              mappedTemporalTiles, failureReason, &mappedOperationNodes)))
        return mlir::failure();
      if (createdRegionCut)
        regionCuts.push_back(&mapped.strategy);
      break;
    }
    case SpatialEdgeAction::Recompute:
      if (mlir::failed(materializeRecompute(
              scope, mapped, materialized, preserved, mappedTemporalTiles,
              mappedOperationNodes, failureReason)))
        return mlir::failure();
      break;
    case SpatialEdgeAction::PeerFragments:
      break;
    case SpatialEdgeAction::LocalPhysicalConversion:
      if (mlir::failed(materializeLocalShardResidency(mapped, failureReason)))
        return mlir::failure();
      break;
    }
  }
  llvm::sort(regionCuts, [](const SpatialEdgeStrategy *lhs,
                            const SpatialEdgeStrategy *rhs) {
    if (lhs->consumer == rhs->consumer)
      return lhs->consumerOperand < rhs->consumerOperand;
    return lhs->consumer->isBeforeInBlock(rhs->consumer);
  });

  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 8>
      rebuiltSupportInputs;
  for (MappedStrategy &anchor : mappedStrategies) {
    if (!anchor.hasSupportPath ||
        anchor.strategy.action != SpatialEdgeAction::RegionCut ||
        llvm::is_contained(
            rebuiltSupportInputs,
            std::pair<mlir::Operation *, unsigned>{
                anchor.consumer, anchor.strategy.consumerOperand}))
      continue;

    mlir::Value original = anchor.supportTemplateOperand;
    mlir::IRMapping supportMapping;
    for (MappedStrategy &mapped : mappedStrategies) {
      if (mapped.consumer != anchor.consumer ||
          mapped.strategy.consumerOperand != anchor.strategy.consumerOperand ||
          mapped.strategy.action != SpatialEdgeAction::RegionCut)
        continue;
      auto spill = llvm::find_if(
          materializedRegionCutSpills,
          [&](const MaterializedRegionCutSpill &candidateSpill) {
            return candidateSpill.producer == mapped.producer &&
                   candidateSpill.result == mapped.strategy.producerResult &&
                   candidateSpill.offsets == mapped.strategy.producerOffsets &&
                   candidateSpill.sizes == mapped.strategy.producerSizes;
          });
      if (spill == materializedRegionCutSpills.end())
        return failResult(
            failureReason,
            "support dependency is missing its selected producer spill");
      if (!mapped.supportTemplateProducer)
        return failResult(
            failureReason,
            "support dependency is missing its pristine producer template");
      mlir::Value producer = mapped.supportTemplateProducer->getResult(
          mapped.strategy.producerResult);
      if (supportMapping.contains(producer) &&
          supportMapping.lookup(producer) != spill->value)
        return failResult(
            failureReason,
            "support dependency has conflicting producer spill domains");
      if (!supportMapping.contains(producer))
        supportMapping.map(producer, spill->value);
    }

    mlir::OpBuilder supportBuilder(anchor.consumer);
    std::function<mlir::FailureOr<mlir::Value>(mlir::Value)> rebuild =
        [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
      if (mlir::Value mapped = supportMapping.lookupOrNull(value))
        return mapped;
      if (mlir::isa<mlir::BlockArgument>(value)) {
        mlir::Value mapped = cloneMapping.lookupOrNull(value);
        if (!mapped)
          return fail<mlir::Value>(
              failureReason,
              "support dependency cannot map its pristine block argument");
        supportMapping.map(value, mapped);
        return mapped;
      }
      mlir::Operation *operation = value.getDefiningOp();
      if (!operation || operation->getBlock() != &supportTemplateBody ||
          !mlir::isMemoryEffectFree(operation))
        return fail<mlir::Value>(
            failureReason,
            "support dependency contains an uncloneable tensor value");
      if (mlir::isa<mlir::TilingInterface>(operation) &&
          mlir::isa<mlir::DestinationStyleOpInterface>(operation))
        return fail<mlir::Value>(
            failureReason,
            "support dependency has an unspilled structured producer");
      for (mlir::Value operand : operation->getOperands()) {
        if (!mlir::isa<mlir::RankedTensorType>(operand.getType()) ||
            supportMapping.contains(operand))
          continue;
        mlir::FailureOr<mlir::Value> replacement = rebuild(operand);
        if (mlir::failed(replacement))
          return mlir::failure();
        if (!supportMapping.contains(operand))
          supportMapping.map(operand, *replacement);
      }
      supportBuilder.clone(*operation, supportMapping);
      mlir::Value mapped = supportMapping.lookupOrNull(value);
      if (!mapped)
        return fail<mlir::Value>(
            failureReason,
            "support dependency clone did not map its selected result");
      return mapped;
    };
    mlir::FailureOr<mlir::Value> replacement = rebuild(original);
    if (mlir::failed(replacement))
      return mlir::failure();
    anchor.consumer->setOperand(anchor.strategy.consumerOperand, *replacement);
    rebuiltSupportInputs.push_back(
        {anchor.consumer, anchor.strategy.consumerOperand});
  }

  llvm::SmallVector<std::pair<MappedStrategy *, mlir::Value>, 8>
      peerSupportAssemblies;
  auto rebuildPeerSupportInput = [&](MappedStrategy &anchor) {
    const std::pair<mlir::Operation *, unsigned> inputKey{
        anchor.consumer, anchor.strategy.consumerOperand};
    if (llvm::is_contained(rebuiltSupportInputs, inputKey))
      return mlir::success();
    mlir::Value original = anchor.supportTemplateOperand;
    mlir::IRMapping supportMapping;
    for (auto [mapped, assembly] : peerSupportAssemblies) {
      if (mapped->consumer != anchor.consumer ||
          mapped->strategy.consumerOperand != anchor.strategy.consumerOperand)
        continue;
      if (!mapped->supportTemplateProducer)
        return failResult(
            failureReason,
            "peer support dependency is missing its pristine producer "
            "template");
      mlir::Value producer = mapped->supportTemplateProducer->getResult(
          mapped->strategy.producerResult);
      if (supportMapping.contains(producer) &&
          supportMapping.lookup(producer) != assembly)
        return failResult(
            failureReason,
            "peer support dependency has conflicting producer assemblies");
      if (!supportMapping.contains(producer))
        supportMapping.map(producer, assembly);
    }

    mlir::OpBuilder supportBuilder(anchor.consumer);
    std::function<mlir::FailureOr<mlir::Value>(mlir::Value)> rebuild =
        [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
      if (mlir::Value mapped = supportMapping.lookupOrNull(value))
        return mapped;
      if (mlir::isa<mlir::BlockArgument>(value)) {
        mlir::Value mapped = cloneMapping.lookupOrNull(value);
        if (!mapped)
          return fail<mlir::Value>(
              failureReason,
              "peer support dependency cannot map its pristine block "
              "argument");
        supportMapping.map(value, mapped);
        return mapped;
      }
      mlir::Operation *operation = value.getDefiningOp();
      if (!operation || operation->getBlock() != &supportTemplateBody ||
          !mlir::isMemoryEffectFree(operation))
        return fail<mlir::Value>(
            failureReason,
            "peer support dependency contains an uncloneable tensor value");
      if (mlir::isa<mlir::TilingInterface>(operation) &&
          mlir::isa<mlir::DestinationStyleOpInterface>(operation))
        return fail<mlir::Value>(
            failureReason,
            (llvm::Twine("peer support dependency has an unassembled "
                         "structured producer ") +
             operation->getName().getStringRef())
                .str());
      for (mlir::Value operand : operation->getOperands()) {
        if (!mlir::isa<mlir::RankedTensorType>(operand.getType()) ||
            supportMapping.contains(operand))
          continue;
        mlir::FailureOr<mlir::Value> replacement = rebuild(operand);
        if (mlir::failed(replacement))
          return mlir::failure();
        if (!supportMapping.contains(operand))
          supportMapping.map(operand, *replacement);
      }
      supportBuilder.clone(*operation, supportMapping);
      mlir::Value mapped = supportMapping.lookupOrNull(value);
      if (!mapped)
        return fail<mlir::Value>(
            failureReason,
            "peer support dependency clone did not map its result");
      return mapped;
    };
    mlir::FailureOr<mlir::Value> replacement = rebuild(original);
    if (mlir::failed(replacement))
      return mlir::failure();
    anchor.consumer->setOperand(anchor.strategy.consumerOperand, *replacement);
    rebuiltSupportInputs.push_back(inputKey);
    return mlir::success();
  };

  llvm::SmallVector<MappedStrategy *, 16> peerOrder;
  for (MappedStrategy &mapped : mappedStrategies)
    if (mapped.strategy.action == SpatialEdgeAction::PeerFragments)
      peerOrder.push_back(&mapped);
  llvm::sort(
      peerOrder, [](const MappedStrategy *lhs, const MappedStrategy *rhs) {
        if (lhs->consumer != rhs->consumer)
          return lhs->consumer->isBeforeInBlock(rhs->consumer);
        if (lhs->strategy.consumerOperand != rhs->strategy.consumerOperand)
          return lhs->strategy.consumerOperand < rhs->strategy.consumerOperand;
        if (lhs->producer != rhs->producer)
          return lhs->producer->isBeforeInBlock(rhs->producer);
        return lhs->strategy.destinationTile.getValue() <
               rhs->strategy.destinationTile.getValue();
      });
  // PeerFragments is the only action with an all-and-only fragment assembly.
  for (size_t groupBegin = 0; groupBegin < peerOrder.size();) {
    size_t groupEnd = groupBegin + 1;
    while (groupEnd < peerOrder.size() &&
           peerOrder[groupEnd]->consumer == peerOrder[groupBegin]->consumer &&
           peerOrder[groupEnd]->strategy.consumerOperand ==
               peerOrder[groupBegin]->strategy.consumerOperand)
      ++groupEnd;
    for (size_t peerIndex = groupBegin; peerIndex < groupEnd; ++peerIndex) {
      MappedStrategy &mapped = *peerOrder[peerIndex];
      SpatialEdgeStrategy &strategy = mapped.strategy;
      if (strategy.fragments.empty())
        return failResult(failureReason,
                          "peer edge strategy has no exact fragments");

      uint64_t coveredElements = 0;
      for (size_t index = 0; index < strategy.fragments.size(); ++index) {
        const SpatialEdgeFragment &fragment = strategy.fragments[index];
        if (!isContained(fragment.offsets, fragment.sizes,
                         strategy.producerOffsets, strategy.producerSizes))
          return failResult(
              failureReason,
              "dependent fragment extends outside its consumer demand");
        std::optional<uint64_t> elements = getDomainElements(fragment.sizes);
        if (!elements ||
            *elements > std::numeric_limits<uint64_t>::max() - coveredElements)
          return failResult(failureReason,
                            "dependent fragment coverage is not representable");
        for (size_t previous = 0; previous < index; ++previous)
          if (overlaps(fragment.offsets, fragment.sizes,
                       strategy.fragments[previous].offsets,
                       strategy.fragments[previous].sizes))
            return failResult(
                failureReason,
                "dependent fragments overlap within one consumer demand");
        coveredElements += *elements;

        if (fragment.kind == SpatialEdgeFragmentKind::Resident) {
          if (fragment.sourceTile != strategy.destinationTile ||
              fragment.bytes != 0 || fragment.communicationId != 0 ||
              fragment.payloadSlice != 0)
            return failResult(
                failureReason,
                "resident fragment must be local and carry no peer message");
          continue;
        }
        auto producerType = mlir::cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        std::optional<uint64_t> elementBytes =
            getElementBytes(producerType.getElementType());
        if (!elementBytes ||
            *elements > std::numeric_limits<uint64_t>::max() / *elementBytes ||
            fragment.bytes != *elements * *elementBytes ||
            fragment.sourceTile == strategy.destinationTile ||
            fragment.communicationId < 0 || fragment.payloadSlice < 0)
          return failResult(
              failureReason,
              "peer fragment has invalid endpoint, bytes, or message");
      }
      std::optional<uint64_t> demandedElements =
          getDomainElements(strategy.producerSizes);
      if (!demandedElements || coveredElements != *demandedElements)
        return failResult(
            failureReason,
            "dependent fragments do not exactly cover the consumer demand");

      if (strategy.destinationTile == currentTile) {
        auto producerType = mlir::cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        mlir::OpBuilder builder(mapped.consumer);
        mlir::Location assemblyLoc = mapped.consumer->getLoc();
        mlir::Value assembled;
        mlir::memref::AllocOp independentAssemblyAllocation;
        if (independentDDRStages) {
          // The logical full-domain assembly may be much larger than SPM.  The
          // baseline stages each exact peer fragment into compiler-owned DDR;
          // downstream temporal waves then load only their demanded windows.
          // Keeping the full carrier in SPM would make temporal refinement
          // ineffective because allocation fails before the consumer wave.
          auto ddrType = mlir::MemRefType::get(
              producerType.getShape(), producerType.getElementType(),
              mlir::MemRefLayoutAttrInterface{},
              MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                              MemLayout::Tensor));
          auto allocation =
              builder.create<mlir::memref::AllocOp>(assemblyLoc, ddrType);
          independentAssemblyAllocation = allocation;
          selectedDDRStages.push_back(CandidateSelectedDDRStage{
              allocation.getResult(), &mapped.strategy});
          auto destination = builder.create<mlir::bufferization::ToTensorOp>(
              assemblyLoc, allocation.getResult(),
              /*restrict=*/true, /*writable=*/true);
          preserved.insert(allocation.getOperation());
          preserved.insert(destination.getOperation());
          assembled = destination.getResult();
          regionCuts.push_back(&mapped.strategy);
        } else {
          auto fullEmpty = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), producerType.getShape(),
              producerType.getElementType(), mlir::ValueRange{},
              producerType.getEncoding());
          assembled = fullEmpty.getResult();
        }
        llvm::SmallVector<const SpatialEdgeFragment *, 4> ordered;
        for (const SpatialEdgeFragment &fragment : strategy.fragments)
          ordered.push_back(&fragment);
        llvm::sort(ordered, [](const auto *lhs, const auto *rhs) {
          if (lhs->offsets != rhs->offsets)
            return lhs->offsets < rhs->offsets;
          if (lhs->sizes != rhs->sizes)
            return lhs->sizes < rhs->sizes;
          return std::tuple(static_cast<uint8_t>(lhs->kind),
                            lhs->sourceTile.getValue()) <
                 std::tuple(static_cast<uint8_t>(rhs->kind),
                            rhs->sourceTile.getValue());
        });
        for (const SpatialEdgeFragment *fragment : ordered) {
          mlir::Value value;
          if (fragment->kind == SpatialEdgeFragmentKind::Resident) {
            if (independentDDRStages) {
              // Keep the resident contribution as a current-SSA window.  The
              // producer is independently materialized later in topological
              // order, and its external fanout rewrite then retargets this
              // slice to the sealed local result without eager recomputation.
              value = createExactSlice(
                  builder, mapped.producer->getLoc(),
                  mapped.producer->getResult(strategy.producerResult),
                  fragment->offsets, fragment->sizes);
            } else {
              mlir::FailureOr<mlir::Value> local = getOrMaterializeSource(
                  scope, mapped.producer, strategy.producerResult,
                  fragment->offsets, fragment->sizes, materialized, preserved,
                  mappedTemporalTiles, failureReason, &mappedOperationNodes);
              if (mlir::failed(local))
                return mlir::failure();
              value = *local;
            }
          } else {
            mlir::Location receiveLoc = mapped.consumer->getLoc();
            auto receiveType = mlir::MemRefType::get(
                fragment->sizes, producerType.getElementType(),
                mlir::MemRefLayoutAttrInterface{},
                MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::SPM,
                                MemLayout::Tensor));
            auto allocation =
                builder.create<mlir::memref::AllocOp>(receiveLoc, receiveType);
            auto received = builder.create<mlir::bufferization::ToTensorOp>(
                receiveLoc, allocation.getResult(), /*restrict=*/true,
                /*writable=*/true);
            preserved.insert(allocation.getOperation());
            preserved.insert(received.getOperation());
            value = received.getResult();
            endpoints.push_back(CandidatePeerEndpoint{
                value, allocation.getResult(),
                CandidatePeerEndpointKind::Receive, fragment->sourceTile,
                fragment->bytes, fragment->communicationId,
                fragment->payloadSlice, mapped.consumerScheduleOrdinal,
                strategy.consumerOperand, fragment});
          }
          mlir::Value inserted =
              insertExactSlice(builder, assemblyLoc, value, assembled,
                               fragment->offsets, fragment->sizes);
          if (independentDDRStages) {
            // Every exact fragment writes a disjoint slice of the same
            // compiler-owned writable DDR allocation. Do not thread those
            // stores through one functional tensor.insert_slice chain: that
            // chain makes all receive buffers appear simultaneously live
            // until the final value, defeating the baseline's fragment-wise
            // staging contract. Preserve each store root independently; the
            // sealed read-only view below is the sole downstream carrier.
            preserved.insert(inserted.getDefiningOp());
          } else {
            assembled = inserted;
          }
        }
        if (independentDDRStages) {
          // Reopen the complete exact allocation as a read-only tensor
          // boundary so consumer waves load only their demanded windows and
          // cannot clone fragment stores into those waves.
          auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
              assemblyLoc, independentAssemblyAllocation.getResult(),
              /*restrict=*/false, /*writable=*/false);
          preserved.insert(sealed.getOperation());
          assembled = sealed.getResult();
        } else {
          // PeerFragments is an explicit physical edge action.  Seal its exact
          // SPM assembly so downstream tiling cannot recompute the producer.
          auto materializedDest = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), producerType.getShape(),
              producerType.getElementType(), mlir::ValueRange{},
              producerType.getEncoding());
          assembled =
              builder
                  .create<mlir::bufferization::MaterializeInDestinationOp>(
                      mapped.consumer->getLoc(), producerType, assembled,
                      materializedDest.getResult(), /*restrict=*/false,
                      /*writable=*/false)
                  .getResult();
          preserved.insert(assembled.getDefiningOp());
        }
        if (mapped.hasSupportPath)
          peerSupportAssemblies.push_back({&mapped, assembled});
        else
          mapped.consumer->setOperand(strategy.consumerOperand, assembled);
      }
    }
    llvm::ArrayRef<MappedStrategy *> group(peerOrder.data() + groupBegin,
                                           groupEnd - groupBegin);
    auto supportAnchor = llvm::find_if(group, [&](MappedStrategy *mapped) {
      return mapped->hasSupportPath &&
             mapped->strategy.destinationTile == currentTile;
    });
    if (supportAnchor != group.end() &&
        mlir::failed(rebuildPeerSupportInput(**supportAnchor)))
      return mlir::failure();
    for (const CandidatePeerEndpoint &endpoint : endpoints) {
      if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
          !endpoint.value.use_empty())
        continue;
      auto groupNode = llvm::find_if(
          operationNodes, [&](const StructuredOperationNodeMapping &entry) {
            return entry.operation ==
                   peerOrder[groupBegin]->supportTemplateConsumer;
          });
      return failResult(
          failureReason,
          (llvm::Twine("selected receive became unused while materializing "
                       "peer consumer group ") +
           (groupNode != operationNodes.end()
                ? llvm::Twine(groupNode->structuredNodeId)
                : llvm::Twine("unknown")) +
           " (communication_id=" + llvm::Twine(endpoint.communicationId) +
           ", payload_slice=" + llvm::Twine(endpoint.payloadSlice) + ")")
              .str());
    }
    groupBegin = groupEnd;
  }
  auto requireLiveReceive = [&](llvm::StringRef stage) {
    for (const CandidatePeerEndpoint &endpoint : endpoints) {
      if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
          !endpoint.value.use_empty())
        continue;
      std::string detail;
      llvm::raw_string_ostream diagnostic(detail);
      diagnostic << "selected receive became unused during " << stage
                 << " (communication_id=" << endpoint.communicationId
                 << ", payload_slice=" << endpoint.payloadSlice << ')';
      return failResult(failureReason, diagnostic.str());
    }
    return mlir::success();
  };
  llvm::DenseSet<mlir::Value> selectedBuffers;
  for (const CandidateSelectedDDRStage &stage : selectedDDRStages)
    selectedBuffers.insert(stage.buffer);
  auto reachesSelectedDDRStage = [&](mlir::Operation *source) {
    bool reachesStage = false;
    candidate->walk([&](mlir::tensor::InsertSliceOp insert) {
      if (reachesStage)
        return;
      mlir::Value destination = insert.getDest();
      while (auto prior =
                 destination.getDefiningOp<mlir::tensor::InsertSliceOp>())
        destination = prior.getDest();
      auto toTensor =
          destination.getDefiningOp<mlir::bufferization::ToTensorOp>();
      auto allocation =
          toTensor ? toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>()
                   : mlir::memref::AllocOp{};
      if (!allocation || !selectedBuffers.contains(allocation.getResult()))
        return;
      llvm::DenseSet<mlir::Value> visited;
      reachesStage = isInBackwardClosure(insert.getSource(), source, visited);
    });
    return reachesStage;
  };
  for (const CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
        !endpoint.selectedFragment)
      continue;
    auto owner = llvm::find_if(mappedStrategies, [&](MappedStrategy &mapped) {
      return llvm::any_of(mapped.strategy.fragments,
                          [&](const SpatialEdgeFragment &fragment) {
                            return &fragment == endpoint.selectedFragment;
                          });
    });
    if (owner == mappedStrategies.end())
      return failResult(failureReason,
                        "selected receive has no mapped edge owner");
    llvm::DenseSet<mlir::Value> visited;
    if (!isInBackwardClosure(
            owner->consumer->getOperand(owner->strategy.consumerOperand),
            endpoint.value.getDefiningOp(), visited) &&
        !reachesSelectedDDRStage(endpoint.value.getDefiningOp()))
      return failResult(
          failureReason,
          (llvm::Twine("selected receive is outside its consumer operand "
                       "closure before output traversal (communication_id=") +
           llvm::Twine(endpoint.communicationId) + ")")
              .str());
  }

  if (independentDDRStages) {
    struct SourceOnlyDomain {
      mlir::Operation *operation = nullptr;
      llvm::SmallVector<int64_t, 4> offsets;
      llvm::SmallVector<int64_t, 4> sizes;
    };
    llvm::SmallVector<SourceOnlyDomain, 4> sourceOnlyDomains;
    auto appendSourceDomain = [&](mlir::Operation *operation,
                                  llvm::ArrayRef<int64_t> offsets,
                                  llvm::ArrayRef<int64_t> sizes) {
      if (!operation || offsets.size() != sizes.size())
        return;
      auto existing =
          llvm::find_if(sourceOnlyDomains, [&](const SourceOnlyDomain &domain) {
            return domain.operation == operation &&
                   llvm::ArrayRef(domain.offsets) == offsets &&
                   llvm::ArrayRef(domain.sizes) == sizes;
          });
      if (existing == sourceOnlyDomains.end())
        sourceOnlyDomains.push_back(SourceOnlyDomain{
            operation, llvm::to_vector(offsets), llvm::to_vector(sizes)});
    };
    for (MappedStrategy &mapped : mappedStrategies) {
      const bool hasLocalIncoming =
          llvm::any_of(mappedStrategies, [&](const MappedStrategy &incoming) {
            return incoming.consumer == mapped.producer &&
                   incoming.strategy.destinationTile == currentTile;
          });
      if (hasLocalIncoming)
        continue;
      if (mapped.strategy.action == SpatialEdgeAction::PeerFragments) {
        for (const SpatialEdgeFragment &fragment : mapped.strategy.fragments)
          if (fragment.sourceTile == currentTile)
            appendSourceDomain(mapped.producer, fragment.offsets,
                               fragment.sizes);
      } else if (mapped.strategy.sourceTile == currentTile) {
        appendSourceDomain(mapped.producer, mapped.strategy.producerOffsets,
                           mapped.strategy.producerSizes);
      }
    }
    llvm::sort(sourceOnlyDomains,
               [](const SourceOnlyDomain &lhs, const SourceOnlyDomain &rhs) {
                 if (lhs.operation != rhs.operation)
                   return lhs.operation->isBeforeInBlock(rhs.operation);
                 if (lhs.offsets != rhs.offsets)
                   return lhs.offsets < rhs.offsets;
                 return lhs.sizes < rhs.sizes;
               });
    for (size_t groupBegin = 0; groupBegin < sourceOnlyDomains.size();) {
      size_t groupEnd = groupBegin + 1;
      while (groupEnd < sourceOnlyDomains.size() &&
             sourceOnlyDomains[groupEnd].operation ==
                 sourceOnlyDomains[groupBegin].operation)
        ++groupEnd;
      mlir::Operation *operation = sourceOnlyDomains[groupBegin].operation;
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          operation->getResult(0).getType());
      if (!resultType || !resultType.hasStaticShape())
        return failResult(
            failureReason,
            "source-only baseline op requires one static ranked result");
      auto ddrType = mlir::MemRefType::get(
          resultType.getShape(), resultType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(operation->getContext(), MemorySpace::DDR,
                          MemLayout::Tensor));
      mlir::OpBuilder builder(operation);
      auto allocation =
          builder.create<mlir::memref::AllocOp>(operation->getLoc(), ddrType);
      auto destination = builder.create<mlir::bufferization::ToTensorOp>(
          operation->getLoc(), allocation.getResult(),
          /*restrict=*/true, /*writable=*/true);
      preserved.insert(allocation.getOperation());
      preserved.insert(destination.getOperation());
      mlir::Value stored = destination.getResult();
      for (size_t index = groupBegin; index < groupEnd; ++index) {
        SourceOnlyDomain &domain = sourceOnlyDomains[index];
        mlir::FailureOr<mlir::Value> updated =
            materializeCandidateRootTileIntoDestination(
                scope, operation, domain.offsets, domain.sizes,
                mappedTemporalTiles, stored, failureReason);
        if (mlir::failed(updated))
          return mlir::failure();
        stored = *updated;
      }
      mlir::Operation *storedDefinition = stored.getDefiningOp();
      if (!storedDefinition)
        return failResult(
            failureReason,
            "source-only baseline store has no materialized definition");
      preserved.insert(storedDefinition);
      builder.setInsertionPointAfter(storedDefinition);
      auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
          operation->getLoc(), allocation.getResult(), /*restrict=*/false,
          /*writable=*/false);
      preserved.insert(sealed.getOperation());
      for (size_t index = groupBegin; index < groupEnd; ++index) {
        SourceOnlyDomain &domain = sourceOnlyDomains[index];
        mlir::Value compact =
            createExactSlice(builder, operation->getLoc(), sealed.getResult(),
                             domain.offsets, domain.sizes);
        preserved.insert(compact.getDefiningOp());
        auto cached = llvm::find_if(
            materialized, [&](const MaterializedSource &candidate) {
              return candidate.producer == operation && candidate.result == 0 &&
                     candidate.offsets == domain.offsets &&
                     candidate.sizes == domain.sizes;
            });
        if (cached == materialized.end())
          materialized.push_back(MaterializedSource{
              operation, /*result=*/0, domain.offsets, domain.sizes, compact});
        else
          cached->value = compact;
      }

      llvm::DenseSet<mlir::Value> visitedValues;
      llvm::DenseSet<mlir::Operation *> dependencyOperations;
      llvm::SmallVector<mlir::Value, 16> worklist{stored};
      while (!worklist.empty()) {
        mlir::Value value = worklist.pop_back_val();
        if (!value || !visitedValues.insert(value).second)
          continue;
        mlir::Operation *definition = value.getDefiningOp();
        if (!definition || !dependencyOperations.insert(definition).second)
          continue;
        worklist.append(definition->operand_begin(), definition->operand_end());
      }
      for (mlir::OpOperand &use :
           llvm::make_early_inc_range(operation->getResult(0).getUses()))
        if (!dependencyOperations.contains(use.getOwner()))
          use.set(sealed.getResult());
      groupBegin = groupEnd;
    }
  }

  if (independentDDRStages) {
    // Every selected baseline consumer is an independent op-wave, including
    // consumers in an observable output closure. Materializing only internal
    // consumers would leave the final output traversal free to fuse and
    // recompute the original functional closure across explicit edge actions.
    // Search-policy CoupledFusion/LocalShardResidency candidates deliberately
    // do not enter this baseline-only path. Their ordinary output traversal
    // remains the materialization owner; only CoupledFusion is later accepted
    // as an actual producer-fusion witness.
    llvm::SmallVector<MappedStrategy *, 16> consumerOrder;
    for (MappedStrategy &mapped : mappedStrategies)
      if (mapped.strategy.destinationTile == currentTile)
        consumerOrder.push_back(&mapped);
    llvm::sort(consumerOrder, [](const MappedStrategy *lhs,
                                 const MappedStrategy *rhs) {
      if (lhs->consumer != rhs->consumer)
        return lhs->consumer->isBeforeInBlock(rhs->consumer);
      if (lhs->strategy.consumerOffsets != rhs->strategy.consumerOffsets)
        return lhs->strategy.consumerOffsets < rhs->strategy.consumerOffsets;
      return lhs->strategy.consumerSizes < rhs->strategy.consumerSizes;
    });
    llvm::SmallVector<MaterializedSource, 16> independentConsumers;
    llvm::SmallVector<MappedStrategy *, 16> deferredRegionCuts;
    for (MappedStrategy *mappedPointer : consumerOrder) {
      MappedStrategy &mapped = *mappedPointer;
      SpatialEdgeStrategy &strategy = mapped.strategy;
      if (llvm::any_of(independentConsumers,
                       [&](const MaterializedSource &existing) {
                         return existing.producer == mapped.consumer &&
                                existing.result == 0 &&
                                existing.offsets == strategy.consumerOffsets &&
                                existing.sizes == strategy.consumerSizes;
                       }))
        continue;
      // A baseline RegionCut is the explicit input boundary of this op. It
      // must be created only after its producer's own incoming boundaries and
      // independent traversal are available; eager whole-function lowering
      // would cache a pre-boundary recursively fused producer closure.
      for (MappedStrategy &incoming : mappedStrategies) {
        if (incoming.consumer != mapped.consumer ||
            incoming.strategy.destinationTile != currentTile ||
            (incoming.strategy.action != SpatialEdgeAction::RegionCut &&
             incoming.strategy.action != SpatialEdgeAction::SpillReload) ||
            llvm::is_contained(deferredRegionCuts, &incoming))
          continue;
        bool createdRegionCut = false;
        if (mlir::failed(
                materializeSpill(scope, incoming, materialized, preserved,
                                 materializedRegionCutSpills, selectedDDRStages,
                                 createdRegionCut, mappedTemporalTiles,
                                 failureReason, &mappedOperationNodes)))
          return mlir::failure();
        deferredRegionCuts.push_back(&incoming);
        if (createdRegionCut)
          regionCuts.push_back(&incoming.strategy);
      }
      mlir::FailureOr<mlir::Value> consumer = getOrMaterializeSource(
          scope, mapped.consumer, /*producerResult=*/0,
          strategy.consumerOffsets, strategy.consumerSizes, materialized,
          preserved, mappedTemporalTiles, failureReason, &mappedOperationNodes);
      if (mlir::failed(consumer))
        return mlir::failure();
      auto selectedType =
          mlir::dyn_cast<mlir::RankedTensorType>(consumer->getType());
      if (!selectedType || !selectedType.hasStaticShape())
        return failResult(
            failureReason,
            "independent consumer materialization requires one static ranked "
            "tensor tile");
      mlir::Operation *selectedDefinition = consumer->getDefiningOp();
      if (!selectedDefinition ||
          selectedDefinition->getBlock() != &scope.getBody())
        return failResult(
            failureReason,
            "independent consumer tile is not available in the tensor-program "
            "body");
      mlir::Value carrierRoot = *consumer;
      while (true) {
        if (auto slice =
                carrierRoot.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
          carrierRoot = slice.getSource();
          continue;
        }
        if (auto expand =
                carrierRoot.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
          carrierRoot = expand.getSrc();
          continue;
        }
        if (auto collapse =
                carrierRoot.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
          carrierRoot = collapse.getSrc();
          continue;
        }
        if (auto cast = carrierRoot.getDefiningOp<mlir::tensor::CastOp>()) {
          carrierRoot = cast.getSource();
          continue;
        }
        break;
      }
      auto existingDDRView =
          carrierRoot.getDefiningOp<mlir::bufferization::ToTensorOp>();
      const bool reusesSealedDDR =
          existingDDRView && !existingDDRView.getWritable() &&
          isWaferDDRMemRefType(existingDDRView.getMemref().getType());
      mlir::Value selectedResult;
      if (reusesSealedDDR) {
        selectedResult = *consumer;
      } else {
        mlir::OpBuilder consumerBuilder(selectedDefinition);
        consumerBuilder.setInsertionPointAfter(selectedDefinition);
        auto selectedDest = consumerBuilder.create<mlir::tensor::EmptyOp>(
            mapped.consumer->getLoc(), selectedType.getShape(),
            selectedType.getElementType(), mlir::ValueRange{},
            selectedType.getEncoding());
        selectedResult =
            consumerBuilder
                .create<mlir::bufferization::MaterializeInDestinationOp>(
                    mapped.consumer->getLoc(), selectedType, *consumer,
                    selectedDest.getResult(), /*restrict=*/false,
                    /*writable=*/false)
                .getResult();
        preserved.insert(selectedResult.getDefiningOp());
      }
      // Wire one typed full-domain carrier into external fanout even when this
      // Tile owns only a spatial shard. Undefined regions are never read by a
      // legal selected local demand; peer fragments for other shards are
      // materialized independently on their owning Tiles. Unless an earlier
      // source-only stage already supplied a read-only DDR carrier, reopen the
      // independently executed result through DDR so outgoing peers and later
      // op waves cannot retain its pre-store SPM cache entry.
      mlir::Value cachedSelectedResult = selectedResult;
      if (!reusesSealedDDR) {
        auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
            mapped.consumer->getResult(0).getType());
        if (!fullType)
          return failResult(
              failureReason,
              "selected consumer result is not one ranked spatial domain");
        mlir::OpBuilder builder(selectedResult.getDefiningOp());
        builder.setInsertionPointAfter(selectedResult.getDefiningOp());
        auto ddrType = mlir::MemRefType::get(
            fullType.getShape(), fullType.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                            MemLayout::Tensor));
        auto allocation = builder.create<mlir::memref::AllocOp>(
            mapped.consumer->getLoc(), ddrType);
        auto destination = builder.create<mlir::bufferization::ToTensorOp>(
            mapped.consumer->getLoc(), allocation.getResult(),
            /*restrict=*/true, /*writable=*/true);
        mlir::Value stored =
            insertExactSlice(builder, mapped.consumer->getLoc(), selectedResult,
                             destination.getResult(), strategy.consumerOffsets,
                             strategy.consumerSizes);
        auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
            mapped.consumer->getLoc(), allocation.getResult(),
            /*restrict=*/false, /*writable=*/false);
        preserved.insert(allocation.getOperation());
        preserved.insert(destination.getOperation());
        preserved.insert(stored.getDefiningOp());
        preserved.insert(sealed.getOperation());
        selectedResult = sealed.getResult();
        cachedSelectedResult =
            createExactSlice(builder, mapped.consumer->getLoc(), selectedResult,
                             strategy.consumerOffsets, strategy.consumerSizes);
        preserved.insert(cachedSelectedResult.getDefiningOp());
      }
      independentConsumers.push_back(MaterializedSource{
          mapped.consumer, /*result=*/0,
          llvm::SmallVector<int64_t, 4>(strategy.consumerOffsets),
          llvm::SmallVector<int64_t, 4>(strategy.consumerSizes),
          cachedSelectedResult});
      for (MaterializedSource &cached : materialized)
        if (cached.producer == mapped.consumer && cached.result == 0 &&
            cached.offsets == strategy.consumerOffsets &&
            cached.sizes == strategy.consumerSizes)
          cached.value = cachedSelectedResult;
      if (selectedResult.getType() == mapped.consumer->getResult(0).getType()) {
        // Some legal destination-style traversals retain the original result as
        // an initialization dependency.  Do not rewrite uses inside the
        // selected value's own backward slice; rewrite every external fanout so
        // the new value becomes the common execution path without creating a
        // cyclic SSA definition.
        llvm::DenseSet<mlir::Value> visitedValues;
        llvm::DenseSet<mlir::Operation *> dependencyOperations;
        llvm::SmallVector<mlir::Value, 16> worklist{selectedResult};
        while (!worklist.empty()) {
          mlir::Value value = worklist.pop_back_val();
          if (!value || !visitedValues.insert(value).second)
            continue;
          mlir::Operation *definition = value.getDefiningOp();
          if (!definition || !dependencyOperations.insert(definition).second)
            continue;
          worklist.append(definition->operand_begin(),
                          definition->operand_end());
        }
        for (mlir::OpOperand &use : llvm::make_early_inc_range(
                 mapped.consumer->getResult(0).getUses()))
          if (!dependencyOperations.contains(use.getOwner()))
            use.set(selectedResult);
      }
    }
  }
  if (mlir::failed(requireLiveReceive("internal consumer materialization")))
    return mlir::failure();
  // Incoming values and every independently executed local consumer are now
  // sealed.  Outgoing endpoints may materialize their exact producer windows
  // only after that point, otherwise their cache can retain a recursively
  // fused pre-boundary closure.
  for (MappedStrategy &mapped : mappedStrategies) {
    SpatialEdgeStrategy &strategy = mapped.strategy;
    if (strategy.action != SpatialEdgeAction::PeerFragments)
      continue;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Peer ||
          fragment.sourceTile != currentTile)
        continue;
      mlir::FailureOr<mlir::Value> value = getOrMaterializeSource(
          scope, mapped.producer, strategy.producerResult, fragment.offsets,
          fragment.sizes, materialized, preserved, mappedTemporalTiles,
          failureReason, &mappedOperationNodes);
      if (mlir::failed(value))
        return mlir::failure();
      endpoints.push_back(CandidatePeerEndpoint{
          *value, /*carrierBuffer=*/{}, CandidatePeerEndpointKind::Send,
          strategy.destinationTile, fragment.bytes, fragment.communicationId,
          fragment.payloadSlice, mapped.consumerScheduleOrdinal,
          strategy.consumerOperand, &fragment});
    }
  }
  if (mlir::failed(requireLiveReceive("outgoing peer materialization")))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 8> preservedOperations;
  for (mlir::Operation *operation : preserved)
    preservedOperations.push_back(operation);
  if (!outputShards.empty()) {
    mlir::func::ReturnOp returnOp = scope.getReturn();
    const bool directFullOutputs =
        outputShards.size() == scope.getOutputCount() &&
        llvm::all_of(outputShards, [&](const SpatialOutputShard &shard) {
          if (shard.outputIndex >= returnOp.getNumOperands() ||
              shard.temporalTileSizes != shard.sizes)
            return false;
          mlir::Operation *root =
              returnOp.getOperand(shard.outputIndex).getDefiningOp();
          return root &&
                 isFullStaticResultDomain(root, /*resultNumber=*/0,
                                          shard.offsets, shard.sizes) &&
                 isOneFullTemporalWave(root, mappedTemporalTiles) &&
                 llvm::any_of(
                     mappedStrategies, [&](const MappedStrategy &mapped) {
                       return mapped.strategy.destinationTile == currentTile &&
                              mapped.consumer == root;
                     });
        });
    if (directFullOutputs) {
      // Keep a one-wave observable consumer on its current SSA path so an
      // incoming selected peer fragment remains the exact value consumed by
      // that operation.  The ordinary full-result insert still writes the
      // functional output boundary; only the unnecessary clone-and-retile is
      // omitted.
      for (const SpatialOutputShard &shard : outputShards) {
        mlir::Value value = returnOp.getOperand(shard.outputIndex);
        mlir::Operation *root = value.getDefiningOp();
        mlir::FailureOr<mlir::Value> boundary =
            getCandidateOutputBoundary(scope, shard.outputIndex, failureReason);
        if (mlir::failed(boundary))
          return mlir::failure();
        mlir::OpBuilder builder(returnOp);
        returnOp->setOperand(shard.outputIndex,
                             insertExactSlice(builder, root->getLoc(), value,
                                              *boundary, shard.offsets,
                                              shard.sizes));
        preserved.insert(root);
      }
    } else if (mlir::failed(materializeCandidateOutputTileSlices(
                   scope, outputShards, mappedTemporalTiles, failureReason,
                   &mappedOperationNodes, preservedOperations))) {
      return mlir::failure();
    }
  } else {
    mlir::func::ReturnOp returnOp = scope.getReturn();
    for (unsigned index = 0; index < scope.getOutputCount(); ++index) {
      mlir::FailureOr<mlir::Value> output =
          getCandidateOutputBoundary(scope, index, failureReason);
      if (mlir::failed(output))
        return mlir::failure();
      returnOp->setOperand(index, *output);
    }
  }
  if (mlir::failed(requireLiveReceive("output traversal materialization")))
    return mlir::failure();
  // A receive being syntactically used is insufficient: after the selected
  // output traversal has sealed the actual result path, its assembled value
  // must reach an observable local output or an outgoing peer payload.  The
  // former pre-output check incorrectly rejected a legal temporally tiled
  // baseline because the output traversal itself was the first observable
  // sink for the independently materialized consumer wave.
  mlir::func::ReturnOp finalReturn = scope.getReturn();
  for (const CandidatePeerEndpoint &receive : endpoints) {
    if (receive.kind != CandidatePeerEndpointKind::Receive)
      continue;
    mlir::Operation *receiveDefinition = receive.value.getDefiningOp();
    bool reachesExecutionSink = false;
    for (mlir::Value output : finalReturn.getOperands()) {
      llvm::DenseSet<mlir::Value> visited;
      if (isInBackwardClosure(output, receiveDefinition, visited)) {
        reachesExecutionSink = true;
        break;
      }
    }
    for (const CandidatePeerEndpoint &send : endpoints) {
      if (reachesExecutionSink || send.kind != CandidatePeerEndpointKind::Send)
        continue;
      llvm::DenseSet<mlir::Value> visited;
      if (isInBackwardClosure(send.value, receiveDefinition, visited))
        reachesExecutionSink = true;
    }
    // An explicit DDR stage intentionally breaks the tensor SSA path after
    // its functional insert: the store stays live, while a fresh read-only
    // to_tensor view prevents downstream temporal tiling from fusing the
    // producer into the consumer wave. Count that selected store as an
    // execution sink only when the receive reaches the inserted source and
    // the destination is rooted in a marked compiler-owned stage. The later
    // TileRegion split still proves the ordered store/reload interval.
    if (!reachesExecutionSink)
      reachesExecutionSink = reachesSelectedDDRStage(receiveDefinition);
    if (!reachesExecutionSink)
      return failResult(
          failureReason,
          (llvm::Twine("selected receive does not reach an execution sink "
                       "after output traversal (communication_id=") +
           llvm::Twine(receive.communicationId) +
           ", payload_slice=" + llvm::Twine(receive.payloadSlice) + ")")
              .str());
  }
  // Send/receive-only Tiles need the unused remainder removed before body
  // conversion. Ordinary local actions retain the same source closure as the
  // direct output-shard path; its materializer and atomic conversion own that
  // cleanup.
  if (!endpoints.empty() || !materialized.empty())
    eraseDeadExcept(scope, preserved);

  if (mlir::failed(rebindSelectedReceiveEndpoints(endpoints, mappedStrategies,
                                                  failureReason)))
    return mlir::failure();

  for (const CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive)
      continue;
    mlir::Operation *defining = endpoint.value.getDefiningOp();
    if (!defining || defining->getBlock() != &scope.getBody() ||
        endpoint.value.use_empty())
      return failResult(
          failureReason,
          "selected consumer traversal does not use an exact receive fragment");
  }

  llvm::DenseSet<mlir::Operation *> liveOperations;
  candidate->walk(
      [&](mlir::Operation *operation) { liveOperations.insert(operation); });
  llvm::erase_if(mappedOperationNodes, [&](const auto &mapping) {
    return !liveOperations.contains(mapping.operation);
  });
  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidate, sourceModule.getContext(), functionalArgumentCount,
          currentLogicalPartition, failureReason,
          /*suppressDiagnostics=*/true,
          /*verifyResult=*/true, /*populateFallbackFailureReason=*/true,
          endpoints, selectedDDRStages, &emissionRelations,
          mappedOperationNodes))) {
    return mlir::failure();
  }
  // Split boundaries in their actual materialized store order. Original DAG
  // consumer order is insufficient when support chains, shared producer
  // spills, or peer assembly reorder the concrete stores. Splitting a later
  // interval first can turn an earlier spill into a TileRegion argument and
  // leave its marked allocation on the other side, losing the exact store /
  // reload identity. The TileRegion IR is the authoritative schedule here.
  struct MaterializedRegionCut {
    const SpatialEdgeStrategy *marker = nullptr;
    TileRegionOp region;
    mlir::memref::AllocOp spillAllocation;
    mlir::Operation *lastStoreRoot = nullptr;
    mlir::Operation *firstPeerActionRoot = nullptr;
  };
  llvm::SmallVector<MaterializedRegionCut, 8> orderedRegionCuts;
  llvm::DenseMap<const SpatialEdgeStrategy *, size_t> regionCutIndices;
  for (const SpatialEdgeStrategy *regionCut : regionCuts) {
    regionCutIndices.try_emplace(regionCut, orderedRegionCuts.size());
    orderedRegionCuts.push_back(MaterializedRegionCut{regionCut});
  }
  // TileRegion lowering reports the exact allocation relation directly. The
  // relation is invocation-local C++ state keyed by current SSA.
  for (const MaterializedSelectedDDRStage &stage :
       emissionRelations.selectedDDRStages) {
    auto found = regionCutIndices.find(stage.strategy);
    if (found == regionCutIndices.end())
      continue;
    MaterializedRegionCut &materialized = orderedRegionCuts[found->second];
    if (materialized.spillAllocation)
      return failResult(
          failureReason,
          "selected DDR stage materialized more than one exact allocation");
    materialized.spillAllocation = stage.allocation;
    materialized.region = stage.allocation->getParentOfType<TileRegionOp>();
  }
  llvm::DenseMap<mlir::Value, size_t> cutByAllocation;
  for (auto [index, materialized] : llvm::enumerate(orderedRegionCuts))
    if (materialized.spillAllocation)
      cutByAllocation.try_emplace(materialized.spillAllocation.getResult(),
                                  index);
  candidate->walk([&](StorageStoreOp store) {
    TileRegionOp owner = store->getParentOfType<TileRegionOp>();
    if (!owner || !owner.getBody().hasOneBlock())
      return;
    auto found = cutByAllocation.find(getViewRoot(store.getDest()));
    if (found == cutByAllocation.end())
      return;
    MaterializedRegionCut &materialized = orderedRegionCuts[found->second];
    mlir::Operation *root = store.getOperation();
    while (root && root->getBlock() != &owner.getBody().front())
      root = root->getParentOp();
    if (!root)
      return;
    if (!materialized.lastStoreRoot || materialized.region != owner ||
        materialized.lastStoreRoot->isBeforeInBlock(root)) {
      materialized.region = owner;
      materialized.lastStoreRoot = root;
    }
  });
  bool ambiguousPeerOwner = false;
  candidate->walk([&](mlir::Operation *operation) {
    if (!mlir::isa<CommPeerSendOp, CommPeerRecvOp>(operation))
      return;
    std::optional<size_t> cutIndex;
    for (auto [index, materialized] : llvm::enumerate(orderedRegionCuts)) {
      if (!isPeerEndpointForStrategy(operation, *materialized.marker))
        continue;
      if (cutIndex && *cutIndex != index) {
        ambiguousPeerOwner = true;
        return;
      }
      cutIndex = index;
    }
    if (!cutIndex)
      return;
    MaterializedRegionCut &materialized = orderedRegionCuts[*cutIndex];
    TileRegionOp owner = operation->getParentOfType<TileRegionOp>();
    if (!owner || owner != materialized.region ||
        !owner.getBody().hasOneBlock())
      return;
    mlir::Operation *root = operation;
    while (root && root->getBlock() != &owner.getBody().front())
      root = root->getParentOp();
    if (root && (!materialized.firstPeerActionRoot ||
                 root->isBeforeInBlock(materialized.firstPeerActionRoot)))
      materialized.firstPeerActionRoot = root;
  });
  if (ambiguousPeerOwner)
    return failResult(failureReason,
                      "selected peer message has multiple DDR stage owners");
  for (const MaterializedRegionCut &materialized : orderedRegionCuts)
    if (!materialized.region || !materialized.spillAllocation ||
        !materialized.lastStoreRoot)
      return failResult(
          failureReason,
          "selected DDR stage did not materialize one marked exact store");
  llvm::sort(orderedRegionCuts, [](const MaterializedRegionCut &lhs,
                                   const MaterializedRegionCut &rhs) {
    if (lhs.region != rhs.region)
      return lhs.region->isBeforeInBlock(rhs.region);
    mlir::Operation *lhsRoot =
        lhs.firstPeerActionRoot ? lhs.firstPeerActionRoot : lhs.lastStoreRoot;
    mlir::Operation *rhsRoot =
        rhs.firstPeerActionRoot ? rhs.firstPeerActionRoot : rhs.lastStoreRoot;
    if (lhsRoot != rhsRoot)
      return lhsRoot->isBeforeInBlock(rhsRoot);
    return lhs.lastStoreRoot->isBeforeInBlock(rhs.lastStoreRoot);
  });
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "transformation-phase", "selected-edge-materialization",
        "split-ddr-stages");
    llvm::SmallVector<mlir::Value, 8> materializedStageBuffers;
    for (const MaterializedSelectedDDRStage &stage :
         emissionRelations.selectedDDRStages)
      materializedStageBuffers.push_back(stage.allocation->getResult(0));
    for (const MaterializedRegionCut &regionCut : orderedRegionCuts)
      if (mlir::failed(splitAtRegionCut(
              regionCut.spillAllocation, regionCut.marker,
              materializedStageBuffers, &emissionRelations.materializedBuffers,
              failureReason)))
        return mlir::failure();
  }
  // Query-local exact carrier caches have to survive until endpoint emission
  // because an endpoint is not itself an SSA use. Repeated DDR stage splits
  // can leave an eagerly converted load in another region after a different
  // containing cache supplies the eventual endpoint. Once every split is
  // final, a load into a direct private allocation with no other use is
  // unobservable; remove that unused peer load and its allocation
  // without changing any load feeding a send, compute, store, view, or region
  // result. The split effect closure above keeps each such reader with its
  // initializing load.
  eraseUnreadDirectPrivateLoads(candidate->getOperation());
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "analysis-phase", "selected-edge-materialization", "verify");
    if (mlir::failed(mlir::verify(*candidate)))
      return failResult(failureReason,
                        "edge-action TileRegion result is not verifier-legal");
  }
  if (materializationRelations)
    *materializationRelations =
        std::move(emissionRelations.materializedBuffers);
  module = std::move(candidate);
  return mlir::success();
}
