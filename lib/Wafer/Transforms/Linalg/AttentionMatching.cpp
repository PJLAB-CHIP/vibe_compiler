//===- AttentionMatching.cpp - Structured attention graph proof --------===//

#include "AttentionMatching.h"
#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#include <array>
#include <functional>
#include <limits>
#include <optional>

namespace wafer::attention_normalization {
namespace {

#define DEBUG_TYPE "wafer-attention-normalization"

struct Contraction {
  mlir::linalg::LinalgOp operation;
  mlir::Value left;
  mlir::Value right;
  llvm::SmallVector<mlir::AffineMap, 3> maps;
  llvm::SmallVector<mlir::utils::IteratorType, 6> iterators;
  llvm::SmallVector<unsigned, 2> reductionDimensions;
};

struct Softmax {
  mlir::linalg::GenericOp rowMax;
  mlir::Value scores;
  mlir::AffineMap scoreMap;
  mlir::Value probability;
};

struct ScoreExpression {
  Contraction contraction;
  mlir::Value scale;
  mlir::Value mask;
  mlir::AffineMap maskMap;
  mlir::AffineMap scoreOutputMap;
};

struct FunctionalCacheAppend {
  mlir::Value updated;
  int64_t dimension = -1;
  int64_t pastExtent = 0;
  int64_t appendedExtent = 0;
};

bool isProjectedPermutation(mlir::AffineMap map) {
  if (!map || map.getNumSymbols() != 0 || !map.isProjectedPermutation())
    return false;
  llvm::SmallBitVector seen(map.getNumDims(), false);
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || seen.test(dimension.getPosition()))
      return false;
    seen.set(dimension.getPosition());
  }
  return true;
}

llvm::SmallBitVector getMapDimensionSet(mlir::AffineMap map) {
  llvm::SmallBitVector set(map.getNumDims(), false);
  for (mlir::AffineExpr expression : map.getResults())
    set.set(mlir::cast<mlir::AffineDimExpr>(expression).getPosition());
  return set;
}

bool isAllParallel(mlir::linalg::LinalgOp operation) {
  return llvm::all_of(operation.getIteratorTypesArray(), [](auto iterator) {
    return iterator == mlir::utils::IteratorType::parallel;
  });
}

template <typename Operation>
bool matchBinaryBody(mlir::linalg::GenericOp generic, bool exactOperandOrder) {
  if (!generic || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      !isAllParallel(generic))
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<Operation>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!operation || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0))
    return false;
  mlir::Value left = generic.getRegionInputArgs()[0];
  mlir::Value right = generic.getRegionInputArgs()[1];
  if (operation->getOperand(0) == left && operation->getOperand(1) == right)
    return true;
  return !exactOperandOrder && operation->getOperand(0) == right &&
         operation->getOperand(1) == left;
}

template <typename Operation>
bool matchUnaryBody(mlir::linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      !isAllParallel(generic))
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<Operation>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return operation && yield && yield.getValues().size() == 1 &&
         operation->getOperand(0) == generic.getRegionInputArgs().front() &&
         yield.getValues().front() == operation->getResult(0);
}

template <typename Operation>
bool matchReductionBody(mlir::linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<Operation>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!operation || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0))
    return false;
  mlir::Value input = generic.getRegionInputArgs().front();
  mlir::Value accumulator = generic.getRegionOutputArgs().front();
  return (operation->getOperand(0) == input &&
          operation->getOperand(1) == accumulator) ||
         (operation->getOperand(1) == input &&
          operation->getOperand(0) == accumulator);
}

bool matchBroadcastBody(mlir::linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      !isAllParallel(generic))
    return false;
  mlir::Block &body = generic.getRegion().front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return body.without_terminator().empty() && yield &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == generic.getRegionInputArgs().front();
}

bool matchMultiplyAccumulateBody(mlir::linalg::LinalgOp operation) {
  if (operation.getRegionInputArgs().size() != 2 ||
      operation.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  if (std::distance(body.begin(), body.end()) != 3)
    return false;
  auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(&body.front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(&*std::next(body.begin()));
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!multiply || !add || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != add.getResult())
    return false;
  mlir::Value left = operation.getRegionInputArgs()[0];
  mlir::Value right = operation.getRegionInputArgs()[1];
  mlir::Value accumulator = operation.getRegionOutputArgs().front();
  const bool exactMultiply =
      (multiply.getLhs() == left && multiply.getRhs() == right) ||
      (multiply.getLhs() == right && multiply.getRhs() == left);
  const bool exactAdd =
      (add.getLhs() == multiply.getResult() && add.getRhs() == accumulator) ||
      (add.getRhs() == multiply.getResult() && add.getLhs() == accumulator);
  return exactMultiply && exactAdd;
}

std::optional<mlir::FloatAttr> getFloatConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attribute = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
  if (!attribute)
    return std::nullopt;
  return attribute;
}

std::optional<mlir::FloatAttr> getFillConstant(mlir::Value value) {
  auto fill = value.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getNumDpsInputs() != 1 || fill.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1)
    return std::nullopt;
  return getFloatConstant(fill.getDpsInputs().front());
}

bool isZeroFill(mlir::Value value) {
  std::optional<mlir::FloatAttr> constant = getFillConstant(value);
  return constant && constant->getValue().isZero();
}

bool isMaximumIdentityFill(mlir::Value value) {
  std::optional<mlir::FloatAttr> constant = getFillConstant(value);
  if (!constant)
    return false;
  const llvm::APFloat &number = constant->getValue();
  if (number.isNegInfinity())
    return true;
  llvm::APFloat lowest = llvm::APFloat::getLargest(number.getSemantics(), true);
  return number.bitwiseIsEqual(lowest);
}

std::optional<Contraction> matchContraction(mlir::Value value) {
  auto operation =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
  if (!operation || operation->getNumResults() != 1 ||
      operation.getNumDpsInputs() != 2 || operation.getNumDpsInits() != 1 ||
      !matchMultiplyAccumulateBody(operation) ||
      !isZeroFill(operation.getDpsInits().front()))
    return std::nullopt;

  llvm::SmallVector<mlir::AffineMap, 3> maps(operation.getIndexingMapsArray());
  if (maps.size() != 3 || llvm::any_of(maps, [](mlir::AffineMap map) {
        return !isProjectedPermutation(map);
      }))
    return std::nullopt;

  llvm::SmallVector<mlir::utils::IteratorType, 6> iterators(
      operation.getIteratorTypesArray());
  llvm::SmallVector<unsigned, 2> reductionDimensions;
  for (auto [dimension, iterator] : llvm::enumerate(iterators)) {
    if (iterator == mlir::utils::IteratorType::reduction)
      reductionDimensions.push_back(dimension);
    else if (iterator != mlir::utils::IteratorType::parallel)
      return std::nullopt;
  }
  if (reductionDimensions.empty())
    return std::nullopt;
  llvm::SmallBitVector leftDimensions = getMapDimensionSet(maps[0]);
  llvm::SmallBitVector rightDimensions = getMapDimensionSet(maps[1]);
  llvm::SmallBitVector outputDimensions = getMapDimensionSet(maps[2]);
  for (unsigned dimension : reductionDimensions) {
    if (!leftDimensions.test(dimension) || !rightDimensions.test(dimension) ||
        outputDimensions.test(dimension))
      return std::nullopt;
  }
  return Contraction{operation,
                     operation.getDpsInputs()[0],
                     operation.getDpsInputs()[1],
                     std::move(maps),
                     std::move(iterators),
                     std::move(reductionDimensions)};
}

bool mapsEqual(mlir::linalg::LinalgOp operation,
               llvm::ArrayRef<mlir::AffineMap> expected) {
  return llvm::equal(operation.getIndexingMapsArray(), expected);
}

bool matchFloatConversionBody(mlir::linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      !isAllParallel(generic))
    return false;
  llvm::SmallVector<mlir::AffineMap, 2> maps(generic.getIndexingMapsArray());
  if (maps.size() != 2 || maps[0] != maps[1] ||
      !isProjectedPermutation(maps[0]))
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  mlir::Operation &conversion = body.front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(conversion) &&
         conversion.getNumOperands() == 1 && conversion.getNumResults() == 1 &&
         conversion.getOperand(0) == generic.getRegionInputArgs().front() &&
         yield && yield.getValues().size() == 1 &&
         yield.getValues().front() == conversion.getResult(0);
}

bool matchPermutationBody(mlir::linalg::GenericOp generic) {
  if (!matchBroadcastBody(generic))
    return false;
  llvm::SmallVector<mlir::AffineMap, 2> maps(generic.getIndexingMapsArray());
  return maps.size() == 2 && isProjectedPermutation(maps[0]) &&
         isProjectedPermutation(maps[1]) &&
         maps[0].getNumResults() == maps[1].getNumResults();
}

mlir::Value getTransparentSource(mlir::Operation *operation) {
  if (!operation || operation->getNumResults() != 1)
    return {};
  if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
    return cast.getSource();
  if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation))
    return collapse.getSrc();
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation))
    return expand.getSrc();
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  if (matchFloatConversionBody(generic) || matchPermutationBody(generic))
    return generic.getDpsInputs().front();
  return {};
}

mlir::Value getTransparentLayoutSource(mlir::Operation *operation) {
  if (!operation || operation->getNumResults() != 1)
    return {};
  if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
    return cast.getSource();
  if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation))
    return collapse.getSrc();
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation))
    return expand.getSrc();
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  return matchPermutationBody(generic) ? generic.getDpsInputs().front()
                                       : mlir::Value{};
}

mlir::Value stripTransparentValue(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    mlir::Value source = getTransparentSource(value.getDefiningOp());
    if (!source)
      return value;
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        sourceType.getNumElements() != resultType.getNumElements())
      return value;
    value = source;
  }
  return {};
}

std::optional<Softmax> matchSoftmax(mlir::Value value) {
  auto probability = value.getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchBinaryBody<mlir::arith::DivFOp>(probability,
                                            /*exactOperandOrder=*/true))
    return std::nullopt;
  auto exponential =
      probability.getDpsInputs()[0].getDefiningOp<mlir::linalg::GenericOp>();
  auto sumBroadcast =
      probability.getDpsInputs()[1].getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchUnaryBody<mlir::math::ExpOp>(exponential) ||
      !matchBroadcastBody(sumBroadcast))
    return std::nullopt;
  auto shifted =
      exponential.getDpsInputs()[0].getDefiningOp<mlir::linalg::GenericOp>();
  auto rowSum =
      sumBroadcast.getDpsInputs()[0].getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchBinaryBody<mlir::arith::SubFOp>(shifted,
                                            /*exactOperandOrder=*/true) ||
      !matchReductionBody<mlir::arith::AddFOp>(rowSum) ||
      rowSum.getDpsInputs()[0] != exponential.getResult(0) ||
      !isZeroFill(rowSum.getDpsInits().front()))
    return std::nullopt;
  auto maxBroadcast =
      shifted.getDpsInputs()[1].getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchBroadcastBody(maxBroadcast))
    return std::nullopt;
  auto rowMax =
      maxBroadcast.getDpsInputs()[0].getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchReductionBody<mlir::arith::MaximumFOp>(rowMax) ||
      rowMax.getDpsInputs()[0] != shifted.getDpsInputs()[0] ||
      !isMaximumIdentityFill(rowMax.getDpsInits().front()))
    return std::nullopt;

  llvm::SmallVector<mlir::AffineMap, 3> rowMaxMaps(
      rowMax.getIndexingMapsArray());
  if (rowMaxMaps.size() != 2 || !isProjectedPermutation(rowMaxMaps[0]) ||
      !isProjectedPermutation(rowMaxMaps[1]) ||
      !mapsEqual(rowSum, rowMaxMaps) ||
      !mapsEqual(maxBroadcast, {rowMaxMaps[1], rowMaxMaps[0]}) ||
      !mapsEqual(sumBroadcast, {rowMaxMaps[1], rowMaxMaps[0]}) ||
      !mapsEqual(shifted, {rowMaxMaps[0], rowMaxMaps[0], rowMaxMaps[0]}) ||
      !mapsEqual(exponential, {rowMaxMaps[0], rowMaxMaps[0]}) ||
      !mapsEqual(probability, {rowMaxMaps[0], rowMaxMaps[0], rowMaxMaps[0]}))
    return std::nullopt;

  llvm::SmallVector<mlir::utils::IteratorType, 6> maxIterators(
      rowMax.getIteratorTypesArray());
  if (!llvm::equal(maxIterators, rowSum.getIteratorTypesArray()) ||
      llvm::none_of(maxIterators, [](auto iterator) {
        return iterator == mlir::utils::IteratorType::reduction;
      }))
    return std::nullopt;
  llvm::SmallBitVector rowDimensions = getMapDimensionSet(rowMaxMaps[1]);
  for (auto [dimension, iterator] : llvm::enumerate(maxIterators)) {
    const bool isReduction = iterator == mlir::utils::IteratorType::reduction;
    if (isReduction == rowDimensions.test(dimension))
      return std::nullopt;
  }

  return Softmax{rowMax, shifted.getDpsInputs()[0], rowMaxMaps[0],
                 probability.getResult(0)};
}

mlir::Value traceScalarSource(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (mlir::isa<mlir::FloatType>(value.getType()))
      return value;
    auto shapedType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (shapedType && shapedType.getRank() == 0 &&
        mlir::isa<mlir::FloatType>(shapedType.getElementType()))
      return value;
    if (shapedType && shapedType.hasStaticShape() &&
        mlir::isa<mlir::FloatType>(shapedType.getElementType())) {
      auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
      auto elements =
          constant
              ? mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())
              : mlir::DenseElementsAttr{};
      if (elements && elements.isSplat())
        return value;
    }
    if (auto cast = value.getDefiningOp<mlir::tensor::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    auto broadcast = value.getDefiningOp<mlir::linalg::GenericOp>();
    if (!matchBroadcastBody(broadcast))
      return {};
    value = broadcast.getDpsInputs().front();
  }
  return {};
}

std::optional<ScoreExpression> matchScaledContraction(mlir::Value value) {
  value = stripTransparentValue(value);
  if (std::optional<Contraction> direct = matchContraction(value))
    return ScoreExpression{*direct, {}, {}, {}, direct->maps[2]};
  auto multiply = value.getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchBinaryBody<mlir::arith::MulFOp>(multiply,
                                            /*exactOperandOrder=*/false))
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 3> maps(multiply.getIndexingMapsArray());
  for (unsigned contractionIndex : {0u, 1u}) {
    std::optional<Contraction> contraction = matchContraction(
        stripTransparentValue(multiply.getDpsInputs()[contractionIndex]));
    mlir::Value scale =
        traceScalarSource(multiply.getDpsInputs()[1 - contractionIndex]);
    if (!contraction || !scale)
      continue;
    if (maps.size() != 3 || maps[contractionIndex] != maps[2] ||
        !isProjectedPermutation(maps[2]))
      continue;
    return ScoreExpression{*contraction, scale, {}, {}, maps[2]};
  }
  return std::nullopt;
}

std::optional<ScoreExpression> matchScoreExpression(mlir::Value value) {
  value = stripTransparentValue(value);
  if (std::optional<ScoreExpression> direct = matchScaledContraction(value))
    return direct;

  auto add = value.getDefiningOp<mlir::linalg::GenericOp>();
  if (!matchBinaryBody<mlir::arith::AddFOp>(add,
                                            /*exactOperandOrder=*/false))
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 3> maps(add.getIndexingMapsArray());
  if (maps.size() != 3 || !isProjectedPermutation(maps[2]))
    return std::nullopt;
  for (unsigned scoreIndex : {0u, 1u}) {
    std::optional<ScoreExpression> score =
        matchScaledContraction(add.getDpsInputs()[scoreIndex]);
    if (!score || maps[scoreIndex] != maps[2] ||
        !isProjectedPermutation(maps[1 - scoreIndex]))
      continue;
    score->mask = add.getDpsInputs()[1 - scoreIndex];
    score->maskMap = maps[1 - scoreIndex];
    score->scoreOutputMap = maps[2];
    // Prove the mask relation while forming the semantic attention operand.
    // A pure forwarding producer need not materialize its broadcast domain.
    // Other uses keep the producer; only the matched attention input changes.
    while (auto producer =
               score->mask.getDefiningOp<mlir::linalg::GenericOp>()) {
      if (!matchBroadcastBody(producer) || !producer.hasPureTensorSemantics())
        break;
      auto producerMaps = producer.getIndexingMapsArray();
      if (producerMaps.size() != 2 ||
          !isProjectedPermutation(producerMaps[0]) ||
          !producerMaps[1].isPermutation())
        break;
      mlir::AffineMap composed =
          producerMaps[0]
              .compose(mlir::inversePermutation(producerMaps[1]))
              .compose(score->maskMap);
      if (!isProjectedPermutation(composed))
        break;
      score->mask = producer.getDpsInputs().front();
      score->maskMap = composed;
    }
    return score;
  }
  return std::nullopt;
}

std::optional<mlir::AffineMap> translateMap(mlir::AffineMap map,
                                            llvm::ArrayRef<int64_t> oldToGlobal,
                                            unsigned globalRank) {
  llvm::SmallVector<mlir::AffineExpr, 4> expressions;
  expressions.reserve(map.getNumResults());
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= oldToGlobal.size() ||
        oldToGlobal[dimension.getPosition()] < 0)
      return std::nullopt;
    expressions.push_back(mlir::getAffineDimExpr(
        oldToGlobal[dimension.getPosition()], map.getContext()));
  }
  return mlir::AffineMap::get(globalRank, 0, expressions, map.getContext());
}

struct LogicalAxisAssignment {
  llvm::SmallVector<std::array<unsigned, 4>, 4> batch;
  llvm::SmallVector<std::array<unsigned, 2>, 2> query;
  llvm::SmallVector<std::array<unsigned, 2>, 2> queryKey;
  llvm::SmallVector<std::array<unsigned, 3>, 2> keyValue;
  llvm::SmallVector<unsigned, 2> valueOutput;
};

struct LogicalAttentionMaps {
  mlir::Value query;
  mlir::Value key;
  mlir::Value value;
  mlir::RankedTensorType outputType;
  llvm::SmallVector<mlir::AffineMap, 6> maps;
  llvm::SmallVector<unsigned, 2> keyValueGlobalDimensions;
};

constexpr uint64_t kMaximumAxisProofWork = 1u << 16;

bool isValueAncestor(mlir::Value ancestor, mlir::Value value);

std::optional<analysis::IndexRelation>
getValueRelation(mlir::Value value, mlir::Value ancestor, uint64_t &work) {
  using analysis::IndexRelation;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  auto relation = IndexRelation::identity(type.getShape());
  llvm::DenseSet<mlir::Value> visited;
  while (value != ancestor) {
    if (++work > kMaximumAxisProofWork || !visited.insert(value).second)
      return std::nullopt;
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    mlir::Value source = getTransparentSource(result.getOwner());
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(result.getOwner());
    if (!source && linalg && isAllParallel(linalg)) {
      // The scalar score/softmax skeleton has already been matched. Here we
      // only follow its unambiguous indexing path to the selected ancestor.
      for (mlir::Value input : linalg.getDpsInputs())
        if (isValueAncestor(ancestor, input)) {
          if (source)
            return std::nullopt;
          source = input;
        }
    }
    if (!source)
      return std::nullopt;
    analysis::IndexRelationResult step;
    if (linalg) {
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          !isAllParallel(linalg) || !linalg.getShapesToLoopsMap())
        return std::nullopt;
      auto operand = llvm::find(linalg->getOperands(), source);
      if (operand == linalg->operand_end())
        return std::nullopt;
      auto outputMap = analysis::getStructuredResultMap(result);
      auto inputMap = analysis::getStructuredOperandMap(
          linalg->getOpOperand(operand - linalg->operand_begin()));
      if (mlir::failed(outputMap) || mlir::failed(inputMap))
        return std::nullopt;
      step = IndexRelation::fromCommonIterationDomain(
          *outputMap,
          mlir::cast<mlir::RankedTensorType>(value.getType()).getShape(),
          *inputMap, sourceType.getShape(), linalg.getStaticLoopRanges());
    } else {
      auto indexing = analysis::deriveTensorResultIndexing(result);
      if (!indexing.isExact() || indexing.indexing->operands.size() != 1)
        return std::nullopt;
      const auto &operand = indexing.indexing->operands.front();
      if (result.getOwner()->getOperand(operand.operand) != source)
        return std::nullopt;
      step.status = analysis::IndexRelationStatus::Exact;
      step.relation = operand.resultToOperand;
    }
    if (!relation.isExact() || !step.isExact())
      return std::nullopt;
    relation = relation.get()->compose(*step.get());
    if (!relation.isExact())
      return std::nullopt;
    value = source;
  }
  return relation.isExact() ? std::optional<IndexRelation>(*relation.get())
                            : std::nullopt;
}

std::optional<analysis::IndexRelation>
getContractionIterationRelation(const Contraction &contraction,
                                const analysis::IndexRelation &globalToLeft,
                                const analysis::IndexRelation &globalToRight,
                                llvm::ArrayRef<int64_t> globalShape) {
  using analysis::IndexRelation;
  auto operation = contraction.operation;
  if (!operation.getShapesToLoopsMap())
    return std::nullopt;
  auto iterations = operation.getStaticLoopRanges();
  std::optional<IndexRelation> iteration;
  for (unsigned operand : {0u, 1u}) {
    auto type = mlir::cast<mlir::RankedTensorType>(
        operation.getDpsInputs()[operand].getType());
    auto access = IndexRelation::fromAffineMap(contraction.maps[operand],
                                               iterations, type.getShape());
    if (!access.isExact())
      return std::nullopt;
    auto inverse = access.get()->inverse();
    if (!inverse.isExact())
      return std::nullopt;
    auto joined =
        (operand == 0 ? globalToLeft : globalToRight).compose(*inverse.get());
    if (!joined.isExact())
      return std::nullopt;
    iteration =
        iteration ? IndexRelation(iteration->getPresburgerRelation().intersect(
                                      joined.get()->getPresburgerRelation()),
                                  analysis::IndexRelationStatus::Exact)
                  : *joined.get();
  }
  auto domain = IndexRelation::staticDomain(globalShape);
  if (!domain.isExact() || !iteration ||
      !iteration->isFunctional().isProvenTrue() ||
      !iteration->getPresburgerRelation().getDomainSet().isEqual(*domain.set))
    return std::nullopt;
  return iteration;
}

bool proveReductionOrder(const analysis::IndexRelation &globalToIteration,
                         mlir::linalg::LinalgOp operation,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<unsigned> globalReductionAxes) {
  using analysis::IndexRelation;
  auto iterationShape = operation.getStaticLoopRanges();
  llvm::SmallVector<mlir::AffineExpr, 4> actualAxes;
  llvm::SmallVector<int64_t, 4> actualShape;
  for (auto [axis, kind] : llvm::enumerate(operation.getIteratorTypesArray()))
    if (kind == mlir::utils::IteratorType::reduction) {
      actualAxes.push_back(
          mlir::getAffineDimExpr(axis, operation.getContext()));
      actualShape.push_back(iterationShape[axis]);
    }
  llvm::SmallVector<mlir::AffineExpr, 4> expectedAxes;
  llvm::SmallVector<int64_t, 4> expectedShape;
  for (unsigned axis : globalReductionAxes) {
    expectedAxes.push_back(
        mlir::getAffineDimExpr(axis, operation.getContext()));
    expectedShape.push_back(globalShape[axis]);
  }
  auto actualProjection = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(iterationShape.size(), 0, actualAxes,
                           operation.getContext()),
      iterationShape, actualShape);
  auto expectedProjection = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(globalShape.size(), 0, expectedAxes,
                           operation.getContext()),
      globalShape, expectedShape);
  auto reshape = IndexRelation::staticReshape(expectedShape, actualShape);
  if (!actualProjection.isExact() || !expectedProjection.isExact() ||
      !reshape.isExact())
    return false;
  auto actual = globalToIteration.compose(*actualProjection.get());
  auto expected = expectedProjection.get()->compose(*reshape.get());
  return actual.isExact() && expected.isExact() &&
         actual.get()->isEquivalentTo(*expected.get()).isProvenTrue();
}

bool proveLogicalAttentionMaps(
    const ScoreExpression &score, const Softmax &softmax,
    const Contraction &valueContraction, unsigned queryInput,
    unsigned probabilityInput, const LogicalAttentionMaps &maps,
    llvm::ArrayRef<int64_t> globalShape, mlir::AffineMap scoreMap,
    llvm::ArrayRef<unsigned> queryKeyAxes, uint64_t &work) {
  using analysis::IndexRelation;
  auto qk = score.contraction.operation;
  auto pv = valueContraction.operation;
  auto rowMax = softmax.rowMax;
  auto globalToOperand =
      [&](mlir::Value raw, mlir::Value selected,
          mlir::AffineMap map) -> std::optional<IndexRelation> {
    auto type = mlir::cast<mlir::RankedTensorType>(selected.getType());
    auto expected =
        IndexRelation::fromAffineMap(map, globalShape, type.getShape());
    auto toSelected = getValueRelation(raw, selected, work);
    if (!expected.isExact() || !toSelected)
      return std::nullopt;
    auto toRaw = toSelected->inverse();
    if (!toRaw.isExact())
      return std::nullopt;
    auto composed = expected.get()->compose(*toRaw.get());
    return composed.isExact() ? std::optional<IndexRelation>(*composed.get())
                              : std::nullopt;
  };
  auto query =
      globalToOperand(qk.getDpsInputs()[queryInput], maps.query, maps.maps[0]);
  auto key = globalToOperand(qk.getDpsInputs()[1 - queryInput], maps.key,
                             maps.maps[1]);
  auto value = globalToOperand(pv.getDpsInputs()[1 - probabilityInput],
                               maps.value, maps.maps[2]);
  auto probability = globalToOperand(pv.getDpsInputs()[probabilityInput],
                                     softmax.probability, scoreMap);
  if (!query || !key || !value || !probability)
    return false;
  auto qkIteration = getContractionIterationRelation(
      score.contraction, queryInput == 0 ? *query : *key,
      queryInput == 0 ? *key : *query, globalShape);
  auto pvIteration = getContractionIterationRelation(
      valueContraction, probabilityInput == 0 ? *probability : *value,
      probabilityInput == 0 ? *value : *probability, globalShape);
  if (!qkIteration || !pvIteration ||
      !proveReductionOrder(*qkIteration, score.contraction.operation,
                           globalShape, queryKeyAxes) ||
      !proveReductionOrder(*pvIteration, valueContraction.operation,
                           globalShape, maps.keyValueGlobalDimensions))
    return false;
  auto scoresType =
      mlir::cast<mlir::RankedTensorType>(softmax.scores.getType());
  auto expectedScores = IndexRelation::fromAffineMap(scoreMap, globalShape,
                                                     scoresType.getShape());
  auto scoresToRaw = getValueRelation(
      softmax.scores, score.contraction.operation->getResult(0), work);
  if (!expectedScores.isExact() || !scoresToRaw)
    return false;
  auto expectedQKOutput = expectedScores.get()->compose(*scoresToRaw);
  auto qkType = mlir::cast<mlir::RankedTensorType>(
      score.contraction.operation->getResult(0).getType());
  auto qkOutput = IndexRelation::fromAffineMap(
      score.contraction.maps[2], qk.getStaticLoopRanges(), qkType.getShape());
  if (!expectedQKOutput.isExact() || !qkOutput.isExact())
    return false;
  auto actualQKOutput = qkIteration->compose(*qkOutput.get());
  if (!actualQKOutput.isExact() ||
      !actualQKOutput.get()
           ->isEquivalentTo(*expectedQKOutput.get())
           .isProvenTrue())
    return false;
  auto pvType = mlir::cast<mlir::RankedTensorType>(
      valueContraction.operation->getResult(0).getType());
  auto pvOutput = IndexRelation::fromAffineMap(
      valueContraction.maps[2], pv.getStaticLoopRanges(), pvType.getShape());
  auto expectedOutput = IndexRelation::fromAffineMap(
      maps.maps.back(), globalShape, maps.outputType.getShape());
  auto outputReshape = IndexRelation::staticReshape(maps.outputType.getShape(),
                                                    pvType.getShape());
  if (!pvOutput.isExact() || !expectedOutput.isExact() ||
      !outputReshape.isExact())
    return false;
  auto actualPVOutput = pvIteration->compose(*pvOutput.get());
  auto expectedPVOutput = expectedOutput.get()->compose(*outputReshape.get());
  if (!actualPVOutput.isExact() || !expectedPVOutput.isExact() ||
      !actualPVOutput.get()
           ->isEquivalentTo(*expectedPVOutput.get())
           .isProvenTrue())
    return false;
  auto softmaxAccess = IndexRelation::fromAffineMap(
      softmax.scoreMap, rowMax.getStaticLoopRanges(), scoresType.getShape());
  if (!softmaxAccess.isExact())
    return false;
  auto inverse = softmaxAccess.get()->inverse();
  if (!inverse.isExact())
    return false;
  auto softmaxIteration = expectedScores.get()->compose(*inverse.get());
  return softmaxIteration.isExact() &&
         proveReductionOrder(*softmaxIteration.get(), softmax.rowMax,
                             globalShape, maps.keyValueGlobalDimensions);
}

std::optional<LogicalAxisAssignment> solveLogicalAttentionAxes(
    mlir::RankedTensorType queryType, mlir::RankedTensorType keyType,
    mlir::RankedTensorType valueType, mlir::RankedTensorType scoreType,
    llvm::ArrayRef<bool> scoreReductionAxes, uint64_t &work,
    llvm::function_ref<bool(const LogicalAxisAssignment &)> accept) {
  if (!queryType || !keyType || !valueType || !scoreType ||
      !queryType.hasStaticShape() || !keyType.hasStaticShape() ||
      !valueType.hasStaticShape() || !scoreType.hasStaticShape() ||
      scoreReductionAxes.size() != static_cast<size_t>(scoreType.getRank()))
    return std::nullopt;

  llvm::SmallVector<bool, 8> usedQuery(queryType.getRank(), false);
  llvm::SmallVector<bool, 8> usedKey(keyType.getRank(), false);
  llvm::SmallVector<bool, 8> usedValue(valueType.getRank(), false);
  LogicalAxisAssignment assignment;

  auto sameExtent = [](mlir::RankedTensorType left, unsigned leftAxis,
                       mlir::RankedTensorType right, unsigned rightAxis) {
    return left.getDimSize(leftAxis) > 0 &&
           left.getDimSize(leftAxis) == right.getDimSize(rightAxis);
  };

  std::function<bool(unsigned)> assignScoreAxis;
  assignScoreAxis = [&](unsigned scoreAxis) -> bool {
    if (++work > kMaximumAxisProofWork)
      return false;
    if (scoreAxis == static_cast<unsigned>(scoreType.getRank())) {
      llvm::SmallVector<unsigned, 4> remainingQuery;
      llvm::SmallVector<unsigned, 4> remainingKey;
      for (unsigned axis = 0; axis < static_cast<unsigned>(queryType.getRank());
           ++axis)
        if (!usedQuery[axis])
          remainingQuery.push_back(axis);
      for (unsigned axis = 0; axis < static_cast<unsigned>(keyType.getRank());
           ++axis)
        if (!usedKey[axis])
          remainingKey.push_back(axis);
      if (remainingQuery.size() != remainingKey.size() ||
          remainingQuery.empty())
        return false;

      std::function<bool(unsigned)> pairQueryKey;
      pairQueryKey = [&](unsigned index) -> bool {
        if (++work > kMaximumAxisProofWork)
          return false;
        if (index == remainingQuery.size()) {
          assignment.valueOutput.clear();
          for (unsigned axis = 0;
               axis < static_cast<unsigned>(valueType.getRank()); ++axis)
            if (!usedValue[axis])
              assignment.valueOutput.push_back(axis);
          return !assignment.query.empty() && !assignment.queryKey.empty() &&
                 !assignment.keyValue.empty() &&
                 !assignment.valueOutput.empty() && accept(assignment);
        }
        const unsigned queryAxis = remainingQuery[index];
        for (unsigned keyAxis : remainingKey) {
          if (usedKey[keyAxis] ||
              !sameExtent(queryType, queryAxis, keyType, keyAxis))
            continue;
          usedKey[keyAxis] = true;
          assignment.queryKey.push_back({queryAxis, keyAxis});
          if (pairQueryKey(index + 1))
            return true;
          assignment.queryKey.pop_back();
          usedKey[keyAxis] = false;
        }
        return false;
      };
      return pairQueryKey(0);
    }

    const int64_t extent = scoreType.getDimSize(scoreAxis);
    if (extent <= 0)
      return false;
    if (scoreReductionAxes[scoreAxis]) {
      for (unsigned keyAxis = 0;
           keyAxis < static_cast<unsigned>(keyType.getRank()); ++keyAxis) {
        if (usedKey[keyAxis] || keyType.getDimSize(keyAxis) != extent)
          continue;
        for (unsigned valueAxis = 0;
             valueAxis < static_cast<unsigned>(valueType.getRank());
             ++valueAxis) {
          if (usedValue[valueAxis] || valueType.getDimSize(valueAxis) != extent)
            continue;
          usedKey[keyAxis] = true;
          usedValue[valueAxis] = true;
          assignment.keyValue.push_back({keyAxis, valueAxis, scoreAxis});
          if (assignScoreAxis(scoreAxis + 1))
            return true;
          assignment.keyValue.pop_back();
          usedValue[valueAxis] = false;
          usedKey[keyAxis] = false;
        }
      }
      return false;
    }

    // A non-reduction score axis is either shared batch/head-like state or a
    // query/output-row coordinate. Try the more constrained shared case first.
    for (unsigned queryAxis = 0;
         queryAxis < static_cast<unsigned>(queryType.getRank()); ++queryAxis) {
      if (usedQuery[queryAxis] || queryType.getDimSize(queryAxis) != extent)
        continue;
      for (unsigned keyAxis = 0;
           keyAxis < static_cast<unsigned>(keyType.getRank()); ++keyAxis) {
        if (usedKey[keyAxis] || keyType.getDimSize(keyAxis) != extent)
          continue;
        for (unsigned valueAxis = 0;
             valueAxis < static_cast<unsigned>(valueType.getRank());
             ++valueAxis) {
          if (usedValue[valueAxis] || valueType.getDimSize(valueAxis) != extent)
            continue;
          usedQuery[queryAxis] = true;
          usedKey[keyAxis] = true;
          usedValue[valueAxis] = true;
          assignment.batch.push_back(
              {queryAxis, keyAxis, valueAxis, scoreAxis});
          if (assignScoreAxis(scoreAxis + 1))
            return true;
          assignment.batch.pop_back();
          usedValue[valueAxis] = false;
          usedKey[keyAxis] = false;
          usedQuery[queryAxis] = false;
        }
      }
    }
    for (unsigned queryAxis = 0;
         queryAxis < static_cast<unsigned>(queryType.getRank()); ++queryAxis) {
      if (usedQuery[queryAxis] || queryType.getDimSize(queryAxis) != extent)
        continue;
      usedQuery[queryAxis] = true;
      assignment.query.push_back({queryAxis, scoreAxis});
      if (assignScoreAxis(scoreAxis + 1))
        return true;
      assignment.query.pop_back();
      usedQuery[queryAxis] = false;
    }
    return false;
  };

  if (!assignScoreAxis(0) || assignment.query.empty() ||
      assignment.queryKey.empty() || assignment.keyValue.empty() ||
      assignment.valueOutput.empty())
    return std::nullopt;
  return assignment;
}

std::optional<LogicalAttentionMaps> buildLogicalAttentionMapsForValues(
    ScoreExpression &score, const Softmax &softmax,
    const Contraction &valueContraction, unsigned probabilityInput,
    unsigned queryInput, mlir::Value query, mlir::Value key, mlir::Value value,
    uint64_t &work) {
  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
  auto valueType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  auto scoreType =
      mlir::dyn_cast<mlir::RankedTensorType>(softmax.scores.getType());
  if (!queryType || !keyType || !valueType || !scoreType)
    return std::nullopt;

  llvm::SmallVector<bool, 8> scoreReductionAxes(scoreType.getRank(), false);
  mlir::linalg::GenericOp rowMax = softmax.rowMax;
  llvm::SmallVector<mlir::utils::IteratorType, 6> scoreIterators(
      rowMax.getIteratorTypesArray());
  if (softmax.scoreMap.getNumResults() !=
      static_cast<unsigned>(scoreType.getRank()))
    return std::nullopt;
  for (auto [axis, expression] :
       llvm::enumerate(softmax.scoreMap.getResults())) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= scoreIterators.size())
      return std::nullopt;
    scoreReductionAxes[axis] = scoreIterators[dimension.getPosition()] ==
                               mlir::utils::IteratorType::reduction;
  }

  auto makeCandidate = [&](const LogicalAxisAssignment &candidate)
      -> std::optional<LogicalAttentionMaps> {
    const auto *assignment = &candidate;
    const unsigned globalRank =
        assignment->batch.size() + assignment->query.size() +
        assignment->queryKey.size() + assignment->keyValue.size() +
        assignment->valueOutput.size();
    llvm::SmallVector<int64_t, 8> queryToGlobal(queryType.getRank(), -1);
    llvm::SmallVector<int64_t, 8> keyToGlobal(keyType.getRank(), -1);
    llvm::SmallVector<int64_t, 8> valueToGlobal(valueType.getRank(), -1);
    llvm::SmallVector<int64_t, 8> scoreToGlobal(scoreType.getRank(), -1);
    llvm::SmallVector<int64_t, 8> globalExtents;
    globalExtents.reserve(globalRank);
    llvm::SmallVector<unsigned, 4> outputGlobalDimensions;
    unsigned nextGlobal = 0;
    for (const auto &axes : assignment->batch) {
      queryToGlobal[axes[0]] = nextGlobal;
      keyToGlobal[axes[1]] = nextGlobal;
      valueToGlobal[axes[2]] = nextGlobal;
      scoreToGlobal[axes[3]] = nextGlobal;
      globalExtents.push_back(scoreType.getDimSize(axes[3]));
      outputGlobalDimensions.push_back(nextGlobal++);
    }
    for (const auto &axes : assignment->query) {
      queryToGlobal[axes[0]] = nextGlobal;
      scoreToGlobal[axes[1]] = nextGlobal;
      globalExtents.push_back(scoreType.getDimSize(axes[1]));
      outputGlobalDimensions.push_back(nextGlobal++);
    }
    llvm::SmallVector<unsigned, 2> queryKeyGlobalDimensions;
    for (const auto &axes : assignment->queryKey) {
      queryKeyGlobalDimensions.push_back(nextGlobal);
      queryToGlobal[axes[0]] = nextGlobal;
      keyToGlobal[axes[1]] = nextGlobal;
      globalExtents.push_back(queryType.getDimSize(axes[0]));
      ++nextGlobal;
    }
    llvm::SmallVector<unsigned, 2> keyValueGlobalDimensions;
    for (const auto &axes : assignment->keyValue) {
      keyToGlobal[axes[0]] = nextGlobal;
      valueToGlobal[axes[1]] = nextGlobal;
      scoreToGlobal[axes[2]] = nextGlobal;
      globalExtents.push_back(scoreType.getDimSize(axes[2]));
      keyValueGlobalDimensions.push_back(nextGlobal++);
    }
    for (unsigned axis : assignment->valueOutput) {
      valueToGlobal[axis] = nextGlobal;
      globalExtents.push_back(valueType.getDimSize(axis));
      outputGlobalDimensions.push_back(nextGlobal++);
    }
    if (nextGlobal != globalRank ||
        llvm::is_contained(queryToGlobal, int64_t{-1}) ||
        llvm::is_contained(keyToGlobal, int64_t{-1}) ||
        llvm::is_contained(valueToGlobal, int64_t{-1}) ||
        llvm::is_contained(scoreToGlobal, int64_t{-1}))
      return std::nullopt;

    auto makeOperandMap = [&](llvm::ArrayRef<int64_t> axisToGlobal) {
      llvm::SmallVector<mlir::AffineExpr, 4> expressions;
      for (int64_t global : axisToGlobal)
        expressions.push_back(
            mlir::getAffineDimExpr(global, query.getContext()));
      return mlir::AffineMap::get(globalRank, 0, expressions,
                                  query.getContext());
    };
    llvm::SmallVector<mlir::AffineMap, 6> maps{
        makeOperandMap(queryToGlobal), makeOperandMap(keyToGlobal),
        makeOperandMap(valueToGlobal),
        mlir::AffineMap::get(globalRank, 0, {}, query.getContext())};
    if (score.mask) {
      llvm::SmallVector<int64_t, 8> expressionToGlobal(
          score.scoreOutputMap.getNumDims(), -1);
      if (score.scoreOutputMap.getNumResults() !=
          static_cast<unsigned>(scoreType.getRank()))
        return std::nullopt;
      for (auto [axis, expression] :
           llvm::enumerate(score.scoreOutputMap.getResults())) {
        unsigned dimension =
            mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
        expressionToGlobal[dimension] = scoreToGlobal[axis];
      }
      std::optional<mlir::AffineMap> maskMap =
          translateMap(score.maskMap, expressionToGlobal, globalRank);
      if (!maskMap)
        return std::nullopt;
      maps.push_back(*maskMap);
    }
    llvm::SmallVector<mlir::AffineExpr, 4> outputExpressions;
    llvm::SmallVector<int64_t, 4> outputShape;
    for (unsigned global : outputGlobalDimensions) {
      outputExpressions.push_back(
          mlir::getAffineDimExpr(global, query.getContext()));
      outputShape.push_back(globalExtents[global]);
    }
    maps.push_back(mlir::AffineMap::get(globalRank, 0, outputExpressions,
                                        query.getContext()));
    auto storageElementType =
        mlir::cast<mlir::ShapedType>(
            valueContraction.operation->getResult(0).getType())
            .getElementType();
    LogicalAttentionMaps proposed{
        query,
        key,
        value,
        mlir::RankedTensorType::get(outputShape, storageElementType),
        std::move(maps),
        std::move(keyValueGlobalDimensions)};
    if (!proveLogicalAttentionMaps(score, softmax, valueContraction, queryInput,
                                   probabilityInput, proposed, globalExtents,
                                   makeOperandMap(scoreToGlobal),
                                   queryKeyGlobalDimensions, work))
      return std::nullopt;
    return proposed;
  };
  std::optional<LogicalAttentionMaps> result;
  solveLogicalAttentionAxes(queryType, keyType, valueType, scoreType,
                            scoreReductionAxes, work,
                            [&](const LogicalAxisAssignment &candidate) {
                              result = makeCandidate(candidate);
                              return result.has_value();
                            });
  return result;
}

std::optional<LogicalAttentionMaps>
buildLogicalAttentionMaps(ScoreExpression &score, const Softmax &softmax,
                          const Contraction &valueContraction,
                          unsigned probabilityInput) {
  uint64_t work = 0;
  auto candidates = [&](mlir::Value value) {
    llvm::SmallVector<mlir::Value, 4> values;
    llvm::DenseSet<mlir::Value> visited;
    while (value && visited.insert(value).second &&
           ++work <= kMaximumAxisProofWork) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape())
        break;
      values.push_back(value);
      mlir::Value source = getTransparentLayoutSource(value.getDefiningOp());
      auto sourceType =
          source ? mlir::dyn_cast<mlir::RankedTensorType>(source.getType())
                 : mlir::RankedTensorType{};
      if (!sourceType || !sourceType.hasStaticShape() ||
          sourceType.getElementType() != type.getElementType() ||
          sourceType.getNumElements() != type.getNumElements())
        break;
      value = source;
    }
    std::reverse(values.begin(), values.end());
    return values;
  };
  auto left = candidates(score.contraction.left);
  auto right = candidates(score.contraction.right);
  auto values = candidates(probabilityInput == 0 ? valueContraction.right
                                                 : valueContraction.left);
  for (unsigned queryInput : {0u, 1u})
    for (mlir::Value query : queryInput == 0 ? left : right)
      for (mlir::Value key : queryInput == 0 ? right : left)
        for (mlir::Value value : values) {
          if (++work > kMaximumAxisProofWork)
            return std::nullopt;
          auto result = buildLogicalAttentionMapsForValues(
              score, softmax, valueContraction, probabilityInput, queryInput,
              query, key, value, work);
          if (result)
            return result;
        }
  return std::nullopt;
}

bool isValueAncestor(mlir::Value ancestor, mlir::Value value) {
  llvm::SmallVector<mlir::Value, 16> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (current == ancestor)
      return true;
    if (!visited.insert(current).second)
      continue;
    if (mlir::Operation *definition = current.getDefiningOp())
      llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

bool hasClosedScalarScore(const AttentionMatch &match) {
  llvm::DenseSet<mlir::Value> visited;
  std::function<bool(mlir::Value)> check = [&](mlir::Value value) {
    if (value == match.rawScores || value == match.scale || value == match.mask)
      return true;
    if (!visited.insert(value).second)
      return true;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return false;
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(definition)) {
      if (!isAllParallel(generic) || generic->getNumResults() != 1 ||
          llvm::any_of(
              generic.getRegionOutputArgs(),
              [](mlir::BlockArgument arg) { return !arg.use_empty(); }) ||
          !llvm::all_of(generic.getDpsInputs(), check))
        return false;
      for (mlir::Operation &scalar : generic.getRegion().front()) {
        if (scalar.getNumRegions() != 0 || !mlir::isMemoryEffectFree(&scalar))
          return false;
        for (mlir::Value operand : scalar.getOperands()) {
          if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
            if (argument.getOwner() == &generic.getRegion().front())
              continue;
          } else if (operand.getDefiningOp()->getBlock() ==
                     &generic.getRegion().front()) {
            continue;
          }
          if (!check(operand))
            return false;
        }
      }
      return true;
    }
    if (mlir::isa<mlir::ShapedType>(value.getType())) {
      mlir::Value source = getTransparentSource(definition);
      return source && check(source);
    }
    return definition->getNumRegions() == 0 &&
           mlir::isMemoryEffectFree(definition) &&
           llvm::all_of(definition->getResultTypes(),
                        [](mlir::Type type) {
                          return mlir::isa<mlir::FloatType, mlir::IntegerType>(
                              type);
                        }) &&
           llvm::all_of(definition->getOperands(), check);
  };
  return check(match.adjustedScores);
}

std::optional<int64_t> getStaticIndex(mlir::OpFoldResult value) {
  return mlir::getConstantIntValue(value);
}

bool hasStaticSlice(mlir::tensor::InsertSliceOp insert,
                    llvm::ArrayRef<int64_t> offsets,
                    llvm::ArrayRef<int64_t> sizes) {
  if (insert.getMixedOffsets().size() != offsets.size() ||
      insert.getMixedSizes().size() != sizes.size() ||
      insert.getMixedStrides().size() != sizes.size())
    return false;
  for (auto [actual, expected] :
       llvm::zip_equal(insert.getMixedOffsets(), offsets)) {
    std::optional<int64_t> value = getStaticIndex(actual);
    if (!value || *value != expected)
      return false;
  }
  for (auto [actual, expected] :
       llvm::zip_equal(insert.getMixedSizes(), sizes)) {
    std::optional<int64_t> value = getStaticIndex(actual);
    if (!value || *value != expected)
      return false;
  }
  return llvm::all_of(insert.getMixedStrides(), [](mlir::OpFoldResult stride) {
    return mlir::getConstantIntValue(stride) == 1;
  });
}

std::optional<FunctionalCacheAppend>
matchFunctionalCacheAppend(mlir::Value returned) {
  auto append = returned.getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!append)
    return std::nullopt;
  auto prefix = append.getDest().getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!prefix)
    return std::nullopt;
  mlir::Value past = prefix.getSource();
  mlir::Value appended = append.getSource();
  auto pastType = mlir::dyn_cast<mlir::RankedTensorType>(past.getType());
  auto appendedType =
      mlir::dyn_cast<mlir::RankedTensorType>(appended.getType());
  auto updatedType = mlir::dyn_cast<mlir::RankedTensorType>(returned.getType());
  if (!pastType || !appendedType || !updatedType ||
      !pastType.hasStaticShape() || !appendedType.hasStaticShape() ||
      !updatedType.hasStaticShape() || pastType.getRank() == 0 ||
      pastType.getRank() != appendedType.getRank() ||
      pastType.getRank() != updatedType.getRank() ||
      pastType.getElementType() != appendedType.getElementType() ||
      pastType.getElementType() != updatedType.getElementType() ||
      !mlir::isa<mlir::BlockArgument>(past) || isValueAncestor(past, appended))
    return std::nullopt;

  std::optional<int64_t> appendDimension;
  for (int64_t dimension = 0; dimension < updatedType.getRank(); ++dimension) {
    const int64_t past = pastType.getDimSize(dimension);
    const int64_t appended = appendedType.getDimSize(dimension);
    const int64_t updated = updatedType.getDimSize(dimension);
    if (updated == past + appended && past != updated && appended != updated) {
      if (appendDimension)
        return std::nullopt;
      appendDimension = dimension;
    } else if (past != updated || appended != updated) {
      return std::nullopt;
    }
  }
  if (!appendDimension)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> prefixOffsets(updatedType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> appendOffsets(updatedType.getRank(), 0);
  appendOffsets[*appendDimension] = pastType.getDimSize(*appendDimension);
  if (!hasStaticSlice(prefix, prefixOffsets, pastType.getShape()) ||
      !hasStaticSlice(append, appendOffsets, appendedType.getShape()))
    return std::nullopt;
  mlir::Value initial = prefix.getDest();
  if (!initial.getDefiningOp<mlir::tensor::EmptyOp>() &&
      !mlir::isa<mlir::BlockArgument>(initial))
    return std::nullopt;
  return FunctionalCacheAppend{returned, *appendDimension,
                               pastType.getDimSize(*appendDimension),
                               appendedType.getDimSize(*appendDimension)};
}

std::optional<unsigned> traceValueAxisToAncestor(mlir::Value value,
                                                 unsigned valueAxis,
                                                 mlir::Value ancestor) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (value == ancestor)
      return valueAxis;
    if (auto cast = value.getDefiningOp<mlir::tensor::CastOp>()) {
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(cast.getSource().getType());
      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(cast.getResult().getType());
      if (!sourceType || !resultType ||
          sourceType.getRank() != resultType.getRank())
        return std::nullopt;
      value = cast.getSource();
      continue;
    }
    auto generic = value.getDefiningOp<mlir::linalg::GenericOp>();
    if (!matchBroadcastBody(generic))
      return std::nullopt;
    llvm::SmallVector<mlir::AffineMap, 2> maps(generic.getIndexingMapsArray());
    if (maps.size() != 2 || valueAxis >= maps[1].getNumResults())
      return std::nullopt;
    auto outputDimension =
        mlir::dyn_cast<mlir::AffineDimExpr>(maps[1].getResult(valueAxis));
    if (!outputDimension)
      return std::nullopt;
    std::optional<unsigned> sourceAxis;
    for (auto [axis, expression] : llvm::enumerate(maps[0].getResults())) {
      auto sourceDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (sourceDimension &&
          sourceDimension.getPosition() == outputDimension.getPosition()) {
        if (sourceAxis)
          return std::nullopt;
        sourceAxis = axis;
      }
    }
    if (!sourceAxis)
      return std::nullopt;
    valueAxis = *sourceAxis;
    value = generic.getDpsInputs().front();
  }
  return std::nullopt;
}

bool appendAxisMapsToGlobalDimension(const FunctionalCacheAppend &append,
                                     mlir::Value consumed,
                                     mlir::AffineMap consumedMap,
                                     unsigned globalDimension) {
  auto consumedType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumed.getType());
  if (!consumedType || consumedMap.getNumResults() !=
                           static_cast<unsigned>(consumedType.getRank()))
    return false;
  for (unsigned axis = 0; axis < consumedMap.getNumResults(); ++axis) {
    auto dimension =
        mlir::dyn_cast<mlir::AffineDimExpr>(consumedMap.getResult(axis));
    if (!dimension || dimension.getPosition() != globalDimension)
      continue;
    std::optional<unsigned> ancestorAxis =
        traceValueAxisToAncestor(consumed, axis, append.updated);
    if (ancestorAxis &&
        *ancestorAxis == static_cast<unsigned>(append.dimension))
      return true;
  }
  return false;
}

AttentionAlgorithm
classifyAlgorithm(mlir::func::FuncOp function, mlir::Value key,
                  mlir::Value value, mlir::AffineMap keyMap,
                  mlir::AffineMap valueMap,
                  llvm::ArrayRef<unsigned> keyValueGlobalDimensions) {
  if (!function.getBody().hasOneBlock())
    return AttentionAlgorithm::FlashAttention;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp)
    return AttentionAlgorithm::FlashAttention;

  std::optional<FunctionalCacheAppend> keyAppend;
  std::optional<FunctionalCacheAppend> valueAppend;
  for (mlir::Value returned : returnOp.getOperands()) {
    std::optional<FunctionalCacheAppend> append =
        matchFunctionalCacheAppend(returned);
    if (!append)
      continue;
    if (isValueAncestor(append->updated, key)) {
      if (keyAppend)
        return AttentionAlgorithm::FlashAttention;
      keyAppend = append;
    }
    if (isValueAncestor(append->updated, value)) {
      if (valueAppend)
        return AttentionAlgorithm::FlashAttention;
      valueAppend = append;
    }
  }
  if (!keyAppend || !valueAppend)
    return AttentionAlgorithm::FlashAttention;
  if (keyAppend->updated == valueAppend->updated)
    return AttentionAlgorithm::FlashAttention;
  if (keyAppend->pastExtent != valueAppend->pastExtent ||
      keyAppend->appendedExtent != valueAppend->appendedExtent)
    return AttentionAlgorithm::FlashAttention;

  const bool sharedK2 =
      llvm::any_of(keyValueGlobalDimensions, [&](unsigned globalDimension) {
        return appendAxisMapsToGlobalDimension(*keyAppend, key, keyMap,
                                               globalDimension) &&
               appendAxisMapsToGlobalDimension(*valueAppend, value, valueMap,
                                               globalDimension);
      });
  if (!sharedK2)
    return AttentionAlgorithm::FlashAttention;
  auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
  return keyType && keyType.hasStaticShape() &&
                 keyType.getDimSize(keyAppend->dimension) >= 2
             ? AttentionAlgorithm::FlashDecoding
             : AttentionAlgorithm::FlashAttention;
}

std::optional<AttentionMatch>
matchAttentionRoot(mlir::func::FuncOp function,
                   mlir::linalg::LinalgOp valueContraction) {
  if (!valueContraction || valueContraction->getNumResults() != 1 ||
      valueContraction.getNumDpsInputs() != 2)
    return std::nullopt;
  bool observable = false;
  function.walk([&](mlir::func::ReturnOp returnOp) {
    observable |= llvm::any_of(returnOp.getOperands(), [&](mlir::Value value) {
      return isValueAncestor(valueContraction->getResult(0), value);
    });
  });
  if (!observable)
    return std::nullopt;
  std::optional<Contraction> matchedValue =
      matchContraction(valueContraction->getResult(0));
  if (!matchedValue) {
    LLVM_DEBUG(llvm::dbgs() << "reject value contraction "
                            << valueContraction->getName() << " result="
                            << (valueContraction->getNumResults()
                                    ? valueContraction->getResult(0).getType()
                                    : mlir::Type{})
                            << "\n");
    return std::nullopt;
  }

  for (unsigned probabilityInput : {0u, 1u}) {
    mlir::Value logicalProbability =
        valueContraction.getDpsInputs()[probabilityInput];
    // Rounding normalized probabilities before PV is not interchangeable
    // with rounding the online algorithm's unnormalized exponentials.
    // Only layout changes are transparent on this side of softmax.
    while (mlir::Value source =
               getTransparentLayoutSource(logicalProbability.getDefiningOp()))
      logicalProbability = source;
    std::optional<Softmax> softmax = matchSoftmax(logicalProbability);
    if (!softmax) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reject softmax input " << probabilityInput << " at "
                 << (logicalProbability.getDefiningOp()
                         ? logicalProbability.getDefiningOp()->getName()
                         : mlir::OperationName(
                               "builtin.unrealized_conversion_cast",
                               function.getContext()))
                 << "\n");
      continue;
    }
    std::optional<ScoreExpression> score =
        matchScoreExpression(softmax->scores);
    if (!score) {
      LLVM_DEBUG(llvm::dbgs() << "reject score expression\n");
      continue;
    }
    auto scoreType =
        mlir::dyn_cast<mlir::RankedTensorType>(softmax->scores.getType());
    auto probabilityType =
        mlir::dyn_cast<mlir::RankedTensorType>(logicalProbability.getType());
    if (!scoreType || !probabilityType || scoreType != probabilityType) {
      LLVM_DEBUG(llvm::dbgs() << "reject logical score/probability type\n");
      continue;
    }

    AttentionMatch match;
    match.scale = score->scale;
    match.mask = score->mask;
    match.rawScores = score->contraction.operation->getResult(0);
    match.adjustedScores = softmax->scores;
    std::optional<LogicalAttentionMaps> logicalMaps = buildLogicalAttentionMaps(
        *score, *softmax, *matchedValue, probabilityInput);
    if (!logicalMaps) {
      LLVM_DEBUG(llvm::dbgs() << "reject logical attention maps\n");
      continue;
    }
    match.query = logicalMaps->query;
    match.key = logicalMaps->key;
    match.value = logicalMaps->value;
    match.outputType = logicalMaps->outputType;
    match.indexingMaps = std::move(logicalMaps->maps);
    auto storageType = match.outputType.getElementType();
    if (probabilityType.getElementType() != storageType) {
      LLVM_DEBUG(llvm::dbgs() << "reject probability precision boundary\n");
      continue;
    }
    if (mlir::cast<mlir::ShapedType>(match.rawScores.getType())
                .getElementType() != storageType ||
        mlir::cast<mlir::ShapedType>(match.query.getType()).getElementType() !=
            storageType ||
        mlir::cast<mlir::ShapedType>(match.key.getType()).getElementType() !=
            storageType ||
        mlir::cast<mlir::ShapedType>(match.value.getType()).getElementType() !=
            storageType ||
        !hasClosedScalarScore(match))
      continue;

    mlir::Value observable = valueContraction->getResult(0);
    auto observableType =
        mlir::dyn_cast<mlir::RankedTensorType>(observable.getType());
    if (!observableType || !observableType.hasStaticShape() ||
        observableType.getElementType() != match.outputType.getElementType() ||
        observableType.getNumElements() != match.outputType.getNumElements()) {
      LLVM_DEBUG(llvm::dbgs() << "reject observable attention result type\n");
      continue;
    }
    if (observableType != match.outputType &&
        !mlir::getReassociationIndicesForReshape(match.outputType,
                                                 observableType)) {
      LLVM_DEBUG(llvm::dbgs() << "reject observable result reshape\n");
      continue;
    }
    match.root = observable.getDefiningOp();
    if (!match.root || match.root->getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "reject observable root\n");
      continue;
    }

    match.algorithm = classifyAlgorithm(
        function, match.key, match.value, match.indexingMaps[1],
        match.indexingMaps[2], logicalMaps->keyValueGlobalDimensions);
    return match;
  }
  return std::nullopt;
}

} // namespace

llvm::SmallVector<AttentionMatch, 4>
collectAttentionMatches(mlir::func::FuncOp function) {
  llvm::SmallVector<AttentionMatch, 4> matches;
  function.walk([&](mlir::linalg::LinalgOp operation) {
    if (mlir::isa<LinalgExtAttentionOp>(operation.getOperation()))
      return;
    std::optional<AttentionMatch> match =
        matchAttentionRoot(function, operation);
    if (match)
      matches.push_back(std::move(*match));
  });
  return matches;
}

mlir::LogicalResult materializeAttentionScoreRegion(mlir::OpBuilder &builder,
                                                    const AttentionMatch &match,
                                                    mlir::Region &region,
                                                    mlir::Type scaleType) {
  if (!region.empty() || !hasClosedScalarScore(match))
    return mlir::failure();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block *block = builder.createBlock(&region);
  mlir::Location location = match.root->getLoc();
  auto dotType =
      mlir::cast<mlir::ShapedType>(match.rawScores.getType()).getElementType();
  mlir::IRMapping mapping;
  mapping.map(match.rawScores, block->addArgument(dotType, location));
  auto scaleArgument = block->addArgument(scaleType, location);
  if (match.scale)
    mapping.map(match.scale, scaleArgument);
  if (match.mask) {
    auto maskType =
        mlir::cast<mlir::ShapedType>(match.mask.getType()).getElementType();
    mapping.map(match.mask, block->addArgument(maskType, location));
  }
  std::function<mlir::Value(mlir::Value)> clone = [&](mlir::Value value) {
    if (auto mapped = mapping.lookupOrNull(value))
      return mapped;
    auto *definition = value.getDefiningOp();
    if (!definition)
      return mlir::Value{};
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(definition)) {
      for (auto [argument, input] : llvm::zip_equal(
               generic.getRegionInputArgs(), generic.getDpsInputs())) {
        mlir::Value scalar = clone(input);
        if (!scalar)
          return mlir::Value{};
        mapping.map(argument, scalar);
      }
      auto yield = mlir::cast<mlir::linalg::YieldOp>(
          generic.getRegion().front().getTerminator());
      mlir::Value result = clone(yield.getValues().front());
      if (result)
        mapping.map(value, result);
      return result;
    }
    if (mlir::isa<mlir::ShapedType>(value.getType())) {
      mlir::Value source = getTransparentSource(definition);
      mlir::Value result = source ? clone(source) : mlir::Value{};
      if (result)
        mapping.map(value, result);
      return result;
    }
    for (mlir::Value operand : definition->getOperands())
      if (!clone(operand))
        return mlir::Value{};
    auto *cloned = builder.clone(*definition, mapping);
    for (auto [source, result] :
         llvm::zip_equal(definition->getResults(), cloned->getResults()))
      mapping.map(source, result);
    return mapping.lookup(value);
  };
  mlir::Value score = clone(match.adjustedScores);
  if (!score)
    return mlir::failure();
  builder.create<LinalgExtAttentionYieldOp>(location, score);
  return mlir::success();
}

} // namespace wafer::attention_normalization
