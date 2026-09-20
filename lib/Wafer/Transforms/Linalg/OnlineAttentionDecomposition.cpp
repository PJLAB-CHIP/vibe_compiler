//===- OnlineAttentionDecomposition.cpp - Lower current online state ---===//

#include "OnlineAttentionDecomposition.h"

#include "AttentionMath.h"
#include "AttentionVisibility.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Linalg/Pipelines.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace wafer {

#define GEN_PASS_DEF_DECOMPOSEONLINEATTENTIONOPSPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

template <typename T>
mlir::FailureOr<T> fail(OnlineAttentionDecompositionFailure *failure,
                        OnlineAttentionDecompositionFailureKind kind,
                        llvm::StringRef detail) {
  if (failure) {
    failure->kind = kind;
    failure->detail = detail.str();
  }
  return mlir::failure();
}

struct DecompositionDescriptor {
  LinalgExtOnlineAttentionOp operation;
  mlir::AffineMap scoreMap;
  llvm::SmallVector<int64_t, 5> scoreShape;
  uint64_t scoreElements = 0;
  struct PositionBias {
    int64_t queryExtent;
    int64_t keyExtent;
    AttentionPositionRange causal;
    AttentionPositionRange valid;
  };
  std::optional<PositionBias> bias;
};

struct DecomposedState {
  mlir::Value accumulator;
  mlir::Value maximum;
  mlir::Value sum;
};

struct DecomposeOnlineAttentionOpsPass final
    : impl::DecomposeOnlineAttentionOpsPassBase<
          DecomposeOnlineAttentionOpsPass> {
  using impl::DecomposeOnlineAttentionOpsPassBase<
      DecomposeOnlineAttentionOpsPass>::DecomposeOnlineAttentionOpsPassBase;

  void runOnOperation() final {
    StructuredMaterializationRelations relations;
    OnlineAttentionDecompositionFailure failure;
    if (mlir::failed(
            decomposeOnlineAttention(getOperation(), relations, &failure))) {
      getOperation()->emitError(failure.detail);
      signalPassFailure();
    }
  }
};

llvm::SmallBitVector getMapDimensions(mlir::AffineMap map) {
  llvm::SmallBitVector dimensions(map.getNumDims(), false);
  for (mlir::AffineExpr expression : map.getResults())
    dimensions.set(mlir::cast<mlir::AffineDimExpr>(expression).getPosition());
  return dimensions;
}

mlir::FailureOr<DecompositionDescriptor>
buildDescriptor(LinalgExtOnlineAttentionOp operation) {
  if (!operation || !operation->getParentOfType<TileRegionOp>() ||
      mlir::failed(mlir::verify(operation)))
    return mlir::failure();
  mlir::FailureOr<AttentionIterationRoles> roles =
      operation.getIterationRoles();
  if (mlir::failed(roles))
    return mlir::failure();
  llvm::SmallVector<int64_t> extents = operation.getStaticLoopRanges();
  if (extents.size() !=
          static_cast<size_t>(operation.getIterationDomainRank()) ||
      llvm::any_of(extents, [](int64_t extent) {
        return extent <= 0 || mlir::ShapedType::isDynamic(extent);
      }))
    return mlir::failure();
  for (mlir::Value value : operation->getOperands()) {
    auto shaped = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (shaped && !shaped.hasStaticShape())
      return mlir::failure();
  }
  // Positions remain integer/index values, including template selection.
  for (auto position : operation.getPositions()) {
    using Bounds = mlir::ValueBoundsConstraintSet;
    auto lower =
        Bounds::computeConstantBound(mlir::presburger::BoundType::LB,
                                     Bounds::Variable(position), nullptr, true);
    if (mlir::failed(lower) || *lower < 0)
      return mlir::failure();
  }

  const unsigned rank = operation.getIterationDomainRank();
  llvm::SmallBitVector scoreDimensions(rank, false);
  for (unsigned dimension : roles->batch)
    scoreDimensions.set(dimension);
  for (unsigned dimension : roles->query)
    scoreDimensions.set(dimension);
  for (unsigned dimension : roles->keyValueReduction)
    scoreDimensions.set(dimension);
  llvm::SmallVector<mlir::AffineExpr, 5> scoreExpressions;
  llvm::SmallVector<int64_t, 5> scoreShape;
  uint64_t scoreElements = 1;
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    if (!scoreDimensions.test(dimension))
      continue;
    scoreExpressions.push_back(
        mlir::getAffineDimExpr(dimension, operation.getContext()));
    scoreShape.push_back(extents[dimension]);
    if (static_cast<uint64_t>(extents[dimension]) >
        std::numeric_limits<uint64_t>::max() / scoreElements)
      return mlir::failure();
    scoreElements *= static_cast<uint64_t>(extents[dimension]);
  }
  mlir::AffineMap scoreMap =
      mlir::AffineMap::get(rank, 0, scoreExpressions, operation.getContext());
  llvm::SmallBitVector expectedQueryKey =
      getMapDimensions(operation.getQueryMap());
  expectedQueryKey |= getMapDimensions(operation.getKeyMap());
  for (unsigned dimension : roles->queryKeyReduction)
    expectedQueryKey.reset(dimension);
  if (scoreShape.empty() || expectedQueryKey != scoreDimensions)
    return mlir::failure();
  DecompositionDescriptor descriptor{operation, scoreMap, std::move(scoreShape),
                                     scoreElements, std::nullopt};
  if (!operation.getPositions().empty()) {
    auto map = operation.getPositionMapAttr().getValue();
    int64_t queryExtent =
        extents[mlir::cast<mlir::AffineDimExpr>(map.getResult(0))
                    .getPosition()];
    int64_t keyExtent =
        extents[mlir::cast<mlir::AffineDimExpr>(map.getResult(1))
                    .getPosition()];
    if (queryExtent > std::numeric_limits<int64_t>::max() - keyExtent)
      return mlir::failure();
    auto positions = operation.getPositions();
    AttentionPositionRange causal{keyExtent - 1, keyExtent - 1, 1};
    if (operation.getCausal())
      causal = getAttentionPositionRange(positions[0], positions[1], operation,
                                         -queryExtent, keyExtent - 1);
    auto valid = getAttentionPositionRange(positions[2], positions[1],
                                           operation, 0, keyExtent);
    int64_t templateElements;
    if (llvm::MulOverflow(causal.size(), valid.size(), templateElements) ||
        llvm::MulOverflow(templateElements, queryExtent, templateElements) ||
        llvm::MulOverflow(templateElements, keyExtent, templateElements))
      return mlir::failure();
    if (causal.first < keyExtent - 1 || valid.first < keyExtent)
      descriptor.bias = DecompositionDescriptor::PositionBias{
          queryExtent, keyExtent, causal, valid};
  }
  return descriptor;
}

llvm::SmallVector<mlir::utils::IteratorType, 6>
getReductionIteratorTypes(mlir::AffineMap outputMap) {
  llvm::SmallVector<mlir::utils::IteratorType, 6> iteratorTypes(
      outputMap.getNumDims(), mlir::utils::IteratorType::reduction);
  for (mlir::AffineExpr expression : outputMap.getResults())
    iteratorTypes[mlir::cast<mlir::AffineDimExpr>(expression).getPosition()] =
        mlir::utils::IteratorType::parallel;
  return iteratorTypes;
}

llvm::SmallVector<mlir::utils::IteratorType, 6>
getParallelIteratorTypes(unsigned rank) {
  return llvm::SmallVector<mlir::utils::IteratorType, 6>(
      rank, mlir::utils::IteratorType::parallel);
}

mlir::Value createZeroTensor(mlir::Location location,
                             mlir::RankedTensorType type,
                             mlir::OpBuilder &builder) {
  mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
      location, type.getShape(), type.getElementType());
  mlir::Value zero =
      mlir::arith::getIdentityValue(mlir::arith::AtomicRMWKind::addf,
                                    type.getElementType(), builder, location);
  return builder.create<mlir::linalg::FillOp>(location, zero, empty)
      .getResult(0);
}

mlir::Value createQK(const DecompositionDescriptor &descriptor,
                     mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  mlir::Location location = operation.getLoc();
  auto scoreType = mlir::RankedTensorType::get(
      descriptor.scoreShape,
      operation.getScoreRegion().front().getArgument(0).getType());
  mlir::Value score = createZeroTensor(location, scoreType, builder);
  llvm::SmallVector<mlir::AffineMap, 3> maps = mlir::compressUnusedDims(
      {operation.getQueryMap(), operation.getKeyMap(), descriptor.scoreMap});
  auto contraction = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{scoreType},
      mlir::ValueRange{operation.getQuery(), operation.getKey()},
      mlir::ValueRange{score}, maps, getReductionIteratorTypes(maps.back()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value lhs = compiler::detail::castAttentionFloatScalar(
            arguments[0], arguments[2].getType(), nestedBuilder,
            nestedLocation);
        mlir::Value rhs = compiler::detail::castAttentionFloatScalar(
            arguments[1], arguments[2].getType(), nestedBuilder,
            nestedLocation);
        mlir::Value product =
            nestedBuilder.create<mlir::arith::MulFOp>(nestedLocation, lhs, rhs);
        mlir::Value result = nestedBuilder.create<mlir::arith::AddFOp>(
            nestedLocation, product, arguments[2]);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, result);
      });
  return contraction.getResult(0);
}

mlir::Value applyScoreRegion(const DecompositionDescriptor &descriptor,
                             mlir::Value rawScores, mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  llvm::SmallVector<mlir::Value, 3> inputs{rawScores, operation.getScale()};
  llvm::SmallVector<mlir::AffineMap, 4> maps{descriptor.scoreMap,
                                             operation.getScaleMap()};
  if (operation.getMask()) {
    inputs.push_back(operation.getMask());
    maps.push_back(*operation.getMaskMap());
  }
  const unsigned scoreArgumentCount = inputs.size();
  if (descriptor.bias) {
    const auto &bias = *descriptor.bias;
    auto loc = operation.getLoc();
    auto scalarType = mlir::cast<mlir::FloatType>(operation.getScoreType());
    auto zero = builder.getFloatAttr(scalarType, 0.0).getValue();
    auto negativeInfinity =
        builder
            .getFloatAttr(scalarType, -std::numeric_limits<double>::infinity())
            .getValue();
    llvm::SmallVector<llvm::APFloat> values;
    int64_t patterns = bias.causal.size() * bias.valid.size();
    values.reserve(patterns * bias.queryExtent * bias.keyExtent);
    for (int64_t c = 0; c < bias.causal.size(); ++c) {
      int64_t diagonal = bias.causal.first + c * bias.causal.step;
      for (int64_t v = 0; v < bias.valid.size(); ++v) {
        int64_t valid = bias.valid.first + v * bias.valid.step;
        for (int64_t q = 0; q < bias.queryExtent; ++q)
          for (int64_t k = 0; k < bias.keyExtent; ++k)
            values.push_back(k - q > diagonal || k >= valid ? negativeInfinity
                                                            : zero);
      }
    }
    llvm::SmallVector<int64_t> shape{bias.queryExtent, bias.keyExtent};
    if (patterns > 1)
      shape.insert(shape.begin(), patterns);
    auto templateType = mlir::RankedTensorType::get(shape, scalarType);
    mlir::Value positionBias = builder.create<mlir::arith::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(templateType, values));
    if (patterns > 1) {
      // ProgramData owns the template collection. The only runtime tensor
      // demand is one actual BQ x BK slice; selection is scalar index math.
      auto index = [&](AttentionPositionRange range, mlir::Value lhs) {
        if (range.size() == 1)
          return mlir::Value(
              builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
        auto first =
            builder.create<mlir::arith::ConstantIndexOp>(loc, range.first);
        auto last =
            builder.create<mlir::arith::ConstantIndexOp>(loc, range.last);
        auto step =
            builder.create<mlir::arith::ConstantIndexOp>(loc, range.step);
        mlir::Value relative = builder.createOrFold<mlir::arith::SubIOp>(
            loc, lhs, operation.getPositions()[1]);
        relative =
            builder.createOrFold<mlir::arith::MaxSIOp>(loc, relative, first);
        relative =
            builder.createOrFold<mlir::arith::MinSIOp>(loc, relative, last);
        relative =
            builder.createOrFold<mlir::arith::SubIOp>(loc, relative, first);
        return builder.createOrFold<mlir::arith::DivUIOp>(loc, relative, step);
      };
      auto causalIndex = index(bias.causal, operation.getPositions()[0]);
      auto validIndex = index(bias.valid, operation.getPositions()[2]);
      auto validCount =
          builder.create<mlir::arith::ConstantIndexOp>(loc, bias.valid.size());
      auto base = builder.createOrFold<mlir::arith::MulIOp>(loc, causalIndex,
                                                            validCount);
      auto selected =
          builder.createOrFold<mlir::arith::AddIOp>(loc, base, validIndex);
      auto resultType = mlir::RankedTensorType::get(
          {bias.queryExtent, bias.keyExtent}, scalarType);
      positionBias = builder.create<mlir::tensor::ExtractSliceOp>(
          loc, resultType, positionBias,
          llvm::ArrayRef<mlir::OpFoldResult>{selected, builder.getIndexAttr(0),
                                             builder.getIndexAttr(0)},
          llvm::ArrayRef<mlir::OpFoldResult>{
              builder.getIndexAttr(1), builder.getIndexAttr(bias.queryExtent),
              builder.getIndexAttr(bias.keyExtent)},
          llvm::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(1),
                                             builder.getIndexAttr(1),
                                             builder.getIndexAttr(1)});
    }
    inputs.push_back(positionBias);
    maps.push_back(operation.getPositionMapAttr().getValue());
  }
  maps.push_back(descriptor.scoreMap);
  maps = mlir::compressUnusedDims(maps);
  auto type = mlir::RankedTensorType::get(descriptor.scoreShape,
                                          operation.getScoreType());
  mlir::Value destination = rawScores;
  if (rawScores.getType() != type)
    destination = builder.create<mlir::tensor::EmptyOp>(
        operation.getLoc(), type.getShape(), type.getElementType());
  return builder
      .create<mlir::linalg::GenericOp>(
          operation.getLoc(), mlir::TypeRange{type}, inputs,
          mlir::ValueRange{destination}, maps,
          getParallelIteratorTypes(maps.back().getNumDims()),
          [&](mlir::OpBuilder &nested, mlir::Location location,
              mlir::ValueRange arguments) {
            mlir::IRMapping mapping;
            auto &body = operation.getScoreRegion().front();
            for (auto [source, target] :
                 llvm::zip_equal(body.getArguments(),
                                 arguments.take_front(scoreArgumentCount)))
              mapping.map(source, target);
            for (mlir::Operation &scalar : body.without_terminator())
              nested.clone(scalar, mapping);
            auto yield =
                mlir::cast<LinalgExtAttentionYieldOp>(body.getTerminator());
            mlir::Value score = mapping.lookup(yield.getValue());
            if (descriptor.bias)
              score = nested.create<mlir::arith::AddFOp>(
                  location, score, arguments[scoreArgumentCount]);
            nested.create<mlir::linalg::YieldOp>(location, score);
          })
      .getResult(0);
}

template <typename CombineOp>
mlir::Value createReduction(mlir::Location location, mlir::Value input,
                            mlir::Value init, mlir::AffineMap inputMap,
                            mlir::AffineMap outputMap,
                            mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::AffineMap, 2> maps =
      mlir::compressUnusedDims({inputMap, outputMap});
  auto reduction = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{init.getType()}, mlir::ValueRange{input},
      mlir::ValueRange{init}, maps, getReductionIteratorTypes(maps.back()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value value = compiler::detail::castAttentionFloatScalar(
            arguments[0], arguments[1].getType(), nestedBuilder,
            nestedLocation);
        mlir::Value result = nestedBuilder.create<CombineOp>(
            nestedLocation, value, arguments[1]);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, result);
      });
  return reduction.getResult(0);
}

mlir::Value createNorm(const DecompositionDescriptor &descriptor,
                       mlir::Value newMaximum, mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  llvm::SmallVector<mlir::AffineMap, 3> maps = mlir::compressUnusedDims(
      {operation.getMaximumMap(), operation.getMaximumMap(),
       operation.getMaximumMap()});
  auto norm = builder.create<mlir::linalg::GenericOp>(
      operation.getLoc(), mlir::TypeRange{operation.getMaximum().getType()},
      mlir::ValueRange{operation.getMaximum(), newMaximum},
      mlir::ValueRange{operation.getMaximum()}, maps,
      getParallelIteratorTypes(maps.front().getNumDims()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location location,
          mlir::ValueRange arguments) {
        mlir::Value result = compiler::detail::computeAttentionExponential(
            arguments[0], arguments[1], nestedBuilder, location);
        nestedBuilder.create<mlir::linalg::YieldOp>(location, result);
      });
  return norm.getResult(0);
}

mlir::Value multiplyState(mlir::Location location, mlir::Value state,
                          mlir::AffineMap stateMap, mlir::Value norm,
                          mlir::AffineMap normMap, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::AffineMap, 3> maps =
      mlir::compressUnusedDims({stateMap, normMap, stateMap});
  auto scaled = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{state.getType()}, mlir::ValueRange{state, norm},
      mlir::ValueRange{state}, maps,
      getParallelIteratorTypes(maps.front().getNumDims()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value scale = compiler::detail::castAttentionFloatScalar(
            arguments[1], arguments[0].getType(), nestedBuilder,
            nestedLocation);
        mlir::Value result = nestedBuilder.create<mlir::arith::MulFOp>(
            nestedLocation, arguments[0], scale);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, result);
      });
  return scaled.getResult(0);
}

mlir::Value createProbability(const DecompositionDescriptor &descriptor,
                              mlir::Value score, mlir::Value newMaximum,
                              mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  llvm::SmallVector<mlir::AffineMap, 3> maps = mlir::compressUnusedDims(
      {descriptor.scoreMap, operation.getMaximumMap(), descriptor.scoreMap});
  auto probability = builder.create<mlir::linalg::GenericOp>(
      operation.getLoc(), mlir::TypeRange{score.getType()},
      mlir::ValueRange{score, newMaximum}, mlir::ValueRange{score}, maps,
      getParallelIteratorTypes(maps.front().getNumDims()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location location,
          mlir::ValueRange arguments) {
        mlir::Value result = compiler::detail::computeAttentionExponential(
            arguments[0], arguments[1], nestedBuilder, location);
        nestedBuilder.create<mlir::linalg::YieldOp>(location, result);
      });
  return probability.getResult(0);
}

mlir::Value createPV(const DecompositionDescriptor &descriptor,
                     mlir::Value probability, mlir::Value scaledAccumulator,
                     mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  auto probabilityType =
      mlir::cast<mlir::RankedTensorType>(probability.getType());
  auto storageType = operation.getValue().getType().getElementType();
  if (probabilityType.getElementType() != storageType) {
    auto narrowType = probabilityType.clone(storageType);
    mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
        operation.getLoc(), narrowType.getShape(), storageType);
    auto identity = builder.getMultiDimIdentityMap(probabilityType.getRank());
    probability =
        builder
            .create<mlir::linalg::GenericOp>(
                operation.getLoc(), mlir::TypeRange{narrowType},
                mlir::ValueRange{probability}, mlir::ValueRange{empty},
                llvm::ArrayRef<mlir::AffineMap>{identity, identity},
                getParallelIteratorTypes(probabilityType.getRank()),
                [&](mlir::OpBuilder &nested, mlir::Location loc,
                    mlir::ValueRange args) {
                  mlir::Value value =
                      compiler::detail::castAttentionFloatScalar(
                          args[0], storageType, nested, loc);
                  nested.create<mlir::linalg::YieldOp>(loc, value);
                })
            .getResult(0);
  }
  llvm::SmallVector<mlir::AffineMap, 3> maps =
      mlir::compressUnusedDims({descriptor.scoreMap, operation.getValueMap(),
                                operation.getAccumulatorMap()});
  auto contraction = builder.create<mlir::linalg::GenericOp>(
      operation.getLoc(), mlir::TypeRange{scaledAccumulator.getType()},
      mlir::ValueRange{probability, operation.getValue()},
      mlir::ValueRange{scaledAccumulator}, maps,
      getReductionIteratorTypes(maps.back()),
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location location,
          mlir::ValueRange arguments) {
        mlir::Value probabilityValue =
            compiler::detail::castAttentionFloatScalar(
                arguments[0], arguments[2].getType(), nestedBuilder, location);
        mlir::Value value = compiler::detail::castAttentionFloatScalar(
            arguments[1], arguments[2].getType(), nestedBuilder, location);
        mlir::Value product = nestedBuilder.create<mlir::arith::MulFOp>(
            location, probabilityValue, value);
        mlir::Value result = nestedBuilder.create<mlir::arith::AddFOp>(
            location, product, arguments[2]);
        nestedBuilder.create<mlir::linalg::YieldOp>(location, result);
      });
  return contraction.getResult(0);
}

DecomposedState decomposeOne(const DecompositionDescriptor &descriptor,
                             mlir::OpBuilder &builder) {
  LinalgExtOnlineAttentionOp operation = descriptor.operation;
  mlir::Value score = createQK(descriptor, builder);
  score = applyScoreRegion(descriptor, score, builder);
  mlir::Value newMaximum = createReduction<mlir::arith::MaximumFOp>(
      operation.getLoc(), score, operation.getMaximum(), descriptor.scoreMap,
      operation.getMaximumMap(), builder);
  mlir::Value norm = createNorm(descriptor, newMaximum, builder);
  mlir::Value normalizedOldSum = multiplyState(
      operation.getLoc(), operation.getSum(), operation.getSumMap(), norm,
      operation.getMaximumMap(), builder);
  mlir::Value probability =
      createProbability(descriptor, score, newMaximum, builder);
  mlir::Value newSum = createReduction<mlir::arith::AddFOp>(
      operation.getLoc(), probability, normalizedOldSum, descriptor.scoreMap,
      operation.getSumMap(), builder);
  mlir::Value scaledAccumulator = multiplyState(
      operation.getLoc(), operation.getAccumulator(),
      operation.getAccumulatorMap(), norm, operation.getMaximumMap(), builder);
  mlir::Value newAccumulator =
      createPV(descriptor, probability, scaledAccumulator, builder);
  return {newAccumulator, newMaximum, newSum};
}

} // namespace

mlir::LogicalResult
verifyOnlineAttentionDecompositionComplete(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<LinalgExtAttentionOp, LinalgExtOnlineAttentionOp>(
            operation)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  return illegal->emitOpError(
      "must be decomposed before layout and bufferization");
}

void buildDecomposeOnlineAttentionPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createDecomposeOnlineAttentionOpsPass());
}

mlir::FailureOr<OnlineAttentionDecompositionStatistics>
decomposeOnlineAttention(mlir::ModuleOp module,
                         StructuredMaterializationRelations &relations,
                         OnlineAttentionDecompositionFailure *failure) {
  if (failure)
    *failure = {};
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<OnlineAttentionDecompositionStatistics>(
        failure, OnlineAttentionDecompositionFailureKind::BrokenContract,
        "online-attention decomposition requires verifier-valid current IR "
        "and live relations");
  bool hasGraphAttention = false;
  module.walk([&](LinalgExtAttentionOp) {
    hasGraphAttention = true;
    return mlir::WalkResult::interrupt();
  });
  if (hasGraphAttention)
    return fail<OnlineAttentionDecompositionStatistics>(
        failure, OnlineAttentionDecompositionFailureKind::BrokenContract,
        "graph attention must be materialized before online decomposition");

  llvm::SmallVector<DecompositionDescriptor, 32> descriptors;
  bool unsupported = false;
  module.walk([&](LinalgExtOnlineAttentionOp operation) {
    mlir::FailureOr<DecompositionDescriptor> descriptor =
        buildDescriptor(operation);
    if (mlir::failed(descriptor)) {
      unsupported = true;
      return mlir::WalkResult::interrupt();
    }
    descriptors.push_back(std::move(*descriptor));
    return mlir::WalkResult::advance();
  });
  if (unsupported)
    return fail<OnlineAttentionDecompositionStatistics>(
        failure, OnlineAttentionDecompositionFailureKind::UnsupportedSemantics,
        "current online-attention maps or static tile types cannot be "
        "decomposed");

  compiler::detail::StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  OnlineAttentionDecompositionStatistics statistics;
  for (const DecompositionDescriptor &descriptor : descriptors) {
    LinalgExtOnlineAttentionOp operation = descriptor.operation;
    rewriter.setInsertionPoint(operation);
    DecomposedState state = decomposeOne(descriptor, rewriter);
    rewriter.replaceOp(operation, mlir::ValueRange{state.accumulator,
                                                   state.maximum, state.sum});
    ++statistics.decomposedOperations;
    ++statistics.qkContractions;
    ++statistics.pvContractions;
    ++statistics.scoreApplications;
    statistics.rowReductions += 2;
    ++statistics.normalizationFactors;
    ++statistics.probabilityUpdates;
    statistics.stateScales += 2;
    ++statistics.scoreScratchTensors;
    statistics.maximumScoreElements =
        std::max(statistics.maximumScoreElements, descriptor.scoreElements);
  }
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyStructuralTileRegions(module)) ||
      mlir::failed(verifyOnlineAttentionDecompositionComplete(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<OnlineAttentionDecompositionStatistics>(
        failure, OnlineAttentionDecompositionFailureKind::CompilerFailure,
        "online-attention decomposition produced invalid current IR");
  return statistics;
}

} // namespace wafer
