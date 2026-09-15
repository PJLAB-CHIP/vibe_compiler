//===- BatchNormFusionTest.cpp - Exact normalization scalar fusion --------===//

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <string>

namespace {
class BatchNormFusionTest : public ::testing::Test {
protected:
  BatchNormFusionTest() {
    mlir::DialectRegistry registry;
    wafer::compiler::detail::registerCompilationDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                    mlir::tensor::TensorDialect>();
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeChain(llvm::ArrayRef<int64_t> shape, unsigned feature,
            mlir::FloatType storage, unsigned reciprocal, bool relu,
            bool broadcast = true, bool fanout = false) {
    mlir::OpBuilder b(context.get());
    auto loc = b.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    auto activationType = mlir::RankedTensorType::get(shape, storage);
    auto featureType = mlir::RankedTensorType::get({shape[feature]}, storage);
    auto f32 = b.getF32Type();
    llvm::SmallVector<mlir::Type> arguments{
        activationType, featureType, featureType,
        featureType,    featureType, mlir::RankedTensorType::get({}, f32)};
    llvm::SmallVector<mlir::Type> results{activationType};
    if (fanout)
      results.push_back(mlir::RankedTensorType::get(shape, f32));
    b.setInsertionPointToStart(module.getBody());
    auto function = b.create<mlir::func::FuncOp>(
        loc, "main", b.getFunctionType(arguments, results));
    b.setInsertionPointToStart(function.addEntryBlock());

    auto map = [&](mlir::Value input, llvm::ArrayRef<int64_t> outputShape) {
      auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
      unsigned rank = outputShape.size();
      if (inputType.getRank() == 0)
        return mlir::AffineMap::get(rank, 0, {}, context.get());
      if (inputType.getShape() == outputShape)
        return b.getMultiDimIdentityMap(rank);
      EXPECT_EQ(inputType.getRank(), 1);
      return mlir::AffineMap::get(rank, 0, {b.getAffineDimExpr(feature)},
                                  context.get());
    };
    auto apply = [&](llvm::ArrayRef<mlir::Value> inputs,
                     llvm::ArrayRef<int64_t> outputShape, mlir::Type element,
                     llvm::StringRef scalarName) -> mlir::Value {
      auto type = mlir::RankedTensorType::get(outputShape, element);
      auto empty = b.create<mlir::tensor::EmptyOp>(loc, outputShape, element);
      llvm::SmallVector<mlir::AffineMap> maps;
      for (auto input : inputs)
        maps.push_back(map(input, outputShape));
      maps.push_back(b.getMultiDimIdentityMap(outputShape.size()));
      auto op = b.create<mlir::linalg::GenericOp>(
          loc, mlir::TypeRange{type}, inputs, mlir::ValueRange{empty}, maps,
          llvm::SmallVector<mlir::utils::IteratorType>(
              outputShape.size(), mlir::utils::IteratorType::parallel),
          [&](mlir::OpBuilder &nested, mlir::Location location,
              mlir::ValueRange values) {
            mlir::Value value = values.front();
            if (!scalarName.empty()) {
              mlir::OperationState state(location, scalarName);
              state.addOperands(values.take_front(inputs.size()));
              state.addTypes(element);
              value = nested.create(state)->getResult(0);
            }
            nested.create<mlir::linalg::YieldOp>(location, value);
          });
      return op.getResult(0);
    };
    auto widen = [&](mlir::Value value) {
      auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
      return storage == f32
                 ? value
                 : apply({value}, type.getShape(), f32, "arith.extf");
    };
    auto expandFeature = [&](mlir::Value value) {
      return broadcast ? apply({value}, shape, f32, "") : value;
    };
    llvm::SmallVector<int64_t> channel{shape[feature]};
    auto x = widen(function.getArgument(0));
    auto mean = widen(function.getArgument(1));
    auto variance = widen(function.getArgument(2));
    auto gamma = widen(function.getArgument(3));
    auto beta = widen(function.getArgument(4));
    mlir::Value epsilon = function.getArgument(5);
    if (broadcast)
      epsilon = apply({epsilon}, channel, f32, "");
    auto stabilized = apply({variance, epsilon}, channel, f32, "arith.addf");
    auto stddev = apply({stabilized}, channel, f32,
                        reciprocal == 2 ? "math.rsqrt" : "math.sqrt");
    auto centered = apply({x, expandFeature(mean)}, shape, f32, "arith.subf");
    mlir::Value normalized;
    if (reciprocal == 1) {
      auto ones = b.create<mlir::arith::ConstantOp>(
          loc, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get(channel, f32),
                   b.getFloatAttr(f32, 1.0)));
      auto inverse = apply({ones, stddev}, channel, f32, "arith.divf");
      normalized =
          apply({centered, expandFeature(inverse)}, shape, f32, "arith.mulf");
    } else {
      normalized = apply({centered, expandFeature(stddev)}, shape, f32,
                         reciprocal == 2 ? "arith.mulf" : "arith.divf");
    }
    auto scaled =
        apply({normalized, expandFeature(gamma)}, shape, f32, "arith.mulf");
    auto shifted =
        apply({scaled, expandFeature(beta)}, shape, f32, "arith.addf");
    auto output = storage == f32
                      ? shifted
                      : apply({shifted}, shape, storage, "arith.truncf");
    if (relu) {
      auto zero = b.create<mlir::arith::ConstantOp>(
          loc, mlir::DenseElementsAttr::get(activationType,
                                            b.getFloatAttr(storage, 0.0)));
      output = apply({output, zero}, shape, storage, "arith.maximumf");
    }
    llvm::SmallVector<mlir::Value> outputs{output};
    if (fanout)
      outputs.push_back(centered);
    b.create<mlir::func::ReturnOp>(loc, outputs);
    return module;
  }

  bool fuse(mlir::ModuleOp module) {
    mlir::PassManager pm(context.get());
    pm.addNestedPass<mlir::func::FuncOp>(
        wafer::createFuseBatchNormInferencePass());
    return mlir::succeeded(pm.run(module));
  }

  static std::string print(mlir::Operation *op) {
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
  }

  static unsigned generics(mlir::ModuleOp module) {
    unsigned result = 0;
    module.walk([&](mlir::linalg::GenericOp) { ++result; });
    return result;
  }

  // A bounded numeric oracle uses each original scalar op's constant folder;
  // it does not reproduce the fusion matcher or its coordinate composition.
  mlir::Attribute evaluate(mlir::Value value,
                           llvm::ArrayRef<int64_t> coordinates) {
    auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      int64_t linear = 0;
      for (auto [axis, coordinate] : llvm::enumerate(coordinates))
        linear = linear * type.getDimSize(axis) + coordinate;
      double number = 0;
      switch (argument.getArgNumber()) {
      case 0:
        number = (linear % 131 - 65) * 0.03125;
        break;
      case 1:
        number = (linear % 5 - 2) * 0.0625;
        break;
      case 2:
        number = 0.5 + (linear % 7) * 0.125;
        break;
      case 3:
        number = 0.75 + (linear % 3) * 0.125;
        break;
      case 4:
        number = (linear % 4 - 2) * 0.0625;
        break;
      case 5:
        number = 1e-5;
        break;
      default:
        ADD_FAILURE();
      }
      return mlir::FloatAttr::get(type.getElementType(), number);
    }
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto data = mlir::cast<mlir::DenseElementsAttr>(constant.getValue());
      return data.getSplatValue<mlir::Attribute>();
    }
    auto generic = value.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic) {
      ADD_FAILURE() << "numeric fixture has an unexpected producer";
      return {};
    }
    llvm::DenseMap<mlir::Value, mlir::Attribute> values;
    for (auto [index, input] : llvm::enumerate(generic.getDpsInputs())) {
      llvm::SmallVector<int64_t> indexValues;
      for (auto expression :
           generic.getIndexingMapsArray()[index].getResults()) {
        if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression))
          indexValues.push_back(coordinates[dim.getPosition()]);
        else
          indexValues.push_back(
              mlir::cast<mlir::AffineConstantExpr>(expression).getValue());
      }
      values[generic.getRegionInputArgs()[index]] =
          evaluate(input, indexValues);
    }
    for (auto &op : generic.getBody()->without_terminator()) {
      // Pinned math.rsqrt has no constant folder. Evaluate its positive F32
      // fixture inputs with one final conversion to the declared result type.
      if (mlir::isa<mlir::math::RsqrtOp>(op)) {
        auto input =
            mlir::cast<mlir::FloatAttr>(values.lookup(op.getOperand(0)));
        values[op.getResult(0)] =
            mlir::FloatAttr::get(op.getResult(0).getType(),
                                 1.0 / std::sqrt(input.getValueAsDouble()));
        continue;
      }
      if (mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(op)) {
        auto input =
            mlir::cast<mlir::FloatAttr>(values.lookup(op.getOperand(0)));
        auto target = mlir::cast<mlir::FloatType>(op.getResult(0).getType());
        llvm::APFloat number = input.getValue();
        bool losesInfo;
        number.convert(target.getFloatSemantics(),
                       llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        values[op.getResult(0)] = mlir::FloatAttr::get(target, number);
        continue;
      }
      llvm::SmallVector<mlir::Attribute> operands;
      for (auto operand : op.getOperands())
        operands.push_back(values.lookup(operand));
      llvm::SmallVector<mlir::OpFoldResult> folded;
      if (mlir::failed(op.fold(operands, folded)) || folded.size() != 1) {
        ADD_FAILURE() << "scalar oracle could not fold "
                      << op.getName().getStringRef().str();
        return {};
      }
      values[op.getResult(0)] =
          mlir::isa<mlir::Attribute>(folded.front())
              ? mlir::cast<mlir::Attribute>(folded.front())
              : values.lookup(mlir::cast<mlir::Value>(folded.front()));
    }
    return values.lookup(
        mlir::cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator())
            .getValues()
            .front());
  }

  static mlir::Value returned(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    return mlir::cast<mlir::func::ReturnOp>(
               function.getBody().front().getTerminator())
        .getOperand(0);
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(BatchNormFusionTest, PreservesFeatureAxesTypesAndScalarBodies) {
  mlir::OpBuilder b(context.get());
  for (unsigned rank : {3u, 4u})
    for (unsigned feature = 0; feature < rank; ++feature)
      for (int64_t extent : {1024, 1025, 1031})
        for (mlir::FloatType storage :
             {b.getF16Type(), b.getBF16Type(), b.getF32Type()})
          for (unsigned reciprocal : {0u, 1u, 2u})
            for (bool relu : {false, true}) {
              SCOPED_TRACE(::testing::Message()
                           << rank << '/' << feature << '/' << extent << '/'
                           << reciprocal << '/' << relu);
              llvm::SmallVector<int64_t> shape(rank, 2);
              shape[feature] = 17;
              shape[(feature + 1) % rank] = extent;
              auto module =
                  makeChain(shape, feature, storage, reciprocal, relu);
              ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
              unsigned originalScalarCount = 0;
              module->walk([&](mlir::Operation *op) {
                originalScalarCount +=
                    op->getParentOfType<mlir::linalg::GenericOp>() &&
                    !mlir::isa<mlir::linalg::YieldOp>(op);
              });
              ASSERT_TRUE(fuse(*module));
              EXPECT_EQ(generics(*module), 1u);
              auto result =
                  returned(*module).getDefiningOp<mlir::linalg::GenericOp>();
              ASSERT_TRUE(result);
              EXPECT_EQ(std::distance(result.getBody()->begin(),
                                      result.getBody()->end()) -
                            1,
                        originalScalarCount);
              EXPECT_EQ(result.getResult(0).getType(),
                        mlir::RankedTensorType::get(shape, storage));
              const std::string once = print(module->getOperation());
              ASSERT_TRUE(fuse(*module));
              EXPECT_EQ(print(module->getOperation()), once);
            }
}

TEST_F(BatchNormFusionTest, RetainsInternalFanoutWithoutDuplicatingCompute) {
  mlir::OpBuilder b(context.get());
  for (int64_t extent : {1024, 1025}) {
    auto module =
        makeChain({2, extent, 17}, 2, b.getF16Type(), true, true, true, true);
    const std::string before = print(module->getOperation());
    ASSERT_TRUE(fuse(*module));
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(BatchNormFusionTest, RejectsDifferentFeatureAxes) {
  mlir::OpBuilder b(context.get());
  for (int64_t extent : {1024, 1025}) {
    auto module = makeChain({17, extent, 17}, 2, b.getF16Type(), true, true);
    mlir::linalg::GenericOp lastBroadcast;
    module->walk([&](mlir::linalg::GenericOp op) {
      if (op.getBody()->without_terminator().empty())
        lastBroadcast = op;
    });
    ASSERT_TRUE(lastBroadcast);
    auto maps = lastBroadcast.getIndexingMapsArray();
    maps[0] =
        mlir::AffineMap::get(3, 0, {b.getAffineDimExpr(0)}, context.get());
    lastBroadcast.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    const std::string before = print(module->getOperation());
    ASSERT_TRUE(fuse(*module));
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(BatchNormFusionTest, KeepsFeatureTemporariesCompactThroughInstrAndSPM) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  mlir::OpBuilder b(context.get());
  for (mlir::FloatType storage : {b.getF16Type(), b.getBF16Type()})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(extent);
      auto module = makeChain({2, extent, 17}, 2, storage, false, false);
      ASSERT_TRUE(fuse(*module));
      ASSERT_EQ(generics(*module), 1u);
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      mlir::IRRewriter rewriter(context.get());
      rewriter.setInsertionPoint(function);
      auto tile = rewriter.create<TileModuleOp>(function.getLoc(),
                                                rewriter.getI64IntegerAttr(0),
                                                rewriter.getI64IntegerAttr(0));
      tile.getBody().push_back(new mlir::Block());
      function->moveBefore(&tile.getBody().front(),
                           tile.getBody().front().end());
      auto ret = mlir::cast<mlir::func::ReturnOp>(
          function.getBody().front().getTerminator());
      llvm::SmallVector<mlir::Operation *> originals;
      for (auto &op : function.getBody().front().without_terminator())
        originals.push_back(&op);
      rewriter.setInsertionPoint(ret);
      auto region = rewriter.create<TileRegionOp>(
          function.getLoc(), ret.getOperandTypes(), function.getArguments());
      auto *body = new mlir::Block();
      region.getBody().push_back(body);
      mlir::IRMapping mapping;
      for (auto argument : function.getArguments())
        mapping.map(argument,
                    body->addArgument(argument.getType(), argument.getLoc()));
      rewriter.setInsertionPointToStart(body);
      for (auto *op : originals)
        rewriter.clone(*op, mapping);
      rewriter.create<TileYieldOp>(function.getLoc(),
                                   mapping.lookup(ret.getOperand(0)));
      ret.setOperand(0, region.getResult(0));
      for (auto *op : llvm::reverse(originals))
        rewriter.eraseOp(op);
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      auto domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded());
      auto first = domain.domain->getFirstChoice();
      auto choice = *first.getChoice();
      ASSERT_EQ(choice.scopes.size(), 1u);
      choice.scopes[0].iteratorTileSizes[1] = 128;
      choice.scopes[0].loopOrder = {1};
      ASSERT_TRUE(domain.domain->contains(choice));
      ASSERT_TRUE(mlir::succeeded(
          applyTemporalTiling({{*domain.domain, choice}}, relations)));
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      unsigned roots = 0, featureConversions = 0;
      module->walk([&](ComputeElementwiseOp op) {
        if (op.getKind() == ComputeElementwiseKind::Sqrt) {
          ++roots;
          auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
          EXPECT_EQ(type.getNumElements(), 17);
        }
      });
      module->walk([&](ComputeConvertOp op) {
        auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
        featureConversions += type.getNumElements() == 17;
      });
      EXPECT_EQ(roots, extent % 128 ? 2u : 1u);
      EXPECT_GE(featureConversions, 4u);
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &owner = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp> regions;
      owner.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (auto current : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(current, session)))
            << print(current);
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*owner.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*owner.module)));
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(owner.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned))
          << static_cast<unsigned>(failure.kind);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
}

TEST_F(BatchNormFusionTest, MatchesAllOutputBitsForMainAndTailShapes) {
  mlir::OpBuilder b(context.get());
  for (mlir::FloatType storage : {b.getF16Type(), b.getBF16Type()})
    for (int64_t extent : {1024, 1025})
      for (unsigned reciprocal : {0u, 1u, 2u}) {
        auto module = makeChain({2, extent, 17}, 2, storage, reciprocal, true);
        auto original = returned(*module);
        llvm::SmallVector<mlir::Attribute> expected;
        for (int64_t n = 0; n < 2; ++n)
          for (int64_t x = 0; x < extent; ++x)
            for (int64_t c = 0; c < 17; ++c) {
              auto value = evaluate(original, {n, x, c});
              ASSERT_TRUE(value);
              expected.push_back(value);
            }
        ASSERT_TRUE(fuse(*module));
        ASSERT_EQ(generics(*module), 1u);
        size_t index = 0;
        for (int64_t n = 0; n < 2; ++n)
          for (int64_t x = 0; x < extent; ++x)
            for (int64_t c = 0; c < 17; ++c)
              ASSERT_EQ(evaluate(returned(*module), {n, x, c}),
                        expected[index++]);
      }
}
} // namespace
