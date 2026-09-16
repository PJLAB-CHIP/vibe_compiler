//===- AttentionNormalizationTest.cpp ----------------------------------===//

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class AttentionNormalizationTest : public ::testing::Test {
protected:
  AttentionNormalizationTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                    mlir::tensor::TensorDialect>();
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  static void replaceAll(std::string &text, llvm::StringRef from,
                         llvm::StringRef to) {
    size_t position = 0;
    while ((position = text.find(from.str(), position)) != std::string::npos) {
      text.replace(position, from.size(), to.str());
      position += to.size();
    }
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  parseRepresentativeDecodeProgram(bool aligned, int64_t keyExtent = 1031) {
    std::string path = std::string(WAFER_TEST_SOURCE_DIR) +
                       "/test/Transforms/Linalg/Inputs/"
                       "attention-decode-representative.mlir";
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer)
      return {};
    std::string source = (*buffer)->getBuffer().str();
    if (aligned) {
      replaceAll(source, "1031", "1024");
      replaceAll(source, "1030", "1023");
    } else if (keyExtent != 1031) {
      replaceAll(source, "1031", std::to_string(keyExtent));
      replaceAll(source, "1030", std::to_string(keyExtent - 1));
    }
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::LogicalResult normalize(mlir::ModuleOp module) {
    mlir::PassManager manager(context.get());
    manager.enableVerifier(true);
    wafer::buildNormalizeAttentionPipeline(manager);
    return manager.run(module);
  }

  template <typename Op> static size_t count(mlir::Operation *root) {
    size_t result = 0;
    root->walk([&](Op) { ++result; });
    return result;
  }

  template <typename Op> static Op findSingle(mlir::Operation *root) {
    Op found;
    root->walk([&](Op operation) {
      EXPECT_FALSE(found);
      found = operation;
    });
    return found;
  }

  static mlir::linalg::LinalgOp findScoreContraction(mlir::Operation *root,
                                                     int64_t keyLength) {
    mlir::linalg::LinalgOp found;
    root->walk([&](mlir::linalg::LinalgOp operation) {
      if (operation->getNumResults() != 1 || operation.getNumDpsInputs() != 2 ||
          !llvm::any_of(operation.getIteratorTypesArray(), [](auto iterator) {
            return iterator == mlir::utils::IteratorType::reduction;
          }))
        return;
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(
          operation->getResult(0).getType());
      if (type &&
          type.getShape() == llvm::ArrayRef<int64_t>({2, 1024, keyLength})) {
        EXPECT_FALSE(found);
        found = operation;
      }
    });
    return found;
  }

  static std::string print(mlir::Operation *root) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    root->print(stream);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(AttentionNormalizationTest,
       AlignedFunctionalDecodeFormsOneFixedFlashDecodingRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/true);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(), wafer::AttentionAlgorithm::FlashDecoding);
  EXPECT_TRUE(llvm::equal(attention.getStaticLoopRanges(),
                          llvm::ArrayRef<int64_t>{2, 1024, 128, 1024, 64}));
  EXPECT_EQ(count<mlir::math::ExpOp>(*module), 0u);
}

TEST_F(AttentionNormalizationTest,
       RaggedFunctionalDecodeFormsOneFixedFlashDecodingRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(), wafer::AttentionAlgorithm::FlashDecoding);
  EXPECT_TRUE(llvm::equal(attention.getStaticLoopRanges(),
                          llvm::ArrayRef<int64_t>{2, 1024, 128, 1031, 64}));
  EXPECT_EQ(count<mlir::math::ExpOp>(*module), 0u);
}

TEST_F(AttentionNormalizationTest,
       ProbabilityNarrowingRemainsAfterNormalization) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool bfloat : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(bfloat);
      std::string path =
          std::string(WAFER_TEST_SOURCE_DIR) +
          "/test/Transforms/Linalg/Inputs/attention-decode-representative.mlir";
      auto file = llvm::MemoryBuffer::getFile(path);
      ASSERT_TRUE(static_cast<bool>(file));
      auto text = (*file)->getBuffer().str();
      replaceAll(text, "f16", "f32");
      replaceAll(text, "-6.550400e+04", "0xFF800000");
      replaceAll(text, "1031", std::to_string(extent));
      replaceAll(text, "1030", std::to_string(extent - 1));
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(module);
      mlir::linalg::GenericOp probability;
      module->walk([&](mlir::linalg::GenericOp op) {
        if (llvm::any_of(op.getBody()->without_terminator(),
                         [](mlir::Operation &nested) {
                           return mlir::isa<mlir::arith::DivFOp>(nested);
                         }))
          probability = op;
      });
      ASSERT_TRUE(probability);
      auto *consumer = *probability.getResult(0).getUsers().begin();
      auto pv = mlir::cast<mlir::linalg::GenericOp>(consumer);
      mlir::OpBuilder builder(pv);
      auto original = mlir::cast<mlir::RankedTensorType>(
          probability.getResult(0).getType());
      mlir::Type narrow = bfloat ? mlir::Type(builder.getBF16Type())
                                 : mlir::Type(builder.getF16Type());
      auto identity = builder.getMultiDimIdentityMap(original.getRank());
      auto convert = [&](mlir::Value input, mlir::Type element, bool truncate) {
        auto type = mlir::RankedTensorType::get(original.getShape(), element);
        auto empty = builder.create<mlir::tensor::EmptyOp>(
            pv.getLoc(), type.getShape(), element);
        return builder
            .create<mlir::linalg::GenericOp>(
                pv.getLoc(), mlir::TypeRange{type}, mlir::ValueRange{input},
                mlir::ValueRange{empty},
                llvm::ArrayRef<mlir::AffineMap>{identity, identity},
                llvm::SmallVector<mlir::utils::IteratorType>(
                    original.getRank(), mlir::utils::IteratorType::parallel),
                [&](mlir::OpBuilder &nested, mlir::Location loc,
                    mlir::ValueRange args) {
                  mlir::Value value;
                  if (truncate)
                    value = nested.create<mlir::arith::TruncFOp>(loc, element,
                                                                 args[0]);
                  else
                    value = nested.create<mlir::arith::ExtFOp>(loc, element,
                                                               args[0]);
                  nested.create<mlir::linalg::YieldOp>(loc, value);
                })
            .getResult(0);
      };
      auto rounded = convert(probability.getResult(0), narrow, true);
      auto restored = convert(rounded, builder.getF32Type(), false);
      pv->setOperand(0, restored);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      ASSERT_TRUE(mlir::succeeded(normalize(*module)));
      EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 0u);
      EXPECT_EQ(count<mlir::arith::TruncFOp>(*module), 1u);
      EXPECT_EQ(count<mlir::math::ExpOp>(*module), 1u);
      EXPECT_EQ(rounded.getDefiningOp()->getOperand(0),
                probability.getResult(0));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST_F(AttentionNormalizationTest,
       MaskAccessCompositionPreservesOtherUsesAndStopsAtArithmetic) {
  for (bool aligned : {true, false}) {
    for (bool arithmetic : {false, true}) {
      SCOPED_TRACE(aligned);
      SCOPED_TRACE(arithmetic);
      auto module = parseRepresentativeDecodeProgram(aligned);
      ASSERT_TRUE(module);
      int64_t length = aligned ? 1024 : 1031;
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto score = findScoreContraction(*module, length);
      ASSERT_TRUE(score);
      mlir::OpBuilder builder(context.get());
      auto location = function.getLoc();
      auto maskType =
          mlir::RankedTensorType::get({length, 1024}, builder.getF16Type());
      function.insertArgument(function.getNumArguments(), maskType,
                              mlir::DictionaryAttr{}, location);
      mlir::Value mask = function.getBody().front().getArguments().back();
      builder.setInsertionPointAfter(score);
      auto d0 = builder.getAffineDimExpr(0);
      auto d1 = builder.getAffineDimExpr(1);
      auto d2 = builder.getAffineDimExpr(2);
      auto map = [&](llvm::ArrayRef<mlir::AffineExpr> results) {
        return mlir::AffineMap::get(3, 0, results, context.get());
      };
      auto createForward = [&](mlir::Value input, llvm::ArrayRef<int64_t> shape,
                               mlir::AffineMap inputMap,
                               mlir::AffineMap outputMap, bool negate) {
        auto type = mlir::RankedTensorType::get(shape, builder.getF16Type());
        mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
            location, shape, builder.getF16Type());
        return builder.create<mlir::linalg::GenericOp>(
            location, mlir::TypeRange{type}, mlir::ValueRange{input},
            mlir::ValueRange{empty},
            llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap},
            llvm::SmallVector<mlir::utils::IteratorType, 3>(
                3, mlir::utils::IteratorType::parallel),
            [&](mlir::OpBuilder &nested, mlir::Location loc,
                mlir::ValueRange args) {
              mlir::Value value = args[0];
              if (negate)
                value = nested.create<mlir::arith::NegFOp>(loc, value);
              nested.create<mlir::linalg::YieldOp>(loc, value);
            });
      };
      auto broadcast = createForward(mask, {2, length, 1024}, map({d2, d1}),
                                     map({d0, d2, d1}), arithmetic);
      auto transpose =
          createForward(broadcast.getResult(0), {2, 1024, length},
                        map({d0, d2, d1}), map({d0, d1, d2}), false);
      mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
          location, llvm::ArrayRef<int64_t>{2, 1024, length},
          builder.getF16Type());
      auto identity = builder.getMultiDimIdentityMap(3);
      auto add = builder.create<mlir::linalg::GenericOp>(
          location, mlir::TypeRange{score->getResult(0).getType()},
          mlir::ValueRange{score->getResult(0), transpose.getResult(0)},
          mlir::ValueRange{empty},
          llvm::ArrayRef<mlir::AffineMap>{identity, identity, identity},
          llvm::SmallVector<mlir::utils::IteratorType, 3>(
              3, mlir::utils::IteratorType::parallel),
          [](mlir::OpBuilder &nested, mlir::Location loc,
             mlir::ValueRange args) {
            mlir::Value sum =
                nested.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
            nested.create<mlir::linalg::YieldOp>(loc, sum);
          });
      score->getResult(0).replaceAllUsesExcept(add.getResult(0), add);
      // The broadcast remains independently observable, including its
      // arithmetic.
      auto returned = mlir::cast<mlir::func::ReturnOp>(
          function.getBody().front().getTerminator());
      returned->insertOperands(returned.getNumOperands(),
                               broadcast.getResult(0));
      auto resultTypes = llvm::to_vector(function.getResultTypes());
      resultTypes.push_back(broadcast.getResult(0).getType());
      function.setType(
          builder.getFunctionType(function.getArgumentTypes(), resultTypes));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      ASSERT_TRUE(mlir::succeeded(normalize(*module)));
      auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
      ASSERT_TRUE(attention);
      EXPECT_EQ(count<mlir::arith::NegFOp>(*module), arithmetic ? 1u : 0u);
      mlir::Value expectedMask = arithmetic ? broadcast.getResult(0) : mask;
      EXPECT_EQ(attention.getMask(), expectedMask);
      auto expectedMap = mlir::AffineMap::get(
          5, 0,
          arithmetic
              ? llvm::ArrayRef<mlir::AffineExpr>{d0,
                                                 builder.getAffineDimExpr(3),
                                                 d1}
              : llvm::ArrayRef<mlir::AffineExpr>{builder.getAffineDimExpr(3),
                                                 d1},
          context.get());
      EXPECT_EQ(*attention.getMaskMap(), expectedMap);
      EXPECT_EQ(returned->getOperands().back(), broadcast.getResult(0));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
  }
}

TEST_F(AttentionNormalizationTest,
       ProjectionViewsAndOutputPermutationArePreserved) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool inputView : {false, true})
      for (bool outputView : {false, true}) {
        SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(inputView) +
                     ":" + std::to_string(outputView));
        auto module = parseRepresentativeDecodeProgram(false, extent);
        ASSERT_TRUE(module);
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        mlir::linalg::LinalgOp pv;
        function.walk([&](mlir::linalg::LinalgOp op) {
          if (op->getNumResults() == 1 && op.getNumDpsInputs() == 2 &&
              llvm::is_contained(op.getIteratorTypesArray(),
                                 mlir::utils::IteratorType::reduction) &&
              mlir::cast<mlir::ShapedType>(op->getResult(0).getType())
                      .getShape() == llvm::ArrayRef<int64_t>{2, 1024, 64})
            pv = op;
        });
        ASSERT_TRUE(pv);
        mlir::OpBuilder builder(pv);
        auto loc = pv.getLoc();
        mlir::Value expectedValue = pv.getDpsInputs()[1];
        if (inputView) {
          auto flatType = mlir::RankedTensorType::get({2 * extent, 64},
                                                      builder.getF16Type());
          auto flat = builder.create<mlir::tensor::CollapseShapeOp>(
              loc, flatType, expectedValue,
              llvm::ArrayRef<mlir::ReassociationIndices>{{0, 1}, {2}});
          mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
              loc, flatType.getShape(), builder.getF16Type());
          auto identity = builder.getMultiDimIdentityMap(2);
          auto projection = builder.create<mlir::linalg::GenericOp>(
              loc, mlir::TypeRange{flatType}, mlir::ValueRange{flat},
              mlir::ValueRange{empty},
              llvm::ArrayRef<mlir::AffineMap>{identity, identity},
              llvm::SmallVector<mlir::utils::IteratorType, 2>(
                  2, mlir::utils::IteratorType::parallel),
              [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                auto neg = b.create<mlir::arith::NegFOp>(l, args[0]);
                b.create<mlir::linalg::YieldOp>(l, neg.getResult());
              });
          expectedValue = builder.create<mlir::tensor::ExpandShapeOp>(
              loc, expectedValue.getType(), projection.getResult(0),
              llvm::ArrayRef<mlir::ReassociationIndices>{{0, 1}, {2}});
          pv.getDpsInputOperand(1)->set(expectedValue);
        }
        mlir::Value returned;
        if (outputView) {
          builder.setInsertionPointAfter(pv);
          auto type =
              mlir::RankedTensorType::get({1024, 2, 64}, builder.getF16Type());
          mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
              loc, type.getShape(), builder.getF16Type());
          auto permutation = mlir::AffineMap::getPermutationMap(
              llvm::ArrayRef<unsigned>{1, 0, 2}, context.get());
          auto transpose = builder.create<mlir::linalg::GenericOp>(
              loc, mlir::TypeRange{type}, mlir::ValueRange{pv->getResult(0)},
              mlir::ValueRange{empty},
              llvm::ArrayRef<mlir::AffineMap>{
                  permutation, builder.getMultiDimIdentityMap(3)},
              llvm::SmallVector<mlir::utils::IteratorType, 3>(
                  3, mlir::utils::IteratorType::parallel),
              [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                b.create<mlir::linalg::YieldOp>(l, args[0]);
              });
          auto flatType =
              mlir::RankedTensorType::get({1024, 128}, builder.getF16Type());
          returned = builder.create<mlir::tensor::CollapseShapeOp>(
              loc, flatType, transpose.getResult(0),
              llvm::ArrayRef<mlir::ReassociationIndices>{{0}, {1, 2}});
          auto resultTypes = llvm::to_vector(function.getResultTypes());
          resultTypes[0] = flatType;
          function.setType(builder.getFunctionType(function.getArgumentTypes(),
                                                   resultTypes));
          auto ret = mlir::cast<mlir::func::ReturnOp>(
              function.getBody().front().getTerminator());
          ret->setOperand(0, returned);
        }
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        ASSERT_TRUE(mlir::succeeded(normalize(*module)));
        auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
        ASSERT_TRUE(attention);
        EXPECT_EQ(attention.getValue(), expectedValue);
        if (outputView) {
          auto ret = mlir::cast<mlir::func::ReturnOp>(
              function.getBody().front().getTerminator());
          EXPECT_EQ(ret.getOperand(0), returned);
          auto collapse =
              returned.getDefiningOp<mlir::tensor::CollapseShapeOp>();
          auto transpose =
              collapse.getSrc().getDefiningOp<mlir::linalg::GenericOp>();
          ASSERT_TRUE(transpose);
          EXPECT_EQ(transpose.getDpsInputs()[0], attention.getResult(0));
        }
        EXPECT_FALSE(attention.getScoreRegion().empty());
      }
}

TEST_F(AttentionNormalizationTest, EqualExtentsDoNotDetermineKeyAxes) {
  auto module = parseRepresentativeDecodeProgram(false, 128);
  ASSERT_TRUE(module);
  auto score = findScoreContraction(*module, 128);
  ASSERT_TRUE(score);
  auto key = score.getDpsInputs()[1];
  auto oldTranspose = key.getDefiningOp<mlir::linalg::GenericOp>();
  ASSERT_TRUE(oldTranspose);
  auto expectedKey = oldTranspose.getDpsInputs()[0];
  mlir::OpBuilder builder(score);
  auto type = mlir::cast<mlir::RankedTensorType>(key.getType());
  mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
      score.getLoc(), type.getShape(), type.getElementType());
  auto transpose = builder.create<mlir::linalg::GenericOp>(
      score.getLoc(), mlir::TypeRange{type}, mlir::ValueRange{key},
      mlir::ValueRange{empty},
      llvm::ArrayRef<mlir::AffineMap>{
          mlir::AffineMap::getPermutationMap(llvm::ArrayRef<unsigned>{0, 2, 1},
                                             context.get()),
          builder.getMultiDimIdentityMap(3)},
      llvm::SmallVector<mlir::utils::IteratorType, 3>(
          3, mlir::utils::IteratorType::parallel),
      [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
        b.create<mlir::linalg::YieldOp>(l, args[0]);
      });
  score.getDpsInputOperand(1)->set(transpose.getResult(0));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getKey(), expectedKey);
  EXPECT_EQ(attention.getKeyMap().getResult(1), builder.getAffineDimExpr(2));
  EXPECT_EQ(attention.getKeyMap().getResult(2), builder.getAffineDimExpr(3));
  EXPECT_EQ(attention.getAlgorithm(),
            wafer::AttentionAlgorithm::FlashAttention);
}

TEST_F(AttentionNormalizationTest, SymbolSpellingDoesNotAffectClassification) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("functional_decode");
  ASSERT_TRUE(function);
  function.setName("unrelated_symbol");
  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(), wafer::AttentionAlgorithm::FlashDecoding);
}

TEST_F(AttentionNormalizationTest,
       AlignedUnreturnedCacheUpdatesClassifyAsFlashAttention) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/true);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("functional_decode");
  ASSERT_TRUE(function);
  auto returnOp = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  returnOp->setOperand(1, function.getArgument(5));
  returnOp->setOperand(2, function.getArgument(6));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(),
            wafer::AttentionAlgorithm::FlashAttention);
  EXPECT_TRUE(llvm::equal(attention.getStaticLoopRanges(),
                          llvm::ArrayRef<int64_t>{2, 1024, 128, 1024, 64}));
}

TEST_F(AttentionNormalizationTest,
       RaggedUnreturnedCacheUpdatesClassifyAsFlashAttention) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("functional_decode");
  ASSERT_TRUE(function);
  auto returnOp = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  returnOp->setOperand(1, function.getArgument(5));
  returnOp->setOperand(2, function.getArgument(6));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(),
            wafer::AttentionAlgorithm::FlashAttention);
  EXPECT_TRUE(llvm::equal(attention.getStaticLoopRanges(),
                          llvm::ArrayRef<int64_t>{2, 1024, 128, 1031, 64}));
}

TEST_F(AttentionNormalizationTest, NormalizesEveryIndependentValueRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  mlir::linalg::LinalgOp valueContraction;
  module->walk([&](mlir::linalg::LinalgOp operation) {
    auto resultType = operation->getNumResults() == 1
                          ? mlir::dyn_cast<mlir::RankedTensorType>(
                                operation->getResult(0).getType())
                          : mlir::RankedTensorType{};
    if (mlir::isa<mlir::linalg::GenericOp>(operation.getOperation()) &&
        operation.getNumDpsInputs() == 2 && resultType &&
        resultType.getShape() == llvm::ArrayRef<int64_t>({2, 1024, 64}))
      valueContraction = operation;
  });
  ASSERT_TRUE(valueContraction);
  mlir::OpBuilder builder(valueContraction);
  builder.setInsertionPointAfter(valueContraction);
  mlir::Operation *cloned = builder.clone(*valueContraction.getOperation());
  auto function = valueContraction->getParentOfType<mlir::func::FuncOp>();
  ASSERT_TRUE(function);
  llvm::SmallVector<mlir::Type, 4> resultTypes(
      function.getResultTypes().begin(), function.getResultTypes().end());
  resultTypes.push_back(cloned->getResult(0).getType());
  function.setType(
      builder.getFunctionType(function.getArgumentTypes(), resultTypes));
  auto oldReturn = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  builder.setInsertionPoint(oldReturn);
  llvm::SmallVector<mlir::Value, 4> returned(oldReturn.getOperands().begin(),
                                             oldReturn.getOperands().end());
  returned.push_back(cloned->getResult(0));
  builder.create<mlir::func::ReturnOp>(oldReturn.getLoc(), returned);
  oldReturn.erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 2u);
  EXPECT_EQ(count<mlir::math::ExpOp>(*module), 0u);
  EXPECT_FALSE(findScoreContraction(*module, 1031));
}

TEST_F(AttentionNormalizationTest, ExtraScoreUseKeepsItsOriginalProducerChain) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  auto score = findScoreContraction(*module, 1031);
  ASSERT_TRUE(score);
  mlir::OpBuilder builder(score);
  builder.setInsertionPointAfter(score);
  builder.create<mlir::tensor::CastOp>(
      score.getLoc(), score->getResult(0).getType(), score->getResult(0));

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 1u);
  EXPECT_TRUE(findScoreContraction(*module, 1031));
  EXPECT_EQ(count<mlir::tensor::CastOp>(*module), 1u);
}

TEST_F(AttentionNormalizationTest, PrecomputedScoresRemainOrdinaryLinalg) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseRepresentativeDecodeProgram(/*aligned=*/false);
  ASSERT_TRUE(module);
  auto score = findScoreContraction(*module, 1031);
  ASSERT_TRUE(score);
  score->getResult(0).replaceAllUsesWith(score.getDpsInits().front());
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = print(module->getOperation());

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 0u);
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
