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
  parseRepresentativeDecodeProgram(bool aligned) {
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
