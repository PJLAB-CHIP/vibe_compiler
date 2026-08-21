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

  mlir::OwningOpRef<mlir::ModuleOp> parseDecodeProgram() {
    std::string path =
        std::string(WAFER_TEST_SOURCE_DIR) +
        "/unittests/Planning/Search/Inputs/functional-decode.mlir";
    return mlir::parseSourceFile<mlir::ModuleOp>(
        path, mlir::ParserConfig(context.get()));
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
       FunctionalDecodeFormsOneFixedFlashDecodingRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  auto attention = findSingle<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  EXPECT_EQ(attention.getAlgorithm(), wafer::AttentionAlgorithm::FlashDecoding);
  EXPECT_EQ(count<mlir::math::ExpOp>(*module), 0u);
}

TEST_F(AttentionNormalizationTest, SymbolSpellingDoesNotAffectClassification) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
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
       UnreturnedCacheUpdatesClassifyAsFlashAttention) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
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
}

TEST_F(AttentionNormalizationTest, NormalizesEveryIndependentValueRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
  ASSERT_TRUE(module);
  mlir::linalg::LinalgOp valueContraction;
  module->walk([&](mlir::linalg::LinalgOp operation) {
    auto resultType = operation->getNumResults() == 1
                          ? mlir::dyn_cast<mlir::RankedTensorType>(
                                operation->getResult(0).getType())
                          : mlir::RankedTensorType{};
    if (mlir::isa<mlir::linalg::GenericOp>(operation.getOperation()) &&
        operation.getNumDpsInputs() == 2 && resultType &&
        resultType.getShape() == llvm::ArrayRef<int64_t>({2, 3}))
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
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
  ASSERT_TRUE(module);
  auto score = findSingle<mlir::linalg::MatmulOp>(*module);
  ASSERT_TRUE(score);
  mlir::OpBuilder builder(score);
  builder.setInsertionPointAfter(score);
  builder.create<mlir::tensor::CastOp>(
      score.getLoc(), score.getResult(0).getType(), score.getResult(0));

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 1u);
  EXPECT_EQ(count<mlir::linalg::MatmulOp>(*module), 1u);
  EXPECT_EQ(count<mlir::tensor::CastOp>(*module), 1u);
}

TEST_F(AttentionNormalizationTest, PrecomputedScoresRemainOrdinaryLinalg) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseDecodeProgram();
  ASSERT_TRUE(module);
  auto score = findSingle<mlir::linalg::MatmulOp>(*module);
  ASSERT_TRUE(score);
  score.getResult(0).replaceAllUsesWith(score.getDpsInits().front());
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = print(module->getOperation());

  ASSERT_TRUE(mlir::succeeded(normalize(*module)));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*module), 0u);
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
