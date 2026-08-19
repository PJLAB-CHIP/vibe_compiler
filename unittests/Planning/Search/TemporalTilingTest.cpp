//===- TemporalTilingTest.cpp -----------------------------------------===//

#include "Wafer/Planning/Search/TemporalTiling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>

namespace {

using wafer::TileId;
using wafer::compiler::detail::StructuredDAGAnalysis;
using wafer::compiler::detail::StructuredDAGNodePlacement;
using wafer::compiler::detail::TemporalNodeAssignment;
using wafer::compiler::detail::TemporalNodeDomain;

class TemporalTilingTest : public ::testing::Test {
protected:
  TemporalTilingTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(TemporalTilingTest, EnumeratesEverySizeAndActiveLoopPermutation) {
  auto module = parse(R"mlir(
module {
  func.func @map(%input: tensor<2x3xf16>) -> tensor<2x3xf16> {
    %empty = tensor.empty() : tensor<2x3xf16>
    %result = linalg.map ins(%input : tensor<2x3xf16>)
        outs(%empty : tensor<2x3xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<2x3xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  StructuredDAGNodePlacement placement{0, {1, 1}, {TileId(0)}, std::nullopt};
  auto domain = TemporalNodeDomain::create(dag->getNodes().front(), placement,
                                           &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;

  std::set<TemporalNodeAssignment> actual;
  std::optional<TemporalNodeAssignment> current = domain->getFirstAssignment();
  while (current) {
    EXPECT_TRUE(domain->contains(*current));
    actual.insert(*current);
    auto next = domain->getNextAssignment(*current);
    ASSERT_TRUE(mlir::succeeded(next));
    current = *next;
  }
  EXPECT_EQ(actual.size(), 8u);
  EXPECT_TRUE(actual.count({0, {2, 3}, {}}));
  EXPECT_TRUE(actual.count({0, {1, 2}, {0, 1}}));
  EXPECT_TRUE(actual.count({0, {1, 2}, {1, 0}}));
  EXPECT_FALSE(domain->contains({0, {2, 3}, {0}}));
}

TEST_F(TemporalTilingTest, SpatialRemainderDefinesSharedLocalMaximum) {
  auto module = parse(R"mlir(
module {
  func.func @map(%input: tensor<5xf16>) -> tensor<5xf16> {
    %empty = tensor.empty() : tensor<5xf16>
    %result = linalg.map ins(%input : tensor<5xf16>)
        outs(%empty : tensor<5xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<5xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  StructuredDAGNodePlacement placement{
      0, {2}, {TileId(0), TileId(1)}, std::nullopt};
  auto domain = TemporalNodeDomain::create(dag->getNodes().front(), placement,
                                           &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  EXPECT_EQ(domain->getFirstAssignment().iteratorTileSizes,
            (llvm::SmallVector<int64_t, 4>{3}));
  unsigned count = 0;
  std::optional<TemporalNodeAssignment> current = domain->getFirstAssignment();
  while (current) {
    ++count;
    auto next = domain->getNextAssignment(*current);
    ASSERT_TRUE(mlir::succeeded(next));
    current = *next;
  }
  EXPECT_EQ(count, 3u);
}

} // namespace
