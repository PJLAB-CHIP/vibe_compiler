//===- TemporalTilingTest.cpp -----------------------------------------===//

#include "Wafer/Planning/Search/TemporalTiling.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
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
    wafer::registerWaferCoreDialects(registry);
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
  StructuredDAGNodePlacement placement{0, {1, 1}, {TileId(0)}};
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
  std::set<TemporalNodeAssignment> reference;
  for (int64_t first = 1; first <= 2; ++first)
    for (int64_t second = 1; second <= 3; ++second) {
      llvm::SmallVector<uint32_t, 4> active;
      if (first < 2)
        active.push_back(0);
      if (second < 3)
        active.push_back(1);
      do {
        reference.insert({0, {first, second}, active});
      } while (std::next_permutation(active.begin(), active.end()));
    }
  EXPECT_EQ(actual, reference);
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
  StructuredDAGNodePlacement placement{0, {2}, {TileId(0), TileId(1)}};
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
