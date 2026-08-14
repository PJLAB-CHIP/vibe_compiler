#include "../../lib/Wafer/Analysis/CollectiveTopologyAnalysis.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

class CollectiveTopologyTest : public ::testing::Test {
protected:
  CollectiveTopologyTest() {
    wafer::registerWaferCoreDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef body) {
    return mlir::parseSourceString<mlir::ModuleOp>(body, &context);
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(CollectiveTopologyTest, FindsMinimumHopCycleFromExplicitPlacement) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 4>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 0, 1, 0>}
}
)mlir");
  ASSERT_TRUE(module);
  const int64_t ranks[] = {0, 1, 2, 3};
  auto order = wafer::analysis::buildMinimumHopCollectiveRingOrder(
      module->getOperation(), ranks);
  ASSERT_TRUE(mlir::succeeded(order));
  EXPECT_EQ(order->groupIndices, (llvm::SmallVector<int64_t, 4>{0, 2, 1, 3}));
}

TEST_F(CollectiveTopologyTest,
       BuildsMinimumHopOrderedTreeAndChoosesCenterRoot) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
}
)mlir");
  ASSERT_TRUE(module);
  const int64_t ranks[] = {0, 1, 2, 3};
  auto tree = wafer::analysis::buildMinimumHopCollectiveTree(
      module->getOperation(), ranks);
  ASSERT_TRUE(mlir::succeeded(tree));
  EXPECT_EQ(tree->rootGroupIndex, 1);
  EXPECT_EQ(tree->parentGroupIndices,
            (llvm::SmallVector<int64_t, 4>{1, -1, 1, 2}));
  ASSERT_EQ(tree->childGroupIndices.size(), 4u);
  EXPECT_EQ(tree->childGroupIndices[1], (llvm::SmallVector<int64_t, 2>{0, 2}));
  EXPECT_EQ(tree->childGroupIndices[2], (llvm::SmallVector<int64_t, 1>{3}));
  EXPECT_EQ(tree->depths, (llvm::SmallVector<int64_t, 4>{1, 0, 1, 2}));
}

TEST_F(CollectiveTopologyTest,
       OrderedTreePreservesGroupOrderAcrossTopologyLocalEdges) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 4>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 0, 1, 0>}
}
)mlir");
  ASSERT_TRUE(module);
  const int64_t ranks[] = {0, 1, 2, 3};
  auto tree = wafer::analysis::buildMinimumHopCollectiveTree(
      module->getOperation(), ranks);
  ASSERT_TRUE(mlir::succeeded(tree));

  // The unconstrained minimum spanning tree for this placement gives group
  // index zero two higher-index children and cannot preserve operand order.
  // The ordered optimum retains the same total hop cost while every subtree
  // remains a contiguous rank_group interval.
  EXPECT_EQ(tree->rootGroupIndex, 0);
  EXPECT_EQ(tree->parentGroupIndices,
            (llvm::SmallVector<int64_t, 4>{-1, 3, 1, 0}));
  ASSERT_EQ(tree->childGroupIndices.size(), 4u);
  EXPECT_EQ(tree->childGroupIndices[0], (llvm::SmallVector<int64_t, 1>{3}));
  EXPECT_EQ(tree->childGroupIndices[1], (llvm::SmallVector<int64_t, 1>{2}));
  EXPECT_EQ(tree->childGroupIndices[3], (llvm::SmallVector<int64_t, 1>{1}));
  EXPECT_EQ(tree->depths, (llvm::SmallVector<int64_t, 4>{0, 2, 3, 1}));
}

TEST_F(CollectiveTopologyTest,
       RejectsDuplicateRanksAndUnboundedExactCandidates) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 17>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 17>,
       policy = "all_available", endpoints = array<i64>}
}
)mlir");
  ASSERT_TRUE(module);
  const int64_t duplicateRanks[] = {0, 0};
  EXPECT_TRUE(mlir::failed(wafer::analysis::buildMinimumHopCollectiveRingOrder(
      module->getOperation(), duplicateRanks)));
  llvm::SmallVector<int64_t, 17> ranks;
  for (int64_t rank = 0; rank < 17; ++rank)
    ranks.push_back(rank);
  EXPECT_TRUE(mlir::failed(wafer::analysis::buildMinimumHopCollectiveRingOrder(
      module->getOperation(), ranks)));
  EXPECT_TRUE(mlir::failed(wafer::analysis::buildMinimumHopCollectiveTree(
      module->getOperation(), ranks)));
}

} // namespace
