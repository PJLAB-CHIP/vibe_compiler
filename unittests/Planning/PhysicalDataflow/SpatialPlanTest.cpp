//===- SpatialPlanTest.cpp ----------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

namespace {

using wafer::TileId;
using namespace wafer::compiler::detail;

SemanticRootKey root(uint32_t result) {
  SemanticRootKey key;
  key.anchorIndex = result;
  return key;
}

SpatialPlanningProblem makeProblem() {
  llvm::SmallVector<TileId, 16> tiles;
  for (int64_t tile = 15; tile >= 0; --tile)
    tiles.push_back(TileId(tile));
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(1), {2, 1024, 128}},
       NodeIterationSpace{root(0), {}}},
      tiles);
  EXPECT_TRUE(mlir::succeeded(problem));
  return std::move(*problem);
}

llvm::SmallVector<TileId, 16> allTiles() {
  llvm::SmallVector<TileId, 16> tiles;
  for (int64_t tile = 0; tile < 16; ++tile)
    tiles.push_back(TileId(tile));
  return tiles;
}

TEST(SpatialPlanTest, ClosesCanonicalPlanIntoExactCartesianAssignment) {
  SpatialPlanningProblem problem = makeProblem();
  EXPECT_EQ(problem.getNodes()[0].root, root(0));
  EXPECT_EQ(problem.getNodes()[1].root, root(1));
  EXPECT_TRUE(llvm::equal(problem.getAvailableTiles(), allTiles()));

  ReductionGroupId group{root(1), 0, {0, 0}};
  SpatialPlan plan;
  plan.nodes.push_back(NodeSpatialPlan{root(0), {}, {TileId(15)}, {}});
  plan.nodes.push_back(
      NodeSpatialPlan{root(1),
                      {{0, IteratorPartitionScheme::BalancedParts, 2},
                       {1, IteratorPartitionScheme::BalancedParts, 8},
                       {2, IteratorPartitionScheme::BalancedParts, 1}},
                      allTiles(),
                      {{group, TileId(15)}}});

  std::string failureReason;
  auto assignment = closeSpatialPlanStructure(problem, plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
  ASSERT_EQ(assignment->nodes.size(), 2u);

  const NodeExecutionPartition &scalar = assignment->nodes[0];
  ASSERT_EQ(scalar.shards.size(), 1u);
  EXPECT_TRUE(scalar.shards[0].shard.coordinate.empty());
  EXPECT_TRUE(scalar.shards[0].iterationDomain.empty());
  EXPECT_EQ(scalar.shards[0].tile, TileId(15));

  const NodeExecutionPartition &matrix = assignment->nodes[1];
  ASSERT_EQ(matrix.shards.size(), 16u);
  EXPECT_TRUE(llvm::equal(matrix.shards[0].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{0, 0, 0}));
  EXPECT_TRUE(llvm::equal(
      matrix.shards[0].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{0, 1}, {0, 128}, {0, 128}}));
  EXPECT_TRUE(llvm::equal(matrix.shards[1].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{0, 1, 0}));
  EXPECT_TRUE(llvm::equal(
      matrix.shards[1].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{0, 1}, {128, 128}, {0, 128}}));
  EXPECT_TRUE(llvm::equal(matrix.shards[8].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{1, 0, 0}));
  EXPECT_TRUE(llvm::equal(
      matrix.shards[15].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{1, 1}, {896, 128}, {0, 128}}));
  EXPECT_EQ(matrix.shards[0].tile, TileId(0));
  EXPECT_EQ(matrix.shards[15].tile, TileId(15));
  ASSERT_EQ(matrix.reductionGroups.size(), 1u);
  EXPECT_EQ(matrix.reductionGroups[0].group, group);
  EXPECT_EQ(matrix.reductionGroups[0].mergeTile, TileId(15));
  EXPECT_TRUE(mlir::succeeded(
      validateSpatialAssignmentStructure(problem, *assignment, &failureReason)))
      << failureReason;
}

TEST(SpatialPlanTest, ClosesRaggedRankThreeAssignmentWithoutCoverageLoss) {
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {2, 1025, 127}}}, allTiles());
  ASSERT_TRUE(mlir::succeeded(problem));
  SpatialPlan plan{{{root(0),
                     {{0, IteratorPartitionScheme::BalancedParts, 2},
                      {1, IteratorPartitionScheme::BalancedParts, 8},
                      {2, IteratorPartitionScheme::BalancedParts, 1}},
                     allTiles(),
                     {}}}};
  std::string failureReason;
  auto assignment = closeSpatialPlanStructure(*problem, plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
  ASSERT_EQ(assignment->nodes.size(), 1u);
  const NodeExecutionPartition &node = assignment->nodes.front();
  ASSERT_EQ(node.shards.size(), 16u);
  EXPECT_TRUE(llvm::equal(
      node.shards[0].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{0, 1}, {0, 129}, {0, 127}}));
  EXPECT_TRUE(llvm::equal(
      node.shards[1].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{0, 1}, {129, 128}, {0, 127}}));
  EXPECT_TRUE(llvm::equal(
      node.shards[15].iterationDomain,
      llvm::ArrayRef<IteratorInterval>{{1, 1}, {897, 128}, {0, 127}}));
  int64_t coveredElements = 0;
  for (const ExecutionShard &shard : node.shards) {
    int64_t elements = 1;
    for (const IteratorInterval &interval : shard.iterationDomain)
      elements *= interval.size;
    coveredElements += elements;
  }
  EXPECT_EQ(coveredElements, int64_t{2} * 1025 * 127);
  EXPECT_TRUE(mlir::succeeded(validateSpatialAssignmentStructure(
      *problem, *assignment, &failureReason)))
      << failureReason;
}

TEST(SpatialPlanTest, RejectsInvalidOrNonCanonicalSpatialPlans) {
  // Small domains are intentional in these single-fault structural negatives.
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {4}}}, {TileId(0), TileId(2)});
  ASSERT_TRUE(mlir::succeeded(problem));

  auto expectFailure = [&](SpatialPlan plan, llvm::StringRef expected) {
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(
        validateSpatialPlanStructure(*problem, plan, &failureReason)));
    EXPECT_NE(failureReason.find(expected.str()), std::string::npos)
        << failureReason;
  };

  expectFailure({{{root(0), {}, {TileId(0)}, {}}}}, "every iterator");
  expectFailure({{{root(0),
                   {{0, IteratorPartitionScheme::UniformExtent, 2}},
                   {TileId(0), TileId(2)},
                   {}}}},
                "duplicates canonical");
  expectFailure({{{root(0),
                   {{0, IteratorPartitionScheme::BalancedParts, 2}},
                   {TileId(0), TileId(0)},
                   {}}}},
                "injective");
  expectFailure({{{root(0),
                   {{0, IteratorPartitionScheme::BalancedParts, 2}},
                   {TileId(0), TileId(7)},
                   {}}}},
                "unavailable");
  expectFailure({{{root(0),
                   {{1, IteratorPartitionScheme::BalancedParts, 2}},
                   {TileId(0), TileId(2)},
                   {}}}},
                "must be ordered");
  expectFailure({{{root(0),
                   {{0, IteratorPartitionScheme::BalancedParts, 0}},
                   {TileId(0)},
                   {}}}},
                "outside iterator extent");
  expectFailure({{{root(0),
                   {{0, IteratorPartitionScheme::UniformExtent, 1}},
                   {TileId(0), TileId(2)},
                   {}}}},
                "exceeds available Tiles");
}

TEST(SpatialPlanTest, RejectsMalformedClosedAssignments) {
  // Compact domains isolate one malformed assignment field at a time; the
  // positive coverage and tail paths use representative all-16 Tile shapes.
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {4}}}, {TileId(0), TileId(2)});
  ASSERT_TRUE(mlir::succeeded(problem));
  SpatialPlan plan{{{root(0),
                     {{0, IteratorPartitionScheme::BalancedParts, 2}},
                     {TileId(0), TileId(2)},
                     {}}}};
  auto assignment = closeSpatialPlanStructure(*problem, plan);
  ASSERT_TRUE(mlir::succeeded(assignment));

  auto expectFailure = [&](SpatialAssignment broken, llvm::StringRef expected) {
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(
        validateSpatialAssignmentStructure(*problem, broken, &failureReason)));
    EXPECT_NE(failureReason.find(expected.str()), std::string::npos)
        << failureReason;
  };

  SpatialAssignment overlapping = *assignment;
  overlapping.nodes[0].shards[1].iterationDomain[0].offset = 1;
  expectFailure(std::move(overlapping), "dense and contiguous");

  SpatialAssignment duplicateTile = *assignment;
  duplicateTile.nodes[0].shards[1].tile = TileId(0);
  expectFailure(std::move(duplicateTile), "injective");

  SpatialAssignment missingCell = *assignment;
  missingCell.nodes[0].shards.pop_back();
  expectFailure(std::move(missingCell), "full extent");

  SpatialAssignment duplicateCoordinate = *assignment;
  duplicateCoordinate.nodes[0].shards[1].shard.coordinate = {0};
  expectFailure(std::move(duplicateCoordinate), "duplicate logical");
}

TEST(SpatialPlanTest, RejectsDuplicateProblemFactsAndMergeGroups) {
  // These are minimal single-contract negatives, not positive shape coverage.
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {0}}}, {TileId(0)}, &failureReason)));
  EXPECT_NE(failureReason.find("non-positive iterator extent"),
            std::string::npos);

  EXPECT_TRUE(mlir::failed(SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {2}}}, {TileId(-1)}, &failureReason)));
  EXPECT_NE(failureReason.find("Tile identity is negative"), std::string::npos);

  EXPECT_TRUE(mlir::failed(SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {2}}, NodeIterationSpace{root(0), {2}}},
      {TileId(0)}, &failureReason)));
  EXPECT_NE(failureReason.find("duplicate root"), std::string::npos);

  EXPECT_TRUE(mlir::failed(
      SpatialPlanningProblem::create({NodeIterationSpace{root(0), {2}}},
                                     {TileId(0), TileId(0)}, &failureReason)));
  EXPECT_NE(failureReason.find("duplicate Tile"), std::string::npos);

  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(0), {2}}}, {TileId(0), TileId(2)});
  ASSERT_TRUE(mlir::succeeded(problem));
  ReductionGroupId group{root(0), 0, {}};
  SpatialPlan duplicateGroups{
      {{root(0),
        {{0, IteratorPartitionScheme::BalancedParts, 1}},
        {TileId(0)},
        {{group, TileId(0)}, {group, TileId(2)}}}}};
  EXPECT_TRUE(mlir::failed(
      validateSpatialPlanStructure(*problem, duplicateGroups, &failureReason)));
  EXPECT_NE(failureReason.find("unique and sorted"), std::string::npos);

  SpatialPlan wrongRootGroup = duplicateGroups;
  wrongRootGroup.nodes[0].reductionMerges.resize(1);
  wrongRootGroup.nodes[0].reductionMerges[0].group.root = root(1);
  EXPECT_TRUE(mlir::failed(
      validateSpatialPlanStructure(*problem, wrongRootGroup, &failureReason)));
  EXPECT_NE(failureReason.find("another semantic root"), std::string::npos);
}

TEST(SpatialPlanTest, CanonicalizesTypedSemanticRootPaths) {
  SemanticRootKey first = root(0);
  first.path.push_back({SemanticRootPathRelation::SSAUseDef, 0, 1});
  SemanticRootKey second = root(0);
  second.path.push_back({SemanticRootPathRelation::SSAUseDef, 0, 2});
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{second, {2, 1024, 128}},
       NodeIterationSpace{first, {2, 1024, 128}}},
      {TileId(0)});
  ASSERT_TRUE(mlir::succeeded(problem));
  EXPECT_EQ(problem->getNodes()[0].root, first);
  EXPECT_EQ(problem->getNodes()[1].root, second);

  SemanticRootKey unknown = root(1);
  unknown.anchorKind = static_cast<SemanticRootAnchorKind>(255);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(SpatialPlanningProblem::create(
      {NodeIterationSpace{unknown, {2, 1024, 128}}}, {TileId(0)},
      &failureReason)));
  EXPECT_NE(failureReason.find("unknown anchor"), std::string::npos);
}

} // namespace
