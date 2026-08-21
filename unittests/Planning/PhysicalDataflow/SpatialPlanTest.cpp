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
  auto problem = SpatialPlanningProblem::create(
      {NodeIterationSpace{root(1), {5, 4}}, NodeIterationSpace{root(0), {}}},
      {TileId(5), TileId(0), TileId(7), TileId(2)});
  EXPECT_TRUE(mlir::succeeded(problem));
  return std::move(*problem);
}

TEST(SpatialPlanTest, ClosesCanonicalPlanIntoExactCartesianAssignment) {
  SpatialPlanningProblem problem = makeProblem();
  EXPECT_EQ(problem.getNodes()[0].root, root(0));
  EXPECT_EQ(problem.getNodes()[1].root, root(1));
  EXPECT_TRUE(llvm::equal(
      problem.getAvailableTiles(),
      llvm::ArrayRef<TileId>{TileId(0), TileId(2), TileId(5), TileId(7)}));

  ReductionGroupId group{root(1), 0, {0}};
  SpatialPlan plan;
  plan.nodes.push_back(NodeSpatialPlan{root(0), {}, {TileId(7)}, {}});
  plan.nodes.push_back(
      NodeSpatialPlan{root(1),
                      {{0, IteratorPartitionScheme::BalancedParts, 2},
                       {1, IteratorPartitionScheme::UniformExtent, 3}},
                      {TileId(7), TileId(0), TileId(5), TileId(2)},
                      {{group, TileId(2)}}});

  std::string failureReason;
  auto assignment = closeSpatialPlanStructure(problem, plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
  ASSERT_EQ(assignment->nodes.size(), 2u);

  const NodeExecutionPartition &scalar = assignment->nodes[0];
  ASSERT_EQ(scalar.shards.size(), 1u);
  EXPECT_TRUE(scalar.shards[0].shard.coordinate.empty());
  EXPECT_TRUE(scalar.shards[0].iterationDomain.empty());
  EXPECT_EQ(scalar.shards[0].tile, TileId(7));

  const NodeExecutionPartition &matrix = assignment->nodes[1];
  ASSERT_EQ(matrix.shards.size(), 4u);
  EXPECT_TRUE(llvm::equal(matrix.shards[0].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{0, 0}));
  EXPECT_TRUE(llvm::equal(matrix.shards[0].iterationDomain,
                          llvm::ArrayRef<IteratorInterval>{{0, 3}, {0, 3}}));
  EXPECT_TRUE(llvm::equal(matrix.shards[1].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{0, 1}));
  EXPECT_TRUE(llvm::equal(matrix.shards[1].iterationDomain,
                          llvm::ArrayRef<IteratorInterval>{{0, 3}, {3, 1}}));
  EXPECT_TRUE(llvm::equal(matrix.shards[2].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{1, 0}));
  EXPECT_TRUE(llvm::equal(matrix.shards[2].iterationDomain,
                          llvm::ArrayRef<IteratorInterval>{{3, 2}, {0, 3}}));
  EXPECT_TRUE(llvm::equal(matrix.shards[3].shard.coordinate,
                          llvm::ArrayRef<uint32_t>{1, 1}));
  EXPECT_TRUE(llvm::equal(matrix.shards[3].iterationDomain,
                          llvm::ArrayRef<IteratorInterval>{{3, 2}, {3, 1}}));
  EXPECT_EQ(matrix.shards[0].tile, TileId(7));
  EXPECT_EQ(matrix.shards[1].tile, TileId(0));
  EXPECT_EQ(matrix.shards[2].tile, TileId(5));
  EXPECT_EQ(matrix.shards[3].tile, TileId(2));
  ASSERT_EQ(matrix.reductionGroups.size(), 1u);
  EXPECT_EQ(matrix.reductionGroups[0].group, group);
  EXPECT_EQ(matrix.reductionGroups[0].mergeTile, TileId(2));
  EXPECT_TRUE(mlir::succeeded(
      validateSpatialAssignmentStructure(problem, *assignment, &failureReason)))
      << failureReason;
}

TEST(SpatialPlanTest, RejectsInvalidOrNonCanonicalSpatialPlans) {
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
      {NodeIterationSpace{second, {2}}, NodeIterationSpace{first, {2}}},
      {TileId(0)});
  ASSERT_TRUE(mlir::succeeded(problem));
  EXPECT_EQ(problem->getNodes()[0].root, first);
  EXPECT_EQ(problem->getNodes()[1].root, second);

  SemanticRootKey unknown = root(1);
  unknown.anchorKind = static_cast<SemanticRootAnchorKind>(255);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(SpatialPlanningProblem::create(
      {NodeIterationSpace{unknown, {2}}}, {TileId(0)}, &failureReason)));
  EXPECT_NE(failureReason.find("unknown anchor"), std::string::npos);
}

} // namespace
