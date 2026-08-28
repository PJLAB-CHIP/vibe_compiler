//===- TemporalDomainTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "gtest/gtest.h"

#include <set>
#include <tuple>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

TemporalScopeId makeScope(uint32_t rootIndex, int64_t tile) {
  SemanticRootKey root;
  root.anchorIndex = rootIndex;
  analysis::RootRegionWorkId work{root, TileId(tile)};
  LogicalShardId shard{root, {rootIndex}};
  ExecutionInstanceId execution{RequiredRootExecution{work, shard}};
  return TemporalScopeId{RegionExecutionId{execution}};
}

TemporalScopeDescriptor makeDescriptor(TemporalScopeId id,
                                       llvm::ArrayRef<int64_t> offsets,
                                       llvm::ArrayRef<int64_t> extents) {
  TemporalScopeDescriptor descriptor;
  descriptor.id = std::move(id);
  descriptor.iterationOffsets.assign(offsets.begin(), offsets.end());
  descriptor.iterationExtents.assign(extents.begin(), extents.end());
  descriptor.iteratorCapabilities.assign(extents.size(),
                                         IteratorTilingCapability::Tileable);
  return descriptor;
}

std::optional<std::vector<TemporalPlan>> enumerate(const TemporalDomain &domain,
                                                   size_t limit = 100000) {
  std::vector<TemporalPlan> plans;
  TemporalSuccessor current = domain.getFirstPlan();
  while (current.getKind() == TemporalSuccessorKind::Plan) {
    if (!current.getPlan() || !current.getCursor() || plans.size() >= limit)
      return std::nullopt;
    plans.push_back(*current.getPlan());
    current = domain.getNextPlan(*current.getCursor());
  }
  return current.getKind() == TemporalSuccessorKind::End
             ? std::optional<std::vector<TemporalPlan>>(std::move(plans))
             : std::nullopt;
}

TEST(TemporalDomainTest,
     EveryPositiveSizeAndActiveOrderMatchesIndependentOracle) {
  TemporalScopeDescriptor descriptor =
      makeDescriptor(makeScope(0, 0), {17, 29}, {2, 3});
  TemporalDomainResult built = buildTemporalDomain({descriptor});
  ASSERT_TRUE(built.succeeded())
      << (built.failure ? built.failure->detail : "");
  auto plans = enumerate(*built.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 8u);

  std::set<std::pair<std::vector<int64_t>, std::vector<uint32_t>>> observed;
  for (const TemporalPlan &plan : *plans) {
    ASSERT_EQ(plan.scopes.size(), 1u);
    const TemporalScopePlan &scope = plan.scopes.front();
    EXPECT_TRUE(built.domain->contains(plan));
    EXPECT_TRUE(
        observed
            .insert({std::vector<int64_t>(scope.iteratorTileSizes.begin(),
                                          scope.iteratorTileSizes.end()),
                     std::vector<uint32_t>(scope.waveLoopOrder.begin(),
                                           scope.waveLoopOrder.end())})
            .second);
  }
}

TEST(TemporalDomainTest, PrecedenceDiamondEnumeratesOnlyLinearExtensions) {
  TemporalScopeDescriptor descriptor =
      makeDescriptor(makeScope(0, 0), {0, 0, 0, 0}, {2, 2, 2, 2});
  descriptor.precedence = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
  TemporalDomainResult built = buildTemporalDomain({descriptor});
  ASSERT_TRUE(built.succeeded());
  TemporalPlan prefix;
  prefix.scopes.push_back({descriptor.id, {1, 1, 1, 1}, {0, 1, 2, 3}});
  TemporalSuccessor first = built.domain->completePrefix(prefix);
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  EXPECT_TRUE(built.domain->contains(*first.getPlan()));
  EXPECT_EQ(first.getPlan()->scopes.front().waveLoopOrder,
            (llvm::SmallVector<uint32_t, 4>{0, 1, 2, 3}));

  TemporalPlan secondPrefix = prefix;
  secondPrefix.scopes.front().waveLoopOrder = {0, 2, 1, 3};
  EXPECT_TRUE(built.domain->contains(secondPrefix));
  TemporalPlan invalid = prefix;
  invalid.scopes.front().waveLoopOrder = {1, 0, 2, 3};
  EXPECT_FALSE(built.domain->contains(invalid));
}

TEST(TemporalDomainTest,
     RankSixRealScaleChoiceKeepsAllAxesInOneExplicitVector) {
  TemporalScopeDescriptor descriptor = makeDescriptor(
      makeScope(0, 0), {0, 0, 0, 0, 0, 0}, {2, 4, 1024, 64, 1031, 128});
  TemporalDomainResult built = buildTemporalDomain({descriptor});
  ASSERT_TRUE(built.succeeded());
  TemporalSuccessor first = built.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_EQ(first.getPlan()->scopes.size(), 1u);
  EXPECT_EQ(first.getPlan()->scopes.front().iteratorTileSizes,
            descriptor.iterationExtents);

  TemporalPlan prefix = *first.getPlan();
  prefix.scopes.front().iteratorTileSizes = {2, 4, 128, 64, 128, 128};
  auto order = buildFirstTemporalWaveLoopOrder(
      descriptor.iterationExtents, prefix.scopes.front().iteratorTileSizes);
  ASSERT_TRUE(mlir::succeeded(order));
  prefix.scopes.front().waveLoopOrder = *order;
  EXPECT_TRUE(built.domain->contains(prefix));
}

TEST(TemporalDomainTest, AxisWavesCoverAlignedRaggedAndNonzeroOffsetExactly) {
  for (auto [extent, expectedCount, expectedTail] :
       {std::tuple<int64_t, size_t, int64_t>{1024, 8, 128},
        std::tuple<int64_t, size_t, int64_t>{1025, 9, 1},
        std::tuple<int64_t, size_t, int64_t>{1031, 9, 7}}) {
    SCOPED_TRACE(extent);
    auto waves = buildTemporalAxisWaves({17, extent}, 128);
    ASSERT_TRUE(mlir::succeeded(waves));
    ASSERT_EQ(waves->size(), expectedCount);
    EXPECT_EQ(waves->front().offset, 17);
    EXPECT_EQ(waves->back().size, expectedTail);
    int64_t next = 17;
    int64_t covered = 0;
    for (const IteratorInterval &wave : *waves) {
      EXPECT_EQ(wave.offset, next);
      next += wave.size;
      covered += wave.size;
    }
    EXPECT_EQ(covered, extent);
  }
}

TEST(TemporalDomainTest, MalformedRanksAndPrecedenceCyclesFailTyped) {
  TemporalScopeDescriptor malformed =
      makeDescriptor(makeScope(0, 0), {0, 0}, {1024, 128});
  malformed.iteratorCapabilities.pop_back();
  TemporalDomainResult rankFailure = buildTemporalDomain({malformed});
  ASSERT_FALSE(rankFailure.succeeded());
  ASSERT_TRUE(rankFailure.failure);
  EXPECT_EQ(rankFailure.failure->kind,
            TemporalDomainFailureKind::BrokenContract);

  TemporalScopeDescriptor cyclic =
      makeDescriptor(makeScope(1, 0), {0, 0}, {1025, 128});
  cyclic.precedence = {{0, 1}, {1, 0}};
  TemporalDomainResult cycleFailure = buildTemporalDomain({cyclic});
  ASSERT_FALSE(cycleFailure.succeeded());
  ASSERT_TRUE(cycleFailure.failure);
  EXPECT_EQ(cycleFailure.failure->kind,
            TemporalDomainFailureKind::BrokenContract);
}

TEST(TemporalDomainTest, IntervalSplitPreservesEveryPoint) {
  auto split = splitTemporalSizeInterval({1, 1031}, 128);
  ASSERT_TRUE(mlir::succeeded(split));
  EXPECT_EQ(split->singleton, (TemporalSizeInterval{128, 128}));
  ASSERT_TRUE(split->above);
  ASSERT_TRUE(split->below);
  EXPECT_EQ(*split->above, (TemporalSizeInterval{129, 1031}));
  EXPECT_EQ(*split->below, (TemporalSizeInterval{1, 127}));
}

} // namespace
