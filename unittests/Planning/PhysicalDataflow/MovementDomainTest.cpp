//===- MovementDomainTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeDomain(llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  StaticRectangularIndexSet box{offsets, llvm::SmallVector<int64_t, 4>(sizes)};
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
}

struct Fixture {
  CanonicalMovementCoordinate movement;
  RepresentationPlan representations;
  DDRBoundaryTransferId transfer;
  PhysicalVersionId selectedDestination;
};

Fixture makeFixture(bool derivedDestination, llvm::ArrayRef<int64_t> sizes) {
  SemanticRootKey sourceRoot;
  SemanticRootKey destinationRoot;
  destinationRoot.anchorIndex = 1;
  RootRegionWorkId sourceWork{sourceRoot, TileId(0)};
  RootRegionWorkId destinationWork{destinationRoot, TileId(3)};
  LogicalShardId sourceShard{sourceRoot, {0}};
  LogicalShardId destinationShard{destinationRoot, {0}};
  ExecutionResultValueId sourceLogical{
      ExecutionInstanceId{RequiredRootExecution{sourceWork, sourceShard}}, 0};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::StructuredResult;
  fragment.source.semantic = sourceRoot;
  fragment.ownerShard = sourceShard;
  fragment.ownerTile = TileId(0);
  fragment.use = {0, destinationShard};
  BoundaryRegionValueId destinationLogical{destinationWork, fragment};
  PhysicalVersionId source{RegionValueVersionId(sourceLogical)};
  PhysicalVersionId destination{RegionValueVersionId(destinationLogical)};
  RepresentationUseId destinationUse =
      BoundaryRepresentationUseId{destinationLogical};

  RepresentationPlan representations;
  representations.logicalValues = {{sourceLogical, source},
                                   {destinationLogical, destination}};
  representations.physicalVersions = {{source, MemLayout::Tensor},
                                      {destination, MemLayout::Tensor}};
  PhysicalVersionId selectedDestination = destination;
  if (derivedDestination) {
    selectedDestination.derivation.push_back(
        {PhysicalVersionDerivationKind::LayoutConversion, MemLayout::Tensor,
         MemLayout::NTensor, SharedRepresentationAnchor{}});
    representations.physicalVersions.push_back(
        {selectedDestination, MemLayout::NTensor});
  }
  representations.uses.push_back({destinationUse, selectedDestination});
  llvm::sort(representations.logicalValues);
  llvm::sort(representations.physicalVersions);
  llvm::sort(representations.uses);

  DDRBoundaryTransferId transfer{destinationLogical};
  CanonicalMovementCoordinate movement;
  movement.plan.ddrTransfers.push_back(
      {transfer, source, destination, MovementRealization{}});
  movement.resources.push_back({MovementActionId(transfer), makeDomain(sizes),
                                mlir::Type{}, TileId(0), TileId(3)});
  return {std::move(movement), std::move(representations), transfer,
          selectedDestination};
}

std::optional<std::vector<MovementPlan>> enumerate(const MovementDomain &domain,
                                                   size_t limit = 10000) {
  std::vector<MovementPlan> plans;
  MovementSuccessor next = domain.getFirstPlan();
  while (next.getKind() == MovementSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() || plans.size() >= limit ||
        !domain.contains(*next.getPlan()))
      return std::nullopt;
    plans.push_back(*next.getPlan());
    MovementCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == MovementSuccessorKind::End
             ? std::optional<std::vector<MovementPlan>>(std::move(plans))
             : std::nullopt;
}

TEST(MovementDomainTest, CrossTileDomainMatchesEverySimpleRelayPermutation) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(false, {2, 1025, 128});
  fixture.movement.resources.front().elementType =
      mlir::Float16Type::get(&context);
  MovementDomainResult result =
      buildMovementDomain(fixture.movement, fixture.representations,
                          {TileId(3), TileId(2), TileId(1), TileId(0)});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 6u);
  std::set<std::vector<int64_t>> paths;
  for (const MovementPlan &plan : *plans) {
    ASSERT_EQ(plan.ddrTransfers.size(), 1u);
    const MovementRealization &realization =
        plan.ddrTransfers.front().realization;
    std::vector<int64_t> path;
    path.push_back(static_cast<int64_t>(realization.kind));
    for (const MovementHop &hop : realization.hops)
      path.push_back(hop.destination.getValue());
    paths.insert(std::move(path));
  }
  EXPECT_TRUE(
      paths.count({static_cast<int64_t>(MovementRealizationKind::DDRStage)}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(MovementRealizationKind::TargetRoutedPeer), 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(MovementRealizationKind::SoftwareRelay), 1, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(MovementRealizationKind::SoftwareRelay), 2, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(MovementRealizationKind::SoftwareRelay), 1, 2, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(MovementRealizationKind::SoftwareRelay), 2, 1, 3}));

  MovementDomainResult reordered =
      buildMovementDomain(fixture.movement, fixture.representations,
                          {TileId(0), TileId(1), TileId(2), TileId(3)});
  ASSERT_TRUE(reordered.succeeded());
  auto reorderedPlans = enumerate(*reordered.domain);
  ASSERT_TRUE(reorderedPlans);
  EXPECT_EQ(*reorderedPlans, *plans);
}

TEST(MovementDomainTest, MismatchedSelectedDestinationKeepsOnlyDDRRealization) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(true, {2, 1031, 128});
  fixture.movement.resources.front().elementType =
      mlir::Float16Type::get(&context);
  MovementDomainResult result =
      buildMovementDomain(fixture.movement, fixture.representations,
                          {TileId(0), TileId(1), TileId(2), TileId(3)});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 1u);
  for (const MovementPlan &plan : *plans)
    EXPECT_EQ(plan.ddrTransfers.front().destination,
              fixture.selectedDestination);
  EXPECT_EQ(plans->front().ddrTransfers.front().realization.kind,
            MovementRealizationKind::DDRStage);
}

TEST(MovementDomainTest, SameTileAndMalformedEndpointsFailClosed) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(false, {});
  fixture.movement.resources.front().elementType =
      mlir::Float16Type::get(&context);
  fixture.movement.resources.front().destinationTile = TileId(0);
  MovementDomainResult sameTile = buildMovementDomain(
      fixture.movement, fixture.representations, {TileId(0), TileId(1)});
  ASSERT_TRUE(sameTile.succeeded());
  auto plans = enumerate(*sameTile.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_EQ(plans->front().ddrTransfers.front().realization.kind,
            MovementRealizationKind::DDRStage);

  MovementDomainResult unavailable = buildMovementDomain(
      fixture.movement, fixture.representations, {TileId(1)});
  ASSERT_FALSE(unavailable.succeeded());
  ASSERT_TRUE(unavailable.failure);
  EXPECT_EQ(unavailable.failure->kind,
            MovementDomainFailureKind::BrokenContract);
}

TEST(MovementDomainTest,
     RemoteReductionPayloadUsesTheSameDDRDirectRelayDomain) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  SemanticRootKey root;
  RootRegionWorkId sourceWork{root, TileId(0)};
  RootRegionWorkId mergeWork{root, TileId(3)};
  LogicalShardId shard{root, {0}};
  ReductionGroupId group{root, 0, {0}};
  ExecutionInstanceId sourceExecution{RequiredRootExecution{sourceWork, shard}};
  ReductionPartialValueId logical{sourceExecution, group, 0};
  PhysicalVersionId source{RegionValueVersionId(logical)};
  RepresentationPlan representations;
  representations.logicalValues.push_back({logical, source});
  representations.physicalVersions.push_back({source, MemLayout::Tensor});
  ReductionGatherId gatherId{group, shard, logical};
  CanonicalMovementCoordinate movement;
  movement.plan.reductionGathers.push_back(
      {gatherId, source,
       ExecutionInstanceId{RequiredMergeExecution{mergeWork, group}},
       MovementRealization{}});
  movement.resources.push_back(
      {MovementActionId(gatherId), makeDomain({2, 1025, 128}),
       mlir::Float16Type::get(&context), TileId(0), TileId(3)});
  MovementDomainResult result = buildMovementDomain(
      movement, representations, {TileId(0), TileId(1), TileId(2), TileId(3)});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 6u);
  for (const MovementPlan &plan : *plans) {
    ASSERT_EQ(plan.reductionGathers.size(), 1u);
    EXPECT_EQ(plan.reductionGathers.front().id, gatherId);
  }
}

} // namespace
