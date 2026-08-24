//===- MovementDomainTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
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

Fixture makeFixture(bool derivedDestination, llvm::ArrayRef<int64_t> sizes,
                    TileId destinationTile = TileId(3)) {
  SemanticRootKey sourceRoot;
  SemanticRootKey destinationRoot;
  destinationRoot.anchorIndex = 1;
  RootRegionWorkId sourceWork{sourceRoot, TileId(0)};
  RootRegionWorkId destinationWork{destinationRoot, destinationTile};
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
  movement.plan.ddrTransfers.push_back({transfer, source, destination});
  movement.resources.push_back({MovementActionId(transfer), makeDomain(sizes),
                                mlir::Type{}, TileId(0), destinationTile});
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

const PeerTransferGraphPlan *findPeerGraph(const MovementPlan &plan,
                                           const MovementActionId &action) {
  auto graph = llvm::find_if(plan.peerGraphs, [&](const auto &candidate) {
    return llvm::is_contained(candidate.actions, action);
  });
  return graph == plan.peerGraphs.end() ? nullptr : &*graph;
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
  constexpr int64_t ddrPath = -1;
  for (const MovementPlan &plan : *plans) {
    ASSERT_EQ(plan.ddrTransfers.size(), 1u);
    const PeerTransferGraphPlan *realization =
        findPeerGraph(plan, MovementActionId(plan.ddrTransfers.front().id));
    std::vector<int64_t> path;
    path.push_back(realization ? static_cast<int64_t>(realization->kind)
                               : ddrPath);
    if (realization) {
      std::map<int64_t, int64_t> parent;
      for (const MovementHop &hop : realization->hops)
        parent.emplace(hop.destination.getValue(), hop.source.getValue());
      std::vector<int64_t> reversePath;
      int64_t current = 3;
      while (current != 0) {
        reversePath.push_back(current);
        auto found = parent.find(current);
        ASSERT_NE(found, parent.end());
        current = found->second;
      }
      std::reverse(reversePath.begin(), reversePath.end());
      path.insert(path.end(), reversePath.begin(), reversePath.end());
    }
    paths.insert(std::move(path));
  }
  EXPECT_TRUE(paths.count({ddrPath}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(PeerTransferGraphKind::TargetRoutedPeer), 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(PeerTransferGraphKind::SoftwareRelay), 1, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(PeerTransferGraphKind::SoftwareRelay), 2, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(PeerTransferGraphKind::SoftwareRelay), 1, 2, 3}));
  EXPECT_TRUE(paths.count(
      {static_cast<int64_t>(PeerTransferGraphKind::SoftwareRelay), 2, 1, 3}));

  MovementDomainResult reordered =
      buildMovementDomain(fixture.movement, fixture.representations,
                          {TileId(0), TileId(1), TileId(2), TileId(3)});
  ASSERT_TRUE(reordered.succeeded());
  auto reorderedPlans = enumerate(*reordered.domain);
  ASSERT_TRUE(reorderedPlans);
  EXPECT_EQ(*reorderedPlans, *plans);
}

TEST(MovementDomainTest,
     FanoutEnumeratesSplitPartialAndMaximalArborescencesAtRealScale) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (int64_t major : {1024, 1025}) {
    SemanticRootKey sourceRoot;
    RootRegionWorkId sourceWork{sourceRoot, TileId(0)};
    LogicalShardId sourceShard{sourceRoot, {0}};
    ExecutionResultValueId sourceLogical{
        ExecutionInstanceId{RequiredRootExecution{sourceWork, sourceShard}}, 0};
    PhysicalVersionId source{RegionValueVersionId(sourceLogical)};
    RepresentationPlan representations;
    representations.logicalValues.push_back({sourceLogical, source});
    representations.physicalVersions.push_back({source, MemLayout::Tensor});
    CanonicalMovementCoordinate movement;
    std::vector<MovementActionId> actionIds;
    for (int64_t tile = 1; tile <= 3; ++tile) {
      SemanticRootKey destinationRoot;
      destinationRoot.anchorIndex = static_cast<uint64_t>(tile);
      RootRegionWorkId destinationWork{destinationRoot, TileId(tile)};
      LogicalShardId destinationShard{destinationRoot, {0}};
      DemandFragmentId fragment;
      fragment.source.kind = RootBoundaryKind::StructuredResult;
      fragment.source.semantic = sourceRoot;
      fragment.ownerShard = sourceShard;
      fragment.ownerTile = TileId(0);
      fragment.use = {0, destinationShard};
      BoundaryRegionValueId destinationLogical{destinationWork, fragment};
      PhysicalVersionId destination{RegionValueVersionId(destinationLogical)};
      representations.logicalValues.push_back(
          {destinationLogical, destination});
      representations.physicalVersions.push_back(
          {destination, MemLayout::Tensor});
      representations.uses.push_back(
          {BoundaryRepresentationUseId{destinationLogical}, destination});
      DDRBoundaryTransferId transfer{destinationLogical};
      movement.plan.ddrTransfers.push_back({transfer, source, destination});
      actionIds.push_back(MovementActionId(transfer));
      movement.resources.push_back(
          {MovementActionId(transfer), makeDomain({2, major, 128}),
           mlir::Float16Type::get(&context), TileId(0), TileId(tile)});
    }
    llvm::sort(representations.logicalValues);
    llvm::sort(representations.physicalVersions);
    llvm::sort(representations.uses);
    llvm::sort(movement.plan.ddrTransfers);
    llvm::sort(movement.resources, [](const auto &lhs, const auto &rhs) {
      return lhs.action < rhs.action;
    });
    llvm::sort(actionIds);

    MovementDomainResult result =
        buildMovementDomain(movement, representations,
                            {TileId(3), TileId(1), TileId(0), TileId(2)});
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    std::vector<MovementPlan> proposals = result.domain->getProposals();
    ASSERT_FALSE(proposals.empty());
    for (const MovementPlan &proposal : proposals)
      EXPECT_TRUE(result.domain->contains(proposal));
    EXPECT_TRUE(llvm::any_of(
        proposals.front().peerGraphs, [&](const PeerTransferGraphPlan &graph) {
          return graph.actions == actionIds &&
                 graph.kind == PeerTransferGraphKind::SoftwareFanout;
        }));
    auto plans = enumerate(*result.domain, /*limit=*/20000);
    ASSERT_TRUE(plans);
    std::set<MovementPlan> unique(plans->begin(), plans->end());
    EXPECT_EQ(unique.size(), plans->size());

    bool sawAllDDR = false;
    bool sawPartial = false;
    bool sawMaximal = false;
    std::set<std::vector<MovementHop>> maximalGraphs;
    std::set<std::vector<uint32_t>> destinationPartitions;
    for (const MovementPlan &plan : *plans) {
      size_t peerActions = 0;
      std::vector<std::vector<MovementActionId>> groups;
      std::vector<uint32_t> partition;
      for (const DDRBoundaryTransferPlan &transfer : plan.ddrTransfers) {
        const PeerTransferGraphPlan *graph =
            findPeerGraph(plan, MovementActionId(transfer.id));
        if (!graph) {
          partition.push_back(0);
          continue;
        }
        ++peerActions;
        ASSERT_TRUE(result.domain->contains(plan));
        auto group = llvm::find(groups, graph->actions);
        if (group == groups.end()) {
          groups.push_back(graph->actions);
          partition.push_back(static_cast<uint32_t>(groups.size()));
        } else {
          partition.push_back(
              static_cast<uint32_t>(std::distance(groups.begin(), group) + 1));
        }
        if (graph->actions.size() == 2)
          sawPartial = true;
        if (graph->actions == actionIds) {
          sawMaximal = true;
          EXPECT_EQ(graph->kind, PeerTransferGraphKind::SoftwareFanout);
          std::set<int64_t> children;
          for (const MovementHop &hop : graph->hops)
            EXPECT_TRUE(children.insert(hop.destination.getValue()).second);
          EXPECT_EQ(children.size(), 3u);
          maximalGraphs.insert(graph->hops);
        }
      }
      destinationPartitions.insert(std::move(partition));
      sawAllDDR |= peerActions == 0;
    }
    EXPECT_TRUE(sawAllDDR);
    EXPECT_TRUE(sawPartial);
    EXPECT_TRUE(sawMaximal);
    // Independent finite combinatorics: choose any peer subset and partition
    // it into sharing groups: sum C(3,k)*Bell(k) = 15. With root plus three
    // terminals and no optional relays, Cayley's formula gives 4^(4-2)=16
    // rooted arborescences for the maximal group.
    EXPECT_EQ(destinationPartitions.size(), 15u);
    EXPECT_EQ(maximalGraphs.size(), 16u);
  }
}

TEST(MovementDomainTest,
     DirectedEndpointFactsAdmitOnlyExplicitRelayAndNeverInventRawRoute) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture =
      makeFixture(false, {2, 1031, 128}, /*destinationTile=*/TileId(2));
  fixture.movement.resources.front().elementType =
      mlir::Float16Type::get(&context);
  MovementTransportFacts facts;
  facts.availableTiles = {TileId(2), TileId(0), TileId(1)};
  facts.endpointTransfers = {{TileId(1), TileId(2)}, {TileId(0), TileId(1)}};
  MovementDomainResult result =
      buildMovementDomain(fixture.movement, fixture.representations, facts);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 2u);
  EXPECT_TRUE(plans->front().peerGraphs.empty());
  const PeerTransferGraphPlan *relay = findPeerGraph(
      plans->back(), MovementActionId(plans->back().ddrTransfers.front().id));
  ASSERT_NE(relay, nullptr);
  EXPECT_EQ(relay->kind, PeerTransferGraphKind::SoftwareRelay);
  EXPECT_EQ(relay->hops, (std::vector<MovementHop>{{TileId(0), TileId(1)},
                                                   {TileId(1), TileId(2)}}));

  for (const MovementPlan &plan : *plans)
    for (const DDRBoundaryTransferPlan &transfer : plan.ddrTransfers)
      if (const PeerTransferGraphPlan *graph =
              findPeerGraph(plan, MovementActionId(transfer.id)))
        EXPECT_EQ(graph->kind, PeerTransferGraphKind::SoftwareRelay);
}

TEST(MovementDomainTest,
     ExternalLoadsEnumerateEveryDestinationRootWithoutHiddenDDRSource) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t major : {1024, 1025}) {
    RepresentationPlan representations;
    CanonicalMovementCoordinate movement;
    std::vector<MovementActionId> actions;
    for (int64_t tile = 1; tile <= 3; ++tile) {
      SemanticRootKey root;
      root.anchorIndex = static_cast<uint64_t>(tile);
      RootRegionWorkId work{root, TileId(tile)};
      LogicalShardId shard{root, {0}};
      DemandFragmentId fragment;
      fragment.source.kind = RootBoundaryKind::ProgramInput;
      fragment.source.index = 0;
      fragment.use = {0, shard};
      BoundaryRegionValueId logical{work, fragment};
      PhysicalVersionId version{RegionValueVersionId(logical)};
      representations.logicalValues.push_back({logical, version});
      representations.physicalVersions.push_back({version, MemLayout::Tensor});
      representations.uses.push_back(
          {BoundaryRepresentationUseId{logical}, version});
      ExternalLoadId load{logical};
      movement.plan.externalLoads.push_back({load, version});
      movement.resources.push_back(
          {MovementActionId(load), makeDomain({2, major, 128}),
           mlir::Float16Type::get(&context), std::nullopt, TileId(tile)});
      actions.push_back(MovementActionId(load));
    }
    llvm::sort(representations.logicalValues);
    llvm::sort(representations.physicalVersions);
    llvm::sort(representations.uses);
    llvm::sort(movement.plan.externalLoads);
    llvm::sort(movement.resources, [](const auto &lhs, const auto &rhs) {
      return lhs.action < rhs.action;
    });
    llvm::sort(actions);
    MovementDomainResult result =
        buildMovementDomain(movement, representations,
                            {TileId(0), TileId(1), TileId(2), TileId(3)});
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    auto plans = enumerate(*result.domain, /*limit=*/50000);
    ASSERT_TRUE(plans);
    std::set<int64_t> maximalRoots;
    bool sawAllDDR = false;
    for (const MovementPlan &plan : *plans) {
      sawAllDDR |= plan.peerGraphs.empty();
      for (const PeerTransferGraphPlan &graph : plan.peerGraphs) {
        if (graph.actions != actions)
          continue;
        ASSERT_EQ(graph.kind, PeerTransferGraphKind::ExternalLoadFanout);
        ASSERT_TRUE(graph.ddrRoot);
        maximalRoots.insert(graph.ddrRoot->destination.work.tile.getValue());
      }
    }
    EXPECT_TRUE(sawAllDDR);
    EXPECT_EQ(maximalRoots, (std::set<int64_t>{1, 2, 3}));
    std::vector<MovementPlan> proposals = result.domain->getProposals();
    ASSERT_FALSE(proposals.empty());
    EXPECT_TRUE(llvm::any_of(proposals.front().peerGraphs,
                             [&](const PeerTransferGraphPlan &graph) {
                               return graph.actions == actions;
                             }));
  }
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
  EXPECT_TRUE(plans->front().peerGraphs.empty());
}

TEST(MovementDomainTest, SameTileAndMalformedEndpointsFailClosed) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(false, {}, /*destinationTile=*/TileId(0));
  fixture.movement.resources.front().elementType =
      mlir::Float16Type::get(&context);
  MovementDomainResult sameTile = buildMovementDomain(
      fixture.movement, fixture.representations, {TileId(0), TileId(1)});
  ASSERT_TRUE(sameTile.succeeded());
  auto plans = enumerate(*sameTile.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_TRUE(plans->front().peerGraphs.empty());

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
       ExecutionInstanceId{RequiredMergeExecution{mergeWork, group}}});
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
