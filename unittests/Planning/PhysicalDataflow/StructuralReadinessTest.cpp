//===- StructuralReadinessTest.cpp -----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StructuralReadiness.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

TemporalScopeDescriptor makeScope(llvm::ArrayRef<int64_t> extents) {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  TemporalScopeDescriptor descriptor;
  descriptor.id.execution =
      ExecutionInstanceId{RequiredRootExecution{work, shard}};
  descriptor.id.invocation = TopLevelWorkPieceId{0};
  descriptor.iterationOffsets.assign(extents.size(), 0);
  descriptor.iterationExtents.assign(extents.begin(), extents.end());
  descriptor.iteratorCapabilities.assign(extents.size(),
                                         IteratorTilingCapability::Tileable);
  return descriptor;
}

TEST(StructuralReadinessTest,
     ClosedRealisticTemporalPrefixAdvancesOnlyToRepresentation) {
  TemporalDomainResult domain =
      buildTemporalDomain({makeScope({2, 1025, 128})});
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  TemporalSuccessor first = domain.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);

  StructuralReadinessResult initial =
      checkStructuralReadiness(*domain.domain, *first.getPlan());
  StructuralReadinessResult repeated =
      checkStructuralReadiness(*domain.domain, *first.getPlan());
  EXPECT_EQ(initial.getKind(), StructuralReadinessKind::ReadyForNextCoordinate);
  EXPECT_EQ(initial.getRequiredCoordinate(),
            RequiredPlanningCoordinate::Representation);
  EXPECT_EQ(repeated.getKind(), initial.getKind());
  EXPECT_EQ(repeated.getRequiredCoordinate(), initial.getRequiredCoordinate());
  EXPECT_TRUE(initial.getDetail().empty());

  TemporalPlan malformed = *first.getPlan();
  malformed.scopes.push_back(malformed.scopes.front());
  StructuralReadinessResult broken =
      checkStructuralReadiness(*domain.domain, malformed);
  EXPECT_EQ(broken.getKind(), StructuralReadinessKind::CompilerBug);
  EXPECT_FALSE(broken.getRequiredCoordinate());
  EXPECT_NE(broken.getDetail().find("outside"), llvm::StringRef::npos);
}

TEST(StructuralReadinessTest,
     RankZeroAndMergeOnlyPrefixesNeedNoSyntheticScopeOrResourceFact) {
  TemporalDomainResult rankZero = buildTemporalDomain({makeScope({})});
  ASSERT_TRUE(rankZero.succeeded())
      << (rankZero.failure ? rankZero.failure->detail : "");
  TemporalSuccessor scalar = rankZero.domain->getFirstPlan();
  ASSERT_EQ(scalar.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(scalar.getPlan(), nullptr);
  ASSERT_EQ(scalar.getPlan()->scopes.size(), 1u);
  EXPECT_TRUE(scalar.getPlan()->scopes.front().iteratorTileSizes.empty());
  EXPECT_EQ(
      checkStructuralReadiness(*rankZero.domain, *scalar.getPlan()).getKind(),
      StructuralReadinessKind::ReadyForNextCoordinate);

  TemporalDomainResult mergeOnly = buildTemporalDomain({});
  ASSERT_TRUE(mergeOnly.succeeded());
  TemporalSuccessor empty = mergeOnly.domain->getFirstPlan();
  ASSERT_EQ(empty.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(empty.getPlan(), nullptr);
  EXPECT_TRUE(empty.getPlan()->scopes.empty());
  EXPECT_EQ(checkStructuralReadiness(*mergeOnly.domain, *empty.getPlan())
                .getRequiredCoordinate(),
            RequiredPlanningCoordinate::Representation);
}

TEST(StructuralReadinessTest,
     NestedAndReplicaRaggedPrefixRemainsPurelyStructural) {
  TemporalScopeDescriptor parent = makeScope({2, 1025, 128});
  SemanticRootKey producerRoot;
  producerRoot.anchorIndex = 1;
  RootRegionWorkId producerWork{producerRoot, TileId(0)};
  LogicalShardId producerShard{producerRoot, {0, 0, 0}};
  RequiredRootExecution producer{producerWork, producerShard};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::StructuredResult;
  fragment.source.semantic = producerRoot;
  fragment.use.destinationShard =
      std::get<RequiredRootExecution>(
          std::get<ExecutionInstanceId>(parent.id.execution).source)
          .shard;
  fragment.ownerShard = producerShard;
  fragment.ownerTile = TileId(0);

  TemporalScopeDescriptor nested;
  nested.id.execution = ExecutionInstanceId{producer};
  NestedInvocationClassId invocation;
  invocation.parent = parent.id.execution;
  invocation.uses.push_back({fragment, /*requestedOffsets=*/{0, 0, 0},
                             /*requestedExtents=*/{1, 128, 64}});
  invocation.producerOffsets = {0, 0, 0};
  invocation.producerExtents = {1, 128, 64};
  nested.id.invocation = invocation;
  nested.iterationOffsets = invocation.producerOffsets;
  nested.iterationExtents = invocation.producerExtents;
  nested.iteratorCapabilities.assign(3, IteratorTilingCapability::Tileable);
  nested.parentScope = parent.id;

  ReplicaExecutionId replicaId{producer, fragment};
  TemporalScopeDescriptor replica;
  replica.id.execution = replicaId;
  replica.iterationOffsets = {0, 0, 0};
  replica.iterationExtents = {2, 1024, 128};
  replica.iteratorCapabilities.assign(3, IteratorTilingCapability::Tileable);

  TemporalDomainResult domain = buildTemporalDomain({nested, replica, parent});
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  TemporalSuccessor first = domain.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->scopes.size(), 3u);
  StructuralReadinessResult ready =
      checkStructuralReadiness(*domain.domain, *first.getPlan());
  EXPECT_EQ(ready.getKind(), StructuralReadinessKind::ReadyForNextCoordinate);
  EXPECT_EQ(ready.getRequiredCoordinate(),
            RequiredPlanningCoordinate::Representation);

  TemporalScopeDescriptor incompatibleNested = nested;
  auto &incompatibleInvocation =
      std::get<NestedInvocationClassId>(incompatibleNested.id.invocation);
  incompatibleInvocation.producerExtents[1] = 129;
  incompatibleNested.iterationExtents[1] = 129;
  TemporalDomainResult other =
      buildTemporalDomain({incompatibleNested, replica, parent});
  ASSERT_TRUE(other.succeeded());
  TemporalSuccessor otherFirst = other.domain->getFirstPlan();
  ASSERT_EQ(otherFirst.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(otherFirst.getPlan(), nullptr);
  StructuralReadinessResult mismatch =
      checkStructuralReadiness(*domain.domain, *otherFirst.getPlan());
  EXPECT_EQ(mismatch.getKind(), StructuralReadinessKind::CompilerBug);
  EXPECT_FALSE(mismatch.getRequiredCoordinate());
}

} // namespace
