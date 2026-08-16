//===- ExactDemandTest.cpp - Policy-free logical demand types -------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

namespace {

using wafer::analysis::DemandDependency;
using wafer::analysis::DemandEdgeKind;
using wafer::analysis::ExactDemandResult;
using wafer::analysis::ExactDemandStatus;
using wafer::analysis::IREpoch;
using wafer::analysis::IndexRelation;
using wafer::analysis::IndexRelationStatus;
using wafer::analysis::LogicalNodeTrial;
using wafer::analysis::LogicalShardTrial;
using wafer::analysis::LogicalTileBinding;
using wafer::analysis::TileRole;
using wafer::TileId;

TEST(IREpochTest, IsStableUntilAdvancedAndAdvancesMonotonically) {
  IREpoch first = IREpoch::current();
  ASSERT_TRUE(first.isValid());
  EXPECT_EQ(first, IREpoch::current());
  EXPECT_EQ(first.getGeneration(), IREpoch::current().getGeneration());

  IREpoch::advance();
  IREpoch second = IREpoch::current();
  EXPECT_NE(first, second);
  EXPECT_GT(second.getGeneration(), first.getGeneration());
  EXPECT_EQ(second, IREpoch::current());

  IREpoch invalid;
  EXPECT_FALSE(invalid.isValid());
  EXPECT_NE(invalid, second);
}

TEST(ExactDemandTest, MapsRelationFailuresToTypedVerdicts) {
  EXPECT_EQ(wafer::analysis::mapIndexRelationStatus(IndexRelationStatus::Unsupported),
            ExactDemandStatus::UnsupportedSemanticRelation);
  EXPECT_EQ(wafer::analysis::mapIndexRelationStatus(IndexRelationStatus::SoundBound),
            ExactDemandStatus::IndeterminateFailure);
  EXPECT_EQ(wafer::analysis::mapIndexRelationStatus(IndexRelationStatus::Invalid),
            ExactDemandStatus::IndeterminateFailure);
  EXPECT_EQ(
      wafer::analysis::mapIndexRelationStatus(IndexRelationStatus::ResourceExhausted),
      ExactDemandStatus::IndeterminateFailure);
}

TEST(ExactDemandTest, OnlyProvenConclusionsAreCacheableAsLegality) {
  EXPECT_TRUE(wafer::analysis::isCacheableLegalityConclusion(
      ExactDemandStatus::Satisfied));
  EXPECT_TRUE(wafer::analysis::isCacheableLegalityConclusion(
      ExactDemandStatus::ProvenLogicalInfeasible));
  EXPECT_FALSE(wafer::analysis::isCacheableLegalityConclusion(
      ExactDemandStatus::UnsupportedSemanticRelation));
  EXPECT_FALSE(wafer::analysis::isCacheableLegalityConclusion(
      ExactDemandStatus::IndeterminateFailure));
}

TEST(ExactDemandTest, TrialCarriesExplicitDomainsRolesAndEpoch) {
  mlir::MLIRContext context;
  auto iteration = IndexRelation::staticRectangularDomain({0, 0}, {8, 4});
  ASSERT_TRUE(iteration.isExact());
  auto firstOwner = IndexRelation::staticRectangularDomain({0, 0}, {4, 4});
  ASSERT_TRUE(firstOwner.isExact());
  auto secondOwner = IndexRelation::staticRectangularDomain({4, 0}, {4, 4});
  ASSERT_TRUE(secondOwner.isExact());

  LogicalNodeTrial node;
  node.node = 3;
  node.completeIterationDomain = *iteration.set;
  node.bindings.push_back(LogicalTileBinding{
      TileId(2), *firstOwner.set, TileRole::PartialReductionContribution});
  node.bindings.push_back(
      LogicalTileBinding{TileId(5), *secondOwner.set, TileRole::UniquePartition});

  LogicalShardTrial trial;
  trial.epoch = IREpoch::current();
  trial.nodes.push_back(std::move(node));

  ASSERT_EQ(trial.nodes.size(), 1u);
  EXPECT_EQ(trial.nodes[0].node, 3u);
  EXPECT_EQ(trial.nodes[0].bindings.size(), 2u);
  EXPECT_EQ(trial.nodes[0].bindings[0].tile, TileId(2));
  EXPECT_EQ(trial.nodes[0].bindings[0].role,
            TileRole::PartialReductionContribution);
  EXPECT_EQ(trial.nodes[0].bindings[1].tile, TileId(5));
  EXPECT_EQ(trial.nodes[0].bindings[1].role, TileRole::UniquePartition);
  EXPECT_FALSE(trial.nodes[0].bindings[0].ownedDomain->isEqual(
      trial.nodes[0].bindings[1].ownedDomain.value()));
  EXPECT_EQ(trial.epoch, IREpoch::current());

  DemandDependency dependency;
  dependency.producerNode = 3;
  dependency.producerResult = 0;
  dependency.consumerNode = 7;
  dependency.consumerOperand = 1;
  dependency.kind = DemandEdgeKind::InitInput;
  EXPECT_TRUE(dependency.supportChain.empty());

  ExactDemandResult result;
  EXPECT_EQ(result.status, ExactDemandStatus::IndeterminateFailure);
  EXPECT_FALSE(result.mergeObligation);
  EXPECT_EQ(result.role, TileRole::UniquePartition);
  EXPECT_FALSE(result.uncoveredWitness);
}

} // namespace
