//===- RepresentationDomainTest.cpp ---------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/PhysicalVersionBuilder.h"

#include "mlir/IR/BuiltinTypes.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeBox(llvm::ArrayRef<int64_t> sizes,
                      mlir::MLIRContext &context) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  StaticRectangularIndexSet box{offsets, llvm::SmallVector<int64_t, 4>(sizes)};
  (void)context;
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
}

struct Fixture {
  CanonicalRepresentationCoordinate coordinate;
  RegionValueVersionId logical;
  RepresentationUseId firstUse;
  RepresentationUseId secondUse;
};

Fixture makeFixture(mlir::MLIRContext &context, llvm::ArrayRef<int64_t> sizes) {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::ProgramInput;
  fragment.use = {0, shard};
  BoundaryRegionValueId boundary{work, fragment};
  RegionValueVersionId logical = boundary;
  PhysicalVersionId primary{logical};
  RepresentationUseId first = BoundaryRepresentationUseId{boundary};
  fragment.use.operand = 1;
  RepresentationUseId second = LocalRepresentationUseId{work, fragment};

  CanonicalRepresentationCoordinate coordinate;
  coordinate.plan.logicalValues.push_back({logical, primary});
  coordinate.plan.physicalVersions.push_back({primary, MemLayout::Tensor});
  coordinate.plan.uses.push_back({first, primary});
  coordinate.plan.uses.push_back({second, primary});
  coordinate.resources.push_back({primary, makeBox(sizes, context),
                                  mlir::Float16Type::get(&context),
                                  MemLayout::Tensor});
  llvm::sort(coordinate.plan.uses);
  return {std::move(coordinate), std::move(logical), std::move(first),
          std::move(second)};
}

std::optional<std::vector<RepresentationPlan>>
enumerate(const RepresentationDomain &domain, size_t limit = 10000) {
  std::vector<RepresentationPlan> plans;
  RepresentationSuccessor next = domain.getFirstPlan();
  while (next.getKind() == RepresentationSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() || plans.size() >= limit ||
        !domain.contains(*next.getPlan()))
      return std::nullopt;
    plans.push_back(*next.getPlan());
    RepresentationCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == RepresentationSuccessorKind::End
             ? std::optional<std::vector<RepresentationPlan>>(std::move(plans))
             : std::nullopt;
}

TEST(RepresentationDomainTest,
     CompletePrimaryAndUseDomainMatchesIndependentCountAtRealisticScale) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    Fixture fixture = makeFixture(context, {2, extent, 128});
    RepresentationDomainResult result =
        buildRepresentationDomain(fixture.coordinate);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    auto plans = enumerate(*result.domain);
    ASSERT_TRUE(plans);
    EXPECT_EQ(plans->size(), 4u * 7u * 7u);
    EXPECT_EQ(std::set<RepresentationPlan>(plans->begin(), plans->end()).size(),
              plans->size());
    std::set<std::tuple<uint32_t, uint32_t, uint32_t>> actualChoices;
    for (const RepresentationPlan &plan : *plans) {
      ASSERT_EQ(plan.logicalValues.size(), 1u);
      ASSERT_EQ(plan.uses.size(), 2u);
      EXPECT_TRUE(result.domain->contains(plan));
      auto primary = llvm::find_if(plan.physicalVersions,
                                   [](const PhysicalVersionPlan &version) {
                                     return version.id.derivation.empty();
                                   });
      ASSERT_NE(primary, plan.physicalVersions.end());
      auto useCode = [&](const PhysicalUseBinding &binding) -> uint32_t {
        if (binding.version.derivation.empty())
          return 0;
        const PhysicalVersionDerivationStep &step =
            binding.version.derivation.back();
        uint32_t targetOrdinal = 0;
        for (MemLayout layout : {MemLayout::Tensor, MemLayout::NTensor,
                                 MemLayout::Cx, MemLayout::NCx}) {
          if (layout == primary->encoding)
            continue;
          if (layout == step.encoding)
            return 1 + targetOrdinal * 2 +
                   std::holds_alternative<UseRepresentationAnchor>(step.anchor);
          ++targetOrdinal;
        }
        return 100;
      };
      actualChoices.emplace(static_cast<uint32_t>(primary->encoding),
                            useCode(plan.uses[0]), useCode(plan.uses[1]));
    }
    std::set<std::tuple<uint32_t, uint32_t, uint32_t>> expectedChoices;
    for (MemLayout primary :
         {MemLayout::Tensor, MemLayout::NTensor, MemLayout::Cx, MemLayout::NCx})
      for (uint32_t first = 0; first < 7; ++first)
        for (uint32_t second = 0; second < 7; ++second)
          expectedChoices.emplace(static_cast<uint32_t>(primary), first,
                                  second);
    EXPECT_EQ(actualChoices, expectedChoices);
    CanonicalRepresentationCoordinate reversed = fixture.coordinate;
    std::reverse(reversed.plan.logicalValues.begin(),
                 reversed.plan.logicalValues.end());
    std::reverse(reversed.plan.physicalVersions.begin(),
                 reversed.plan.physicalVersions.end());
    std::reverse(reversed.plan.uses.begin(), reversed.plan.uses.end());
    std::reverse(reversed.resources.begin(), reversed.resources.end());
    RepresentationDomainResult reordered = buildRepresentationDomain(reversed);
    ASSERT_TRUE(reordered.succeeded())
        << (reordered.failure ? reordered.failure->detail : "");
    RepresentationSuccessor reorderedFirst = reordered.domain->getFirstPlan();
    ASSERT_NE(reorderedFirst.getPlan(), nullptr);
    EXPECT_EQ(*reorderedFirst.getPlan(), plans->front());
  }
}

TEST(RepresentationDomainTest,
     SharedAndPerUseConversionsHaveExplicitDistinctIdentities) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(context, {2, 1031, 128});
  RepresentationDomainResult result =
      buildRepresentationDomain(fixture.coordinate);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);

  bool sawShared = false;
  bool sawPerUse = false;
  for (const RepresentationPlan &plan : *plans) {
    auto primary =
        llvm::find_if(plan.physicalVersions, [](const auto &version) {
          return version.id.derivation.empty();
        });
    if (primary == plan.physicalVersions.end() ||
        primary->encoding != MemLayout::Tensor)
      continue;
    if (plan.uses[0].version == plan.uses[1].version &&
        !plan.uses[0].version.derivation.empty() &&
        plan.uses[0].version.derivation.back().encoding == MemLayout::NTensor) {
      sawShared = true;
      EXPECT_EQ(plan.physicalVersions.size(), 2u);
      EXPECT_TRUE(std::holds_alternative<SharedRepresentationAnchor>(
          plan.uses[0].version.derivation.back().anchor));
      std::string failureReason;
      auto prepared =
          prepareRepresentationPlan(*result.domain, plan, &failureReason);
      ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
      EXPECT_EQ(prepared->primaryVersions.size(), 1u);
      EXPECT_EQ(prepared->derivedVersions.size(), 1u);
    }
    if (!(plan.uses[0].version == plan.uses[1].version) &&
        !plan.uses[0].version.derivation.empty() &&
        !plan.uses[1].version.derivation.empty() &&
        plan.uses[0].version.derivation.back().encoding == MemLayout::NTensor &&
        plan.uses[1].version.derivation.back().encoding == MemLayout::NTensor &&
        std::holds_alternative<UseRepresentationAnchor>(
            plan.uses[0].version.derivation.back().anchor) &&
        std::holds_alternative<UseRepresentationAnchor>(
            plan.uses[1].version.derivation.back().anchor)) {
      sawPerUse = true;
      EXPECT_EQ(plan.physicalVersions.size(), 3u);
    }
  }
  EXPECT_TRUE(sawShared);
  EXPECT_TRUE(sawPerUse);
}

TEST(RepresentationDomainTest,
     ExactIdentityAliasIsExplicitAndUnsupportedAliasHasNoState) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture fixture = makeFixture(context, {2, 1025, 128});
  const auto &boundary = std::get<BoundaryRegionValueId>(fixture.logical);
  SupportRegionValueId aliasLogical{boundary.work,
                                    SupportValueId{boundary.work.root, 0}};
  PhysicalVersionId aliasPrimary{RegionValueVersionId(aliasLogical)};
  fixture.coordinate.plan.logicalValues.push_back({aliasLogical, aliasPrimary});
  fixture.coordinate.plan.physicalVersions.push_back(
      {aliasPrimary, MemLayout::Tensor});
  fixture.coordinate.resources.push_back(
      {aliasPrimary, makeBox({2, 1025, 128}, context),
       mlir::Float16Type::get(&context), MemLayout::Tensor});
  auto secondBinding = llvm::find_if(fixture.coordinate.plan.uses,
                                     [&](const PhysicalUseBinding &binding) {
                                       return binding.use == fixture.secondUse;
                                     });
  ASSERT_NE(secondBinding, fixture.coordinate.plan.uses.end());
  secondBinding->version = aliasPrimary;
  llvm::sort(fixture.coordinate.plan.logicalValues);
  llvm::sort(fixture.coordinate.plan.physicalVersions);

  IdentityAliasRequirement alias{fixture.secondUse, fixture.logical};
  RepresentationDomainResult result =
      buildRepresentationDomain(fixture.coordinate, {alias});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  auto aliasPlan = llvm::find_if(*plans, [&](const RepresentationPlan &plan) {
    auto binding = llvm::find_if(plan.uses, [&](const auto &candidate) {
      return candidate.use == fixture.secondUse;
    });
    return binding != plan.uses.end() && !binding->version.derivation.empty() &&
           binding->version.derivation.back().kind ==
               PhysicalVersionDerivationKind::AliasView;
  });
  ASSERT_NE(aliasPlan, plans->end());
  auto binding = llvm::find_if(aliasPlan->uses, [&](const auto &candidate) {
    return candidate.use == fixture.secondUse;
  });
  ASSERT_TRUE(binding->version.derivation.back().sourceLogicalValue);
  EXPECT_EQ(*binding->version.derivation.back().sourceLogicalValue,
            fixture.logical);
  std::string failureReason;
  auto prepared =
      prepareRepresentationPlan(*result.domain, *aliasPlan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;

  RepresentationLayoutTupleConstraint matching;
  matching.values = {fixture.logical, RegionValueVersionId(aliasLogical)};
  for (MemLayout layout :
       {MemLayout::Tensor, MemLayout::NTensor, MemLayout::Cx, MemLayout::NCx})
    matching.legalTuples.push_back({layout, layout});
  RepresentationDomainResult constrained =
      buildRepresentationDomain(fixture.coordinate, {alias}, {matching});
  ASSERT_TRUE(constrained.succeeded())
      << (constrained.failure ? constrained.failure->detail : "");
  auto constrainedPlans = enumerate(*constrained.domain);
  ASSERT_TRUE(constrainedPlans);
  EXPECT_EQ(constrainedPlans->size(), 4u * 7u * 8u);
  for (const RepresentationPlan &plan : *constrainedPlans) {
    ASSERT_EQ(plan.logicalValues.size(), 2u);
    auto sourceVersion = llvm::find_if(
        plan.physicalVersions, [&](const PhysicalVersionPlan &version) {
          return version.id.logicalValue == fixture.logical &&
                 version.id.derivation.empty();
        });
    auto targetVersion = llvm::find_if(
        plan.physicalVersions, [&](const PhysicalVersionPlan &version) {
          return version.id.logicalValue ==
                     RegionValueVersionId(aliasLogical) &&
                 version.id.derivation.empty();
        });
    ASSERT_NE(sourceVersion, plan.physicalVersions.end());
    ASSERT_NE(targetVersion, plan.physicalVersions.end());
    EXPECT_EQ(sourceVersion->encoding, targetVersion->encoding);
  }
  RepresentationLayoutTupleConstraint different;
  different.values = matching.values;
  different.legalTuples = {{MemLayout::Tensor, MemLayout::NTensor},
                           {MemLayout::NTensor, MemLayout::Tensor}};
  RepresentationDomainResult noSolution = buildRepresentationDomain(
      fixture.coordinate, {alias}, {matching, different});
  ASSERT_TRUE(noSolution.succeeded())
      << (noSolution.failure ? noSolution.failure->detail : "");
  EXPECT_EQ(noSolution.domain->getFirstPlan().getKind(),
            RepresentationSuccessorKind::End);

  CanonicalRepresentationCoordinate mismatched = fixture.coordinate;
  mismatched.resources.back().exactDomain = makeBox({2, 1024, 128}, context);
  RepresentationDomainResult unsupported =
      buildRepresentationDomain(mismatched, {alias});
  ASSERT_FALSE(unsupported.succeeded());
  ASSERT_TRUE(unsupported.failure);
  EXPECT_EQ(unsupported.failure->kind,
            RepresentationDomainFailureKind::UnsupportedSemantics);
}

TEST(RepresentationDomainTest, RankZeroAndMalformedInventoriesRemainTyped) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  Fixture scalar = makeFixture(context, {});
  RepresentationDomainResult rankZero =
      buildRepresentationDomain(scalar.coordinate);
  ASSERT_TRUE(rankZero.succeeded())
      << (rankZero.failure ? rankZero.failure->detail : "");
  EXPECT_EQ(rankZero.domain->getFirstPlan().getKind(),
            RepresentationSuccessorKind::Plan);

  Fixture coupled = makeFixture(context, {1, 2, 1024, 64, 1031, 128});
  RepresentationDomainResult rankSix =
      buildRepresentationDomain(coupled.coordinate);
  ASSERT_TRUE(rankSix.succeeded())
      << (rankSix.failure ? rankSix.failure->detail : "");
  EXPECT_EQ(rankSix.domain->getFirstPlan().getKind(),
            RepresentationSuccessorKind::Plan);

  CanonicalRepresentationCoordinate duplicate = scalar.coordinate;
  duplicate.resources.push_back(duplicate.resources.front());
  duplicate.plan.logicalValues.push_back(duplicate.plan.logicalValues.front());
  RepresentationDomainResult broken = buildRepresentationDomain(duplicate);
  ASSERT_FALSE(broken.succeeded());
  ASSERT_TRUE(broken.failure);
  EXPECT_EQ(broken.failure->kind,
            RepresentationDomainFailureKind::BrokenContract);

  RepresentationSuccessor first = rankSix.domain->getFirstPlan();
  ASSERT_NE(first.getPlan(), nullptr);
  RepresentationPlan unsupportedAlias = *first.getPlan();
  PhysicalVersionId alias = unsupportedAlias.logicalValues.front().primary;
  alias.derivation.push_back({PhysicalVersionDerivationKind::AliasView,
                              MemLayout::Tensor, MemLayout::Tensor,
                              SharedRepresentationAnchor{}});
  unsupportedAlias.physicalVersions.push_back({alias, MemLayout::Tensor});
  EXPECT_FALSE(rankSix.domain->contains(unsupportedAlias));
}

} // namespace
