//===- CanonicalMovementPlanTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class CanonicalMovementPlanTest : public ::testing::Test {
protected:
  CanonicalMovementPlanTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static ExactIndexSet box(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
    IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
    EXPECT_TRUE(set.isExact());
    StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                        llvm::to_vector(sizes)};
    return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                         {rectangle});
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalMovementPlanTest,
       AlignedAndRaggedSingleRootUsesLoadsAndPublications) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @map(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#id, #id],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    auto module = parse(source);
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                            &failureReason);
    ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;

    // This pair is the cross-boundary coverage witness for the completed
    // canonical B--H prefix. Check every intermediate contract instead of
    // treating successful construction of the final movement plan as proof.
    ASSERT_EQ(prefix->spatial.nodes.size(), 1u);
    const NodeExecutionPartition &partition = prefix->spatial.nodes.front();
    ASSERT_EQ(partition.shards.size(), 16u);
    EXPECT_TRUE(partition.reductionGroups.empty());
    std::array<bool, 16> observedTiles{};
    uint64_t coveredIterations = 0;
    llvm::SmallVector<int64_t, 16> shardQuerySizes;
    const std::array<int64_t, 3> logicalShape = {2, extent, 128};
    for (const ExecutionShard &shard : partition.shards) {
      ASSERT_GE(shard.tile.getValue(), 0);
      ASSERT_LT(shard.tile.getValue(), 16);
      EXPECT_FALSE(observedTiles[shard.tile.getValue()]);
      observedTiles[shard.tile.getValue()] = true;
      ASSERT_EQ(shard.iterationDomain.size(), logicalShape.size());
      uint64_t shardVolume = 1;
      for (auto [interval, dimension] :
           llvm::zip_equal(shard.iterationDomain, logicalShape)) {
        EXPECT_GE(interval.offset, 0);
        EXPECT_GT(interval.size, 0);
        EXPECT_LE(interval.getEnd(), dimension);
        shardVolume *= static_cast<uint64_t>(interval.size);
      }
      coveredIterations += shardVolume;
      shardQuerySizes.push_back(shard.iterationDomain[1].size);
    }
    EXPECT_TRUE(
        llvm::all_of(observedTiles, [](bool observed) { return observed; }));
    EXPECT_EQ(coveredIterations, static_cast<uint64_t>(2) * extent * 128);
    for (size_t lhs = 0; lhs < partition.shards.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < partition.shards.size(); ++rhs) {
        bool disjoint = false;
        for (auto [lhsInterval, rhsInterval] :
             llvm::zip_equal(partition.shards[lhs].iterationDomain,
                             partition.shards[rhs].iterationDomain))
          disjoint |= lhsInterval.getEnd() <= rhsInterval.offset ||
                      rhsInterval.getEnd() <= lhsInterval.offset;
        EXPECT_TRUE(disjoint) << "spatial shards must not overlap";
      }
    if (extent == 1024)
      EXPECT_EQ(*llvm::min_element(shardQuerySizes),
                *llvm::max_element(shardQuerySizes));
    else
      EXPECT_LT(*llvm::min_element(shardQuerySizes),
                *llvm::max_element(shardQuerySizes));

    ASSERT_EQ(prefix->demand.finalOwners.size(), 16u);
    ASSERT_EQ(prefix->demand.dependencyDemands.size(), 1u);
    EXPECT_TRUE(prefix->demand.reductionMerges.empty());
    ASSERT_EQ(prefix->demand.dependencyDemands.front().perDestination.size(),
              16u);
    uint64_t ownedIterations = 0;
    for (const FinalResultOwner &owner : prefix->demand.finalOwners) {
      ASSERT_TRUE(owner.shard.has_value());
      EXPECT_FALSE(owner.reductionGroup.has_value());
      ASSERT_EQ(owner.domain.getBoxes().size(), 1u);
      const StaticRectangularIndexSet &ownerBox =
          owner.domain.getBoxes().front();
      ASSERT_EQ(ownerBox.sizes.size(), logicalShape.size());
      uint64_t ownerVolume = 1;
      for (int64_t size : ownerBox.sizes) {
        EXPECT_GT(size, 0);
        ownerVolume *= static_cast<uint64_t>(size);
      }
      ownedIterations += ownerVolume;
      auto shard =
          llvm::find_if(partition.shards, [&](const ExecutionShard &s) {
            return s.shard == *owner.shard;
          });
      ASSERT_NE(shard, partition.shards.end());
      EXPECT_EQ(owner.tile, shard->tile);
      for (size_t axis = 0; axis < logicalShape.size(); ++axis) {
        EXPECT_EQ(ownerBox.offsets[axis], shard->iterationDomain[axis].offset);
        EXPECT_EQ(ownerBox.sizes[axis], shard->iterationDomain[axis].size);
      }
    }
    EXPECT_EQ(ownedIterations, static_cast<uint64_t>(2) * extent * 128);
    for (const DestinationDemand &destination :
         prefix->demand.dependencyDemands.front().perDestination) {
      ASSERT_EQ(destination.consumerExecutionDomain.getBoxes().size(), 1u);
      ASSERT_EQ(destination.operandDemand.getBoxes().size(), 1u);
      ASSERT_EQ(destination.sources.size(), 1u);
      EXPECT_TRUE(std::holds_alternative<ProgramInputSource>(
          destination.sources.front().source));
      ASSERT_EQ(destination.sources.front().requiredDomain.getBoxes().size(),
                1u);
      const auto &executionBox =
          destination.consumerExecutionDomain.getBoxes().front();
      const auto &demandBox = destination.operandDemand.getBoxes().front();
      const auto &sourceBox =
          destination.sources.front().requiredDomain.getBoxes().front();
      EXPECT_EQ(demandBox.offsets, executionBox.offsets);
      EXPECT_EQ(demandBox.sizes, executionBox.sizes);
      EXPECT_EQ(sourceBox.offsets, demandBox.offsets);
      EXPECT_EQ(sourceBox.sizes, demandBox.sizes);
    }

    ASSERT_EQ(prefix->rootWorks.size(), 16u);
    for (const RootRegionWork &work : prefix->rootWorks) {
      ASSERT_EQ(work.execution.size(), 1u);
      EXPECT_TRUE(work.contributions.empty());
      EXPECT_TRUE(work.merges.empty());
      ASSERT_EQ(work.operands.size(), 1u);
      ASSERT_EQ(work.operands.front().uses.size(), 1u);
      ASSERT_EQ(work.boundaries.size(), 1u);
      EXPECT_EQ(work.boundaries.front().id.kind,
                RootBoundaryKind::ProgramInput);
      ASSERT_TRUE(work.boundaries.front().requiredDomain.has_value());
      ASSERT_EQ(work.boundaries.front().consumerUses.size(), 1u);
      ASSERT_EQ(work.results.size(), 1u);
      ASSERT_TRUE(work.results.front().ownerShard.has_value());
      EXPECT_EQ(*work.results.front().ownerShard, work.execution.front().shard);
      EXPECT_FALSE(work.results.front().reductionGroup.has_value());
    }

    ASSERT_EQ(prefix->regions.groups.size(), 16u);
    for (const RegionGroupPlan &group : prefix->regions.groups) {
      ASSERT_EQ(group.mandatoryRoots.size(), 1u);
      ASSERT_EQ(group.executions.size(), 1u);
      ASSERT_EQ(group.externalBindings.size(), 1u);
      EXPECT_EQ(group.tile, group.mandatoryRoots.front().tile);
      EXPECT_TRUE(std::holds_alternative<RequiredRootExecution>(
          group.executions.front().id.source));
      EXPECT_EQ(group.externalBindings.front().fragment.source.kind,
                RootBoundaryKind::ProgramInput);
    }
    ASSERT_EQ(prefix->temporal.scopes.size(), 16u);
    for (const TemporalScopePlan &scope : prefix->temporal.scopes) {
      EXPECT_TRUE(scope.waveLoopOrder.empty());
      const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
      const auto *required =
          execution ? std::get_if<RequiredRootExecution>(&execution->source)
                    : nullptr;
      ASSERT_NE(required, nullptr);
      auto work =
          llvm::find_if(prefix->rootWorks, [&](const RootRegionWork &w) {
            return w.id == required->work;
          });
      ASSERT_NE(work, prefix->rootWorks.end());
      ASSERT_EQ(work->execution.size(), 1u);
      ASSERT_EQ(scope.iteratorTileSizes,
                llvm::map_to_vector(work->execution.front().iterationDomain,
                                    [](const IteratorInterval &interval) {
                                      return interval.size;
                                    }));
    }

    auto representation = buildCanonicalRepresentationPlan(
        prefix->regions, prefix->temporal, prefix->rootWorks);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representation);
    ASSERT_NE(representations, nullptr);
    ASSERT_EQ(representations->plan.primaryVersions.size(), 32u);
    ASSERT_EQ(representations->resources.size(), 32u);
    EXPECT_EQ(
        llvm::count_if(representations->plan.primaryVersions,
                       [](const PhysicalVersionPlan &version) {
                         return std::holds_alternative<BoundaryRegionValueId>(
                             version.id.logicalValue);
                       }),
        16u);
    EXPECT_EQ(
        llvm::count_if(representations->plan.primaryVersions,
                       [](const PhysicalVersionPlan &version) {
                         return std::holds_alternative<ExecutionResultValueId>(
                             version.id.logicalValue);
                       }),
        16u);
    CanonicalMovementPlanOutcome outcome = buildCanonicalMovementPlan(
        prefix->regions, *representations, prefix->rootWorks);
    const CanonicalMovementCoordinate *coordinate =
        getCanonicalMovementCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    EXPECT_EQ(coordinate->plan.externalLoads.size(), 16u);
    EXPECT_TRUE(coordinate->plan.ddrTransfers.empty());
    EXPECT_TRUE(coordinate->plan.reductionGathers.empty());
    EXPECT_EQ(coordinate->plan.publications.size(), 16u);
    EXPECT_EQ(coordinate->resources.size(), 32u);
    llvm::SmallVector<int64_t, 32> querySizes;
    for (const MovementResourceDescription &resource : coordinate->resources) {
      EXPECT_TRUE(resource.elementType.isF16());
      ASSERT_EQ(resource.exactDomain.getBoxes().size(), 1u);
      querySizes.push_back(resource.exactDomain.getBoxes().front().sizes[1]);
    }
    if (extent == 1024)
      EXPECT_EQ(*llvm::min_element(querySizes), *llvm::max_element(querySizes));
    else
      EXPECT_LT(*llvm::min_element(querySizes), *llvm::max_element(querySizes));
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalMovementPlanTest,
       DiamondSingletonRegionsUseExplicitDDRAndPublishBothSinks) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @diamond(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%producer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16): linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %left_empty = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%left_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16): linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %right_empty = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%right_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16): linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %join_empty = tensor.empty() : tensor<2x1025x128xf16>
    %join = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right : tensor<2x1025x128xf16>,
                            tensor<2x1025x128xf16>)
        outs(%join_empty : tensor<2x1025x128xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %value = arith.addf %lhs, %rhs : f16
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    return %join, %right : tensor<2x1025x128xf16>,
                           tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                          &failureReason);
  ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
  auto representation = buildCanonicalRepresentationPlan(
      prefix->regions, prefix->temporal, prefix->rootWorks);
  const CanonicalRepresentationCoordinate *representations =
      getCanonicalRepresentationCoordinate(representation);
  ASSERT_NE(representations, nullptr);
  CanonicalMovementPlanOutcome outcome = buildCanonicalMovementPlan(
      prefix->regions, *representations, prefix->rootWorks);
  const CanonicalMovementCoordinate *coordinate =
      getCanonicalMovementCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  EXPECT_EQ(coordinate->plan.externalLoads.size(), 16u);
  EXPECT_EQ(coordinate->plan.ddrTransfers.size(), 64u);
  EXPECT_EQ(coordinate->plan.publications.size(), 32u);
  EXPECT_TRUE(coordinate->plan.reductionGathers.empty());
  EXPECT_EQ(coordinate->resources.size(), 112u);
  for (const DDRBoundaryTransferPlan &transfer :
       coordinate->plan.ddrTransfers) {
    EXPECT_NE(transfer.source, transfer.destination);
    EXPECT_EQ(transfer.id.destination.fragment.source.kind,
              RootBoundaryKind::StructuredResult);
  }

  std::reverse(prefix->rootWorks.begin(), prefix->rootWorks.end());
  CanonicalMovementPlanOutcome reversed = buildCanonicalMovementPlan(
      prefix->regions, *representations, prefix->rootWorks);
  const CanonicalMovementCoordinate *reversedCoordinate =
      getCanonicalMovementCoordinate(reversed);
  ASSERT_NE(reversedCoordinate, nullptr);
  ASSERT_EQ(reversedCoordinate->plan.ddrTransfers.size(),
            coordinate->plan.ddrTransfers.size());
  for (auto [expected, actual] :
       llvm::zip_equal(coordinate->plan.ddrTransfers,
                       reversedCoordinate->plan.ddrTransfers))
    EXPECT_EQ(actual.id, expected.id);
}

TEST_F(CanonicalMovementPlanTest,
       FlashDecodingMovesEveryRemoteComponentAndSkipsLocalContributions) {
  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
        queryExtent, keyValueExtent));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                            &failureReason);
    ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
    auto representation = buildCanonicalRepresentationPlan(
        prefix->regions, prefix->temporal, prefix->rootWorks);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representation);
    ASSERT_NE(representations, nullptr);
    CanonicalMovementPlanOutcome outcome = buildCanonicalMovementPlan(
        prefix->regions, *representations, prefix->rootWorks);
    const CanonicalMovementCoordinate *coordinate =
        getCanonicalMovementCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    unsigned expectedGathers = 0;
    for (const ReductionMergeRequirement &merge :
         prefix->demand.reductionMerges)
      for (const ReductionContribution &contribution : merge.contributions)
        if (contribution.tile != merge.mergeTile)
          expectedGathers += contribution.components.size();
    EXPECT_EQ(coordinate->plan.reductionGathers.size(), expectedGathers);
    EXPECT_EQ(coordinate->plan.externalLoads.size(), 64u);
    EXPECT_TRUE(coordinate->plan.ddrTransfers.empty());
    EXPECT_EQ(coordinate->plan.publications.size(),
              prefix->demand.reductionMerges.size());
    for (const ReductionGatherPlan &gather :
         coordinate->plan.reductionGathers) {
      const auto *component =
          std::get_if<CoupledComponentValueId>(&gather.id.value);
      ASSERT_NE(component, nullptr);
      const auto *source =
          std::get_if<RequiredRootExecution>(&component->execution.source);
      const auto *merge =
          std::get_if<RequiredMergeExecution>(&gather.mergeExecution.source);
      ASSERT_NE(source, nullptr);
      ASSERT_NE(merge, nullptr);
      EXPECT_NE(source->work.tile, merge->work.tile);
    }
  }
}

TEST_F(CanonicalMovementPlanTest,
       OrdinaryReductionGathersOnlyTheRemotePartialResult) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module {\n"
           << "  func.func @holder() -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %result = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    auto module = parse(source);
    ASSERT_TRUE(module);
    mlir::tensor::EmptyOp root;
    module->walk([&](mlir::tensor::EmptyOp empty) { root = empty; });
    ASSERT_TRUE(root);
    SemanticRootKey rootKey;
    LogicalShardId remoteShard{rootKey, {0}};
    LogicalShardId localShard{rootKey, {1}};
    ReductionGroupId group{rootKey, 0, {0}};
    ExactIndexSet resultDomain = box({0, 0, 0}, {2, extent, 128});

    ReductionContribution remoteContribution;
    remoteContribution.shard = remoteShard;
    remoteContribution.tile = TileId(0);
    remoteContribution.results.push_back({0, resultDomain});
    ReductionContribution localContribution;
    localContribution.shard = localShard;
    localContribution.tile = TileId(1);
    localContribution.results.push_back({0, resultDomain});
    ReductionMergeRequirement merge;
    merge.group = group;
    merge.mergeTile = TileId(1);
    merge.results.push_back({0, resultDomain});
    merge.contributions = {remoteContribution, localContribution};

    RootRegionWork remote;
    remote.id = {rootKey, TileId(0)};
    remote.rootOperation = root;
    remote.execution.push_back({remoteShard, {{0, 1}, {0, extent}, {0, 128}}});
    remote.contributions.push_back(
        {group, TileId(1),
         ReductionInitialization::IdentityPerContributionInitOnceAtMerge,
         ReductionAlgebraKind::StandardPartialReduction, std::nullopt,
         remoteContribution});

    RootRegionWork local;
    local.id = {rootKey, TileId(1)};
    local.rootOperation = root;
    local.execution.push_back({localShard, {{1, 1}, {0, extent}, {0, 128}}});
    local.contributions.push_back(
        {group, TileId(1),
         ReductionInitialization::IdentityPerContributionInitOnceAtMerge,
         ReductionAlgebraKind::StandardPartialReduction, std::nullopt,
         localContribution});
    local.merges.push_back(merge);
    local.results.push_back({0, std::nullopt, group, resultDomain});

    std::array<RootRegionWork, 2> works{remote, local};
    CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan(works);
    const RegionPlan *regions = getRegionPlan(regionOutcome);
    ASSERT_NE(regions, nullptr);
    CanonicalTemporalPlanOutcome temporalOutcome =
        buildCanonicalTemporalPlan(*regions, works);
    const TemporalPlan *temporal = getTemporalPlan(temporalOutcome);
    ASSERT_NE(temporal, nullptr);
    auto representation =
        buildCanonicalRepresentationPlan(*regions, *temporal, works);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representation);
    ASSERT_NE(representations, nullptr);
    CanonicalMovementPlanOutcome outcome =
        buildCanonicalMovementPlan(*regions, *representations, works);
    const CanonicalMovementCoordinate *coordinate =
        getCanonicalMovementCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.reductionGathers.size(), 1u);
    const ReductionGatherPlan &gather =
        coordinate->plan.reductionGathers.front();
    EXPECT_TRUE(
        std::holds_alternative<ReductionPartialValueId>(gather.id.value));
    EXPECT_EQ(gather.id.contribution, remoteShard);
    EXPECT_EQ(gather.id.group, group);
    EXPECT_EQ(coordinate->plan.publications.size(), 1u);
    ASSERT_EQ(coordinate->resources.size(), 2u);
    for (const MovementResourceDescription &resource : coordinate->resources) {
      ASSERT_EQ(resource.exactDomain.getBoxes().size(), 1u);
      EXPECT_EQ(resource.exactDomain.getBoxes().front().sizes,
                (llvm::SmallVector<int64_t, 3>{2, extent, 128}));
    }
  }
}

TEST_F(CanonicalMovementPlanTest,
       RankZeroMovesWhileMissingVersionsAndCoverageMismatchFailClosed) {
  auto module = parse(R"mlir(
#scalar = affine_map<() -> ()>
module {
  func.func @scalar(%input: tensor<f16>) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.generic {
        indexing_maps = [#scalar, #scalar], iterator_types = []}
        ins(%input : tensor<f16>) outs(%empty : tensor<f16>) {
      ^bb0(%value: f16, %old: f16): linalg.yield %value : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  llvm::SmallVector<TileId, 1> tile{TileId(0)};
  auto prefix =
      wafer::test::buildCanonicalPlanningPrefix(*dag, tile, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
  auto representation = buildCanonicalRepresentationPlan(
      prefix->regions, prefix->temporal, prefix->rootWorks);
  const CanonicalRepresentationCoordinate *representations =
      getCanonicalRepresentationCoordinate(representation);
  ASSERT_NE(representations, nullptr);
  CanonicalMovementPlanOutcome valid = buildCanonicalMovementPlan(
      prefix->regions, *representations, prefix->rootWorks);
  const CanonicalMovementCoordinate *coordinate =
      getCanonicalMovementCoordinate(valid);
  ASSERT_NE(coordinate, nullptr);
  ASSERT_EQ(coordinate->plan.externalLoads.size(), 1u);
  ASSERT_EQ(coordinate->plan.publications.size(), 1u);
  for (const MovementResourceDescription &resource : coordinate->resources)
    EXPECT_EQ(resource.exactDomain.getRank(), 0u);

  CanonicalRepresentationCoordinate missing = *representations;
  missing.resources.pop_back();
  CanonicalMovementPlanOutcome missingOutcome =
      buildCanonicalMovementPlan(prefix->regions, missing, prefix->rootWorks);
  const auto *missingFailure = std::get_if<BrokenMovementPlan>(&missingOutcome);
  ASSERT_NE(missingFailure, nullptr);
  EXPECT_EQ(missingFailure->reason,
            BrokenMovementPlanReason::MissingPhysicalVersion);

  auto diamond = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @chain(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %a0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%a0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %b0 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%b0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    return %b : tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(diamond);
  auto chainDag =
      StructuredDAGAnalysis::create(function(*diamond), &failureReason);
  ASSERT_TRUE(mlir::succeeded(chainDag)) << failureReason;
  auto chainPrefix = wafer::test::buildCanonicalPlanningPrefix(
      *chainDag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(chainPrefix)) << failureReason;
  auto chainRepresentation = buildCanonicalRepresentationPlan(
      chainPrefix->regions, chainPrefix->temporal, chainPrefix->rootWorks);
  const CanonicalRepresentationCoordinate *chainRepresentations =
      getCanonicalRepresentationCoordinate(chainRepresentation);
  ASSERT_NE(chainRepresentations, nullptr);
  CanonicalMovementPlanOutcome validChain = buildCanonicalMovementPlan(
      chainPrefix->regions, *chainRepresentations, chainPrefix->rootWorks);
  const CanonicalMovementCoordinate *validChainCoordinate =
      getCanonicalMovementCoordinate(validChain);
  ASSERT_NE(validChainCoordinate, nullptr);
  ASSERT_FALSE(validChainCoordinate->plan.ddrTransfers.empty());
  PhysicalVersionId ddrSource =
      validChainCoordinate->plan.ddrTransfers.front().source;
  CanonicalRepresentationCoordinate badCoverage = *chainRepresentations;
  auto source =
      llvm::find_if(badCoverage.resources,
                    [&](const RepresentationResourceDescription &resource) {
                      return resource.version == ddrSource;
                    });
  ASSERT_NE(source, badCoverage.resources.end());
  source->exactDomain = box({0, 0, 0}, {1, 1, 1});
  CanonicalMovementPlanOutcome coverage = buildCanonicalMovementPlan(
      chainPrefix->regions, badCoverage, chainPrefix->rootWorks);
  const auto *coverageFailure = std::get_if<BrokenMovementPlan>(&coverage);
  ASSERT_NE(coverageFailure, nullptr);
  EXPECT_EQ(coverageFailure->reason,
            BrokenMovementPlanReason::CoverageMismatch);
}

} // namespace
