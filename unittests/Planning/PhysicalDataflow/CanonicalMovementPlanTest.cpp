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
  auto module = parse(R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @decode(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x1031x128xf16>,
      %value: tensor<2x1031x64xf16>, %scale: f32,
      %mask: tensor<1025x1031xf16>) -> tensor<2x1025x64xf16> {
    %out = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale, %mask :
            tensor<2x1025x128xf16>, tensor<2x1031x128xf16>,
            tensor<2x1031x64xf16>, f32, tensor<1025x1031xf16>)
        outs(%out : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
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
  unsigned expectedGathers = 0;
  for (const ReductionMergeRequirement &merge : prefix->demand.reductionMerges)
    for (const ReductionContribution &contribution : merge.contributions)
      if (contribution.tile != merge.mergeTile)
        expectedGathers += contribution.components.size();
  EXPECT_EQ(coordinate->plan.reductionGathers.size(), expectedGathers);
  EXPECT_EQ(coordinate->plan.externalLoads.size(), 64u);
  EXPECT_TRUE(coordinate->plan.ddrTransfers.empty());
  EXPECT_EQ(coordinate->plan.publications.size(),
            prefix->demand.reductionMerges.size());
  for (const ReductionGatherPlan &gather : coordinate->plan.reductionGathers) {
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

TEST_F(CanonicalMovementPlanTest,
       OrdinaryReductionGathersOnlyTheRemotePartialResult) {
  auto module = parse(R"mlir(
module {
  func.func @holder() -> tensor<2x1025xf16> {
    %result = tensor.empty() : tensor<2x1025xf16>
    return %result : tensor<2x1025xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::tensor::EmptyOp root;
  module->walk([&](mlir::tensor::EmptyOp empty) { root = empty; });
  ASSERT_TRUE(root);
  SemanticRootKey rootKey;
  LogicalShardId remoteShard{rootKey, {0}};
  LogicalShardId localShard{rootKey, {1}};
  ReductionGroupId group{rootKey, 0, {0}};
  ExactIndexSet resultDomain = box({0, 0}, {2, 1025});

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
  remote.execution.push_back({remoteShard, {{0, 1}}});
  remote.contributions.push_back(
      {group, TileId(1),
       ReductionInitialization::IdentityPerContributionInitOnceAtMerge,
       ReductionAlgebraKind::StandardPartialReduction, std::nullopt,
       remoteContribution});

  RootRegionWork local;
  local.id = {rootKey, TileId(1)};
  local.rootOperation = root;
  local.execution.push_back({localShard, {{1, 1}}});
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
  const ReductionGatherPlan &gather = coordinate->plan.reductionGathers.front();
  EXPECT_TRUE(std::holds_alternative<ReductionPartialValueId>(gather.id.value));
  EXPECT_EQ(gather.id.contribution, remoteShard);
  EXPECT_EQ(gather.id.group, group);
  EXPECT_EQ(coordinate->plan.publications.size(), 1u);
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
