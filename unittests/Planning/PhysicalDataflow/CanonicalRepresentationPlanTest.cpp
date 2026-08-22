//===- CanonicalRepresentationPlanTest.cpp ----------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class CanonicalRepresentationPlanTest : public ::testing::Test {
protected:
  CanonicalRepresentationPlanTest() {
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

TEST_F(CanonicalRepresentationPlanTest,
       AlignedAndRaggedValuesHaveOneTensorPrimaryAndExactResource) {
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
    CanonicalRepresentationPlanOutcome outcome =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *coordinate =
        getCanonicalRepresentationCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.primaryVersions.size(), 32u);
    ASSERT_EQ(coordinate->resources.size(), 32u);
    EXPECT_TRUE(llvm::all_of(coordinate->plan.primaryVersions,
                             [](const PhysicalVersionPlan &version) {
                               return version.encoding ==
                                      wafer::MemLayout::Tensor;
                             }));
    EXPECT_EQ(
        llvm::count_if(coordinate->plan.primaryVersions,
                       [](const PhysicalVersionPlan &version) {
                         return std::holds_alternative<BoundaryRegionValueId>(
                             version.id.logicalValue);
                       }),
        16u);
    EXPECT_EQ(
        llvm::count_if(coordinate->plan.primaryVersions,
                       [](const PhysicalVersionPlan &version) {
                         return std::holds_alternative<ExecutionResultValueId>(
                             version.id.logicalValue);
                       }),
        16u);
    llvm::SmallVector<int64_t, 16> querySizes;
    for (const RepresentationResourceDescription &resource :
         coordinate->resources) {
      EXPECT_TRUE(resource.elementType.isF16());
      EXPECT_EQ(resource.encoding, wafer::MemLayout::Tensor);
      ASSERT_EQ(resource.exactDomain.getBoxes().size(), 1u);
      querySizes.push_back(resource.exactDomain.getBoxes().front().sizes[1]);
    }
    if (extent == 1024)
      EXPECT_EQ(*llvm::min_element(querySizes), *llvm::max_element(querySizes));
    else
      EXPECT_LT(*llvm::min_element(querySizes), *llvm::max_element(querySizes));

    std::reverse(prefix->rootWorks.begin(), prefix->rootWorks.end());
    CanonicalRepresentationPlanOutcome reversed =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *reversedCoordinate =
        getCanonicalRepresentationCoordinate(reversed);
    ASSERT_NE(reversedCoordinate, nullptr);
    ASSERT_EQ(reversedCoordinate->plan.primaryVersions.size(),
              coordinate->plan.primaryVersions.size());
    for (auto [expected, actual] :
         llvm::zip_equal(coordinate->plan.primaryVersions,
                         reversedCoordinate->plan.primaryVersions)) {
      EXPECT_EQ(actual.id, expected.id);
      EXPECT_EQ(actual.encoding, expected.encoding);
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalRepresentationPlanTest,
       PadAndMultiResultInventorySkipsScalarCaptureAndKeepsSupportOnce) {
  auto module = parse(R"mlir(
#input = affine_map<(b, m, n) -> (b, m, n)>
#output = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @pad_pair(%input: tensor<2x1025x128xf16>,
                      %lhs_init: tensor<2x1026x128xf16>,
                      %rhs_init: tensor<2x1026x128xf16>)
      -> (tensor<2x1026x128xf16>, tensor<2x1026x128xf16>) {
    %zero = arith.constant 0.0 : f16
    %padded = tensor.pad %input low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%b: index, %m: index, %n: index):
        tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    %lhs, %rhs = linalg.generic {
        indexing_maps = [#input, #output, #output],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%padded : tensor<2x1026x128xf16>)
        outs(%lhs_init, %rhs_init : tensor<2x1026x128xf16>,
                                     tensor<2x1026x128xf16>) {
      ^bb0(%value: f16, %old_lhs: f16, %old_rhs: f16):
        %next_lhs = arith.addf %value, %old_lhs : f16
        %next_rhs = arith.addf %value, %old_rhs : f16
        linalg.yield %next_lhs, %next_rhs : f16, f16
    } -> (tensor<2x1026x128xf16>, tensor<2x1026x128xf16>)
    return %lhs, %rhs : tensor<2x1026x128xf16>, tensor<2x1026x128xf16>
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
  CanonicalRepresentationPlanOutcome outcome = buildCanonicalRepresentationPlan(
      prefix->regions, prefix->temporal, prefix->rootWorks);
  const CanonicalRepresentationCoordinate *coordinate =
      getCanonicalRepresentationCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  EXPECT_EQ(
      llvm::count_if(coordinate->plan.primaryVersions,
                     [](const PhysicalVersionPlan &version) {
                       return std::holds_alternative<SupportRegionValueId>(
                           version.id.logicalValue);
                     }),
      16u);
  EXPECT_EQ(
      llvm::count_if(coordinate->plan.primaryVersions,
                     [](const PhysicalVersionPlan &version) {
                       return std::holds_alternative<ExecutionResultValueId>(
                           version.id.logicalValue);
                     }),
      32u);
  EXPECT_EQ(
      llvm::count_if(coordinate->plan.primaryVersions,
                     [](const PhysicalVersionPlan &version) {
                       return std::holds_alternative<BoundaryRegionValueId>(
                           version.id.logicalValue);
                     }),
      48u);
  EXPECT_EQ(coordinate->plan.primaryVersions.size(), 96u);
}

TEST_F(CanonicalRepresentationPlanTest,
       FlashDecodingComponentsKeepRequirementTypesAndDomains) {
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
    CanonicalRepresentationPlanOutcome outcome =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *coordinate =
        getCanonicalRepresentationCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);

    unsigned expectedComponents = 0;
    for (const RootRegionWork &work : prefix->rootWorks) {
      for (const RootContributionWork &contribution : work.contributions)
        expectedComponents += contribution.contribution.components.size();
      for (const ReductionMergeRequirement &merge : work.merges)
        expectedComponents += merge.components.size();
    }
    unsigned actualComponents = 0;
    unsigned maximumF32 = 0;
    unsigned accumulatorF16 = 0;
    for (const RepresentationResourceDescription &resource :
         coordinate->resources) {
      const auto *component =
          std::get_if<CoupledComponentValueId>(&resource.version.logicalValue);
      if (!component)
        continue;
      ++actualComponents;
      maximumF32 += component->component ==
                        wafer::CoupledReductionComponentKind::Maximum &&
                    resource.elementType.isF32();
      accumulatorF16 += component->component ==
                            wafer::CoupledReductionComponentKind::Accumulator &&
                        resource.elementType.isF16();
      EXPECT_FALSE(resource.exactDomain.isEmpty());
    }
    EXPECT_EQ(actualComponents, expectedComponents);
    EXPECT_GT(maximumF32, 0u);
    EXPECT_GT(accumulatorF16, 0u);
  }
}

TEST_F(CanonicalRepresentationPlanTest,
       EmptyAndScalarValuesAreSkippedWhileRankZeroTensorIsRepresented) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    %empty_index = tensor.empty() : tensor<1024xindex>
    %empty_scalar = tensor.empty() : tensor<f16>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().back();
  mlir::tensor::EmptyOp emptyIndex;
  mlir::tensor::EmptyOp emptyScalar;
  module->walk([&](mlir::tensor::EmptyOp empty) {
    auto type = mlir::cast<mlir::RankedTensorType>(empty.getType());
    if (type.getRank() == 0)
      emptyScalar = empty;
    else
      emptyIndex = empty;
  });
  ASSERT_TRUE(emptyIndex);
  ASSERT_TRUE(emptyScalar);
  SemanticRootKey root;
  RootRegionWork work;
  work.id = {root, TileId(0)};
  work.rootOperation = holder;
  LogicalShardId shard{root, {}};
  work.execution.push_back({shard, {}});
  RootSupportValueWork support;
  support.id = {{}, 0};
  support.operation = emptyIndex;
  support.result = 0;
  support.requiredDomain = box({0}, {1024});
  work.supportValues.push_back(support);
  CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan({work});
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  ASSERT_NE(regions, nullptr);
  CanonicalTemporalPlanOutcome temporalOutcome =
      buildCanonicalTemporalPlan(*regions, {work});
  const TemporalPlan *temporal = getTemporalPlan(temporalOutcome);
  ASSERT_NE(temporal, nullptr);
  CanonicalRepresentationPlanOutcome invalid =
      buildCanonicalRepresentationPlan(*regions, *temporal, {work});
  const auto *failure = std::get_if<UnsupportedRepresentationPlan>(&invalid);
  ASSERT_NE(failure, nullptr);
  EXPECT_EQ(failure->feature, UnsupportedRepresentationFeature::ElementType);

  support.operation = emptyScalar;
  support.requiredDomain = box({}, {});
  work.supportValues = {support};
  CanonicalRepresentationPlanOutcome rankZero =
      buildCanonicalRepresentationPlan(*regions, *temporal, {work});
  const CanonicalRepresentationCoordinate *rankZeroCoordinate =
      getCanonicalRepresentationCoordinate(rankZero);
  ASSERT_NE(rankZeroCoordinate, nullptr);
  ASSERT_EQ(rankZeroCoordinate->resources.size(), 1u);
  EXPECT_EQ(rankZeroCoordinate->resources.front().exactDomain.getRank(), 0u);
  EXPECT_TRUE(rankZeroCoordinate->resources.front().elementType.isF16());

  work.supportValues.push_back(support);
  CanonicalRepresentationPlanOutcome duplicate =
      buildCanonicalRepresentationPlan(*regions, *temporal, {work});
  const auto *duplicateFailure =
      std::get_if<BrokenRepresentationPlan>(&duplicate);
  ASSERT_NE(duplicateFailure, nullptr);
  EXPECT_EQ(duplicateFailure->reason,
            BrokenRepresentationPlanReason::DuplicateLogicalValue);
  work.supportValues.pop_back();

  TemporalPlan missingTemporal = *temporal;
  missingTemporal.scopes.clear();
  CanonicalRepresentationPlanOutcome missing =
      buildCanonicalRepresentationPlan(*regions, missingTemporal, {work});
  const auto *missingFailure = std::get_if<BrokenRepresentationPlan>(&missing);
  ASSERT_NE(missingFailure, nullptr);
  EXPECT_EQ(missingFailure->reason,
            BrokenRepresentationPlanReason::PlanWorkMismatch);

  work.supportValues.clear();
  RootBoundaryWork emptyBoundary;
  emptyBoundary.id.kind = RootBoundaryKind::ProgramInput;
  emptyBoundary.requiredDomain =
      ExactIndexSet(mlir::presburger::PresburgerSet::getEmpty(
                        mlir::presburger::PresburgerSpace::getSetSpace(0)),
                    ExactIndexSetForm::BoxUnion);
  emptyBoundary.consumerUses.push_back(
      {{0, shard}, *emptyBoundary.requiredDomain, {}});
  work.boundaries.push_back(emptyBoundary);
  RegionPlan emptyRegions = *regions;
  DemandFragmentId fragment;
  fragment.source = emptyBoundary.id;
  fragment.use = {0, shard};
  emptyRegions.groups.front().externalBindings.push_back({fragment});
  CanonicalRepresentationPlanOutcome empty =
      buildCanonicalRepresentationPlan(emptyRegions, *temporal, {work});
  const CanonicalRepresentationCoordinate *emptyCoordinate =
      getCanonicalRepresentationCoordinate(empty);
  ASSERT_NE(emptyCoordinate, nullptr);
  EXPECT_TRUE(emptyCoordinate->plan.primaryVersions.empty());
}

TEST_F(CanonicalRepresentationPlanTest,
       NormalizesFiniteGeneralDomainIntoDisjointTensorPieces) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    %value = tensor.empty() : tensor<2x1025x128xf16>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::tensor::EmptyOp value;
  module->walk([&](mlir::tensor::EmptyOp candidate) { value = candidate; });
  ASSERT_TRUE(value);

  IndexSetResult lower =
      IndexRelation::staticRectangularDomain({0, 0, 0}, {2, 500, 128});
  IndexSetResult upper =
      IndexRelation::staticRectangularDomain({0, 500, 0}, {2, 525, 128});
  ASSERT_TRUE(lower.isExact() && upper.isExact());
  ExactIndexSet general(lower.set->unionSet(*upper.set),
                        ExactIndexSetForm::GeneralPresburger);

  SemanticRootKey root;
  LogicalShardId shard{root, {}};
  RootRegionWork work;
  work.id = {root, TileId(0)};
  work.rootOperation = &function(*module).getBody().front().back();
  work.execution.push_back({shard, {{0, 2}, {0, 1025}, {0, 128}}});
  RootSupportValueWork support;
  support.id = {{}, 0};
  support.operation = value;
  support.result = 0;
  support.requiredDomain = general;
  work.supportValues.push_back(support);

  CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan({work});
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  ASSERT_NE(regions, nullptr);
  CanonicalTemporalPlanOutcome temporalOutcome =
      buildCanonicalTemporalPlan(*regions, {work});
  const TemporalPlan *temporal = getTemporalPlan(temporalOutcome);
  ASSERT_NE(temporal, nullptr);
  CanonicalRepresentationPlanOutcome outcome =
      buildCanonicalRepresentationPlan(*regions, *temporal, {work});
  const CanonicalRepresentationCoordinate *coordinate =
      getCanonicalRepresentationCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  ASSERT_EQ(coordinate->resources.size(), 1u);
  const ExactIndexSet &normalized = coordinate->resources.front().exactDomain;
  EXPECT_EQ(normalized.getForm(), ExactIndexSetForm::BoxUnion);
  ASSERT_EQ(normalized.getBoxes().size(), 2u);
  EXPECT_TRUE(
      normalized.getPresburgerSet().isEqual(general.getPresburgerSet()));
  EXPECT_EQ(normalized.getBoxes()[0].offsets,
            (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
  EXPECT_EQ(normalized.getBoxes()[0].sizes,
            (llvm::SmallVector<int64_t, 4>{2, 500, 128}));
  EXPECT_EQ(normalized.getBoxes()[1].offsets,
            (llvm::SmallVector<int64_t, 4>{0, 500, 0}));
  EXPECT_EQ(normalized.getBoxes()[1].sizes,
            (llvm::SmallVector<int64_t, 4>{2, 525, 128}));
}

} // namespace
