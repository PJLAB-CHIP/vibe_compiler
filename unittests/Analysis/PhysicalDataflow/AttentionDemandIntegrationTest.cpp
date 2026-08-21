//===- AttentionDemandIntegrationTest.cpp -----------------------------===//

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"

#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class AttentionDemandIntegrationTest : public ::testing::Test {
protected:
  AttentionDemandIntegrationTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static wafer::LinalgExtAttentionOp attention(mlir::ModuleOp module) {
    wafer::LinalgExtAttentionOp result;
    module.walk([&](wafer::LinalgExtAttentionOp candidate) {
      if (!result)
        result = candidate;
    });
    return result;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static std::string singleK2Source(wafer::AttentionAlgorithm algorithm,
                                    int64_t queryExtent, int64_t keyValueExtent,
                                    bool withMask) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @main(%query: tensor<2x)mlir"
           << queryExtent << "x128xf16>, %key: tensor<2x" << keyValueExtent
           << "x128xf16>, %value: tensor<2x" << keyValueExtent
           << "x64xf16>, %scale: f32";
    if (withMask)
      stream << ", %mask: tensor<" << queryExtent << "x" << keyValueExtent
             << "xf16>";
    stream << ") -> tensor<2x" << queryExtent << "x64xf16> {\n"
           << "    %out = tensor.empty() : tensor<2x" << queryExtent
           << "x64xf16>\n"
           << "    %result = wafer.linalg_ext.attention\n"
           << "        ins(%query, %key, %value, %scale";
    if (withMask)
      stream << ", %mask";
    stream << " : tensor<2x" << queryExtent << "x128xf16>, tensor<2x"
           << keyValueExtent << "x128xf16>, tensor<2x" << keyValueExtent
           << "x64xf16>, f32";
    if (withMask)
      stream << ", tensor<" << queryExtent << "x" << keyValueExtent << "xf16>";
    stream << ")\n"
           << "        outs(%out : tensor<2x" << queryExtent << "x64xf16>)\n"
           << "        algorithm(<"
           << (algorithm == wafer::AttentionAlgorithm::FlashAttention
                   ? "flash_attention"
                   : "flash_decoding")
           << ">)\n"
           << "        indexing_maps = [#q, #k, #v, #s";
    if (withMask)
      stream << ", #mask";
    stream << ", #o]\n"
           << "        -> tensor<2x" << queryExtent << "x64xf16>\n"
           << "    return %result : tensor<2x" << queryExtent << "x64xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static std::string multiK2Source() {
    return R"mlir(
#q = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, k1)>
#v = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, n)>
#s = affine_map<(b, m, k1, k2a, k2b, n) -> ()>
#mask = affine_map<(b, m, k1, k2a, k2b, n) -> (m, k2a, k2b)>
#o = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, n)>
module {
  func.func @main(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x33x31x128xf16>,
      %value: tensor<2x33x31x64xf16>, %scale: f32,
      %mask: tensor<1025x33x31xf16>) -> tensor<2x1025x64xf16> {
    %out = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale, %mask :
            tensor<2x1025x128xf16>, tensor<2x33x31x128xf16>,
            tensor<2x33x31x64xf16>, f32, tensor<1025x33x31xf16>)
        outs(%out : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
  }

  static mlir::FailureOr<CanonicalSpatialCoordinate>
  buildCoordinate(const StructuredDAGAnalysis &dag,
                  llvm::ArrayRef<int64_t> factors,
                  llvm::ArrayRef<unsigned> outputParallelIterators,
                  bool addMergeGroups, std::string *failureReason) {
    mlir::FailureOr<CanonicalSpatialCoordinate> coordinate =
        buildCanonicalSpatialAssignment(dag, allTiles(), failureReason);
    if (mlir::failed(coordinate) || coordinate->plan.nodes.size() != 1 ||
        coordinate->plan.nodes.front().axes.size() != factors.size())
      return mlir::failure();

    uint64_t cellCount = 1;
    NodeSpatialPlan &node = coordinate->plan.nodes.front();
    node.axes.clear();
    node.reductionMerges.clear();
    for (auto [iterator, factor] : llvm::enumerate(factors)) {
      node.axes.push_back({static_cast<uint32_t>(iterator),
                           IteratorPartitionScheme::BalancedParts, factor});
      cellCount *= static_cast<uint64_t>(factor);
    }
    if (cellCount != 16)
      return mlir::failure();
    node.embedding = allTiles();

    mlir::FailureOr<SpatialAssignment> provisional = closeSpatialPlanStructure(
        coordinate->problem, coordinate->plan, failureReason);
    if (mlir::failed(provisional))
      return mlir::failure();
    if (addMergeGroups) {
      std::map<std::vector<uint32_t>, TileId> firstContributors;
      for (const ExecutionShard &shard : provisional->nodes.front().shards) {
        std::vector<uint32_t> parallelCoordinate;
        for (unsigned iterator : outputParallelIterators)
          parallelCoordinate.push_back(shard.shard.coordinate[iterator]);
        firstContributors.try_emplace(std::move(parallelCoordinate),
                                      shard.tile);
      }
      for (const auto &[parallelCoordinate, tile] : firstContributors) {
        ReductionGroupId group;
        group.root = node.root;
        group.resultGroup = 0;
        group.parallelCoordinate.assign(parallelCoordinate.begin(),
                                        parallelCoordinate.end());
        node.reductionMerges.push_back({std::move(group), tile});
      }
    }
    mlir::FailureOr<SpatialAssignment> assignment = closeSpatialPlanStructure(
        coordinate->problem, coordinate->plan, failureReason);
    if (mlir::failed(assignment))
      return mlir::failure();
    coordinate->assignment = std::move(*assignment);
    return coordinate;
  }

  static const DependencyDemand *findDependency(const ExactDemandProof &proof,
                                                uint32_t operand) {
    auto found = llvm::find_if(proof.dependencyDemands,
                               [&](const DependencyDemand &dependency) {
                                 return dependency.consumerOperand == operand;
                               });
    return found == proof.dependencyDemands.end() ? nullptr : &*found;
  }

  static const StaticRectangularIndexSet *singleBox(const ExactIndexSet &set) {
    return set.getForm() == ExactIndexSetForm::BoxUnion &&
                   set.getBoxes().size() == 1
               ? &set.getBoxes().front()
               : nullptr;
  }

  static uint64_t volume(const StaticRectangularIndexSet &box) {
    uint64_t result = 1;
    for (int64_t size : box.sizes)
      result *= static_cast<uint64_t>(size);
    return result;
  }

  static bool overlaps(const StaticRectangularIndexSet &lhs,
                       const StaticRectangularIndexSet &rhs) {
    if (lhs.offsets.size() != rhs.offsets.size())
      return false;
    for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
         llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes))
      if (lhsOffset >= rhsOffset + rhsSize || rhsOffset >= lhsOffset + lhsSize)
        return false;
    return true;
  }

  static std::string outcomeDetail(const ExactDemandOutcome &outcome) {
    return std::visit(
        [](const auto &value) -> std::string {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, ExactDemandProof>)
            return {};
          else
            return value.detail;
        },
        outcome);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(AttentionDemandIntegrationTest,
       FlashAttentionAlignedAndRaggedDemandRemainUncoupled) {
  struct Case {
    int64_t queryExtent;
    int64_t keyValueExtent;
    bool withMask;
  };
  for (const Case &testCase :
       {Case{1024, 1024, false}, Case{1025, 1031, true}}) {
    SCOPED_TRACE(testCase.queryExtent);
    mlir::OwningOpRef<mlir::ModuleOp> module = parse(singleK2Source(
        wafer::AttentionAlgorithm::FlashAttention, testCase.queryExtent,
        testCase.keyValueExtent, testCase.withMask));
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate = buildCoordinate(*dag, {2, 4, 1, 1, 2}, {0, 1, 4},
                                      /*addMergeGroups=*/false, &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                                 &failureReason);
    ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
    ExactDemandOutcome outcome = session->query(coordinate->assignment);
    const ExactDemandProof *proof = getExactDemandProof(outcome);
    ASSERT_NE(proof, nullptr) << outcomeDetail(outcome);
    EXPECT_TRUE(proof->reductionMerges.empty());
    EXPECT_EQ(proof->finalOwners.size(), 16u);
    EXPECT_EQ(proof->dependencyDemands.size(), testCase.withMask ? 4u : 3u);
    EXPECT_TRUE(llvm::all_of(proof->finalOwners, [](const auto &owner) {
      return owner.shard.has_value() && !owner.reductionGroup;
    }));

    uint64_t outputElements = 0;
    for (const FinalResultOwner &owner : proof->finalOwners) {
      const StaticRectangularIndexSet *box = singleBox(owner.domain);
      ASSERT_NE(box, nullptr);
      outputElements += volume(*box);
    }
    for (size_t lhs = 0; lhs < proof->finalOwners.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < proof->finalOwners.size(); ++rhs) {
        const StaticRectangularIndexSet *lhsBox =
            singleBox(proof->finalOwners[lhs].domain);
        const StaticRectangularIndexSet *rhsBox =
            singleBox(proof->finalOwners[rhs].domain);
        ASSERT_NE(lhsBox, nullptr);
        ASSERT_NE(rhsBox, nullptr);
        EXPECT_FALSE(overlaps(*lhsBox, *rhsBox));
      }
    EXPECT_EQ(outputElements,
              static_cast<uint64_t>(2 * testCase.queryExtent * 64));

    const DependencyDemand *query = findDependency(*proof, 0);
    const DependencyDemand *key = findDependency(*proof, 1);
    const DependencyDemand *value = findDependency(*proof, 2);
    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);
    ASSERT_EQ(query->perDestination.size(), 16u);
    for (auto [queryDestination, keyDestination, valueDestination] :
         llvm::zip_equal(query->perDestination, key->perDestination,
                         value->perDestination)) {
      const StaticRectangularIndexSet *queryBox =
          singleBox(queryDestination.operandDemand);
      const StaticRectangularIndexSet *keyBox =
          singleBox(keyDestination.operandDemand);
      const StaticRectangularIndexSet *valueBox =
          singleBox(valueDestination.operandDemand);
      ASSERT_NE(queryBox, nullptr);
      ASSERT_NE(keyBox, nullptr);
      ASSERT_NE(valueBox, nullptr);
      EXPECT_EQ(queryBox->sizes[2], 128);
      EXPECT_EQ(keyBox->sizes, (llvm::SmallVector<int64_t, 4>{
                                   1, testCase.keyValueExtent, 128}));
      EXPECT_EQ(valueBox->sizes[1], testCase.keyValueExtent);
      EXPECT_EQ(valueBox->sizes[2], 32);
    }
    if (testCase.withMask) {
      const DependencyDemand *mask = findDependency(*proof, 4);
      ASSERT_NE(mask, nullptr);
      for (const DestinationDemand &destination : mask->perDestination) {
        const StaticRectangularIndexSet *box =
            singleBox(destination.operandDemand);
        ASSERT_NE(box, nullptr);
        EXPECT_EQ(box->sizes[1], testCase.keyValueExtent);
      }
    }

    auto view = StructuredDemandView::create(*dag, coordinate->assignment,
                                             *proof, &failureReason);
    ASSERT_TRUE(mlir::succeeded(view)) << failureReason;
    EXPECT_FALSE(view->hasSpatialReduction(0));
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(AttentionDemandIntegrationTest,
       FlashDecodingAlignedAndRaggedProduceOneCoupledStatePerOutputPiece) {
  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    mlir::OwningOpRef<mlir::ModuleOp> module = parse(singleK2Source(
        wafer::AttentionAlgorithm::FlashDecoding, queryExtent, keyValueExtent,
        /*withMask=*/true));
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    wafer::LinalgExtAttentionOp sourceAttention = attention(*module);
    ASSERT_TRUE(sourceAttention);
    wafer::CoupledReductionDescription sourceDescription =
        sourceAttention.getCoupledReductionDescription();
    ASSERT_EQ(sourceDescription.components.size(), 3u);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate = buildCoordinate(*dag, {2, 2, 1, 4, 1}, {0, 1, 4},
                                      /*addMergeGroups=*/true, &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                                 &failureReason);
    ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
    ExactDemandOutcome outcome = session->query(coordinate->assignment);
    const ExactDemandProof *proof = getExactDemandProof(outcome);
    ASSERT_NE(proof, nullptr) << outcomeDetail(outcome);
    ASSERT_EQ(proof->reductionMerges.size(), 4u);
    ASSERT_EQ(proof->finalOwners.size(), 4u);
    EXPECT_EQ(proof->dependencyDemands.size(), 4u);

    for (const ReductionMergeRequirement &merge : proof->reductionMerges) {
      EXPECT_EQ(merge.algebra, ReductionAlgebraKind::CoupledReduction);
      EXPECT_EQ(merge.initialization,
                ReductionInitialization::CoupledIdentityPerContribution);
      ASSERT_TRUE(merge.coupledRule.has_value());
      EXPECT_EQ(merge.coupledRule->mergeKind,
                wafer::CoupledReductionMergeKind::OnlineAttention);
      EXPECT_EQ(merge.coupledRule->finalizationKind,
                wafer::CoupledReductionFinalizationKind::NormalizeAccumulator);
      ASSERT_EQ(merge.results.size(), 1u);
      ASSERT_EQ(merge.components.size(), 3u);
      ASSERT_EQ(merge.contributions.size(), 4u);
      EXPECT_EQ(merge.components[0].kind,
                wafer::CoupledReductionComponentKind::Maximum);
      EXPECT_EQ(merge.components[1].kind,
                wafer::CoupledReductionComponentKind::Sum);
      EXPECT_EQ(merge.components[2].kind,
                wafer::CoupledReductionComponentKind::Accumulator);
      for (size_t component = 0; component < 3; ++component) {
        EXPECT_EQ(merge.components[component].kind,
                  sourceDescription.components[component].kind);
        EXPECT_EQ(merge.components[component].indexingMap,
                  sourceDescription.components[component].indexingMap);
        EXPECT_EQ(merge.components[component].elementType,
                  sourceDescription.components[component].elementType);
      }
      EXPECT_EQ(merge.components[0].domain.getRank(), 2u);
      EXPECT_EQ(merge.components[1].domain.getRank(), 2u);
      EXPECT_EQ(merge.components[2].domain.getRank(), 3u);
      const StaticRectangularIndexSet *maximumDomain =
          singleBox(merge.components[0].domain);
      const StaticRectangularIndexSet *sumDomain =
          singleBox(merge.components[1].domain);
      const StaticRectangularIndexSet *accumulatorDomain =
          singleBox(merge.components[2].domain);
      ASSERT_NE(maximumDomain, nullptr);
      ASSERT_NE(sumDomain, nullptr);
      ASSERT_NE(accumulatorDomain, nullptr);
      EXPECT_EQ(maximumDomain->offsets, sumDomain->offsets);
      EXPECT_EQ(maximumDomain->sizes, sumDomain->sizes);
      EXPECT_EQ(accumulatorDomain->offsets[0], maximumDomain->offsets[0]);
      EXPECT_EQ(accumulatorDomain->offsets[1], maximumDomain->offsets[1]);
      EXPECT_EQ(accumulatorDomain->sizes[0], maximumDomain->sizes[0]);
      EXPECT_EQ(accumulatorDomain->sizes[1], maximumDomain->sizes[1]);
      EXPECT_EQ(accumulatorDomain->sizes[2], 64);

      int64_t nextKeyValueOffset = 0;
      for (const ReductionContribution &contribution : merge.contributions) {
        ASSERT_TRUE(contribution.results.empty());
        ASSERT_EQ(contribution.components.size(), 3u);
        EXPECT_EQ(contribution.components[0].kind,
                  wafer::CoupledReductionComponentKind::Maximum);
        EXPECT_EQ(contribution.components[1].kind,
                  wafer::CoupledReductionComponentKind::Sum);
        EXPECT_EQ(contribution.components[2].kind,
                  wafer::CoupledReductionComponentKind::Accumulator);
        for (size_t component = 0; component < 3; ++component) {
          const StaticRectangularIndexSet *expected =
              singleBox(merge.components[component].domain);
          const StaticRectangularIndexSet *actual =
              singleBox(contribution.components[component].domain);
          ASSERT_NE(expected, nullptr);
          ASSERT_NE(actual, nullptr);
          EXPECT_EQ(actual->offsets, expected->offsets);
          EXPECT_EQ(actual->sizes, expected->sizes);
        }
        const StaticRectangularIndexSet *iteration =
            singleBox(contribution.iterationDomain);
        ASSERT_NE(iteration, nullptr);
        EXPECT_EQ(iteration->offsets[3], nextKeyValueOffset);
        nextKeyValueOffset += iteration->sizes[3];
      }
      EXPECT_EQ(nextKeyValueOffset, keyValueExtent);
      EXPECT_TRUE(
          llvm::any_of(proof->finalOwners, [&](const FinalResultOwner &owner) {
            return owner.reductionGroup == merge.group && !owner.shard &&
                   owner.tile == merge.mergeTile;
          }));
    }

    const DependencyDemand *query = findDependency(*proof, 0);
    const DependencyDemand *key = findDependency(*proof, 1);
    const DependencyDemand *value = findDependency(*proof, 2);
    const DependencyDemand *mask = findDependency(*proof, 4);
    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);
    ASSERT_NE(mask, nullptr);
    for (size_t base = 0; base < query->perDestination.size(); base += 4) {
      const StaticRectangularIndexSet *firstQuery =
          singleBox(query->perDestination[base].operandDemand);
      ASSERT_NE(firstQuery, nullptr);
      int64_t nextOffset = 0;
      for (size_t contribution = 0; contribution < 4; ++contribution) {
        const size_t index = base + contribution;
        const StaticRectangularIndexSet *queryBox =
            singleBox(query->perDestination[index].operandDemand);
        const StaticRectangularIndexSet *keyBox =
            singleBox(key->perDestination[index].operandDemand);
        const StaticRectangularIndexSet *valueBox =
            singleBox(value->perDestination[index].operandDemand);
        const StaticRectangularIndexSet *maskBox =
            singleBox(mask->perDestination[index].operandDemand);
        ASSERT_NE(queryBox, nullptr);
        ASSERT_NE(keyBox, nullptr);
        ASSERT_NE(valueBox, nullptr);
        ASSERT_NE(maskBox, nullptr);
        EXPECT_EQ(queryBox->offsets, firstQuery->offsets);
        EXPECT_EQ(queryBox->sizes, firstQuery->sizes);
        EXPECT_EQ(keyBox->offsets[1], nextOffset);
        EXPECT_EQ(valueBox->offsets[1], nextOffset);
        EXPECT_EQ(maskBox->offsets[1], nextOffset);
        nextOffset += keyBox->sizes[1];
      }
      EXPECT_EQ(nextOffset, keyValueExtent);
    }

    auto view = StructuredDemandView::create(*dag, coordinate->assignment,
                                             *proof, &failureReason);
    ASSERT_TRUE(mlir::succeeded(view)) << failureReason;
    EXPECT_TRUE(view->hasSpatialReduction(0));

    SpatialAssignment permuted = coordinate->assignment;
    std::reverse(permuted.nodes.front().shards.begin(),
                 permuted.nodes.front().shards.end());
    ExactDemandOutcome permutedOutcome = session->query(permuted);
    const ExactDemandProof *permutedProof =
        getExactDemandProof(permutedOutcome);
    ASSERT_NE(permutedProof, nullptr) << outcomeDetail(permutedOutcome);
    ASSERT_EQ(permutedProof->reductionMerges.size(),
              proof->reductionMerges.size());
    for (auto [expected, actual] : llvm::zip_equal(
             proof->reductionMerges, permutedProof->reductionMerges)) {
      EXPECT_EQ(actual.group, expected.group);
      EXPECT_EQ(actual.mergeTile, expected.mergeTile);
      EXPECT_EQ(actual.contributions.size(), expected.contributions.size());
      for (auto [expectedContribution, actualContribution] :
           llvm::zip_equal(expected.contributions, actual.contributions)) {
        EXPECT_EQ(actualContribution.shard, expectedContribution.shard);
        EXPECT_EQ(actualContribution.tile, expectedContribution.tile);
      }
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(AttentionDemandIntegrationTest,
       FlashDecodingMultiK2CoversTheCompleteTwoDimensionalFiber) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(multiK2Source());
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate = buildCoordinate(*dag, {2, 1, 1, 2, 4, 1}, {0, 1, 5},
                                    /*addMergeGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr) << outcomeDetail(outcome);
  ASSERT_EQ(proof->reductionMerges.size(), 2u);
  ASSERT_EQ(proof->finalOwners.size(), 2u);
  for (const ReductionMergeRequirement &merge : proof->reductionMerges) {
    ASSERT_EQ(merge.contributions.size(), 8u);
    uint64_t covered = 0;
    for (size_t lhs = 0; lhs < merge.contributions.size(); ++lhs) {
      const StaticRectangularIndexSet *lhsBox =
          singleBox(merge.contributions[lhs].iterationDomain);
      ASSERT_NE(lhsBox, nullptr);
      covered += static_cast<uint64_t>(lhsBox->sizes[3] * lhsBox->sizes[4]);
      for (size_t rhs = lhs + 1; rhs < merge.contributions.size(); ++rhs) {
        const StaticRectangularIndexSet *rhsBox =
            singleBox(merge.contributions[rhs].iterationDomain);
        ASSERT_NE(rhsBox, nullptr);
        const bool disjointK2 =
            lhsBox->offsets[3] >= rhsBox->offsets[3] + rhsBox->sizes[3] ||
            rhsBox->offsets[3] >= lhsBox->offsets[3] + lhsBox->sizes[3] ||
            lhsBox->offsets[4] >= rhsBox->offsets[4] + rhsBox->sizes[4] ||
            rhsBox->offsets[4] >= lhsBox->offsets[4] + lhsBox->sizes[4];
        EXPECT_TRUE(disjointK2);
      }
    }
    EXPECT_EQ(covered, 33u * 31u);
    ASSERT_EQ(merge.components.size(), 3u);
    EXPECT_EQ(merge.components[0].domain.getRank(), 2u);
    EXPECT_EQ(merge.components[2].domain.getRank(), 3u);
  }
}

TEST_F(AttentionDemandIntegrationTest,
       ModeAndMergeContractFailuresRemainTypedAndDoNotPoisonTheSession) {
  std::string failureReason;
  auto fdModule =
      parse(singleK2Source(wafer::AttentionAlgorithm::FlashDecoding, 1024, 1024,
                           /*withMask=*/true));
  ASSERT_TRUE(fdModule);
  auto fdDag =
      StructuredDAGAnalysis::create(function(*fdModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(fdDag)) << failureReason;
  auto validFD = buildCoordinate(*fdDag, {2, 2, 1, 4, 1}, {0, 1, 4},
                                 /*addMergeGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(validFD)) << failureReason;
  auto fdSession = DemandPlanningSession::create(*fdDag, IndexRelationLimits(),
                                                 &failureReason);
  ASSERT_TRUE(mlir::succeeded(fdSession)) << failureReason;

  SpatialAssignment missingMerge = validFD->assignment;
  missingMerge.nodes.front().reductionGroups.pop_back();
  ExactDemandOutcome missing = fdSession->query(missingMerge);
  const auto *missingFailure = std::get_if<InvalidSpatialAssignment>(&missing);
  ASSERT_NE(missingFailure, nullptr);
  EXPECT_EQ(missingFailure->reason,
            InvalidSpatialAssignmentReason::ReductionGroup);

  auto unpartitionedFD =
      buildCoordinate(*fdDag, {2, 4, 1, 1, 2}, {0, 1, 4},
                      /*addMergeGroups=*/false, &failureReason);
  ASSERT_TRUE(mlir::succeeded(unpartitionedFD)) << failureReason;
  ExactDemandOutcome unpartitioned =
      fdSession->query(unpartitionedFD->assignment);
  EXPECT_TRUE(std::holds_alternative<InvalidSpatialAssignment>(unpartitioned));

  ExactDemandOutcome valid = fdSession->query(validFD->assignment);
  ASSERT_NE(getExactDemandProof(valid), nullptr) << outcomeDetail(valid);

  auto faModule = parse(
      singleK2Source(wafer::AttentionAlgorithm::FlashAttention, 1024, 1024,
                     /*withMask=*/false));
  ASSERT_TRUE(faModule);
  auto faDag =
      StructuredDAGAnalysis::create(function(*faModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(faDag)) << failureReason;
  auto partitionedFA = buildCoordinate(*faDag, {2, 2, 1, 4, 1}, {0, 1, 4},
                                       /*addMergeGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(partitionedFA)) << failureReason;
  auto faSession = DemandPlanningSession::create(*faDag, IndexRelationLimits(),
                                                 &failureReason);
  ASSERT_TRUE(mlir::succeeded(faSession)) << failureReason;
  ExactDemandOutcome partitioned = faSession->query(partitionedFA->assignment);
  EXPECT_TRUE(std::holds_alternative<InvalidSpatialAssignment>(partitioned));
}

} // namespace
