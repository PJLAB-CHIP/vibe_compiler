//===- StructuredDemandAnalysisTest.cpp -------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.h"

#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class StructuredDemandAnalysisTest : public ::testing::Test {
protected:
  StructuredDemandAnalysisTest() {
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

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

struct ObservedStructuredRelationAnalysis {
  explicit ObservedStructuredRelationAnalysis(mlir::Operation *operation)
      : analysis(operation) {
    ++constructionCount;
  }

  static unsigned constructionCount;
  StructuredRelationAnalysis analysis;
};

unsigned ObservedStructuredRelationAnalysis::constructionCount = 0;

struct ObserveRelationPass
    : mlir::PassWrapper<ObserveRelationPass,
                        mlir::OperationPass<mlir::func::FuncOp>> {
  explicit ObserveRelationPass(llvm::SmallVectorImpl<unsigned> *observations)
      : observations(observations) {}

  void runOnOperation() override {
    auto &analysis = getAnalysis<ObservedStructuredRelationAnalysis>();
    if (!analysis.analysis.isValid()) {
      signalPassFailure();
      return;
    }
    observations->push_back(
        ObservedStructuredRelationAnalysis::constructionCount);
    markAllAnalysesPreserved();
  }

  llvm::SmallVectorImpl<unsigned> *observations;
};

struct MutateRelationInputPass
    : mlir::PassWrapper<MutateRelationInputPass,
                        mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    getOperation()->setDiscardableAttr("test.relation_mutation",
                                       mlir::UnitAttr::get(&getContext()));
  }
};

TEST_F(StructuredDemandAnalysisTest,
       DerivesStructuredProgramInputAndConstantBoundariesWithoutMutation) {
  auto module = parse(R"mlir(
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%input: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %constant = arith.constant dense<1.0> : tensor<2x1024x128xf16>
    %producer_empty = tensor.empty() : tensor<2x1024x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%producer_empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x1024x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer, %constant : tensor<2x1024x128xf16>,
                                   tensor<2x1024x128xf16>)
        outs(%consumer_empty : tensor<2x1024x128xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %output: f16):
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
    } -> tensor<2x1024x128xf16>
    return %consumer : tensor<2x1024x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr) << std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
  EXPECT_EQ(proof->finalOwners.size(), 32u);
  EXPECT_EQ(proof->dependencyDemands.size(), 3u);
  unsigned structuredSources = 0;
  unsigned programInputs = 0;
  unsigned constants = 0;
  for (const DependencyDemand &dependency : proof->dependencyDemands)
    for (const DestinationDemand &destination : dependency.perDestination)
      for (const SourceDemand &source : destination.sources) {
        structuredSources +=
            std::holds_alternative<StructuredResultSource>(source.source);
        programInputs +=
            std::holds_alternative<ProgramInputSource>(source.source);
        constants += std::holds_alternative<ConstantSource>(source.source);
      }
  EXPECT_GT(structuredSources, 0u);
  EXPECT_GT(programInputs, 0u);
  EXPECT_GT(constants, 0u);
  EXPECT_EQ(print(module->getOperation()), before);

  ExactDemandOutcome repeated = session->query(coordinate->assignment);
  const ExactDemandProof *repeatedProof = getExactDemandProof(repeated);
  ASSERT_NE(repeatedProof, nullptr);
  ASSERT_EQ(repeatedProof->finalOwners.size(), proof->finalOwners.size());
  EXPECT_EQ(repeatedProof->finalOwners.front().tile,
            proof->finalOwners.front().tile);

  SpatialAssignment permuted = coordinate->assignment;
  for (NodeExecutionPartition &node : permuted.nodes)
    std::reverse(node.shards.begin(), node.shards.end());
  ExactDemandOutcome permutedOutcome = session->query(permuted);
  const ExactDemandProof *permutedProof = getExactDemandProof(permutedOutcome);
  ASSERT_NE(permutedProof, nullptr);
  ASSERT_EQ(permutedProof->finalOwners.size(), proof->finalOwners.size());
  for (auto [expected, actual] :
       llvm::zip_equal(proof->finalOwners, permutedProof->finalOwners)) {
    EXPECT_EQ(actual.root, expected.root);
    EXPECT_EQ(actual.result, expected.result);
    EXPECT_EQ(actual.shard, expected.shard);
    EXPECT_EQ(actual.reductionGroup, expected.reductionGroup);
    EXPECT_EQ(actual.tile, expected.tile);
    ASSERT_EQ(actual.domain.getBoxes().size(),
              expected.domain.getBoxes().size());
    for (auto [expectedBox, actualBox] : llvm::zip_equal(
             expected.domain.getBoxes(), actual.domain.getBoxes())) {
      EXPECT_EQ(actualBox.offsets, expectedBox.offsets);
      EXPECT_EQ(actualBox.sizes, expectedBox.sizes);
    }
  }

  SpatialAssignment changedTile = coordinate->assignment;
  ASSERT_GE(changedTile.nodes.front().shards.size(), 2u);
  std::swap(changedTile.nodes.front().shards[0].tile,
            changedTile.nodes.front().shards[1].tile);
  ExactDemandOutcome changed = session->query(changedTile);
  const ExactDemandProof *changedProof = getExactDemandProof(changed);
  ASSERT_NE(changedProof, nullptr);
  ASSERT_EQ(changedProof->finalOwners.size(), proof->finalOwners.size());
  EXPECT_NE(changedProof->finalOwners.front().tile,
            proof->finalOwners.front().tile)
      << "assignment memo key must include physical Tile binding";

  session->close();
  ExactDemandOutcome closed = session->query(coordinate->assignment);
  EXPECT_TRUE(std::holds_alternative<BrokenDemandContract>(closed));
}

TEST_F(StructuredDemandAnalysisTest,
       PreservesExactEmptyBranchesInMultiOperandSupportGraph) {
  auto module = parse(R"mlir(
#source_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#dest_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @insert(%source_input: tensor<2x512x128xf16>,
                    %dest_input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %source_empty = tensor.empty() : tensor<2x512x128xf16>
    %source = linalg.generic {
        indexing_maps = [#source_map, #source_map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%source_input : tensor<2x512x128xf16>)
        outs(%source_empty : tensor<2x512x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x512x128xf16>
    %dest_empty = tensor.empty() : tensor<2x1025x128xf16>
    %dest = linalg.generic {
        indexing_maps = [#dest_map, #dest_map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%dest_input : tensor<2x1025x128xf16>)
        outs(%dest_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %inserted = tensor.insert_slice %source into %dest[0, 1, 0]
        [2, 512, 128] [1, 1, 1]
        : tensor<2x512x128xf16> into tensor<2x1025x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#dest_map, #dest_map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%inserted : tensor<2x1025x128xf16>)
        outs(%consumer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    return %consumer : tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr);

  auto consumer = llvm::find_if(
      proof->dependencyDemands, [](const DependencyDemand &dependency) {
        return llvm::any_of(dependency.perDestination,
                            [](const DestinationDemand &destination) {
                              return llvm::any_of(
                                  destination.reconstruction.steps,
                                  [](const TensorTransform &step) {
                                    return step.kind ==
                                           TensorTransformKind::InsertSlice;
                                  });
                            });
      });
  ASSERT_NE(consumer, proof->dependencyDemands.end());
  ASSERT_EQ(consumer->perDestination.size(), 16u);
  unsigned destinationsWithExactEmptySource = 0;
  unsigned destinationsCrossingInsertBoundary = 0;
  for (const DestinationDemand &destination : consumer->perDestination) {
    ASSERT_EQ(destination.sources.size(), 2u);
    const bool hasEmptySource =
        llvm::any_of(destination.sources, [](const SourceDemand &source) {
          return source.requiredDomain.isEmpty();
        });
    if (hasEmptySource)
      ++destinationsWithExactEmptySource;
    else
      ++destinationsCrossingInsertBoundary;
    EXPECT_TRUE(
        llvm::any_of(destination.sources, [](const SourceDemand &source) {
          return !source.requiredDomain.isEmpty();
        }));
    ASSERT_EQ(destination.reconstruction.steps.size(), 1u);
    EXPECT_EQ(destination.reconstruction.steps.front().kind,
              TensorTransformKind::InsertSlice);
  }
  EXPECT_EQ(destinationsWithExactEmptySource, 14u);
  EXPECT_EQ(destinationsCrossingInsertBoundary, 2u);
}

TEST_F(StructuredDemandAnalysisTest,
       PreservesMultiResultFanoutAndFaninSourceIdentity) {
  auto module = parse(R"mlir(
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @multi_result(%input: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %split_empty0 = tensor.empty() : tensor<2x1024x128xf16>
    %split_empty1 = tensor.empty() : tensor<2x1024x128xf16>
    %first, %second = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%split_empty0, %split_empty1
            : tensor<2x1024x128xf16>, tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %output0: f16, %output1: f16):
        linalg.yield %value, %value : f16, f16
    } -> (tensor<2x1024x128xf16>, tensor<2x1024x128xf16>)
    %left_empty = tensor.empty() : tensor<2x1024x128xf16>
    %left = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%first : tensor<2x1024x128xf16>)
        outs(%left_empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    %right_empty = tensor.empty() : tensor<2x1024x128xf16>
    %right = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%first : tensor<2x1024x128xf16>)
        outs(%right_empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    %sum_empty = tensor.empty() : tensor<2x1024x128xf16>
    %sum = linalg.generic {
        indexing_maps = [#id, #id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right, %second : tensor<2x1024x128xf16>,
              tensor<2x1024x128xf16>, tensor<2x1024x128xf16>)
        outs(%sum_empty : tensor<2x1024x128xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %extra: f16, %output: f16):
        %pair = arith.addf %lhs, %rhs : f16
        %value = arith.addf %pair, %extra : f16
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    return %sum : tensor<2x1024x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr);
  EXPECT_EQ(proof->finalOwners.size(), 80u);
  EXPECT_EQ(proof->dependencyDemands.size(), 6u);

  mlir::Operation *split = nullptr;
  module->walk([&](mlir::linalg::GenericOp generic) {
    if (generic->getNumResults() == 2)
      split = generic;
  });
  ASSERT_NE(split, nullptr);
  const SemanticRootBinding *binding = coordinate->semanticRoots.find(split);
  ASSERT_NE(binding, nullptr);
  for (uint32_t result = 0; result < 2; ++result)
    EXPECT_EQ(llvm::count_if(proof->finalOwners,
                             [&](const auto &owner) {
                               return owner.root == binding->key &&
                                      owner.result == result;
                             }),
              16u);

  unsigned consumersOfFirstResult = 0;
  for (const DependencyDemand &dependency : proof->dependencyDemands) {
    const bool consumesFirst = llvm::any_of(
        dependency.perDestination, [&](const DestinationDemand &destination) {
          return llvm::any_of(
              destination.sources, [&](const SourceDemand &source) {
                const auto *structured =
                    std::get_if<StructuredResultSource>(&source.source);
                return structured && structured->root == binding->key &&
                       structured->result == 0;
              });
        });
    consumersOfFirstResult += consumesFirst;
  }
  EXPECT_EQ(consumersOfFirstResult, 2u);
}

TEST_F(StructuredDemandAnalysisTest,
       ComposesRaggedStridedReshapeAndPadSupportExactly) {
  auto module = parse(R"mlir(
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @support(%input: tensor<2x2049x128xf16>)
      -> tensor<2x1026x128xf16> {
    %zero = arith.constant 0.0 : f16
    %slice = tensor.extract_slice %input[0, 0, 0] [2, 1025, 128]
        [1, 2, 1] : tensor<2x2049x128xf16> to tensor<2x1025x128xf16>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
        : tensor<2x1025x128xf16> into tensor<2050x128xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]]
        output_shape [2, 1025, 128]
        : tensor<2050x128xf16> into tensor<2x1025x128xf16>
    %padded = tensor.pad %expanded low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%batch: index, %sequence: index, %feature: index):
        tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    %empty = tensor.empty() : tensor<2x1026x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%padded : tensor<2x1026x128xf16>)
        outs(%empty : tensor<2x1026x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1026x128xf16>
    return %result : tensor<2x1026x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr);
  ASSERT_EQ(proof->dependencyDemands.size(), 1u);
  ASSERT_EQ(proof->dependencyDemands.front().perDestination.size(), 16u);

  bool sawStridedProgramInput = false;
  for (const DestinationDemand &destination :
       proof->dependencyDemands.front().perDestination) {
    llvm::SmallVector<TensorTransformKind, 4> kinds;
    for (const TensorTransform &step : destination.reconstruction.steps)
      kinds.push_back(step.kind);
    EXPECT_TRUE(llvm::is_contained(kinds, TensorTransformKind::ExtractSlice));
    EXPECT_TRUE(llvm::is_contained(kinds, TensorTransformKind::CollapseShape));
    EXPECT_TRUE(llvm::is_contained(kinds, TensorTransformKind::ExpandShape));
    EXPECT_TRUE(llvm::is_contained(kinds, TensorTransformKind::Pad));
    for (const SourceDemand &source : destination.sources) {
      if (!std::holds_alternative<ProgramInputSource>(source.source) ||
          source.requiredDomain.isEmpty())
        continue;
      if (source.requiredDomain.getPresburgerSet().containsPoint({0, 0, 0})) {
        sawStridedProgramInput = true;
        EXPECT_TRUE(
            source.requiredDomain.getPresburgerSet().containsPoint({0, 2, 0}));
        EXPECT_FALSE(
            source.requiredDomain.getPresburgerSet().containsPoint({0, 1, 0}));
      }
    }
  }
  EXPECT_TRUE(sawStridedProgramInput);
}

TEST_F(StructuredDemandAnalysisTest,
       DerivesPerOutputSpatialReductionRequirementAndFinalOwner) {
  auto module = parse(R"mlir(
#input = affine_map<(b, k0, k1, n) -> (b, k0, k1, n)>
#output = affine_map<(b, k0, k1, n) -> (b, n)>
module {
  func.func @reduce(%input: tensor<2x33x33x1024xf16>)
      -> tensor<2x1024xf16> {
    %zero = arith.constant 0.0 : f16
    %init_empty = tensor.empty() : tensor<2x1024xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%init_empty : tensor<2x1024xf16>) -> tensor<2x1024xf16>
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
        ins(%input : tensor<2x33x33x1024xf16>)
        outs(%init : tensor<2x1024xf16>) {
      ^bb0(%value: f16, %accumulator: f16):
        %sum = arith.addf %value, %accumulator : f16
        linalg.yield %sum : f16
    } -> tensor<2x1024xf16>
    return %result : tensor<2x1024xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;

  mlir::Operation *reduction = nullptr;
  module->walk([&](mlir::linalg::GenericOp generic) {
    if (llvm::is_contained(generic.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      reduction = generic;
  });
  ASSERT_NE(reduction, nullptr);
  const SemanticRootBinding *binding =
      coordinate->semanticRoots.find(reduction);
  ASSERT_NE(binding, nullptr);
  auto planNode =
      llvm::find_if(coordinate->plan.nodes, [&](const NodeSpatialPlan &node) {
        return node.root == binding->key;
      });
  ASSERT_NE(planNode, coordinate->plan.nodes.end());
  planNode->axes[0].parameter = 2;
  planNode->axes[1].parameter = 2;
  planNode->axes[2].parameter = 4;
  planNode->axes[3].parameter = 1;
  planNode->embedding = allTiles();
  ReductionGroupId firstGroup{binding->key, 0, {0, 0}};
  ReductionGroupId secondGroup{binding->key, 0, {1, 0}};
  planNode->reductionMerges = {{firstGroup, TileId(0)},
                               {secondGroup, TileId(8)}};
  auto spatial = closeSpatialPlanStructure(coordinate->problem,
                                           coordinate->plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(spatial)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(*spatial);
  const ExactDemandProof *proof = getExactDemandProof(outcome);
  ASSERT_NE(proof, nullptr) << std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
  ASSERT_EQ(proof->reductionMerges.size(), 2u);
  EXPECT_EQ(proof->reductionMerges[0].group, firstGroup);
  EXPECT_EQ(proof->reductionMerges[0].mergeTile, TileId(0));
  EXPECT_EQ(proof->reductionMerges[1].group, secondGroup);
  EXPECT_EQ(proof->reductionMerges[1].mergeTile, TileId(8));
  for (const ReductionMergeRequirement &requirement : proof->reductionMerges)
    EXPECT_EQ(requirement.contributions.size(), 8u);
  ASSERT_EQ(proof->finalOwners.size(), 18u);
  EXPECT_EQ(llvm::count_if(proof->finalOwners,
                           [](const auto &owner) {
                             return owner.reductionGroup.has_value();
                           }),
            2u);
}

TEST_F(StructuredDemandAnalysisTest,
       InvalidAssignmentAndUnsupportedTransferRemainTyped) {
  // A compact shape is intentional: this is a typed single-fault negative.
  auto module = parse(R"mlir(
#id = affine_map<(d0) -> (d0)>
module {
  func.func @unsupported(%input: tensor<4xf16>, %condition: i1)
      -> tensor<4xf16> {
    %other = arith.select %condition, %input, %input : tensor<4xf16>
    %empty = tensor.empty() : tensor<4xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%other : tensor<4xf16>) outs(%empty : tensor<4xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate = buildCanonicalSpatialAssignment(
      *dag, {TileId(0), TileId(1)}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, IndexRelationLimits(),
                                               &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome unsupported = session->query(coordinate->assignment);
  EXPECT_TRUE(std::holds_alternative<UnsupportedDemandSemantics>(unsupported));

  SpatialAssignment invalid = coordinate->assignment;
  invalid.nodes.front().shards.pop_back();
  ExactDemandOutcome malformed = session->query(invalid);
  EXPECT_TRUE(std::holds_alternative<InvalidSpatialAssignment>(malformed));
}

TEST_F(StructuredDemandAnalysisTest,
       WorkLimitStopsBeforeExpandingInsertRemainderPieces) {
  auto module = parse(R"mlir(
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @limited(%source: tensor<2x17x128xf16>,
                     %dest: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %inserted = tensor.insert_slice %source into %dest[0, 64, 0]
        [2, 17, 128] [1, 1, 1]
        : tensor<2x17x128xf16> into tensor<2x1025x128xf16>
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%inserted : tensor<2x1025x128xf16>)
        outs(%empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    return %result : tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, {TileId(0)}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  IndexRelationLimits limits;
  limits.maxRectangularPieces = 1;
  auto session = DemandPlanningSession::create(*dag, limits, &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  ExactDemandOutcome outcome = session->query(coordinate->assignment);
  EXPECT_TRUE(std::holds_alternative<DemandWorkLimitReached>(outcome));
}

TEST_F(StructuredDemandAnalysisTest,
       AnalysisManagerPreservesReadOnlyFactsAndRebuildsAfterMutation) {
  auto module = parse(R"mlir(
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @analysis(%input: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    return %result : tensor<2x1024x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ObservedStructuredRelationAnalysis::constructionCount = 0;
  llvm::SmallVector<unsigned, 3> observations;
  mlir::PassManager manager(context.get());
  manager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ObserveRelationPass>(&observations));
  manager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ObserveRelationPass>(&observations));
  manager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<MutateRelationInputPass>());
  manager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ObserveRelationPass>(&observations));
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
  EXPECT_EQ(observations, (llvm::SmallVector<unsigned, 3>{1, 1, 2}));
}

TEST(StructuredDemandRootFactTest,
     NonLinalgRootKeepsTypedUnsupportedDemandOutcome) {
  // `tensor.pack` is DestinationStyle and TilingInterface, so it enters the
  // structured DAG, but it is neither a linalg op nor an attention op and
  // therefore cannot expose operand/result indexing relations. The typed
  // reason must survive as an unsupported demand outcome that closes one
  // choice, instead of becoming a broken-contract session failure. The tensor
  // tiling models are registered here because production registers them too.
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModelsForPackUnPackOps(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<2x1024x128xf16>) -> tensor<2x1024x128xf16> {
    %packed_empty = tensor.empty() : tensor<1x8x128x2x128xf16>
    %packed = tensor.pack %input inner_dims_pos = [0, 1] inner_tiles = [2, 128]
        into %packed_empty : tensor<2x1024x128xf16> -> tensor<1x8x128x2x128xf16>
    %result_empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = tensor.unpack %packed inner_dims_pos = [0, 1]
        inner_tiles = [2, 128]
        into %result_empty : tensor<1x8x128x2x128xf16> -> tensor<2x1024x128xf16>
    return %result : tensor<2x1024x128xf16>
  }
}
)mlir",
                                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto function = *module->getOps<mlir::func::FuncOp>().begin();

  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;

  auto session = DemandPlanningSession::create(*dag, {}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  EXPECT_TRUE(session->hasUnanalyzableRoot());
  // The session holds typed facts, so no spatial assignment can satisfy them;
  // the outcome category is what PlanningSession maps to a closed choice.
  ExactDemandOutcome outcome = session->query(SpatialAssignment{});
  EXPECT_EQ(classifyExactDemandOutcome(outcome),
            ExactDemandOutcomeCategory::UnsupportedSemantics);
  ASSERT_TRUE(std::holds_alternative<UnsupportedDemandSemantics>(outcome));
  EXPECT_FALSE(std::get<UnsupportedDemandSemantics>(outcome).detail.empty());
}

TEST(StructuredDemandRootFactTest, LinalgProgramHasNoUnanalyzableRoot) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<2x1024x128xf16>) -> tensor<2x1024x128xf16> {
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.map ins(%input : tensor<2x1024x128xf16>)
        outs(%empty : tensor<2x1024x128xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<2x1024x128xf16>
  }
}
)mlir",
                                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto session = DemandPlanningSession::create(*dag, {}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(session)) << failureReason;
  EXPECT_FALSE(session->hasUnanalyzableRoot());
}

} // namespace
