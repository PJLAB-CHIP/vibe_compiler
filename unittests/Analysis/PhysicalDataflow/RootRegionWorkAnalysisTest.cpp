//===- RootRegionWorkAnalysisTest.cpp ---------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class RootRegionWorkAnalysisTest : public ::testing::Test {
protected:
  RootRegionWorkAnalysisTest() {
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

  static std::string outcomeDetail(const RootRegionWorkOutcome &outcome) {
    return std::visit(
        [](const auto &value) -> std::string {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, RootRegionWork> ||
                        std::is_same_v<T, NoRootRegionWork>)
            return {};
          else
            return value.detail;
        },
        outcome);
  }

  static mlir::FailureOr<ExactDemandProof>
  deriveDemand(const StructuredDAGAnalysis &dag,
               const SpatialAssignment &assignment,
               std::string *failureReason) {
    auto session = DemandPlanningSession::create(dag, IndexRelationLimits(),
                                                 failureReason);
    if (mlir::failed(session))
      return mlir::failure();
    ExactDemandOutcome outcome = session->query(assignment);
    const ExactDemandProof *proof = getExactDemandProof(outcome);
    if (!proof) {
      if (failureReason)
        *failureReason = std::visit(
            [](const auto &value) -> std::string {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, ExactDemandProof>)
                return {};
              else
                return value.detail;
            },
            outcome);
      return mlir::failure();
    }
    return *proof;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(RootRegionWorkAnalysisTest,
       DeduplicatesRaggedMultiProducerSupportAndKeepsExactEmptyBoundaries) {
  auto module = parse(R"mlir(
#left = affine_map<(b, m, n) -> (b, m, n)>
#right = affine_map<(b, m, n) -> (b, m, n)>
#full = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @fanin(%lhs: tensor<2x513x128xf16>,
                   %rhs: tensor<2x512x128xf16>)
      -> tensor<2x1025x128xf16> {
    %left_empty = tensor.empty() : tensor<2x513x128xf16>
    %left_value = linalg.generic {
        indexing_maps = [#left, #left],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%lhs : tensor<2x513x128xf16>)
        outs(%left_empty : tensor<2x513x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x513x128xf16>
    %right_empty = tensor.empty() : tensor<2x512x128xf16>
    %right_value = linalg.generic {
        indexing_maps = [#right, #right],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%rhs : tensor<2x512x128xf16>)
        outs(%right_empty : tensor<2x512x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x512x128xf16>
    %assembled_empty = tensor.empty() : tensor<2x1025x128xf16>
    %with_left = tensor.insert_slice %left_value into %assembled_empty
        [0, 0, 0] [2, 513, 128] [1, 1, 1]
        : tensor<2x513x128xf16> into tensor<2x1025x128xf16>
    %assembled = tensor.insert_slice %right_value into %with_left
        [0, 513, 0] [2, 512, 128] [1, 1, 1]
        : tensor<2x512x128xf16> into tensor<2x1025x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#full, #full],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%assembled : tensor<2x1025x128xf16>)
        outs(%consumer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    return %consumer : tensor<2x1025x128xf16>
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;

  mlir::Operation *consumer = nullptr;
  module->walk([&](mlir::linalg::GenericOp generic) {
    if (generic.getResult(0).getType() ==
        mlir::RankedTensorType::get({2, 1025, 128},
                                    mlir::Float16Type::get(context.get())))
      consumer = generic;
  });
  ASSERT_NE(consumer, nullptr);
  const SemanticRootBinding *binding = coordinate->semanticRoots.find(consumer);
  ASSERT_NE(binding, nullptr);
  auto consumerNode =
      llvm::find_if(dag->getNodes(), [&](const StructuredDAGNode &node) {
        return node.operation == consumer;
      });
  ASSERT_NE(consumerNode, dag->getNodes().end());

  unsigned exactEmptyBoundaries = 0;
  std::map<RootBoundaryId, unsigned> nonemptyBoundaries;
  llvm::SmallVector<SupportValueId, 2> firstSupportIds;
  llvm::SmallVector<RootBoundaryId, 2> firstBoundaryIds;
  for (TileId tile : allTiles()) {
    RootRegionWorkOutcome outcome = analysis->query(binding->key, tile);
    const RootRegionWork *work = getRootRegionWork(outcome);
    ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
    ASSERT_EQ(work->execution.size(), 1u);
    ASSERT_EQ(work->operands.size(), 1u);
    ASSERT_EQ(work->supportValues.size(), 2u);
    ASSERT_EQ(work->boundaries.size(), 2u);
    auto leaf = wafer::prepareStructuredRootLeaf(consumerNode->id, *work,
                                                 &failureReason);
    ASSERT_TRUE(mlir::succeeded(leaf)) << failureReason;
    ASSERT_TRUE(leaf->has_value());
    EXPECT_EQ((*leaf)->tile, tile);
    ASSERT_EQ((*leaf)->offsets.size(),
              work->execution.front().iterationDomain.size());
    for (auto [offset, size, interval] :
         llvm::zip_equal((*leaf)->offsets, (*leaf)->sizes,
                         work->execution.front().iterationDomain)) {
      EXPECT_EQ(offset, interval.offset);
      EXPECT_EQ(size, interval.size);
    }
    EXPECT_EQ(work->supportValues[0].operation->getNumResults(), 1u);
    EXPECT_NE(work->supportValues[0].id, work->supportValues[1].id);
    EXPECT_TRUE(llvm::all_of(work->supportValues, [](const auto &support) {
      return support.inputs.size() == 2;
    }));
    if (tile == TileId(0)) {
      for (const RootSupportValueWork &support : work->supportValues)
        firstSupportIds.push_back(support.id);
      for (const RootBoundaryWork &boundary : work->boundaries)
        firstBoundaryIds.push_back(boundary.id);
    }
    unsigned nonempty = 0;
    for (const RootBoundaryWork &boundary : work->boundaries) {
      ASSERT_TRUE(boundary.requiredDomain.has_value());
      if (boundary.requiredDomain->isEmpty())
        ++exactEmptyBoundaries;
      else {
        ++nonempty;
        ++nonemptyBoundaries[boundary.id];
      }
      EXPECT_EQ(boundary.id.kind, RootBoundaryKind::StructuredResult);
      ASSERT_EQ(boundary.consumerUses.size(), 1u);
      ASSERT_TRUE(boundary.consumerUses.front().requiredDomain.has_value());
      if (!boundary.consumerUses.front().requiredDomain->isEmpty())
        EXPECT_FALSE(
            boundary.consumerUses.front().eligibleFinalOwners.empty());
    }
    EXPECT_GE(nonempty, 1u);
  }
  EXPECT_GT(exactEmptyBoundaries, 0u);
  ASSERT_EQ(nonemptyBoundaries.size(), 2u);
  EXPECT_TRUE(llvm::all_of(nonemptyBoundaries,
                           [](const auto &entry) { return entry.second > 0; }));
  SpatialAssignment permuted = coordinate->assignment;
  for (NodeExecutionPartition &node : permuted.nodes)
    std::reverse(node.shards.begin(), node.shards.end());
  auto permutedAnalysis =
      RootRegionWorkAnalysis::create(*dag, permuted, *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(permutedAnalysis)) << failureReason;
  RootRegionWorkOutcome permutedOutcome =
      permutedAnalysis->query(binding->key, TileId(0));
  const RootRegionWork *permutedWork = getRootRegionWork(permutedOutcome);
  ASSERT_NE(permutedWork, nullptr) << outcomeDetail(permutedOutcome);
  EXPECT_TRUE(llvm::equal(
      firstSupportIds, llvm::map_range(permutedWork->supportValues,
                                       [](const RootSupportValueWork &support) {
                                         return support.id;
                                       })));
  EXPECT_TRUE(llvm::equal(firstBoundaryIds,
                          llvm::map_range(permutedWork->boundaries,
                                          [](const RootBoundaryWork &boundary) {
                                            return boundary.id;
                                          })));
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST_F(RootRegionWorkAnalysisTest,
       PreservesAlignedMultiResultAndDpsInitBoundaries) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @pair(%input: tensor<2x1024x128xf16>,
                  %lhs_init: tensor<2x1024x128xf16>,
                  %rhs_init: tensor<2x1024x128xf16>)
      -> (tensor<2x1024x128xf16>, tensor<2x1024x128xf16>) {
    %lhs, %rhs = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%lhs_init, %rhs_init : tensor<2x1024x128xf16>,
                                     tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %old_lhs: f16, %old_rhs: f16):
        %next_lhs = arith.addf %value, %old_lhs : f16
        %next_rhs = arith.mulf %value, %old_rhs : f16
        linalg.yield %next_lhs, %next_rhs : f16, f16
    } -> (tensor<2x1024x128xf16>, tensor<2x1024x128xf16>)
    return %lhs, %rhs : tensor<2x1024x128xf16>, tensor<2x1024x128xf16>
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  const SemanticRootKey root = coordinate->semanticRoots.getRoots().front().key;
  for (TileId tile : allTiles()) {
    RootRegionWorkOutcome outcome = analysis->query(root, tile);
    const RootRegionWork *work = getRootRegionWork(outcome);
    ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
    EXPECT_EQ(work->results.size(), 2u);
    ASSERT_EQ(work->operands.size(), 3u);
    EXPECT_EQ(work->operands[0].kind, DemandOperandKind::DataInput);
    EXPECT_EQ(work->operands[1].kind, DemandOperandKind::InitInput);
    EXPECT_EQ(work->operands[2].kind, DemandOperandKind::InitInput);
    ASSERT_EQ(work->boundaries.size(), 3u);
    EXPECT_TRUE(llvm::all_of(work->boundaries, [](const auto &boundary) {
      return boundary.id.kind == RootBoundaryKind::ProgramInput &&
             boundary.requiredDomain.has_value() &&
             !boundary.requiredDomain->isEmpty();
    }));
  }
}

TEST_F(RootRegionWorkAnalysisTest,
       SameProducerMultiPathUnionsOneStructuredBoundaryPerRootUse) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @diamond(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%producer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %left = tensor.extract_slice %producer[0, 0, 0] [2, 500, 128]
        [1, 1, 1] : tensor<2x1025x128xf16> to tensor<2x500x128xf16>
    %right = tensor.extract_slice %producer[0, 500, 0] [2, 525, 128]
        [1, 1, 1] : tensor<2x1025x128xf16> to tensor<2x525x128xf16>
    %assembled_empty = tensor.empty() : tensor<2x1025x128xf16>
    %with_left = tensor.insert_slice %left into %assembled_empty
        [0, 0, 0] [2, 500, 128] [1, 1, 1]
        : tensor<2x500x128xf16> into tensor<2x1025x128xf16>
    %assembled = tensor.insert_slice %right into %with_left
        [0, 500, 0] [2, 525, 128] [1, 1, 1]
        : tensor<2x525x128xf16> into tensor<2x1025x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%assembled : tensor<2x1025x128xf16>)
        outs(%consumer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  mlir::Operation *consumer = nullptr;
  module->walk([&](mlir::linalg::GenericOp generic) {
    if (generic.getOperand(0).getDefiningOp<mlir::tensor::InsertSliceOp>())
      consumer = generic;
  });
  ASSERT_NE(consumer, nullptr);
  const SemanticRootBinding *binding = coordinate->semanticRoots.find(consumer);
  ASSERT_NE(binding, nullptr);

  unsigned crossingUses = 0;
  for (TileId tile : allTiles()) {
    RootRegionWorkOutcome outcome = analysis->query(binding->key, tile);
    const RootRegionWork *work = getRootRegionWork(outcome);
    ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
    ASSERT_EQ(work->supportValues.size(), 4u);
    auto structured =
        llvm::find_if(work->boundaries, [](const RootBoundaryWork &boundary) {
          return boundary.id.kind == RootBoundaryKind::StructuredResult;
        });
    ASSERT_NE(structured, work->boundaries.end());
    EXPECT_EQ(llvm::count_if(work->boundaries,
                             [](const RootBoundaryWork &boundary) {
                               return boundary.id.kind ==
                                      RootBoundaryKind::StructuredResult;
                             }),
              1u);
    ASSERT_TRUE(structured->requiredDomain.has_value());
    ASSERT_EQ(structured->consumerUses.size(), 1u);
    ASSERT_TRUE(structured->consumerUses.front().requiredDomain.has_value());
    EXPECT_FALSE(structured->consumerUses.front().eligibleFinalOwners.empty());
    if (structured->requiredDomain->getPresburgerSet().containsPoint(
            {0, 499, 0}) &&
        structured->requiredDomain->getPresburgerSet().containsPoint(
            {0, 500, 0}))
      ++crossingUses;
  }
  EXPECT_GT(crossingUses, 0u);
}

TEST_F(RootRegionWorkAnalysisTest,
       PadSupportKeepsExactSourceDemandAndScalarCaptureInOneWorkNode) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @pad(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1026x128xf16> {
    %zero = arith.constant 0.0 : f16
    %padded = tensor.pad %input low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%b: index, %m: index, %n: index):
        tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    %empty = tensor.empty() : tensor<2x1026x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%padded : tensor<2x1026x128xf16>)
        outs(%empty : tensor<2x1026x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1026x128xf16>
    return %result : tensor<2x1026x128xf16>
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  const SemanticRootKey root = coordinate->semanticRoots.getRoots().front().key;

  unsigned clippedInputWorks = 0;
  for (TileId tile : allTiles()) {
    RootRegionWorkOutcome outcome = analysis->query(root, tile);
    const RootRegionWork *work = getRootRegionWork(outcome);
    ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
    ASSERT_EQ(work->supportValues.size(), 1u);
    const RootSupportValueWork &support = work->supportValues.front();
    ASSERT_EQ(support.captures.size(), 1u);
    ASSERT_EQ(support.inputs.size(), 1u);
    EXPECT_EQ(support.inputs.front().kind, RootSupportInputKind::Boundary);
    auto capture =
        llvm::find_if(work->boundaries, [&](const RootBoundaryWork &boundary) {
          return boundary.id == support.captures.front();
        });
    ASSERT_NE(capture, work->boundaries.end());
    EXPECT_EQ(capture->id.kind, RootBoundaryKind::Constant);
    EXPECT_FALSE(capture->requiredDomain.has_value());
    auto input =
        llvm::find_if(work->boundaries, [](const RootBoundaryWork &boundary) {
          return boundary.id.kind == RootBoundaryKind::ProgramInput;
        });
    ASSERT_NE(input, work->boundaries.end());
    ASSERT_TRUE(input->requiredDomain.has_value());
    ASSERT_EQ(input->requiredDomain->getBoxes().size(), 1u);
    ASSERT_EQ(support.requiredDomain.getBoxes().size(), 1u);
    const StaticRectangularIndexSet &inputBox =
        input->requiredDomain->getBoxes().front();
    const StaticRectangularIndexSet &outputBox =
        support.requiredDomain.getBoxes().front();
    clippedInputWorks += inputBox.sizes[1] < outputBox.sizes[1];
    ASSERT_EQ(support.inputs.front().requiredDomain.getBoxes().size(), 1u);
    EXPECT_EQ(support.inputs.front().requiredDomain.getBoxes().front().offsets,
              inputBox.offsets);
    EXPECT_EQ(support.inputs.front().requiredDomain.getBoxes().front().sizes,
              inputBox.sizes);
  }
  EXPECT_GT(clippedInputWorks, 0u);
}

TEST_F(RootRegionWorkAnalysisTest,
       SpatialReductionContributionAndMergeWorkMatchAlignedAndRaggedProofs) {
  for (int64_t reductionExtent : {32, 33}) {
    SCOPED_TRACE(reductionExtent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#input = affine_map<(b, k0, k1, n) -> (b, k0, k1, n)>
#output = affine_map<(b, k0, k1, n) -> (b, n)>
module {
  func.func @reduce(%input: tensor<2x)mlir"
           << reductionExtent << "x" << reductionExtent
           << "x1025xf16>) -> tensor<2x1025xf16> {\n"
           << "    %zero = arith.constant 0.0 : f16\n"
           << "    %init_empty = tensor.empty() : tensor<2x1025xf16>\n"
           << "    %init = linalg.fill ins(%zero : f16)\n"
           << "        outs(%init_empty : tensor<2x1025xf16>) "
              "-> tensor<2x1025xf16>\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#input, #output],\n"
           << "        iterator_types = [\"parallel\", \"reduction\", "
              "\"reduction\", \"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << reductionExtent << "x"
           << reductionExtent << "x1025xf16>)\n"
           << "        outs(%init : tensor<2x1025xf16>) {\n"
           << "      ^bb0(%value: f16, %acc: f16):\n"
           << "        %next = arith.addf %value, %acc : f16\n"
           << "        linalg.yield %next : f16\n"
           << "    } -> tensor<2x1025xf16>\n"
           << "    return %result : tensor<2x1025xf16>\n"
           << "  }\n"
           << "}\n";
    auto module = parse(source);
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate =
        buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    NodeSpatialPlan &node = coordinate->plan.nodes.front();
    node.axes = {{0, IteratorPartitionScheme::BalancedParts, 2},
                 {1, IteratorPartitionScheme::BalancedParts, 2},
                 {2, IteratorPartitionScheme::BalancedParts, 4},
                 {3, IteratorPartitionScheme::BalancedParts, 1}};
    node.embedding = allTiles();
    node.reductionMerges = {
        {{node.root, 0, {0, 0}}, TileId(0)},
        {{node.root, 0, {1, 0}}, TileId(0)},
    };
    auto assignment = closeSpatialPlanStructure(
        coordinate->problem, coordinate->plan, &failureReason);
    ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
    auto proof = deriveDemand(*dag, *assignment, &failureReason);
    ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
    auto analysis = RootRegionWorkAnalysis::create(*dag, *assignment, *proof,
                                                   &failureReason);
    ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;

    unsigned contributions = 0;
    unsigned merges = 0;
    unsigned finalResults = 0;
    for (TileId tile : allTiles()) {
      RootRegionWorkOutcome outcome = analysis->query(node.root, tile);
      const RootRegionWork *work = getRootRegionWork(outcome);
      ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
      contributions += work->contributions.size();
      merges += work->merges.size();
      finalResults += work->results.size();
      ASSERT_EQ(work->contributions.size(), 1u);
      EXPECT_EQ(work->contributions.front().contribution.tile, tile);
      if (tile == TileId(0)) {
        ASSERT_EQ(work->merges.size(), 2u);
        ASSERT_EQ(work->results.size(), 2u);
        for (auto [result, merge] :
             llvm::zip_equal(work->results, work->merges))
          EXPECT_EQ(result.reductionGroup, merge.group);
      } else {
        EXPECT_TRUE(work->merges.empty());
        EXPECT_TRUE(work->results.empty());
      }
    }
    EXPECT_EQ(contributions, 16u);
    EXPECT_EQ(merges, 2u);
    EXPECT_EQ(finalResults, 2u);
  }
}

TEST_F(RootRegionWorkAnalysisTest,
       CarriesStableScalarRegionCaptureWithoutASecondClosureWalk) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @captured(%input: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %bias = arith.constant 1.0 : f16
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %bias : f16
        linalg.yield %next : f16
    } -> tensor<2x1024x128xf16>
    return %result : tensor<2x1024x128xf16>
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  RootRegionWorkOutcome outcome = analysis->query(
      coordinate->semanticRoots.getRoots().front().key, TileId(0));
  const RootRegionWork *work = getRootRegionWork(outcome);
  ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
  ASSERT_EQ(work->invariantInputs.size(), 1u);
  EXPECT_EQ(work->invariantInputs.front().kind,
            RootInvariantUseKind::RegionCapture);
  auto capture =
      llvm::find_if(work->boundaries, [&](const RootBoundaryWork &boundary) {
        return boundary.id == work->invariantInputs.front().boundary;
      });
  ASSERT_NE(capture, work->boundaries.end());
  EXPECT_EQ(capture->id.kind, RootBoundaryKind::Constant);
  EXPECT_FALSE(capture->requiredDomain.has_value());
  EXPECT_TRUE(capture->sourceValue.getDefiningOp<mlir::arith::ConstantOp>());
}

TEST_F(RootRegionWorkAnalysisTest,
       RankZeroUsesOneEmptyVectorExecutionPieceAndNoWorkElsewhere) {
  // A scalar-shaped structured contract is the intentional bounded exception
  // to the representative-shape rule; the same query is covered above at
  // 1024/1025 scale.
  auto module = parse(R"mlir(
#scalar = affine_map<() -> ()>
module {
  func.func @scalar(%input: tensor<f16>) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.generic {
        indexing_maps = [#scalar, #scalar], iterator_types = []}
        ins(%input : tensor<f16>) outs(%empty : tensor<f16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<f16>
    return %result : tensor<f16>
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
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  const SemanticRootKey root = coordinate->semanticRoots.getRoots().front().key;
  RootRegionWorkOutcome local = analysis->query(root, TileId(0));
  const RootRegionWork *work = getRootRegionWork(local);
  ASSERT_NE(work, nullptr) << outcomeDetail(local);
  ASSERT_EQ(work->execution.size(), 1u);
  EXPECT_TRUE(work->execution.front().iterationDomain.empty());
  EXPECT_EQ(work->results.size(), 1u);
  RootRegionWorkOutcome empty = analysis->query(root, TileId(1));
  EXPECT_TRUE(std::holds_alternative<NoRootRegionWork>(empty));
}

TEST_F(RootRegionWorkAnalysisTest,
       FlashDecodingKeepsThreeComponentsInOneRootContributionAndMergeWork) {
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
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  const SemanticRootKey root = coordinate->semanticRoots.getRoots().front().key;

  unsigned contributions = 0;
  unsigned merges = 0;
  unsigned results = 0;
  for (TileId tile : allTiles()) {
    RootRegionWorkOutcome outcome = analysis->query(root, tile);
    const RootRegionWork *work = getRootRegionWork(outcome);
    ASSERT_NE(work, nullptr) << outcomeDetail(outcome);
    ASSERT_EQ(work->execution.size(), 1u);
    ASSERT_EQ(work->operands.size(), 4u);
    ASSERT_EQ(work->invariantInputs.size(), 1u);
    EXPECT_EQ(work->invariantInputs.front().kind,
              RootInvariantUseKind::Operand);
    contributions += work->contributions.size();
    merges += work->merges.size();
    results += work->results.size();
    ASSERT_EQ(work->contributions.size(), 1u);
    const RootContributionWork &contribution = work->contributions.front();
    EXPECT_EQ(contribution.algebra, ReductionAlgebraKind::CoupledReduction);
    EXPECT_TRUE(contribution.coupledRule.has_value());
    EXPECT_TRUE(contribution.contribution.results.empty());
    EXPECT_EQ(contribution.contribution.components.size(), 3u);
    for (const ReductionMergeRequirement &merge : work->merges) {
      EXPECT_EQ(merge.algebra, ReductionAlgebraKind::CoupledReduction);
      EXPECT_EQ(merge.components.size(), 3u);
      EXPECT_EQ(merge.mergeTile, tile);
    }
  }
  unsigned expectedContributions = 0;
  for (const ReductionMergeRequirement &merge : proof->reductionMerges)
    expectedContributions += merge.contributions.size();
  EXPECT_EQ(contributions, expectedContributions);
  EXPECT_EQ(merges, proof->reductionMerges.size());
  EXPECT_EQ(results, proof->reductionMerges.size());
}

TEST_F(RootRegionWorkAnalysisTest,
       MissingBoundaryIsACompilerContractFailureWithoutMutation) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @main(%input: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%empty : tensor<2x1024x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    return %result : tensor<2x1024x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  ASSERT_FALSE(proof->dependencyDemands.empty());
  ASSERT_FALSE(proof->dependencyDemands.front().perDestination.empty());
  proof->dependencyDemands.front().perDestination.front().sources.clear();
  auto analysis = RootRegionWorkAnalysis::create(*dag, coordinate->assignment,
                                                 *proof, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  RootRegionWorkOutcome outcome = analysis->query(
      coordinate->semanticRoots.getRoots().front().key, TileId(0));
  const auto *failure = std::get_if<BrokenRootRegionWork>(&outcome);
  ASSERT_NE(failure, nullptr);
  EXPECT_EQ(failure->reason, BrokenRootRegionWorkReason::MissingBoundary);
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
