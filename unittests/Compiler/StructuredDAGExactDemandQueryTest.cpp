//===- StructuredDAGExactDemandQueryTest.cpp
//------------------------------------------===//

#include "../../lib/Wafer/Compiler/StructuredDAGExactDemandQuery.h"
#include "../../lib/Wafer/Compiler/StructuredDAGCandidateSchedule.h"
#include "../../lib/Wafer/Compiler/StructuredDAGEdgeDemandPlan.h"
#include "../../lib/Wafer/Compiler/StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <string>

using namespace wafer;
using namespace wafer::compiler::detail;

namespace {

class StructuredDAGExactDemandQueryTest : public ::testing::Test {
protected:
  StructuredDAGExactDemandQueryTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  StructuredDAGAnalysis buildDAG(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    EXPECT_TRUE(mlir::succeeded(dag)) << failureReason;
    if (mlir::failed(dag))
      std::abort();
    return std::move(*dag);
  }

  static StructuredDAGNodePlacement
  placement(StructuredDAGNodeID node, unsigned dimension,
            std::initializer_list<int64_t> tileValues) {
    StructuredDAGNodePlacement result;
    result.node = node;
    result.shardDimension = dimension;
    for (int64_t value : tileValues)
      result.tiles.push_back(TileId(value));
    return result;
  }

  static llvm::SmallVector<StructuredDAGNodePlacement, 16>
  fullPlacements(const StructuredDAGAnalysis &dag, unsigned dimension,
                 std::initializer_list<int64_t> tileValues) {
    llvm::SmallVector<StructuredDAGNodePlacement, 16> result;
    for (const StructuredDAGNode &node : dag.getNodes())
      result.push_back(placement(node.id, dimension, tileValues));
    return result;
  }

  static analysis::LogicalShardTrial buildTrial(
      const StructuredDAGAnalysis &dag,
      llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
    std::string failureReason;
    auto trial =
        buildLogicalShardTrial(dag, placements, analysis::IREpoch::current(),
                               &failureReason);
    EXPECT_TRUE(mlir::succeeded(trial)) << failureReason;
    if (mlir::failed(trial))
      std::abort();
    return std::move(*trial);
  }

  static StructuredDAGEdgeID findEdge(const StructuredDAGAnalysis &dag,
                                      StructuredDAGNodeID producer,
                                      StructuredDAGNodeID consumer) {
    for (const StructuredDAGEdge &edge : dag.getEdges())
      if (edge.producer == producer && edge.consumer == consumer)
        return edge.id;
    std::abort();
  }

  static analysis::LogicalNodeTrial &findNode(analysis::LogicalShardTrial &trial,
                                              uint32_t node) {
    for (analysis::LogicalNodeTrial &entry : trial.nodes)
      if (entry.node == node)
        return entry;
    std::abort();
  }

  static void expectDemandPoints(
      const analysis::ExactDemandResult &result,
      std::initializer_list<std::initializer_list<int64_t>> present,
      std::initializer_list<std::initializer_list<int64_t>> absent) {
    ASSERT_TRUE(result.producerDemand);
    for (auto point : present) {
      llvm::SmallVector<int64_t, 4> coords(point);
      EXPECT_TRUE(result.producerDemand->containsPoint(coords))
          << "expected demand point to be present";
    }
    for (auto point : absent) {
      llvm::SmallVector<int64_t, 4> coords(point);
      EXPECT_FALSE(result.producerDemand->containsPoint(coords))
          << "expected demand point to be absent";
    }
  }

  static const analysis::ExactOwnershipIntersection *findIntersection(
      const analysis::ExactDemandResult &result, TileId tile) {
    for (const analysis::ExactOwnershipIntersection &intersection :
         result.ownershipIntersections)
      if (intersection.tile == tile)
        return &intersection;
    return nullptr;
  }

  static constexpr llvm::StringLiteral kChain = R"mlir(
module {
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kTransposeRead = R"mlir(
module {
  func.func @transpose_read(%input: tensor<4x8xf16>) -> tensor<8x4xf16> {
    %out0 = tensor.empty() : tensor<4x8xf16>
    %producer = linalg.map ins(%input : tensor<4x8xf16>)
        outs(%out0 : tensor<4x8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<8x4xf16>
    %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%producer : tensor<4x8xf16>) outs(%out1 : tensor<8x4xf16>) {
    ^bb0(%in: f16, %out: f16):
      %result = arith.addf %in, %out : f16
      linalg.yield %result : f16
    } -> tensor<8x4xf16>
    return %consumer : tensor<8x4xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kStridedSupportChain = R"mlir(
module {
  func.func @strided(%input: tensor<8xf16>) -> tensor<4xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %view = tensor.extract_slice %producer[0] [4] [2] : tensor<8xf16> to tensor<4xf16>
    %out1 = tensor.empty() : tensor<4xf16>
    %consumer = linalg.map ins(%view : tensor<4xf16>)
        outs(%out1 : tensor<4xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<4xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kStridedDirect = R"mlir(
module {
  func.func @strided_direct(%input: tensor<8xf16>) -> tensor<4xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<4xf16>
    %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0 * 2)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%producer : tensor<8xf16>) outs(%out1 : tensor<4xf16>) {
    ^bb0(%in: f16, %out: f16):
      %result = arith.mulf %in, %out : f16
      linalg.yield %result : f16
    } -> tensor<4xf16>
    return %consumer : tensor<4xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kReduction = R"mlir(
module {
  func.func @reduction(%a_input: tensor<4x8xf16>, %b: tensor<8x4xf16>)
      -> tensor<4x4xf16> {
    %out0 = tensor.empty() : tensor<4x8xf16>
    %producer = linalg.map ins(%a_input : tensor<4x8xf16>)
        outs(%out0 : tensor<4x8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<4x4xf16>
    %consumer = linalg.matmul ins(%producer, %b : tensor<4x8xf16>, tensor<8x4xf16>)
        outs(%out1 : tensor<4x4xf16>) -> tensor<4x4xf16>
    return %consumer : tensor<4x4xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kBroadcast = R"mlir(
module {
  func.func @broadcast(%input: tensor<4xf16>) -> tensor<4x8xf16> {
    %out0 = tensor.empty() : tensor<4xf16>
    %producer = linalg.map ins(%input : tensor<4xf16>)
        outs(%out0 : tensor<4xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<4x8xf16>
    %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%producer : tensor<4xf16>) outs(%out1 : tensor<4x8xf16>) {
    ^bb0(%in: f16, %out: f16):
      %result = arith.addf %in, %out : f16
      linalg.yield %result : f16
    } -> tensor<4x8xf16>
    return %consumer : tensor<4x8xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kInitRoot = R"mlir(
module {
  func.func @init_root(%a: tensor<4x8xf16>, %b: tensor<8x4xf16>)
      -> tensor<4x4xf16> {
    %out0 = tensor.empty() : tensor<4x4xf16>
    %c0 = arith.constant 0.0 : f16
    %init = linalg.fill ins(%c0 : f16) outs(%out0 : tensor<4x4xf16>)
        -> tensor<4x4xf16>
    %result = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x4xf16>)
        outs(%init : tensor<4x4xf16>) -> tensor<4x4xf16>
    return %result : tensor<4x4xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kConcatChain = R"mlir(
module {
  func.func @concat_chain(%left: tensor<4xf16>, %right: tensor<4xf16>)
      -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<4xf16>
    %producer_a = linalg.map ins(%left : tensor<4xf16>)
        outs(%out0 : tensor<4xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<4xf16>
    %producer_b = linalg.map ins(%right : tensor<4xf16>)
        outs(%out1 : tensor<4xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    %piece_a = tensor.extract_slice %producer_a[0] [2] [1]
        : tensor<4xf16> to tensor<2xf16>
    %piece_b = tensor.extract_slice %producer_b[1] [2] [1]
        : tensor<4xf16> to tensor<2xf16>
    %empty0 = tensor.empty() : tensor<8xf16>
    %stage0 = tensor.insert_slice %piece_a into %empty0[0] [2] [1]
        : tensor<2xf16> into tensor<8xf16>
    %stage1 = tensor.insert_slice %piece_b into %stage0[4] [2] [1]
        : tensor<2xf16> into tensor<8xf16>
    %out2 = tensor.empty() : tensor<8xf16>
    %consumer = linalg.map ins(%stage1 : tensor<8xf16>)
        outs(%out2 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kAmbiguousSupport = R"mlir(
module {
  func.func @ambiguous(%input: tensor<16xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<16xf16>
    %producer = linalg.map ins(%input : tensor<16xf16>)
        outs(%out0 : tensor<16xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %t = tensor.extract_slice %producer[0] [8] [1] : tensor<16xf16> to tensor<8xf16>
    %empty0 = tensor.empty() : tensor<8xf16>
    %stage = tensor.insert_slice %t into %empty0[0] [8] [1]
        : tensor<8xf16> into tensor<8xf16>
    %cat = tensor.insert_slice %t into %stage[0] [8] [1]
        : tensor<8xf16> into tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %consumer = linalg.map ins(%cat : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kPadded = R"mlir(
module {
  func.func @padded(%input: tensor<8xf16>) -> tensor<10xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %c0 = arith.constant 0.0 : f16
    %padded = tensor.pad %producer low[1] high[1] {
    ^bb0(%index: index):
      tensor.yield %c0 : f16
    } : tensor<8xf16> to tensor<10xf16>
    %out1 = tensor.empty() : tensor<10xf16>
    %consumer = linalg.map ins(%padded : tensor<10xf16>)
        outs(%out1 : tensor<10xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<10xf16>
  }
}
)mlir";

  static std::string rank17Module() {
    std::string dims = "1";
    for (unsigned index = 1; index < 17; ++index)
      dims += "x1";
    return std::string(R"mlir(
module {
  func.func @rank17(%input: tensor<)mlir") +
           dims + R"mlir(xf16>) -> tensor<)mlir" + dims + R"mlir(xf16> {
    %out0 = tensor.empty() : tensor<)mlir" +
           dims + R"mlir(xf16>
    %producer = linalg.map ins(%input : tensor<)mlir" +
           dims + R"mlir(xf16>)
        outs(%out0 : tensor<)mlir" +
           dims + R"mlir(xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %out1 = tensor.empty() : tensor<)mlir" +
           dims + R"mlir(xf16>
    %consumer = linalg.map ins(%producer : tensor<)mlir" +
           dims + R"mlir(xf16>)
        outs(%out1 : tensor<)mlir" +
           dims + R"mlir(xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<)mlir" +
           dims + R"mlir(xf16>
  }
}
)mlir";
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredDAGExactDemandQueryTest,
       DirectDataEdgeIsSatisfiedWithExactDemandAndRole) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_EQ(result.dependencyKind, analysis::DemandEdgeKind::DataInput);
  EXPECT_EQ(result.role, analysis::TileRole::UniquePartition);
  EXPECT_FALSE(result.uncoveredWitness);
  EXPECT_FALSE(result.mergeObligation);
  expectDemandPoints(result, {{0}, {3}, {4}, {7}}, {});
  ASSERT_EQ(result.ownershipIntersections.size(), 2u);
  const analysis::ExactOwnershipIntersection *tile0 =
      findIntersection(result, TileId(0));
  const analysis::ExactOwnershipIntersection *tile1 =
      findIntersection(result, TileId(1));
  ASSERT_TRUE(tile0 && tile1);
  llvm::SmallVector<int64_t, 4> inside0({0});
  llvm::SmallVector<int64_t, 4> inside1({6});
  llvm::SmallVector<int64_t, 4> cross0({6});
  llvm::SmallVector<int64_t, 4> cross1({1});
  EXPECT_TRUE(tile0->set->containsPoint(inside0));
  EXPECT_FALSE(tile0->set->containsPoint(cross0));
  EXPECT_TRUE(tile1->set->containsPoint(inside1));
  EXPECT_FALSE(tile1->set->containsPoint(cross1));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PermutedIndexingMapProducesExactProducerDemand) {
  auto module = parse(kTransposeRead);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  expectDemandPoints(result, {{0, 0}, {3, 7}, {0, 7}, {3, 0}}, {});
}

TEST_F(StructuredDAGExactDemandQueryTest,
       StridedSupportChainKeepsExactDemandWithoutBoundingBox) {
  auto module = parse(kStridedSupportChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  // The exact strided demand is {0,2,4,6}: a bounding box would wrongly
  // include 1,3,5,7.
  expectDemandPoints(result, {{0}, {2}, {4}, {6}}, {{1}, {3}, {5}, {7}});
}

TEST_F(StructuredDAGExactDemandQueryTest,
       OneToManyReductionDemandCoversEveryRequiredProducerElement) {
  auto module = parse(kReduction);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  // The reduction iterator projects many consumer iterations onto one
  // producer element; the exact image is the full producer domain.
  expectDemandPoints(result, {{0, 0}, {0, 7}, {3, 0}, {3, 7}}, {});
}

TEST_F(StructuredDAGExactDemandQueryTest, BroadcastRelationKeepsUniqueSourceSet) {
  auto module = parse(kBroadcast);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  expectDemandPoints(result, {{0}, {3}}, {{5}});
}

TEST_F(StructuredDAGExactDemandQueryTest, ExplicitInitProducerCarriesExactDemand) {
  auto module = parse(kInitRoot);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_EQ(result.dependencyKind, analysis::DemandEdgeKind::InitInput);
  expectDemandPoints(result, {{0, 0}, {3, 3}}, {});
}

TEST_F(StructuredDAGExactDemandQueryTest,
       MultiOperandSupportGraphKeepsPerPredecessorDependencies) {
  auto module = parse(kConcatChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edgeA = findEdge(dag, 0, 2);
  StructuredDAGEdgeID edgeB = findEdge(dag, 1, 2);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult resultA = query.query(edgeA, trial);
  analysis::ExactDemandResult resultB = query.query(edgeB, trial);

  EXPECT_EQ(resultA.status, analysis::ExactDemandStatus::Satisfied)
      << resultA.detail;
  EXPECT_EQ(resultB.status, analysis::ExactDemandStatus::Satisfied)
      << resultB.detail;
  // The insert_slice chain splits the consumer operand into per-producer
  // pieces expressed in each producer's own result space: producer A serves
  // the [0,2) piece of its domain, producer B the [1,3) piece of its domain.
  expectDemandPoints(resultA, {{0}, {1}}, {{2}, {3}});
  expectDemandPoints(resultB, {{1}, {2}}, {{0}, {3}});
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PadSupportChainDemandsTheInteriorProducerDomain) {
  auto module = parse(kPadded);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  // The pad boundary reads the constant fill, so the producer demand is the
  // interior [0,8), never the padded extents.
  expectDemandPoints(result, {{0}, {7}}, {{9}});
}

TEST_F(StructuredDAGExactDemandQueryTest,
       OwnershipHoleProducesProvenLogicalInfeasibleWithWitness) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  // Drop the second owner: the upper half of the demand is uncovered.
  analysis::LogicalNodeTrial &producerTrial = findNode(trial, 0);
  producerTrial.bindings.pop_back();

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::ProvenLogicalInfeasible);
  ASSERT_TRUE(result.uncoveredWitness);
  llvm::SmallVector<int64_t, 4> covered({0});
  llvm::SmallVector<int64_t, 4> uncovered({7});
  EXPECT_FALSE(result.uncoveredWitness->containsPoint(covered));
  EXPECT_TRUE(result.uncoveredWitness->containsPoint(uncovered));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       UniquePartitionOverlapIsProvenLogicalInfeasible) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  // Both owners claim the first half: a provable partition contradiction.
  analysis::LogicalNodeTrial &producerTrial = findNode(trial, 0);
  producerTrial.bindings[1].ownedDomain =
      producerTrial.bindings[0].ownedDomain;

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::ProvenLogicalInfeasible);
  ASSERT_TRUE(result.uncoveredWitness);
  llvm::SmallVector<int64_t, 4> overlapped({2});
  EXPECT_TRUE(result.uncoveredWitness->containsPoint(overlapped));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       AmbiguousSupportPathIsUnsupportedSemanticRelation) {
  auto module = parse(kAmbiguousSupport);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status,
            analysis::ExactDemandStatus::UnsupportedSemanticRelation)
      << result.detail;
  EXPECT_FALSE(result.producerDemand);
}

TEST_F(StructuredDAGExactDemandQueryTest, StaleEpochTrialIsIndeterminate) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  auto placements = fullPlacements(dag, 0, {0, 1});
  analysis::IREpoch firstEpoch = analysis::IREpoch::current();
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, firstEpoch);
  EXPECT_EQ(query.query(edge, trial).status,
            analysis::ExactDemandStatus::Satisfied);

  analysis::IREpoch::advance();
  analysis::LogicalShardTrial newerTrial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  EXPECT_EQ(query.query(edge, newerTrial).status,
            analysis::ExactDemandStatus::IndeterminateFailure);

  // A query built for the new epoch proves the same trial again.
  StructuredDAGExactDemandQuery newerQuery(dag, newerTrial.epoch);
  EXPECT_EQ(newerQuery.query(edge, newerTrial).status,
            analysis::ExactDemandStatus::Satisfied);

  // A trial without any epoch is malformed, never a placement failure.
  analysis::LogicalShardTrial epochless;
  epochless.nodes = trial.nodes;
  EXPECT_EQ(query.query(edge, epochless).status,
            analysis::ExactDemandStatus::IndeterminateFailure);
}

TEST_F(StructuredDAGExactDemandQueryTest, MalformedTrialIsIndeterminate) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);

  // Missing producer node entry.
  {
    analysis::LogicalShardTrial broken = trial;
    broken.nodes.erase(
        llvm::find_if(broken.nodes, [](const analysis::LogicalNodeTrial &entry) {
          return entry.node == 0;
        }));
    EXPECT_EQ(query.query(edge, broken).status,
              analysis::ExactDemandStatus::IndeterminateFailure);
  }
  // No ownership.
  {
    analysis::LogicalShardTrial broken = trial;
    findNode(broken, 0).bindings.clear();
    EXPECT_EQ(query.query(edge, broken).status,
              analysis::ExactDemandStatus::IndeterminateFailure);
  }
  // One Tile bound to several owner domains.
  {
    analysis::LogicalShardTrial broken = trial;
    findNode(broken, 0).bindings[1].tile = TileId(0);
    EXPECT_EQ(query.query(edge, broken).status,
              analysis::ExactDemandStatus::IndeterminateFailure);
  }
  // Missing ownership domain.
  {
    analysis::LogicalShardTrial broken = trial;
    findNode(broken, 0).bindings[0].ownedDomain.reset();
    EXPECT_EQ(query.query(edge, broken).status,
              analysis::ExactDemandStatus::IndeterminateFailure);
  }
  // Missing consumer iteration domain.
  {
    analysis::LogicalShardTrial broken = trial;
    findNode(broken, 1).completeIterationDomain.reset();
    EXPECT_EQ(query.query(edge, broken).status,
              analysis::ExactDemandStatus::IndeterminateFailure);
  }
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PresburgerBudgetExhaustionIsIndeterminate) {
  auto module = parse(rank17Module());
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  // The 34-variable relation space exceeds the IndexRelation budget; this is
  // a machinery failure, never a placement rejection.
  EXPECT_EQ(result.status, analysis::ExactDemandStatus::IndeterminateFailure);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       MetamorphicCarrierFailureDoesNotChangeTheOutcome) {
  auto module = parse(kStridedDirect);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  auto placements = fullPlacements(dag, 0, {0, 1});
  analysis::LogicalShardTrial trial = buildTrial(dag, placements);

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult before = query.query(edge, trial);
  EXPECT_EQ(before.status, analysis::ExactDemandStatus::Satisfied);
  expectDemandPoints(before, {{0}, {2}, {4}, {6}}, {{1}, {3}, {5}, {7}});

  // The canonical carrier cannot express the strided exact set; that failure
  // belongs to the physical assignment and must not touch the logical proof.
  StructuredDAGEdgeDemandPlanner demandPlanner(dag);
  std::string demandReason;
  auto demandPlan =
      demandPlanner.derive(edge, placements[0], placements[1], &demandReason);
  ASSERT_TRUE(mlir::succeeded(demandPlan)) << demandReason;
  std::string carrierReason;
  auto strategies = lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
      dag, *demandPlan, &carrierReason);
  EXPECT_TRUE(mlir::failed(strategies));

  analysis::ExactDemandResult after = query.query(edge, trial);
  EXPECT_EQ(after.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_TRUE(after.producerDemand->isEqual(*before.producerDemand));
  EXPECT_EQ(after.ownershipIntersections.size(),
            before.ownershipIntersections.size());
}

TEST_F(StructuredDAGExactDemandQueryTest,
       EquivalentPlacementsRecomputeToTheSameProof) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trialA =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  analysis::LogicalShardTrial trialB =
      buildTrial(dag, fullPlacements(dag, 0, {2, 3}));

  StructuredDAGExactDemandQuery query(dag, trialA.epoch);
  analysis::ExactDemandResult resultA = query.query(edge, trialA);
  analysis::ExactDemandResult resultB = query.query(edge, trialB);
  EXPECT_EQ(resultA.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_EQ(resultB.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_TRUE(resultA.producerDemand->isEqual(*resultB.producerDemand));
  // Distinct Tile identities keep distinct intersections: the cache must not
  // conflate extensionally equal placements with different bindings.
  ASSERT_TRUE(findIntersection(resultA, TileId(0)));
  ASSERT_TRUE(findIntersection(resultA, TileId(1)));
  ASSERT_TRUE(findIntersection(resultB, TileId(2)));
  ASSERT_TRUE(findIntersection(resultB, TileId(3)));
  EXPECT_FALSE(findIntersection(resultA, TileId(2)));

  // Reversing the binding order of the same trial content yields the same
  // per-Tile proof: the outcome never depends on caller enumeration order.
  analysis::LogicalShardTrial reversed = trialA;
  std::reverse(findNode(reversed, 0).bindings.begin(),
               findNode(reversed, 0).bindings.end());
  analysis::ExactDemandResult resultR = query.query(edge, reversed);
  EXPECT_EQ(resultR.status, analysis::ExactDemandStatus::Satisfied);
  const analysis::ExactOwnershipIntersection *tile0 =
      findIntersection(resultR, TileId(0));
  ASSERT_TRUE(tile0);
  EXPECT_TRUE(tile0->set->isEqual(
      *findIntersection(resultA, TileId(0))->set));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       TrialAdapterBuildsExplicitDomainsAndRejectsMalformedPlacements) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);

  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  ASSERT_EQ(trial.nodes.size(), 2u);
  const analysis::LogicalNodeTrial &producerTrial = findNode(trial, 0);
  ASSERT_TRUE(producerTrial.completeIterationDomain);
  llvm::SmallVector<int64_t, 4> iterPoint({7});
  EXPECT_TRUE(producerTrial.completeIterationDomain->containsPoint(iterPoint));
  ASSERT_EQ(producerTrial.bindings.size(), 2u);
  EXPECT_EQ(producerTrial.bindings[0].role, analysis::TileRole::UniquePartition);
  llvm::SmallVector<int64_t, 4> owned({0});
  llvm::SmallVector<int64_t, 4> notOwned({6});
  EXPECT_TRUE(producerTrial.bindings[0].ownedDomain->containsPoint(owned));
  EXPECT_FALSE(producerTrial.bindings[0].ownedDomain->containsPoint(notOwned));

  // Duplicate node placement.
  {
    auto placements = fullPlacements(dag, 0, {0, 1});
    placements[1].node = 0;
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::current(), &failureReason)));
  }
  // Missing node placement.
  {
    auto placements = fullPlacements(dag, 0, {0, 1});
    placements.pop_back();
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::current(), &failureReason)));
  }
  // Empty Tile group.
  {
    auto placements = fullPlacements(dag, 0, {0, 1});
    placements[0].tiles.clear();
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::current(), &failureReason)));
  }
  // Shard dimension outside the result rank.
  {
    auto placements = fullPlacements(dag, 5, {0, 1});
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::current(), &failureReason)));
  }
}

} // namespace
