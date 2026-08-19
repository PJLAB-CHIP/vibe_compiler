//===- StructuredDAGExactDemandQueryTest.cpp
//------------------------------------------===//

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGEdgeDemandPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Builders.h"
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
    result.spatialPartition =
        StructuredDAGSpatialPartition{dimension, dimension};
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

  static analysis::LogicalShardTrial
  buildTrial(const StructuredDAGAnalysis &dag,
             llvm::ArrayRef<StructuredDAGNodePlacement> placements,
             analysis::IREpoch epoch = analysis::IREpoch::mint()) {
    std::string failureReason;
    auto trial = buildLogicalShardTrial(dag, placements, epoch, &failureReason);
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

  static analysis::LogicalNodeTrial &
  findNode(analysis::LogicalShardTrial &trial, uint32_t node) {
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

  static const analysis::ExactOwnershipIntersection *
  findIntersection(const analysis::ExactDemandResult &result, TileId tile) {
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

  static constexpr llvm::StringLiteral kCompleteInsertAssembly = R"mlir(
module {
  func.func @complete_insert(%left: tensor<4xf16>, %right: tensor<4xf16>)
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
    %empty = tensor.empty() : tensor<8xf16>
    %lower = tensor.insert_slice %producer_a into %empty[0] [4] [1]
        : tensor<4xf16> into tensor<8xf16>
    %assembled = tensor.insert_slice %producer_b into %lower[4] [4] [1]
        : tensor<4xf16> into tensor<8xf16>
    %out2 = tensor.empty() : tensor<8xf16>
    %consumer = linalg.map ins(%assembled : tensor<8xf16>)
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
  EXPECT_EQ(result.producerResult, 0u);
  EXPECT_EQ(result.consumerOperand, 0u);
  EXPECT_EQ(result.role, analysis::TileRole::UniquePartition);
  EXPECT_FALSE(result.uncoveredWitness);
  EXPECT_FALSE(result.mergeObligation);
  // The successful outcome carries the complete consumer iteration domain
  // and per-owner intersections ordered by Tile id.
  ASSERT_TRUE(result.consumerIterationDomain);
  llvm::SmallVector<int64_t, 4> iterPoint({7});
  EXPECT_TRUE(result.consumerIterationDomain->containsPoint(iterPoint));
  expectDemandPoints(result, {{0}, {3}, {4}, {7}}, {});
  ASSERT_EQ(result.ownershipIntersections.size(), 2u);
  EXPECT_EQ(result.ownershipIntersections[0].tile, TileId(0));
  EXPECT_EQ(result.ownershipIntersections[1].tile, TileId(1));
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

TEST_F(StructuredDAGExactDemandQueryTest,
       BroadcastRelationKeepsUniqueSourceSet) {
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

TEST_F(StructuredDAGExactDemandQueryTest,
       ExplicitInitProducerCarriesExactDemand) {
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
       GroupedOperandRecipeRetainsExactEmptyInsertBranches) {
  auto module = parse(kCompleteInsertAssembly);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 3u);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ConsumerInputDemand demand =
      query.queryOperand(/*consumer=*/2, /*consumerOperand=*/0, trial);

  ASSERT_EQ(demand.status, analysis::ExactDemandStatus::Satisfied)
      << demand.detail;
  ASSERT_EQ(demand.perDestination.size(), 2u);
  for (auto [tileIndex, recipe] : llvm::enumerate(demand.perDestination)) {
    EXPECT_EQ(recipe.destinationTile,
              TileId(static_cast<int64_t>(tileIndex)));
    ASSERT_EQ(recipe.boundaries.size(), 2u);
    const analysis::ProducerValueRequirement &producerA = recipe.boundaries[0];
    const analysis::ProducerValueRequirement &producerB = recipe.boundaries[1];
    ASSERT_TRUE(producerA.requiredDomain);
    ASSERT_TRUE(producerB.requiredDomain);
    if (tileIndex == 0) {
      EXPECT_FALSE(producerA.requiredDomain->isIntegerEmpty());
      EXPECT_TRUE(producerB.requiredDomain->isIntegerEmpty());
      ASSERT_EQ(recipe.steps.size(), 2u);
    } else {
      EXPECT_TRUE(producerA.requiredDomain->isIntegerEmpty());
      EXPECT_FALSE(producerB.requiredDomain->isIntegerEmpty());
      ASSERT_EQ(recipe.steps.size(), 1u);
    }
    EXPECT_EQ(recipe.steps.back().kind,
              analysis::TensorTransformKind::InsertSlice);
  }
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

  EXPECT_EQ(result.status,
            analysis::ExactDemandStatus::ProvenLogicalInfeasible);
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
  producerTrial.bindings[1].ownedDomain = producerTrial.bindings[0].ownedDomain;

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status,
            analysis::ExactDemandStatus::ProvenLogicalInfeasible);
  ASSERT_TRUE(result.uncoveredWitness);
  llvm::SmallVector<int64_t, 4> overlapped({2});
  EXPECT_TRUE(result.uncoveredWitness->containsPoint(overlapped));

  // The witness is the union of pairwise overlaps and independent of the
  // caller's binding enumeration order.
  analysis::LogicalShardTrial reversedOverlap = trial;
  std::reverse(findNode(reversedOverlap, 0).bindings.begin(),
               findNode(reversedOverlap, 0).bindings.end());
  analysis::ExactDemandResult reversedResult =
      query.query(edge, reversedOverlap);
  EXPECT_EQ(reversedResult.status,
            analysis::ExactDemandStatus::ProvenLogicalInfeasible);
  ASSERT_TRUE(reversedResult.uncoveredWitness);
  EXPECT_TRUE(
      reversedResult.uncoveredWitness->isEqual(*result.uncoveredWitness));
  ASSERT_EQ(reversedResult.ownershipIntersections.size(), 2u);
  EXPECT_EQ(reversedResult.ownershipIntersections[0].tile, TileId(0));
  EXPECT_EQ(reversedResult.ownershipIntersections[1].tile, TileId(1));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       AmbiguousProducerToConsumerChainIsUnsupportedSemanticRelation) {
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

TEST_F(StructuredDAGExactDemandQueryTest,
       TrialFromAnotherBorrowIsIndeterminate) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  auto placements = fullPlacements(dag, 0, {0, 1});

  analysis::IREpoch epoch = analysis::IREpoch::mint();
  std::string failureReason;
  auto sameBorrowTrial =
      buildLogicalShardTrial(dag, placements, epoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(sameBorrowTrial)) << failureReason;
  StructuredDAGExactDemandQuery query(dag, epoch);
  EXPECT_EQ(query.query(edge, *sameBorrowTrial).status,
            analysis::ExactDemandStatus::Satisfied);

  // A trial minted for another borrow is rejected before any derivation.
  analysis::IREpoch otherEpoch = analysis::IREpoch::mint();
  auto otherTrial =
      buildLogicalShardTrial(dag, placements, otherEpoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(otherTrial)) << failureReason;
  EXPECT_EQ(query.query(edge, *otherTrial).status,
            analysis::ExactDemandStatus::IndeterminateFailure);

  // A query built for the other borrow proves the same trial again.
  StructuredDAGExactDemandQuery otherQuery(dag, otherEpoch);
  EXPECT_EQ(otherQuery.query(edge, *otherTrial).status,
            analysis::ExactDemandStatus::Satisfied);

  // A trial without any epoch is malformed, never a placement failure.
  analysis::LogicalShardTrial epochless;
  epochless.nodes = sameBorrowTrial->nodes;
  EXPECT_EQ(query.query(edge, epochless).status,
            analysis::ExactDemandStatus::IndeterminateFailure);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       InPlaceMutationInvalidatesTheQuerySnapshot) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  auto placements = fullPlacements(dag, 0, {0, 1});
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  std::string failureReason;
  auto trial = buildLogicalShardTrial(dag, placements, epoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(trial)) << failureReason;

  StructuredDAGExactDemandQuery query(dag, epoch);
  EXPECT_EQ(query.query(edge, *trial).status,
            analysis::ExactDemandStatus::Satisfied);

  // Mutate a nested operation attribute without changing any operation count.
  // The old query must fail closed on every derived fact, including its
  // relation cache; the epoch token itself is not an invalidation mechanism.
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  const size_t operationCount =
      function.getBody().front().getOperations().size();
  mlir::linalg::MapOp consumer =
      *std::next(function.getOps<mlir::linalg::MapOp>().begin());
  consumer->setAttr("test.semantic_revision",
                    mlir::UnitAttr::get(function.getContext()));
  EXPECT_EQ(operationCount, function.getBody().front().getOperations().size());
  EXPECT_EQ(query.query(edge, *trial).status,
            analysis::ExactDemandStatus::IndeterminateFailure);

  // A fresh query over the current structure works again: invalidation is
  // the structural snapshot captured at query construction.
  StructuredDAGExactDemandQuery freshQuery(dag, epoch);
  EXPECT_EQ(freshQuery.query(edge, *trial).status,
            analysis::ExactDemandStatus::Satisfied);
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
    broken.nodes.erase(llvm::find_if(
        broken.nodes, [](const analysis::LogicalNodeTrial &entry) {
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
  // Present destination shards must all-and-only cover the complete domain.
  {
    analysis::LogicalShardTrial broken = trial;
    findNode(broken, 1).executionShards.pop_back();
    analysis::ExactDemandResult result = query.query(edge, broken);
    EXPECT_EQ(result.status, analysis::ExactDemandStatus::IndeterminateFailure);
    EXPECT_NE(result.detail.find("do not cover"), std::string::npos);
  }
  // Overlapping execution domains are malformed even when their union covers.
  {
    analysis::LogicalShardTrial broken = trial;
    analysis::LogicalNodeTrial &consumer = findNode(broken, 1);
    consumer.executionShards[1].executionDomain =
        consumer.executionShards[0].executionDomain;
    analysis::ExactDemandResult result = query.query(edge, broken);
    EXPECT_EQ(result.status, analysis::ExactDemandStatus::IndeterminateFailure);
    EXPECT_NE(result.detail.find("overlap"), std::string::npos);
  }
  // A Tile has exactly one execution shard.
  {
    analysis::LogicalShardTrial broken = trial;
    analysis::LogicalNodeTrial &consumer = findNode(broken, 1);
    consumer.executionShards[1].tile = consumer.executionShards[0].tile;
    analysis::ExactDemandResult result = query.query(edge, broken);
    EXPECT_EQ(result.status, analysis::ExactDemandStatus::IndeterminateFailure);
    EXPECT_NE(result.detail.find("repeat"), std::string::npos);
  }
  // Every execution shard uses the complete iteration space.
  {
    analysis::LogicalShardTrial broken = trial;
    analysis::IndexSetResult wrongRank =
        analysis::IndexRelation::staticDomain({8, 8});
    ASSERT_TRUE(wrongRank.isExact());
    findNode(broken, 1).executionShards[0].executionDomain = *wrongRank.set;
    analysis::ExactDemandResult result = query.query(edge, broken);
    EXPECT_EQ(result.status, analysis::ExactDemandStatus::IndeterminateFailure);
    EXPECT_NE(result.detail.find("incompatible iteration space"),
              std::string::npos);
  }
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PresburgerBudgetExhaustionIsIndeterminate) {
  auto module = parse(rank17Module());
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  llvm::SmallVector<int64_t, 17> shape(17, 1);
  analysis::IndexSetResult full = analysis::IndexRelation::staticDomain(shape);
  ASSERT_TRUE(full.isExact());
  analysis::LogicalShardTrial trial;
  trial.epoch = analysis::IREpoch::mint();
  for (const StructuredDAGNode &node : dag.getNodes()) {
    analysis::LogicalNodeTrial nodeTrial;
    nodeTrial.node = node.id;
    nodeTrial.completeIterationDomain = *full.set;
    nodeTrial.executionShards.push_back(
        analysis::LogicalExecutionShard{TileId(0), *full.set});
    nodeTrial.bindings.push_back(
        analysis::LogicalTileBinding{TileId(0), /*resultIndex=*/0, *full.set,
                                     analysis::TileRole::UniquePartition});
    trial.nodes.push_back(std::move(nodeTrial));
  }

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
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  analysis::LogicalShardTrial trialA =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}), epoch);
  analysis::LogicalShardTrial trialB =
      buildTrial(dag, fullPlacements(dag, 0, {2, 3}), epoch);

  StructuredDAGExactDemandQuery query(dag, epoch);
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
  EXPECT_TRUE(tile0->set->isEqual(*findIntersection(resultA, TileId(0))->set));
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
  EXPECT_EQ(producerTrial.bindings[0].role,
            analysis::TileRole::UniquePartition);
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
        dag, placements, analysis::IREpoch::mint(), &failureReason)));
  }
  // Missing node placement.
  {
    auto placements = fullPlacements(dag, 0, {0, 1});
    placements.pop_back();
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::mint(), &failureReason)));
  }
  // Empty Tile group.
  {
    auto placements = fullPlacements(dag, 0, {0, 1});
    placements[0].tiles.clear();
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::mint(), &failureReason)));
  }
  // A spatial iterator outside the loop rank is a malformed placement.
  {
    auto placements = fullPlacements(dag, 5, {0, 1});
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(buildLogicalShardTrial(
        dag, placements, analysis::IREpoch::mint(), &failureReason)));
  }
}

TEST_F(StructuredDAGExactDemandQueryTest,
       MultiResultProducerCarriesPerResultOwnership) {
  auto module = parse(R"mlir(
module {
  func.func @multi(%input: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>)
        outs(%out0, %out1 : tensor<8xf16>, tensor<8xf16>) {
      ^bb0(%in: f16, %o0: f16, %o1: f16):
        %sum = arith.addf %in, %o0 : f16
        linalg.yield %sum, %sum : f16, f16
    } -> (tensor<8xf16>, tensor<8xf16>)
    %out2 = tensor.empty() : tensor<8xf16>
    %consumer0 = linalg.map ins(%producer#0 : tensor<8xf16>)
        outs(%out2 : tensor<8xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    %out3 = tensor.empty() : tensor<8xf16>
    %consumer1 = linalg.map ins(%producer#1 : tensor<8xf16>)
        outs(%out3 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer0, %consumer1 : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 3u);
  StructuredDAGEdgeID edge0 = findEdge(dag, 0, 1);
  StructuredDAGEdgeID edge1 = findEdge(dag, 0, 2);
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  auto placements = fullPlacements(dag, 0, {0, 1});
  std::string failureReason;
  auto trial = buildLogicalShardTrial(dag, placements, epoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(trial)) << failureReason;

  StructuredDAGExactDemandQuery query(dag, epoch);
  analysis::ExactDemandResult result0 = query.query(edge0, *trial);
  analysis::ExactDemandResult result1 = query.query(edge1, *trial);
  EXPECT_EQ(result0.status, analysis::ExactDemandStatus::Satisfied)
      << result0.detail;
  EXPECT_EQ(result1.status, analysis::ExactDemandStatus::Satisfied)
      << result1.detail;
  // Each edge names its concrete producer result and proves it with the
  // ownership of that result only.
  EXPECT_EQ(result0.producerResult, 0u);
  EXPECT_EQ(result1.producerResult, 1u);
  expectDemandPoints(result0, {{0}, {7}}, {});
  expectDemandPoints(result1, {{0}, {7}}, {});
  ASSERT_EQ(result0.ownershipIntersections.size(), 2u);
  ASSERT_EQ(result1.ownershipIntersections.size(), 2u);
  EXPECT_EQ(result0.ownershipIntersections[0].tile, TileId(0));
  EXPECT_EQ(result1.ownershipIntersections[1].tile, TileId(1));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       UnshardableResultFallsBackToExplicitReplication) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8}, mlir::FloatType::getF16(&context));
  // The rank-1 result cannot express shard dimension 2: every Tile of the
  // group owns the complete result as an explicit replica.
  std::string failureReason;
  auto ownership =
      buildBalancedOwnership(type, /*shardDimension=*/2, {TileId(0), TileId(1)},
                             /*resultIndex=*/0, &failureReason);
  ASSERT_TRUE(mlir::succeeded(ownership)) << failureReason;
  ASSERT_EQ(ownership->size(), 2u);
  for (const analysis::LogicalTileBinding &binding : *ownership) {
    EXPECT_EQ(binding.resultIndex, 0u);
    EXPECT_EQ(binding.role, analysis::TileRole::ExplicitReplication);
    ASSERT_TRUE(binding.ownedDomain);
    llvm::SmallVector<int64_t, 4> point({0});
    EXPECT_TRUE(binding.ownedDomain->containsPoint(point));
  }

  // A manual replication trial proves the query keeps all eligible owners
  // and never picks a source.
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}), epoch);
  analysis::LogicalNodeTrial &producerTrial = findNode(trial, 0);
  for (analysis::LogicalTileBinding &binding : producerTrial.bindings) {
    auto full = analysis::IndexRelation::staticDomain({8});
    ASSERT_TRUE(full.isExact());
    binding.ownedDomain = *full.set;
    binding.role = analysis::TileRole::ExplicitReplication;
  }

  StructuredDAGExactDemandQuery query(dag, epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);
  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied)
      << result.detail;
  EXPECT_EQ(result.role, analysis::TileRole::ExplicitReplication);
  ASSERT_EQ(result.ownershipIntersections.size(), 2u);
  llvm::SmallVector<int64_t, 4> point({0});
  for (const analysis::ExactOwnershipIntersection &intersection :
       result.ownershipIntersections)
    EXPECT_TRUE(intersection.set->containsPoint(point));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PartialReductionContributionsPreserveEveryRequiredOwner) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  analysis::IndexSetResult full = analysis::IndexRelation::staticDomain({8});
  ASSERT_TRUE(full.isExact());
  analysis::LogicalNodeTrial &producer = findNode(trial, 0);
  for (analysis::LogicalTileBinding &binding : producer.bindings) {
    binding.ownedDomain = *full.set;
    binding.role = analysis::TileRole::PartialReductionContribution;
  }
  std::reverse(producer.bindings.begin(), producer.bindings.end());

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied)
      << result.detail;
  EXPECT_TRUE(result.mergeObligation);
  EXPECT_EQ(result.role, analysis::TileRole::PartialReductionContribution);
  ASSERT_EQ(result.ownershipIntersections.size(), 2u);
  EXPECT_EQ(result.ownershipIntersections[0].tile, TileId(0));
  EXPECT_EQ(result.ownershipIntersections[1].tile, TileId(1));
  ASSERT_EQ(result.perDestination.size(), 2u);
  for (const analysis::ExactDestinationDemand &destination :
       result.perDestination)
    EXPECT_EQ(destination.ownershipIntersections.size(), 2u);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PerDestinationFactsCoverEveryConsumerExecutionShard) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  // One exact demand per consumer execution shard, ordered by Tile id.
  ASSERT_EQ(result.perDestination.size(), 2u);
  EXPECT_EQ(result.perDestination[0].destinationTile, TileId(0));
  EXPECT_EQ(result.perDestination[1].destinationTile, TileId(1));

  // Tile 0 executes iterations [0,4) and demands producer elements [0,4);
  // its intersection is owned by Tile 0 alone.
  const analysis::ExactDestinationDemand &first = result.perDestination[0];
  ASSERT_TRUE(first.consumerExecutionDomain);
  ASSERT_TRUE(first.producerDemand);
  EXPECT_TRUE(first.consumerExecutionDomain->containsPoint({0}));
  EXPECT_TRUE(first.consumerExecutionDomain->containsPoint({3}));
  EXPECT_FALSE(first.consumerExecutionDomain->containsPoint({4}));
  EXPECT_TRUE(first.producerDemand->containsPoint({0}));
  EXPECT_TRUE(first.producerDemand->containsPoint({3}));
  EXPECT_FALSE(first.producerDemand->containsPoint({4}));
  ASSERT_EQ(first.ownershipIntersections.size(), 1u);
  EXPECT_EQ(first.ownershipIntersections[0].tile, TileId(0));
  EXPECT_TRUE(first.ownershipIntersections[0].set->containsPoint({3}));
  EXPECT_FALSE(first.ownershipIntersections[0].set->containsPoint({4}));
  EXPECT_FALSE(first.uncoveredWitness);

  // Tile 1 executes iterations [4,8) and demands producer elements [4,8).
  const analysis::ExactDestinationDemand &second = result.perDestination[1];
  ASSERT_TRUE(second.consumerExecutionDomain);
  ASSERT_TRUE(second.producerDemand);
  EXPECT_FALSE(second.consumerExecutionDomain->containsPoint({3}));
  EXPECT_TRUE(second.consumerExecutionDomain->containsPoint({4}));
  EXPECT_FALSE(second.producerDemand->containsPoint({3}));
  EXPECT_TRUE(second.producerDemand->containsPoint({4}));
  EXPECT_TRUE(second.producerDemand->containsPoint({7}));
  ASSERT_EQ(second.ownershipIntersections.size(), 1u);
  EXPECT_EQ(second.ownershipIntersections[0].tile, TileId(1));
  EXPECT_FALSE(second.uncoveredWitness);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       SixteenLargeBalancedRectanglesUseExactIntervalCoverage) {
  auto module = parse(R"mlir(
module {
  func.func @large_chain(%input: tensor<11008xf16>) -> tensor<11008xf16> {
    %out0 = tensor.empty() : tensor<11008xf16>
    %out1 = tensor.empty() : tensor<11008xf16>
    %producer = linalg.map ins(%input : tensor<11008xf16>)
        outs(%out0 : tensor<11008xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %consumer = linalg.map ins(%producer : tensor<11008xf16>)
        outs(%out1 : tensor<11008xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<11008xf16>
  }
}
)mlir");
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial = buildTrial(
      dag, fullPlacements(dag, 0, {0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied)
      << result.detail;
  EXPECT_EQ(result.ownershipIntersections.size(), 16u);
  EXPECT_EQ(result.perDestination.size(), 16u);
  for (const analysis::ExactDestinationDemand &destination :
       result.perDestination)
    EXPECT_FALSE(destination.uncoveredWitness);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       PerDestinationWitnessIdentifiesTheUncoveredShard) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));
  // Drop the second owner: the upper half of the demand is uncovered, and
  // only the Tile-1 destination shard carries the direct witness.
  analysis::LogicalNodeTrial &producerTrial = findNode(trial, 0);
  producerTrial.bindings.pop_back();

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status,
            analysis::ExactDemandStatus::ProvenLogicalInfeasible);
  ASSERT_TRUE(result.uncoveredWitness);
  EXPECT_TRUE(result.uncoveredWitness->containsPoint({7}));
  ASSERT_EQ(result.perDestination.size(), 2u);
  const analysis::ExactDestinationDemand &first = result.perDestination[0];
  EXPECT_FALSE(first.uncoveredWitness);
  const analysis::ExactDestinationDemand &second = result.perDestination[1];
  ASSERT_TRUE(second.uncoveredWitness);
  EXPECT_FALSE(second.uncoveredWitness->containsPoint({0}));
  EXPECT_TRUE(second.uncoveredWitness->containsPoint({7}));
}

TEST_F(StructuredDAGExactDemandQueryTest,
       BalancedRemainderProducesExactPerDestinationFacts) {
  auto module = parse(R"mlir(
module {
  func.func @chain(%input: tensor<10xf16>) -> tensor<10xf16> {
    %out0 = tensor.empty() : tensor<10xf16>
    %out1 = tensor.empty() : tensor<10xf16>
    %producer = linalg.map ins(%input : tensor<10xf16>)
        outs(%out0 : tensor<10xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %consumer = linalg.map ins(%producer : tensor<10xf16>)
        outs(%out1 : tensor<10xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<10xf16>
  }
}
)mlir");
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1, 2}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult result = query.query(edge, trial);

  EXPECT_EQ(result.status, analysis::ExactDemandStatus::Satisfied);
  // 10 elements over 3 Tiles: balanced shards of sizes 4, 3, 3.
  ASSERT_EQ(result.perDestination.size(), 3u);
  const auto bounds = [](const analysis::ExactDestinationDemand &destination,
                         int64_t first, int64_t last) {
    ASSERT_TRUE(destination.producerDemand);
    EXPECT_TRUE(destination.producerDemand->containsPoint({first}));
    EXPECT_TRUE(destination.producerDemand->containsPoint({last}));
    EXPECT_FALSE(destination.producerDemand->containsPoint({last + 1}));
  };
  bounds(result.perDestination[0], 0, 3);
  bounds(result.perDestination[1], 4, 6);
  bounds(result.perDestination[2], 7, 9);
  for (const analysis::ExactDestinationDemand &destination :
       result.perDestination)
    EXPECT_FALSE(destination.uncoveredWitness);
}

TEST_F(StructuredDAGExactDemandQueryTest,
       CarrierAssemblyConsumesTheQueryPerDestinationFacts) {
  auto module = parse(kChain);
  StructuredDAGAnalysis dag = buildDAG(*module);
  StructuredDAGEdgeID edge = findEdge(dag, 0, 1);
  analysis::LogicalShardTrial trial =
      buildTrial(dag, fullPlacements(dag, 0, {0, 1}));

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  analysis::ExactDemandResult demand = query.query(edge, trial);
  ASSERT_EQ(demand.status, analysis::ExactDemandStatus::Satisfied);

  StructuredDAGEdgeDemandPlan plan;
  std::string failureReason;
  auto placements = fullPlacements(dag, 0, {0, 1});
  ASSERT_TRUE(mlir::succeeded(assembleStructuredDAGEdgeDemandPlan(
      dag, *dag.getEdge(edge), placements[0], placements[1], trial, demand,
      &plan, &failureReason)))
      << failureReason;

  // The carrier plan carries exactly the query's per-destination facts: one
  // demand per consumer shard, producerDemand equal to the query fact, and
  // the consumer access window equal to the consumer's own result shard.
  ASSERT_EQ(plan.demands.size(), 2u);
  const analysis::LogicalNodeTrial &consumerTrial = findNode(trial, 1);
  for (const StructuredDAGEdgeDemand &entry : plan.demands) {
    const analysis::ExactDestinationDemand *fact = nullptr;
    for (const analysis::ExactDestinationDemand &destination :
         demand.perDestination)
      if (destination.destinationTile == entry.destinationTile) {
        fact = &destination;
        break;
      }
    ASSERT_TRUE(fact);
    ASSERT_TRUE(fact->producerDemand);
    EXPECT_TRUE(entry.producerDemand.isEqual(*fact->producerDemand));
    const analysis::LogicalTileBinding *binding = nullptr;
    for (const analysis::LogicalTileBinding &candidate : consumerTrial.bindings)
      if (candidate.tile == entry.destinationTile &&
          candidate.resultIndex == 0) {
        binding = &candidate;
        break;
      }
    ASSERT_TRUE(binding);
    ASSERT_TRUE(binding->ownedDomain);
    EXPECT_TRUE(entry.consumerDomain.isEqual(*binding->ownedDomain));
    EXPECT_EQ(entry.producerShardOwnership.size(), 2u);
  }

  // A fabricated satisfied payload that omits one placement destination is
  // rejected by the carrier instead of silently dropping its movement.
  demand.perDestination.pop_back();
  StructuredDAGEdgeDemandPlan incompletePlan;
  failureReason.clear();
  EXPECT_TRUE(mlir::failed(assembleStructuredDAGEdgeDemandPlan(
      dag, *dag.getEdge(edge), placements[0], placements[1], trial, demand,
      &incompletePlan, &failureReason)));
  EXPECT_NE(failureReason.find("every destination Tile"), std::string::npos);
}

} // namespace
