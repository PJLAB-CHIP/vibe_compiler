//===- SingleRootTileRegionTest.cpp -----------------------------------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"
#include "Wafer/Planning/Search/SingleRootRegion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/Twine.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>

namespace {

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

template <typename OpT> unsigned countOps(mlir::Operation *operation) {
  unsigned count = 0;
  operation->walk([&](OpT) { ++count; });
  return count;
}

constexpr llvm::StringLiteral topology = R"mlir(
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
)mlir";

TEST(SingleRootTileRegionTest, MaterializesExactNodeShardsAndEmptyTiles) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @map(%input: tensor<4x6xf16>) -> tensor<4x6xf16> {
    %empty = tensor.empty() : tensor<4x6xf16>
    %result = linalg.map ins(%input : tensor<4x6xf16>)
        outs(%empty : tensor<4x6xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<4x6xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::MapOp root;
  source->walk([&](mlir::linalg::MapOp operation) { root = operation; });
  ASSERT_TRUE(root);
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 7}};
  std::array<wafer::StructuredNodeIterationShard, 2> shards = {
      wafer::StructuredNodeIterationShard{7, wafer::TileId(0), {0, 0}, {2, 6}},
      wafer::StructuredNodeIterationShard{7, wafer::TileId(3), {2, 0}, {2, 6}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  wafer::StructuredMaterializationRelations relations;
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      &relations, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
  EXPECT_EQ(countOps<wafer::TileModuleOp>(materialized->getOperation()), 4u);
  EXPECT_EQ(countOps<mlir::func::FuncOp>(materialized->getOperation()), 2u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->getOperation()), 2u);
  materialized->walk([&](mlir::func::FuncOp function) {
    EXPECT_TRUE(function->getParentOfType<wafer::TileModuleOp>());
  });
  ASSERT_FALSE(relations.operationEmissions.empty());
  for (const wafer::StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    EXPECT_EQ(relation.structuredNodeId, 7u);
}

TEST(SingleRootTileRegionTest, AppliesClosedMultiAxisPlacement) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @map(%input: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %empty = tensor.empty() : tensor<5x7xf16>
    %result = linalg.map ins(%input : tensor<5x7xf16>)
        outs(%empty : tensor<5x7xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<5x7xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  wafer::analysis::IREpoch epoch = wafer::analysis::IREpoch::mint();
  wafer::compiler::detail::StructuredDAGNodePlacement placement{
      dag->getNodes().front().id,
      /*iteratorPartitionFactors=*/{2, 2},
      /*tiles=*/
      {wafer::TileId(3), wafer::TileId(0), wafer::TileId(2), wafer::TileId(1)}};
  auto trial = wafer::compiler::detail::buildLogicalShardTrial(
      *dag, llvm::ArrayRef(&placement, 1), epoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(trial)) << failureReason;
  auto target = wafer::TargetTopology::create(*source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(target)) << failureReason;
  llvm::SmallVector<wafer::TileId, 16> available{
      wafer::TileId(0), wafer::TileId(1), wafer::TileId(2), wafer::TileId(3)};
  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 16> operationNodes{
      {dag->getNodes().front().operation, dag->getNodes().front().id}};
  wafer::compiler::detail::CardProgramAnalysis program(
      std::move(*target), available, std::move(*dag),
      wafer::compiler::detail::StaticOutputDomains{{5, 7}}, operationNodes,
      epoch);
  auto materialized = wafer::compiler::detail::materializeCardSingleRootRegions(
      *source, program, wafer::CardId(0), *trial, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            4u);
}

TEST(SingleRootTileRegionTest,
     KeepsStructuredInitProducerOutsideConsumerRegion) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @init_boundary(%input: tensor<4xf16>) -> tensor<4xf16> {
    %producer_empty = tensor.empty() : tensor<4xf16>
    %producer = linalg.map ins(%input : tensor<4xf16>)
        outs(%producer_empty : tensor<4xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %view = tensor.extract_slice %producer[0] [4] [1]
        : tensor<4xf16> to tensor<4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%view : tensor<4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %old : f16
        linalg.yield %sum : f16
    } -> tensor<4xf16>
    return %consumer : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::Operation *, 2> roots;
  source->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
        mlir::isa<mlir::TilingInterface>(operation))
      roots.push_back(operation);
  });
  ASSERT_EQ(roots.size(), 2u);
  std::array<wafer::StructuredOperationNodeMapping, 2> operationNodes = {
      wafer::StructuredOperationNodeMapping{roots[0], 0},
      wafer::StructuredOperationNodeMapping{roots[1], 1}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{1, wafer::TileId(0), {0}, {2}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  wafer::StructuredMaterializationRelations relations;
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      &relations, &failureReason)))
      << failureReason;
  ASSERT_FALSE(relations.operationEmissions.empty());
  for (const wafer::StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    EXPECT_EQ(relation.structuredNodeId, 1u);
}

TEST(SingleRootTileRegionTest,
     KeepsEveryMultiProducerSupportBoundaryOutsideConsumerRegion) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @fanin(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> tensor<4xf16> {
    %left_empty = tensor.empty() : tensor<2xf16>
    %left = linalg.map ins(%lhs : tensor<2xf16>)
        outs(%left_empty : tensor<2xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %right_empty = tensor.empty() : tensor<2xf16>
    %right = linalg.map ins(%rhs : tensor<2xf16>)
        outs(%right_empty : tensor<2xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %assembled_empty = tensor.empty() : tensor<4xf16>
    %with_left = tensor.insert_slice %left into %assembled_empty[0] [2] [1]
        : tensor<2xf16> into tensor<4xf16>
    %assembled = tensor.insert_slice %right into %with_left[2] [2] [1]
        : tensor<2xf16> into tensor<4xf16>
    %consumer_empty = tensor.empty() : tensor<4xf16>
    %consumer = linalg.map ins(%assembled : tensor<4xf16>)
        outs(%consumer_empty : tensor<4xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %consumer : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 3> roots;
  source->walk(
      [&](mlir::linalg::MapOp operation) { roots.push_back(operation); });
  ASSERT_EQ(roots.size(), 3u);
  std::array<wafer::StructuredOperationNodeMapping, 3> operationNodes = {
      wafer::StructuredOperationNodeMapping{roots[0].getOperation(), 0},
      wafer::StructuredOperationNodeMapping{roots[1].getOperation(), 1},
      wafer::StructuredOperationNodeMapping{roots[2].getOperation(), 2}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{2, wafer::TileId(0), {0}, {4}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  wafer::StructuredMaterializationRelations relations;
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      &relations, &failureReason)))
      << failureReason;
  ASSERT_FALSE(relations.operationEmissions.empty());
  for (const wafer::StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    EXPECT_EQ(relation.structuredNodeId, 2u);
}

TEST(SingleRootTileRegionTest, MaterializesMultipleStructuredResults) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @pair(%input: tensor<4xf16>, %lhs_init: tensor<4xf16>,
                  %rhs_init: tensor<4xf16>)
      -> (tensor<4xf16>, tensor<4xf16>) {
    %lhs, %rhs = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>)
        outs(%lhs_init, %rhs_init : tensor<4xf16>, tensor<4xf16>) {
      ^bb0(%value: f16, %old_lhs: f16, %old_rhs: f16):
        %next_lhs = arith.addf %value, %old_lhs : f16
        %next_rhs = arith.mulf %value, %old_rhs : f16
        linalg.yield %next_lhs, %next_rhs : f16, f16
    } -> (tensor<4xf16>, tensor<4xf16>)
    return %lhs, %rhs : tensor<4xf16>, tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp root;
  source->walk([&](mlir::linalg::GenericOp operation) { root = operation; });
  ASSERT_TRUE(root);
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{0, wafer::TileId(0), {1}, {2}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      nullptr, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->getOperation()), 1u);
  mlir::func::FuncOp function;
  materialized->walk([&](mlir::func::FuncOp candidate) {
    if (!function)
      function = candidate;
  });
  ASSERT_TRUE(function);
  EXPECT_EQ(function.getNumResults(), 2u);
}

TEST(SingleRootTileRegionTest, MaterializesUnpartitionedReductionWork) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @reduce(%input: tensor<4x8xf16>, %init: tensor<4xf16>)
      -> tensor<4xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<4xf16>
    return %sum : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp root;
  source->walk([&](mlir::linalg::GenericOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{0, wafer::TileId(0), {0, 0}, {2, 8}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      nullptr, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->getOperation()), 1u);
}

TEST(SingleRootTileRegionTest, MaterializesZeroDimensionalIteratorShard) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @scalar(%value: f16) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.fill ins(%value : f16) outs(%empty : tensor<f16>)
        -> tensor<f16>
    return %result : tensor<f16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::FillOp root;
  source->walk([&](mlir::linalg::FillOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{0, wafer::TileId(1), {}, {}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, materialized,
      nullptr, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->getOperation()), 1u);
}

TEST(SingleRootTileRegionTest,
     MaterializesSelectedTemporalLoopOrderAndFiniteTails) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @map(%input: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %empty = tensor.empty() : tensor<5x7xf16>
    %result = linalg.map ins(%input : tensor<5x7xf16>)
        outs(%empty : tensor<5x7xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<5x7xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::MapOp root;
  source->walk([&](mlir::linalg::MapOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};

  auto materialize = [&](llvm::SmallVector<uint32_t, 2> order) {
    wafer::StructuredNodeShardGroup group;
    group.shards.push_back(wafer::StructuredNodeIterationShard{
        0, wafer::TileId(0), {0, 0}, {5, 7}});
    group.temporalTiles.push_back(
        wafer::StructuredNodeTemporalTile{0, {2, 3}, std::move(order)});
    mlir::OwningOpRef<mlir::ModuleOp> result;
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeGroupsToCardModule(
        *source, wafer::CardId(0), tiles, operationNodes,
        llvm::ArrayRef(&group, 1), result, nullptr, &failureReason)))
        << failureReason;
    return result;
  };

  mlir::OwningOpRef<mlir::ModuleOp> rowMajor = materialize({0, 1});
  mlir::OwningOpRef<mlir::ModuleOp> columnMajor = materialize({1, 0});
  ASSERT_TRUE(rowMajor && columnMajor);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(rowMajor->getOperation()), 4u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(columnMajor->getOperation()), 4u);

  auto getNestingStep = [](mlir::ModuleOp module) -> std::optional<int64_t> {
    std::optional<int64_t> result;
    module.walk([&](mlir::scf::ForOp loop) {
      bool hasNestedLoop = false;
      loop.getRegion().walk(
          [&](mlir::scf::ForOp nested) { hasNestedLoop |= nested != loop; });
      if (hasNestedLoop && !result)
        result = mlir::getConstantIntValue(loop.getStep());
    });
    return result;
  };
  EXPECT_EQ(getNestingStep(*rowMajor), 2);
  EXPECT_EQ(getNestingStep(*columnMajor), 3);
}

TEST(SingleRootTileRegionTest,
     CarriesReductionAccumulatorAcrossTemporalPrologueSteadyAndTail) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @reduce(%input: tensor<2x10xf16>, %init: tensor<2xf16>)
      -> tensor<2xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<2x10xf16>) outs(%init : tensor<2xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2xf16>
    return %sum : tensor<2xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp root;
  source->walk([&](mlir::linalg::GenericOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  wafer::StructuredNodeShardGroup group;
  group.shards.push_back(wafer::StructuredNodeIterationShard{
      0, wafer::TileId(0), {0, 0}, {2, 10}});
  group.temporalTiles.push_back(
      wafer::StructuredNodeTemporalTile{0, {2, 4}, {1}});
  wafer::StructuredMaterializationRelations relations;
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeGroupsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes,
      llvm::ArrayRef(&group, 1), materialized, &relations, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<mlir::scf::ForOp>(materialized->getOperation()), 1u);
  EXPECT_GE(relations.operationEmissions.size(), 3u);
  EXPECT_TRUE(llvm::all_of(
      relations.operationEmissions,
      [](const wafer::StructuredOperationEmissionRelation &relation) {
        return relation.structuredNodeId == 0 && relation.operation;
      }));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
}

TEST(SingleRootTileRegionTest,
     MaterializesDistinctMultiReductionWaveLoopOrders) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @reduce2d(%input: tensor<4x6xf16>, %init: tensor<f16>)
      -> tensor<f16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> ()>],
        iterator_types = ["reduction", "reduction"]
      } ins(%input : tensor<4x6xf16>) outs(%init : tensor<f16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<f16>
    return %sum : tensor<f16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp root;
  source->walk([&](mlir::linalg::GenericOp operation) { root = operation; });
  ASSERT_TRUE(root);
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  auto materialize = [&](llvm::SmallVector<uint32_t, 2> order) {
    wafer::StructuredNodeShardGroup group;
    group.shards.push_back(wafer::StructuredNodeIterationShard{
        0, wafer::TileId(0), {0, 0}, {4, 6}});
    group.temporalTiles.push_back(
        wafer::StructuredNodeTemporalTile{0, {2, 3}, std::move(order)});
    mlir::OwningOpRef<mlir::ModuleOp> result;
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeGroupsToCardModule(
        *source, wafer::CardId(0), tiles, operationNodes,
        llvm::ArrayRef(&group, 1), result, nullptr, &failureReason)))
        << failureReason;
    return result;
  };
  mlir::OwningOpRef<mlir::ModuleOp> firstMajor = materialize({0, 1});
  mlir::OwningOpRef<mlir::ModuleOp> secondMajor = materialize({1, 0});
  ASSERT_TRUE(firstMajor && secondMajor);
  auto getNestingStep = [](mlir::ModuleOp module) -> std::optional<int64_t> {
    std::optional<int64_t> result;
    module.walk([&](mlir::scf::ForOp loop) {
      bool hasNestedLoop = false;
      loop.getRegion().walk([&](mlir::scf::ForOp) { hasNestedLoop = true; });
      if (hasNestedLoop && !result)
        result = mlir::getConstantIntValue(loop.getStep());
    });
    return result;
  };
  EXPECT_EQ(getNestingStep(*firstMajor), 2);
  EXPECT_EQ(getNestingStep(*secondMajor), 3);
}

TEST(SingleRootTileRegionTest,
     MaterializesPartialContributionsAndSelectedMergeRegion) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @reduce(%input: tensor<4x8xf16>, %init: tensor<4xf16>)
      -> tensor<4xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<4xf16>
    return %sum : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  wafer::analysis::IREpoch epoch = wafer::analysis::IREpoch::mint();
  wafer::compiler::detail::StructuredDAGNodePlacement placement{
      dag->getNodes().front().id,
      /*iteratorPartitionFactors=*/{2, 2},
      /*tiles=*/
      {wafer::TileId(0), wafer::TileId(1), wafer::TileId(2), wafer::TileId(3)},
      /*reductionMergeTile=*/wafer::TileId(2)};
  auto trial = wafer::compiler::detail::buildLogicalShardTrial(
      *dag, llvm::ArrayRef(&placement, 1), epoch, &failureReason);
  ASSERT_TRUE(mlir::succeeded(trial)) << failureReason;
  auto target = wafer::TargetTopology::create(*source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(target)) << failureReason;
  llvm::SmallVector<wafer::TileId, 16> available{
      wafer::TileId(0), wafer::TileId(1), wafer::TileId(2), wafer::TileId(3)};
  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 16> operationNodes{
      {dag->getNodes().front().operation, dag->getNodes().front().id}};
  wafer::compiler::detail::CardProgramAnalysis program(
      std::move(*target), available, std::move(*dag),
      wafer::compiler::detail::StaticOutputDomains{{4}}, operationNodes, epoch);
  auto materialized = wafer::compiler::detail::materializeCardSingleRootRegions(
      *source, program, wafer::CardId(0), *trial, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            5u);
  EXPECT_EQ(countOps<mlir::func::FuncOp>(materialized->module->getOperation()),
            5u);
  llvm::SmallVector<unsigned, 4> regionsByTile(4, 0);
  materialized->module->walk([&](wafer::TileModuleOp tile) {
    tile.walk([&](wafer::TileRegionOp) {
      ++regionsByTile[tile.getTileIdAttr().getInt()];
    });
  });
  EXPECT_EQ(regionsByTile, (llvm::SmallVector<unsigned, 4>{1, 1, 2, 1}));
}

TEST(SingleRootTileRegionTest,
     MaterializesTemporalWavesInsideSpatialReductionContributions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @reduce(%input: tensor<4x8xf16>, %init: tensor<4xf16>)
      -> tensor<4xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<4xf16>
    return %sum : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp root;
  source->walk([&](mlir::linalg::GenericOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  std::array<std::array<int64_t, 2>, 4> offsets = {
      std::array<int64_t, 2>{0, 0}, {0, 4}, {2, 0}, {2, 4}};
  llvm::SmallVector<wafer::StructuredNodeShardGroup, 4> groups;
  for (unsigned tile = 0; tile < 4; ++tile) {
    wafer::StructuredNodeShardGroup group;
    group.shards.push_back(wafer::StructuredNodeIterationShard{
        0,
        wafer::TileId(tile),
        {offsets[tile][0], offsets[tile][1]},
        {2, 4},
        wafer::StructuredNodeIterationShardRole::PartialReductionContribution,
        wafer::TileId(2)});
    group.temporalTiles.push_back(
        wafer::StructuredNodeTemporalTile{0, {1, 2}, {0, 1}});
    groups.push_back(std::move(group));
  }
  mlir::OwningOpRef<mlir::ModuleOp> materialized;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeGroupsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, groups, materialized,
      nullptr, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->getOperation()), 5u);
  EXPECT_GT(countOps<mlir::scf::ForOp>(materialized->getOperation()), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
}

TEST(SingleRootTileRegionTest, RejectsEffectfulDependencyAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  std::string sourceText = (llvm::Twine("module {") + topology + R"mlir(
  func.func @effect(%buffer: memref<1xf16>) -> tensor<4xf16> {
    %index = arith.constant 0 : index
    %value = memref.load %buffer[%index] : memref<1xf16>
    %empty = tensor.empty() : tensor<4xf16>
    %result = linalg.fill ins(%value : f16) outs(%empty : tensor<4xf16>)
        -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
})mlir")
                               .str();
  auto source = parse(*context, sourceText);
  ASSERT_TRUE(source);
  mlir::linalg::FillOp root;
  source->walk([&](mlir::linalg::FillOp operation) { root = operation; });
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{root.getOperation(), 0}};
  std::array<wafer::StructuredNodeIterationShard, 1> shards = {
      wafer::StructuredNodeIterationShard{0, wafer::TileId(0), {0}, {4}}};
  std::array<wafer::TileId, 4> tiles = {wafer::TileId(0), wafer::TileId(1),
                                        wafer::TileId(2), wafer::TileId(3)};
  mlir::OwningOpRef<mlir::ModuleOp> output;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerStructuredNodeShardsToCardModule(
      *source, wafer::CardId(0), tiles, operationNodes, shards, output, nullptr,
      &failureReason)));
  EXPECT_FALSE(output);
  EXPECT_NE(failureReason.find("effectful"), std::string::npos)
      << failureReason;
}

} // namespace
