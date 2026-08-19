//===- WaferTensorProgramToCardModuleTest.cpp - Card baseline tests -----===//

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Conversion/WaferCardModuleToTileModules/WaferCardModuleToTileModules.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/MemoryPlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <tuple>

namespace {

static std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
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

static std::string printOperation(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

template <typename OpT> static unsigned countOps(mlir::Operation *operation) {
  unsigned count = 0;
  operation->walk([&](OpT) { ++count; });
  return count;
}

static unsigned countUnreadDirectPrivateLoads(mlir::Operation *operation) {
  unsigned count = 0;
  operation->walk([&](wafer::StorageLoadOp load) {
    auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && allocation.getResult().hasOneUse())
      ++count;
  });
  return count;
}

struct StoreShard {
  int64_t tileId = -1;
  int64_t offset = -1;
  int64_t size = -1;

  friend bool operator==(const StoreShard &lhs, const StoreShard &rhs) {
    return std::tie(lhs.tileId, lhs.offset, lhs.size) ==
           std::tie(rhs.tileId, rhs.offset, rhs.size);
  }
};

static llvm::SmallVector<StoreShard, 16>
collectSecondDimensionStoreShards(wafer::CardModuleOp card) {
  llvm::SmallVector<StoreShard, 16> shards;
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    tile.walk([&](wafer::StorageStoreOp store) {
      auto view = store.getDest().getDefiningOp<mlir::memref::SubViewOp>();
      EXPECT_TRUE(view);
      if (!view || view.getStaticOffsets().size() < 2 ||
          view.getStaticSizes().size() < 2)
        return;
      shards.push_back(StoreShard{tile.getTileIdAttr().getInt(),
                                  view.getStaticOffsets()[1],
                                  view.getStaticSizes()[1]});
    });
  }
  llvm::sort(shards, [](const StoreShard &lhs, const StoreShard &rhs) {
    return lhs.tileId < rhs.tileId;
  });
  return shards;
}

static mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                                     llvm::StringRef text) {
  return mlir::parseSourceString<mlir::ModuleOp>(text,
                                                 mlir::ParserConfig(&context));
}

static wafer::TileMapping
mapping(unsigned shardDimension, std::initializer_list<int64_t> tileIds,
        std::initializer_list<int64_t> temporalTileSizes) {
  wafer::TileMapping result;
  wafer::OutputTileMapping output;
  output.outputIndex = 0;
  output.shardDimension = shardDimension;
  for (int64_t tileId : tileIds)
    output.activeTileIds.push_back(wafer::TileId(tileId));
  output.temporalTileSizes.append(temporalTileSizes.begin(),
                                  temporalTileSizes.end());
  result.outputs.push_back(std::move(output));
  return result;
}

static wafer::OutputTileMapping
outputMapping(unsigned outputIndex, unsigned shardDimension,
              std::initializer_list<int64_t> tileIds,
              std::initializer_list<int64_t> temporalTileSizes) {
  wafer::OutputTileMapping output;
  output.outputIndex = outputIndex;
  output.shardDimension = shardDimension;
  for (int64_t tileId : tileIds)
    output.activeTileIds.push_back(wafer::TileId(tileId));
  output.temporalTileSizes.append(temporalTileSizes.begin(),
                                  temporalTileSizes.end());
  return output;
}

static wafer::TileMapping
multiOutputMapping(std::initializer_list<wafer::OutputTileMapping> outputs) {
  wafer::TileMapping result;
  result.outputs.append(outputs.begin(), outputs.end());
  return result;
}

static wafer::SpatialEdgeStrategy
edgeStrategy(mlir::Operation *producer, mlir::Operation *consumer,
             wafer::TileId tile, wafer::SpatialEdgeAction action,
             llvm::ArrayRef<int64_t> producerOffsets,
             llvm::ArrayRef<int64_t> producerSizes,
             llvm::ArrayRef<int64_t> consumerOffsets = {},
             llvm::ArrayRef<int64_t> consumerSizes = {}) {
  wafer::SpatialEdgeStrategy result;
  result.producer = producer;
  result.consumer = consumer;
  result.producerOffsets.assign(producerOffsets.begin(), producerOffsets.end());
  result.producerSizes.assign(producerSizes.begin(), producerSizes.end());
  result.consumerOffsets.assign(consumerOffsets.begin(), consumerOffsets.end());
  result.consumerSizes.assign(consumerSizes.begin(), consumerSizes.end());
  result.sourceTile = tile;
  result.destinationTile = tile;
  result.action = action;
  return result;
}

static wafer::TileMapping completeTemporalMapping(mlir::ModuleOp source,
                                                  wafer::TileMapping mapping) {
  llvm::DenseSet<mlir::Operation *> configured;
  for (const wafer::StructuredOpTemporalTile &tile :
       mapping.operationTemporalTiles)
    configured.insert(tile.operation);

  for (mlir::func::FuncOp function : source.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal() || !function.getBody().hasOneBlock())
      continue;
    for (mlir::Operation &operation :
         function.getBody().front().without_terminator()) {
      auto tiling = mlir::dyn_cast<mlir::TilingInterface>(&operation);
      if (!tiling ||
          !mlir::isa<mlir::DestinationStyleOpInterface>(&operation) ||
          configured.contains(&operation))
        continue;
      llvm::SmallVector<int64_t, 4> ranges;
      if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(&operation)) {
        ranges = linalg.getStaticLoopRanges();
      } else if (operation.getNumResults() == 1) {
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
            operation.getResult(0).getType());
        if (resultType && resultType.hasStaticShape())
          ranges.assign(resultType.getShape().begin(),
                        resultType.getShape().end());
      }
      if (ranges.size() != tiling.getLoopIteratorTypes().size()) {
        ADD_FAILURE() << "test fixture cannot derive complete iterator tiles";
        continue;
      }
      mapping.operationTemporalTiles.push_back(
          wafer::StructuredOpTemporalTile{&operation, std::move(ranges)});
      configured.insert(&operation);
    }

    // Test mappings spell only the dimension under test. Complete every
    // direct structured data edge with the ordinary fused action so fixtures
    // exercise the production all-edge contract without a compatibility API.
    for (mlir::Operation &operation :
         function.getBody().front().without_terminator()) {
      auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(&operation);
      if (!dps || !mlir::isa<mlir::TilingInterface>(&operation) ||
          mapping.outputs.empty() ||
          mapping.outputs.front().activeTileIds.empty())
        continue;
      for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
        auto producerResult = mlir::dyn_cast<mlir::OpResult>(operand->get());
        mlir::Operation *producer =
            producerResult ? producerResult.getOwner() : nullptr;
        if (!producer || producer->getBlock() != operation.getBlock() ||
            !mlir::isa<mlir::TilingInterface>(producer) ||
            !mlir::isa<mlir::DestinationStyleOpInterface>(producer))
          continue;
        bool configuredEdge = llvm::any_of(
            mapping.edgeStrategies,
            [&](const wafer::SpatialEdgeStrategy &edge) {
              return edge.producer == producer && edge.consumer == &operation &&
                     edge.consumerOperand == operand->getOperandNumber();
            });
        if (configuredEdge)
          continue;
        auto producerType =
            mlir::dyn_cast<mlir::RankedTensorType>(producerResult.getType());
        if (!producerType || !producerType.hasStaticShape()) {
          ADD_FAILURE() << "test fixture cannot derive a static edge demand";
          continue;
        }
        llvm::SmallVector<int64_t, 4> offsets(producerType.getRank(), 0);
        wafer::SpatialEdgeStrategy fused = edgeStrategy(
            producer, &operation, mapping.outputs.front().activeTileIds.front(),
            wafer::SpatialEdgeAction::CoupledFusion, offsets,
            producerType.getShape());
        fused.producerResult = producerResult.getResultNumber();
        fused.consumerOperand = operand->getOperandNumber();
        mapping.edgeStrategies.push_back(std::move(fused));
      }
    }
  }
  return mapping;
}

static mlir::LogicalResult lowerCompleteTensorProgramToCardModule(
    mlir::ModuleOp source, wafer::CardId cardId, wafer::TileMapping mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<wafer::StructuredOperationNodeMapping> operationNodes = {},
    wafer::StructuredMaterializationRelations *materializationRelations =
        nullptr) {
  mlir::func::FuncOp function = *source.getOps<mlir::func::FuncOp>().begin();
  std::string dagFailure;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &dagFailure);
  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 16> derivedNodes;
  if (mlir::succeeded(dag) && operationNodes.empty()) {
    derivedNodes.reserve(dag->getNodes().size());
    for (const auto &node : dag->getNodes())
      derivedNodes.push_back({node.operation, node.id});
    operationNodes = derivedNodes;
  }
  return wafer::lowerTensorProgramToCardModule(
      source, cardId, completeTemporalMapping(source, std::move(mapping)),
      cardModule, failureReason, operationNodes, materializationRelations);
}

static wafer::TileMapping
mixedLocalRemoteFaninMapping(mlir::Operation *producer,
                             mlir::Operation *consumer) {
  wafer::TileMapping result = mapping(/*shardDimension=*/1, {1, 2}, {4, 2});
  result.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;
  wafer::SpatialEdgeStrategy first =
      edgeStrategy(producer, consumer, wafer::TileId(1),
                   wafer::SpatialEdgeAction::PeerFragments,
                   /*producerOffsets=*/{0, 0}, /*producerSizes=*/{4, 2},
                   /*consumerOffsets=*/{0, 0}, /*consumerSizes=*/{4, 2});
  first.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {0, 0},
                                 {2, 2},
                                 wafer::TileId(0),
                                 /*bytes=*/8,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/0});
  first.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Resident,
                                 {2, 0},
                                 {2, 2},
                                 wafer::TileId(1)});
  result.edgeStrategies.push_back(std::move(first));

  wafer::SpatialEdgeStrategy second =
      edgeStrategy(producer, consumer, wafer::TileId(2),
                   wafer::SpatialEdgeAction::PeerFragments,
                   /*producerOffsets=*/{0, 2}, /*producerSizes=*/{4, 2},
                   /*consumerOffsets=*/{0, 2}, /*consumerSizes=*/{4, 2});
  second.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {0, 2},
                                 {2, 2},
                                 wafer::TileId(0),
                                 /*bytes=*/8,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/1});
  second.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {2, 2},
                                 {2, 2},
                                 wafer::TileId(1),
                                 /*bytes=*/8,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/2});
  result.edgeStrategies.push_back(std::move(second));
  return result;
}

TEST(WaferTensorProgramToCardModuleTest,
     CoversFourByFourCardWithDistinctBalancedTileBodies) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module attributes {test.card_baseline = "preserved"} {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  memref.global "private" @shared_weights
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  func.func @matmul(%lhs: tensor<8x16xf16>, %rhs: tensor<16x64xf16>)
      -> tensor<8x64xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<8x64xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<8x64xf16>) -> tensor<8x64xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<8x16xf16>, tensor<16x64xf16>)
        outs(%init : tensor<8x64xf16>) -> tensor<8x64xf16>
    return %result : tensor<8x64xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  const std::string sourceBefore = printOperation(source->getOperation());

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      mapping(/*shardDimension=*/1,
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {8, 4}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(cardModule);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(printOperation(source->getOperation()), sourceBefore);
  EXPECT_TRUE(cardModule->getOperation()->hasAttr("test.card_baseline"));
  EXPECT_EQ(countOps<wafer::TargetTopologyOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::ExecutionMeshOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<mlir::memref::GlobalOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::CardModuleOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::TileModuleOp>(cardModule->getOperation()), 16u);
  EXPECT_EQ(countOps<mlir::func::FuncOp>(cardModule->getOperation()), 16u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(cardModule->getOperation()), 16u);

  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  llvm::SmallVector<StoreShard, 16> shards =
      collectSecondDimensionStoreShards(card);
  ASSERT_EQ(shards.size(), 16u);
  for (unsigned index = 0; index < shards.size(); ++index) {
    EXPECT_EQ(shards[index].tileId, static_cast<int64_t>(index));
    EXPECT_EQ(shards[index].offset, static_cast<int64_t>(index * 4));
    EXPECT_EQ(shards[index].size, 4);
  }
  for (unsigned index = 1; index < shards.size(); ++index)
    EXPECT_EQ(shards[index - 1].offset + shards[index - 1].size,
              shards[index].offset);
  EXPECT_EQ(shards.back().offset + shards.back().size, 64);

  llvm::SmallVector<std::string, 16> tileBodies;
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>())
    tileBodies.push_back(printOperation(tile.getOperation()));
  EXPECT_NE(tileBodies.front(), tileBodies.back());
}

TEST(WaferTensorProgramToCardModuleTest,
     OperandBufferRelationsIdentifyExplicitStructuredInputs) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @input_role(%input: tensor<4x4xf16>,
                        %init: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(m, n) -> (m, n)>,
                         affine_map<(m, n) -> (m, n)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<4x4xf16>)
        outs(%init : tensor<4x4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %old : f16
        linalg.yield %sum : f16
    } -> tensor<4x4xf16>
    return %result : tensor<4x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  auto generic = *function.getOps<mlir::linalg::GenericOp>().begin();

  wafer::TileMapping selected = completeTemporalMapping(
      *source, mapping(/*shardDimension=*/0, {0}, {4, 4}));
  selected.outputs.front().temporalTileSizes = {4, 1};
  ASSERT_EQ(selected.operationTemporalTiles.size(), 1u);
  selected.operationTemporalTiles.front().iteratorTileSizes = {4, 1};
  std::array<wafer::StructuredOperationNodeMapping, 1> operationNodes = {
      wafer::StructuredOperationNodeMapping{generic.getOperation(), 0}};
  wafer::StructuredMaterializationRelations relations;

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason,
      operationNodes, &relations)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  ASSERT_FALSE(relations.operandBuffers.empty());
  for (const wafer::StructuredOperationBufferRelation &relation :
       relations.operandBuffers) {
    EXPECT_EQ(relation.structuredNodeId, 0u);
    auto type = mlir::cast<mlir::MemRefType>(relation.buffer.getType());
    EXPECT_LE(type.getNumElements(), 4);
    EXPECT_TRUE(llvm::any_of(cardModule->getOps<wafer::CardModuleOp>(),
                             [&](wafer::CardModuleOp card) {
                               bool producedByLoad = false;
                               card.walk([&](wafer::StorageLoadOp load) {
                                 producedByLoad |=
                                     load.getDest() == relation.buffer;
                               });
                               return producedByLoad;
                             }));
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     OutputUpdateTileAllocationRetainsOutputBufferRelation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @cache_update(%past: tensor<7x4xf16>,
                          %row: tensor<1x4xf16>) -> tensor<8x4xf16> {
    %empty = tensor.empty() : tensor<8x4xf16>
    %prefix = tensor.insert_slice %past into %empty[0, 0] [7, 4] [1, 1]
        : tensor<7x4xf16> into tensor<8x4xf16>
    %result = tensor.insert_slice %row into %prefix[7, 0] [1, 4] [1, 1]
        : tensor<1x4xf16> into tensor<8x4xf16>
    return %result : tensor<8x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  wafer::TileMapping selected = completeTemporalMapping(
      *source, mapping(/*shardDimension=*/0, {0}, {4, 4}));
  wafer::StructuredMaterializationRelations relations;

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason,
      /*operationNodes=*/{}, &relations)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  ASSERT_FALSE(relations.outputBuffers.empty());
  for (const wafer::SpatialOutputBufferRelation &relation :
       relations.outputBuffers) {
    EXPECT_EQ(relation.outputIndex, 0u);
    auto type = mlir::cast<mlir::MemRefType>(relation.buffer.getType());
    EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>({8, 4}));
    EXPECT_TRUE(wafer::isWaferDDRMemRefType(type));
  }
  unsigned boundedTileAllocations = 0;
  cardModule->walk([&](mlir::memref::AllocOp allocation) {
    if (!wafer::isWaferSPMMemRefType(allocation.getType()))
      return;
    ++boundedTileAllocations;
    EXPECT_LE(allocation.getType().getNumElements(), 16)
        << printOperation(allocation.getOperation());
  });
  EXPECT_GT(boundedTileAllocations, 0u);

  // The 7-row prefix intersects 4-row traversal waves as the finite static
  // classes {3, 4}. Their mutually exclusive scf.if branches may reuse the
  // same pre-insert destination version, and must remain representable all the
  // way through Tile instruction lowering.
  auto tileModules = wafer::splitCardModuleIntoTileModules(
      std::move(cardModule), &failureReason, &relations);
  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  for (wafer::TileModule &tile : *tileModules) {
    EXPECT_EQ(tile.materializationRelations.outputBuffers.empty(),
              tile.tileId != wafer::TileId(0));
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstrModule(*tile.module)))
        << "Tile " << tile.tileId.getValue();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*tile.module)));
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     CoupledFullWindowKeepsProducerOperandLoadsTemporallyTiled) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @coupled(%weight: tensor<8x8xf16>, %lhs: tensor<1x8xf16>,
                     %init: tensor<1x8xf16>) -> tensor<1x8xf16> {
    %empty = tensor.empty() : tensor<8x8xf16>
    %transpose = linalg.generic {
        indexing_maps = [affine_map<(m, n) -> (n, m)>,
                         affine_map<(m, n) -> (m, n)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<8x8xf16>) outs(%empty : tensor<8x8xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<8x8xf16>
    %result = linalg.matmul ins(%lhs, %transpose
        : tensor<1x8xf16>, tensor<8x8xf16>)
        outs(%init : tensor<1x8xf16>) -> tensor<1x8xf16>
    return %result : tensor<1x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  llvm::SmallVector<mlir::linalg::LinalgOp, 2> structured;
  function.walk([&](mlir::linalg::LinalgOp op) { structured.push_back(op); });
  ASSERT_EQ(structured.size(), 2u);

  wafer::TileMapping selected = completeTemporalMapping(
      *source, mapping(/*shardDimension=*/1, {0}, {1, 8}));
  ASSERT_EQ(selected.operationTemporalTiles.size(), 2u);
  selected.operationTemporalTiles[0].iteratorTileSizes = {8, 2};
  selected.operationTemporalTiles[1].iteratorTileSizes = {1, 2, 8};

  std::array<wafer::StructuredOperationNodeMapping, 2> operationNodes = {
      wafer::StructuredOperationNodeMapping{structured[0].getOperation(), 0},
      wafer::StructuredOperationNodeMapping{structured[1].getOperation(), 1}};
  wafer::StructuredMaterializationRelations relations;

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason,
      operationNodes, &relations)))
      << failureReason;

  unsigned producerBuffers = 0;
  for (const wafer::StructuredOperationBufferRelation &relation :
       relations.operandBuffers) {
    if (relation.structuredNodeId != 0)
      continue;
    ++producerBuffers;
    auto type = mlir::cast<mlir::MemRefType>(relation.buffer.getType());
    EXPECT_LE(type.getNumElements(), 16);
  }
  EXPECT_GT(producerBuffers, 0u);

  cardModule->walk([&](mlir::memref::AllocOp allocation) {
    auto type = allocation.getType();
    EXPECT_LT(type.getNumElements(), 64)
        << "a consumer-driven coupled wave must not assemble the full "
           "producer window: "
        << printOperation(allocation.getOperation());
  });
}

TEST(WaferTensorProgramToCardModuleTest,
     SharedOutputClosuresKeepCoupledWeightWindowsTemporallyTiled) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @shared_outputs(%weight: tensor<8x8xf16>,
                            %lhs: tensor<1x8xf16>,
                            %init: tensor<1x8xf16>)
      -> (tensor<1x2x1x4xf16>, tensor<1x2x1x4xf16>) {
    %empty = tensor.empty() : tensor<8x8xf16>
    %transpose = linalg.generic {
        indexing_maps = [affine_map<(m, n) -> (n, m)>,
                         affine_map<(m, n) -> (m, n)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<8x8xf16>) outs(%empty : tensor<8x8xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<8x8xf16>
    %product = linalg.matmul ins(%lhs, %transpose
        : tensor<1x8xf16>, tensor<8x8xf16>)
        outs(%init : tensor<1x8xf16>) -> tensor<1x8xf16>
    %expanded = tensor.expand_shape %product [[0], [1, 2, 3]]
        output_shape [1, 2, 1, 4]
        : tensor<1x8xf16> into tensor<1x2x1x4xf16>
    %first_out = tensor.empty() : tensor<1x2x1x4xf16>
    %first = linalg.map ins(%expanded : tensor<1x2x1x4xf16>)
        outs(%first_out : tensor<1x2x1x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %second_out = tensor.empty() : tensor<1x2x1x4xf16>
    %second = linalg.map ins(%expanded : tensor<1x2x1x4xf16>)
        outs(%second_out : tensor<1x2x1x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %first_consumer_out = tensor.empty() : tensor<1x2x1x4xf16>
    %first_consumer = linalg.map ins(%first : tensor<1x2x1x4xf16>)
        outs(%first_consumer_out : tensor<1x2x1x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %second_consumer_out = tensor.empty() : tensor<1x2x1x4xf16>
    %second_consumer = linalg.map ins(%second : tensor<1x2x1x4xf16>)
        outs(%second_consumer_out : tensor<1x2x1x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %first_consumer, %second_consumer
        : tensor<1x2x1x4xf16>, tensor<1x2x1x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  llvm::SmallVector<mlir::linalg::LinalgOp, 6> structured;
  function.walk([&](mlir::linalg::LinalgOp op) { structured.push_back(op); });
  ASSERT_EQ(structured.size(), 6u);

  wafer::TileMapping selected =
      multiOutputMapping({outputMapping(0, 3, {0}, {1, 2, 1, 2}),
                          outputMapping(1, 3, {0}, {1, 2, 1, 2})});
  for (auto [producer, consumer, communicationId] :
       {std::tuple(structured[2].getOperation(), structured[4].getOperation(),
                   int64_t{0}),
        std::tuple(structured[3].getOperation(), structured[5].getOperation(),
                   int64_t{1})}) {
    wafer::SpatialEdgeStrategy peer =
        edgeStrategy(producer, consumer, wafer::TileId(0),
                     wafer::SpatialEdgeAction::PeerFragments,
                     /*producerOffsets=*/{0, 0, 0, 0},
                     /*producerSizes=*/{1, 2, 1, 4},
                     /*consumerOffsets=*/{0, 0, 0, 0},
                     /*consumerSizes=*/{1, 2, 1, 4});
    peer.sourceTile = wafer::TileId(1);
    peer.fragments.push_back(wafer::SpatialEdgeFragment{
        wafer::SpatialEdgeFragmentKind::Peer,
        /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 2, 1, 4}, wafer::TileId(1),
        /*bytes=*/16, communicationId,
        /*payloadSlice=*/0});
    selected.edgeStrategies.push_back(std::move(peer));
  }
  selected = completeTemporalMapping(*source, std::move(selected));
  ASSERT_EQ(selected.operationTemporalTiles.size(), 6u);
  selected.operationTemporalTiles[0].iteratorTileSizes = {8, 2};
  selected.operationTemporalTiles[1].iteratorTileSizes = {1, 2, 4};

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  bool sawCoupledWeightWindow = false;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
    if (type && type.getShape() == llvm::ArrayRef<int64_t>({2, 4}))
      sawCoupledWeightWindow = true;
  });
  cardModule->walk([&](mlir::memref::AllocOp allocation) {
    auto type = allocation.getType();
    EXPECT_LT(type.getNumElements(), 64)
        << "each shared-output traversal must consume the selected coupled "
           "weight wave without assembling the full producer: "
        << printOperation(allocation.getOperation());
  });
  EXPECT_TRUE(sawCoupledWeightWindow)
      << printOperation(cardModule->getOperation());
}

TEST(WaferTensorProgramToCardModuleTest,
     DirectlyLowersExactGenericGemmFromStructuredSemantics) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @generic_gemm(%lhs: tensor<2x4xf16>, %rhs: tensor<4x3xf16>)
      -> tensor<2x3xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x3xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x3xf16>) -> tensor<2x3xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                         affine_map<(m, n, k) -> (k, n)>,
                         affine_map<(m, n, k) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) {
      ^bb0(%left: f16, %right: f16, %acc: f16):
        %product = arith.mulf %left, %right : f16
        %sum = arith.addf %product, %acc : f16
        linalg.yield %sum : f16
    } -> tensor<2x3xf16>
    return %result : tensor<2x3xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), mapping(/*shardDimension=*/1, {0}, {2, 3}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(cardModule->getOperation()), 1u);

  // Program boundaries use the one canonical staged path: DDR transfers
  // enter and leave Tensor-layout SPM, while compute-specific layouts are
  // represented by explicit materialization inside the Tile region.
  unsigned loadCount = 0;
  unsigned storeCount = 0;
  bool boundaryTransfersUseTensorLayout = true;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    ++loadCount;
    auto type = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
    auto memory = type ? wafer::getWaferMemoryAttr(type) : wafer::MemoryAttr{};
    boundaryTransfersUseTensorLayout &=
        memory && memory.getLayout() == wafer::MemLayout::Tensor;
  });
  cardModule->walk([&](wafer::StorageStoreOp store) {
    ++storeCount;
    auto type = mlir::dyn_cast<mlir::MemRefType>(store.getSource().getType());
    auto memory = type ? wafer::getWaferMemoryAttr(type) : wafer::MemoryAttr{};
    boundaryTransfersUseTensorLayout &=
        memory && memory.getLayout() == wafer::MemLayout::Tensor;
  });
  EXPECT_GT(loadCount, 0u);
  EXPECT_GT(storeCount, 0u);
  EXPECT_TRUE(boundaryTransfersUseTensorLayout);
  EXPECT_GT(countOps<wafer::LayoutMaterializeOp>(cardModule->getOperation()),
            0u);
}

TEST(WaferTensorProgramToCardModuleTest,
     DerivesBoundaryOperandWindowThroughStructuredProducerRelations) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @transpose_matmul(%lhs: tensor<1x2x4xf16>,
                              %stored_rhs: tensor<3x4xf16>,
                              %residual: tensor<1x2x3xf16>)
      -> tensor<1x2x3xf16> {
    %collapsed_lhs = tensor.collapse_shape %lhs [[0, 1], [2]]
        : tensor<1x2x4xf16> into tensor<2x4xf16>
    %rhs_out = tensor.empty() : tensor<4x3xf16>
    %rhs = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%stored_rhs : tensor<3x4xf16>)
        outs(%rhs_out : tensor<4x3xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4x3xf16>
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x3xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x3xf16>) -> tensor<2x3xf16>
    %result = linalg.matmul
        ins(%collapsed_lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) -> tensor<2x3xf16>
    %expanded = tensor.expand_shape %result [[0, 1], [2]]
        output_shape [1, 2, 3]
        : tensor<2x3xf16> into tensor<1x2x3xf16>
    %add_out = tensor.empty() : tensor<1x2x3xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%expanded, %residual : tensor<1x2x3xf16>, tensor<1x2x3xf16>)
        outs(%add_out : tensor<1x2x3xf16>) {
      ^bb0(%value: f16, %other: f16, %old: f16):
        %added = arith.addf %value, %other : f16
        linalg.yield %added : f16
    } -> tensor<1x2x3xf16>
    return %sum : tensor<1x2x3xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), mapping(/*shardDimension=*/2, {0}, {1, 2, 1}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  bool sawExactStoredRhsWindow = false;
  bool sawFullStoredRhs = false;
  bool sawFullTransposedRhs = false;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(load.getSource().getType());
    if (!sourceType || sourceType.getRank() != 2)
      return;
    sawExactStoredRhsWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({1, 4});
    sawFullStoredRhs |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({3, 4});
  });
  cardModule->walk([&](mlir::Operation *operation) {
    for (mlir::Type type : operation->getResultTypes()) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
      sawFullTransposedRhs |=
          memref && memref.getShape() == llvm::ArrayRef<int64_t>({4, 3});
    }
  });
  EXPECT_TRUE(sawExactStoredRhsWindow);
  EXPECT_FALSE(sawFullStoredRhs);
  EXPECT_FALSE(sawFullTransposedRhs);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesInternalReductionTemporalPrologueSteadyAndTail) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @internal_reduction(%lhs: tensor<5x10xf16>,
                                %rhs: tensor<10x3xf16>)
      -> tensor<5x3xf16> {
    %zero = arith.constant 0.0 : f16
    %matmul_out = tensor.empty() : tensor<5x3xf16>
    %matmul_init = linalg.fill ins(%zero : f16)
        outs(%matmul_out : tensor<5x3xf16>) -> tensor<5x3xf16>
    %product = linalg.matmul
        ins(%lhs, %rhs : tensor<5x10xf16>, tensor<10x3xf16>)
        outs(%matmul_init : tensor<5x3xf16>) -> tensor<5x3xf16>
    %result_out = tensor.empty() : tensor<5x3xf16>
    %result = linalg.map ins(%product : tensor<5x3xf16>)
        outs(%result_out : tensor<5x3xf16>) (%value: f16) {
      %doubled = arith.addf %value, %value : f16
      linalg.yield %doubled : f16
    }
    return %result : tensor<5x3xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::linalg::MatmulOp sourceMatmul;
  source->walk(
      [&](mlir::linalg::MatmulOp operation) { sourceMatmul = operation; });
  ASSERT_TRUE(sourceMatmul);
  wafer::TileMapping selected = mapping(/*shardDimension=*/1, {0}, {5, 3});
  selected.operationTemporalTiles.push_back(
      wafer::StructuredOpTemporalTile{sourceMatmul.getOperation(), {5, 3, 4}});

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  bool sawSteadyRhsWindow = false;
  bool sawTailRhsWindow = false;
  bool sawFullRhs = false;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(load.getSource().getType());
    if (!sourceType || sourceType.getRank() != 2)
      return;
    sawSteadyRhsWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({4, 3});
    sawTailRhsWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({2, 3});
    sawFullRhs |= sourceType.getShape() == llvm::ArrayRef<int64_t>({10, 3});
  });
  EXPECT_TRUE(sawSteadyRhsWindow);
  EXPECT_TRUE(sawTailRhsWindow);
  EXPECT_FALSE(sawFullRhs);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(cardModule->getOperation()), 3u);
  unsigned accumulatorUpdates = 0;
  cardModule->walk([&](wafer::ComputeElementwiseIntoOp update) {
    ++accumulatorUpdates;
    ASSERT_EQ(update.getInputs().size(), 2u);
    EXPECT_EQ(update.getDest(), update.getInputs().front());
  });
  EXPECT_EQ(accumulatorUpdates, 2u);
  bool sawStableLoopCarriedAccumulator = false;
  cardModule->walk([&](mlir::scf::ForOp loop) {
    mlir::scf::YieldOp yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [argument, yielded] :
         llvm::zip_equal(loop.getRegionIterArgs(), yield.getOperands()))
      sawStableLoopCarriedAccumulator |= argument == yielded;
  });
  EXPECT_TRUE(sawStableLoopCarriedAccumulator);

  wafer::TileMapping mixedParallelReduction =
      mapping(/*shardDimension=*/1, {0}, {5, 3});
  mixedParallelReduction.operationTemporalTiles.push_back(
      wafer::StructuredOpTemporalTile{sourceMatmul.getOperation(), {2, 3, 4}});
  mlir::OwningOpRef<mlir::ModuleOp> mixedModule;
  failureReason.clear();
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), mixedParallelReduction, mixedModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*mixedModule)));
  EXPECT_GT(countOps<mlir::scf::ForOp>(mixedModule->getOperation()), 1u);
  bool sawParallelTail = false;
  mixedModule->walk([&](wafer::ComputeGemmOp gemm) {
    auto resultType = mlir::cast<mlir::MemRefType>(gemm.getResult().getType());
    sawParallelTail |= resultType.getShape() == llvm::ArrayRef<int64_t>({1, 3});
  });
  EXPECT_TRUE(sawParallelTail);

  auto tileModules = wafer::splitCardModuleIntoTileModules(
      std::move(cardModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  ASSERT_EQ(tileModules->size(), 1u);
  mlir::OwningOpRef<mlir::ModuleOp> tileModule =
      std::move(tileModules->front().module);
  ASSERT_TRUE(
      mlir::succeeded(wafer::convertTileRegionToInstrModule(*tileModule)));
  EXPECT_EQ(
      countOps<wafer::ComputeElementwiseIntoOp>(tileModule->getOperation()),
      0u);
  unsigned inPlaceAccumulatorUpdates = 0;
  tileModule->walk([&](wafer::InstrElementwiseOp operation) {
    if (!operation.getInputs().empty() &&
        operation.getInputs().front() == operation.getDest())
      ++inPlaceAccumulatorUpdates;
  });
  EXPECT_EQ(inPlaceAccumulatorUpdates, 2u);
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *tileModule, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*tileModule)));
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesEveryConfiguredReductionIteratorCompactly) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @two_reduction_axes(%input: tensor<2x10x11xf16>,
                                %init: tensor<2xf16>) -> tensor<2xf16> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0)>],
        iterator_types = ["parallel", "reduction", "reduction"]
      } ins(%input : tensor<2x10x11xf16>) outs(%init : tensor<2xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %sum = arith.addf %value, %acc : f16
        linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %result : tensor<2xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::linalg::GenericOp sourceReduction;
  source->walk(
      [&](mlir::linalg::GenericOp operation) { sourceReduction = operation; });
  ASSERT_TRUE(sourceReduction);
  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {2});
  selected.operationTemporalTiles.push_back(wafer::StructuredOpTemporalTile{
      sourceReduction.getOperation(), {2, 4, 3}});

  // Floating-point reduction reassociation (split and tree) is a supported
  // numeric transformation: no fast-math flag is consumed as a semantics
  // switch, and the typed comparator owns acceptance.
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  bool sawSteadyWindow = false;
  bool sawBothAxesTailWindow = false;
  bool sawFullInput = false;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(load.getSource().getType());
    if (!sourceType || sourceType.getRank() != 3)
      return;
    sawSteadyWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({2, 4, 3});
    sawBothAxesTailWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({2, 2, 2});
    sawFullInput |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({2, 10, 11});
  });
  EXPECT_TRUE(sawSteadyWindow);
  EXPECT_TRUE(sawBothAxesTailWindow);
  EXPECT_FALSE(sawFullInput);
  // Static IR size depends on reduction rank, not the 3 x 4 runtime chunks:
  // each configured axis contributes one prologue, steady loop and tail.
  EXPECT_EQ(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 4u);
  EXPECT_EQ(countOps<wafer::ComputeReduceOp>(cardModule->getOperation()), 9u);

  // Check the exact static templates represented by the nested compact
  // traversal. A dynamic offset must be the induction variable of the
  // corresponding reduction axis; when both are dynamic the inner axis must
  // be nested in the outer one. This catches mutable offset state leaking
  // from one outer prologue/steady/tail sibling into the next.
  using WindowSignature = std::tuple<int64_t, int64_t, int64_t, int64_t>;
  llvm::SmallVector<WindowSignature, 9> windows;
  auto getLoop = [](mlir::OpFoldResult offset) -> mlir::scf::ForOp {
    mlir::Value dynamic = mlir::dyn_cast<mlir::Value>(offset);
    if (!dynamic)
      return {};
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(dynamic);
    if (!argument)
      return {};
    return mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
  };
  auto expectLoopBounds = [](mlir::scf::ForOp loop, int64_t lower,
                             int64_t upper, int64_t step) {
    ASSERT_TRUE(loop);
    EXPECT_EQ(mlir::getConstantIntValue(
                  mlir::getAsOpFoldResult(loop.getLowerBound())),
              lower);
    EXPECT_EQ(mlir::getConstantIntValue(
                  mlir::getAsOpFoldResult(loop.getUpperBound())),
              upper);
    EXPECT_EQ(
        mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep())),
        step);
  };
  cardModule->walk([&](mlir::memref::SubViewOp view) {
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(view.getSourceType());
    if (!sourceType || sourceType.getRank() != 3)
      return;
    ASSERT_EQ(view.getMixedOffsets().size(), 3u);
    auto resultType = mlir::cast<mlir::MemRefType>(view.getType());
    ASSERT_EQ(resultType.getRank(), 3);
    EXPECT_EQ(resultType.getShape()[0], 2);

    mlir::OpFoldResult outerOffset = view.getMixedOffsets()[1];
    mlir::OpFoldResult innerOffset = view.getMixedOffsets()[2];
    std::optional<int64_t> outerConstant =
        mlir::getConstantIntValue(outerOffset);
    std::optional<int64_t> innerConstant =
        mlir::getConstantIntValue(innerOffset);
    mlir::scf::ForOp outerLoop = getLoop(outerOffset);
    mlir::scf::ForOp innerLoop = getLoop(innerOffset);
    if (!outerConstant)
      expectLoopBounds(outerLoop, 4, 8, 4);
    if (!innerConstant)
      expectLoopBounds(innerLoop, 3, 9, 3);
    if (outerLoop && innerLoop)
      EXPECT_TRUE(
          outerLoop.getOperation()->isProperAncestor(innerLoop.getOperation()));

    windows.emplace_back(outerConstant.value_or(-1), innerConstant.value_or(-1),
                         resultType.getShape()[1], resultType.getShape()[2]);
  });
  llvm::SmallVector<WindowSignature, 9> expectedWindows{
      {0, 0, 4, 3},  {0, -1, 4, 3},  {0, 9, 4, 2},
      {-1, 0, 4, 3}, {-1, -1, 4, 3}, {-1, 9, 4, 2},
      {8, 0, 2, 3},  {8, -1, 2, 3},  {8, 9, 2, 2},
  };
  llvm::sort(windows);
  llvm::sort(expectedWindows);
  EXPECT_EQ(windows, expectedWindows);

  // Numerically execute the runtime instances described by those checked
  // prologue/steady/tail templates. Every source element must contribute
  // exactly once, and the carried accumulator must match the unsplit
  // two-axis reduction for both output elements.
  constexpr std::array<std::pair<int64_t, int64_t>, 3> outerChunks{
      std::pair<int64_t, int64_t>{0, 4}, {4, 4}, {8, 2}};
  constexpr std::array<std::pair<int64_t, int64_t>, 4> innerChunks{
      std::pair<int64_t, int64_t>{0, 3}, {3, 3}, {6, 3}, {9, 2}};
  std::array<std::array<std::array<unsigned, 11>, 10>, 2> coverage{};
  std::array<int64_t, 2> direct{3, 5};
  std::array<int64_t, 2> chunked = direct;
  auto inputValue = [](int64_t output, int64_t outer, int64_t inner) {
    return 1 + (output * 3 + outer * 5 + inner) % 7;
  };
  for (int64_t output = 0; output < 2; ++output)
    for (int64_t outer = 0; outer < 10; ++outer)
      for (int64_t inner = 0; inner < 11; ++inner)
        direct[output] += inputValue(output, outer, inner);
  for (auto [outerOffset, outerSize] : outerChunks)
    for (auto [innerOffset, innerSize] : innerChunks)
      for (int64_t output = 0; output < 2; ++output)
        for (int64_t outer = outerOffset; outer < outerOffset + outerSize;
             ++outer)
          for (int64_t inner = innerOffset; inner < innerOffset + innerSize;
               ++inner) {
            ++coverage[output][outer][inner];
            chunked[output] += inputValue(output, outer, inner);
          }
  EXPECT_EQ(chunked, direct);
  for (const auto &outputCoverage : coverage)
    for (const auto &outerCoverage : outputCoverage)
      for (unsigned count : outerCoverage)
        EXPECT_EQ(count, 1u);
}

TEST(WaferTensorProgramToCardModuleTest,
     RejectsMissingStructuredIteratorTilesAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @pointwise(%input: tensor<4xf16>) -> tensor<4xf16> {
    %out = tensor.empty() : tensor<4xf16>
    %result = linalg.map ins(%input : tensor<4xf16>)
        outs(%out : tensor<4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string sourceBefore = printOperation(source->getOperation());
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), mapping(/*shardDimension=*/0, {0}, {4}),
      cardModule, &failureReason)));
  EXPECT_EQ(failureReason,
            "card structured temporal mapping must cover every scheduled "
            "operation exactly once");
  EXPECT_FALSE(cardModule);
  EXPECT_EQ(printOperation(source->getOperation()), sourceBefore);
}

TEST(WaferTensorProgramToCardModuleTest,
     DoesNotMakeAFullInternal4096SquareWeightResident) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @large_internal_weight(%lhs: tensor<1x4096xf16>,
                                   %rhs: tensor<4096x4096xf16>)
      -> tensor<1x4096xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<1x4096xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<1x4096xf16>) -> tensor<1x4096xf16>
    %product = linalg.matmul
        ins(%lhs, %rhs : tensor<1x4096xf16>, tensor<4096x4096xf16>)
        outs(%init : tensor<1x4096xf16>) -> tensor<1x4096xf16>
    %root_out = tensor.empty() : tensor<1x4096xf16>
    %root = linalg.map ins(%product : tensor<1x4096xf16>)
        outs(%root_out : tensor<1x4096xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %root : tensor<1x4096xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::linalg::MatmulOp sourceMatmul;
  source->walk(
      [&](mlir::linalg::MatmulOp operation) { sourceMatmul = operation; });
  ASSERT_TRUE(sourceMatmul);
  wafer::TileMapping selected = mapping(/*shardDimension=*/1, {0}, {1, 64});
  selected.operationTemporalTiles.push_back(wafer::StructuredOpTemporalTile{
      sourceMatmul.getOperation(), {1, 4096, 256}});

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  bool sawChunkWindow = false;
  bool sawFullWeightLoad = false;
  cardModule->walk([&](wafer::StorageLoadOp load) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(load.getSource().getType());
    if (!sourceType || sourceType.getRank() != 2)
      return;
    sawChunkWindow |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({256, 64});
    sawFullWeightLoad |=
        sourceType.getShape() == llvm::ArrayRef<int64_t>({4096, 4096});
  });
  EXPECT_TRUE(sawChunkWindow);
  EXPECT_FALSE(sawFullWeightLoad);
  EXPECT_GT(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 0u);
}

TEST(WaferTensorProgramToCardModuleTest,
     RejectsUnsupportedReductionRegroupingBeforeCardMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @unsupported_regrouping(%input: tensor<4x8xi32>,
                                    %init: tensor<4xi32>)
      -> tensor<4xi32> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xi32>) outs(%init : tensor<4xi32>) {
      ^bb0(%value: i32, %acc: i32):
        %maximum = arith.maxui %value, %acc : i32
        linalg.yield %maximum : i32
    } -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp reduction;
  source->walk(
      [&](mlir::linalg::GenericOp operation) { reduction = operation; });
  ASSERT_TRUE(reduction);

  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {4});
  selected.operationTemporalTiles.push_back(
      wafer::StructuredOpTemporalTile{reduction.getOperation(), {4, 4}});
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)));
  EXPECT_EQ(failureReason,
            "reduction partition cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
  EXPECT_FALSE(cardModule);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToCardModuleTest,
     DirectlyLowersOrdinaryGenericThroughBaseline) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @pointwise(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %value : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), mapping(/*shardDimension=*/0, {0}, {8}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(cardModule->getOperation()), 0u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(cardModule->getOperation()),
            1u);
  llvm::SmallVector<mlir::func::FuncOp, 1> entries;
  cardModule->walk([&](mlir::func::FuncOp entry) { entries.push_back(entry); });
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries.front().getNumArguments(), 1u);
  EXPECT_EQ(
      entries.front().getArgument(0).getType(),
      mlir::RankedTensorType::get({8}, mlir::Float16Type::get(context.get())));
}

TEST(WaferTensorProgramToCardModuleTest,
     BaselineConstructionProducesOneRootPerRegionWithRetargetedReload) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %value : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    %final = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>) outs(%final : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %old : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  mlir::linalg::GenericOp producer;
  mlir::linalg::GenericOp consumer;
  source->walk([&](mlir::linalg::GenericOp operation) {
    if (!producer)
      producer = operation;
    else
      consumer = operation;
  });
  ASSERT_TRUE(producer);
  ASSERT_TRUE(consumer);

  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
  selected.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;
  wafer::SpatialEdgeStrategy cut =
      edgeStrategy(producer.getOperation(), consumer.getOperation(),
                   wafer::TileId(0), wafer::SpatialEdgeAction::RegionCut,
                   /*producerOffsets=*/{0}, /*producerSizes=*/{8},
                   /*consumerOffsets=*/{0}, /*consumerSizes=*/{8});
  cut.producerResult = 0;
  cut.consumerOperand = 0;
  selected.edgeStrategies.push_back(std::move(cut));
  selected = completeTemporalMapping(*source, std::move(selected));

  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 2> operationNodes = {
      {producer.getOperation(), /*structuredNodeId=*/0},
      {consumer.getOperation(), /*structuredNodeId=*/1}};

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  wafer::StructuredMaterializationRelations relations;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason,
      operationNodes, &relations)))
      << failureReason;
  // Structure: the two roots form separate regions; every compute region
  // carries exactly one structured root.
  llvm::SmallVector<wafer::TileRegionOp, 4> regions;
  cardModule->walk([&](wafer::TileRegionOp region) {
    if (!region->getParentOfType<wafer::TileRegionOp>())
      regions.push_back(region);
  });
  // Structure: two sequential regions on the shared Tile, each carrying its
  // own compute, connected by explicit DDR boundary movement.
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(cardModule->getOperation()),
            2u);
  // The direct construction writes the producer stage and final output once
  // each. It does not recreate the former intermediate DDR assemblies.
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(cardModule->getOperation()), 2u);
  EXPECT_EQ(countOps<wafer::StorageLoadOp>(cardModule->getOperation()), 2u);
  // Both roots retain current-SSA operand evidence after the split. The
  // consumer relation names the explicit reload allocation rather than the
  // erased pre-cut value.
  ASSERT_EQ(relations.operandBuffers.size(), 2u);
  auto producerRelation =
      llvm::find_if(relations.operandBuffers, [](const auto &relation) {
        return relation.structuredNodeId == 0;
      });
  auto consumerRelation =
      llvm::find_if(relations.operandBuffers, [](const auto &relation) {
        return relation.structuredNodeId == 1;
      });
  ASSERT_NE(producerRelation, relations.operandBuffers.end());
  ASSERT_NE(consumerRelation, relations.operandBuffers.end());
  mlir::Value consumerOperand = consumerRelation->buffer;
  ASSERT_TRUE(consumerOperand);
  auto reloadAlloc = consumerOperand.getDefiningOp<mlir::memref::AllocOp>();
  ASSERT_TRUE(reloadAlloc);
  auto reloadType = mlir::dyn_cast<mlir::MemRefType>(reloadAlloc.getType());
  ASSERT_TRUE(reloadType);
  EXPECT_TRUE(mlir::isa<wafer::MemoryAttr>(reloadType.getMemorySpace()));
  EXPECT_EQ(
      mlir::cast<wafer::MemoryAttr>(reloadType.getMemorySpace()).getSpace(),
      wafer::MemorySpace::SPM);

  // The structural postcondition is relation-backed, not inferred from a
  // single visible compute op. Dropping the structured-node mapping must
  // reject the construction instead of admitting a zero-root compute region.
  mlir::OwningOpRef<mlir::ModuleOp> missingRelationModule;
  failureReason.clear();
  EXPECT_TRUE(mlir::failed(wafer::lowerTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, missingRelationModule,
      &failureReason,
      /*operationNodes=*/{}, /*materializationRelations=*/nullptr)));
  EXPECT_FALSE(missingRelationModule);
  EXPECT_NE(failureReason.find("no structured node identity"),
            std::string::npos)
      << failureReason;
}

TEST(WaferTensorProgramToCardModuleTest,
     IndependentRootsMaterializeTheirCapturedScalarConstantsLocally) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @cut_chain() -> tensor<8xf16> {
    %one = arith.constant 1.0 : f16
    %producer_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } outs(%producer_empty : tensor<8xf16>) {
      ^bb0(%old: f16):
        %sum = arith.addf %one, %one : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %one : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::linalg::GenericOp producer;
  mlir::linalg::GenericOp consumer;
  source->walk([&](mlir::linalg::GenericOp operation) {
    if (!producer)
      producer = operation;
    else
      consumer = operation;
  });
  ASSERT_TRUE(producer);
  ASSERT_TRUE(consumer);

  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
  selected.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;
  wafer::SpatialEdgeStrategy cut =
      edgeStrategy(producer.getOperation(), consumer.getOperation(),
                   wafer::TileId(0), wafer::SpatialEdgeAction::RegionCut,
                   /*producerOffsets=*/{0}, /*producerSizes=*/{8},
                   /*consumerOffsets=*/{0}, /*consumerSizes=*/{8});
  cut.producerResult = 0;
  cut.consumerOperand = 0;
  selected.edgeStrategies.push_back(std::move(cut));
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  unsigned rootCount = 0;
  cardModule->walk([&](wafer::TileRegionOp region) {
    if (region->getParentOfType<wafer::TileRegionOp>())
      return;
    ++rootCount;
    EXPECT_EQ(countOps<mlir::arith::ConstantOp>(region.getOperation()), 1u)
        << printOperation(region.getOperation());
  });
  EXPECT_EQ(rootCount, 2u);
}

TEST(WaferTensorProgramToCardModuleTest,
     SkipsUnavailableTopologyIdsWithoutRenumberingShards) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>,
       unavailable_tiles = array<i64: 0, 0, 0, 1>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @matmul(%lhs: tensor<2x4xf16>, %rhs: tensor<4x6xf16>)
      -> tensor<2x6xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x6xf16>) -> tensor<2x6xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x6xf16>)
        outs(%init : tensor<2x6xf16>) -> tensor<2x6xf16>
    return %result : tensor<2x6xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      mapping(/*shardDimension=*/1, {0, 2, 3}, {2, 2}), cardModule,
      &failureReason)))
      << failureReason;
  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  llvm::SmallVector<StoreShard, 16> shards =
      collectSecondDimensionStoreShards(card);
  ASSERT_EQ(shards.size(), 3u);
  EXPECT_EQ(shards[0], (StoreShard{0, 0, 2}));
  EXPECT_EQ(shards[1], (StoreShard{2, 2, 2}));
  EXPECT_EQ(shards[2], (StoreShard{3, 4, 2}));
}

TEST(WaferTensorProgramToCardModuleTest,
     CreatesDefinedNoWorkEntriesOnlyBeyondNonemptyShardCount) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @fill() -> tensor<2x3xf16> {
    %one = arith.constant 1.0 : f16
    %out = tensor.empty() : tensor<2x3xf16>
    %result = linalg.fill ins(%one : f16)
        outs(%out : tensor<2x3xf16>) -> tensor<2x3xf16>
    return %result : tensor<2x3xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      mapping(/*shardDimension=*/1, {0, 1, 2}, {2, 1}), cardModule,
      &failureReason)))
      << failureReason;
  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  unsigned active = 0;
  unsigned noWork = 0;
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    llvm::SmallVector<mlir::func::FuncOp, 1> functions(
        tile.getBody().front().getOps<mlir::func::FuncOp>());
    ASSERT_EQ(functions.size(), 1u);
    ASSERT_FALSE(functions.front().isExternal());
    if (countOps<wafer::TileRegionOp>(tile.getOperation()) == 0) {
      ++noWork;
      EXPECT_EQ(functions.front().getNumArguments(), 0u);
      EXPECT_EQ(functions.front().getNumResults(), 1u);
      EXPECT_EQ(countOps<mlir::tensor::EmptyOp>(tile.getOperation()), 1u);
      EXPECT_EQ(functions.front().getBody().front().getOperations().size(), 2u);
    } else {
      ++active;
    }
  }
  EXPECT_EQ(active, 3u);
  EXPECT_EQ(noWork, 1u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesIndependentHeterogeneousOutputsOnDisjointTileGroups) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @branches(%lhs: tensor<8xf16>, %rhs: tensor<3x6xf16>)
      -> (tensor<8xf16>, tensor<3x6xf16>) {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<3x6xf16>
    %first = linalg.map ins(%lhs : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %second = linalg.map ins(%rhs : tensor<3x6xf16>)
        outs(%out1 : tensor<3x6xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %first, %second : tensor<8xf16>, tensor<3x6xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      multiOutputMapping({outputMapping(0, 0, {0, 1}, {4}),
                          outputMapping(1, 1, {2, 3}, {3, 3})}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    const int64_t tileId = tile.getTileIdAttr().getInt();
    llvm::SmallVector<mlir::func::FuncOp, 1> functions(
        tile.getBody().front().getOps<mlir::func::FuncOp>());
    ASSERT_EQ(functions.size(), 1u);
    EXPECT_EQ(functions.front().getNumArguments(), 2u);
    EXPECT_EQ(functions.front().getNumResults(), 2u);
    EXPECT_EQ(countOps<wafer::TileRegionOp>(tile.getOperation()), 1u);
    EXPECT_EQ(countOps<wafer::StorageStoreOp>(tile.getOperation()), 1u);
    llvm::SmallVector<wafer::ComputeElementwiseOp, 1> elementwise;
    tile.walk([&](wafer::ComputeElementwiseOp operation) {
      elementwise.push_back(operation);
    });
    ASSERT_EQ(elementwise.size(), 1u);
    if (tileId < 2) {
      EXPECT_EQ(elementwise.front().getKind(),
                wafer::ComputeElementwiseKind::Add);
    } else {
      EXPECT_EQ(elementwise.front().getKind(),
                wafer::ComputeElementwiseKind::Mul);
    }
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesObservableProducerAndDependentResultTogether) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @dependent_outputs(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
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
    return %producer, %consumer : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      multiOutputMapping(
          {outputMapping(0, 0, {0}, {4}), outputMapping(1, 0, {0}, {4})}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::TileModuleOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(cardModule->getOperation()), 4u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(cardModule->getOperation()),
            6u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesShapeViewResultThroughStructuredOutputAnchor) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @shape_views(%input: tensor<1x1x8xf16>)
      -> tensor<1x1x8xf16> {
    %collapsed = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<1x1x8xf16> into tensor<1x8xf16>
    %out = tensor.empty() : tensor<1x8xf16>
    %mapped = linalg.map ins(%collapsed : tensor<1x8xf16>)
        outs(%out : tensor<1x8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %expanded = tensor.expand_shape %mapped [[0], [1, 2]]
        output_shape [1, 1, 8]
        : tensor<1x8xf16> into tensor<1x1x8xf16>
    return %expanded : tensor<1x1x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), mapping(/*shardDimension=*/2, {0}, {1, 1, 4}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(cardModule->getOperation()),
            2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(cardModule->getOperation()), 2u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesPerOutputTemporalLoopsAndFiniteTailClasses) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @branches(%lhs: tensor<1x10xf16>, %rhs: tensor<1x9xf16>)
      -> (tensor<1x10xf16>, tensor<1x9xf16>) {
    %out0 = tensor.empty() : tensor<1x10xf16>
    %out1 = tensor.empty() : tensor<1x9xf16>
    %first = linalg.map ins(%lhs : tensor<1x10xf16>)
        outs(%out0 : tensor<1x10xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %second = linalg.map ins(%rhs : tensor<1x9xf16>)
        outs(%out1 : tensor<1x9xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %first, %second : tensor<1x10xf16>, tensor<1x9xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      multiOutputMapping(
          {outputMapping(0, 1, {0}, {1, 4}), outputMapping(1, 1, {0}, {1, 3})}),
      cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 2u);
  // Output 0 has prologue, steady and a 2-element tail; output 1 has a
  // prologue and steady loop with no tail.
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(cardModule->getOperation()), 5u);

  llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 4> shapes;
  cardModule->walk([&](wafer::ComputeElementwiseOp operation) {
    auto type = mlir::cast<mlir::MemRefType>(operation.getResult().getType());
    shapes.emplace_back(type.getShape());
  });
  EXPECT_TRUE(llvm::is_contained(shapes, llvm::SmallVector<int64_t, 2>{1, 2}));
  EXPECT_TRUE(llvm::is_contained(shapes, llvm::SmallVector<int64_t, 2>{1, 3}));
  EXPECT_TRUE(llvm::is_contained(shapes, llvm::SmallVector<int64_t, 2>{1, 4}));
}

TEST(WaferTensorProgramToCardModuleTest,
     RejectsIncompleteOrDuplicatedOutputMappingsAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @branches(%lhs: tensor<4xf16>, %rhs: tensor<3xf16>)
      -> (tensor<4xf16>, tensor<3xf16>) {
    %out0 = tensor.empty() : tensor<4xf16>
    %out1 = tensor.empty() : tensor<3xf16>
    %first = linalg.map ins(%lhs : tensor<4xf16>)
        outs(%out0 : tensor<4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %second = linalg.map ins(%rhs : tensor<3xf16>)
        outs(%out1 : tensor<3xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %first, %second : tensor<4xf16>, tensor<3xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string sourceBefore = printOperation(source->getOperation());
  mlir::OwningOpRef<mlir::ModuleOp> unchanged =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  unchanged->getOperation()->setAttr("test.unchanged",
                                     mlir::UnitAttr::get(context.get()));
  mlir::ModuleOp unchangedPointer = *unchanged;
  std::string failureReason;

  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      multiOutputMapping({outputMapping(0, 0, {0, 1}, {2})}), unchanged,
      &failureReason)));
  EXPECT_EQ(failureReason,
            "card spatial mapping must cover every function result exactly "
            "once");
  EXPECT_EQ(*unchanged, unchangedPointer);

  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      multiOutputMapping(
          {outputMapping(0, 0, {0}, {4}), outputMapping(0, 0, {1}, {4})}),
      unchanged, &failureReason)));
  EXPECT_EQ(failureReason, "card spatial mapping output index is duplicated");
  EXPECT_EQ(*unchanged, unchangedPointer);
  EXPECT_TRUE(unchanged->getOperation()->hasAttr("test.unchanged"));
  EXPECT_EQ(printOperation(source->getOperation()), sourceBefore);
}

TEST(WaferTensorProgramToCardModuleTest, IsDeterministicThroughTileProjection) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 3>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @fill() -> tensor<2x7xf16> {
    %one = arith.constant 1.0 : f16
    %out = tensor.empty() : tensor<2x7xf16>
    %result = linalg.fill ins(%one : f16)
        outs(%out : tensor<2x7xf16>) -> tensor<2x7xf16>
    return %result : tensor<2x7xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  mlir::OwningOpRef<mlir::ModuleOp> first;
  mlir::OwningOpRef<mlir::ModuleOp> second;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      mapping(/*shardDimension=*/1, {0, 1, 2}, {2, 3}), first, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0),
      mapping(/*shardDimension=*/1, {0, 1, 2}, {2, 3}), second,
      &failureReason)))
      << failureReason;
  EXPECT_EQ(printOperation(first->getOperation()),
            printOperation(second->getOperation()));

  auto firstTileModules =
      wafer::splitCardModuleIntoTileModules(std::move(first), &failureReason);
  auto secondTileModules =
      wafer::splitCardModuleIntoTileModules(std::move(second), &failureReason);
  ASSERT_TRUE(mlir::succeeded(firstTileModules)) << failureReason;
  ASSERT_TRUE(mlir::succeeded(secondTileModules)) << failureReason;
  ASSERT_EQ(firstTileModules->size(), 3u);
  ASSERT_EQ(secondTileModules->size(), 3u);
  for (unsigned index = 0; index < firstTileModules->size(); ++index) {
    EXPECT_EQ((*firstTileModules)[index].tileId.getValue(),
              static_cast<int64_t>(index));
    EXPECT_EQ(
        printOperation((*firstTileModules)[index].module->getOperation()),
        printOperation((*secondTileModules)[index].module->getOperation()));
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesOnlyMissingDependentDomainsAsPeerInstructions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 6>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out0 : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.addf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>) outs(%out1 : tensor<8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.mulf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::GenericOp, 2> structured;
  source->walk([&](mlir::linalg::GenericOp operation) {
    structured.push_back(operation);
  });
  ASSERT_EQ(structured.size(), 2u);

  wafer::TileMapping selected =
      mapping(/*shardDimension=*/0, {0, 1, 4, 5}, {8});
  wafer::SpatialEdgeStrategy first =
      edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                   wafer::TileId(4), wafer::SpatialEdgeAction::PeerFragments,
                   /*producerOffsets=*/{4}, /*producerSizes=*/{2},
                   /*consumerOffsets=*/{4}, /*consumerSizes=*/{2});
  first.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {4},
                                 {2},
                                 wafer::TileId(2),
                                 /*bytes=*/4,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/0});
  selected.edgeStrategies.push_back(std::move(first));
  wafer::SpatialEdgeStrategy second =
      edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                   wafer::TileId(5), wafer::SpatialEdgeAction::PeerFragments,
                   /*producerOffsets=*/{6}, /*producerSizes=*/{2},
                   /*consumerOffsets=*/{6}, /*consumerSizes=*/{2});
  second.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {6},
                                 {2},
                                 wafer::TileId(3),
                                 /*bytes=*/4,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/1});
  selected.edgeStrategies.push_back(std::move(second));

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::CommPeerSendOp>(cardModule->getOperation()), 2u);
  EXPECT_EQ(countOps<wafer::CommPeerRecvOp>(cardModule->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::async::AwaitOp>(cardModule->getOperation()), 4u);

  auto tileModules = wafer::splitCardModuleIntoTileModules(
      std::move(cardModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  ASSERT_EQ(tileModules->size(), 6u);
  unsigned sends = 0;
  unsigned receives = 0;
  unsigned waits = 0;
  for (wafer::TileModule &tile : *tileModules) {
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstrModule(*tile.module)))
        << "Tile " << tile.tileId.getValue();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*tile.module)));
    sends += countOps<wafer::InstrDTESendOp>(tile.module->getOperation());
    receives += countOps<wafer::InstrDTERecvOp>(tile.module->getOperation());
    waits += countOps<wafer::InstrDTEWaitOp>(tile.module->getOperation());
  }
  EXPECT_EQ(sends, 2u);
  EXPECT_EQ(receives, 2u);
  EXPECT_EQ(waits, 4u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesExactFaninWhileOneTileSendsReceivesAndOwnsOutput) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 3>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @partial_fanin(%input: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %producer_empty = tensor.empty() : tensor<4x4xf16>
    %consumer_empty = tensor.empty() : tensor<4x4xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<4x4xf16>)
        outs(%producer_empty : tensor<4x4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %value : f16
        linalg.yield %sum : f16
    } -> tensor<4x4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<4x4xf16>)
        outs(%consumer_empty : tensor<4x4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %product = arith.mulf %value, %value : f16
        linalg.yield %product : f16
    } -> tensor<4x4xf16>
    return %consumer : tensor<4x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::GenericOp, 2> structured;
  source->walk([&](mlir::linalg::GenericOp operation) {
    structured.push_back(operation);
  });
  ASSERT_EQ(structured.size(), 2u);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  wafer::TileMapping selected =
      mixedLocalRemoteFaninMapping(structured[0], structured[1]);
  // Each remote fragment spans more than one temporal consumer wave. The
  // selected peer receive is one stable SPM endpoint outside those waves;
  // prologue/steady/tail traversal must not clone the message endpoint. The
  // baseline also completes one globally ordered endpoint before issuing the
  // next, so the number of live receiver FSMs is independent of fan-in.
  selected.outputs.front().temporalTileSizes = {1, 2};
  selected.operationTemporalTiles.push_back(
      {structured[1].getOperation(), {1, 2}});
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_GT(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 0u);
  EXPECT_EQ(countOps<wafer::CommPeerSendOp>(cardModule->getOperation()), 3u);
  EXPECT_EQ(countOps<wafer::CommPeerRecvOp>(cardModule->getOperation()), 3u);
  EXPECT_EQ(countOps<mlir::async::AwaitOp>(cardModule->getOperation()), 6u);
  EXPECT_EQ(countUnreadDirectPrivateLoads(cardModule->getOperation()), 0u);

  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    const int64_t tileId = tile.getTileIdAttr().getInt();
    const unsigned sends = countOps<wafer::CommPeerSendOp>(tile.getOperation());
    const unsigned receives =
        countOps<wafer::CommPeerRecvOp>(tile.getOperation());
    const unsigned stores =
        countOps<wafer::StorageStoreOp>(tile.getOperation());
    if (tileId == 0) {
      EXPECT_EQ(sends, 2u);
      EXPECT_EQ(receives, 0u);
      // The no-fusion baseline seals this source op in compiler-owned DDR
      // before its exact shards are reloaded into the two peer sends.
      EXPECT_GT(stores, 0u);
    } else if (tileId == 1) {
      EXPECT_EQ(sends, 1u);
      EXPECT_EQ(receives, 1u);
      EXPECT_GT(stores, 0u);
    } else {
      EXPECT_EQ(sends, 0u);
      EXPECT_EQ(receives, 2u);
      EXPECT_GT(stores, 0u);
      unsigned liveReceives = 0;
      unsigned maximumLiveReceives = 0;
      bool sawReceive = false;
      bool storedAfterReceive = false;
      tile.walk([&](mlir::Operation *operation) {
        if (mlir::isa<wafer::CommPeerRecvOp>(operation)) {
          if (sawReceive)
            EXPECT_TRUE(storedAfterReceive)
                << "peer fragment was not staged to DDR before the next "
                   "receive";
          sawReceive = true;
          storedAfterReceive = false;
          ++liveReceives;
          maximumLiveReceives = std::max(maximumLiveReceives, liveReceives);
        }
        if (sawReceive && mlir::isa<wafer::StorageStoreOp>(operation))
          storedAfterReceive = true;
        if (auto await = mlir::dyn_cast<mlir::async::AwaitOp>(operation))
          if (await.getOperand().getDefiningOp<wafer::CommPeerRecvOp>()) {
            ASSERT_GT(liveReceives, 0u);
            --liveReceives;
          }
      });
      EXPECT_TRUE(storedAfterReceive);
      EXPECT_EQ(liveReceives, 0u);
      EXPECT_EQ(maximumLiveReceives, 1u);
    }
  }

  auto tileModules = wafer::splitCardModuleIntoTileModules(
      std::move(cardModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  unsigned sends = 0;
  unsigned receives = 0;
  unsigned waits = 0;
  for (wafer::TileModule &tile : *tileModules) {
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstrModule(*tile.module)))
        << "Tile " << tile.tileId.getValue();
    sends += countOps<wafer::InstrDTESendOp>(tile.module->getOperation());
    receives += countOps<wafer::InstrDTERecvOp>(tile.module->getOperation());
    waits += countOps<wafer::InstrDTEWaitOp>(tile.module->getOperation());
  }
  EXPECT_EQ(sends, 3u);
  EXPECT_EQ(receives, 3u);
  EXPECT_EQ(waits, 6u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesPeerAndRegionCutInputsForOneStructuredConsumer) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @two_inputs(%left: tensor<8xf16>, %right: tensor<8xf16>)
      -> tensor<8xf16> {
    %left_empty = tensor.empty() : tensor<8xf16>
    %left_value = linalg.map ins(%left : tensor<8xf16>)
        outs(%left_empty : tensor<8xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %right_empty = tensor.empty() : tensor<8xf16>
    %right_value = linalg.map ins(%right : tensor<8xf16>)
        outs(%right_empty : tensor<8xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %result_empty = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%left_value, %right_value : tensor<8xf16>, tensor<8xf16>)
        outs(%result_empty : tensor<8xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);

  llvm::SmallVector<mlir::Operation *, 3> structured;
  source->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::LinalgOp>(operation) &&
        operation->getNumResults() == 1)
      structured.push_back(operation);
  });
  ASSERT_EQ(structured.size(), 3u);

  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {1}, {8});
  selected.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;

  wafer::SpatialEdgeStrategy remote = edgeStrategy(
      structured[0], structured[2], wafer::TileId(1),
      wafer::SpatialEdgeAction::PeerFragments, /*producerOffsets=*/{0},
      /*producerSizes=*/{8}, /*consumerOffsets=*/{0},
      /*consumerSizes=*/{8});
  remote.consumerOperand = 0;
  remote.fragments.push_back(
      wafer::SpatialEdgeFragment{wafer::SpatialEdgeFragmentKind::Peer,
                                 {0},
                                 {8},
                                 wafer::TileId(0),
                                 /*bytes=*/16,
                                 /*communicationId=*/0,
                                 /*payloadSlice=*/0});
  selected.edgeStrategies.push_back(std::move(remote));

  wafer::SpatialEdgeStrategy local =
      edgeStrategy(structured[1], structured[2], wafer::TileId(1),
                   wafer::SpatialEdgeAction::RegionCut, /*producerOffsets=*/{0},
                   /*producerSizes=*/{8}, /*consumerOffsets=*/{0},
                   /*consumerSizes=*/{8});
  local.consumerOperand = 1;
  selected.edgeStrategies.push_back(std::move(local));

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::CommPeerSendOp>(cardModule->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::CommPeerRecvOp>(cardModule->getOperation()), 1u);
  EXPECT_GT(countOps<wafer::StorageStoreOp>(cardModule->getOperation()), 0u);
  EXPECT_GT(countOps<wafer::StorageLoadOp>(cardModule->getOperation()), 0u);

  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    unsigned consumerRoots = 0;
    tile.walk([&](wafer::TileRegionOp region) {
      unsigned structuredRoots = 0;
      region.walk([&](mlir::Operation *operation) {
        structuredRoots += mlir::isa<wafer::ComputeElementwiseOp>(operation);
      });
      consumerRoots += structuredRoots == 1;
      EXPECT_LE(structuredRoots, 1u);
    });
    if (tile.getTileIdAttr().getInt() == 1)
      EXPECT_GT(consumerRoots, 0u);
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     OrdersIndependentPeerMessagesByConsumerUseNotMessageIdentity) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 3>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @crossed_message_order(%first: tensor<8xf16>,
                                   %second: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
    %producer0_empty = tensor.empty() : tensor<8xf16>
    %producer1_empty = tensor.empty() : tensor<8xf16>
    %early_empty = tensor.empty() : tensor<8xf16>
    %late_empty = tensor.empty() : tensor<8xf16>
    %producer0 = linalg.map ins(%first : tensor<8xf16>)
        outs(%producer0_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %producer1 = linalg.map ins(%second : tensor<8xf16>)
        outs(%producer1_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %early = linalg.map ins(%producer1 : tensor<8xf16>)
        outs(%early_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    %late = linalg.map ins(%producer0 : tensor<8xf16>)
        outs(%late_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %early, %late : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 4> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 4u);

  wafer::TileMapping selected = multiOutputMapping(
      {outputMapping(0, 0, {2}, {8}), outputMapping(1, 0, {2}, {8})});
  selected.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;
  auto appendPeer = [&](mlir::Operation *producer, mlir::Operation *consumer,
                        wafer::TileId sourceTile, int64_t communicationId) {
    wafer::SpatialEdgeStrategy strategy =
        edgeStrategy(producer, consumer, wafer::TileId(2),
                     wafer::SpatialEdgeAction::PeerFragments,
                     /*producerOffsets=*/{0}, /*producerSizes=*/{8},
                     /*consumerOffsets=*/{0}, /*consumerSizes=*/{8});
    strategy.sourceTile = sourceTile;
    strategy.fragments.push_back(wafer::SpatialEdgeFragment{
        wafer::SpatialEdgeFragmentKind::Peer,
        /*offsets=*/{0}, /*sizes=*/{8}, sourceTile, /*bytes=*/16,
        communicationId, /*payloadSlice=*/0});
    selected.edgeStrategies.push_back(std::move(strategy));
  };
  // Message 0 feeds the later consumer, while message 1 feeds the earlier
  // consumer. Message identity remains unchanged; only query-local issue order
  // follows the first consumer use.
  appendPeer(structured[0], structured[3], wafer::TileId(0), 0);
  appendPeer(structured[1], structured[2], wafer::TileId(1), 1);

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), selected, cardModule, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  llvm::SmallVector<int64_t, 2> destinationReceiveOrder;
  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    if (tile.getTileIdAttr().getInt() != 2)
      continue;
    tile.walk([&](wafer::CommPeerRecvOp receive) {
      destinationReceiveOrder.push_back(
          receive.getMessage().getCommunicationId());
    });
  }
  ASSERT_EQ(destinationReceiveOrder.size(), 2u);
  EXPECT_EQ(destinationReceiveOrder[0], 1);
  EXPECT_EQ(destinationReceiveOrder[1], 0);
}

TEST(WaferTensorProgramToCardModuleTest,
     AcceptsExactBroadcastConsumerInputDemandForOneConsumerShard) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @broadcast(%input: tensor<8xf16>) -> tensor<1x2x8xf16> {
    %producer_empty = tensor.empty() : tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<1x2x8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producer_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<1x2x8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %sum = arith.addf %value, %old : f16
        linalg.yield %sum : f16
    } -> tensor<1x2x8xf16>
    return %consumer : tensor<1x2x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::Operation *, 2> structured;
  source->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::LinalgOp>(operation))
      structured.push_back(operation);
  });
  ASSERT_EQ(structured.size(), 2u);

  auto makeSelected = [&](int64_t producerSize) {
    wafer::TileMapping selected =
        mapping(/*shardDimension=*/1, {0, 1}, {1, 1, 8});
    for (int64_t tile = 0; tile < 2; ++tile)
      selected.edgeStrategies.push_back(edgeStrategy(
          structured[0], structured[1], wafer::TileId(tile),
          wafer::SpatialEdgeAction::SpillReload,
          /*producerOffsets=*/{0}, /*producerSizes=*/{producerSize},
          /*consumerOffsets=*/{0, tile, 0},
          /*consumerSizes=*/{1, 1, 8}));
    return selected;
  };

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), makeSelected(/*producerSize=*/8), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  mlir::OwningOpRef<mlir::ModuleOp> invalidModule;
  failureReason.clear();
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), makeSelected(/*producerSize=*/4),
      invalidModule, &failureReason)));
  EXPECT_NE(failureReason.find("differs from the exact relation image"),
            std::string::npos)
      << failureReason;
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesRelationDerivedReductionPeerFragmentsToActualCardIR) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @reduce(%input: tensor<8x4xf16>) -> tensor<8xf16> {
    %producer_empty = tensor.empty() : tensor<8x4xf16>
    %result_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8x4xf16>)
        outs(%producer_empty : tensor<8x4xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%result_empty : tensor<8xf16>) -> tensor<8xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%producer : tensor<8x4xf16>) outs(%init : tensor<8xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<8xf16>
    return %sum : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  auto function = *source->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;

  auto placement = [&](wafer::compiler::detail::StructuredDAGNodeID node,
                       std::initializer_list<int64_t> tiles) {
    wafer::compiler::detail::StructuredDAGNodePlacement result;
    result.node = node;
    auto linalg =
        mlir::cast<mlir::linalg::LinalgOp>(dag->getNode(node)->operation);
    result.iteratorPartitionFactors.assign(
        linalg.getIteratorTypesArray().size(), 1);
    result.iteratorPartitionFactors[0] = tiles.size();
    for (int64_t tile : tiles)
      result.tiles.push_back(wafer::TileId(tile));
    return result;
  };
  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 3>
      placements = {placement(0, {0, 1}), placement(1, {2, 3}),
                    placement(2, {2, 3})};
  auto edgePlan = wafer::compiler::detail::deriveStructuredDAGEdgeStrategyPlan(
      *dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(edgePlan)) << failureReason;
  ASSERT_EQ(edgePlan->strategies.size(), 2u);
  ASSERT_EQ(edgePlan->totalPeerBytes, 64u);

  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {2, 3}, {4});
  selected.materializationMode =
      wafer::SpatialDataflowMaterializationMode::IndependentDDRStages;
  selected.edgeStrategies.append(edgePlan->strategies.begin(),
                                 edgePlan->strategies.end());
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::CommPeerSendOp>(cardModule->getOperation()), 2u);
  EXPECT_EQ(countOps<wafer::CommPeerRecvOp>(cardModule->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::async::AwaitOp>(cardModule->getOperation()), 4u);
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesDistinctLocalEdgeActionsInActualTileIR) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_empty = tensor.empty() : tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producer_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);

  auto lowerAction = [&](wafer::SpatialEdgeAction action) {
    wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
    selected.edgeStrategies.push_back(
        edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                     wafer::TileId(0), action, /*producerOffsets=*/{0},
                     /*producerSizes=*/{8}, /*consumerOffsets=*/{0},
                     /*consumerSizes=*/{8}));
    mlir::OwningOpRef<mlir::ModuleOp> result;
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
        *source, wafer::CardId(0), std::move(selected), result,
        &failureReason)))
        << failureReason;
    return result;
  };

  mlir::OwningOpRef<mlir::ModuleOp> fused =
      lowerAction(wafer::SpatialEdgeAction::CoupledFusion);
  mlir::OwningOpRef<mlir::ModuleOp> retained =
      lowerAction(wafer::SpatialEdgeAction::LocalShardResidency);
  mlir::OwningOpRef<mlir::ModuleOp> spilled =
      lowerAction(wafer::SpatialEdgeAction::SpillReload);
  mlir::OwningOpRef<mlir::ModuleOp> cut =
      lowerAction(wafer::SpatialEdgeAction::RegionCut);
  ASSERT_TRUE(fused && retained && spilled && cut);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(fused->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(retained->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(spilled->getOperation()), 1u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(cut->getOperation()), 2u);
  EXPECT_GT(countOps<wafer::MoveInsertSliceOp>(retained->getOperation()),
            countOps<wafer::MoveInsertSliceOp>(fused->getOperation()));
  EXPECT_GT(countOps<wafer::StorageStoreOp>(spilled->getOperation()),
            countOps<wafer::StorageStoreOp>(fused->getOperation()));
  EXPECT_GT(countOps<wafer::StorageLoadOp>(spilled->getOperation()),
            countOps<wafer::StorageLoadOp>(fused->getOperation()));
  EXPECT_GT(countOps<wafer::StorageStoreOp>(cut->getOperation()),
            countOps<wafer::StorageStoreOp>(fused->getOperation()));
  EXPECT_GT(countOps<wafer::StorageLoadOp>(cut->getOperation()),
            countOps<wafer::StorageLoadOp>(fused->getOperation()));

  std::string failureReason;
  for (mlir::OwningOpRef<mlir::ModuleOp> *actual : {&spilled, &cut}) {
    auto tileModules = wafer::splitCardModuleIntoTileModules(std::move(*actual),
                                                             &failureReason);
    ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
    ASSERT_EQ(tileModules->size(), 1u);
    ASSERT_TRUE(mlir::succeeded(
        wafer::convertTileRegionToInstrModule(*tileModules->front().module)));
    EXPECT_GT(countOps<wafer::InstrRDMAOp>(
                  tileModules->front().module->getOperation()),
              1u);
    EXPECT_GT(countOps<wafer::InstrWDMAOp>(
                  tileModules->front().module->getOperation()),
              1u);
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     TemporallyFusedProducerRetainsCurrentIRNodeRelations) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_empty = tensor.empty() : tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producer_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);

  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 2> operationNodes{
      {structured[0].getOperation(), 0}, {structured[1].getOperation(), 1}};
  wafer::StructuredMaterializationRelations relations;
  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
  selected.operationTemporalTiles.push_back(
      {structured[0].getOperation(), {4}});
  selected.operationTemporalTiles.push_back(
      {structured[1].getOperation(), {4}});
  selected.edgeStrategies.push_back(
      edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                   wafer::TileId(0), wafer::SpatialEdgeAction::CoupledFusion,
                   /*producerOffsets=*/{0}, /*producerSizes=*/{8},
                   /*consumerOffsets=*/{0}, /*consumerSizes=*/{8}));

  mlir::OwningOpRef<mlir::ModuleOp> fused;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), fused, &failureReason,
      operationNodes, &relations)))
      << failureReason;

  EXPECT_FALSE(relations.operationResultBuffers.empty());
  EXPECT_FALSE(relations.operandBuffers.empty());
  bool sawSharedProducerConsumerBuffer = false;
  for (const wafer::StructuredOperationBufferRelation &producer :
       relations.operationResultBuffers)
    for (const wafer::StructuredOperationBufferRelation &consumer :
         relations.operandBuffers)
      sawSharedProducerConsumerBuffer |=
          producer.structuredNodeId == 0 && consumer.structuredNodeId == 1 &&
          wafer::compiler::detail::shareStructuredBufferStorage(
              producer.buffer, consumer.buffer);
  EXPECT_TRUE(sawSharedProducerConsumerBuffer)
      << "coupled producer/consumer work must remain related by current SSA";

  auto tileModules = wafer::splitCardModuleIntoTileModules(
      std::move(fused), &failureReason, &relations);
  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  ASSERT_EQ(tileModules->size(), 1u);
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*tileModules->front().module)));
  EXPECT_FALSE(tileModules->front()
                   .materializationRelations.operationResultBuffers.empty());
  EXPECT_FALSE(
      tileModules->front().materializationRelations.operandBuffers.empty());
}

TEST(WaferTensorProgramToCardModuleTest,
     NestedTemporalTraversalPropagatesTensorSplatStateSafely) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @splat_chain() -> tensor<2x2x2x2xf16> {
    %c0 = arith.constant dense<0.0> : tensor<2x2x2x2xf16>
    %c1 = arith.constant dense<1.0> : tensor<2x2x2x2xf16>
    %c2 = arith.constant dense<2.0> : tensor<2x2x2x2xf16>
    %c3 = arith.constant dense<3.0> : tensor<2x2x2x2xf16>
    %c4 = arith.constant dense<4.0> : tensor<2x2x2x2xf16>
    %c5 = arith.constant dense<5.0> : tensor<2x2x2x2xf16>
    %c6 = arith.constant dense<6.0> : tensor<2x2x2x2xf16>
    %c7 = arith.constant dense<7.0> : tensor<2x2x2x2xf16>
    %out = tensor.empty() : tensor<2x2x2x2xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>,
                         affine_map<(i, j, k, l) -> (i, j, k, l)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 :
            tensor<2x2x2x2xf16>, tensor<2x2x2x2xf16>,
            tensor<2x2x2x2xf16>, tensor<2x2x2x2xf16>,
            tensor<2x2x2x2xf16>, tensor<2x2x2x2xf16>,
            tensor<2x2x2x2xf16>, tensor<2x2x2x2xf16>)
        outs(%out : tensor<2x2x2x2xf16>) {
      ^bb0(%v0: f16, %v1: f16, %v2: f16, %v3: f16,
           %v4: f16, %v5: f16, %v6: f16, %v7: f16, %unused: f16):
        %s0 = arith.addf %v0, %v1 : f16
        %s1 = arith.addf %s0, %v2 : f16
        %s2 = arith.addf %s1, %v3 : f16
        %s3 = arith.addf %s2, %v4 : f16
        %s4 = arith.addf %s3, %v5 : f16
        %s5 = arith.addf %s4, %v6 : f16
        %s6 = arith.addf %s5, %v7 : f16
        linalg.yield %s6 : f16
    } -> tensor<2x2x2x2xf16>
    return %result : tensor<2x2x2x2xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = *source->getOps<mlir::func::FuncOp>().begin();
  mlir::linalg::GenericOp generic =
      *function.getOps<mlir::linalg::GenericOp>().begin();

  wafer::TileMapping selected =
      mapping(/*shardDimension=*/3, {0}, {2, 2, 2, 2});
  selected.operationTemporalTiles.push_back(
      {generic.getOperation(), {1, 1, 1, 1}});
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_GE(countOps<mlir::scf::ForOp>(cardModule->getOperation()), 4u);
  EXPECT_GT(countOps<wafer::ComputeElementwiseOp>(cardModule->getOperation()),
            0u);
}

TEST(WaferTensorProgramToCardModuleTest,
     RecomputeClonesPureProducerInActualTileIR) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @fanout(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
    %producer_empty = tensor.empty() : tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producer_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %producer, %consumer : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);

  auto lowerAction = [&](wafer::SpatialEdgeAction action) {
    wafer::TileMapping selected = multiOutputMapping(
        {outputMapping(0, 0, {0}, {8}), outputMapping(1, 0, {0}, {8})});
    selected.edgeStrategies.push_back(
        edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                     wafer::TileId(0), action, /*producerOffsets=*/{0},
                     /*producerSizes=*/{8}, /*consumerOffsets=*/{0},
                     /*consumerSizes=*/{8}));
    mlir::OwningOpRef<mlir::ModuleOp> result;
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
        *source, wafer::CardId(0), std::move(selected), result,
        &failureReason)))
        << failureReason;
    return result;
  };
  mlir::OwningOpRef<mlir::ModuleOp> fused =
      lowerAction(wafer::SpatialEdgeAction::CoupledFusion);
  mlir::OwningOpRef<mlir::ModuleOp> recomputed =
      lowerAction(wafer::SpatialEdgeAction::Recompute);
  ASSERT_TRUE(fused && recomputed);
  EXPECT_GT(countOps<wafer::ComputeElementwiseOp>(recomputed->getOperation()),
            countOps<wafer::ComputeElementwiseOp>(fused->getOperation()));
}

TEST(WaferTensorProgramToCardModuleTest,
     PartialOutputTileKeepsSelectedSpillStorageExplicit) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @fanout(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
    %producer_empty = tensor.empty() : tensor<8xf16>
    %consumer_empty = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producer_empty : tensor<8xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%consumer_empty : tensor<8xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %producer, %consumer : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);

  wafer::TileMapping selected = multiOutputMapping(
      {outputMapping(0, 0, {0}, {8}), outputMapping(1, 0, {1}, {8})});
  selected.edgeStrategies.push_back(
      edgeStrategy(structured[0].getOperation(), structured[1].getOperation(),
                   wafer::TileId(1), wafer::SpatialEdgeAction::SpillReload,
                   /*producerOffsets=*/{0}, /*producerSizes=*/{8},
                   /*consumerOffsets=*/{0}, /*consumerSizes=*/{8}));

  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));

  wafer::CardModuleOp card = *cardModule->getOps<wafer::CardModuleOp>().begin();
  for (wafer::TileModuleOp tile :
       card.getBody().front().getOps<wafer::TileModuleOp>()) {
    if (tile.getTileIdAttr().getInt() != 1)
      continue;
    EXPECT_GT(countOps<wafer::StorageStoreOp>(tile.getOperation()), 0u);
    EXPECT_GT(countOps<wafer::StorageLoadOp>(tile.getOperation()), 0u);
  }
}

TEST(WaferTensorProgramToCardModuleTest,
     MaterializesMultipleSelectedRegionCutsAsConsecutiveRegions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %empty0 = tensor.empty() : tensor<8xf16>
    %empty1 = tensor.empty() : tensor<8xf16>
    %empty2 = tensor.empty() : tensor<8xf16>
    %first = linalg.map ins(%input : tensor<8xf16>)
        outs(%empty0 : tensor<8xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %second = linalg.map ins(%first : tensor<8xf16>)
        outs(%empty1 : tensor<8xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %third = linalg.map ins(%second : tensor<8xf16>)
        outs(%empty2 : tensor<8xf16>) (%value: f16) {
      %next = arith.subf %value, %value : f16
      linalg.yield %next : f16
    }
    return %third : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 3> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 3u);
  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
  for (unsigned index = 0; index < 2; ++index)
    selected.edgeStrategies.push_back(edgeStrategy(
        structured[index].getOperation(), structured[index + 1].getOperation(),
        wafer::TileId(0), wafer::SpatialEdgeAction::RegionCut,
        /*producerOffsets=*/{0}, /*producerSizes=*/{8},
        /*consumerOffsets=*/{0}, /*consumerSizes=*/{8}));
  mlir::OwningOpRef<mlir::ModuleOp> cardModule;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), cardModule,
      &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*cardModule)));
  EXPECT_EQ(countOps<wafer::TileRegionOp>(cardModule->getOperation()), 3u);
}

TEST(WaferTensorProgramToCardModuleTest,
     RejectsLocalConversionWithoutTypedLayoutRequestAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %empty0 = tensor.empty() : tensor<8xf16>
    %empty1 = tensor.empty() : tensor<8xf16>
    %first = linalg.map ins(%input : tensor<8xf16>)
        outs(%empty0 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %second = linalg.map ins(%first : tensor<8xf16>)
        outs(%empty1 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %second : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string sourceBefore = printOperation(source->getOperation());
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);
  wafer::TileMapping selected = mapping(/*shardDimension=*/0, {0}, {8});
  selected.edgeStrategies.push_back(edgeStrategy(
      structured[0].getOperation(), structured[1].getOperation(),
      wafer::TileId(0), wafer::SpatialEdgeAction::LocalPhysicalConversion,
      /*producerOffsets=*/{0}, /*producerSizes=*/{8},
      /*consumerOffsets=*/{0}, /*consumerSizes=*/{8}));
  mlir::OwningOpRef<mlir::ModuleOp> unchanged =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::ModuleOp unchangedPointer = *unchanged;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), std::move(selected), unchanged,
      &failureReason)));
  EXPECT_EQ(failureReason,
            "local physical conversion requires distinct typed layouts");
  EXPECT_EQ(*unchanged, unchangedPointer);
  EXPECT_EQ(printOperation(source->getOperation()), sourceBefore);
}

TEST(WaferTensorProgramToCardModuleTest,
     RejectsDependentFragmentCoverageHolesAndOverlapAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 3>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @partial_fanin(%input: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %producer_empty = tensor.empty() : tensor<4x4xf16>
    %consumer_empty = tensor.empty() : tensor<4x4xf16>
    %producer = linalg.map ins(%input : tensor<4x4xf16>)
        outs(%producer_empty : tensor<4x4xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    %consumer = linalg.map ins(%producer : tensor<4x4xf16>)
        outs(%consumer_empty : tensor<4x4xf16>) (%value: f16) {
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    }
    return %consumer : tensor<4x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::linalg::MapOp, 2> structured;
  source->walk(
      [&](mlir::linalg::MapOp operation) { structured.push_back(operation); });
  ASSERT_EQ(structured.size(), 2u);
  const std::string sourceBefore = printOperation(source->getOperation());

  mlir::OwningOpRef<mlir::ModuleOp> unchanged =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  unchanged->getOperation()->setAttr("test.unchanged",
                                     mlir::UnitAttr::get(context.get()));
  mlir::ModuleOp unchangedPointer = *unchanged;
  std::string failureReason;

  wafer::TileMapping hole =
      mixedLocalRemoteFaninMapping(structured[0], structured[1]);
  hole.edgeStrategies.front().fragments.pop_back();
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), hole, unchanged, &failureReason)));
  EXPECT_EQ(failureReason,
            "dependent fragments do not exactly cover the consumer demand");
  EXPECT_EQ(*unchanged, unchangedPointer);

  wafer::TileMapping overlap =
      mixedLocalRemoteFaninMapping(structured[0], structured[1]);
  overlap.edgeStrategies.front().fragments.back().offsets = {0, 0};
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), overlap, unchanged, &failureReason)));
  EXPECT_EQ(failureReason,
            "dependent fragments overlap within one consumer demand");
  EXPECT_EQ(*unchanged, unchangedPointer);

  wafer::TileMapping overflow =
      mixedLocalRemoteFaninMapping(structured[0], structured[1]);
  overflow.edgeStrategies.front().fragments.back().offsets = {
      std::numeric_limits<int64_t>::max(), 0};
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *source, wafer::CardId(0), overflow, unchanged, &failureReason)));
  EXPECT_EQ(failureReason,
            "dependent fragment extends outside its consumer demand");
  EXPECT_EQ(*unchanged, unchangedPointer);
  EXPECT_TRUE(unchanged->getOperation()->hasAttr("test.unchanged"));
  EXPECT_EQ(printOperation(source->getOperation()), sourceBefore);
}

TEST(WaferTensorProgramToCardModuleTest, RejectsDynamicDomainsAtomically) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  mlir::OwningOpRef<mlir::ModuleOp> unchanged =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  unchanged->getOperation()->setAttr("test.unchanged",
                                     mlir::UnitAttr::get(context.get()));
  mlir::ModuleOp unchangedPointer = *unchanged;
  std::string failureReason;
  auto dynamic = parseModule(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  func.func @dynamic(%size: index) -> tensor<?x8xf16> {
    %out = tensor.empty(%size) : tensor<?x8xf16>
    return %out : tensor<?x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(dynamic);
  EXPECT_TRUE(mlir::failed(lowerCompleteTensorProgramToCardModule(
      *dynamic, wafer::CardId(0), mapping(/*shardDimension=*/0, {0}, {1, 8}),
      unchanged, &failureReason)));
  EXPECT_EQ(failureReason,
            "card spatial mapping requires static tensor-program "
            "output shapes");
  EXPECT_EQ(*unchanged, unchangedPointer);
  EXPECT_TRUE(unchanged->getOperation()->hasAttr("test.unchanged"));
}

} // namespace
