//===- TemporalTilingTest.cpp -----------------------------------------===//

#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionMaterialization.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef body,
                                              llvm::StringRef inputType,
                                              llvm::StringRef resultType) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module {\n"
            "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
            "    func.func @entry(%input: "
         << inputType << ") -> " << resultType
         << " {\n"
            "      %result = wafer.tile.region(%input : "
         << inputType << ") -> (" << resultType
         << ") {\n"
            "      ^bb0(%arg: "
         << inputType << "):\n"
         << body << "\n        wafer.tile.yield %value : " << resultType
         << "\n      }\n"
            "      return %result : "
         << resultType
         << "\n    }\n"
            "  }\n"
            "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(stream.str(),
                                                 mlir::ParserConfig(&context));
}

TileRegionOp findRegion(mlir::ModuleOp module) {
  TileRegionOp result;
  module.walk([&](TileRegionOp region) { result = region; });
  return result;
}

TemporalChoice selectTileSizes(const TemporalDomain &domain,
                               llvm::ArrayRef<int64_t> sizes) {
  TemporalSuccessor first = domain.getFirstChoice();
  EXPECT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  EXPECT_NE(first.getChoice(), nullptr);
  TemporalChoice choice = *first.getChoice();
  EXPECT_EQ(choice.scopes.size(), 1u);
  if (choice.scopes.size() != 1)
    return choice;
  choice.scopes.front().iteratorTileSizes.assign(sizes.begin(), sizes.end());
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.getScopeDescriptors();
  auto order =
      buildFirstTemporalLoopOrder(descriptors.front().iterationExtents, sizes,
                                  descriptors.front().precedence);
  EXPECT_TRUE(mlir::succeeded(order));
  if (mlir::succeeded(order))
    choice.scopes.front().loopOrder = std::move(*order);
  EXPECT_TRUE(domain.contains(choice));
  return choice;
}

template <typename Op> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](Op) { ++count; });
  return count;
}

TEST(TemporalTilingTest, AlignedAndRaggedOrdinaryLoopsHaveOnlyOneTail) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream << "        %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
              "        %value = linalg.generic {\n"
              "            indexing_maps = ["
              "affine_map<(b, m, n) -> (b, m, n)>,\n"
              "                             "
              "affine_map<(b, m, n) -> (b, m, n)>],\n"
              "            iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
              "            ins(%arg : tensor<2x"
           << extent
           << "x128xf16>)\n"
              "            outs(%empty : tensor<2x"
           << extent
           << "x128xf16>) {\n"
              "          ^bb0(%element: f16, %old: f16):\n"
              "            %doubled = arith.addf %element, %element : f16\n"
              "            linalg.yield %doubled : f16\n"
              "        } -> tensor<2x"
           << extent << "x128xf16>";
    std::string type = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    auto module = parseModule(*context, stream.str(), type, type);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    ASSERT_TRUE(region);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->tiledTraversals, 1u);
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, extent == 1024 ? 0u : 1u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<mlir::linalg::GenericOp>(module->getOperation()),
              extent == 1024 ? 1u : 2u);

    std::set<int64_t> observedM;
    module->walk([&](mlir::linalg::GenericOp generic) {
      auto type =
          mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
      EXPECT_TRUE(type.hasStaticShape());
      observedM.insert(type.getShape()[1]);
    });
    std::set<int64_t> expected{128};
    if (extent != 1024)
      expected.insert(extent % 128);
    EXPECT_EQ(observedM, expected);
    module->walk([&](mlir::tensor::ExtractSliceOp slice) {
      EXPECT_TRUE(slice.getType().hasStaticShape());
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
  }
}

TEST(TemporalTilingTest, SlicedUnitCollapseReadsOnlyTheSelectedInputTile) {
  for (int64_t extent : {1024, 1025, 1031, 4096, 4097}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    std::string inputType =
        "tensor<2x1x" + std::to_string(extent) + "x128xf16>";
    std::string resultType = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream << "%collapsed = tensor.collapse_shape %arg [[0], [1, 2], [3]] : "
           << inputType << " into " << resultType << "\n"
           << "%empty = tensor.empty() : " << resultType << "\n"
           << "%value = linalg.generic {indexing_maps = ["
              "affine_map<(b, m, n) -> (b, m, n)>, "
              "affine_map<(b, m, n) -> (b, m, n)>], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]} "
              "ins(%collapsed : "
           << resultType << ") outs(%empty : " << resultType << ") {\n"
           << "^bb0(%element: f16, %old: f16):\n"
              "  %doubled = arith.addf %element, %element : f16\n"
              "  linalg.yield %doubled : f16\n"
              "} -> "
           << resultType;
    auto module = parseModule(*context, stream.str(), inputType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    mlir::Value input = region.getBody().front().getArgument(0);
    auto domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, extent % 128 == 0 ? 0u : 1u);
    EXPECT_EQ(countOps<mlir::tensor::CollapseShapeOp>(module->getOperation()),
              0u);
    unsigned reads = 0;
    module->walk([&](mlir::linalg::GenericOp generic) {
      auto slice = generic.getDpsInputs()
                       .front()
                       .getDefiningOp<mlir::tensor::ExtractSliceOp>();
      ASSERT_TRUE(slice);
      EXPECT_EQ(slice.getSource(), input);
      EXPECT_EQ(slice.getStaticStrides(),
                (llvm::ArrayRef<int64_t>{1, 1, 1, 1}));
      auto sizes = slice.getStaticSizes();
      ASSERT_EQ(sizes.size(), 4u);
      EXPECT_EQ(sizes[0], 2);
      EXPECT_EQ(sizes[1], 1);
      EXPECT_EQ(sizes[3], 128);
      auto offsets = slice.getStaticOffsets();
      EXPECT_EQ(offsets[0], 0);
      EXPECT_EQ(offsets[1], 0);
      EXPECT_EQ(offsets[3], 0);
      if (sizes[2] == 128) {
        auto loop = slice->getParentOfType<mlir::scf::ForOp>();
        ASSERT_TRUE(loop);
        EXPECT_EQ(slice.getOffsets().front(), loop.getInductionVar());
      } else {
        EXPECT_EQ(sizes[2], extent % 128);
        EXPECT_EQ(offsets[2], extent - extent % 128);
      }
      ++reads;
    });
    EXPECT_EQ(reads, extent % 128 == 0 ? 1u : 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
    auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    auto movement = materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    unsigned loads = 0;
    module->walk([&](StorageLoadOp load) {
      auto source = mlir::cast<mlir::MemRefType>(load->getOperand(0).getType());
      auto destination =
          mlir::cast<mlir::MemRefType>(load->getOperand(1).getType());
      EXPECT_EQ(source.getShape(), destination.getShape());
      EXPECT_EQ(source.getRank(), 3);
      EXPECT_LE(source.getShape()[1], 128);
      ++loads;
    });
    EXPECT_EQ(loads, reads);
    EXPECT_EQ(movement.statistics.streamedOutputCarriers, 1u);
    EXPECT_EQ(countOps<StorageStoreOp>(module->getOperation()), reads);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    module->walk([&](mlir::memref::AllocOp allocation) {
      if (isWaferSPMMemRefType(allocation.getType())) {
        EXPECT_LE(allocation.getType().getShape()[1], 128);
      }
    });
    std::string detail;
    auto standalone =
        createStandaloneTileModules(std::move(module), &detail, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
    ASSERT_EQ(standalone->size(), 1u);
    auto &tile = standalone->front();
    TileRegionToInstrLoweringSession session(*context);
    llvm::SmallVector<TileRegionOp, 2> tileRegions;
    tile.module->walk([&](TileRegionOp op) { tileRegions.push_back(op); });
    for (TileRegionOp op : tileRegions)
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(op, session)));
    ASSERT_TRUE(mlir::succeeded(
        convertBufferizationCopiesToInstr(*tile.module, session)));
    ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
    TileMemoryPlanningFailure memoryFailure;
    auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
    ASSERT_TRUE(mlir::succeeded(planned));
    EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
  }
}

TEST(TemporalTilingTest, TwoRaggedAxesProduceAtMostFourStaticBodies) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module =
      parseModule(*context,
                  R"mlir(
        %empty = tensor.empty() : tensor<2x1025x1031xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x1031xf16>)
            outs(%empty : tensor<2x1025x1031xf16>) {
          ^bb0(%element: f16, %old: f16):
            %doubled = arith.addf %element, %element : f16
            linalg.yield %doubled : f16
        } -> tensor<2x1025x1031xf16>)mlir",
                  "tensor<2x1025x1031xf16>", "tensor<2x1025x1031xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->loops, 2u);
  EXPECT_EQ(tiled->specializedTails, 2u);
  EXPECT_EQ(countOps<mlir::linalg::GenericOp>(module->getOperation()), 4u);
  EXPECT_LE(countOps<mlir::scf::ForOp>(module->getOperation()), 3u);
  module->walk([&](mlir::linalg::GenericOp generic) {
    EXPECT_TRUE(
        mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType())
            .hasStaticShape());
  });
}

TEST(TemporalTilingTest, ExactSingleUseChainFusesWithoutFullShardClone) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %first_empty = tensor.empty() : tensor<2x1025x128xf16>
        %first = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%first_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %twice = arith.addf %element, %element : f16
            linalg.yield %twice : f16
        } -> tensor<2x1025x128xf16>
        %second_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%first : tensor<2x1025x128xf16>)
            outs(%second_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %doubled = arith.addf %element, %element : f16
            linalg.yield %doubled : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->fusedProducers, 1u);
  EXPECT_EQ(countOps<mlir::linalg::GenericOp>(module->getOperation()), 4u);
  module->walk([&](mlir::linalg::GenericOp generic) {
    auto type =
        mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
    EXPECT_NE(type.getShape()[1], 1025);
  });
}

TEST(TemporalTilingTest,
     IndependentChoiceKeepsProducerAndConsumerAsSeparateTraversals) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025x128xf16>)
            outs(%consumer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.mulf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalSuccessor first = domain.domain->getFirstIndependentChoice();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  TemporalChoice choice = *first.getChoice();
  ASSERT_EQ(choice.kind, TemporalTraversalKind::Independent);
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent);
  ASSERT_EQ(choice.scopes.size(), 2u);
  ASSERT_EQ(descriptors.size(), 2u);
  for (auto [scope, descriptor] : llvm::zip_equal(choice.scopes, descriptors)) {
    scope.iteratorTileSizes = {2, 128, 128};
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    ASSERT_TRUE(mlir::succeeded(order));
    scope.loopOrder = std::move(*order);
  }
  ASSERT_TRUE(domain.domain->contains(choice));
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  mlir::FailureOr<TemporalTilingStatistics> tiled =
      applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->tiledTraversals, 2u);
  EXPECT_EQ(tiled->fusedProducers, 0u);
  EXPECT_EQ(tiled->loops, 2u);
  EXPECT_EQ(tiled->specializedTails, 2u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()), 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalTilingTest,
     ExactSliceAndUnitReshapeChainFusesProducerIntoConsumer) {
  for (int64_t sourceExtent : {1024, 1025, 1031}) {
    SCOPED_TRACE(sourceExtent);
    const int64_t offset = sourceExtent == 1024 ? 0 : sourceExtent - 1025;
    const int64_t resultExtent = sourceExtent == 1024 ? 1024 : 1025;
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %producer_empty = tensor.empty() : tensor<2x"
        << sourceExtent
        << "x128xf16>\n"
           "        %producer = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%arg : tensor<2x"
        << sourceExtent
        << "x128xf16>)\n"
           "            outs(%producer_empty : tensor<2x"
        << sourceExtent
        << "x128xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << sourceExtent
        << "x128xf16>\n"
           "        %slice = tensor.extract_slice %producer[0, "
        << offset << ", 0] [2, " << resultExtent
        << ", 128] [1, 1, 1]\n"
           "            : tensor<2x"
        << sourceExtent << "x128xf16> to tensor<2x" << resultExtent
        << "x128xf16>\n"
           "        %expanded = tensor.expand_shape %slice [[0], [1], [2, 3]]\n"
           "            output_shape [2, "
        << resultExtent << ", 1, 128] : tensor<2x" << resultExtent
        << "x128xf16> into tensor<2x" << resultExtent
        << "x1x128xf16>\n"
           "        %collapsed = tensor.collapse_shape %expanded [[0], [1], "
           "[2, 3]]\n"
           "            : tensor<2x"
        << resultExtent << "x1x128xf16> into tensor<2x" << resultExtent
        << "x128xf16>\n"
           "        %consumer_empty = tensor.empty() : tensor<2x"
        << resultExtent
        << "x128xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%collapsed : tensor<2x"
        << resultExtent
        << "x128xf16>)\n"
           "            outs(%consumer_empty : tensor<2x"
        << resultExtent
        << "x128xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << resultExtent << "x128xf16>";
    std::string sourceType =
        "tensor<2x" + std::to_string(sourceExtent) + "x128xf16>";
    std::string resultType =
        "tensor<2x" + std::to_string(resultExtent) + "x128xf16>";
    auto module = parseModule(*context, stream.str(), sourceType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    llvm::SmallVector<mlir::linalg::GenericOp, 2> sourceOperations;
    region.walk([&](mlir::linalg::GenericOp operation) {
      sourceOperations.push_back(operation);
    });
    ASSERT_EQ(sourceOperations.size(), 2u);
    TemporalFusionPathResult path = queryTemporalProducerFusionPath(
        mlir::cast<mlir::OpResult>(sourceOperations.front().getResult(0)));
    ASSERT_EQ(path.kind, TemporalFusionQueryKind::ExactDerived) << path.detail;
    ASSERT_TRUE(path.viewTransparent);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->viewTransparentProducers, 1u);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(countOps<mlir::tensor::ExpandShapeOp>(module->getOperation()),
              0u);
    EXPECT_EQ(countOps<mlir::tensor::CollapseShapeOp>(module->getOperation()),
              0u);
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto type =
          mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
      EXPECT_TRUE(type.hasStaticShape());
      EXPECT_NE(type.getShape()[1], sourceExtent);
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (sourceExtent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      EXPECT_EQ(layout.statistics.bufferizationInvocations, 1u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
      unsigned fullSourceAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        auto type = allocation.getType();
        if (type.getShape() == llvm::ArrayRef<int64_t>({2, 1031, 128}))
          ++fullSourceAllocations;
      });
      // The input remains a boundary view and the fused producer contributes
      // no full-shape destination root.
      EXPECT_EQ(fullSourceAllocations, 0u);
      EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
    }
  }
}

TEST(TemporalTilingTest,
     GeneralFlattenUsesFullReshapeGroupAndTilesTheRemainingAxisExactly) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    const int64_t flattened = 2 * extent;
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %producer_empty = tensor.empty() : tensor<2x" << extent
        << "x129xf16>\n"
           "        %producer = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%arg : tensor<2x"
        << extent
        << "x129xf16>)\n"
           "            outs(%producer_empty : tensor<2x"
        << extent
        << "x129xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << extent
        << "x129xf16>\n"
           "        %flat = tensor.collapse_shape %producer [[0, 1], [2]] : "
           "tensor<2x"
        << extent << "x129xf16> into tensor<" << flattened
        << "x129xf16>\n"
           "        %consumer_empty = tensor.empty() : tensor<"
        << flattened
        << "x129xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(m, n) -> (m, n)>,\n"
           "                             affine_map<(m, n) -> (m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\"]}\n"
           "            ins(%flat : tensor<"
        << flattened
        << "x129xf16>)\n"
           "            outs(%consumer_empty : tensor<"
        << flattened
        << "x129xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.mulf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<"
        << flattened << "x129xf16>";
    const std::string sourceType =
        "tensor<2x" + std::to_string(extent) + "x129xf16>";
    const std::string resultType =
        "tensor<" + std::to_string(flattened) + "x129xf16>";
    auto module = parseModule(*context, stream.str(), sourceType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    ASSERT_EQ(
        domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
            .size(),
        2u);
    llvm::ArrayRef<TemporalScopeDescriptor> jointDescriptors =
        domain.domain->getScopeDescriptors();
    const TemporalScopeDescriptor &descriptor = jointDescriptors.front();
    ASSERT_EQ(descriptor.iteratorCapabilities.size(), 2u);
    EXPECT_EQ(descriptor.iteratorCapabilities[0],
              IteratorTilingCapability::Tileable);
    EXPECT_EQ(descriptor.exactReshapeDimensions,
              (llvm::SmallVector<uint32_t, 2>{0}));
    TemporalChoice choice = selectTileSizes(*domain.domain, {flattened, 64});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    mlir::FailureOr<TemporalTilingStatistics> tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->viewTransparentProducers, 1u);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, 1u);
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto type =
          mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
      EXPECT_TRUE(type.hasStaticShape());
      EXPECT_NE(type.getShape().back(), 129);
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullProducerAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 1031, 129}))
          ++fullProducerAllocations;
      });
      EXPECT_EQ(fullProducerAllocations, 0u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest,
     GeneralUnflattenUsesFullReshapeGroupAndTilesTheRemainingAxisExactly) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %producer_empty = tensor.empty() : tensor<2050x129xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(m, n) -> (m, n)>,
                             affine_map<(m, n) -> (m, n)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%arg : tensor<2050x129xf16>)
            outs(%producer_empty : tensor<2050x129xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2050x129xf16>
        %expanded = tensor.expand_shape %producer [[0, 1], [2]]
            output_shape [2, 1025, 129] :
            tensor<2050x129xf16> into tensor<2x1025x129xf16>
        %consumer_empty = tensor.empty() : tensor<2x1025x129xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%expanded : tensor<2x1025x129xf16>)
            outs(%consumer_empty : tensor<2x1025x129xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.mulf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x129xf16>)mlir",
                            "tensor<2050x129xf16>", "tensor<2x1025x129xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.domain->getScopeDescriptors();
  ASSERT_EQ(descriptors.size(), 1u);
  ASSERT_EQ(descriptors.front().iteratorCapabilities.size(), 3u);
  EXPECT_EQ(descriptors.front().exactReshapeDimensions,
            (llvm::SmallVector<uint32_t, 2>{0, 1}));
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 1025, 64});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  mlir::FailureOr<TemporalTilingStatistics> tiled =
      applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->viewTransparentProducers, 1u);
  EXPECT_EQ(tiled->fusedProducers, 1u);
  EXPECT_EQ(tiled->loops, 1u);
  EXPECT_EQ(tiled->specializedTails, 1u);
  module->walk([&](mlir::linalg::GenericOp operation) {
    auto type =
        mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
    EXPECT_TRUE(type.hasStaticShape());
    EXPECT_NE(type.getShape().back(), 129);
  });
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalTilingTest,
     GeneralFlattenBuildsBoundedExactPiecesForOneMainAndTail) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %flat = tensor.collapse_shape %producer [[0, 1], [2]] :
            tensor<2x1025x128xf16> into tensor<2050x128xf16>
        %consumer_empty = tensor.empty() : tensor<2050x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(m, n) -> (m, n)>,
                             affine_map<(m, n) -> (m, n)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%flat : tensor<2050x128xf16>)
            outs(%consumer_empty : tensor<2050x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.mulf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2050x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2050x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  mlir::linalg::GenericOp producer;
  region.walk([&](mlir::linalg::GenericOp operation) {
    if (!producer)
      producer = operation;
  });
  ASSERT_TRUE(producer);
  producer->setLoc(
      mlir::NameLoc::get(mlir::StringAttr::get(context.get(), "pieces")));
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalChoice choice = selectTileSizes(*domain.domain, {1500, 128});
  ASSERT_TRUE(domain.domain->contains(choice));
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  mlir::FailureOr<TemporalTilingStatistics> tiled =
      applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->viewTransparentProducers, 1u);
  EXPECT_EQ(tiled->fusedProducers, 1u);
  EXPECT_EQ(tiled->loops, 1u);
  EXPECT_EQ(tiled->specializedTails, 1u);
  unsigned producerPieces = 0;
  module->walk([&](mlir::linalg::GenericOp operation) {
    auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
    if (name && name.getName().strref() == "pieces")
      ++producerPieces;
  });
  // [0,1500) crosses the batch boundary and is exactly two producer
  // rectangles; [1500,2050) is one tail rectangle.
  EXPECT_EQ(producerPieces, 3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalTilingTest, RankFiveViewChainUsesTheSameExactFusionPath) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module =
      parseModule(*context,
                  R"mlir(
        %producer_empty = tensor.empty() : tensor<2x4x8x1031x16xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(a, b, c, m, n) -> (a, b, c, m, n)>,
                             affine_map<(a, b, c, m, n) -> (a, b, c, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel",
                              "parallel", "parallel"]}
            ins(%arg : tensor<2x4x8x1031x16xf16>)
            outs(%producer_empty : tensor<2x4x8x1031x16xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x4x8x1031x16xf16>
        %slice = tensor.extract_slice %producer[0, 0, 0, 6, 0]
            [2, 4, 8, 1025, 16] [1, 1, 1, 1, 1]
            : tensor<2x4x8x1031x16xf16> to tensor<2x4x8x1025x16xf16>
        %expanded = tensor.expand_shape %slice [[0], [1], [2], [3], [4, 5]]
            output_shape [2, 4, 8, 1025, 1, 16]
            : tensor<2x4x8x1025x16xf16> into tensor<2x4x8x1025x1x16xf16>
        %collapsed = tensor.collapse_shape %expanded
            [[0], [1], [2], [3], [4, 5]]
            : tensor<2x4x8x1025x1x16xf16> into tensor<2x4x8x1025x16xf16>
        %consumer_empty = tensor.empty() : tensor<2x4x8x1025x16xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(a, b, c, m, n) -> (a, b, c, m, n)>,
                             affine_map<(a, b, c, m, n) -> (a, b, c, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel",
                              "parallel", "parallel"]}
            ins(%collapsed : tensor<2x4x8x1025x16xf16>)
            outs(%consumer_empty : tensor<2x4x8x1025x16xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x4x8x1025x16xf16>)mlir",
                  "tensor<2x4x8x1031x16xf16>", "tensor<2x4x8x1025x16xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    operations.push_back(operation);
  });
  ASSERT_EQ(operations.size(), 2u);
  TemporalFusionPathResult path = queryTemporalProducerFusionPath(
      mlir::cast<mlir::OpResult>(operations.front().getResult(0)));
  ASSERT_TRUE(path.isExact()) << path.detail;
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 4, 8, 128, 16});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->viewTransparentProducers, 1u);
  module->walk([&](mlir::linalg::GenericOp operation) {
    auto type =
        mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
    EXPECT_NE(type.getShape()[3], 1031);
  });
}

TEST(TemporalTilingTest, TensorPadUsesItsPinnedTilingInterfaceInConsumerLoop) {
  for (int64_t outputExtent : {1024, 1025, 1031}) {
    SCOPED_TRACE(outputExtent);
    const int64_t inputExtent = outputExtent - 1;
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %zero = arith.constant 0.0 : f16\n"
           "        %padded = tensor.pad %arg low[0, 1, 0] high[0, 0, 0] {\n"
           "          ^bb0(%b: index, %m: index, %n: index):\n"
           "            tensor.yield %zero : f16\n"
           "        } : tensor<2x"
        << inputExtent << "x128xf16> to tensor<2x" << outputExtent
        << "x128xf16>\n"
           "        %empty = tensor.empty() : tensor<2x"
        << outputExtent
        << "x128xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%padded : tensor<2x"
        << outputExtent
        << "x128xf16>)\n"
           "            outs(%empty : tensor<2x"
        << outputExtent
        << "x128xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << outputExtent << "x128xf16>";
    std::string inputType =
        "tensor<2x" + std::to_string(inputExtent) + "x128xf16>";
    std::string resultType =
        "tensor<2x" + std::to_string(outputExtent) + "x128xf16>";
    auto module = parseModule(*context, stream.str(), inputType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    ASSERT_EQ(
        domain.domain->getScopeDescriptors().front().iteratorCapabilities[1],
        IteratorTilingCapability::FullExtentOnly);
    TemporalChoice choice =
        selectTileSizes(*domain.domain, {2, outputExtent, 64});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->viewTransparentProducers, 0u);
    EXPECT_EQ(tiled->decomposedPads, 1u);
    EXPECT_EQ(tiled->decomposedConstantGenerates, 1u);
    EXPECT_EQ(countOps<mlir::tensor::PadOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<mlir::tensor::GenerateOp>(module->getOperation()), 0u);
    module->walk([&](mlir::linalg::FillOp fill) {
      auto type =
          mlir::cast<mlir::RankedTensorType>(fill.getResult(0).getType());
      EXPECT_TRUE(type.hasStaticShape());
      EXPECT_EQ(type.getShape()[1], outputExtent);
      EXPECT_EQ(type.getShape()[2], 64);
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (outputExtent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullOutputAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 1031, 128}))
          ++fullOutputAllocations;
      });
      EXPECT_LE(fullOutputAllocations, 1u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest,
     UnpackElementwisePackChainTilesAsOneConsumerTraversal) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %unpacked_empty = tensor.empty() : tensor<2x" << extent
        << "x128xf16>\n"
           "        %unpacked = tensor.unpack %arg inner_dims_pos = [2] "
           "inner_tiles = [16] into %unpacked_empty : tensor<2x"
        << extent << "x8x16xf16> -> tensor<2x" << extent
        << "x128xf16>\n"
           "        %mapped_empty = tensor.empty() : tensor<2x"
        << extent
        << "x128xf16>\n"
           "        %mapped = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%unpacked : tensor<2x"
        << extent
        << "x128xf16>)\n"
           "            outs(%mapped_empty : tensor<2x"
        << extent
        << "x128xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << extent
        << "x128xf16>\n"
           "        %packed_empty = tensor.empty() : tensor<2x"
        << extent
        << "x8x16xf16>\n"
           "        %value = tensor.pack %mapped inner_dims_pos = [2] "
           "inner_tiles = [16] into %packed_empty : tensor<2x"
        << extent << "x128xf16> -> tensor<2x" << extent << "x8x16xf16>";
    std::string type = "tensor<2x" + std::to_string(extent) + "x8x16xf16>";
    auto module = parseModule(*context, stream.str(), type, type);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 8});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->fusedProducers, 2u);
    EXPECT_EQ(tiled->viewTransparentProducers, 0u);
    EXPECT_EQ(countOps<mlir::tensor::PackOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<mlir::tensor::UnPackOp>(module->getOperation()), 0u);
    unsigned localReshapes = 0;
    module->walk([&](mlir::tensor::ExpandShapeOp expand) {
      ++localReshapes;
      EXPECT_NE(expand.getResultType().getShape()[1], extent);
    });
    module->walk([&](mlir::tensor::CollapseShapeOp collapse) {
      ++localReshapes;
      EXPECT_NE(collapse.getResultType().getShape()[1], extent);
    });
    EXPECT_GT(localReshapes, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullUnpackedAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 1031, 128}))
          ++fullUnpackedAllocations;
      });
      EXPECT_EQ(fullUnpackedAllocations, 0u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest, ConcatInsertChainBuildsOnlyRequestedConsumerTile) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %left = tensor.extract_slice %arg[0, 0, 0] [2, " << extent
        << ", 64] [1, 1, 1] : tensor<2x" << extent << "x128xf16> to tensor<2x"
        << extent
        << "x64xf16>\n"
           "        %right = tensor.extract_slice %arg[0, 0, 64] [2, "
        << extent << ", 64] [1, 1, 1] : tensor<2x" << extent
        << "x128xf16> to tensor<2x" << extent
        << "x64xf16>\n"
           "        %left_empty = tensor.empty() : tensor<2x"
        << extent
        << "x64xf16>\n"
           "        %left_computed = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%left : tensor<2x"
        << extent
        << "x64xf16>)\n"
           "            outs(%left_empty : tensor<2x"
        << extent
        << "x64xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << extent
        << "x64xf16>\n"
           "        %assembly_empty = tensor.empty() : tensor<2x"
        << extent
        << "x128xf16>\n"
           "        %with_left = tensor.insert_slice %left_computed into "
           "%assembly_empty[0, 0, 0] [2, "
        << extent << ", 64] [1, 1, 1] : tensor<2x" << extent
        << "x64xf16> into tensor<2x" << extent
        << "x128xf16>\n"
           "        %joined = tensor.insert_slice %right into "
           "%with_left[0, 0, 64] [2, "
        << extent << ", 64] [1, 1, 1] : tensor<2x" << extent
        << "x64xf16> into tensor<2x" << extent
        << "x128xf16>\n"
           "        %result_empty = tensor.empty() : tensor<2x"
        << extent
        << "x128xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%joined : tensor<2x"
        << extent
        << "x128xf16>)\n"
           "            outs(%result_empty : tensor<2x"
        << extent
        << "x128xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << extent << "x128xf16>";
    std::string type = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    auto module = parseModule(*context, stream.str(), type, type);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    mlir::linalg::GenericOp consumer;
    region.walk(
        [&](mlir::linalg::GenericOp operation) { consumer = operation; });
    ASSERT_TRUE(consumer);
    TemporalConcatQueryResult concat =
        queryTemporalConcatAssembly(consumer->getOpOperand(0));
    ASSERT_TRUE(concat.isExact()) << concat.detail;
    ASSERT_EQ(concat.segments.size(), 2u);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    const uint64_t variants = extent == 1024 ? 1 : 2;
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->tileLocalAssemblies, variants);
    EXPECT_EQ(tiled->assembledSegments, variants * 2);
    EXPECT_EQ(countOps<mlir::scf::IfOp>(module->getOperation()), 0u);
    module->walk([&](mlir::tensor::InsertSliceOp insert) {
      auto destinationType =
          mlir::cast<mlir::RankedTensorType>(insert.getDest().getType());
      auto sourceType = insert.getSourceType();
      const bool fullConcatSegment = destinationType.getShape()[1] == extent &&
                                     sourceType.getShape().back() == 64;
      EXPECT_FALSE(fullConcatSegment);
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 1031, 128}))
          ++fullAllocations;
      });
      EXPECT_LE(fullAllocations, 1u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest,
     UnalignedStaticConcatBoundaryKeepsEveryExecutableTileStatic) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(
      *context,
      R"mlir(
        %past = tensor.extract_slice %arg[0, 0, 0, 0]
            [1, 4, 1023, 128] [1, 1, 1, 1]
            : tensor<1x4x1024x128xf16> to tensor<1x4x1023x128xf16>
        %token = tensor.extract_slice %arg[0, 0, 1023, 0]
            [1, 4, 1, 128] [1, 1, 1, 1]
            : tensor<1x4x1024x128xf16> to tensor<1x4x1x128xf16>
        %assembly_empty = tensor.empty() : tensor<1x4x1024x128xf16>
        %with_past = tensor.insert_slice %past into %assembly_empty
            [0, 0, 0, 0] [1, 4, 1023, 128] [1, 1, 1, 1]
            : tensor<1x4x1023x128xf16> into tensor<1x4x1024x128xf16>
        %joined = tensor.insert_slice %token into %with_past
            [0, 0, 1023, 0] [1, 4, 1, 128] [1, 1, 1, 1]
            : tensor<1x4x1x128xf16> into tensor<1x4x1024x128xf16>
        %result_empty = tensor.empty() : tensor<1x4x1024x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, h, s, d) -> (b, h, s, d)>,
                             affine_map<(b, h, s, d) -> (b, h, s, d)>],
            iterator_types = ["parallel", "parallel", "parallel",
                              "parallel"]}
            ins(%joined : tensor<1x4x1024x128xf16>)
            outs(%result_empty : tensor<1x4x1024x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<1x4x1024x128xf16>)mlir",
      "tensor<1x4x1024x128xf16>", "tensor<1x4x1024x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  mlir::linalg::GenericOp consumer;
  region.walk(
      [&](mlir::linalg::GenericOp operation) { consumer = operation; });
  ASSERT_TRUE(consumer);
  TemporalConcatQueryResult concat =
      queryTemporalConcatAssembly(consumer->getOpOperand(0));
  ASSERT_TRUE(concat.isExact()) << concat.detail;
  ASSERT_EQ(concat.segments.size(), 2u);

  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  TemporalChoice choice =
      selectTileSizes(*domain.domain, {1, 2, 512, 128});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->tileLocalAssemblies, 2u);
  EXPECT_EQ(tiled->assembledSegments, 3u);
  EXPECT_EQ(tiled->specializedConcatBoundaries, 1u);
  EXPECT_EQ(countOps<mlir::scf::IfOp>(module->getOperation()), 0u);

  std::set<int64_t> insertedSequenceSizes;
  std::set<int64_t> extractedSequenceSizes;
  module->walk([&](mlir::tensor::InsertSliceOp insert) {
    auto sourceType = insert.getSourceType();
    auto destinationType = mlir::dyn_cast<mlir::RankedTensorType>(
        insert.getDest().getType());
    if (sourceType.getRank() == 4 && destinationType &&
        destinationType.getDimSize(2) == 512)
      insertedSequenceSizes.insert(sourceType.getDimSize(2));
  });
  module->walk([&](mlir::tensor::ExtractSliceOp extract) {
    auto type = extract.getType();
    if (type.getRank() == 4)
      extractedSequenceSizes.insert(type.getDimSize(2));
  });
  EXPECT_EQ(insertedSequenceSizes, (std::set<int64_t>{1, 511}));
  EXPECT_EQ(extractedSequenceSizes.count(512), 1u);

  module->walk([&](mlir::Operation *operation) {
    for (mlir::Type type : operation->getOperandTypes()) {
      if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type); shaped &&
          shaped.hasRank()) {
        EXPECT_TRUE(shaped.hasStaticShape())
            << operation->getName().getStringRef().str();
      }
    }
    for (mlir::Type type : operation->getResultTypes()) {
      if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type); shaped &&
          shaped.hasRank()) {
        EXPECT_TRUE(shaped.hasStaticShape())
            << operation->getName().getStringRef().str();
      }
    }
    for (mlir::Region &nested : operation->getRegions())
      for (mlir::Block &block : nested)
        for (mlir::BlockArgument argument : block.getArguments()) {
          if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(argument.getType());
              shaped && shaped.hasRank()) {
            EXPECT_TRUE(shaped.hasStaticShape())
                << operation->getName().getStringRef().str();
          }
        }
  });
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  LayoutOptimizationResult layout =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  EXPECT_TRUE(layout.succeeded()) << layout.detail;
}

TEST(TemporalTilingTest,
     MultiUseProducerIsMaterializedOncePerJointMainOrTailBody) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %left_empty = tensor.empty() : tensor<2x1025x128xf16>
        %left = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025x128xf16>)
            outs(%left_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %right_empty = tensor.empty() : tensor<2x1025x128xf16>
        %right = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025x128xf16>)
            outs(%right_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.mulf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %sum_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%left, %right : tensor<2x1025x128xf16>,
                  tensor<2x1025x128xf16>)
            outs(%sum_empty : tensor<2x1025x128xf16>) {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %next = arith.addf %lhs, %rhs : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  llvm::SmallVector<mlir::linalg::GenericOp, 4> beforeOperations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    beforeOperations.push_back(operation);
  });
  ASSERT_EQ(beforeOperations.size(), 4u);
  beforeOperations.front()->setLoc(
      mlir::NameLoc::get(mlir::StringAttr::get(context.get(), "shared")));

  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  ASSERT_EQ(domain.domain->getJointProducerGroups().size(), 1u);
  TemporalSuccessor first = domain.domain->getFirstChoice();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  TemporalChoice choice = *first.getChoice();
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.domain->getScopeDescriptors();
  for (auto [scope, descriptor] : llvm::zip_equal(choice.scopes, descriptors)) {
    scope.iteratorTileSizes[1] = 128;
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    ASSERT_TRUE(mlir::succeeded(order));
    scope.loopOrder = std::move(*order);
  }
  ASSERT_TRUE(domain.domain->contains(choice));
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->tiledTraversals, 1u);
  EXPECT_EQ(tiled->fusedProducers, 3u);
  EXPECT_EQ(tiled->loops, 1u);
  EXPECT_EQ(tiled->specializedTails, 1u);
  unsigned sharedOccurrences = 0;
  module->walk([&](mlir::linalg::GenericOp operation) {
    auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
    if (name && name.getName().strref() == "shared")
      ++sharedOccurrences;
    auto type =
        mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
    EXPECT_TRUE(type.hasStaticShape());
    EXPECT_NE(type.getShape()[1], 1025);
  });
  EXPECT_EQ(sharedOccurrences, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalTilingTest,
     TwoAndFifteenObservableRootsShareOneJointLoopAndProducerTile) {
  const std::pair<int64_t, unsigned> cases[]{{1024, 2}, {1025, 15}};
  for (auto [extent, useCount] : cases) {
    SCOPED_TRACE("extent=" + std::to_string(extent) +
                 ", roots=" + std::to_string(useCount));
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    const std::string type = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module {\n"
              "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
              "    func.func @entry(%input: "
           << type << ") -> (";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << type;
    }
    stream << ") {\n      %results:" << useCount
           << " = wafer.tile.region(%input : " << type << ") -> (";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << type;
    }
    stream << ") {\n      ^bb0(%arg: " << type << "):\n"
           << "        %producer_empty = tensor.empty() : " << type << "\n"
           << R"mlir(        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : )mlir"
           << type << ") outs(%producer_empty : " << type << R"mlir() {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> )mlir"
           << type << "\n";
    for (unsigned index = 0; index < useCount; ++index) {
      stream << "        %empty" << index << " = tensor.empty() : " << type
             << "\n";
      stream << "        %consumer" << index << R"mlir( = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : )mlir"
             << type << ") outs(%empty" << index << " : " << type << R"mlir() {
          ^bb0(%element: f16, %old: f16):
            %constant = arith.constant )mlir"
             << index + 1 << ".0 : f16\n"
             << R"mlir(            %next = arith.addf %element, %constant : f16
            linalg.yield %next : f16
        } -> )mlir"
             << type << "\n";
    }
    stream << "        wafer.tile.yield ";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << "%consumer" << index;
    }
    stream << " : ";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << type;
    }
    stream << "\n      }\n      return ";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << "%results#" << index;
    }
    stream << " : ";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << type;
    }
    stream << "\n    }\n  }\n}\n";
    stream.flush();

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(
            source, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(module) << source;
    TileRegionOp region = findRegion(*module);
    llvm::SmallVector<mlir::linalg::GenericOp, 16> sourceOperations;
    region.walk([&](mlir::linalg::GenericOp operation) {
      sourceOperations.push_back(operation);
    });
    ASSERT_EQ(sourceOperations.size(), useCount + 1u);
    sourceOperations.front()->setLoc(
        mlir::NameLoc::get(mlir::StringAttr::get(context.get(), "shared")));

    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), useCount);
    ASSERT_EQ(domain.domain->getJointProducerGroups().size(), 1u);
    ASSERT_EQ(
        domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
            .size(),
        useCount + 1u);
    TemporalChoice choice = *domain.domain->getFirstChoice().getChoice();
    llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
        domain.domain->getScopeDescriptors();
    if (useCount == 2) {
      TemporalChoice incompatible = choice;
      incompatible.scopes[0].iteratorTileSizes = {2, 128, 128};
      incompatible.scopes[1].iteratorTileSizes = {2, 256, 128};
      for (auto [scope, descriptor] :
           llvm::zip_equal(incompatible.scopes, descriptors)) {
        auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                                 scope.iteratorTileSizes,
                                                 descriptor.precedence);
        ASSERT_TRUE(mlir::succeeded(order));
        scope.loopOrder = std::move(*order);
      }
      EXPECT_FALSE(domain.domain->contains(incompatible));
    }
    for (auto [scope, descriptor] :
         llvm::zip_equal(choice.scopes, descriptors)) {
      scope.iteratorTileSizes = {2, 128, 128};
      auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                               scope.iteratorTileSizes,
                                               descriptor.precedence);
      ASSERT_TRUE(mlir::succeeded(order));
      scope.loopOrder = std::move(*order);
    }
    ASSERT_TRUE(domain.domain->contains(choice));
    StructuredMaterializationRelations relations;
    for (unsigned index = 0; index < region.getNumResults(); ++index)
      relations.structuralOutputs.push_back({index, region.getResult(index)});
    TemporalTilingFailure failure;
    mlir::FailureOr<TemporalTilingStatistics> tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->tiledTraversals, useCount);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, extent == 1025 ? 1u : 0u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()), 1u);
    unsigned sharedOccurrences = 0;
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
      if (name && name.getName().strref() == "shared")
        ++sharedOccurrences;
    });
    EXPECT_EQ(sharedOccurrences, extent == 1025 ? 2u : 1u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1025) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      EXPECT_EQ(layout.statistics.bufferizationInvocations, 1u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest, BroadcastProducerIsOutsideEveryInvariantConsumerLoop) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %producer_empty = tensor.empty() : tensor<2x4x" << extent
        << "xf16>\n"
           "        %producer = linalg.generic {\n"
           "            indexing_maps = [affine_map<(g, b, m) -> (g, b, m)>,\n"
           "                             affine_map<(g, b, m) -> (g, b, m)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%arg : tensor<2x4x"
        << extent << "xf16>) outs(%producer_empty : tensor<2x4x" << extent
        << "xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x4x"
        << extent
        << "xf16>\n"
           "        %consumer_empty = tensor.empty() : tensor<2x4x"
        << extent
        << "x1025xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(g, b, m, n) -> (g, b, "
           "m)>,\n"
           "                             affine_map<(g, b, m, n) -> (g, b, m, "
           "n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\", \"parallel\"]}\n"
           "            ins(%producer : tensor<2x4x"
        << extent << "xf16>) outs(%consumer_empty : tensor<2x4x" << extent
        << "x1025xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            linalg.yield %element : f16\n"
           "        } -> tensor<2x4x"
        << extent << "x1025xf16>";
    const std::string inputType =
        "tensor<2x4x" + std::to_string(extent) + "xf16>";
    const std::string resultType =
        "tensor<2x4x" + std::to_string(extent) + "x1025xf16>";
    auto module = parseModule(*context, stream.str(), inputType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
    region.walk([&](mlir::linalg::GenericOp operation) {
      operations.push_back(operation);
    });
    ASSERT_EQ(operations.size(), 2u);
    operations.front()->setLoc(mlir::NameLoc::get(
        mlir::StringAttr::get(context.get(), "broadcast-invariant")));
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    ASSERT_EQ(domain.domain->getBroadcastFusions().size(), 1u);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 4, 128, 64});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->tiledTraversals, 1u);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->loops, 2u);
    EXPECT_EQ(tiled->specializedTails, extent == 1024 ? 1u : 2u);
    unsigned producerOccurrences = 0;
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
      if (!name || name.getName().strref() != "broadcast-invariant")
        return;
      ++producerOccurrences;
      unsigned loopDepth = 0;
      for (mlir::Operation *parent = operation->getParentOp(); parent;
           parent = parent->getParentOp())
        loopDepth += mlir::isa<mlir::scf::ForOp>(parent);
      EXPECT_LE(loopDepth, 1u);
      auto resultType =
          mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
      EXPECT_NE(resultType.getShape()[2], extent);
    });
    EXPECT_EQ(producerOccurrences, extent == 1024 ? 1u : 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullProducerAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 4, 1031}))
          ++fullProducerAllocations;
      });
      EXPECT_EQ(fullProducerAllocations, 0u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest,
     DisjointAffineWindowFusesOnlyItsActualMainAndTailRectangles) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    const int64_t inputExtent = 3 * extent;
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %producer_empty = tensor.empty() : tensor<2x" << inputExtent
        << "x129xf16>\n"
           "        %producer = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, q, n) -> (b, q, n)>,\n"
           "                             affine_map<(b, q, n) -> (b, q, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%arg : tensor<2x"
        << inputExtent << "x129xf16>) outs(%producer_empty : tensor<2x"
        << inputExtent
        << "x129xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            %next = arith.addf %element, %element : f16\n"
           "            linalg.yield %next : f16\n"
           "        } -> tensor<2x"
        << inputExtent
        << "x129xf16>\n"
           "        %kernel = tensor.empty() : tensor<3xf16>\n"
           "        %consumer_empty = tensor.empty() : tensor<2x"
        << extent
        << "x129xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, k, n) -> "
           "(b, m * 3 + k, n)>,\n"
           "                             affine_map<(b, m, k, n) -> (k)>,\n"
           "                             affine_map<(b, m, k, n) -> (b, m, "
           "n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"reduction\", \"parallel\"]}\n"
           "            ins(%producer, %kernel : tensor<2x"
        << inputExtent
        << "x129xf16>, tensor<3xf16>) outs(%consumer_empty : tensor<2x"
        << extent
        << "x129xf16>) {\n"
           "          ^bb0(%window_input: f16, %weight: f16, %acc: f16):\n"
           "            %product = arith.mulf %window_input, %weight : f16\n"
           "            %sum = arith.addf %product, %acc : f16\n"
           "            linalg.yield %sum : f16\n"
           "        } -> tensor<2x"
        << extent << "x129xf16>";
    const std::string inputType =
        "tensor<2x" + std::to_string(inputExtent) + "x129xf16>";
    const std::string resultType =
        "tensor<2x" + std::to_string(extent) + "x129xf16>";
    auto module = parseModule(*context, stream.str(), inputType, resultType);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
    region.walk([&](mlir::linalg::GenericOp operation) {
      operations.push_back(operation);
    });
    ASSERT_EQ(operations.size(), 2u);
    operations.front()->setLoc(mlir::NameLoc::get(
        mlir::StringAttr::get(context.get(), "disjoint-window")));
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getWindowFusions().size(), 1u);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 3, 64});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->tiledTraversals, 1u);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->loops, 2u);
    EXPECT_EQ(tiled->specializedTails, extent == 1024 ? 1u : 2u);
    unsigned producerOccurrences = 0;
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
      if (!name || name.getName().strref() != "disjoint-window")
        return;
      ++producerOccurrences;
      auto type =
          mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
      EXPECT_LE(type.getShape()[1], 384);
      EXPECT_LE(type.getShape()[2], 64);
      EXPECT_NE(type.getShape()[1], inputExtent);
    });
    EXPECT_EQ(producerOccurrences, extent == 1024 ? 2u : 4u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1031) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullProducerAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, inputExtent, 129}))
          ++fullProducerAllocations;
      });
      EXPECT_EQ(fullProducerAllocations, 0u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest,
     OverlappingWindowRemainsTwoIndependentCurrentTraversals) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1027x129xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, q, n) -> (b, q, n)>,
                             affine_map<(b, q, n) -> (b, q, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1027x129xf16>)
            outs(%producer_empty : tensor<2x1027x129xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1027x129xf16>
        %kernel = tensor.empty() : tensor<3xf16>
        %consumer_empty = tensor.empty() : tensor<2x1025x129xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, k, n) -> (b, m + k, n)>,
                             affine_map<(b, m, k, n) -> (k)>,
                             affine_map<(b, m, k, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "reduction",
                              "parallel"]}
            ins(%producer, %kernel : tensor<2x1027x129xf16>, tensor<3xf16>)
            outs(%consumer_empty : tensor<2x1025x129xf16>) {
          ^bb0(%window_input: f16, %weight: f16, %acc: f16):
            %product = arith.mulf %window_input, %weight : f16
            %sum = arith.addf %product, %acc : f16
            linalg.yield %sum : f16
        } -> tensor<2x1025x129xf16>)mlir",
                            "tensor<2x1027x129xf16>", "tensor<2x1025x129xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  ASSERT_EQ(domain.domain->getWindowFusions().size(), 1u);
  TemporalChoice rejectedJoint = *domain.domain->getFirstChoice().getChoice();
  ASSERT_EQ(rejectedJoint.scopes.size(), 1u);
  rejectedJoint.scopes.front().iteratorTileSizes = {2, 128, 3, 129};
  rejectedJoint.scopes.front().loopOrder = {1};
  EXPECT_FALSE(domain.domain->contains(rejectedJoint));

  TemporalSuccessor independent = domain.domain->getFirstIndependentChoice();
  ASSERT_EQ(independent.getKind(), TemporalSuccessorKind::Choice);
  TemporalChoice choice = *independent.getChoice();
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent);
  ASSERT_EQ(choice.scopes.size(), 2u);
  ASSERT_EQ(descriptors.size(), 2u);
  for (auto [scope, descriptor] : llvm::zip_equal(choice.scopes, descriptors)) {
    if (descriptor.iterationExtents.size() == 3)
      scope.iteratorTileSizes = {2, 128, 129};
    else
      scope.iteratorTileSizes = {2, 128, 3, 129};
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    ASSERT_TRUE(mlir::succeeded(order));
    scope.loopOrder = std::move(*order);
  }
  ASSERT_TRUE(domain.domain->contains(choice));
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->tiledTraversals, 2u);
  EXPECT_EQ(tiled->fusedProducers, 0u);
  EXPECT_EQ(tiled->loops, 2u);
  EXPECT_EQ(tiled->specializedTails, 2u);
  unsigned fullProducerValues = 0;
  module->walk([&](mlir::tensor::InsertSliceOp insert) {
    auto destinationType =
        mlir::dyn_cast<mlir::RankedTensorType>(insert.getDest().getType());
    if (destinationType &&
        destinationType.getShape() == llvm::ArrayRef<int64_t>({2, 1027, 129}))
      ++fullProducerValues;
  });
  EXPECT_GT(fullProducerValues, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalTilingTest,
     TwoAndFifteenViewUsesShareOneProducerTilePerMainOrTail) {
  const std::pair<int64_t, unsigned> cases[]{{1024, 2}, {1025, 15}};
  for (auto [extent, useCount] : cases) {
    SCOPED_TRACE("extent=" + std::to_string(extent) +
                 ", uses=" + std::to_string(useCount));
    const int64_t sourceExtent = extent + 6;
    const std::string sourceType =
        "tensor<2x" + std::to_string(sourceExtent) + "x128xf16>";
    const std::string viewType =
        "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream
        << "        %producer_empty = tensor.empty() : " << sourceType
        << "\n"
           "        %producer = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,\n"
           "                             affine_map<(b, m, n) -> (b, m, n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"parallel\"]}\n"
           "            ins(%arg : "
        << sourceType << ") outs(%producer_empty : " << sourceType << R"mlir() {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> )mlir"
        << sourceType
        << "\n        %slice = tensor.extract_slice %producer[0, 6, 0] "
           "[2, "
        << extent << ", 128] [1, 1, 1] : " << sourceType << " to " << viewType
        << "\n        %expanded = tensor.expand_shape %slice [[0], [1], "
           "[2, 3]] output_shape [2, "
        << extent << ", 1, 128] : " << viewType << " into tensor<2x" << extent
        << "x1x128xf16>\n        %collapsed = tensor.collapse_shape "
           "%expanded [[0], [1], [2, 3]] : tensor<2x"
        << extent << "x1x128xf16> into " << viewType
        << "\n        %view = tensor.cast %collapsed : " << viewType << " to "
        << viewType
        << "\n        %consumer_empty = tensor.empty() : " << viewType
        << "\n        %value = linalg.generic {\n"
           "            indexing_maps = [";
    for (unsigned index = 0; index < useCount + 1; ++index) {
      if (index)
        stream << ",\n                             ";
      stream << "affine_map<(b, m, n) -> (b, m, n)>";
    }
    stream << "],\n            iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n            ins(";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << "%view";
    }
    stream << " : ";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << viewType;
    }
    stream << ") outs(%consumer_empty : " << viewType << ") {\n          ^bb0(";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index)
        stream << ", ";
      stream << "%input" << index << ": f16";
    }
    stream << ", %old: f16):\n            linalg.yield %input0 : f16\n"
              "        } -> "
           << viewType;
    auto module = parseModule(*context, stream.str(), sourceType, viewType);
    ASSERT_TRUE(module) << body;
    TileRegionOp region = findRegion(*module);
    llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
    region.walk([&](mlir::linalg::GenericOp operation) {
      operations.push_back(operation);
    });
    ASSERT_EQ(operations.size(), 2u);
    operations.front()->setLoc(mlir::NameLoc::get(
        mlir::StringAttr::get(context.get(), "shared-view")));
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getJointProducerGroups().size(), 1u);
    EXPECT_TRUE(
        domain.domain->getJointProducerGroups().front().isViewTransparent());
    EXPECT_EQ(
        domain.domain->getJointProducerGroups().front().consumerOperands.size(),
        useCount);
    TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->tiledTraversals, 1u);
    EXPECT_EQ(tiled->fusedProducers, 1u);
    EXPECT_EQ(tiled->viewTransparentProducers, 1u);
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, extent == 1024 ? 0u : 1u);
    unsigned producerOccurrences = 0;
    module->walk([&](mlir::linalg::GenericOp operation) {
      auto name = mlir::dyn_cast<mlir::NameLoc>(operation.getLoc());
      if (!name || name.getName().strref() != "shared-view")
        return;
      ++producerOccurrences;
      auto type =
          mlir::cast<mlir::RankedTensorType>(operation.getResult(0).getType());
      EXPECT_LE(type.getShape()[1], 128);
      EXPECT_NE(type.getShape()[1], sourceExtent);
    });
    EXPECT_EQ(producerOccurrences, extent == 1024 ? 1u : 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    if (extent == 1025) {
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned fullProducerAllocations = 0;
      module->walk([&](mlir::memref::AllocOp allocation) {
        if (allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, sourceExtent, 128}))
          ++fullProducerAllocations;
      });
      EXPECT_EQ(fullProducerAllocations, 0u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    }
  }
}

TEST(TemporalTilingTest, ExactContractionProducerUsesFullReductionFiber) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %rhs = tensor.empty() : tensor<2x64x128xf16>
        %matmul_empty = tensor.empty() : tensor<2x1025x128xf16>
        %matmul = linalg.batch_matmul
            ins(%arg, %rhs : tensor<2x1025x64xf16>,
                tensor<2x64x128xf16>)
            outs(%matmul_empty : tensor<2x1025x128xf16>)
            -> tensor<2x1025x128xf16>
        %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%matmul : tensor<2x1025x128xf16>)
            outs(%consumer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x64xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->fusedProducers, 1u);
  EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(module->getOperation()), 2u);
  module->walk([&](mlir::linalg::BatchMatmulOp matmul) {
    auto lhsType = mlir::cast<mlir::RankedTensorType>(
        matmul.getInputs().front().getType());
    EXPECT_TRUE(lhsType.hasStaticShape());
    EXPECT_EQ(lhsType.getShape()[2], 64);
    EXPECT_TRUE(lhsType.getShape()[1] == 128 || lhsType.getShape()[1] == 1);
  });
}

TEST(TemporalTilingTest,
     OrdinaryReductionCarriesOneAccumulatorAcrossTwoRaggedAxes) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %init = tensor.empty() : tensor<2x1025xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, k) -> (b, m, k)>,
                             affine_map<(b, m, k) -> (b, m)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%arg : tensor<2x1025x1031xf16>)
            outs(%init : tensor<2x1025xf16>) {
          ^bb0(%element: f16, %accumulator: f16):
            %next = arith.addf %element, %accumulator : f16
            linalg.yield %next : f16
        } -> tensor<2x1025xf16>)mlir",
                            "tensor<2x1025x1031xf16>", "tensor<2x1025xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalChoice choice = selectTileSizes(*domain.domain, {2, 128, 128});
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, choice, relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->loops, 2u);
  EXPECT_EQ(tiled->specializedTails, 2u);
  EXPECT_EQ(countOps<mlir::linalg::GenericOp>(module->getOperation()), 4u);
  module->walk([&](mlir::linalg::GenericOp reduction) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(
        reduction.getDpsInputs().front().getType());
    auto outputType =
        mlir::cast<mlir::RankedTensorType>(reduction.getResult(0).getType());
    EXPECT_TRUE(inputType.hasStaticShape());
    EXPECT_TRUE(outputType.hasStaticShape());
    EXPECT_TRUE(inputType.getShape()[1] == 128 || inputType.getShape()[1] == 1);
    EXPECT_TRUE(inputType.getShape()[2] == 128 || inputType.getShape()[2] == 7);
    EXPECT_EQ(outputType.getShape()[1], inputType.getShape()[1]);
  });
}

TEST(TemporalTilingTest, OnlineK2CarriesThreeStatesThroughMainAndTail) {
  for (int64_t keyValueExtent : {1024, 1025, 1031}) {
    SCOPED_TRACE(keyValueExtent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string body;
    llvm::raw_string_ostream stream(body);
    stream << "        %key = tensor.empty() : tensor<2x4x" << keyValueExtent
           << "x64xf16>\n"
              "        %value_input = tensor.empty() : tensor<2x4x"
           << keyValueExtent
           << "x128xf16>\n"
              "        %scale = arith.constant 1.0 : f32\n"
              "        %accumulator = tensor.empty() : "
              "tensor<2x4x1025x128xf16>\n"
              "        %maximum = tensor.empty() : tensor<2x4x1025xf32>\n"
              "        %sum = tensor.empty() : tensor<2x4x1025xf32>\n"
              "        %value, %next_maximum, %next_sum =\n"
              "            wafer.linalg_ext.online_attention\n"
              "            ins(%arg, %key, %value_input, %scale : "
              "tensor<2x4x1025x64xf16>,\n"
              "                tensor<2x4x"
           << keyValueExtent << "x64xf16>, tensor<2x4x" << keyValueExtent
           << "x128xf16>, f32)\n"
              "            outs(%accumulator, %maximum, %sum : "
              "tensor<2x4x1025x128xf16>,\n"
              "                tensor<2x4x1025xf32>, "
              "tensor<2x4x1025xf32>)\n"
              "            indexing_maps = [\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, m, k1)>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, k2, k1)>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, k2, n)>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> ()>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, m, n)>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, m)>,\n"
              "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, h, m)>]\n"
              "            -> (tensor<2x4x1025x128xf16>, "
              "tensor<2x4x1025xf32>,\n"
              "                tensor<2x4x1025xf32>)";
    auto module = parseModule(*context, stream.str(), "tensor<2x4x1025x64xf16>",
                              "tensor<2x4x1025x128xf16>");
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    TemporalChoice choice =
        selectTileSizes(*domain.domain, {2, 4, 1025, 64, 128, 128});
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
    EXPECT_EQ(tiled->loops, 1u);
    EXPECT_EQ(tiled->specializedTails, keyValueExtent == 1024 ? 0u : 1u);
    mlir::scf::ForOp loop;
    module->walk([&](mlir::scf::ForOp operation) {
      if (!loop)
        loop = operation;
    });
    ASSERT_TRUE(loop);
    EXPECT_EQ(loop.getNumRegionIterArgs(), 3u);
    EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()),
              keyValueExtent == 1024 ? 1u : 2u);
    module->walk([&](LinalgExtOnlineAttentionOp online) {
      auto keyType =
          mlir::cast<mlir::RankedTensorType>(online.getKey().getType());
      EXPECT_TRUE(keyType.hasStaticShape());
      EXPECT_EQ(keyType.getShape()[3], 64);
      EXPECT_LE(keyType.getShape()[2], 128);
    });
    EXPECT_EQ(countOps<LinalgExtAttentionOp>(module->getOperation()), 0u);
  }
}

TEST(TemporalTilingTest, FullExtentChoiceIsByteIdentical) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalSuccessor first = domain.domain->getFirstChoice();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  auto tiled = applyTemporalTiling(*domain.domain, *first.getChoice(),
                                   relations, &failure);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
  EXPECT_EQ(tiled->tiledTraversals, 0u);
  EXPECT_EQ(tiled->loops, 0u);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(TemporalTilingTest, InvalidChoiceFailsBeforeMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parseModule(*context,
                            R"mlir(
        %empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>)mlir",
                            "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  TemporalChoice choice = *domain.domain->getFirstChoice().getChoice();
  choice.scopes.front().iteratorTileSizes[1] = 0;
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  TemporalTilingFailure failure;
  EXPECT_TRUE(mlir::failed(
      applyTemporalTiling(*domain.domain, choice, relations, &failure)));
  EXPECT_EQ(failure.kind, TemporalTilingFailureKind::BrokenContract);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

mlir::OwningOpRef<mlir::ModuleOp>
createOnlineFinalizerModule(mlir::MLIRContext &context, int64_t extent,
                            llvm::StringRef elementType) {
  std::string text = R"mlir(
    %scale = arith.constant 0.0883883461 : f32
    %empty = tensor.empty() : tensor<1x2x4096x128xf16>
    %value = wafer.linalg_ext.attention
      ins(%arg, %arg, %arg, %scale : tensor<1x2x4096x128xf16>, tensor<1x2x4096x128xf16>, tensor<1x2x4096x128xf16>, f32)
      outs(%empty : tensor<1x2x4096x128xf16>) algorithm(<flash_attention>)
      indexing_maps = [
        affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>,
        affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>,
        affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>,
        affine_map<(b, h, m, k1, k2, n) -> ()>,
        affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>]
      -> tensor<1x2x4096x128xf16>)mlir";
  auto substitute = [&](llvm::StringRef from, llvm::StringRef to) {
    size_t offset = 0;
    while ((offset = text.find(from.str(), offset)) != std::string::npos) {
      text.replace(offset, from.size(), to.str());
      offset += to.size();
    }
  };
  substitute("4096", std::to_string(extent));
  substitute("f16", elementType);
  std::string type = "tensor<1x2x" + std::to_string(extent) + "x128x" +
                     elementType.str() + ">";
  auto module = parseModule(context, text, type, type);
  if (!module)
    return {};
  LinalgExtAttentionOp attention;
  module->walk([&](LinalgExtAttentionOp op) { attention = op; });
  mlir::OpBuilder builder(attention);
  llvm::SmallVector<mlir::OpFoldResult, 6> offsets(6, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
  llvm::SmallVector<int64_t, 6> extents{1, 2, extent, 128, extent, 128};
  for (int64_t size : extents)
    sizes.push_back(builder.getIndexAttr(size));
  auto state = materializeOnlineAttentionTile(
      attention, attention.getQuery(), attention.getKey(), attention.getValue(),
      attention.getScale(), {}, offsets, sizes, builder);
  if (mlir::failed(state))
    return {};
  auto output = materializeOnlineAttentionFinalize(attention, *state, builder);
  if (mlir::failed(output))
    return {};
  attention.getResult(0).replaceAllUsesWith(*output);
  attention.erase();
  return module;
}

TEST(TemporalTilingTest, CoupledConsumerKeepsUnprovenTraversalsSeparate) {
  enum class Boundary {
    ReadInit,
    DifferentGrid,
    ReductionFirst,
    SharedAxis,
    NonFillInit,
    ExtraStateUse
  };
  for (Boundary boundary :
       {Boundary::ReadInit, Boundary::DifferentGrid, Boundary::ReductionFirst,
        Boundary::SharedAxis, Boundary::NonFillInit, Boundary::ExtraStateUse}) {
    SCOPED_TRACE(static_cast<int>(boundary));
    auto context = createContext();
    auto module = createOnlineFinalizerModule(*context, 1025, "f16");
    ASSERT_TRUE(module);
    LinalgExtOnlineAttentionOp online;
    mlir::linalg::GenericOp consumer;
    module->walk([&](LinalgExtOnlineAttentionOp op) { online = op; });
    module->walk([&](mlir::linalg::GenericOp op) { consumer = op; });
    mlir::OpBuilder builder(consumer);
    if (boundary == Boundary::ReadInit) {
      auto yield = mlir::cast<mlir::linalg::YieldOp>(
          consumer.getBody()->getTerminator());
      builder.setInsertionPoint(yield);
      auto add = builder.create<mlir::arith::AddFOp>(
          yield.getLoc(), yield.getValues()[0],
          consumer.getBody()->getArguments().back());
      yield->setOperand(0, add);
    } else if (boundary == Boundary::NonFillInit) {
      auto dps =
          mlir::cast<mlir::DestinationStyleOpInterface>(online.getOperation());
      auto init = dps.getDpsInitOperand(0);
      init->set(
          init->get().getDefiningOp<mlir::linalg::FillOp>().getOutputs()[0]);
    } else if (boundary == Boundary::ExtraStateUse) {
      // An observable scalar read of a second state feeds the consumer payload.
      auto zero =
          builder.create<mlir::arith::ConstantIndexOp>(online.getLoc(), 0);
      auto scalar = builder.create<mlir::tensor::ExtractOp>(
          online.getLoc(), online.getResult(1),
          mlir::ValueRange{zero, zero, zero});
      builder.setInsertionPoint(consumer.getBody()->getTerminator());
      auto yield = mlir::cast<mlir::linalg::YieldOp>(
          consumer.getBody()->getTerminator());
      auto cast = builder.create<mlir::arith::TruncFOp>(
          online.getLoc(), builder.getF16Type(), scalar);
      auto add = builder.create<mlir::arith::AddFOp>(
          online.getLoc(), yield.getValues()[0], cast);
      yield->setOperand(0, add);
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    TileRegionOp region = findRegion(*module);
    auto domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded()) << domain.failure->detail;
    auto choice = *domain.domain->getFirstChoice().getChoice();
    for (auto [scope, descriptor] :
         llvm::zip_equal(choice.scopes, domain.domain->getScopeDescriptors())) {
      for (int64_t &size : scope.iteratorTileSizes)
        size = std::min<int64_t>(size, 128);
      if (scope.operation == consumer) {
        if (boundary == Boundary::DifferentGrid)
          scope.iteratorTileSizes[2] = 64;
        if (boundary == Boundary::SharedAxis)
          scope.iteratorTileSizes[3] = 64;
      }
      auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                               scope.iteratorTileSizes,
                                               descriptor.precedence);
      ASSERT_TRUE(mlir::succeeded(order));
      scope.loopOrder = std::move(*order);
      if (scope.operation == online && boundary == Boundary::ReductionFirst)
        std::reverse(scope.loopOrder.begin(), scope.loopOrder.end());
    }
    ASSERT_TRUE(domain.domain->contains(choice));
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure failure;
    auto result =
        applyTemporalTiling(*domain.domain, choice, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(result)) << failure.detail;
    EXPECT_EQ(result->fusedProducers, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    unsigned fullStates = 0;
    module->walk([&](mlir::scf::ForOp loop) {
      for (mlir::Value state : loop.getRegionIterArgs()) {
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(state.getType());
        if (loop.getNumRegionIterArgs() == 3 && type &&
            type.getShape()[2] == 1025)
          ++fullStates;
      }
    });
    EXPECT_GT(fullStates, 0u);
  }
}

TEST(TemporalTilingTest, CoupledStateFinalizesInsideOutputTile) {
  // Rank-four actual materialization, both Q and K tails, and more than one
  // output wave. Q/K/V alias only to keep this structural witness
  // self-contained; the production PyTorch case supplies independent Q/K/V
  // inputs.
  for (bool permuted : {false, true}) {
    for (llvm::StringRef dtype : {"f16", "bf16"}) {
      for (int64_t extent : {1024, 1025, 1031, 4096, 4097}) {
        SCOPED_TRACE(dtype.str() + ":" + std::to_string(extent));
        auto context = createContext();
        auto module = createOnlineFinalizerModule(*context, extent, dtype);
        ASSERT_TRUE(module);
        if (permuted) {
          module->walk([&](mlir::linalg::GenericOp generic) {
            auto permutation = mlir::AffineMap::getPermutationMap(
                llvm::ArrayRef<unsigned>{0, 2, 1, 3}, context.get());
            llvm::SmallVector<mlir::AffineMap, 3> maps;
            for (mlir::AffineMap map : generic.getIndexingMapsArray())
              maps.push_back(map.compose(permutation));
            generic.setIndexingMapsAttr(
                mlir::Builder(context.get()).getAffineMapArrayAttr(maps));
          });
        }
        TileRegionOp region = findRegion(*module);
        auto domain = buildTemporalDomain(region);
        ASSERT_TRUE(domain.succeeded()) << domain.failure->detail;
        TemporalChoice choice = *domain.domain->getFirstChoice().getChoice();
        auto descriptors = domain.domain->getScopeDescriptors();
        for (auto [scope, descriptor] :
             llvm::zip_equal(choice.scopes, descriptors)) {
          for (int64_t &size : scope.iteratorTileSizes)
            size = std::min<int64_t>(size, 128);
          auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                                   scope.iteratorTileSizes,
                                                   descriptor.precedence);
          ASSERT_TRUE(mlir::succeeded(order));
          scope.loopOrder = std::move(*order);
        }
        StructuredMaterializationRelations relations;
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        TemporalTilingFailure failure;
        auto tiled =
            applyTemporalTiling(*domain.domain, choice, relations, &failure);
        ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
        EXPECT_EQ(tiled->fusedProducers, 1u);
        unsigned onlineBodies = 0;
        int64_t coveredWork = 0;
        module->walk([&](LinalgExtOnlineAttentionOp online) {
          ++onlineBodies;
          auto accumulatorType = mlir::cast<mlir::RankedTensorType>(
              online.getAccumulator().getType());
          auto keyType =
              mlir::cast<mlir::RankedTensorType>(online.getKey().getType());
          EXPECT_LE(accumulatorType.getShape()[2], 128);
          EXPECT_LE(keyType.getShape()[2], 128);
          int64_t dynamicCount = 1;
          for (mlir::Operation *parent = online->getParentOp();
               parent != region; parent = parent->getParentOp()) {
            if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
              auto lower = mlir::getConstantIntValue(loop.getLowerBound());
              auto upper = mlir::getConstantIntValue(loop.getUpperBound());
              auto step = mlir::getConstantIntValue(loop.getStep());
              ASSERT_TRUE(lower && upper && step);
              dynamicCount *= (*upper - *lower + *step - 1) / *step;
              if (loop.getNumRegionIterArgs() == 3) {
                for (mlir::Value state : loop.getRegionIterArgs()) {
                  auto type =
                      mlir::cast<mlir::RankedTensorType>(state.getType());
                  EXPECT_LE(type.getShape()[2], 128);
                }
              }
            }
          }
          coveredWork += dynamicCount * accumulatorType.getShape()[2] *
                         keyType.getShape()[2];
        });
        EXPECT_EQ(onlineBodies, extent % 128 == 0 ? 1u : 4u);
        EXPECT_EQ(coveredWork, extent * extent);
        unsigned finalizers = 0;
        module->walk([&](mlir::linalg::GenericOp generic) {
          ++finalizers;
          auto type = mlir::cast<mlir::RankedTensorType>(
              generic.getResult(0).getType());
          EXPECT_LE(type.getShape()[2], 128);
          EXPECT_EQ(generic.getNumReductionLoops(), 0u);
          auto accumulator =
              mlir::cast<mlir::OpResult>(generic.getDpsInputs()[0]);
          auto sum = mlir::cast<mlir::OpResult>(generic.getDpsInputs()[1]);
          EXPECT_EQ(accumulator.getOwner(), sum.getOwner());
          EXPECT_EQ(accumulator.getResultNumber(), 0u);
          EXPECT_EQ(sum.getResultNumber(), 2u);
          EXPECT_EQ(accumulator.getOwner()->getBlock(), generic->getBlock());
          EXPECT_TRUE(accumulator.getOwner()->isBeforeInBlock(generic));
        });
        EXPECT_EQ(finalizers, extent % 128 == 0 ? 1u : 2u);
        OnlineAttentionDecompositionFailure decompositionFailure;
        ASSERT_TRUE(mlir::succeeded(decomposeOnlineAttention(
            *module, relations, &decompositionFailure)))
            << decompositionFailure.detail;
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_EQ(movement.statistics.streamedOutputCarriers, 1u);
        unsigned fullStateFills = 0;
        module->walk([&](ComputeFillOp fill) {
          auto type = mlir::cast<mlir::MemRefType>(fill.getDest().getType());
          if (type.getShape() == llvm::ArrayRef<int64_t>{1, 2, extent, 128})
            ++fullStateFills;
        });
        EXPECT_EQ(fullStateFills, 0u);
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        ASSERT_EQ(standalone->size(), 1u);
        auto &tile = standalone->front();
        TileRegionToInstrLoweringSession session(*context);
        llvm::SmallVector<TileRegionOp, 2> tileRegions;
        tile.module->walk([&](TileRegionOp op) { tileRegions.push_back(op); });
        for (TileRegionOp op : tileRegions)
          ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(op, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        TileMemoryPlanningFailure memoryFailure;
        auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
        ASSERT_TRUE(mlir::succeeded(planned));
        EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
      }
    }
  }
}

} // namespace
