//===- TemporalTilingTest.cpp -----------------------------------------===//

#include "Wafer/Transforms/Linalg/TemporalTiling.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
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
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
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
     MultiUseProducerStaysInOneIndependentTraversalWhileBranchesFuse) {
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
  ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 2u);
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
  EXPECT_EQ(tiled->tiledTraversals, 2u);
  EXPECT_EQ(tiled->fusedProducers, 2u);
  EXPECT_EQ(tiled->loops, 2u);
  EXPECT_EQ(tiled->specializedTails, 2u);
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

} // namespace
