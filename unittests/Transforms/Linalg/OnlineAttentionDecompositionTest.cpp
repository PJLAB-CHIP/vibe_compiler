//===- OnlineAttentionDecompositionTest.cpp ---------------------------===//

#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

std::string tensorType(llvm::StringRef shape, llvm::StringRef elementType) {
  return "tensor<" + shape.str() + "x" + elementType.str() + ">";
}

mlir::OwningOpRef<mlir::ModuleOp> parseOnlineModule(mlir::MLIRContext &context,
                                                    int64_t keyValueExtent,
                                                    llvm::StringRef elementType,
                                                    bool withMask) {
  const std::string queryType = tensorType("2x4x1025x64", elementType);
  const std::string keyType =
      tensorType("2x4x" + std::to_string(keyValueExtent) + "x64", elementType);
  const std::string valueType =
      tensorType("2x4x" + std::to_string(keyValueExtent) + "x128", elementType);
  const std::string accumulatorType = tensorType("2x4x1025x128", elementType);
  const std::string maskType =
      "tensor<2x1025x" + std::to_string(keyValueExtent) + "xf32>";
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << R"mlir(module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%query: )mlir"
         << queryType << ", %key: " << keyType
         << ", %value_input: " << valueType;
  if (withMask)
    stream << ", %mask: " << maskType;
  stream << ") -> " << accumulatorType << R"mlir( {
      %result = wafer.tile.region(%query, %key, %value_input)mlir";
  if (withMask)
    stream << ", %mask";
  stream << " : " << queryType << ", " << keyType << ", " << valueType;
  if (withMask)
    stream << ", " << maskType;
  stream << ") -> (" << accumulatorType << R"mlir() {
      ^bb0(%query_arg: )mlir"
         << queryType << ", %key_arg: " << keyType
         << ", %value_arg: " << valueType;
  if (withMask)
    stream << ", %mask_arg: " << maskType;
  stream << R"mlir():
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : )mlir"
         << accumulatorType << R"mlir(
        %maximum = tensor.empty() : tensor<2x4x1025xf32>
        %sum = tensor.empty() : tensor<2x4x1025xf32>
        %next_accumulator, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%query_arg, %key_arg, %value_arg, %scale)mlir";
  if (withMask)
    stream << ", %mask_arg";
  stream << " : " << queryType << ", " << keyType << ", " << valueType
         << ", f32";
  if (withMask)
    stream << ", " << maskType;
  stream << R"mlir()
            outs(%accumulator, %maximum, %sum : )mlir"
         << accumulatorType
         << R"mlir(, tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
            indexing_maps = [
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>,
              affine_map<(b, h, m, k1, k2, n) -> ()>,
)mlir";
  if (withMask)
    stream << "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, m, k2)>,\n";
  stream
      << R"mlir(              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>]
            -> ()mlir"
      << accumulatorType << R"mlir(, tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
        wafer.tile.yield %next_accumulator : )mlir"
      << accumulatorType << R"mlir(
      }
      return %result : )mlir"
      << accumulatorType << R"mlir(
    }
  }
}
)mlir";
  return mlir::parseSourceString<mlir::ModuleOp>(stream.str(),
                                                 mlir::ParserConfig(&context));
}

TileRegionOp findRegion(mlir::ModuleOp module) {
  TileRegionOp result;
  module.walk([&](TileRegionOp region) { result = region; });
  return result;
}

template <typename Op> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](Op) { ++count; });
  return count;
}

TemporalChoice selectK2Tile(const TemporalDomain &domain, int64_t tileSize) {
  TemporalSuccessor first = domain.getFirstChoice();
  EXPECT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  TemporalChoice choice = *first.getChoice();
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.getScopeDescriptors();
  EXPECT_EQ(choice.scopes.size(), descriptors.size());
  for (auto [scope, descriptor] : llvm::zip_equal(choice.scopes, descriptors)) {
    auto online =
        mlir::dyn_cast<LinalgExtOnlineAttentionOp>(descriptor.operation);
    if (!online)
      continue;
    auto roles = online.getIterationRoles();
    EXPECT_TRUE(mlir::succeeded(roles));
    for (unsigned dimension : roles->keyValueReduction)
      scope.iteratorTileSizes[dimension] =
          std::min(tileSize, descriptor.iterationExtents[dimension]);
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    EXPECT_TRUE(mlir::succeeded(order));
    if (mlir::succeeded(order))
      scope.loopOrder = std::move(*order);
  }
  EXPECT_TRUE(domain.contains(choice));
  return choice;
}

unsigned countContractions(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::linalg::LinalgOp operation) {
    if (operation.getNumDpsInputs() == 2 &&
        llvm::is_contained(operation.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      ++count;
  });
  return count;
}

unsigned countRowReductions(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::linalg::LinalgOp operation) {
    if (operation.getNumDpsInputs() == 1 &&
        llvm::is_contained(operation.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      ++count;
  });
  return count;
}

TEST(OnlineAttentionDecompositionTest,
     MainAndTailBecomeActualQKStateAndPVWithoutNewLoops) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto module = parseOnlineModule(*context, extent, "f16",
                                    /*withMask=*/false);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    TemporalChoice choice = selectK2Tile(*domain.domain, 128);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    TemporalTilingFailure tilingFailure;
    auto tiled =
        applyTemporalTiling(*domain.domain, choice, relations, &tilingFailure);
    ASSERT_TRUE(mlir::succeeded(tiled)) << tilingFailure.detail;
    const unsigned onlineBefore =
        countOps<LinalgExtOnlineAttentionOp>(module->getOperation());
    const unsigned loopsBefore =
        countOps<mlir::scf::ForOp>(module->getOperation());
    ASSERT_EQ(onlineBefore, extent == 1024 ? 1u : 2u);

    OnlineAttentionDecompositionFailure failure;
    auto decomposed = decomposeOnlineAttention(*module, relations, &failure);
    ASSERT_TRUE(mlir::succeeded(decomposed)) << failure.detail;
    EXPECT_EQ(decomposed->decomposedOperations, onlineBefore);
    EXPECT_EQ(decomposed->qkContractions, onlineBefore);
    EXPECT_EQ(decomposed->pvContractions, onlineBefore);
    EXPECT_EQ(decomposed->scaleApplications, onlineBefore);
    EXPECT_EQ(decomposed->maskApplications, 0u);
    EXPECT_EQ(decomposed->rowReductions, onlineBefore * 2);
    EXPECT_EQ(decomposed->normalizationFactors, onlineBefore);
    EXPECT_EQ(decomposed->probabilityUpdates, onlineBefore);
    EXPECT_EQ(decomposed->stateScales, onlineBefore * 2);
    EXPECT_EQ(decomposed->scoreScratchTensors, onlineBefore);
    EXPECT_EQ(decomposed->maximumScoreElements, UINT64_C(2) * 4 * 1025 * 128);
    EXPECT_EQ(countOps<LinalgExtAttentionOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()), loopsBefore);
    EXPECT_EQ(countContractions(module->getOperation()), onlineBefore * 2);
    EXPECT_EQ(countRowReductions(module->getOperation()), onlineBefore * 2);
    EXPECT_EQ(countOps<mlir::math::ExpOp>(module->getOperation()),
              onlineBefore * 2);
    module->walk([&](mlir::scf::ForOp loop) {
      EXPECT_EQ(loop.getNumRegionIterArgs(), 3u);
    });
    unsigned scoreScratch = 0;
    module->walk([&](mlir::tensor::EmptyOp empty) {
      auto type = empty.getType();
      if (type.getRank() != 4 || !type.getElementType().isF32())
        return;
      ++scoreScratch;
      EXPECT_EQ(type.getShape()[0], 2);
      EXPECT_EQ(type.getShape()[1], 4);
      EXPECT_EQ(type.getShape()[2], 1025);
      EXPECT_LE(type.getShape()[3], 128);
    });
    EXPECT_EQ(scoreScratch, onlineBefore);
    EXPECT_TRUE(
        mlir::succeeded(verifyOnlineAttentionDecompositionComplete(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
  }
}

TEST(OnlineAttentionDecompositionTest,
     BroadcastMaskAndBF16UseTheSameModeNeutralDecomposition) {
  for (auto [elementType, withMask] :
       {std::pair<llvm::StringRef, bool>{"f16", true}, {"bf16", false}}) {
    SCOPED_TRACE(elementType.str());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto module = parseOnlineModule(*context, 1031, elementType, withMask);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    TemporalChoice choice = selectK2Tile(*domain.domain, 128);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    ASSERT_TRUE(mlir::succeeded(
        applyTemporalTiling(*domain.domain, choice, relations)));
    const unsigned onlineBefore =
        countOps<LinalgExtOnlineAttentionOp>(module->getOperation());
    auto decomposed = decomposeOnlineAttention(*module, relations);
    ASSERT_TRUE(mlir::succeeded(decomposed));
    EXPECT_EQ(decomposed->decomposedOperations, onlineBefore);
    EXPECT_EQ(decomposed->scaleApplications, onlineBefore);
    EXPECT_EQ(decomposed->maskApplications, withMask ? onlineBefore : 0u);
    unsigned maskConsumers = 0;
    module->walk([&](mlir::linalg::GenericOp generic) {
      if (llvm::any_of(generic.getDpsInputs(), [](mlir::Value value) {
            auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
            return type && type.getRank() == 3 && type.getShape()[1] == 1025;
          }))
        ++maskConsumers;
    });
    EXPECT_EQ(maskConsumers, withMask ? onlineBefore : 0u);
    EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()), 0u);
  }
}

TEST(OnlineAttentionDecompositionTest, GraphAttentionFailsBeforeMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @main(%query: tensor<2x1025x64xf16>,
                  %key: tensor<2x1031x64xf16>,
                  %value: tensor<2x1031x128xf16>, %scale: f32)
      -> tensor<2x1025x128xf16> {
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x64xf16>,
            tensor<2x1031x64xf16>, tensor<2x1031x128xf16>, f32)
        outs(%empty : tensor<2x1025x128xf16>)
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x128xf16>
    return %result : tensor<2x1025x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  StructuredMaterializationRelations relations;
  OnlineAttentionDecompositionFailure failure;
  EXPECT_TRUE(
      mlir::failed(decomposeOnlineAttention(*module, relations, &failure)));
  EXPECT_EQ(failure.kind,
            OnlineAttentionDecompositionFailureKind::BrokenContract);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(OnlineAttentionDecompositionTest,
     DynamicOnlineTileReturnsUnsupportedWithoutMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#acc = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#row = affine_map<(b, m, k1, k2, n) -> (b, m)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%query: tensor<2x1025x64xf16>,
                     %key: tensor<2x?x64xf16>,
                     %value: tensor<2x?x128xf16>)
        -> tensor<2x1025x128xf16> {
      %result = wafer.tile.region(
          %query, %key, %value : tensor<2x1025x64xf16>,
          tensor<2x?x64xf16>, tensor<2x?x128xf16>)
          -> (tensor<2x1025x128xf16>) {
      ^bb0(%query_arg: tensor<2x1025x64xf16>,
           %key_arg: tensor<2x?x64xf16>,
           %value_arg: tensor<2x?x128xf16>):
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : tensor<2x1025x128xf16>
        %maximum = tensor.empty() : tensor<2x1025xf32>
        %sum = tensor.empty() : tensor<2x1025xf32>
        %next_accumulator, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%query_arg, %key_arg, %value_arg, %scale :
                tensor<2x1025x64xf16>, tensor<2x?x64xf16>,
                tensor<2x?x128xf16>, f32)
            outs(%accumulator, %maximum, %sum : tensor<2x1025x128xf16>,
                tensor<2x1025xf32>, tensor<2x1025xf32>)
            indexing_maps = [#q, #k, #v, #s, #acc, #row, #row]
            -> (tensor<2x1025x128xf16>, tensor<2x1025xf32>,
                tensor<2x1025xf32>)
        wafer.tile.yield %next_accumulator : tensor<2x1025x128xf16>
      }
      return %result : tensor<2x1025x128xf16>
    }
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  OnlineAttentionDecompositionFailure failure;
  EXPECT_TRUE(
      mlir::failed(decomposeOnlineAttention(*module, relations, &failure)));
  EXPECT_EQ(failure.kind,
            OnlineAttentionDecompositionFailureKind::UnsupportedSemantics);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

} // namespace
