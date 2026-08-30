//===- TemporalDomainTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef body,
                                        llvm::StringRef argumentType,
                                        llvm::StringRef resultType) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module {\n"
            "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
            "    func.func @entry(%input: "
         << argumentType << ") -> " << resultType
         << " {\n"
            "      %result = wafer.tile.region(%input : "
         << argumentType << ") -> (" << resultType
         << ") {\n"
            "      ^bb0(%arg: "
         << argumentType << "):\n"
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

std::optional<std::vector<TemporalChoice>>
enumerate(const TemporalDomain &domain, size_t limit = 100000) {
  std::vector<TemporalChoice> choices;
  TemporalSuccessor current = domain.getFirstChoice();
  while (current.getKind() == TemporalSuccessorKind::Choice) {
    if (!current.getChoice() || !current.getCursor() || choices.size() >= limit)
      return std::nullopt;
    choices.push_back(*current.getChoice());
    current = domain.getNextChoice(*current.getCursor());
  }
  return current.getKind() == TemporalSuccessorKind::End
             ? std::optional<std::vector<TemporalChoice>>(std::move(choices))
             : std::nullopt;
}

TEST(TemporalDomainTest,
     EveryPositiveSizeAndActiveOrderMatchesIndependentBoundedOracle) {
  // The 1x2x3 shape intentionally bounds exhaustive enumeration. Item 13's
  // transformation behavior is covered separately at 1024/1025/1031 scale.
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %empty = tensor.empty() : tensor<1x2x3xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<1x2x3xf16>)
            outs(%empty : tensor<1x2x3xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<1x2x3xf16>)mlir",
                      "tensor<1x2x3xf16>", "tensor<1x2x3xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  ASSERT_TRUE(region);
  TemporalDomainResult built = buildTemporalDomain(region);
  ASSERT_TRUE(built.succeeded())
      << (built.failure ? built.failure->detail : "");
  auto choices = enumerate(*built.domain);
  ASSERT_TRUE(choices);
  EXPECT_EQ(choices->size(), 16u);

  std::set<std::tuple<TemporalTraversalKind, std::vector<int64_t>,
                      std::vector<uint32_t>>>
      observed;
  unsigned jointChoices = 0;
  unsigned independentChoices = 0;
  for (const TemporalChoice &choice : *choices) {
    ASSERT_EQ(choice.scopes.size(), 1u);
    EXPECT_TRUE(built.domain->contains(choice));
    if (choice.kind == TemporalTraversalKind::Joint)
      ++jointChoices;
    else
      ++independentChoices;
    const TemporalScopeChoice &scope = choice.scopes.front();
    EXPECT_TRUE(
        observed
            .insert({choice.kind,
                     std::vector<int64_t>(scope.iteratorTileSizes.begin(),
                                          scope.iteratorTileSizes.end()),
                     std::vector<uint32_t>(scope.loopOrder.begin(),
                                           scope.loopOrder.end())})
            .second);
  }
  EXPECT_EQ(jointChoices, 8u);
  EXPECT_EQ(independentChoices, 8u);
}

TEST(TemporalDomainTest, PrecedenceDiamondEnumeratesOnlyLinearExtensions) {
  llvm::SmallVector<int64_t, 4> extents{2, 2, 2, 2};
  llvm::SmallVector<int64_t, 4> sizes{1, 1, 1, 1};
  llvm::SmallVector<TemporalPrecedenceEdge, 4> precedence{
      {0, 1}, {0, 2}, {1, 3}, {2, 3}};
  auto first = buildFirstTemporalLoopOrder(extents, sizes, precedence);
  ASSERT_TRUE(mlir::succeeded(first));
  EXPECT_EQ(*first, (llvm::SmallVector<uint32_t, 4>{0, 1, 2, 3}));

  llvm::SmallVector<TemporalPrecedenceEdge, 2> cyclic{{0, 1}, {1, 0}};
  EXPECT_TRUE(
      mlir::failed(buildFirstTemporalLoopOrder({2, 2}, {1, 1}, cyclic)));
}

TEST(TemporalDomainTest,
     ExactSingleUseChainHasOneRootWhileMultiUseRemainsIndependent) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral chainBody = R"mlir(
        %first_empty = tensor.empty() : tensor<2x1025x128xf16>
        %first = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%first_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>
        %second_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%first : tensor<2x1025x128xf16>)
            outs(%second_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>)mlir";
  auto chain = parse(*context, chainBody, "tensor<2x1025x128xf16>",
                     "tensor<2x1025x128xf16>");
  ASSERT_TRUE(chain);
  std::string chainBefore;
  llvm::raw_string_ostream chainBeforeStream(chainBefore);
  chain->print(chainBeforeStream);
  chainBeforeStream.flush();
  TileRegionOp chainRegion;
  chain->walk([&](TileRegionOp operation) { chainRegion = operation; });
  TemporalDomainResult chainDomain = buildTemporalDomain(chainRegion);
  ASSERT_TRUE(chainDomain.succeeded());
  ASSERT_EQ(chainDomain.domain->getScopeDescriptors().size(), 1u);
  llvm::SmallVector<mlir::linalg::GenericOp, 2> chainOps;
  chainRegion.walk([&](mlir::linalg::GenericOp operation) {
    chainOps.push_back(operation);
  });
  ASSERT_EQ(chainOps.size(), 2u);
  ASSERT_TRUE(chainOps.front()->getResult(0).hasOneUse());
  auto query = queryTemporalProducerFusion(
      chainOps.front()->getResult(0),
      *chainOps.front()->getResult(0).getUses().begin());
  EXPECT_EQ(query.kind, TemporalFusionQueryKind::ExactDerived);
  EXPECT_EQ(chainDomain.domain->getScopeDescriptors().front().operation,
            chainOps.back().getOperation());
  std::string chainAfter;
  llvm::raw_string_ostream chainAfterStream(chainAfter);
  chain->print(chainAfterStream);
  chainAfterStream.flush();
  EXPECT_EQ(chainAfter, chainBefore);

  constexpr llvm::StringLiteral fanoutBody = R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>
        %left_empty = tensor.empty() : tensor<2x1025x128xf16>
        %left = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025x128xf16>)
            outs(%left_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>
        %right_empty = tensor.empty() : tensor<2x1025x128xf16>
        %right = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025x128xf16>)
            outs(%right_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
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
            %sum = arith.addf %lhs, %rhs : f16
            linalg.yield %sum : f16
        } -> tensor<2x1025x128xf16>)mlir";
  auto fanout = parse(*context, fanoutBody, "tensor<2x1025x128xf16>",
                      "tensor<2x1025x128xf16>");
  ASSERT_TRUE(fanout);
  TileRegionOp fanoutRegion;
  fanout->walk([&](TileRegionOp operation) { fanoutRegion = operation; });
  TemporalDomainResult fanoutDomain = buildTemporalDomain(fanoutRegion);
  ASSERT_TRUE(fanoutDomain.succeeded());
  // Joint traversal derives the shared all-use producer and both branches;
  // the independent domain still exposes every original traversal.
  EXPECT_EQ(fanoutDomain.domain->getScopeDescriptors().size(), 1u);
  EXPECT_EQ(fanoutDomain.domain
                ->getScopeDescriptors(TemporalTraversalKind::Independent)
                .size(),
            4u);
  EXPECT_EQ(fanoutDomain.domain->getJointProducerGroups().size(), 1u);
}

TEST(TemporalDomainTest,
     GeneralFlattenViewHasJointAndIndependentTemporalChoices) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>
        %flattened = tensor.collapse_shape %producer [[0, 1], [2]]
            : tensor<2x1025x128xf16> into tensor<2050x128xf16>
        %consumer_empty = tensor.empty() : tensor<2050x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(m, n) -> (m, n)>,
                             affine_map<(m, n) -> (m, n)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%flattened : tensor<2050x128xf16>)
            outs(%consumer_empty : tensor<2050x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2050x128xf16>)mlir",
                      "tensor<2x1025x128xf16>", "tensor<2050x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    operations.push_back(operation);
  });
  ASSERT_EQ(operations.size(), 2u);
  TemporalFusionPathResult path = queryTemporalProducerFusionPath(
      mlir::cast<mlir::OpResult>(operations.front().getResult(0)));
  EXPECT_EQ(path.kind, TemporalFusionQueryKind::ExactDerived) << path.detail;
  EXPECT_TRUE(path.consumerViewToProducer.has_value());
  EXPECT_EQ(path.generalReshapeConsumerDimensions,
            (llvm::SmallVector<uint32_t, 2>{0}));
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
          .size(),
      2u);
}

TEST(TemporalDomainTest, MultiUseViewResultDoesNotCloneOrDeriveItsProducer) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1031x128xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x1031x128xf16>)
            outs(%producer_empty : tensor<2x1031x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1031x128xf16>
        %view = tensor.extract_slice %producer[0, 0, 0] [2, 1025, 128]
            [1, 1, 1] : tensor<2x1031x128xf16> to tensor<2x1025x128xf16>
        %left_empty = tensor.empty() : tensor<2x1025x128xf16>
        %left = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%view : tensor<2x1025x128xf16>)
            outs(%left_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>
        %right_empty = tensor.empty() : tensor<2x1025x128xf16>
        %right = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%view : tensor<2x1025x128xf16>)
            outs(%right_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
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
            %sum = arith.addf %lhs, %rhs : f16
            linalg.yield %sum : f16
        } -> tensor<2x1025x128xf16>)mlir",
                      "tensor<2x1031x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  llvm::SmallVector<mlir::linalg::GenericOp, 4> operations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    operations.push_back(operation);
  });
  ASSERT_EQ(operations.size(), 4u);
  TemporalFusionPathResult path = queryTemporalProducerFusionPath(
      mlir::cast<mlir::OpResult>(operations.front().getResult(0)));
  EXPECT_EQ(path.kind, TemporalFusionQueryKind::NonUnique);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 2u);
}

TEST(TemporalDomainTest,
     IncompleteInsertAssemblyIsUnsupportedWithoutRejectingTheRegion) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %left = tensor.extract_slice %arg[0, 0, 0] [2, 1025, 63]
            [1, 1, 1] : tensor<2x1025x128xf16> to tensor<2x1025x63xf16>
        %right = tensor.extract_slice %arg[0, 0, 64] [2, 1025, 64]
            [1, 1, 1] : tensor<2x1025x128xf16> to tensor<2x1025x64xf16>
        %assembly_empty = tensor.empty() : tensor<2x1025x128xf16>
        %with_left = tensor.insert_slice %left into
            %assembly_empty[0, 0, 0] [2, 1025, 63] [1, 1, 1]
            : tensor<2x1025x63xf16> into tensor<2x1025x128xf16>
        %joined = tensor.insert_slice %right into
            %with_left[0, 0, 64] [2, 1025, 64] [1, 1, 1]
            : tensor<2x1025x64xf16> into tensor<2x1025x128xf16>
        %result_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%joined : tensor<2x1025x128xf16>)
            outs(%result_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>)mlir",
                      "tensor<2x1025x128xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  mlir::linalg::GenericOp consumer;
  region.walk([&](mlir::linalg::GenericOp operation) { consumer = operation; });
  ASSERT_TRUE(consumer);
  TemporalConcatQueryResult concat =
      queryTemporalConcatAssembly(consumer->getOpOperand(0));
  EXPECT_EQ(concat.kind, TemporalConcatQueryKind::Unsupported);
  EXPECT_FALSE(concat.detail.empty());
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
}

TEST(TemporalDomainTest, BroadcastEdgeKeepsAnIndependentProducerTraversal) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %producer_empty = tensor.empty() : tensor<2x1025xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(b, m) -> (b, m)>,
                             affine_map<(b, m) -> (b, m)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%arg : tensor<2x1025xf16>)
            outs(%producer_empty : tensor<2x1025xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x1025xf16>
        %consumer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%producer : tensor<2x1025xf16>)
            outs(%consumer_empty : tensor<2x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x1025x128xf16>)mlir",
                      "tensor<2x1025xf16>", "tensor<2x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    operations.push_back(operation);
  });
  ASSERT_EQ(operations.size(), 2u);
  auto query = queryTemporalProducerFusion(
      operations.front()->getResult(0),
      *operations.front()->getResult(0).getUses().begin());
  EXPECT_EQ(query.kind, TemporalFusionQueryKind::NonUnique);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 2u);
}

TEST(TemporalDomainTest, RankSixOnlineAttentionKeepsK1FullAndK2InTheRawDomain) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %key = tensor.empty() : tensor<2x4x1031x64xf16>
        %value_input = tensor.empty() : tensor<2x4x1031x128xf16>
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : tensor<2x4x1025x128xf16>
        %maximum = tensor.empty() : tensor<2x4x1025xf32>
        %sum = tensor.empty() : tensor<2x4x1025xf32>
        %value, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%arg, %key, %value_input, %scale : tensor<2x4x1025x64xf16>,
                tensor<2x4x1031x64xf16>, tensor<2x4x1031x128xf16>, f32)
            outs(%accumulator, %maximum, %sum : tensor<2x4x1025x128xf16>,
                tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
            indexing_maps = [
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>,
              affine_map<(b, h, m, k1, k2, n) -> ()>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>]
            -> (tensor<2x4x1025x128xf16>, tensor<2x4x1025xf32>,
                tensor<2x4x1025xf32>))mlir",
                      "tensor<2x4x1025x64xf16>", "tensor<2x4x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  TemporalDomainResult built = buildTemporalDomain(region);
  ASSERT_TRUE(built.succeeded())
      << (built.failure ? built.failure->detail : "");
  ASSERT_EQ(built.domain->getScopeDescriptors().size(), 1u);
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      built.domain->getScopeDescriptors();
  const TemporalScopeDescriptor &descriptor = descriptors.front();
  ASSERT_EQ(descriptor.iterationExtents.size(), 6u);
  EXPECT_EQ(descriptor.iteratorCapabilities[3],
            IteratorTilingCapability::FullExtentOnly);
  EXPECT_EQ(descriptor.iteratorCapabilities[4],
            IteratorTilingCapability::Tileable);

  TemporalChoice choice = *built.domain->getFirstChoice().getChoice();
  choice.scopes.front().iteratorTileSizes[3] = 32;
  choice.scopes.front().loopOrder = {3};
  EXPECT_FALSE(built.domain->contains(choice));
  choice = *built.domain->getFirstChoice().getChoice();
  choice.scopes.front().iteratorTileSizes[4] = 128;
  choice.scopes.front().loopOrder = {4};
  EXPECT_TRUE(built.domain->contains(choice));
}

TEST(TemporalDomainTest, DynamicTraversalStaysFullExtentWithoutChangingIR) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %value = linalg.generic {
            indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                             affine_map<(b, m, n) -> (b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x?x128xf16>)
            outs(%arg : tensor<2x?x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x?x128xf16>)mlir",
                      "tensor<2x?x128xf16>", "tensor<2x?x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  TemporalDomainResult built = buildTemporalDomain(region);
  ASSERT_TRUE(built.succeeded());
  EXPECT_TRUE(built.domain->getScopeDescriptors().empty());
  TemporalSuccessor first = built.domain->getFirstChoice();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  ASSERT_TRUE(first.getChoice());
  EXPECT_TRUE(first.getChoice()->scopes.empty());
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(TemporalDomainTest, IntervalSplitPreservesEveryPoint) {
  auto split = splitTemporalSizeInterval({1, 1031}, 128);
  ASSERT_TRUE(mlir::succeeded(split));
  EXPECT_EQ(split->singleton, (TemporalSizeInterval{128, 128}));
  ASSERT_TRUE(split->above);
  ASSERT_TRUE(split->below);
  EXPECT_EQ(*split->above, (TemporalSizeInterval{129, 1031}));
  EXPECT_EQ(*split->below, (TemporalSizeInterval{1, 127}));
}

} // namespace
