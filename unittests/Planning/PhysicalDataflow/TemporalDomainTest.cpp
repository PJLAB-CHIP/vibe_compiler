//===- TemporalDomainTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::vector<TemporalFusion> sharedFusions(const TemporalDomain &domain) {
  std::vector<TemporalFusion> result;
  for (const auto &fusion : domain.getFusions())
    if (fusion.uses.size() > 1)
      result.push_back(fusion);
  return result;
}

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
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
  auto query = queryTemporalFusion(chainOps.front()->getResult(0));
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
  EXPECT_EQ(sharedFusions(*fanoutDomain.domain).size(), 1u);
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
  TemporalFusionQueryResult path = queryTemporalFusion(
      mlir::cast<mlir::OpResult>(operations.front().getResult(0)));
  EXPECT_EQ(path.kind, TemporalFusionQueryKind::ExactDerived) << path.detail;
  EXPECT_TRUE(path.fusion->uses.front().viewToProducer.has_value());
  EXPECT_EQ(path.fusion->uses.front().reshapeDimensions,
            (llvm::SmallVector<uint32_t, 2>{0}));
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
          .size(),
      2u);
}

TEST(TemporalDomainTest, MultiUseViewResultFormsOneAllUseJointProducerGroup) {
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
  TemporalFusionQueryResult path = queryTemporalFusion(
      mlir::cast<mlir::OpResult>(operations.front().getResult(0)));
  ASSERT_TRUE(path.isExact()) << path.detail;
  EXPECT_EQ(path.fusion->uses.size(), 2u);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  auto groups = sharedFusions(*domain.domain);
  ASSERT_EQ(groups.size(), 1u);
  const TemporalFusion &group = groups.front();
  EXPECT_TRUE((group.uses.front().consumerValue != group.producer));
  EXPECT_EQ(group.uses.size(), 2u);
  EXPECT_EQ(group.uses.front().consumerValue.getDefiningOp(),
            operations.front()->getNextNode());
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
          .size(),
      4u);
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

TEST(TemporalDomainTest,
     BroadcastJointRequiresDependentLoopsBeforeInvariantLoops) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %producer_empty = tensor.empty() : tensor<2x4x1025xf16>
        %producer = linalg.generic {
            indexing_maps = [affine_map<(g, b, m) -> (g, b, m)>,
                             affine_map<(g, b, m) -> (g, b, m)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%arg : tensor<2x4x1025xf16>)
            outs(%producer_empty : tensor<2x4x1025xf16>) {
          ^bb0(%element: f16, %old: f16):
            %next = arith.addf %element, %element : f16
            linalg.yield %next : f16
        } -> tensor<2x4x1025xf16>
        %consumer_empty = tensor.empty() : tensor<2x4x1025x128xf16>
        %value = linalg.generic {
            indexing_maps = [affine_map<(g, b, m, n) -> (g, b, m)>,
                             affine_map<(g, b, m, n) -> (g, b, m, n)>],
            iterator_types = ["parallel", "parallel", "parallel",
                              "parallel"]}
            ins(%producer : tensor<2x4x1025xf16>)
            outs(%consumer_empty : tensor<2x4x1025x128xf16>) {
          ^bb0(%element: f16, %old: f16):
            linalg.yield %element : f16
        } -> tensor<2x4x1025x128xf16>)mlir",
                      "tensor<2x4x1025xf16>", "tensor<2x4x1025x128xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp operation) { region = operation; });
  llvm::SmallVector<mlir::linalg::GenericOp, 2> operations;
  region.walk([&](mlir::linalg::GenericOp operation) {
    operations.push_back(operation);
  });
  ASSERT_EQ(operations.size(), 2u);
  auto query = queryTemporalFusion(operations.front()->getResult(0));
  ASSERT_TRUE(query.isExact()) << query.detail;
  EXPECT_EQ(query.fusion->uses.front().representation,
            TemporalTileRepresentation::RectangularImage);
  TemporalDomainResult domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
  ASSERT_EQ(domain.domain->getFusions().size(), 1u);
  auto first = *domain.domain->getFirstChoice().getChoice();
  auto demand = queryTemporalFusionTile(
      domain.domain->getFusions().front(),
      domain.domain->getFusions().front().uses.front(), first.scopes.front());
  ASSERT_TRUE(demand.isExact()) << demand.reason;
  EXPECT_EQ(demand.image->invariantDimensions,
            (llvm::SmallVector<uint32_t, 4>{3}));
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
          .size(),
      2u);
  TemporalChoice joint = *domain.domain->getFirstChoice().getChoice();
  ASSERT_EQ(joint.scopes.size(), 1u);
  joint.scopes.front().iteratorTileSizes = {2, 4, 128, 64};
  joint.scopes.front().loopOrder = {2, 3};
  EXPECT_TRUE(domain.domain->contains(joint));
  joint.scopes.front().loopOrder = {3, 2};
  EXPECT_FALSE(domain.domain->contains(joint));
}

TEST(TemporalDomainTest,
     WindowJointAcceptsDisjointDemandAndRejectsOverlappingHalo) {
  for (bool disjoint : {true, false}) {
    SCOPED_TRACE(disjoint ? "disjoint" : "overlap");
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    const int64_t inputExtent = disjoint ? 3075 : 1027;
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
        << inputExtent
        << "x129xf16>)\n"
           "            outs(%producer_empty : tensor<2x"
        << inputExtent
        << "x129xf16>) {\n"
           "          ^bb0(%element: f16, %old: f16):\n"
           "            linalg.yield %element : f16\n"
           "        } -> tensor<2x"
        << inputExtent
        << "x129xf16>\n"
           "        %kernel = tensor.empty() : tensor<3xf16>\n"
           "        %consumer_empty = tensor.empty() : "
           "tensor<2x1025x129xf16>\n"
           "        %value = linalg.generic {\n"
           "            indexing_maps = [affine_map<(b, m, k, n) -> "
        << (disjoint ? "(b, m * 3 + k, n)>" : "(b, m + k, n)>")
        << ",\n"
           "                             affine_map<(b, m, k, n) -> (k)>,\n"
           "                             affine_map<(b, m, k, n) -> (b, m, "
           "n)>],\n"
           "            iterator_types = [\"parallel\", \"parallel\", "
           "\"reduction\", \"parallel\"]}\n"
           "            ins(%producer, %kernel : tensor<2x"
        << inputExtent
        << "x129xf16>, tensor<3xf16>)\n"
           "            outs(%consumer_empty : tensor<2x1025x129xf16>) {\n"
           "          ^bb0(%window_input: f16, %weight: f16, %acc: f16):\n"
           "            %product = arith.mulf %window_input, %weight : f16\n"
           "            %sum = arith.addf %product, %acc : f16\n"
           "            linalg.yield %sum : f16\n"
           "        } -> tensor<2x1025x129xf16>";
    auto module = parse(*context, stream.str(),
                        "tensor<2x" + std::to_string(inputExtent) + "x129xf16>",
                        "tensor<2x1025x129xf16>");
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp operation) { region = operation; });
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ASSERT_EQ(domain.domain->getFusions().size(), 1u);
    ASSERT_EQ(domain.domain->getScopeDescriptors().size(), 1u);
    TemporalChoice joint = *domain.domain->getFirstChoice().getChoice();
    ASSERT_EQ(joint.scopes.size(), 1u);
    joint.scopes.front().iteratorTileSizes = {2, 128, 3, 64};
    joint.scopes.front().loopOrder = {1, 3};
    EXPECT_EQ(domain.domain->contains(joint), disjoint);
    EXPECT_EQ(
        domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
            .size(),
        2u);
  }
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
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>] score {
            ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
              %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
              %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
              wafer.linalg_ext.attention.yield %attention_0_scaled : f32
            }
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
  EXPECT_EQ(descriptor.iteratorCapabilities[5],
            IteratorTilingCapability::FullExtentOnly);

  TemporalChoice choice = *built.domain->getFirstChoice().getChoice();
  choice.scopes.front().iteratorTileSizes[3] = 32;
  choice.scopes.front().loopOrder = {3};
  EXPECT_FALSE(built.domain->contains(choice));
  choice = *built.domain->getFirstChoice().getChoice();
  choice.scopes.front().iteratorTileSizes[4] = 128;
  choice.scopes.front().loopOrder = {4};
  EXPECT_TRUE(built.domain->contains(choice));
  choice.scopes.front().iteratorTileSizes[5] = 64;
  choice.scopes.front().loopOrder = {4, 5};
  EXPECT_FALSE(built.domain->contains(choice));

  // A caller bypassing the domain must get the same rejection without any
  // partially materialized slices or changes to the actual operation.
  auto online = mlir::cast<LinalgExtOnlineAttentionOp>(descriptor.operation);
  mlir::OpBuilder builder(online);
  llvm::SmallVector<mlir::OpFoldResult> offsets(6, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  for (int64_t extent : descriptor.iterationExtents)
    sizes.push_back(builder.getIndexAttr(extent));
  sizes[5] = builder.getIndexAttr(64);
  auto before = online->getBlock()->getOperations().size();
  EXPECT_TRUE(
      mlir::failed(online.getTiledImplementation(builder, offsets, sizes)));
  EXPECT_EQ(online->getBlock()->getOperations().size(), before);
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

TEST(TemporalDomainTest, FusedReductionRetainsEverySizeAndRemapsItsRole) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
    %e = tensor.empty() : tensor<2x1024x1024xf16>
    %p = linalg.batch_matmul ins(%arg, %arg : tensor<2x1024x1024xf16>, tensor<2x1024x1024xf16>) outs(%e : tensor<2x1024x1024xf16>) -> tensor<2x1024x1024xf16>
    %out = tensor.empty() : tensor<2x1024x1024xf16>
    %value = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m,n)>], iterator_types = ["parallel","parallel","parallel"]} ins(%p : tensor<2x1024x1024xf16>) outs(%out : tensor<2x1024x1024xf16>) {
      ^bb0(%x: f16, %old: f16): linalg.yield %x : f16
    } -> tensor<2x1024x1024xf16>)mlir",
                      "tensor<2x1024x1024xf16>", "tensor<2x1024x1024xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp op) { region = op; });
  auto domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  auto descriptors = domain.domain->getScopeDescriptors();
  ASSERT_EQ(descriptors.size(), 2u);
  ASSERT_EQ(descriptors[0].role, TemporalScopeRole::FusedReduction);
  EXPECT_EQ(descriptors[0].iteratorCapabilities[0],
            IteratorTilingCapability::FullExtentOnly);
  EXPECT_EQ(descriptors[0].iteratorCapabilities[3],
            IteratorTilingCapability::Tileable);
  auto choice = *domain.domain->getFirstChoice().getChoice();
  choice.scopes[1].iteratorTileSizes = {2, 128, 128};
  choice.scopes[1].loopOrder = {1, 2};
  for (int64_t size = 1; size <= 1024; ++size) {
    choice.scopes[0].iteratorTileSizes[3] = size;
    choice.scopes[0].loopOrder = size == 1024
                                     ? llvm::SmallVector<uint32_t, 4>{}
                                     : llvm::SmallVector<uint32_t, 4>{3};
    EXPECT_TRUE(domain.domain->contains(choice));
  }
  choice.scopes[0].iteratorTileSizes[1] = 128;
  EXPECT_FALSE(domain.domain->contains(choice));
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)[0]
          .iteratorCapabilities[1],
      IteratorTilingCapability::Tileable);
  auto function = region->getParentOfType<mlir::func::FuncOp>();
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::func::FuncOp> clone(
      mlir::cast<mlir::func::FuncOp>(function->clone(mapping)));
  auto mappedRegion =
      mlir::cast<TileRegionOp>(mapping.lookup(region.getOperation()));
  auto mapped = remapTemporalDomain(*domain.domain, mappedRegion, mapping);
  ASSERT_TRUE(mlir::succeeded(mapped));
  EXPECT_EQ(mapped->getScopeDescriptors()[0].role,
            TemporalScopeRole::FusedReduction);
  EXPECT_EQ(mapped->getScopeDescriptors()[0].operation,
            mapping.lookup(descriptors[0].operation));
}

TEST(TemporalDomainTest, DynamicTensorInterfaceDoesNotMaterializeADomainQuery) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
    %value = tensor.pad %arg low[0, 0, 1] high[0, 0, 1] {
      ^bb0(%b: index, %m: index, %n: index):
        %zero = arith.constant 0.0 : f16
        tensor.yield %zero : f16
    } : tensor<2x?x128xf16> to tensor<2x?x130xf16>)mlir",
                      "tensor<2x?x128xf16>", "tensor<2x?x130xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp op) { region = op; });
  size_t before = region.getBody().front().getOperations().size();
  auto domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  EXPECT_TRUE(domain.domain->getScopeDescriptors().empty());
  EXPECT_EQ(region.getBody().front().getOperations().size(), before);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalDomainTest, ExactRelationDoesNotAuthorizeUnsupportedTilingBounds) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
    %pe = tensor.empty() : tensor<2x1031x16xf16>
    %producer = linalg.generic {
      indexing_maps = [affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m,n)>],
      iterator_types = ["parallel","parallel","parallel"]}
      ins(%arg : tensor<2x1031x16xf16>) outs(%pe : tensor<2x1031x16xf16>) {
      ^bb0(%x: f16, %old: f16):
        %p = arith.addf %x, %x : f16
        linalg.yield %p : f16
    } -> tensor<2x1031x16xf16>
    %ce = tensor.empty() : tensor<2x1031x16xf16>
    %value = linalg.generic {
      indexing_maps = [affine_map<(b,m,n)->(b,1030-m,n)>, affine_map<(b,m,n)->(b,m,n)>],
      iterator_types = ["parallel","parallel","parallel"]}
      ins(%producer : tensor<2x1031x16xf16>) outs(%ce : tensor<2x1031x16xf16>) {
      ^bb0(%x: f16, %old: f16):
        %p = arith.addf %x, %x : f16
        linalg.yield %p : f16
    } -> tensor<2x1031x16xf16>
  )mlir",
                      "tensor<2x1031x16xf16>", "tensor<2x1031x16xf16>");
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp op) { region = op; });
  auto before = region.getBody().front().getOperations().size();
  auto domain = buildTemporalDomain(region);
  ASSERT_TRUE(domain.succeeded());
  ASSERT_EQ(domain.domain->getFusions().size(), 1u);
  auto choice = *domain.domain->getFirstChoice().getChoice();
  ASSERT_EQ(choice.scopes.size(), 1u);
  choice.scopes.front().iteratorTileSizes = {2, 128, 16};
  choice.scopes.front().loopOrder = {1};
  auto fusions = domain.domain->getFusions();
  const auto &fusion = fusions.front();
  auto exact = fusion.uses.front().iterationToProducer->getRectangularTileImage(
      context.get(), fusion.uses.front().iterationShape, {2, 1031, 16},
      {2, 128, 16});
  ASSERT_TRUE(exact.isExact()) << exact.reason;
  EXPECT_TRUE(exact.image->distinctTilesDisjoint);
  auto generated = queryTemporalFusionTile(fusion, fusion.uses.front(),
                                           choice.scopes.front());
  EXPECT_EQ(generated.status, analysis::IndexRelationStatus::Unsupported);
  EXPECT_NE(generated.reason.find("tiling interface"), std::string::npos);
  EXPECT_FALSE(domain.domain->contains(choice));
  EXPECT_EQ(
      domain.domain->getScopeDescriptors(TemporalTraversalKind::Independent)
          .size(),
      2u);
  EXPECT_EQ(region.getBody().front().getOperations().size(), before);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(TemporalDomainTest, SharedRequestsFollowTheirActualSelectedRoot) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool transposeAtSink : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(transposeAtSink);
      auto context = createContext();
      auto type = "tensor<2x" + std::to_string(extent) + "x" +
                  std::to_string(extent) + "xf16>";
      std::string body;
      llvm::raw_string_ostream b(body);
      auto pointwise = [&](llvm::StringRef name, llvm::StringRef input) {
        b << "%e_" << name << " = tensor.empty() : " << type << "\n%" << name
          << " = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,"
             "affine_map<(b,m,n)->(b,m,n)>], iterator_types = "
             "[\"parallel\",\"parallel\",\"parallel\"]} "
             "ins(%"
          << input << " : " << type << ") outs(%e_" << name << " : " << type
          << ") { ^bb0(%x: f16, %old: f16): %p = arith.addf %x, %x : f16 "
             "linalg.yield %p : f16 } -> "
          << type << "\n";
      };
      pointwise("producer", "arg");
      pointwise("left", "producer");
      pointwise("right", "producer");
      b << "%out = tensor.empty() : " << type
        << "\n%value = linalg.generic {indexing_maps = ["
           "affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->"
        << (transposeAtSink ? "(b,n,m)>" : "(b,m,n)>")
        << ", affine_map<(b,m,n)->(b,m,n)>], iterator_types = "
           "[\"parallel\",\"parallel\",\"parallel\"]} "
           "ins(%left, %right : "
        << type << ", " << type << ") outs(%out : " << type
        << ") { ^bb0(%x: f16, %y: f16, %old: f16): %p = arith.addf %x, %y : "
           "f16 linalg.yield %p : f16 } -> "
        << type;
      auto module = parse(*context, b.str(), type, type);
      ASSERT_TRUE(module);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      mlir::OpResult producer;
      region.walk([&](mlir::linalg::GenericOp op) {
        if (!producer)
          producer = op->getResult(0);
      });
      auto before = region.getBody().front().getOperations().size();
      auto query = queryTemporalFusion(producer);
      ASSERT_TRUE(query.isExact()) << query.detail;
      ASSERT_EQ(query.fusion->uses.size(), 2u);
      analysis::IndexRelationLimits limits;
      limits.maxRectangularPieces = 1;
      EXPECT_EQ(queryTemporalFusion(producer, limits).kind,
                TemporalFusionQueryKind::Indeterminate);
      auto domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded());
      auto choice = *domain.domain->getFirstChoice().getChoice();
      ASSERT_EQ(choice.scopes.size(), 1u);
      EXPECT_TRUE(domain.domain->contains(choice));
      choice.scopes.front().iteratorTileSizes = {2, 128, 64};
      choice.scopes.front().loopOrder = {1, 2};
      EXPECT_EQ(domain.domain->contains(choice), !transposeAtSink);
      EXPECT_EQ(region.getBody().front().getOperations().size(), before);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST(TemporalDomainTest, AnObservableUseRejectsTheEntireFusionGroup) {
  auto context = createContext();
  const std::string original = R"mlir(
    module { wafer.tile.module card_id = 0 tile_id = 0 {
      func.func private @observe(tensor<2x1031x16xf16>)
      func.func @entry(%input: tensor<2x1031x16xf16>) -> tensor<2x1031x16xf16> {
        %result = wafer.tile.region(%input : tensor<2x1031x16xf16>) -> (tensor<2x1031x16xf16>) {
        ^bb0(%arg: tensor<2x1031x16xf16>):
          %e = tensor.empty() : tensor<2x1031x16xf16>
          %p = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,affine_map<(b,m,n)->(b,m,n)>],
            iterator_types = ["parallel","parallel","parallel"]}
            ins(%arg : tensor<2x1031x16xf16>) outs(%e : tensor<2x1031x16xf16>) {
            ^bb0(%x: f16, %old: f16): %v = arith.addf %x, %x : f16 linalg.yield %v : f16
          } -> tensor<2x1031x16xf16>
          func.call @observe(%p) : (tensor<2x1031x16xf16>) -> ()
          %out = tensor.empty() : tensor<2x1031x16xf16>
          %value = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,affine_map<(b,m,n)->(b,m,n)>],
            iterator_types = ["parallel","parallel","parallel"]}
            ins(%p : tensor<2x1031x16xf16>) outs(%out : tensor<2x1031x16xf16>) {
            ^bb0(%x: f16, %old: f16): %v = arith.addf %x, %x : f16 linalg.yield %v : f16
          } -> tensor<2x1031x16xf16>
          wafer.tile.yield %value : tensor<2x1031x16xf16>
        }
        return %result : tensor<2x1031x16xf16>
      }
    } }
  )mlir";
  for (bool observeProducer : {true, false}) {
    SCOPED_TRACE(observeProducer);
    std::string text = original;
    if (!observeProducer) {
      const std::string call = "func.call @observe(%p)";
      text.replace(text.find(call), call.size(), R"mlir(
          %le = tensor.empty() : tensor<2x1031x16xf16>
          %left = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,affine_map<(b,m,n)->(b,m,n)>],
            iterator_types = ["parallel","parallel","parallel"]}
            ins(%p : tensor<2x1031x16xf16>) outs(%le : tensor<2x1031x16xf16>) {
            ^bb0(%x: f16, %old: f16): %v = arith.addf %x, %x : f16 linalg.yield %v : f16
          } -> tensor<2x1031x16xf16>
          func.call @observe(%left))mlir");
    }
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(module);
    mlir::OpResult producer;
    module->walk([&](mlir::linalg::GenericOp op) {
      if (!producer)
        producer = op->getResult(0);
    });
    auto uses = llvm::range_size(producer.getUses());
    auto query = queryTemporalFusion(producer);
    EXPECT_EQ(query.kind, observeProducer
                              ? TemporalFusionQueryKind::NonUnique
                              : TemporalFusionQueryKind::Unsupported);
    EXPECT_FALSE(query.fusion.has_value());
    EXPECT_EQ(llvm::range_size(producer.getUses()), uses);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

} // namespace
