//===- ComputeImplementationTest.cpp ---------------------------------===//

#include "Wafer/Planning/Search/ComputeImplementation.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

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

mlir::OwningOpRef<mlir::ModuleOp> parseProgram(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<4xf16>) -> tensor<4xf16> {
    %empty = tensor.empty() : tensor<4xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%empty : tensor<4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %one = arith.constant 1.000000e+00 : f16
        %reciprocal = arith.divf %one, %value : f16
        linalg.yield %reciprocal : f16
    } -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}
)mlir",
                                                 mlir::ParserConfig(&context));
}

std::unique_ptr<CardProgramAnalysis> analyze(mlir::ModuleOp module,
                                             std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, failureReason);
  auto topology = TargetTopology::create(module, failureReason);
  if (mlir::failed(dag) || mlir::failed(topology))
    return {};
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  for (const StructuredDAGNode &node : dag->getNodes())
    operationNodes.push_back({node.operation, node.id});
  return std::make_unique<CardProgramAnalysis>(
      std::move(*topology), llvm::SmallVector<TileId, 16>{TileId(0)},
      std::move(*dag), StaticOutputDomains{{4}}, std::move(operationNodes));
}

unsigned countKind(mlir::Operation *root, ComputeElementwiseKind kind) {
  unsigned count = 0;
  root->walk([&](ComputeElementwiseOp operation) {
    count += operation.getKind() == kind;
  });
  return count;
}

TEST(ComputeImplementationTest,
     KeepsNaturalAndExactReciprocalAsDistinctActualChoices) {
  auto context = createContext();
  auto module = parseProgram(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = analyze(*module, &failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto domain =
      CardComputeImplementationDomain::create(*program, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardComputeImplementationAssignment natural = domain->getFirstAssignment();
  ASSERT_EQ(natural.nodes.size(), 1u);
  EXPECT_EQ(natural.nodes.front().implementation,
            StructuredComputeImplementation::Natural);
  auto next = domain->getNextAssignment(natural);
  ASSERT_TRUE(mlir::succeeded(next));
  ASSERT_TRUE(*next);
  CardComputeImplementationAssignment reciprocal = std::move(**next);
  EXPECT_EQ(reciprocal.nodes.front().implementation,
            StructuredComputeImplementation::Reciprocal);
  auto end = domain->getNextAssignment(reciprocal);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);

  StructuredNodeShardGroup naturalGroup;
  naturalGroup.shards.push_back(
      StructuredNodeIterationShard{0,
                                   TileId(0),
                                   {0},
                                   {4},
                                   {}});
  naturalGroup.implementations.push_back(
      {0, StructuredComputeImplementation::Natural});
  StructuredNodeShardGroup reciprocalGroup = naturalGroup;
  reciprocalGroup.implementations.front().implementation =
      StructuredComputeImplementation::Reciprocal;
  mlir::OwningOpRef<mlir::ModuleOp> naturalCard;
  mlir::OwningOpRef<mlir::ModuleOp> reciprocalCard;
  ASSERT_TRUE(mlir::succeeded(lowerStructuredNodeGroupsToCardModule(
      *module, CardId(0), llvm::ArrayRef<TileId>{TileId(0)},
      program->operationNodes, llvm::ArrayRef{naturalGroup}, naturalCard,
      nullptr, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(lowerStructuredNodeGroupsToCardModule(
      *module, CardId(0), llvm::ArrayRef<TileId>{TileId(0)},
      program->operationNodes, llvm::ArrayRef{reciprocalGroup}, reciprocalCard,
      nullptr, &failureReason)))
      << failureReason;
  EXPECT_GT(countKind(naturalCard->getOperation(), ComputeElementwiseKind::Div),
            0u);
  EXPECT_EQ(
      countKind(naturalCard->getOperation(), ComputeElementwiseKind::Recip),
      0u);
  EXPECT_EQ(
      countKind(reciprocalCard->getOperation(), ComputeElementwiseKind::Div),
      0u);
  EXPECT_GT(
      countKind(reciprocalCard->getOperation(), ComputeElementwiseKind::Recip),
      0u);
}

TEST(ComputeImplementationTest, NonUnitNumeratorHasNoReciprocalSibling) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<4xf16>) -> tensor<4xf16> {
    %empty = tensor.empty() : tensor<4xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%empty : tensor<4xf16>) {
      ^bb0(%value: f16, %old: f16):
        %two = arith.constant 2.000000e+00 : f16
        %division = arith.divf %two, %value : f16
        linalg.yield %division : f16
    } -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = analyze(*module, &failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto domain =
      CardComputeImplementationDomain::create(*program, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardComputeImplementationAssignment natural = domain->getFirstAssignment();
  auto end = domain->getNextAssignment(natural);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);
  CardComputeImplementationAssignment invalid = natural;
  invalid.nodes.front().implementation =
      StructuredComputeImplementation::Reciprocal;
  EXPECT_FALSE(domain->contains(invalid));

  StructuredNodeShardGroup group;
  group.shards.push_back(
      StructuredNodeIterationShard{0,
                                   TileId(0),
                                   {0},
                                   {4},
                                   {}});
  group.implementations.push_back(
      {0, StructuredComputeImplementation::Reciprocal});
  mlir::OwningOpRef<mlir::ModuleOp> card;
  EXPECT_TRUE(mlir::failed(lowerStructuredNodeGroupsToCardModule(
      *module, CardId(0), llvm::ArrayRef<TileId>{TileId(0)},
      program->operationNodes, llvm::ArrayRef{group}, card, nullptr,
      &failureReason)));
  EXPECT_NE(failureReason.find("requires exact 1/x"), std::string::npos);
}

} // namespace
