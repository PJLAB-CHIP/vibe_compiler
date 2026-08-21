//===- CanonicalBaselinePlanTest.cpp ---------------------------------===//

#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class CanonicalBaselinePlanTest : public ::testing::Test {
protected:
  CanonicalBaselinePlanTest() {
    registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static std::string withCardFacts(std::string source) {
    size_t insertion = source.find("module {");
    EXPECT_NE(insertion, std::string::npos);
    insertion += std::string("module {").size();
    source.insert(insertion, R"mlir(
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir");
    return source;
  }

  static std::string mapSource(int64_t extent) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
module {
  func.func @main(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.map ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent
           << "x128xf16>) (%value: f16) {\n"
           << "      %next = arith.addf %value, %value : f16\n"
           << "      linalg.yield %next : f16\n"
           << "    }\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return withCardFacts(source);
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::FailureOr<std::unique_ptr<CardProgramAnalysis>>
  analyze(mlir::ModuleOp module, std::string *failureReason) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    auto dag = StructuredDAGAnalysis::create(function, failureReason);
    if (mlir::failed(dag))
      return mlir::failure();
    auto topology = TargetTopology::create(module, failureReason);
    if (mlir::failed(topology))
      return mlir::failure();
    StaticOutputDomains outputs;
    for (mlir::Type type : function.getResultTypes()) {
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!tensor || !tensor.hasStaticShape())
        return mlir::failure();
      outputs.emplace_back(tensor.getShape().begin(), tensor.getShape().end());
    }
    llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
    for (const StructuredDAGNode &node : dag->getNodes())
      operationNodes.push_back({node.operation, node.id});
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return std::make_unique<CardProgramAnalysis>(
        std::move(*topology), std::move(tiles), std::move(*dag),
        std::move(outputs), std::move(operationNodes));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalBaselinePlanTest, OrdinaryAlignedAndRaggedPlansCloseWithoutIR) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(mapSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto program = analyze(*module, &failureReason);
    ASSERT_TRUE(mlir::succeeded(program)) << failureReason;
    auto plan = buildCanonicalBaselinePlan(**program, getTargetMemoryPolicy(),
                                           &failureReason);
    ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
    EXPECT_FALSE(plan->spatial.nodes.empty());
    EXPECT_FALSE(plan->regions.groups.empty());
    EXPECT_FALSE(plan->temporal.scopes.empty());
    EXPECT_FALSE(plan->storage.plan.storageObjects.empty());
    EXPECT_FALSE(plan->schedule.plan.order.empty());
    EXPECT_TRUE(plan->attention.roots.empty());
    EXPECT_TRUE(plan->preparedAttention.work.roots.empty());
    EXPECT_EQ(plan->feasibility.coverage,
              FullFeasibilityCoverage::EveryPlannedResourceClosed);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalBaselinePlanTest,
       FlashAttentionAndDecodingPrepareOneResolvedPlan) {
  struct Case {
    std::string source;
    AttentionAlgorithm algorithm;
  };
  std::vector<Case> cases;
  cases.push_back(
      {withCardFacts(wafer::test::buildFlashAttentionPlanningFixture(
           /*queryExtent=*/1024, /*keyValueExtent=*/1024,
           /*withMask=*/false)),
       AttentionAlgorithm::FlashAttention});
  cases.push_back({withCardFacts(wafer::test::buildFlashDecodingPlanningFixture(
                       /*queryExtent=*/1025, /*keyValueExtent=*/1031)),
                   AttentionAlgorithm::FlashDecoding});
  for (const Case &testCase : cases) {
    auto module = parse(testCase.source);
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto program = analyze(*module, &failureReason);
    ASSERT_TRUE(mlir::succeeded(program)) << failureReason;
    auto plan = buildCanonicalBaselinePlan(**program, getTargetMemoryPolicy(),
                                           &failureReason);
    ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
    ASSERT_EQ(plan->attention.roots.size(), 1u);
    EXPECT_EQ(plan->attention.roots.front().algorithm, testCase.algorithm);
    EXPECT_EQ(plan->preparedAttention.work.roots.size(), 1u);
    EXPECT_FALSE(plan->feasibility.dependencyKey.attentionActions.empty());
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

} // namespace
