//===- PlanningSessionTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningSession.h"

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
#include <optional>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class PlanningSessionTest : public ::testing::Test {
protected:
  PlanningSessionTest() {
    registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(uint64_t extent) {
    std::string source = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main(%input: tensor<2x2xEXTENTx128xbf16>)
      -> tensor<2x2xEXTENTx128xbf16> {
    %empty = tensor.empty() : tensor<2x2xEXTENTx128xbf16>
    %result = linalg.map ins(%input : tensor<2x2xEXTENTx128xbf16>)
        outs(%empty : tensor<2x2xEXTENTx128xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    return %result : tensor<2x2xEXTENTx128xbf16>
  }
}
)mlir";
    for (size_t offset = source.find("EXTENT"); offset != std::string::npos;) {
      const std::string replacement = std::to_string(extent);
      source.replace(offset, 6, replacement);
      offset = source.find("EXTENT", offset + replacement.size());
    }
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static std::unique_ptr<StructuredProgramAnalysis>
  buildProgram(mlir::ModuleOp module, std::string &failureReason) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    if (mlir::failed(dag))
      return nullptr;
    auto topology = TargetTopology::create(module, &failureReason);
    if (mlir::failed(topology))
      return nullptr;
    std::optional<llvm::ArrayRef<TileId>> available =
        topology->getAvailableTileIds(CardId(0));
    if (!available)
      return nullptr;
    StaticOutputDomains outputs;
    for (mlir::Type result : function.getResultTypes()) {
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(result);
      if (!tensor || !tensor.hasStaticShape())
        return nullptr;
      outputs.emplace_back(tensor.getShape().begin(), tensor.getShape().end());
    }
    llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
    for (const StructuredDAGNode &node : dag->getNodes())
      operationNodes.push_back({node.operation, node.id});
    llvm::SmallVector<TileId, 16> tiles(available->begin(), available->end());
    return std::make_unique<StructuredProgramAnalysis>(
        std::move(*topology), std::move(tiles), std::move(*dag),
        std::move(outputs), std::move(operationNodes));
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    stream.flush();
    return text;
  }

  static std::optional<SpatialState>
  takeFirstSpatial(PhysicalDataflowPlanningSession &session) {
    for (unsigned attempt = 0; attempt < 10000; ++attempt) {
      SpatialExpansionResult expansion = session.resumeSpatial();
      if (expansion.getKind() == SpatialExpansionKind::Unsupported)
        continue;
      if (expansion.getKind() != SpatialExpansionKind::StateQueued)
        return std::nullopt;
      return session.takeNextSpatialState();
    }
    return std::nullopt;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(PlanningSessionTest,
       RealScalePlanningStopsAtRegionChoiceBeforeCurrentIRMaterialization) {
  for (uint64_t extent : {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(extent);
    auto module = parse(extent);
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto program = buildProgram(*module, failureReason);
    ASSERT_TRUE(program) << failureReason;
    auto admission = PhysicalDataflowPlanningProblem::create(
        *program, CardId(0), analysis::IndexRelationLimits());
    auto *problem = std::get_if<PhysicalDataflowPlanningProblem>(&admission);
    ASSERT_NE(problem, nullptr);

    PhysicalDataflowPlanningSession session(*problem, 16);
    std::optional<SpatialState> spatial = takeFirstSpatial(session);
    ASSERT_TRUE(spatial);
    RegionContinuation region =
        session.createRegionContinuation(std::move(*spatial));
    auto regionState = session.resumeRegion(region, &failureReason);
    ASSERT_TRUE(mlir::succeeded(regionState)) << failureReason;
    ASSERT_TRUE(*regionState);
    EXPECT_FALSE((*regionState)->getRegionPlan().groups.empty());
    EXPECT_EQ(print(module->getOperation()), before);
    EXPECT_GT(session.getWork().spatialDemandQueries, 0u);
    EXPECT_GT(session.getWork().regionStatesQueued, 0u);
  }
}

TEST_F(PlanningSessionTest,
       IndependentSessionsProduceTheSameFirstStructuralChoice) {
  auto module = parse(1025);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto admission = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), analysis::IndexRelationLimits());
  auto *problem = std::get_if<PhysicalDataflowPlanningProblem>(&admission);
  ASSERT_NE(problem, nullptr);

  auto firstChoice = [&](PhysicalDataflowPlanningSession &session)
      -> std::optional<RegionState> {
    std::optional<SpatialState> spatial = takeFirstSpatial(session);
    if (!spatial)
      return std::nullopt;
    RegionContinuation region =
        session.createRegionContinuation(std::move(*spatial));
    auto regionState = session.resumeRegion(region, &failureReason);
    if (mlir::failed(regionState) || !*regionState)
      return std::nullopt;
    return std::move(**regionState);
  };

  PhysicalDataflowPlanningSession first(*problem, 16);
  PhysicalDataflowPlanningSession second(*problem, 16);
  std::optional<RegionState> lhs = firstChoice(first);
  std::optional<RegionState> rhs = firstChoice(second);
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  EXPECT_EQ(*lhs, *rhs);
  EXPECT_EQ(first.getWork().spatialSuccessorSteps,
            second.getWork().spatialSuccessorSteps);
}

TEST_F(PlanningSessionTest, UnsupportedDemandClosesChoicesInAnAdmittedDomain) {
  for (int64_t extent : {1024, 1025}) {
    auto module = parse(extent);
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    mlir::OpBuilder builder(&function.getBody().front(),
                            function.getBody().front().begin());
    auto input = function.getArgument(0);
    auto condition =
        builder.create<mlir::arith::ConstantIntOp>(function.getLoc(), 1, 1);
    auto unsupported = builder.create<mlir::arith::SelectOp>(
        function.getLoc(), condition, input, input);
    function.walk(
        [&](mlir::linalg::MapOp op) { op->setOperand(0, unsupported); });
    const auto before = print(module->getOperation());
    std::string detail;
    auto program = buildProgram(*module, detail);
    ASSERT_TRUE(program) << detail;
    auto admission =
        PhysicalDataflowPlanningProblem::create(*program, CardId(0));
    auto *problem = std::get_if<PhysicalDataflowPlanningProblem>(&admission);
    ASSERT_NE(problem, nullptr);
    PhysicalDataflowPlanningSession session(*problem, 16);
    for (unsigned query = 0; query < 2; ++query) {
      auto result = session.resumeSpatial();
      EXPECT_EQ(result.getKind(), SpatialExpansionKind::Unsupported);
    }
    EXPECT_EQ(session.getWork().unsupportedSpatialChoices, 2u);
    EXPECT_FALSE(session.takeNextSpatialState());
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

} // namespace
