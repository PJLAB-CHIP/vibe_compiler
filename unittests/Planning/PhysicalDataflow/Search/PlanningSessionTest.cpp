//===- PlanningSessionTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

#include "TestSupport/Planning/SpatialPlanReference.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

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

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static std::unique_ptr<CardProgramAnalysis>
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
    return std::make_unique<CardProgramAnalysis>(
        std::move(*topology), std::move(tiles), std::move(*dag),
        std::move(outputs), std::move(operationNodes));
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static std::optional<std::vector<SpatialState>>
  exhaust(PhysicalDataflowPlanningSession &session, size_t limit = 10000) {
    std::vector<SpatialState> states;
    while (states.size() <= limit) {
      SpatialExpansionResult expansion = session.resumeSpatial();
      if (expansion.getKind() == SpatialExpansionKind::ParentExhausted)
        return states;
      if (expansion.getKind() == SpatialExpansionKind::Unsupported)
        continue;
      if (expansion.getKind() != SpatialExpansionKind::StateQueued)
        return std::nullopt;
      std::optional<SpatialState> state = session.takeNextSpatialState();
      if (!state)
        return std::nullopt;
      states.push_back(std::move(*state));
    }
    return std::nullopt;
  }

  static constexpr llvm::StringLiteral kRealSource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main(%input: tensor<2x2x1025x128xbf16>)
      -> tensor<2x2x1025x128xbf16> {
    %empty = tensor.empty() : tensor<2x2x1025x128xbf16>
    %result = linalg.map ins(%input : tensor<2x2x1025x128xbf16>)
        outs(%empty : tensor<2x2x1025x128xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    return %result : tensor<2x2x1025x128xbf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kTinySource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  func.func @main(%input: tensor<1x4x1xf16>) -> tensor<1x4x1xf16> {
    %empty = tensor.empty() : tensor<1x4x1xf16>
    %result = linalg.map ins(%input : tensor<1x4x1xf16>)
        outs(%empty : tensor<1x4x1xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<1x4x1xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kLimitedSource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main(%source: tensor<2x17x128xf16>,
                  %dest: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %inserted = tensor.insert_slice %source into %dest[0, 64, 0]
        [2, 17, 128] [1, 1, 1]
        : tensor<2x17x128xf16> into tensor<2x1025x128xf16>
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = linalg.map ins(%inserted : tensor<2x1025x128xf16>)
        outs(%empty : tensor<2x1025x128xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<2x1025x128xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kTinyChainSource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  func.func @main(%input: tensor<1x2x1xf16>) -> tensor<1x2x1xf16> {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %first = linalg.map ins(%input : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %second = linalg.map ins(%first : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %second : tensor<1x2x1xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kScalarSource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  func.func @main(%value: f16) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.fill ins(%value : f16) outs(%empty : tensor<f16>)
        -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kUnsupportedSource = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main(%input: tensor<2x1024x128xf16>, %condition: i1)
      -> tensor<2x1024x128xf16> {
    %selected = arith.select %condition, %input, %input
        : tensor<2x1024x128xf16>
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.map ins(%selected : tensor<2x1024x128xf16>)
        outs(%empty : tensor<2x1024x128xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<2x1024x128xf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(PlanningSessionTest,
       RealSpatialPrefixReturnsTypedIncompleteWithoutMutatingSource) {
  auto module = parse(kRealSource);
  ASSERT_TRUE(module);
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto problem = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
  PhysicalDataflowPlanningSession session(*problem);
  auto incomplete = session.getFirstIncompleteState(&failureReason);
  ASSERT_TRUE(mlir::succeeded(incomplete)) << failureReason;
  EXPECT_EQ(incomplete->getRequiredCoordinate(),
            RequiredPlanningCoordinate::PartialFeasibility);
  EXPECT_TRUE(problem->getSpatialDomain().contains(
      incomplete->getState().getSpatialPlan()));
  EXPECT_FALSE(incomplete->getState().getRegionPlan().groups.empty());
  EXPECT_TRUE(incomplete->hasRemainingSpatialWork());
  EXPECT_EQ(incomplete->getWork().spatialStatesQueued, 1u);
  EXPECT_EQ(incomplete->getWork().spatialDemandQueries, 1u);
  EXPECT_GT(incomplete->getWork().rootWorksValidated, 0u);
  EXPECT_EQ(incomplete->getWork().rootWorkSuccessorSteps,
            incomplete->getWork().rootWorksValidated + 1);
  EXPECT_EQ(incomplete->getWork().regionSuccessorSteps, 1u);
  EXPECT_EQ(incomplete->getWork().regionStatesQueued, 1u);
  EXPECT_EQ(incomplete->getWork().temporalSuccessorSteps, 1u);
  EXPECT_EQ(incomplete->getWork().temporalStatesQueued, 1u);
  EXPECT_FALSE(incomplete->getState().getTemporalPlan().scopes.empty());
  for (const TemporalScopePlan &scope :
       incomplete->getState().getTemporalPlan().scopes) {
    EXPECT_TRUE(isTopLevelScope(scope.id));
    EXPECT_TRUE(scope.waveLoopOrder.empty());
  }
  EXPECT_EQ(print(module->getOperation()), before);

  auto secondModule = parse(kRealSource);
  ASSERT_TRUE(secondModule);
  auto secondProgram = buildProgram(*secondModule, failureReason);
  ASSERT_TRUE(secondProgram) << failureReason;
  auto secondProblem = PhysicalDataflowPlanningProblem::create(
      *secondProgram, CardId(0), analysis::IndexRelationLimits(),
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(secondProblem)) << failureReason;
  PhysicalDataflowPlanningSession secondSession(*secondProblem);
  auto secondIncomplete = secondSession.getFirstIncompleteState(&failureReason);
  ASSERT_TRUE(mlir::succeeded(secondIncomplete)) << failureReason;
  EXPECT_EQ(secondIncomplete->getState(), incomplete->getState());
  EXPECT_NE(secondProgram->dag.getNodes().front().operation,
            program->dag.getNodes().front().operation);
}

TEST_F(PlanningSessionTest,
       TinyContinuationMatchesIndependentDomainAndIsDeterministic) {
  auto module = parse(kTinySource);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto problem = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
  const SpatialRootDomainFacts &root =
      problem->getSpatialDomain().getProblem().getRoots().front();
  wafer::test::ReferenceSpatialRoot reference;
  reference.root = root.root;
  reference.iteratorExtents = {1, 4, 1};
  reference.partitionableIterators = {1, 1, 1};
  reference.reductionIterators = {0, 0, 0};
  reference.resultParallelIterators = {1, 1, 1};
  std::set<SpatialPlan> expected = wafer::test::enumerateReferenceSpatialPlans(
      {reference}, problem->getSpatialDomain()
                       .getProblem()
                       .getStructuralProblem()
                       .getAvailableTiles());

  PhysicalDataflowPlanningSession first(*problem);
  auto firstStates = exhaust(first);
  ASSERT_TRUE(firstStates);
  std::set<SpatialPlan> actual;
  for (const SpatialState &state : *firstStates) {
    EXPECT_EQ(state.getRequiredCoordinate(),
              RequiredPlanningCoordinate::Region);
    actual.insert(state.getPlan());
  }
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(firstStates->size(), expected.size());
  EXPECT_EQ(first.getWork().spatialStatesQueued, expected.size());
  EXPECT_EQ(first.getWork().spatialDemandQueries, expected.size());
  EXPECT_GT(first.getWork().rootWorksValidated, expected.size());
  EXPECT_GT(first.getWork().duplicateSpatialChoices, 0u);
  EXPECT_TRUE(first.isSpatialExhausted());

  PhysicalDataflowPlanningSession second(*problem);
  auto secondStates = exhaust(second);
  ASSERT_TRUE(secondStates);
  ASSERT_EQ(secondStates->size(), firstStates->size());
  for (auto [lhs, rhs] : llvm::zip_equal(*firstStates, *secondStates))
    EXPECT_EQ(lhs, rhs);

  PhysicalDataflowPlanningSession queued(*problem);
  for (unsigned index = 0; index < 3; ++index)
    EXPECT_EQ(queued.resumeSpatial().getKind(),
              SpatialExpansionKind::StateQueued);
  std::vector<SpatialState> ordered;
  while (std::optional<SpatialState> state = queued.takeNextSpatialState())
    ordered.push_back(std::move(*state));
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(std::is_sorted(ordered.begin(), ordered.end()));
}

TEST_F(PlanningSessionTest,
       IndeterminateDemandPreservesTheCurrentSpatialChoice) {
  auto module = parse(kLimitedSource);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  analysis::IndexRelationLimits limits;
  limits.maxRectangularPieces = 1;
  auto problem = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), limits, &failureReason);
  ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
  PhysicalDataflowPlanningSession session(*problem);
  SpatialExpansionResult first = session.resumeSpatial();
  EXPECT_EQ(first.getKind(), SpatialExpansionKind::Indeterminate)
      << first.getDetail().str();
  EXPECT_FALSE(session.takeNextSpatialState());
  SpatialExpansionResult second = session.resumeSpatial();
  EXPECT_EQ(second.getKind(), SpatialExpansionKind::Indeterminate)
      << second.getDetail().str();
  EXPECT_EQ(first.getDetail(), second.getDetail());
  EXPECT_EQ(session.getWork().spatialSuccessorSteps, 2u);
  EXPECT_EQ(session.getWork().spatialDemandQueries, 2u);
  EXPECT_EQ(session.getWork().spatialStatesQueued, 0u);
  EXPECT_FALSE(session.isSpatialExhausted());
}

TEST_F(PlanningSessionTest, ScalarAndChainPrefixesRemainComplete) {
  {
    auto module = parse(kScalarSource);
    ASSERT_TRUE(module);
    std::string failureReason;
    auto program = buildProgram(*module, failureReason);
    ASSERT_TRUE(program) << failureReason;
    auto problem = PhysicalDataflowPlanningProblem::create(
        *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
    PhysicalDataflowPlanningSession session(*problem);
    auto states = exhaust(session);
    ASSERT_TRUE(states);
    ASSERT_EQ(states->size(), 2u);
    for (const SpatialState &state : *states) {
      EXPECT_TRUE(state.getPlan().nodes.front().axes.empty());
      EXPECT_EQ(state.getRequiredCoordinate(),
                RequiredPlanningCoordinate::Region);
    }
  }

  {
    auto module = parse(kTinyChainSource);
    ASSERT_TRUE(module);
    std::string failureReason;
    auto program = buildProgram(*module, failureReason);
    ASSERT_TRUE(program) << failureReason;
    auto problem = PhysicalDataflowPlanningProblem::create(
        *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
    llvm::SmallVector<wafer::test::ReferenceSpatialRoot, 2> roots;
    for (const SpatialRootDomainFacts &root :
         problem->getSpatialDomain().getProblem().getRoots()) {
      wafer::test::ReferenceSpatialRoot reference;
      reference.root = root.root;
      reference.iteratorExtents = {1, 2, 1};
      reference.partitionableIterators = {1, 1, 1};
      reference.reductionIterators = {0, 0, 0};
      reference.resultParallelIterators = {1, 1, 1};
      roots.push_back(std::move(reference));
    }
    std::set<SpatialPlan> expected =
        wafer::test::enumerateReferenceSpatialPlans(roots,
                                                    problem->getSpatialDomain()
                                                        .getProblem()
                                                        .getStructuralProblem()
                                                        .getAvailableTiles());
    PhysicalDataflowPlanningSession session(*problem);
    auto states = exhaust(session);
    ASSERT_TRUE(states);
    std::set<SpatialPlan> actual;
    for (const SpatialState &state : *states)
      actual.insert(state.getPlan());
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(states->size(), 16u);

    auto selected = llvm::find_if(*states, [](const SpatialState &state) {
      return llvm::all_of(state.getPlan().nodes,
                          [](const NodeSpatialPlan &node) {
                            return node.embedding.size() == 1 &&
                                   node.embedding.front() == TileId(0);
                          });
    });
    ASSERT_NE(selected, states->end());
    RegionContinuation continuation =
        session.createRegionContinuation(*selected);
    std::set<RegionPlan> regionPlans;
    while (true) {
      auto region = session.resumeRegion(continuation, &failureReason);
      ASSERT_TRUE(mlir::succeeded(region)) << failureReason;
      if (!*region)
        break;
      EXPECT_EQ((*region)->getRequiredCoordinate(),
                RequiredPlanningCoordinate::Temporal);
      regionPlans.insert((*region)->getRegionPlan());
    }
    EXPECT_TRUE(continuation.isExhausted());
    EXPECT_EQ(regionPlans.size(), 7u);

    RegionContinuation firstRegionContinuation =
        session.createRegionContinuation(*selected);
    auto firstRegion =
        session.resumeRegion(firstRegionContinuation, &failureReason);
    ASSERT_TRUE(mlir::succeeded(firstRegion)) << failureReason;
    ASSERT_TRUE(*firstRegion);
    TemporalContinuation temporalContinuation =
        session.createTemporalContinuation(**firstRegion);
    std::set<TemporalPlan> temporalPlans;
    while (true) {
      TemporalExpansionResult temporal =
          session.resumeTemporal(temporalContinuation);
      if (temporal.getKind() == TemporalExpansionKind::ParentExhausted)
        break;
      ASSERT_EQ(temporal.getKind(), TemporalExpansionKind::State)
          << temporal.getDetail().str();
      std::optional<TemporalState> state = temporal.takeState();
      ASSERT_TRUE(state);
      temporalPlans.insert(state->getTemporalPlan());
    }
    EXPECT_TRUE(temporalContinuation.isExhausted());
    EXPECT_EQ(temporalPlans.size(), 4u);
  }
}

TEST_F(PlanningSessionTest,
       UnsupportedSpatialChoiceLeavesItsSiblingContinuationReachable) {
  auto module = parse(kUnsupportedSource);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto problem = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
  PhysicalDataflowPlanningSession session(*problem);
  SpatialExpansionResult first = session.resumeSpatial();
  EXPECT_EQ(first.getKind(), SpatialExpansionKind::Unsupported)
      << first.getDetail().str();
  EXPECT_FALSE(session.takeNextSpatialState());
  SpatialExpansionResult second = session.resumeSpatial();
  EXPECT_EQ(second.getKind(), SpatialExpansionKind::Unsupported)
      << second.getDetail().str();
  EXPECT_FALSE(session.takeNextSpatialState());
  EXPECT_EQ(session.getWork().unsupportedSpatialChoices, 2u);
  EXPECT_EQ(session.getWork().spatialSuccessorSteps, 2u);
  EXPECT_TRUE(session.hasRemainingSpatialWork());
}

TEST_F(PlanningSessionTest, TypedOutcomeRoutingAndInvalidStateFailClosed) {
  EXPECT_EQ(classifySpatialChoiceOutcome(analysis::ExactDemandProof{}),
            SpatialChoiceOutcomeKind::Satisfied);
  EXPECT_EQ(
      classifySpatialChoiceOutcome(analysis::UnsupportedDemandSemantics{}),
      SpatialChoiceOutcomeKind::Unsupported);
  EXPECT_EQ(classifySpatialChoiceOutcome(analysis::DemandWorkLimitReached{}),
            SpatialChoiceOutcomeKind::Indeterminate);
  EXPECT_EQ(classifySpatialChoiceOutcome(analysis::BrokenDemandContract{}),
            SpatialChoiceOutcomeKind::CompilerBug);

  auto module = parse(kTinySource);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto program = buildProgram(*module, failureReason);
  ASSERT_TRUE(program) << failureReason;
  auto problem = PhysicalDataflowPlanningProblem::create(
      *program, CardId(0), analysis::IndexRelationLimits(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
  SpatialPlan invalid = problem->getSpatialDomain().getFirstPlan();
  invalid.nodes.front().embedding = {TileId(0), TileId(0)};
  EXPECT_TRUE(mlir::failed(
      SpatialState::create(*problem, std::move(invalid), &failureReason)));
  EXPECT_NE(failureReason.find("outside"), std::string::npos);
}

} // namespace
