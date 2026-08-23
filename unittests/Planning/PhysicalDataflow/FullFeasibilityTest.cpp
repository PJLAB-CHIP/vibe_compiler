//===- FullFeasibilityTest.cpp ---------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;
using namespace wafer::compiler::testing;

ParsedProgram parseElementwise(int64_t extent) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<2x)mlir"
         << extent << "x128xf16>, %rhs: tensor<2x" << extent
         << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
         << "    %out = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
         << R"mlir(    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs, %rhs : )mlir"
         << "tensor<2x" << extent << "x128xf16>, tensor<2x" << extent
         << "x128xf16>) outs(%out : tensor<2x" << extent << "x128xf16>) {\n"
         << R"mlir(      ^bb0(%a: f16, %b: f16, %old: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    })mlir"
         << " -> tensor<2x" << extent << "x128xf16>\n"
         << "    return %sum : tensor<2x" << extent << "x128xf16>\n"
         << "  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      stream.str(), mlir::ParserConfig(context.get()));
  return {std::move(context), std::move(module)};
}

frontend::FrontendProgramVerificationResult
elementwiseMetadata(int64_t extent) {
  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {2, extent, 128}),
                               boundary(1, {2, extent, 128})};
  program.distributedOutputs = {boundary(0, {2, extent, 128})};
  return program;
}

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}

struct CompletePrefix {
  std::unique_ptr<CardProgramAnalysis> analysis;
  std::optional<PhysicalDataflowPlanningProblem> problem;
  std::unique_ptr<PhysicalDataflowPlanningSession> session;
  std::optional<IncompletePlanningDomain> incomplete;
};

CompletePrefix
buildPrefix(mlir::ModuleOp module,
            const frontend::FrontendProgramVerificationResult &metadata,
            llvm::raw_ostream &diagnostics, std::string &failureReason) {
  CompletePrefix result;
  auto analysis =
      analyzeCardProgram(module, metadata, executionConfig(), diagnostics);
  if (mlir::failed(analysis))
    return result;
  result.analysis = std::move(*analysis);
  auto problem = PhysicalDataflowPlanningProblem::create(
      *result.analysis, CardId(0), analysis::IndexRelationLimits(),
      &failureReason);
  if (mlir::failed(problem))
    return result;
  result.problem.emplace(std::move(*problem));
  result.session =
      std::make_unique<PhysicalDataflowPlanningSession>(*result.problem);
  auto incomplete = result.session->getFirstIncompleteState(&failureReason);
  if (mlir::failed(incomplete))
    return result;
  result.incomplete.emplace(std::move(*incomplete));
  return result;
}

ScheduleDomainInput makeScheduleInput(const ScheduledState &state,
                                      const EventGraph &graph) {
  ScheduleDomainInput input;
  input.structure = state.getExecutionStructurePlan();
  input.buffers = state.getBufferPlan();
  input.events.assign(graph.getEvents().begin(), graph.getEvents().end());
  input.hardDependencies.assign(graph.getHardDependencies().begin(),
                                graph.getHardDependencies().end());
  input.orderChoices.assign(graph.getOrderChoices().begin(),
                            graph.getOrderChoices().end());
  input.completionObligations.assign(graph.getCompletionObligations().begin(),
                                     graph.getCompletionObligations().end());
  input.resourceUses.assign(graph.getResourceUses().begin(),
                            graph.getResourceUses().end());
  input.components.assign(graph.getComponents().begin(),
                          graph.getComponents().end());
  return input;
}

TEST(FullFeasibilityTest,
     RealAlignedAndRaggedCandidatesActualizeOnceAndRetainAcceptedResult) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    ParsedProgram parsed = parseElementwise(extent);
    ASSERT_TRUE(parsed.module);
    const std::string before = print(parsed.module->getOperation());
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    auto metadata = elementwiseMetadata(extent);
    CompletePrefix prefix =
        buildPrefix(*parsed.module, metadata, diagnostics, failureReason);
    ASSERT_TRUE(prefix.analysis) << failureReason;
    ASSERT_TRUE(prefix.problem) << failureReason;
    ASSERT_TRUE(prefix.session) << failureReason;
    ASSERT_TRUE(prefix.incomplete) << failureReason;
    ASSERT_EQ(prefix.incomplete->getRequiredCoordinate(),
              RequiredPlanningCoordinate::FullFeasibility);

    wafer::compiler::ProgramDataHandoff programData;
    FullFeasibilityStatistics statistics;
    FullFeasibilityResult evaluated = prefix.session->evaluateScheduledState(
        *parsed.module, prefix.incomplete->getState(), metadata,
        executionConfig(), diagnostics, programData, &statistics,
        /*tilePipelineParallelism=*/0,
        /*captureTileDataflowIRTrace=*/true);
    ASSERT_TRUE(evaluated.isAccepted()) << evaluated.detail;
    EXPECT_EQ(statistics.evaluations, 1u);
    EXPECT_EQ(statistics.candidateActualizations, 1u);
    EXPECT_EQ(statistics.executableGateInvocations, 1u);
    ASSERT_TRUE(evaluated.compilation);
    EXPECT_EQ(evaluated.compilation->tileDataflowIRTrace.size(), 16u);
    EXPECT_EQ(print(parsed.module->getOperation()), before);
    CardExecutableLoweringResult executable = evaluated.takeExecutable();
    expectCompleteTileDomain(executable,
                             evaluated.compilation->tileDataflowIRTrace);
  }
}

TEST(FullFeasibilityTest,
     RaggedOverfullCandidateReturnsOwnedActualCapacityWitness) {
  ParsedProgram parsed = parseLargeTemporalProgram();
  ASSERT_TRUE(parsed.module);
  const std::string before = print(parsed.module->getOperation());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  CompletePrefix prefix =
      buildPrefix(*parsed.module, largeTemporalProgramMetadata(), diagnostics,
                  failureReason);
  ASSERT_TRUE(prefix.incomplete) << failureReason;
  std::vector<std::vector<SemanticRootKey>> causalRoots;
  std::vector<std::string> gates;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    wafer::compiler::ProgramDataHandoff programData;
    FullFeasibilityStatistics statistics;
    FullFeasibilityResult evaluated = prefix.session->evaluateScheduledState(
        *parsed.module, prefix.incomplete->getState(),
        largeTemporalProgramMetadata(), executionConfig(), diagnostics,
        programData, &statistics, /*tilePipelineParallelism=*/0,
        /*captureTileDataflowIRTrace=*/false);
    ASSERT_TRUE(evaluated.isExactRejection()) << evaluated.detail;
    EXPECT_EQ(statistics.candidateActualizations, 1u);
    EXPECT_EQ(statistics.executableGateInvocations, 1u);
    EXPECT_FALSE(evaluated.causalRoots.empty());
    ASSERT_TRUE(evaluated.compilation);
    EXPECT_TRUE(evaluated.compilation->isProvenExactRejection());
    EXPECT_TRUE(llvm::all_of(evaluated.compilation->tileFailures,
                             [](const CardExecutableTileFailure &failure) {
                               return isProvenExactTileMemoryPlanningFailure(
                                   failure.memoryPlanning);
                             }));
    causalRoots.push_back(evaluated.causalRoots);
    gates.push_back(evaluated.compilation->gate);
  }
  EXPECT_EQ(causalRoots[0], causalRoots[1]);
  EXPECT_EQ(gates[0], gates[1]);
  EXPECT_EQ(print(parsed.module->getOperation()), before);
}

TEST(
    FullFeasibilityTest,
    RepeatedCompleteStateProducesTheSameFreshActualResultWithoutSourceMutation) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  const std::string before = print(parsed.module->getOperation());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  CompletePrefix prefix = buildPrefix(*parsed.module, programMetadata(),
                                      diagnostics, failureReason);
  ASSERT_TRUE(prefix.incomplete) << failureReason;

  std::vector<std::string> traces;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    wafer::compiler::ProgramDataHandoff programData;
    FullFeasibilityStatistics statistics;
    FullFeasibilityResult evaluated = prefix.session->evaluateScheduledState(
        *parsed.module, prefix.incomplete->getState(), programMetadata(),
        executionConfig(), diagnostics, programData, &statistics,
        /*tilePipelineParallelism=*/1,
        /*captureTileDataflowIRTrace=*/true);
    ASSERT_TRUE(evaluated.isAccepted()) << evaluated.detail;
    ASSERT_TRUE(evaluated.compilation);
    EXPECT_EQ(statistics.candidateActualizations, 1u);
    std::string joined;
    for (const std::string &trace : evaluated.compilation->tileDataflowIRTrace)
      joined += trace;
    traces.push_back(std::move(joined));
  }
  ASSERT_EQ(traces.size(), 2u);
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(print(parsed.module->getOperation()), before);
  EXPECT_EQ(prefix.session->getWork().fullFeasibilityEvaluations, 2u);
  EXPECT_EQ(prefix.session->getWork().candidateActualizations, 2u);
}

TEST(FullFeasibilityTest,
     UnsupportedSelectedScheduleReturnsBeforeCandidateActualization) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  const std::string before = print(parsed.module->getOperation());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  CompletePrefix prefix = buildPrefix(*parsed.module, programMetadata(),
                                      diagnostics, failureReason);
  ASSERT_TRUE(prefix.incomplete) << failureReason;
  ScheduleDomainResult domain = buildScheduleDomain(makeScheduleInput(
      prefix.incomplete->getState(), prefix.incomplete->getEventGraph()));
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ScheduleSuccessor first = domain.domain->getFirstPlan();
  ASSERT_NE(first.getCursor(), nullptr);
  ScheduleSuccessor next = domain.domain->getNextPlan(*first.getCursor());
  ASSERT_EQ(next.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_NE(next.getPlan(), nullptr);
  auto alternative = ScheduledState::create(
      *domain.domain, prefix.incomplete->getState().getBufferState(),
      *next.getPlan(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(alternative)) << failureReason;

  wafer::compiler::ProgramDataHandoff programData;
  FullFeasibilityStatistics statistics;
  FullFeasibilityResult evaluated = prefix.session->evaluateScheduledState(
      *parsed.module, *alternative, programMetadata(), executionConfig(),
      diagnostics, programData, &statistics);
  EXPECT_EQ(evaluated.status, FullFeasibilityStatus::Unsupported);
  EXPECT_EQ(statistics.evaluations, 1u);
  EXPECT_EQ(statistics.candidateActualizations, 0u);
  EXPECT_EQ(statistics.executableGateInvocations, 0u);
  EXPECT_FALSE(evaluated.compilation.has_value());
  EXPECT_EQ(print(parsed.module->getOperation()), before);
}

} // namespace
