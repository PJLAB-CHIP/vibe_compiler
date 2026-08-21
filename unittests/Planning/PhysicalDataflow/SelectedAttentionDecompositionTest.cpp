//===- SelectedAttentionDecompositionTest.cpp ------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalFeasibilityProof.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct SelectedInputs {
  CanonicalAttentionWorkCoordinate attention;
  FullFeasibilityProof proof;
};

class SelectedAttentionDecompositionTest : public ::testing::Test {
protected:
  SelectedAttentionDecompositionTest() {
    wafer::registerWaferCoreDialects(registry);
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

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::FailureOr<SelectedInputs> buildInputs(const StructuredDAGAnalysis &dag,
                                              std::string *failureReason) {
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(dag, allTiles(),
                                                            failureReason);
    if (mlir::failed(prefix))
      return mlir::failure();
    CanonicalRepresentationPlanOutcome representationOutcome =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representationOutcome);
    if (!representations)
      return mlir::failure();
    CanonicalMovementPlanOutcome movementOutcome = buildCanonicalMovementPlan(
        prefix->regions, *representations, prefix->rootWorks);
    const CanonicalMovementCoordinate *movements =
        getCanonicalMovementCoordinate(movementOutcome);
    if (!movements)
      return mlir::failure();
    CanonicalSerializedExecutionPlanOutcome serializedOutcome =
        buildCanonicalSerializedExecutionPlan(prefix->regions,
                                              prefix->temporal);
    const SerializedExecutionPlan *serialized =
        getSerializedExecutionPlan(serializedOutcome);
    if (!serialized)
      return mlir::failure();
    CanonicalStoragePlanOutcome storageOutcome =
        buildCanonicalStoragePlan(*representations, *movements, *serialized);
    const CanonicalStorageCoordinate *storage =
        getCanonicalStorageCoordinate(storageOutcome);
    if (!storage)
      return mlir::failure();
    CanonicalSchedulePlanOutcome scheduleOutcome =
        buildCanonicalSchedulePlan(*storage, *serialized);
    const CanonicalScheduleCoordinate *schedule =
        getCanonicalScheduleCoordinate(scheduleOutcome);
    if (!schedule)
      return mlir::failure();
    CanonicalAttentionWorkProjectionOutcome attentionOutcome =
        buildCanonicalAttentionWorkProjection(prefix->rootWorks,
                                              *representations, *movements,
                                              *storage, *schedule);
    const CanonicalAttentionWorkCoordinate *attention =
        getCanonicalAttentionWorkCoordinate(attentionOutcome);
    if (!attention)
      return mlir::failure();
    CanonicalFeasibilityOutcome feasibility = buildCanonicalFeasibilityProof(
        *storage, *movements, *schedule, *attention,
        wafer::getTargetMemoryPolicy());
    const CanonicalFeasibilityCoordinate *proof =
        getCanonicalFeasibilityCoordinate(feasibility);
    if (!proof)
      return mlir::failure();
    return SelectedInputs{*attention, proof->proof};
  }

  static wafer::LinalgExtAttentionOp attention(mlir::ModuleOp module) {
    wafer::LinalgExtAttentionOp result;
    module.walk([&](wafer::LinalgExtAttentionOp candidate) {
      if (!result)
        result = candidate;
    });
    return result;
  }

  template <typename OpT> static unsigned count(mlir::ModuleOp module) {
    unsigned result = 0;
    module.walk([&](OpT) { ++result; });
    return result;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(SelectedAttentionDecompositionTest,
       PrepareRequiresExactFullProofActionCoverage) {
  auto module = parse(wafer::test::buildFlashAttentionPlanningFixture(
      /*queryExtent=*/1024, /*keyValueExtent=*/1024, /*withMask=*/false));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  PreparedAttentionDecompositionOutcome prepared =
      prepareSelectedAttentionDecomposition(inputs->attention, inputs->proof);
  EXPECT_TRUE(std::holds_alternative<PreparedAttentionDecomposition>(prepared));

  FullFeasibilityProof missing = inputs->proof;
  ASSERT_FALSE(missing.dependencyKey.attentionActions.empty());
  missing.dependencyKey.attentionActions.pop_back();
  PreparedAttentionDecompositionOutcome rejected =
      prepareSelectedAttentionDecomposition(inputs->attention, missing);
  const auto *failure = std::get_if<BrokenPreparedAttention>(&rejected);
  ASSERT_NE(failure, nullptr);
  EXPECT_EQ(failure->reason,
            BrokenPreparedAttentionReason::ActionCoverageMismatch);
}

TEST_F(SelectedAttentionDecompositionTest,
       FlashAttentionAlignedAndRaggedEmitOnlyBlockSizedLinalg) {
  struct Case {
    int64_t queryExtent;
    int64_t keyValueExtent;
    bool withMask;
  };
  for (const Case testCase :
       {Case{1024, 1024, false}, Case{1025, 1031, true}}) {
    SCOPED_TRACE(testCase.queryExtent);
    auto module = parse(wafer::test::buildFlashAttentionPlanningFixture(
        testCase.queryExtent, testCase.keyValueExtent, testCase.withMask));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    ASSERT_EQ(inputs->attention.roots.size(), 1u);

    mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
    ASSERT_TRUE(selected);
    mlir::IRRewriter rewriter(context.get());
    auto materialized = emitSelectedAttentionDecomposition(
        rewriter, attention(*selected), inputs->attention.roots.front(),
        &failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
    EXPECT_EQ(materialized->actions.size(),
              inputs->attention.roots.front().actions.size());
    EXPECT_EQ(materialized->values.size(),
              inputs->attention.roots.front().values.size());
    for (const AttentionActionMaterialization &action : materialized->actions)
      EXPECT_FALSE(action.operations.empty());
    for (const AttentionValueMaterialization &value : materialized->values) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.value.getType());
      ASSERT_TRUE(type);
      if (value.id.kind == AttentionValueKind::ScoreBlock)
        EXPECT_LT(type.getNumElements(), static_cast<int64_t>(2) *
                                             testCase.queryExtent *
                                             testCase.keyValueExtent);
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(SelectedAttentionDecompositionTest,
       FlashDecodingEmitsContributionStatesAndOneFinalizePerMerge) {
  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
        queryExtent, keyValueExtent));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    ASSERT_EQ(inputs->attention.roots.size(), 1u);

    mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
    mlir::IRRewriter rewriter(context.get());
    auto materialized = emitSelectedAttentionDecomposition(
        rewriter, attention(*selected), inputs->attention.roots.front(),
        &failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
    EXPECT_EQ(materialized->actions.size(),
              inputs->attention.roots.front().actions.size());
    unsigned stateMerges = 0;
    unsigned finalizes = 0;
    for (const AttentionActionMaterialization &action : materialized->actions) {
      stateMerges += action.id.kind == AttentionActionKind::StateMerge;
      finalizes += action.id.kind == AttentionActionKind::Finalize;
    }
    unsigned expectedMerges = llvm::count_if(
        inputs->attention.roots.front().actions,
        [](const AttentionActionDescription &action) {
          return action.id.kind == AttentionActionKind::StateMerge;
        });
    EXPECT_EQ(stateMerges, expectedMerges);
    EXPECT_EQ(finalizes, expectedMerges);
  }
}

TEST_F(SelectedAttentionDecompositionTest,
       MultiKeyValueAxesEmitCartesianScoreBlocks) {
  auto module = parse(wafer::test::buildMultiK2FlashDecodingPlanningFixture());
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  ASSERT_EQ(inputs->attention.roots.size(), 1u);

  mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
  mlir::IRRewriter rewriter(context.get());
  auto materialized = emitSelectedAttentionDecomposition(
      rewriter, attention(*selected), inputs->attention.roots.front(),
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
  unsigned rankFourScores = 0;
  for (const AttentionValueMaterialization &value : materialized->values) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.value.getType());
    ASSERT_TRUE(type);
    rankFourScores +=
        value.id.kind == AttentionValueKind::ScoreBlock && type.getRank() == 4;
  }
  EXPECT_EQ(rankFourScores, 16u);
}

TEST_F(SelectedAttentionDecompositionTest,
       MultiplePreparedRootsMaterializeIndependently) {
  auto module = parse(wafer::test::buildTwoFlashAttentionPlanningFixture());
  ASSERT_TRUE(module);
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  PreparedAttentionDecompositionOutcome preparedOutcome =
      prepareSelectedAttentionDecomposition(inputs->attention, inputs->proof);
  const auto *prepared =
      std::get_if<PreparedAttentionDecomposition>(&preparedOutcome);
  ASSERT_NE(prepared, nullptr);
  ASSERT_EQ(prepared->work.roots.size(), 2u);

  mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
  llvm::SmallVector<wafer::LinalgExtAttentionOp, 2> selectedOps;
  selected->walk(
      [&](wafer::LinalgExtAttentionOp op) { selectedOps.push_back(op); });
  ASSERT_EQ(selectedOps.size(), prepared->work.roots.size());
  mlir::IRRewriter rewriter(context.get());
  for (auto [op, description] :
       llvm::zip_equal(selectedOps, prepared->work.roots)) {
    auto materialized = emitSelectedAttentionDecomposition(
        rewriter, op, description, &failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(materialized->root, description.root);
  }
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST_F(SelectedAttentionDecompositionTest,
       EmissionMismatchFailsBeforeMutatingTheSelectedOp) {
  auto fa = parse(wafer::test::buildFlashAttentionPlanningFixture(
      /*queryExtent=*/1024, /*keyValueExtent=*/1024, /*withMask=*/false));
  auto fd = parse(wafer::test::buildFlashDecodingPlanningFixture(
      /*queryExtent=*/1024, /*keyValueExtent=*/1024));
  ASSERT_TRUE(fa && fd);
  std::string failureReason;
  auto fdDag = StructuredDAGAnalysis::create(function(*fd), &failureReason);
  ASSERT_TRUE(mlir::succeeded(fdDag)) << failureReason;
  auto fdInputs = buildInputs(*fdDag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(fdInputs)) << failureReason;
  ASSERT_EQ(fdInputs->attention.roots.size(), 1u);
  const std::string before = print(fa->getOperation());
  mlir::IRRewriter rewriter(context.get());
  auto failed = emitSelectedAttentionDecomposition(
      rewriter, attention(*fa), fdInputs->attention.roots.front(),
      &failureReason);
  EXPECT_TRUE(mlir::failed(failed));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*fa), 1u);
  EXPECT_EQ(print(fa->getOperation()), before);
}

TEST_F(SelectedAttentionDecompositionTest,
       MidEmissionFailureIsConfinedToDisposableSelectedClone) {
  auto source = parse(wafer::test::buildFlashAttentionPlanningFixture(
      /*queryExtent=*/1025, /*keyValueExtent=*/1031, /*withMask=*/true));
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*source), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  AttentionWorkDescription malformed = inputs->attention.roots.front();
  ASSERT_FALSE(malformed.operands.empty());
  AttentionOperandProjection &last = malformed.operands.back();
  last.exactDomain = ExactIndexSet(last.exactDomain.getPresburgerSet(),
                                   ExactIndexSetForm::GeneralPresburger);

  mlir::OwningOpRef<mlir::ModuleOp> selected = source->clone();
  mlir::IRRewriter rewriter(context.get());
  auto failed = emitSelectedAttentionDecomposition(
      rewriter, attention(*selected), malformed, &failureReason);
  EXPECT_TRUE(mlir::failed(failed));
  EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 1u);
  EXPECT_GT(count<mlir::linalg::GenericOp>(*selected), 0u)
      << "earlier scopes may exist only in the disposable selected clone";
  EXPECT_EQ(print(source->getOperation()), before);
}

} // namespace
