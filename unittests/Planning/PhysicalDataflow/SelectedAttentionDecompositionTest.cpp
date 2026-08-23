//===- SelectedAttentionDecompositionTest.cpp ------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
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

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct SelectedInputs {
  CanonicalAttentionWorkCoordinate attention;
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
    return SelectedInputs{*attention};
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

  static bool
  setKeyValueResidentTiles(AttentionWorkDescription &description,
                           const wafer::AttentionIterationRoles &roles,
                           int64_t maximumTile) {
    if (maximumTile <= 0 || roles.keyValueReduction.empty())
      return false;
    for (AttentionOperandDescription &operand : description.operands) {
      if (operand.exactDomain.getForm() != ExactIndexSetForm::BoxUnion ||
          operand.exactDomain.getBoxes().size() != 1 ||
          operand.indexingMap.getNumResults() !=
              operand.exactDomain.getBoxes().front().sizes.size())
        return false;
      StaticRectangularIndexSet box = operand.exactDomain.getBoxes().front();
      for (auto [dimension, expression] :
           llvm::enumerate(operand.indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!iterator)
          return false;
        if (!llvm::is_contained(roles.keyValueReduction,
                                iterator.getPosition()))
          continue;
        const int64_t exact = box.sizes[dimension];
        box.sizes[dimension] =
            std::min(maximumTile, std::max<int64_t>(1, exact / 2));
      }
      IndexSetResult set =
          IndexRelation::staticRectangularDomain(box.offsets, box.sizes);
      if (!set.isExact())
        return false;
      operand.residentDomain = ExactIndexSet(
          std::move(*set.set), ExactIndexSetForm::BoxUnion, {std::move(box)});
    }
    for (AttentionValueDescription &value : description.values) {
      if (value.exactDomain.getForm() != ExactIndexSetForm::BoxUnion ||
          value.exactDomain.getBoxes().size() != 1 ||
          value.indexingMap.getNumResults() !=
              value.exactDomain.getBoxes().front().sizes.size())
        return false;
      StaticRectangularIndexSet box = value.exactDomain.getBoxes().front();
      for (auto [dimension, expression] :
           llvm::enumerate(value.indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!iterator)
          return false;
        if (!llvm::is_contained(roles.keyValueReduction,
                                iterator.getPosition()))
          continue;
        const int64_t exact = box.sizes[dimension];
        box.sizes[dimension] =
            std::min(maximumTile, std::max<int64_t>(1, exact / 2));
      }
      IndexSetResult set =
          IndexRelation::staticRectangularDomain(box.offsets, box.sizes);
      if (!set.isExact())
        return false;
      value.residentDomain = ExactIndexSet(
          std::move(*set.set), ExactIndexSetForm::BoxUnion, {std::move(box)});
    }
    for (AttentionScratchDescription &scratch : description.scratch) {
      if (scratch.exactDomain.getForm() != ExactIndexSetForm::BoxUnion ||
          scratch.exactDomain.getBoxes().size() != 1 ||
          scratch.indexingMap.getNumResults() !=
              scratch.exactDomain.getBoxes().front().sizes.size())
        return false;
      StaticRectangularIndexSet box = scratch.exactDomain.getBoxes().front();
      for (auto [dimension, expression] :
           llvm::enumerate(scratch.indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!iterator)
          return false;
        if (!llvm::is_contained(roles.keyValueReduction,
                                iterator.getPosition()))
          continue;
        const int64_t exact = box.sizes[dimension];
        box.sizes[dimension] =
            std::min(maximumTile, std::max<int64_t>(1, exact / 2));
      }
      IndexSetResult set =
          IndexRelation::staticRectangularDomain(box.offsets, box.sizes);
      if (!set.isExact())
        return false;
      scratch.residentDomain = ExactIndexSet(
          std::move(*set.set), ExactIndexSetForm::BoxUnion, {std::move(box)});
    }
    return true;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(SelectedAttentionDecompositionTest,
       PrepareValidatesTheSelectedWorkDescriptionWithoutResourcePrediction) {
  auto module = parse(wafer::test::buildFlashAttentionPlanningFixture(
      /*queryExtent=*/1024, /*keyValueExtent=*/1024, /*withMask=*/false));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  PreparedAttentionDecompositionOutcome prepared =
      prepareSelectedAttentionDecomposition(inputs->attention);
  EXPECT_TRUE(std::holds_alternative<PreparedAttentionDecomposition>(prepared));

  CanonicalAttentionWorkCoordinate nonResident = inputs->attention;
  ASSERT_FALSE(nonResident.roots.front().values.empty());
  AttentionValueDescription &value = nonResident.roots.front().values.front();
  ASSERT_EQ(value.exactDomain.getBoxes().size(), 1u);
  StaticRectangularIndexSet outside = value.exactDomain.getBoxes().front();
  ASSERT_FALSE(outside.offsets.empty());
  outside.offsets.back() += outside.sizes.back();
  IndexSetResult outsideSet =
      IndexRelation::staticRectangularDomain(outside.offsets, outside.sizes);
  ASSERT_TRUE(outsideSet.isExact());
  value.residentDomain =
      ExactIndexSet(std::move(*outsideSet.set), ExactIndexSetForm::BoxUnion,
                    {std::move(outside)});
  PreparedAttentionDecompositionOutcome invalidResident =
      prepareSelectedAttentionDecomposition(nonResident);
  const auto *invalid = std::get_if<BrokenPreparedAttention>(&invalidResident);
  ASSERT_NE(invalid, nullptr);
  EXPECT_EQ(invalid->reason,
            BrokenPreparedAttentionReason::InvalidWorkDescription);

  CanonicalAttentionWorkCoordinate nonResidentOperand = inputs->attention;
  ASSERT_FALSE(nonResidentOperand.roots.front().operands.empty());
  AttentionOperandDescription &operand =
      nonResidentOperand.roots.front().operands.front();
  ASSERT_EQ(operand.exactDomain.getBoxes().size(), 1u);
  StaticRectangularIndexSet outsideOperand =
      operand.exactDomain.getBoxes().front();
  ASSERT_FALSE(outsideOperand.offsets.empty());
  outsideOperand.offsets.back() += outsideOperand.sizes.back();
  IndexSetResult outsideOperandSet = IndexRelation::staticRectangularDomain(
      outsideOperand.offsets, outsideOperand.sizes);
  ASSERT_TRUE(outsideOperandSet.isExact());
  operand.residentDomain =
      ExactIndexSet(std::move(*outsideOperandSet.set),
                    ExactIndexSetForm::BoxUnion, {std::move(outsideOperand)});
  PreparedAttentionDecompositionOutcome invalidOperandResident =
      prepareSelectedAttentionDecomposition(nonResidentOperand);
  const auto *invalidOperand =
      std::get_if<BrokenPreparedAttention>(&invalidOperandResident);
  ASSERT_NE(invalidOperand, nullptr);
  EXPECT_EQ(invalidOperand->reason,
            BrokenPreparedAttentionReason::InvalidWorkDescription);
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
    std::set<mlir::Operation *> actionStructuredOperations;
    for (const AttentionActionMaterialization &action : materialized->actions) {
      EXPECT_FALSE(action.operations.empty());
      for (mlir::Operation *operation : action.structuredOperations)
        EXPECT_TRUE(actionStructuredOperations.insert(operation).second);
    }
    std::set<mlir::Operation *> scopeStructuredOperations;
    for (const AttentionScopeOperationMaterialization &scope :
         materialized->scopes)
      for (mlir::Operation *operation : scope.operations)
        EXPECT_TRUE(scopeStructuredOperations.insert(operation).second);
    EXPECT_EQ(actionStructuredOperations, scopeStructuredOperations);
    for (const AttentionValueMaterialization &value : materialized->values) {
      ASSERT_FALSE(value.occurrences.empty());
      for (mlir::Value occurrence : value.occurrences) {
        auto type =
            mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
        ASSERT_TRUE(type);
        if (value.id.kind == AttentionValueKind::ScoreBlock)
          EXPECT_LT(type.getNumElements(), static_cast<int64_t>(2) *
                                               testCase.queryExtent *
                                               testCase.keyValueExtent);
      }
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
       TemporalK2BlocksCarryOnlineStateAcrossAlignedAndRaggedFAFD) {
  struct Case {
    bool decoding;
    int64_t queryExtent;
    int64_t keyValueExtent;
  };
  for (const Case testCase : {Case{false, 1024, 1024}, Case{false, 1025, 1031},
                              Case{true, 1024, 1024}, Case{true, 1025, 1031}}) {
    SCOPED_TRACE(testCase.decoding);
    SCOPED_TRACE(testCase.keyValueExtent);
    auto module = parse(
        testCase.decoding ? wafer::test::buildFlashDecodingPlanningFixture(
                                testCase.queryExtent, testCase.keyValueExtent)
                          : wafer::test::buildFlashAttentionPlanningFixture(
                                testCase.queryExtent, testCase.keyValueExtent,
                                /*withMask=*/testCase.keyValueExtent == 1031));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    ASSERT_EQ(inputs->attention.roots.size(), 1u);
    auto sourceAttention = attention(*module);
    auto roles = sourceAttention.getIterationRoles();
    ASSERT_TRUE(mlir::succeeded(roles));
    AttentionWorkDescription description = inputs->attention.roots.front();
    ASSERT_TRUE(setKeyValueResidentTiles(description, *roles,
                                         /*maximumTile=*/512));

    mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
    mlir::IRRewriter rewriter(context.get());
    auto materialized = emitSelectedAttentionDecomposition(
        rewriter, attention(*selected), description, &failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*selected)));
    EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
    unsigned stateUpdates = 0;
    for (const AttentionActionMaterialization &action : materialized->actions)
      if (action.id.kind == AttentionActionKind::StateUpdate) {
        ++stateUpdates;
        EXPECT_GT(action.operations.size(), 3u);
      }
    EXPECT_GT(stateUpdates, 0u);

    unsigned scoreValueIds = 0;
    size_t expectedCompletionCount = 0;
    for (const AttentionValueMaterialization &materializedValue :
         materialized->values) {
      if (materializedValue.id.kind != AttentionValueKind::ScoreBlock)
        continue;
      ++scoreValueIds;
      auto expected = llvm::find_if(
          description.values, [&](const AttentionValueDescription &value) {
            return value.id == materializedValue.id;
          });
      ASSERT_NE(expected, description.values.end());
      ASSERT_EQ(expected->exactDomain.getBoxes().size(), 1u);
      ASSERT_EQ(expected->residentDomain.getBoxes().size(), 1u);
      const auto &exact = expected->exactDomain.getBoxes().front();
      const auto &resident = expected->residentDomain.getBoxes().front();
      ASSERT_EQ(exact.sizes.size(), expected->indexingMap.getNumResults());
      ASSERT_EQ(resident.sizes.size(), expected->indexingMap.getNumResults());
      unsigned keyValueDimension = std::numeric_limits<unsigned>::max();
      for (auto [dimension, expression] :
           llvm::enumerate(expected->indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        ASSERT_TRUE(iterator);
        if (llvm::is_contained(roles->keyValueReduction,
                               iterator.getPosition()))
          keyValueDimension = dimension;
      }
      ASSERT_NE(keyValueDimension, std::numeric_limits<unsigned>::max());
      const int64_t exactExtent = exact.sizes[keyValueDimension];
      const int64_t residentExtent = resident.sizes[keyValueDimension];
      ASSERT_LT(residentExtent, exactExtent);
      const size_t expectedBlocks =
          (exactExtent + residentExtent - 1) / residentExtent;
      expectedCompletionCount += expectedBlocks;
      EXPECT_EQ(materializedValue.occurrences.size(), expectedBlocks);
      for (mlir::Value occurrence : materializedValue.occurrences) {
        auto type =
            mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
        ASSERT_TRUE(type);
        EXPECT_LE(type.getDimSize(keyValueDimension), residentExtent);
        EXPECT_LT(type.getNumElements(),
                  std::accumulate(exact.sizes.begin(), exact.sizes.end(),
                                  int64_t{1}, std::multiplies<int64_t>()));
      }
    }
    EXPECT_GT(scoreValueIds, 0u);
    EXPECT_EQ(materialized->scratch.size(), description.scratch.size());
    EXPECT_EQ(count<wafer::TensorCompletionOp>(*selected),
              expectedCompletionCount);
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
    ASSERT_FALSE(value.occurrences.empty());
    rankFourScores +=
        value.id.kind == AttentionValueKind::ScoreBlock &&
        llvm::any_of(value.occurrences, [](mlir::Value occurrence) {
          auto type =
              mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
          return type && type.getRank() == 4;
        });
  }
  EXPECT_EQ(rankFourScores, 16u);
}

TEST_F(SelectedAttentionDecompositionTest,
       MultiKeyValueAxesCarryStateThroughCartesianTailBlocks) {
  auto module = parse(wafer::test::buildMultiK2FlashDecodingPlanningFixture());
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  ASSERT_EQ(inputs->attention.roots.size(), 1u);
  auto roles = attention(*module).getIterationRoles();
  ASSERT_TRUE(mlir::succeeded(roles));
  ASSERT_EQ(roles->keyValueReduction.size(), 2u);
  AttentionWorkDescription description = inputs->attention.roots.front();
  ASSERT_TRUE(
      setKeyValueResidentTiles(description, *roles, /*maximumTile=*/16));

  mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
  mlir::IRRewriter rewriter(context.get());
  auto materialized = emitSelectedAttentionDecomposition(
      rewriter, attention(*selected), description, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*selected)));
  unsigned multiAxisScores = 0;
  for (const AttentionValueMaterialization &materializedValue :
       materialized->values) {
    if (materializedValue.id.kind != AttentionValueKind::ScoreBlock)
      continue;
    auto expected = llvm::find_if(description.values,
                                  [&](const AttentionValueDescription &value) {
                                    return value.id == materializedValue.id;
                                  });
    ASSERT_NE(expected, description.values.end());
    const auto &resident = expected->residentDomain.getBoxes().front();
    for (mlir::Value occurrence : materializedValue.occurrences) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
      ASSERT_TRUE(type);
      for (auto [dimension, expression] :
           llvm::enumerate(expected->indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        ASSERT_TRUE(iterator);
        if (!llvm::is_contained(roles->keyValueReduction,
                                iterator.getPosition()))
          continue;
        EXPECT_LE(type.getDimSize(dimension), resident.sizes[dimension]);
        ++multiAxisScores;
      }
    }
  }
  EXPECT_GT(multiAxisScores, 0u);
}

TEST_F(SelectedAttentionDecompositionTest,
       AdjacentOperandPiecesAssembleOneDenseSelectedOperand) {
  auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
      /*queryExtent=*/1025, /*keyValueExtent=*/1031));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  AttentionWorkDescription &description = inputs->attention.roots.front();
  auto operand = llvm::find_if(
      description.operands, [](const AttentionOperandDescription &candidate) {
        return candidate.role == AttentionOperandRole::Key &&
               candidate.exactDomain.getForm() == ExactIndexSetForm::BoxUnion &&
               candidate.exactDomain.getBoxes().size() == 1;
      });
  ASSERT_NE(operand, description.operands.end());
  StaticRectangularIndexSet original = operand->exactDomain.getBoxes().front();
  auto splitDimension =
      llvm::find_if(original.sizes, [](int64_t extent) { return extent > 1; });
  ASSERT_NE(splitDimension, original.sizes.end());
  unsigned dimension = splitDimension - original.sizes.begin();
  int64_t firstExtent = original.sizes[dimension] / 2;
  llvm::SmallVector<int64_t, 4> firstSizes(original.sizes);
  llvm::SmallVector<int64_t, 4> secondOffsets(original.offsets);
  llvm::SmallVector<int64_t, 4> secondSizes(original.sizes);
  firstSizes[dimension] = firstExtent;
  secondOffsets[dimension] += firstExtent;
  secondSizes[dimension] -= firstExtent;
  IndexSetResult first =
      IndexRelation::staticRectangularDomain(original.offsets, firstSizes);
  IndexSetResult second =
      IndexRelation::staticRectangularDomain(secondOffsets, secondSizes);
  ASSERT_TRUE(first.isExact() && second.isExact());
  mlir::presburger::PresburgerSet combined = std::move(*first.set);
  combined.unionInPlace(*second.set);
  operand->exactDomain =
      ExactIndexSet(std::move(combined), ExactIndexSetForm::BoxUnion,
                    {StaticRectangularIndexSet{original.offsets, firstSizes},
                     StaticRectangularIndexSet{secondOffsets, secondSizes}});

  mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
  mlir::IRRewriter rewriter(context.get());
  auto materialized = emitSelectedAttentionDecomposition(
      rewriter, attention(*selected), description, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
  EXPECT_GT(count<mlir::tensor::InsertSliceOp>(*selected), 0u);
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
      prepareSelectedAttentionDecomposition(inputs->attention);
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
  AttentionOperandDescription &last = malformed.operands.back();
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
