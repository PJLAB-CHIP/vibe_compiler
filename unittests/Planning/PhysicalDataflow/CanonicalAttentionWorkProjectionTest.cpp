//===- CanonicalAttentionWorkProjectionTest.cpp ----------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
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
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct AttentionProjectionInputs {
  wafer::test::CanonicalPlanningPrefix prefix;
  CanonicalRepresentationCoordinate representations;
  CanonicalMovementCoordinate movements;
  CanonicalStorageCoordinate storage;
  CanonicalScheduleCoordinate schedule;
};

class CanonicalAttentionWorkProjectionTest : public ::testing::Test {
protected:
  CanonicalAttentionWorkProjectionTest() {
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

  static std::string attentionSource(wafer::AttentionAlgorithm algorithm,
                                     int64_t queryExtent,
                                     int64_t keyValueExtent, bool withMask) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @attention(
      %query: tensor<2x)mlir"
           << queryExtent << "x128xf16>, %key: tensor<2x" << keyValueExtent
           << "x128xf16>,\n"
           << "      %value: tensor<2x" << keyValueExtent
           << "x64xf16>, %scale: f32";
    if (withMask)
      stream << ", %mask: tensor<" << queryExtent << "x" << keyValueExtent
             << "xf16>";
    stream << ") -> tensor<2x" << queryExtent << "x64xf16> {\n"
           << "    %out = tensor.empty() : tensor<2x" << queryExtent
           << "x64xf16>\n"
           << "    %result = wafer.linalg_ext.attention\n"
           << "        ins(%query, %key, %value, %scale";
    if (withMask)
      stream << ", %mask";
    stream << " : tensor<2x" << queryExtent << "x128xf16>, tensor<2x"
           << keyValueExtent << "x128xf16>, tensor<2x" << keyValueExtent
           << "x64xf16>, f32";
    if (withMask)
      stream << ", tensor<" << queryExtent << "x" << keyValueExtent << "xf16>";
    stream << ")\n"
           << "        outs(%out : tensor<2x" << queryExtent << "x64xf16>)\n"
           << "        algorithm(<"
           << (algorithm == wafer::AttentionAlgorithm::FlashAttention
                   ? "flash_attention"
                   : "flash_decoding")
           << ">)\n"
           << "        indexing_maps = [#q, #k, #v, #s";
    if (withMask)
      stream << ", #mask";
    stream << ", #o]\n"
           << "        -> tensor<2x" << queryExtent << "x64xf16>\n"
           << "    return %result : tensor<2x" << queryExtent << "x64xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static std::string multiK2Source() {
    return R"mlir(
#q = affine_map<(b, m, k1, k20, k21, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k20, k21, n) -> (b, k20, k21, k1)>
#v = affine_map<(b, m, k1, k20, k21, n) -> (b, k20, k21, n)>
#s = affine_map<(b, m, k1, k20, k21, n) -> ()>
#mask = affine_map<(b, m, k1, k20, k21, n) -> (m, k20, k21)>
#o = affine_map<(b, m, k1, k20, k21, n) -> (b, m, n)>
module {
  func.func @decode(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x33x31x128xf16>,
      %value: tensor<2x33x31x64xf16>, %scale: f32,
      %mask: tensor<1025x33x31xf16>) -> tensor<2x1025x64xf16> {
    %out = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale, %mask :
            tensor<2x1025x128xf16>, tensor<2x33x31x128xf16>,
            tensor<2x33x31x64xf16>, f32, tensor<1025x33x31xf16>)
        outs(%out : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
  }

  static std::string multiRootSource() {
    return R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @two_roots(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x1031x128xf16>,
      %value: tensor<2x1031x64xf16>, %scale: f32)
      -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>) {
    %left_out = tensor.empty() : tensor<2x1025x64xf16>
    %left = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%left_out : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    %right_out = tensor.empty() : tensor<2x1025x64xf16>
    %right = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%right_out : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %left, %right : tensor<2x1025x64xf16>, tensor<2x1025x64xf16>
  }
}
)mlir";
  }

  mlir::FailureOr<AttentionProjectionInputs>
  buildInputs(const StructuredDAGAnalysis &dag, std::string *failureReason) {
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
    return AttentionProjectionInputs{std::move(*prefix), *representations,
                                     *movements, *storage, *schedule};
  }

  static const AttentionValueDescription *
  findValue(const AttentionWorkDescription &description,
            const AttentionValueId &id) {
    auto value = llvm::find_if(
        description.values,
        [&](const AttentionValueDescription &entry) { return entry.id == id; });
    return value == description.values.end() ? nullptr : &*value;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalAttentionWorkProjectionTest,
       FlashAttentionAlignedAndRaggedDescribeOnlyBlockScratch) {
  struct Case {
    int64_t queryExtent;
    int64_t keyValueExtent;
    bool withMask;
  };
  for (const Case testCase :
       {Case{1024, 1024, false}, Case{1025, 1031, true}}) {
    SCOPED_TRACE(testCase.queryExtent);
    auto module = parse(attentionSource(
        wafer::AttentionAlgorithm::FlashAttention, testCase.queryExtent,
        testCase.keyValueExtent, testCase.withMask));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalAttentionWorkProjectionOutcome outcome =
        buildCanonicalAttentionWorkProjection(
            inputs->prefix.rootWorks, inputs->representations,
            inputs->movements, inputs->storage, inputs->schedule);
    const CanonicalAttentionWorkCoordinate *coordinate =
        getCanonicalAttentionWorkCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->roots.size(), 1u);
    const AttentionWorkDescription &description = coordinate->roots.front();
    EXPECT_EQ(description.algorithm, wafer::AttentionAlgorithm::FlashAttention);
    EXPECT_TRUE(description.gathers.empty());

    std::set<AttentionWorkScopeId> scopes;
    for (const AttentionActionDescription &action : description.actions)
      scopes.insert(action.id.scope);
    ASSERT_EQ(scopes.size(), 16u);
    EXPECT_EQ(description.actions.size(), 16u * 8u);
    EXPECT_EQ(description.values.size(), 16u * 10u);
    EXPECT_EQ(description.simultaneousValues.size(), 16u * 2u);
    EXPECT_EQ(description.operands.size(), 16u * (testCase.withMask ? 4u : 3u));
    for (const AttentionWorkScopeId &scope : scopes) {
      EXPECT_FALSE(scope.group.has_value());
      for (AttentionActionKind kind :
           {AttentionActionKind::QueryKeyContraction,
            AttentionActionKind::ScaleMask, AttentionActionKind::RowMaximum,
            AttentionActionKind::Exponential, AttentionActionKind::RowSum,
            AttentionActionKind::ValueContraction,
            AttentionActionKind::StateUpdate, AttentionActionKind::Finalize})
        EXPECT_EQ(llvm::count_if(description.actions,
                                 [&](const AttentionActionDescription &action) {
                                   return action.id.scope == scope &&
                                          action.id.kind == kind;
                                 }),
                  1u);
      for (AttentionValueKind kind :
           {AttentionValueKind::ScoreBlock,
            AttentionValueKind::ScaledMaskedScoreBlock,
            AttentionValueKind::ProbabilityBlock,
            AttentionValueKind::RunningMaximum, AttentionValueKind::RunningSum,
            AttentionValueKind::RunningAccumulator}) {
        const AttentionValueDescription *value =
            findValue(description, {scope, kind});
        ASSERT_NE(value, nullptr);
        EXPECT_FALSE(value->physicalVersion.has_value());
        EXPECT_FALSE(value->storage.has_value());
        ASSERT_EQ(value->exactDomain.getBoxes().size(), 1u);
        if (kind == AttentionValueKind::ScoreBlock)
          EXPECT_EQ(value->exactDomain.getBoxes().front().sizes.back(),
                    testCase.keyValueExtent);
      }
      const AttentionValueDescription *output =
          findValue(description, {scope, AttentionValueKind::FinalOutput});
      ASSERT_NE(output, nullptr);
      EXPECT_TRUE(output->physicalVersion.has_value());
      EXPECT_TRUE(output->storage.has_value());
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalAttentionWorkProjectionTest,
       FlashDecodingProjectsEveryRemoteAndLocalComponent) {
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
    CanonicalAttentionWorkProjectionOutcome outcome =
        buildCanonicalAttentionWorkProjection(
            inputs->prefix.rootWorks, inputs->representations,
            inputs->movements, inputs->storage, inputs->schedule);
    const CanonicalAttentionWorkCoordinate *coordinate =
        getCanonicalAttentionWorkCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->roots.size(), 1u);
    const AttentionWorkDescription &description = coordinate->roots.front();
    EXPECT_EQ(description.algorithm, wafer::AttentionAlgorithm::FlashDecoding);
    EXPECT_EQ(description.gathers.size(),
              inputs->movements.plan.reductionGathers.size());

    unsigned contributionScopes = 0;
    unsigned mergeScopes = 0;
    std::set<AttentionWorkScopeId> scopes;
    for (const AttentionActionDescription &action : description.actions)
      scopes.insert(action.id.scope);
    for (const AttentionWorkScopeId &scope : scopes) {
      if (std::holds_alternative<RequiredRootExecution>(
              scope.execution.source)) {
        ++contributionScopes;
        EXPECT_TRUE(scope.group.has_value());
        EXPECT_EQ(llvm::count_if(description.actions,
                                 [&](const AttentionActionDescription &action) {
                                   return action.id.scope == scope;
                                 }),
                  7u);
        EXPECT_EQ(llvm::count_if(description.actions,
                                 [&](const AttentionActionDescription &action) {
                                   return action.id.scope == scope &&
                                          action.id.kind ==
                                              AttentionActionKind::Finalize;
                                 }),
                  0u);
      } else {
        ++mergeScopes;
        EXPECT_EQ(llvm::count_if(description.actions,
                                 [&](const AttentionActionDescription &action) {
                                   return action.id.scope == scope;
                                 }),
                  2u);
      }
      for (AttentionValueKind kind :
           {AttentionValueKind::RunningMaximum, AttentionValueKind::RunningSum,
            AttentionValueKind::RunningAccumulator}) {
        const AttentionValueDescription *value =
            findValue(description, {scope, kind});
        ASSERT_NE(value, nullptr);
        EXPECT_TRUE(value->physicalVersion.has_value());
        EXPECT_TRUE(value->storage.has_value());
      }
    }
    EXPECT_EQ(contributionScopes, 16u);
    EXPECT_EQ(mergeScopes, inputs->prefix.demand.reductionMerges.size());
    for (const AttentionGatherProjection &gather : description.gathers) {
      EXPECT_TRUE(std::holds_alternative<RequiredRootExecution>(
          gather.value.scope.execution.source));
      EXPECT_TRUE(gather.value.scope.group.has_value());
      EXPECT_EQ(gather.value.scope.group, std::optional{gather.gather.group});
    }
  }
}

TEST_F(CanonicalAttentionWorkProjectionTest,
       MultiKeyValueAxesRemainAProjectedCartesianDomain) {
  auto module = parse(multiK2Source());
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  CanonicalAttentionWorkProjectionOutcome outcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          inputs->storage, inputs->schedule);
  const CanonicalAttentionWorkCoordinate *coordinate =
      getCanonicalAttentionWorkCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  ASSERT_EQ(coordinate->roots.size(), 1u);
  unsigned rankFourScores = 0;
  for (const AttentionValueDescription &value :
       coordinate->roots.front().values)
    rankFourScores += value.id.kind == AttentionValueKind::ScoreBlock &&
                      value.exactDomain.getRank() == 4;
  EXPECT_EQ(rankFourScores, 16u);
}

TEST_F(CanonicalAttentionWorkProjectionTest,
       MultipleRootsAndInputOrderProduceTheSameStableIdentities) {
  auto module = parse(multiRootSource());
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  CanonicalAttentionWorkProjectionOutcome outcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          inputs->storage, inputs->schedule);
  const CanonicalAttentionWorkCoordinate *coordinate =
      getCanonicalAttentionWorkCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  ASSERT_EQ(coordinate->roots.size(), 2u);

  std::reverse(inputs->prefix.rootWorks.begin(),
               inputs->prefix.rootWorks.end());
  std::reverse(inputs->representations.plan.primaryVersions.begin(),
               inputs->representations.plan.primaryVersions.end());
  std::reverse(inputs->representations.resources.begin(),
               inputs->representations.resources.end());
  std::reverse(inputs->movements.plan.externalLoads.begin(),
               inputs->movements.plan.externalLoads.end());
  std::reverse(inputs->movements.plan.ddrTransfers.begin(),
               inputs->movements.plan.ddrTransfers.end());
  std::reverse(inputs->movements.plan.publications.begin(),
               inputs->movements.plan.publications.end());
  std::reverse(inputs->storage.plan.versionBindings.begin(),
               inputs->storage.plan.versionBindings.end());
  std::reverse(inputs->storage.resources.begin(),
               inputs->storage.resources.end());
  std::reverse(inputs->schedule.plan.order.begin(),
               inputs->schedule.plan.order.end());
  std::reverse(inputs->schedule.plan.workerBindings.begin(),
               inputs->schedule.plan.workerBindings.end());
  CanonicalAttentionWorkProjectionOutcome reversed =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          inputs->storage, inputs->schedule);
  const CanonicalAttentionWorkCoordinate *reversedCoordinate =
      getCanonicalAttentionWorkCoordinate(reversed);
  ASSERT_NE(reversedCoordinate, nullptr);
  ASSERT_EQ(reversedCoordinate->roots.size(), coordinate->roots.size());
  for (auto [expected, actual] :
       llvm::zip_equal(coordinate->roots, reversedCoordinate->roots)) {
    EXPECT_EQ(actual.root, expected.root);
    EXPECT_EQ(actual.algorithm, expected.algorithm);
    EXPECT_EQ(
        llvm::map_to_vector<4>(
            actual.actions,
            [](const AttentionActionDescription &action) { return action.id; }),
        llvm::map_to_vector<4>(expected.actions,
                               [](const AttentionActionDescription &action) {
                                 return action.id;
                               }));
    EXPECT_EQ(
        llvm::map_to_vector<4>(
            actual.values,
            [](const AttentionValueDescription &value) { return value.id; }),
        llvm::map_to_vector<4>(
            expected.values,
            [](const AttentionValueDescription &value) { return value.id; }));
  }
}

TEST_F(CanonicalAttentionWorkProjectionTest,
       MissingComponentStorageGatherAndScheduleFactsFailClosed) {
  auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
      /*queryExtent=*/1025, /*keyValueExtent=*/1031));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;

  CanonicalRepresentationCoordinate missingVersion = inputs->representations;
  auto component =
      llvm::find_if(missingVersion.resources,
                    [](const RepresentationResourceDescription &resource) {
                      return std::holds_alternative<CoupledComponentValueId>(
                          resource.version.logicalValue);
                    });
  ASSERT_NE(component, missingVersion.resources.end());
  PhysicalVersionId removedVersion = component->version;
  missingVersion.resources.erase(component);
  CanonicalAttentionWorkProjectionOutcome missingVersionOutcome =
      buildCanonicalAttentionWorkProjection(inputs->prefix.rootWorks,
                                            missingVersion, inputs->movements,
                                            inputs->storage, inputs->schedule);
  const auto *missingVersionFailure =
      std::get_if<BrokenAttentionWorkProjection>(&missingVersionOutcome);
  ASSERT_NE(missingVersionFailure, nullptr);
  EXPECT_EQ(missingVersionFailure->reason,
            BrokenAttentionWorkProjectionReason::MissingPhysicalVersion);

  CanonicalRepresentationCoordinate mismatchedComponent =
      inputs->representations;
  auto mismatched =
      llvm::find_if(mismatchedComponent.resources,
                    [](const RepresentationResourceDescription &resource) {
                      return std::holds_alternative<CoupledComponentValueId>(
                          resource.version.logicalValue);
                    });
  ASSERT_NE(mismatched, mismatchedComponent.resources.end());
  mismatched->elementType =
      mismatched->elementType.isF32()
          ? mlir::Type(mlir::Float16Type::get(context.get()))
          : mlir::Type(mlir::Float32Type::get(context.get()));
  CanonicalAttentionWorkProjectionOutcome mismatchedComponentOutcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, mismatchedComponent, inputs->movements,
          inputs->storage, inputs->schedule);
  const auto *mismatchedComponentFailure =
      std::get_if<BrokenAttentionWorkProjection>(&mismatchedComponentOutcome);
  ASSERT_NE(mismatchedComponentFailure, nullptr);
  EXPECT_EQ(mismatchedComponentFailure->reason,
            BrokenAttentionWorkProjectionReason::ComponentMismatch);

  CanonicalStorageCoordinate missingStorage = inputs->storage;
  llvm::erase_if(missingStorage.plan.versionBindings,
                 [&](const PhysicalVersionStorageBinding &binding) {
                   return binding.version == removedVersion;
                 });
  CanonicalAttentionWorkProjectionOutcome missingStorageOutcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          missingStorage, inputs->schedule);
  const auto *missingStorageFailure =
      std::get_if<BrokenAttentionWorkProjection>(&missingStorageOutcome);
  ASSERT_NE(missingStorageFailure, nullptr);
  EXPECT_EQ(missingStorageFailure->reason,
            BrokenAttentionWorkProjectionReason::MissingStorageBinding);

  CanonicalStorageCoordinate missingStaging = inputs->storage;
  ASSERT_FALSE(missingStaging.plan.gatherStagingBindings.empty());
  missingStaging.plan.gatherStagingBindings.pop_back();
  CanonicalAttentionWorkProjectionOutcome missingStagingOutcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          missingStaging, inputs->schedule);
  const auto *missingStagingFailure =
      std::get_if<BrokenAttentionWorkProjection>(&missingStagingOutcome);
  ASSERT_NE(missingStagingFailure, nullptr);
  EXPECT_EQ(missingStagingFailure->reason,
            BrokenAttentionWorkProjectionReason::MovementMismatch);

  CanonicalScheduleCoordinate missingSchedule = inputs->schedule;
  auto executionNode =
      llvm::find_if(missingSchedule.plan.order, [](const ScheduleNodeId &node) {
        return std::holds_alternative<ExecutionInstanceId>(node);
      });
  ASSERT_NE(executionNode, missingSchedule.plan.order.end());
  ScheduleNodeId removedNode = *executionNode;
  missingSchedule.plan.order.erase(executionNode);
  llvm::erase_if(missingSchedule.plan.workerBindings,
                 [&](const ScheduleWorkerBinding &binding) {
                   return binding.node == removedNode;
                 });
  CanonicalAttentionWorkProjectionOutcome missingScheduleOutcome =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          inputs->storage, missingSchedule);
  const auto *missingScheduleFailure =
      std::get_if<BrokenAttentionWorkProjection>(&missingScheduleOutcome);
  ASSERT_NE(missingScheduleFailure, nullptr);
  EXPECT_EQ(missingScheduleFailure->reason,
            BrokenAttentionWorkProjectionReason::MissingScheduleNode);

  CanonicalAttentionWorkProjectionOutcome repeated =
      buildCanonicalAttentionWorkProjection(
          inputs->prefix.rootWorks, inputs->representations, inputs->movements,
          inputs->storage, inputs->schedule);
  EXPECT_NE(getCanonicalAttentionWorkCoordinate(repeated), nullptr);
}

} // namespace
