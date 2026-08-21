//===- CanonicalFeasibilityProofTest.cpp ------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalFeasibilityProof.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
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
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct FeasibilityInputs {
  wafer::test::CanonicalPlanningPrefix prefix;
  CanonicalMovementCoordinate movements;
  CanonicalStorageCoordinate storage;
  CanonicalScheduleCoordinate schedule;
  CanonicalAttentionWorkCoordinate attention;
};

class CanonicalFeasibilityProofTest : public ::testing::Test {
protected:
  CanonicalFeasibilityProofTest() {
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

  static RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
    return std::visit([](const auto &source) { return source.work; },
                      execution.source);
  }

  static ExactIndexSet box(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
    IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
    EXPECT_TRUE(set.isExact());
    StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                        llvm::to_vector(sizes)};
    return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                         {rectangle});
  }

  static ExactIndexSet twoBoxUnion(llvm::ArrayRef<int64_t> firstOffsets,
                                   llvm::ArrayRef<int64_t> firstSizes,
                                   llvm::ArrayRef<int64_t> secondOffsets,
                                   llvm::ArrayRef<int64_t> secondSizes) {
    ExactIndexSet first = box(firstOffsets, firstSizes);
    ExactIndexSet second = box(secondOffsets, secondSizes);
    mlir::presburger::PresburgerSet combined = first.getPresburgerSet();
    combined.unionInPlace(second.getPresburgerSet());
    return ExactIndexSet(
        std::move(combined), ExactIndexSetForm::BoxUnion,
        {StaticRectangularIndexSet{llvm::to_vector(firstOffsets),
                                   llvm::to_vector(firstSizes)},
         StaticRectangularIndexSet{llvm::to_vector(secondOffsets),
                                   llvm::to_vector(secondSizes)}});
  }

  static std::string mapSource(int64_t extent) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @map(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {indexing_maps = [#id, #id],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  mlir::FailureOr<FeasibilityInputs>
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
    CanonicalAttentionWorkProjectionOutcome attentionOutcome =
        buildCanonicalAttentionWorkProjection(prefix->rootWorks,
                                              *representations, *movements,
                                              *storage, *schedule);
    const CanonicalAttentionWorkCoordinate *attention =
        getCanonicalAttentionWorkCoordinate(attentionOutcome);
    if (!attention)
      return mlir::failure();
    return FeasibilityInputs{std::move(*prefix), *movements, *storage,
                             *schedule, *attention};
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalFeasibilityProofTest,
       OrdinaryAlignedAndRaggedPlansProduceValidatedProofs) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(mapSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalFeasibilityOutcome outcome = buildCanonicalFeasibilityProof(
        inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
        wafer::getTargetMemoryPolicy());
    const CanonicalFeasibilityCoordinate *coordinate =
        getCanonicalFeasibilityCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    EXPECT_EQ(coordinate->problem.tileSPM.size(), 16u);
    EXPECT_EQ(coordinate->problem.ddrPayloads.size(), 32u);
    EXPECT_EQ(coordinate->problem.scheduleNodes.size(), 48u);
    EXPECT_TRUE(coordinate->problem.attentionActions.empty());
    EXPECT_EQ(coordinate->proof.coverage,
              FullFeasibilityCoverage::EveryPlannedResourceClosed);
    EXPECT_EQ(coordinate->proof.dependencyKey.scheduleNodes,
              coordinate->problem.scheduleNodes);
    EXPECT_EQ(coordinate->proof.dependencyKey.movements.size(),
              coordinate->problem.ddrPayloads.size());
    EXPECT_EQ(llvm::count_if(coordinate->proof.resourceProblems,
                             [](const ResourceProblemProof &proof) {
                               return proof.kind == ResourceProofKind::TileSPM;
                             }),
              16u);
    for (const TileSPMResourceProblem &problem : coordinate->problem.tileSPM) {
      EXPECT_FALSE(problem.demands.empty());
      EXPECT_EQ(problem.arenaBegin, wafer::getTargetMemoryPolicy().spmBase);
      EXPECT_EQ(problem.arenaEnd, wafer::getTargetMemoryPolicy().spmLimit);
    }

    std::reverse(inputs->storage.plan.storageObjects.begin(),
                 inputs->storage.plan.storageObjects.end());
    std::reverse(inputs->storage.resources.begin(),
                 inputs->storage.resources.end());
    std::reverse(inputs->storage.lifetimes.begin(),
                 inputs->storage.lifetimes.end());
    std::reverse(inputs->movements.resources.begin(),
                 inputs->movements.resources.end());
    std::reverse(inputs->schedule.plan.workerBindings.begin(),
                 inputs->schedule.plan.workerBindings.end());
    CanonicalFeasibilityOutcome reversed = buildCanonicalFeasibilityProof(
        inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
        wafer::getTargetMemoryPolicy());
    const CanonicalFeasibilityCoordinate *reversedCoordinate =
        getCanonicalFeasibilityCoordinate(reversed);
    ASSERT_NE(reversedCoordinate, nullptr);
    ASSERT_EQ(reversedCoordinate->problem.tileSPM.size(),
              coordinate->problem.tileSPM.size());
    for (auto [expected, actual] :
         llvm::zip_equal(coordinate->problem.tileSPM,
                         reversedCoordinate->problem.tileSPM)) {
      EXPECT_EQ(actual.tile, expected.tile);
      EXPECT_EQ(
          llvm::map_to_vector<4>(actual.demands,
                                 [](const FeasibilityResourceDemand &demand) {
                                   return demand.id;
                                 }),
          llvm::map_to_vector<4>(expected.demands,
                                 [](const FeasibilityResourceDemand &demand) {
                                   return demand.id;
                                 }));
      EXPECT_EQ(actual.conflicts, expected.conflicts);
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalFeasibilityProofTest,
       IndependentRootsShareNoHiddenResourceIdentity) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @two_maps(%lhs: tensor<2x1025x128xf16>,
                      %rhs: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %left_empty = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%lhs : tensor<2x1025x128xf16>)
        outs(%left_empty : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %right_empty = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%rhs : tensor<2x1025x128xf16>)
        outs(%right_empty : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    return %left, %right : tensor<2x1025x128xf16>,
                           tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  CanonicalFeasibilityOutcome outcome = buildCanonicalFeasibilityProof(
      inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
      wafer::getTargetMemoryPolicy());
  const CanonicalFeasibilityCoordinate *coordinate =
      getCanonicalFeasibilityCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  std::set<FeasibilityResourceId> resources;
  for (const TileSPMResourceProblem &problem : coordinate->problem.tileSPM)
    for (const FeasibilityResourceDemand &demand : problem.demands)
      EXPECT_TRUE(resources.insert(demand.id).second);
  EXPECT_EQ(resources.size(), inputs->storage.plan.storageObjects.size());
}

TEST_F(CanonicalFeasibilityProofTest,
       AttentionScratchAndPhysicalStateAreClosedExactlyOnce) {
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
    CanonicalFeasibilityOutcome outcome = buildCanonicalFeasibilityProof(
        inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
        wafer::getTargetMemoryPolicy());
    const CanonicalFeasibilityCoordinate *coordinate =
        getCanonicalFeasibilityCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(inputs->attention.roots.size(), 1u);
    EXPECT_EQ(coordinate->problem.attentionActions.size(),
              inputs->attention.roots.front().actions.size());

    std::set<FeasibilityResourceId> demandIds;
    unsigned scratchDemands = 0;
    for (const TileSPMResourceProblem &problem : coordinate->problem.tileSPM)
      for (const FeasibilityResourceDemand &demand : problem.demands) {
        EXPECT_TRUE(demandIds.insert(demand.id).second);
        scratchDemands +=
            std::holds_alternative<AttentionValueId>(demand.id.origin);
      }
    unsigned expectedScratch =
        llvm::count_if(inputs->attention.roots.front().values,
                       [](const AttentionValueDescription &value) {
                         return !value.physicalVersion.has_value();
                       });
    EXPECT_EQ(scratchDemands, expectedScratch);
    for (const AttentionValueDescription &value :
         inputs->attention.roots.front().values) {
      FeasibilityResourceId id{value.physicalVersion
                                   ? FeasibilityResourceOrigin{*value.storage}
                                   : FeasibilityResourceOrigin{value.id}};
      EXPECT_TRUE(demandIds.count(id));
    }
  }
}

TEST_F(CanonicalFeasibilityProofTest,
       FlashAttentionScratchFitsForAlignedAndRaggedMasks) {
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
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalFeasibilityOutcome outcome = buildCanonicalFeasibilityProof(
        inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
        wafer::getTargetMemoryPolicy());
    const CanonicalFeasibilityCoordinate *coordinate =
        getCanonicalFeasibilityCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(inputs->attention.roots.size(), 1u);
    EXPECT_EQ(inputs->attention.roots.front().algorithm,
              wafer::AttentionAlgorithm::FlashAttention);
    unsigned scratch = 0;
    for (const TileSPMResourceProblem &problem : coordinate->problem.tileSPM)
      scratch += llvm::count_if(
          problem.demands, [](const FeasibilityResourceDemand &demand) {
            return std::holds_alternative<AttentionValueId>(demand.id.origin);
          });
    EXPECT_EQ(scratch,
              llvm::count_if(inputs->attention.roots.front().values,
                             [](const AttentionValueDescription &value) {
                               return !value.physicalVersion.has_value();
                             }));
  }
}

TEST_F(CanonicalFeasibilityProofTest,
       MultiPieceAndRankZeroResourcesProduceFiniteProblems) {
  SemanticRootKey pieceRoot;
  RootRegionWorkId pieceWork{pieceRoot, TileId(0)};
  ExecutionInstanceId pieceExecution{
      RequiredRootExecution{pieceWork, LogicalShardId{pieceRoot, {0}}}};
  PhysicalVersionId pieceVersion{
      SupportRegionValueId{pieceWork, SupportValueId{}}};
  StorageObjectId pieceObject{StorageObjectOrigin{pieceVersion}};

  SemanticRootKey scalarRoot;
  scalarRoot.anchorIndex = 1;
  RootRegionWorkId scalarWork{scalarRoot, TileId(1)};
  ExecutionInstanceId scalarExecution{
      RequiredRootExecution{scalarWork, LogicalShardId{scalarRoot, {}}}};
  PhysicalVersionId scalarVersion{ExecutionResultValueId{scalarExecution, 0}};
  StorageObjectId scalarObject{StorageObjectOrigin{scalarVersion}};
  mlir::Type f16 = mlir::Float16Type::get(context.get());

  CanonicalStorageCoordinate storage;
  storage.plan.storageObjects = {{pieceObject, TileId(0)},
                                 {scalarObject, TileId(1)}};
  storage.plan.versionBindings = {{pieceVersion, pieceObject},
                                  {scalarVersion, scalarObject}};
  storage.resources = {
      {pieceObject,
       twoBoxUnion({0, 0, 0}, {1, 512, 128}, {0, 512, 0}, {1, 513, 128}), f16,
       wafer::MemLayout::Tensor},
      {scalarObject, box({}, {}), f16, wafer::MemLayout::Tensor}};
  storage.lifetimes = {{pieceObject, pieceExecution, {pieceExecution}},
                       {scalarObject, scalarExecution, {scalarExecution}}};
  CanonicalScheduleCoordinate schedule;
  schedule.plan.order = {pieceExecution, scalarExecution};
  schedule.plan.workerBindings = {{pieceExecution, wafer::NCCWorker::Worker0},
                                  {scalarExecution, wafer::NCCWorker::Worker0}};
  CanonicalFeasibilityOutcome outcome = buildCanonicalFeasibilityProof(
      storage, /*movements=*/{}, schedule, /*attention=*/{},
      wafer::getTargetMemoryPolicy());
  const CanonicalFeasibilityCoordinate *coordinate =
      getCanonicalFeasibilityCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  ASSERT_EQ(coordinate->problem.tileSPM.size(), 2u);
  auto pieceProblem = llvm::find_if(coordinate->problem.tileSPM,
                                    [](const TileSPMResourceProblem &problem) {
                                      return problem.tile == TileId(0);
                                    });
  auto scalarProblem = llvm::find_if(coordinate->problem.tileSPM,
                                     [](const TileSPMResourceProblem &problem) {
                                       return problem.tile == TileId(1);
                                     });
  ASSERT_NE(pieceProblem, coordinate->problem.tileSPM.end());
  ASSERT_NE(scalarProblem, coordinate->problem.tileSPM.end());
  ASSERT_EQ(pieceProblem->demands.size(), 1u);
  ASSERT_EQ(scalarProblem->demands.size(), 1u);
  EXPECT_GT(pieceProblem->demands.front().sizeBytes, 1025 * 128);
  EXPECT_GT(scalarProblem->demands.front().sizeBytes, 0);
}

TEST_F(CanonicalFeasibilityProofTest,
       HardSPMAndDDRLimitsProduceTypedExactRejections) {
  auto module = parse(mapSource(1025));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;

  wafer::TargetMemoryPolicy tinySPM = wafer::getTargetMemoryPolicy();
  tinySPM.spmLimit = tinySPM.spmBase + 1;
  CanonicalFeasibilityOutcome spmOutcome = buildCanonicalFeasibilityProof(
      inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
      tinySPM);
  const auto *spmRejection =
      std::get_if<CanonicalResourceRejection>(&spmOutcome);
  ASSERT_NE(spmRejection, nullptr);
  EXPECT_EQ(spmRejection->reason,
            CanonicalResourceRejectionReason::SPMCapacity);
  EXPECT_FALSE(spmRejection->resources.empty());
  EXPECT_TRUE(spmRejection->requiredBytes.has_value());
  EXPECT_GT(*spmRejection->requiredBytes, spmRejection->capacityBytes);

  wafer::TargetMemoryPolicy tinyDDR = wafer::getTargetMemoryPolicy();
  tinyDDR.ddrLargestContiguousBytes = 1;
  CanonicalFeasibilityOutcome ddrOutcome = buildCanonicalFeasibilityProof(
      inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
      tinyDDR);
  const auto *ddrRejection =
      std::get_if<CanonicalResourceRejection>(&ddrOutcome);
  ASSERT_NE(ddrRejection, nullptr);
  EXPECT_EQ(ddrRejection->reason,
            CanonicalResourceRejectionReason::DDRPayloadLimit);
  EXPECT_TRUE(ddrRejection->movement.has_value());
  EXPECT_TRUE(ddrRejection->requiredBytes.has_value());
  EXPECT_GT(*ddrRejection->requiredBytes, ddrRejection->capacityBytes);
}

TEST_F(CanonicalFeasibilityProofTest,
       BoundedCounterexampleKeepsExhaustionIndeterminate) {
  // A four-demand bounded oracle is intentionally small: it exercises solver
  // status strength. Representative 1024/1025 resource coverage is provided
  // by the production-shaped tests above.
  SemanticRootKey root;
  llvm::SmallVector<ExecutionInstanceId, 5> executions;
  for (unsigned index = 0; index < 4; ++index) {
    RootRegionWorkId work{root, TileId(0)};
    executions.push_back(
        {RequiredRootExecution{work, LogicalShardId{root, {index}}}});
  }
  SemanticRootKey storageRoot = root;
  storageRoot.anchorIndex = 99;
  RootRegionWorkId storageWork{storageRoot, TileId(1)};
  executions.push_back(
      {RequiredRootExecution{storageWork, LogicalShardId{storageRoot, {0}}}});

  PhysicalVersionId storageVersion{
      SupportRegionValueId{workOf(executions.back()), SupportValueId{}}};
  StorageObjectId storageObject{StorageObjectOrigin{storageVersion}};
  mlir::Type i8 = mlir::IntegerType::get(context.get(), 8);
  CanonicalStorageCoordinate storage;
  storage.plan.storageObjects.push_back({storageObject, TileId(1)});
  storage.plan.versionBindings.push_back({storageVersion, storageObject});
  storage.resources.push_back(
      {storageObject, box({0}, {1}), i8, wafer::MemLayout::Tensor});
  storage.lifetimes.push_back(
      {storageObject, executions.back(), {executions.back()}});

  CanonicalScheduleCoordinate schedule;
  for (const ExecutionInstanceId &execution : executions) {
    schedule.plan.order.push_back(execution);
    schedule.plan.workerBindings.push_back(
        {execution, wafer::NCCWorker::Worker0});
  }

  AttentionWorkDescription attentionRoot;
  attentionRoot.root = root;
  std::array<int64_t, 4> sizes{1, 1, 3, 3};
  std::array<AttentionValueId, 4> values;
  for (unsigned index = 0; index < 4; ++index) {
    AttentionWorkScopeId scope{executions[index], std::nullopt};
    AttentionValueId value{scope, AttentionValueKind::ScoreBlock};
    values[index] = value;
    mlir::AffineMap identity =
        mlir::AffineMap::getMultiDimIdentityMap(1, context.get());
    attentionRoot.values.push_back({value, box({0}, {sizes[index]}), i8,
                                    identity, std::nullopt, std::nullopt});
    AttentionActionId action{scope, AttentionActionKind::StateUpdate};
    attentionRoot.actions.push_back(
        {action, box({0}, {sizes[index]}), {value}, {value}});
  }
  attentionRoot.simultaneousValues = {
      {values[0].scope, {values[0], values[1]}},
      {values[0].scope, {values[0], values[3]}},
      {values[1].scope, {values[1], values[2]}}};
  CanonicalAttentionWorkCoordinate attention{{attentionRoot}};
  wafer::TargetMemoryPolicy memory = wafer::getTargetMemoryPolicy();
  memory.spmBase = 0;
  memory.spmLimit = 4;
  memory.spmAlignment = 1;
  CanonicalFeasibilityOptions noWork;
  noWork.packingSearchNodeBudget = 0;
  CanonicalFeasibilityOutcome exhausted = buildCanonicalFeasibilityProof(
      storage, /*movements=*/{}, schedule, attention, memory, noWork);
  const auto *indeterminate =
      std::get_if<CanonicalFeasibilityIndeterminate>(&exhausted);
  ASSERT_NE(indeterminate, nullptr);
  EXPECT_EQ(indeterminate->reason,
            CanonicalFeasibilityIndeterminateReason::ResourceWorkExhausted);

  CanonicalFeasibilityOutcome resolved = buildCanonicalFeasibilityProof(
      storage, /*movements=*/{}, schedule, attention, memory);
  EXPECT_NE(getCanonicalFeasibilityCoordinate(resolved), nullptr);
}

TEST_F(CanonicalFeasibilityProofTest,
       UnsupportedAndBrokenInputsRemainSeparate) {
  auto module = parse(mapSource(1024));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;

  CanonicalStorageCoordinate unsupported = inputs->storage;
  ASSERT_FALSE(unsupported.resources.empty());
  const ExactIndexSet &original = unsupported.resources.front().exactDomain;
  unsupported.resources.front().exactDomain = ExactIndexSet(
      original.getPresburgerSet(), ExactIndexSetForm::GeneralPresburger);
  CanonicalFeasibilityOutcome unsupportedOutcome =
      buildCanonicalFeasibilityProof(unsupported, inputs->movements,
                                     inputs->schedule, inputs->attention,
                                     wafer::getTargetMemoryPolicy());
  EXPECT_TRUE(std::holds_alternative<UnsupportedCanonicalFeasibility>(
      unsupportedOutcome));

  CanonicalStorageCoordinate brokenStorage = inputs->storage;
  brokenStorage.lifetimes.pop_back();
  CanonicalFeasibilityOutcome brokenCoverage = buildCanonicalFeasibilityProof(
      brokenStorage, inputs->movements, inputs->schedule, inputs->attention,
      wafer::getTargetMemoryPolicy());
  const auto *brokenResult =
      std::get_if<BrokenCanonicalFeasibility>(&brokenCoverage);
  ASSERT_NE(brokenResult, nullptr);
  EXPECT_EQ(brokenResult->reason,
            BrokenCanonicalFeasibilityReason::PlanCoverageMismatch);

  wafer::TargetMemoryPolicy invalidMemory = wafer::getTargetMemoryPolicy();
  invalidMemory.spmLimit = invalidMemory.spmBase;
  CanonicalFeasibilityOutcome invalidPolicy = buildCanonicalFeasibilityProof(
      inputs->storage, inputs->movements, inputs->schedule, inputs->attention,
      invalidMemory);
  const auto *invalidPolicyFailure =
      std::get_if<BrokenCanonicalFeasibility>(&invalidPolicy);
  ASSERT_NE(invalidPolicyFailure, nullptr);
  EXPECT_EQ(invalidPolicyFailure->reason,
            BrokenCanonicalFeasibilityReason::InvalidResourceProblem);
}

} // namespace
