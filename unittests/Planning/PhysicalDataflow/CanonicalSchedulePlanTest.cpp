//===- CanonicalSchedulePlanTest.cpp ---------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
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

struct CanonicalScheduleInputs {
  CanonicalStorageCoordinate storage;
  SerializedExecutionPlan serialized;
};

class CanonicalSchedulePlanTest : public ::testing::Test {
protected:
  CanonicalSchedulePlanTest() {
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

  static ExactIndexSet box(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
    IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
    EXPECT_TRUE(set.isExact());
    StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                        llvm::to_vector(sizes)};
    return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                         {rectangle});
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
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#id, #id],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  mlir::FailureOr<CanonicalScheduleInputs>
  buildInputs(const StructuredDAGAnalysis &dag, llvm::ArrayRef<TileId> tiles,
              std::string *failureReason) {
    auto prefix =
        wafer::test::buildCanonicalPlanningPrefix(dag, tiles, failureReason);
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
    return CanonicalScheduleInputs{*storage, *serialized};
  }

  static std::map<ScheduleNodeId, size_t>
  positions(const ClosedSchedulePlan &plan) {
    std::map<ScheduleNodeId, size_t> result;
    for (auto [position, node] : llvm::enumerate(plan.order))
      result.try_emplace(node, position);
    return result;
  }

  static bool hasDependency(const CanonicalScheduleCoordinate &coordinate,
                            const ScheduleNodeId &predecessor,
                            const ScheduleNodeId &successor) {
    return llvm::any_of(coordinate.dependencies,
                        [&](const ScheduleDependency &dependency) {
                          return dependency.predecessor == predecessor &&
                                 dependency.successor == successor;
                        });
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalSchedulePlanTest,
       AlignedAndRaggedLoadsExecuteAndPublishInWorkerZeroOrder) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(mapSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalSchedulePlanOutcome outcome =
        buildCanonicalSchedulePlan(inputs->storage, inputs->serialized);
    const CanonicalScheduleCoordinate *coordinate =
        getCanonicalScheduleCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.order.size(), 48u);
    ASSERT_EQ(coordinate->plan.workerBindings.size(), 48u);
    ASSERT_EQ(coordinate->dependencies.size(), 32u);
    EXPECT_TRUE(llvm::all_of(coordinate->plan.workerBindings,
                             [](const ScheduleWorkerBinding &binding) {
                               return binding.worker ==
                                      wafer::NCCWorker::Worker0;
                             }));
    std::map<ScheduleNodeId, size_t> order = positions(coordinate->plan);
    EXPECT_EQ(order.size(), coordinate->plan.order.size());
    for (const ScheduleDependency &dependency : coordinate->dependencies) {
      ASSERT_TRUE(order.count(dependency.predecessor));
      ASSERT_TRUE(order.count(dependency.successor));
      EXPECT_LT(order[dependency.predecessor], order[dependency.successor]);
    }

    std::reverse(inputs->storage.plan.storageObjects.begin(),
                 inputs->storage.plan.storageObjects.end());
    std::reverse(inputs->storage.resources.begin(),
                 inputs->storage.resources.end());
    std::reverse(inputs->storage.lifetimes.begin(),
                 inputs->storage.lifetimes.end());
    for (StorageLifetimeDescription &lifetime : inputs->storage.lifetimes)
      std::reverse(lifetime.uses.begin(), lifetime.uses.end());
    std::reverse(inputs->serialized.executions.begin(),
                 inputs->serialized.executions.end());
    CanonicalSchedulePlanOutcome reversed =
        buildCanonicalSchedulePlan(inputs->storage, inputs->serialized);
    const CanonicalScheduleCoordinate *reversedCoordinate =
        getCanonicalScheduleCoordinate(reversed);
    ASSERT_NE(reversedCoordinate, nullptr);
    EXPECT_EQ(reversedCoordinate->plan, coordinate->plan);
    EXPECT_EQ(reversedCoordinate->dependencies, coordinate->dependencies);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalSchedulePlanTest,
       DiamondKeepsOneProducerBeforeAllTransferConsumers) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @diamond(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %a0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%a0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %b0 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%b0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %c0 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%c0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    return %b, %c : tensor<2x1025x128xf16>, tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  CanonicalSchedulePlanOutcome outcome =
      buildCanonicalSchedulePlan(inputs->storage, inputs->serialized);
  const CanonicalScheduleCoordinate *coordinate =
      getCanonicalScheduleCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  std::map<ScheduleNodeId, size_t> order = positions(coordinate->plan);
  unsigned sharedProducers = 0;
  std::map<ExecutionInstanceId, unsigned> ddrSuccessors;
  for (const ScheduleDependency &dependency : coordinate->dependencies) {
    const auto *producer =
        std::get_if<ExecutionInstanceId>(&dependency.predecessor);
    const auto *action = std::get_if<MovementActionId>(&dependency.successor);
    if (producer && action &&
        std::holds_alternative<DDRBoundaryTransferId>(*action))
      ++ddrSuccessors[*producer];
    EXPECT_LT(order[dependency.predecessor], order[dependency.successor]);
  }
  for (const auto &[producer, count] : ddrSuccessors)
    sharedProducers += count > 1;
  EXPECT_GT(sharedProducers, 0u);
}

TEST_F(CanonicalSchedulePlanTest,
       FlashDecodingOrdersRemoteAndLocalComponentsIntoOneMerge) {
  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
        queryExtent, keyValueExtent));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalSchedulePlanOutcome outcome =
        buildCanonicalSchedulePlan(inputs->storage, inputs->serialized);
    const CanonicalScheduleCoordinate *coordinate =
        getCanonicalScheduleCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    std::map<ScheduleNodeId, size_t> order = positions(coordinate->plan);
    for (const ReductionGatherStorageBinding &binding :
         inputs->storage.plan.gatherStagingBindings) {
      ScheduleNodeId gather = MovementActionId{binding.gather};
      auto lifetime =
          llvm::find_if(inputs->storage.lifetimes,
                        [&](const StorageLifetimeDescription &entry) {
                          return entry.object == binding.stagingObject;
                        });
      ASSERT_NE(lifetime, inputs->storage.lifetimes.end());
      ASSERT_EQ(lifetime->uses.size(), 1u);
      const ScheduleNodeId merge = lifetime->uses.front();
      EXPECT_TRUE(hasDependency(*coordinate, gather, merge));
      EXPECT_LT(order[gather], order[merge]);
    }
    unsigned directLocalMergeEdges = 0;
    for (const ScheduleDependency &dependency : coordinate->dependencies)
      directLocalMergeEdges +=
          std::holds_alternative<ExecutionInstanceId>(dependency.predecessor) &&
          std::holds_alternative<ExecutionInstanceId>(dependency.successor);
    EXPECT_GT(directLocalMergeEdges, 0u);
  }
}

TEST_F(CanonicalSchedulePlanTest,
       OrdinaryReductionOrdersRemoteStagingAndLocalPartial) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    SemanticRootKey root;
    RootRegionWorkId remoteWork{root, TileId(0)};
    RootRegionWorkId mergeWork{root, TileId(1)};
    LogicalShardId remoteShard{root, {0}};
    LogicalShardId localShard{root, {1}};
    ReductionGroupId group{root, 0, {0}};
    ExecutionInstanceId remoteExecution{
        RequiredRootExecution{remoteWork, remoteShard}};
    ExecutionInstanceId localExecution{
        RequiredRootExecution{mergeWork, localShard}};
    ExecutionInstanceId mergeExecution{
        RequiredMergeExecution{mergeWork, group}};
    ReductionPartialValueId remoteLogical{remoteExecution, group, 0};
    ReductionPartialValueId localLogical{localExecution, group, 0};
    PhysicalVersionId remoteVersion{remoteLogical};
    PhysicalVersionId localVersion{localLogical};
    ReductionGatherId gatherId{group, remoteShard, remoteLogical};
    ResultPublicationId publicationId{
        ExecutionResultValueId{mergeExecution, 0}};
    StorageObjectId remoteObject{StorageObjectOrigin{remoteVersion}};
    StorageObjectId localObject{StorageObjectOrigin{localVersion}};
    StorageObjectId stagingObject{
        StorageObjectOrigin{ReductionGatherStagingId{gatherId}}};
    PhysicalVersionId resultVersion{publicationId.source};
    StorageObjectId resultObject{StorageObjectOrigin{resultVersion}};
    ExactIndexSet domain = box({0, 0, 0}, {2, extent, 128});
    mlir::Type f16 = mlir::Float16Type::get(context.get());

    CanonicalStorageCoordinate storage;
    storage.plan.storageObjects = {{remoteObject, TileId(0)},
                                   {localObject, TileId(1)},
                                   {stagingObject, TileId(1)},
                                   {resultObject, TileId(1)}};
    storage.plan.versionBindings = {{remoteVersion, remoteObject},
                                    {localVersion, localObject},
                                    {resultVersion, resultObject}};
    storage.plan.gatherStagingBindings = {{gatherId, stagingObject}};
    for (const StorageObjectId &object :
         {remoteObject, localObject, stagingObject, resultObject})
      storage.resources.push_back(
          {object, domain, f16, wafer::MemLayout::Tensor});
    storage.lifetimes = {{remoteObject,
                          ScheduleNodeId{remoteExecution},
                          {ScheduleNodeId{MovementActionId{gatherId}}}},
                         {localObject,
                          ScheduleNodeId{localExecution},
                          {ScheduleNodeId{mergeExecution}}},
                         {stagingObject,
                          ScheduleNodeId{MovementActionId{gatherId}},
                          {ScheduleNodeId{mergeExecution}}},
                         {resultObject,
                          ScheduleNodeId{mergeExecution},
                          {ScheduleNodeId{MovementActionId{publicationId}}}}};
    SerializedExecutionPlan serialized{
        {remoteExecution, localExecution, mergeExecution}};

    CanonicalSchedulePlanOutcome outcome =
        buildCanonicalSchedulePlan(storage, serialized);
    const CanonicalScheduleCoordinate *coordinate =
        getCanonicalScheduleCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ScheduleNodeId gather = MovementActionId{gatherId};
    ScheduleNodeId merge = mergeExecution;
    EXPECT_TRUE(hasDependency(*coordinate, remoteExecution, gather));
    EXPECT_TRUE(hasDependency(*coordinate, gather, merge));
    EXPECT_TRUE(hasDependency(*coordinate, localExecution, merge));
    EXPECT_TRUE(
        hasDependency(*coordinate, merge, MovementActionId{publicationId}));
  }
}

TEST_F(CanonicalSchedulePlanTest, RankZeroAndSupportSelfUsesDoNotCreateCycles) {
  auto scalar = parse(R"mlir(
#scalar = affine_map<() -> ()>
module {
  func.func @scalar(%input: tensor<f16>) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.generic {indexing_maps = [#scalar, #scalar],
        iterator_types = []} ins(%input : tensor<f16>)
        outs(%empty : tensor<f16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(scalar);
  std::string failureReason;
  auto scalarDag =
      StructuredDAGAnalysis::create(function(*scalar), &failureReason);
  ASSERT_TRUE(mlir::succeeded(scalarDag)) << failureReason;
  llvm::SmallVector<TileId, 1> tile{TileId(0)};
  auto scalarInputs = buildInputs(*scalarDag, tile, &failureReason);
  ASSERT_TRUE(mlir::succeeded(scalarInputs)) << failureReason;
  CanonicalSchedulePlanOutcome scalarOutcome = buildCanonicalSchedulePlan(
      scalarInputs->storage, scalarInputs->serialized);
  const CanonicalScheduleCoordinate *scalarCoordinate =
      getCanonicalScheduleCoordinate(scalarOutcome);
  ASSERT_NE(scalarCoordinate, nullptr);
  EXPECT_TRUE(llvm::none_of(
      scalarCoordinate->dependencies, [](const ScheduleDependency &dependency) {
        return dependency.predecessor == dependency.successor;
      }));

  auto support = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @pad(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1026x128xf16> {
    %zero = arith.constant 0.0 : f16
    %padded = tensor.pad %input low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%b: index, %m: index, %n: index): tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    %empty = tensor.empty() : tensor<2x1026x128xf16>
    %result = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%padded : tensor<2x1026x128xf16>)
        outs(%empty : tensor<2x1026x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1026x128xf16>
    return %result : tensor<2x1026x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(support);
  auto supportDag =
      StructuredDAGAnalysis::create(function(*support), &failureReason);
  ASSERT_TRUE(mlir::succeeded(supportDag)) << failureReason;
  auto supportInputs = buildInputs(*supportDag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(supportInputs)) << failureReason;
  CanonicalSchedulePlanOutcome supportOutcome = buildCanonicalSchedulePlan(
      supportInputs->storage, supportInputs->serialized);
  const CanonicalScheduleCoordinate *supportCoordinate =
      getCanonicalScheduleCoordinate(supportOutcome);
  ASSERT_NE(supportCoordinate, nullptr);
  EXPECT_TRUE(llvm::none_of(supportCoordinate->dependencies,
                            [](const ScheduleDependency &dependency) {
                              return dependency.predecessor ==
                                     dependency.successor;
                            }));
}

TEST_F(CanonicalSchedulePlanTest,
       InvalidCoverageUnknownSitesAndCyclesFailClosed) {
  auto module = parse(mapSource(1024));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;

  CanonicalSchedulePlanOutcome emptyOutcome =
      buildCanonicalSchedulePlan({}, {});
  const auto *emptyFailure = std::get_if<BrokenSchedulePlan>(&emptyOutcome);
  ASSERT_NE(emptyFailure, nullptr);
  EXPECT_EQ(emptyFailure->reason, BrokenSchedulePlanReason::EmptyScheduleInput);

  SerializedExecutionPlan duplicateExecution = inputs->serialized;
  duplicateExecution.executions.push_back(
      duplicateExecution.executions.front());
  CanonicalSchedulePlanOutcome duplicateExecutionOutcome =
      buildCanonicalSchedulePlan(inputs->storage, duplicateExecution);
  const auto *duplicateExecutionFailure =
      std::get_if<BrokenSchedulePlan>(&duplicateExecutionOutcome);
  ASSERT_NE(duplicateExecutionFailure, nullptr);
  EXPECT_EQ(duplicateExecutionFailure->reason,
            BrokenSchedulePlanReason::DuplicateSerializedExecution);

  CanonicalStorageCoordinate duplicateObject = inputs->storage;
  duplicateObject.plan.storageObjects.push_back(
      duplicateObject.plan.storageObjects.front());
  CanonicalSchedulePlanOutcome duplicateObjectOutcome =
      buildCanonicalSchedulePlan(duplicateObject, inputs->serialized);
  const auto *duplicateObjectFailure =
      std::get_if<BrokenSchedulePlan>(&duplicateObjectOutcome);
  ASSERT_NE(duplicateObjectFailure, nullptr);
  EXPECT_EQ(duplicateObjectFailure->reason,
            BrokenSchedulePlanReason::DuplicateStorageObject);

  CanonicalStorageCoordinate missingResource = inputs->storage;
  missingResource.resources.pop_back();
  CanonicalSchedulePlanOutcome missingResourceOutcome =
      buildCanonicalSchedulePlan(missingResource, inputs->serialized);
  const auto *missingResourceFailure =
      std::get_if<BrokenSchedulePlan>(&missingResourceOutcome);
  ASSERT_NE(missingResourceFailure, nullptr);
  EXPECT_EQ(missingResourceFailure->reason,
            BrokenSchedulePlanReason::MissingStorageResource);

  CanonicalStorageCoordinate duplicateResource = inputs->storage;
  duplicateResource.resources.push_back(duplicateResource.resources.front());
  CanonicalSchedulePlanOutcome duplicateResourceOutcome =
      buildCanonicalSchedulePlan(duplicateResource, inputs->serialized);
  const auto *duplicateResourceFailure =
      std::get_if<BrokenSchedulePlan>(&duplicateResourceOutcome);
  ASSERT_NE(duplicateResourceFailure, nullptr);
  EXPECT_EQ(duplicateResourceFailure->reason,
            BrokenSchedulePlanReason::DuplicateStorageResource);

  CanonicalStorageCoordinate duplicateLifetime = inputs->storage;
  duplicateLifetime.lifetimes.push_back(duplicateLifetime.lifetimes.front());
  CanonicalSchedulePlanOutcome duplicateLifetimeOutcome =
      buildCanonicalSchedulePlan(duplicateLifetime, inputs->serialized);
  const auto *duplicateLifetimeFailure =
      std::get_if<BrokenSchedulePlan>(&duplicateLifetimeOutcome);
  ASSERT_NE(duplicateLifetimeFailure, nullptr);
  EXPECT_EQ(duplicateLifetimeFailure->reason,
            BrokenSchedulePlanReason::DuplicateLifetime);

  CanonicalStorageCoordinate missingLifetime = inputs->storage;
  missingLifetime.lifetimes.pop_back();
  CanonicalSchedulePlanOutcome missingLifetimeOutcome =
      buildCanonicalSchedulePlan(missingLifetime, inputs->serialized);
  const auto *missingLifetimeFailure =
      std::get_if<BrokenSchedulePlan>(&missingLifetimeOutcome);
  ASSERT_NE(missingLifetimeFailure, nullptr);
  EXPECT_EQ(missingLifetimeFailure->reason,
            BrokenSchedulePlanReason::MissingLifetime);

  CanonicalStorageCoordinate emptyUse = inputs->storage;
  emptyUse.lifetimes.front().uses.clear();
  CanonicalSchedulePlanOutcome emptyUseOutcome =
      buildCanonicalSchedulePlan(emptyUse, inputs->serialized);
  const auto *emptyUseFailure =
      std::get_if<BrokenSchedulePlan>(&emptyUseOutcome);
  ASSERT_NE(emptyUseFailure, nullptr);
  EXPECT_EQ(emptyUseFailure->reason, BrokenSchedulePlanReason::EmptyUseSet);

  CanonicalStorageCoordinate unknownObject = inputs->storage;
  StorageObjectId absentObject{StorageObjectOrigin{ReductionGatherStagingId{}}};
  unknownObject.lifetimes.front().object = absentObject;
  CanonicalSchedulePlanOutcome unknownObjectOutcome =
      buildCanonicalSchedulePlan(unknownObject, inputs->serialized);
  const auto *unknownObjectFailure =
      std::get_if<BrokenSchedulePlan>(&unknownObjectOutcome);
  ASSERT_NE(unknownObjectFailure, nullptr);
  EXPECT_EQ(unknownObjectFailure->reason,
            BrokenSchedulePlanReason::UnknownStorageObject);

  CanonicalStorageCoordinate unknownExecution = inputs->storage;
  SemanticRootKey unknownRoot;
  unknownRoot.anchorIndex = 999;
  ExecutionInstanceId unknown{
      RequiredRootExecution{RootRegionWorkId{unknownRoot, TileId(0)},
                            LogicalShardId{unknownRoot, {0}}}};
  unknownExecution.lifetimes.front().uses.push_back(unknown);
  CanonicalSchedulePlanOutcome unknownExecutionOutcome =
      buildCanonicalSchedulePlan(unknownExecution, inputs->serialized);
  const auto *unknownExecutionFailure =
      std::get_if<BrokenSchedulePlan>(&unknownExecutionOutcome);
  ASSERT_NE(unknownExecutionFailure, nullptr);
  EXPECT_EQ(unknownExecutionFailure->reason,
            BrokenSchedulePlanReason::UnknownExecutionSite);

  CanonicalStorageCoordinate cyclic = inputs->storage;
  ASSERT_GE(inputs->serialized.executions.size(), 2u);
  const ExecutionInstanceId first = inputs->serialized.executions[0];
  const ExecutionInstanceId second = inputs->serialized.executions[1];
  bool firstAdded = false;
  bool secondAdded = false;
  for (StorageLifetimeDescription &lifetime : cyclic.lifetimes) {
    if (!firstAdded && lifetime.definition == ScheduleNodeId{first}) {
      lifetime.uses.push_back(second);
      firstAdded = true;
    } else if (!secondAdded && lifetime.definition == ScheduleNodeId{second}) {
      lifetime.uses.push_back(first);
      secondAdded = true;
    }
  }
  ASSERT_TRUE(firstAdded && secondAdded);
  CanonicalSchedulePlanOutcome cyclicOutcome =
      buildCanonicalSchedulePlan(cyclic, inputs->serialized);
  const auto *cyclicFailure = std::get_if<BrokenSchedulePlan>(&cyclicOutcome);
  ASSERT_NE(cyclicFailure, nullptr);
  EXPECT_EQ(cyclicFailure->reason, BrokenSchedulePlanReason::CyclicDependency);

  CanonicalSchedulePlanOutcome repeated =
      buildCanonicalSchedulePlan(inputs->storage, inputs->serialized);
  EXPECT_NE(getCanonicalScheduleCoordinate(repeated), nullptr);
}

} // namespace
