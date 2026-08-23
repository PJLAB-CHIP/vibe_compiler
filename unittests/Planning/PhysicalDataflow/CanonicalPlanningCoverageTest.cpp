//===- CanonicalPlanningCoverageTest.cpp -----------------------------===//

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"
#include "Wafer/Target/Core/TargetMemory.h"

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct CanonicalPlanningChain {
  wafer::test::CanonicalPlanningPrefix prefix;
  CanonicalRepresentationCoordinate representations;
  CanonicalMovementCoordinate movements;
  SerializedExecutionPlan serialized;
  CanonicalStorageCoordinate storage;
  CanonicalScheduleCoordinate schedule;
  CanonicalAttentionWorkCoordinate attention;
};

class CanonicalPlanningCoverageTest : public ::testing::Test {
protected:
  CanonicalPlanningCoverageTest() {
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

  static void setStageFailure(std::string *failureReason,
                              llvm::StringRef stage) {
    if (failureReason && failureReason->empty())
      *failureReason = (stage + " failed").str();
  }

  mlir::FailureOr<CanonicalPlanningChain>
  buildChain(const StructuredDAGAnalysis &dag, std::string *failureReason) {
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(dag, allTiles(),
                                                            failureReason);
    if (mlir::failed(prefix))
      return mlir::failure();

    CanonicalRepresentationPlanOutcome representationOutcome =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representationOutcome);
    if (!representations) {
      setStageFailure(failureReason, "canonical representation");
      return mlir::failure();
    }

    CanonicalMovementPlanOutcome movementOutcome = buildCanonicalMovementPlan(
        prefix->regions, *representations, prefix->rootWorks);
    const CanonicalMovementCoordinate *movements =
        getCanonicalMovementCoordinate(movementOutcome);
    if (!movements) {
      setStageFailure(failureReason, "canonical movement");
      return mlir::failure();
    }

    CanonicalSerializedExecutionPlanOutcome serializedOutcome =
        buildCanonicalSerializedExecutionPlan(prefix->regions,
                                              prefix->temporal);
    const SerializedExecutionPlan *serialized =
        getSerializedExecutionPlan(serializedOutcome);
    if (!serialized) {
      setStageFailure(failureReason, "canonical serialized execution");
      return mlir::failure();
    }

    CanonicalStoragePlanOutcome storageOutcome =
        buildCanonicalStoragePlan(*representations, *movements, *serialized);
    const CanonicalStorageCoordinate *storage =
        getCanonicalStorageCoordinate(storageOutcome);
    if (!storage) {
      setStageFailure(failureReason, "canonical storage");
      return mlir::failure();
    }

    CanonicalSchedulePlanOutcome scheduleOutcome =
        buildCanonicalSchedulePlan(*storage, *serialized);
    const CanonicalScheduleCoordinate *schedule =
        getCanonicalScheduleCoordinate(scheduleOutcome);
    if (!schedule) {
      setStageFailure(failureReason, "canonical schedule");
      return mlir::failure();
    }

    CanonicalAttentionWorkProjectionOutcome attentionOutcome =
        buildCanonicalAttentionWorkProjection(prefix->rootWorks,
                                              *representations, *movements,
                                              *storage, *schedule);
    const CanonicalAttentionWorkCoordinate *attention =
        getCanonicalAttentionWorkCoordinate(attentionOutcome);
    if (!attention) {
      setStageFailure(failureReason, "canonical attention projection");
      return mlir::failure();
    }

    return CanonicalPlanningChain{
        std::move(*prefix), *representations, *movements, *serialized,
        *storage,           *schedule,        *attention};
  }

  static std::set<MovementActionId>
  getMovementActions(const MovementPlan &plan) {
    std::set<MovementActionId> actions;
    for (const ExternalLoadPlan &load : plan.externalLoads)
      actions.insert(MovementActionId{load.id});
    for (const DDRBoundaryTransferPlan &transfer : plan.ddrTransfers)
      actions.insert(MovementActionId{transfer.id});
    for (const ReductionGatherPlan &gather : plan.reductionGathers)
      actions.insert(MovementActionId{gather.id});
    for (const ResultPublicationPlan &publication : plan.publications)
      actions.insert(MovementActionId{publication.id});
    return actions;
  }

  void expectClosedIdentityChain(const CanonicalPlanningChain &chain) {
    std::set<ExecutionInstanceId> regionExecutions;
    for (const RegionGroupPlan &group : chain.prefix.regions.groups)
      for (const ExecutionInstancePlan &execution : group.executions)
        EXPECT_TRUE(regionExecutions.insert(execution.id).second);
    const std::set<ExecutionInstanceId> serializedExecutions(
        chain.serialized.executions.begin(), chain.serialized.executions.end());
    EXPECT_EQ(regionExecutions, serializedExecutions);
    EXPECT_EQ(serializedExecutions.size(), chain.serialized.executions.size());

    std::set<ExecutionInstanceId> temporalExecutions;
    for (const TemporalScopePlan &scope : chain.prefix.temporal.scopes) {
      const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
      ASSERT_NE(execution, nullptr);
      EXPECT_TRUE(isTopLevelScope(scope.id));
      EXPECT_TRUE(temporalExecutions.insert(*execution).second);
    }
    for (const ExecutionInstanceId &execution : chain.serialized.executions) {
      if (std::holds_alternative<RequiredRootExecution>(execution.source))
        EXPECT_EQ(temporalExecutions.count(execution), 1u);
      else
        EXPECT_EQ(temporalExecutions.count(execution), 0u);
    }

    std::set<PhysicalVersionId> plannedVersions;
    for (const PhysicalVersionPlan &version :
         chain.representations.plan.physicalVersions)
      EXPECT_TRUE(plannedVersions.insert(version.id).second);
    std::set<PhysicalVersionId> describedVersions;
    for (const RepresentationResourceDescription &resource :
         chain.representations.resources)
      EXPECT_TRUE(describedVersions.insert(resource.version).second);
    EXPECT_EQ(plannedVersions, describedVersions);

    const std::set<MovementActionId> movementActions =
        getMovementActions(chain.movements.plan);
    std::set<MovementActionId> movementResources;
    for (const MovementResourceDescription &resource :
         chain.movements.resources)
      EXPECT_TRUE(movementResources.insert(resource.action).second);
    EXPECT_EQ(movementActions, movementResources);

    std::set<PhysicalVersionId> boundVersions;
    for (const PhysicalVersionStorageBinding &binding :
         chain.storage.plan.versionBindings) {
      EXPECT_TRUE(boundVersions.insert(binding.version).second);
      EXPECT_EQ(binding.object,
                (StorageObjectId{StorageObjectOrigin{binding.version}}));
    }
    EXPECT_EQ(boundVersions, plannedVersions);

    std::set<StorageObjectId> plannedObjects;
    for (const StorageObjectPlan &object : chain.storage.plan.storageObjects)
      EXPECT_TRUE(plannedObjects.insert(object.id).second);
    std::set<StorageObjectId> resourceObjects;
    for (const StorageResourceDescription &resource : chain.storage.resources)
      EXPECT_TRUE(resourceObjects.insert(resource.object).second);
    std::set<StorageObjectId> lifetimeObjects;
    for (const StorageLifetimeDescription &lifetime : chain.storage.lifetimes) {
      EXPECT_TRUE(lifetimeObjects.insert(lifetime.object).second);
      EXPECT_FALSE(lifetime.uses.empty());
    }
    EXPECT_EQ(plannedObjects, resourceObjects);
    EXPECT_EQ(plannedObjects, lifetimeObjects);

    std::set<ScheduleNodeId> expectedScheduleNodes;
    for (const ExecutionInstanceId &execution : chain.serialized.executions)
      expectedScheduleNodes.insert(ScheduleNodeId{execution});
    for (const MovementActionId &action : movementActions)
      expectedScheduleNodes.insert(ScheduleNodeId{action});
    const std::set<ScheduleNodeId> scheduleNodes(
        chain.schedule.plan.order.begin(), chain.schedule.plan.order.end());
    EXPECT_EQ(scheduleNodes, expectedScheduleNodes);
    EXPECT_EQ(scheduleNodes.size(), chain.schedule.plan.order.size());
    ASSERT_EQ(chain.schedule.plan.workerBindings.size(),
              chain.schedule.plan.order.size());
    std::set<ScheduleNodeId> workerNodes;
    for (const ScheduleWorkerBinding &binding :
         chain.schedule.plan.workerBindings) {
      EXPECT_EQ(binding.worker, wafer::NCCWorker::Worker0);
      EXPECT_TRUE(workerNodes.insert(binding.node).second);
    }
    EXPECT_EQ(workerNodes, scheduleNodes);
    for (const ScheduleDependency &dependency : chain.schedule.dependencies) {
      EXPECT_EQ(scheduleNodes.count(dependency.predecessor), 1u);
      EXPECT_EQ(scheduleNodes.count(dependency.successor), 1u);
      EXPECT_EQ(plannedObjects.count(dependency.object), 1u);
    }

    std::set<AttentionActionId> attentionActions;
    for (const AttentionWorkDescription &root : chain.attention.roots)
      for (const AttentionActionDescription &action : root.actions)
        EXPECT_TRUE(attentionActions.insert(action.id).second);
  }

  static std::string diamondSource(int64_t extent) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @diamond_join(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> (tensor<2x" << extent
           << "x128xf16>, tensor<2x" << extent << "x128xf16>) {\n"
           << "    %a0 = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
           << "    %a = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%a0 : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    %b0 = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
           << "    %b = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%a : tensor<2x" << extent
           << "x128xf16>) outs(%b0 : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    %c0 = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
           << "    %c = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%a : tensor<2x" << extent
           << "x128xf16>) outs(%c0 : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    %d0 = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
           << "    %d = linalg.generic {indexing_maps = [#id, #id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%b, %c : tensor<2x" << extent
           << "x128xf16>, tensor<2x" << extent
           << "x128xf16>) outs(%d0 : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%lhs: f16, %rhs: f16, %o: f16):\n"
           << "        %sum = arith.addf %lhs, %rhs : f16\n"
           << "        linalg.yield %sum : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %d, %c : tensor<2x" << extent
           << "x128xf16>, tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static std::string reductionSource(int64_t extent) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#input = affine_map<(b, m, k) -> (b, m, k)>
#output = affine_map<(b, m, k) -> ()>
module {
  func.func @reduce(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<f16> {\n"
           << "    %zero = arith.constant 0.0 : f16\n"
           << "    %empty = tensor.empty() : tensor<f16>\n"
           << "    %init = linalg.fill ins(%zero : f16)\n"
           << "        outs(%empty : tensor<f16>) -> tensor<f16>\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#input, #output],\n"
           << "        iterator_types = [\"reduction\", \"reduction\", "
              "\"reduction\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%init : tensor<f16>) {\n"
           << "      ^bb0(%value: f16, %accumulator: f16):\n"
           << "        %sum = arith.addf %value, %accumulator : f16\n"
           << "        linalg.yield %sum : f16\n"
           << "    } -> tensor<f16>\n"
           << "    return %result : tensor<f16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static std::string multiPieceSource(int64_t extent) {
    const int64_t middleExtent = extent == 1024 ? 256 : 257;
    const int64_t middleOffset = 384;
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @multi_piece(%left: tensor<2x)mlir"
           << extent << "x128xf16>, %middle: tensor<2x" << middleExtent
           << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %producer_out = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %producer = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%left : tensor<2x" << extent
           << "x128xf16>) outs(%producer_out : tensor<2x" << extent
           << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    %assembled = tensor.insert_slice %middle into %producer"
           << "[0, " << middleOffset << ", 0] [2, " << middleExtent
           << ", 128] [1, 1, 1] : tensor<2x" << middleExtent
           << "x128xf16> into tensor<2x" << extent << "x128xf16>\n"
           << "    %result_out = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%assembled : tensor<2x" << extent
           << "x128xf16>) outs(%result_out : tensor<2x" << extent
           << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static std::string multiProducerSource(int64_t extent) {
    const int64_t leftExtent = 500;
    const int64_t rightExtent = extent - leftExtent;
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @multi_producer(%left: tensor<2x)mlir"
           << leftExtent << "x128xf16>, %right: tensor<2x" << rightExtent
           << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %left_out = tensor.empty() : tensor<2x" << leftExtent
           << "x128xf16>\n"
           << "    %producer_a = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%left : tensor<2x" << leftExtent
           << "x128xf16>) outs(%left_out : tensor<2x" << leftExtent
           << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << leftExtent << "x128xf16>\n"
           << "    %right_out = tensor.empty() : tensor<2x" << rightExtent
           << "x128xf16>\n"
           << "    %producer_b = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%right : tensor<2x" << rightExtent
           << "x128xf16>) outs(%right_out : tensor<2x" << rightExtent
           << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << rightExtent << "x128xf16>\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %lower = tensor.insert_slice %producer_a into %empty"
           << "[0, 0, 0] [2, " << leftExtent << ", 128] [1, 1, 1] : tensor<2x"
           << leftExtent << "x128xf16> into tensor<2x" << extent
           << "x128xf16>\n"
           << "    %assembled = tensor.insert_slice %producer_b into %lower"
           << "[0, " << leftExtent << ", 0] [2, " << rightExtent
           << ", 128] [1, 1, 1] : tensor<2x" << rightExtent
           << "x128xf16> into tensor<2x" << extent << "x128xf16>\n"
           << "    %result_out = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
           << "        ins(%assembled : tensor<2x" << extent
           << "x128xf16>) outs(%result_out : tensor<2x" << extent
           << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  static wafer::LinalgExtAttentionOp firstAttention(mlir::ModuleOp module) {
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

TEST_F(CanonicalPlanningCoverageTest,
       OrdinaryAlignedAndRaggedDiamondClosesEveryCanonicalIdentity) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(diamondSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto chain = buildChain(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(chain)) << failureReason;

    ASSERT_EQ(chain->prefix.spatial.nodes.size(), 4u);
    for (const NodeExecutionPartition &node : chain->prefix.spatial.nodes) {
      ASSERT_EQ(node.shards.size(), 16u);
      int64_t coveredElements = 0;
      std::vector<int64_t> shardElements;
      for (const ExecutionShard &shard : node.shards) {
        int64_t elements = 1;
        for (const IteratorInterval &interval : shard.iterationDomain)
          elements *= interval.size;
        coveredElements += elements;
        shardElements.push_back(elements);
      }
      EXPECT_EQ(coveredElements, int64_t{2} * extent * 128);
      if (extent == 1024)
        EXPECT_EQ(*llvm::min_element(shardElements),
                  *llvm::max_element(shardElements));
      else
        EXPECT_LT(*llvm::min_element(shardElements),
                  *llvm::max_element(shardElements));
    }
    EXPECT_EQ(chain->prefix.rootWorks.size(), 64u);
    EXPECT_EQ(chain->serialized.executions.size(), 64u);
    EXPECT_TRUE(chain->attention.roots.empty());
    expectClosedIdentityChain(*chain);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalPlanningCoverageTest,
       ReductionAlignedAndRaggedKeepsReductionRootUnpartitioned) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(reductionSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto chain = buildChain(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(chain)) << failureReason;

    ASSERT_EQ(chain->prefix.spatial.nodes.size(), 2u);
    const NodeExecutionPartition *reduction = nullptr;
    for (const NodeExecutionPartition &node : chain->prefix.spatial.nodes) {
      ASSERT_EQ(node.shards.size(), 1u);
      if (node.shards.front().iterationDomain.size() == 3)
        reduction = &node;
      else
        EXPECT_TRUE(node.shards.front().iterationDomain.empty());
    }
    ASSERT_NE(reduction, nullptr);
    const ExecutionShard &shard = reduction->shards.front();
    ASSERT_EQ(shard.iterationDomain.size(), 3u);
    EXPECT_EQ(shard.iterationDomain[0], (IteratorInterval{0, 2}));
    EXPECT_EQ(shard.iterationDomain[1], (IteratorInterval{0, extent}));
    EXPECT_EQ(shard.iterationDomain[2], (IteratorInterval{0, 128}));
    ASSERT_EQ(chain->serialized.executions.size(), 2u);
    ASSERT_EQ(chain->prefix.temporal.scopes.size(), 2u);
    auto reductionScope = llvm::find_if(
        chain->prefix.temporal.scopes, [](const TemporalScopePlan &scope) {
          return scope.iteratorTileSizes.size() == 3;
        });
    ASSERT_NE(reductionScope, chain->prefix.temporal.scopes.end());
    EXPECT_TRUE(llvm::equal(reductionScope->iteratorTileSizes,
                            llvm::ArrayRef<int64_t>{2, extent, 128}));
    EXPECT_TRUE(chain->attention.roots.empty());
    expectClosedIdentityChain(*chain);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalPlanningCoverageTest,
       MultiPieceAndMultiProducerCarriersCloseAtRepresentativeScale) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto multiPiece = parse(multiPieceSource(extent));
    ASSERT_TRUE(multiPiece);
    const std::string multiPieceBefore = print(multiPiece->getOperation());
    std::string failureReason;
    auto multiPieceDag =
        StructuredDAGAnalysis::create(function(*multiPiece), &failureReason);
    ASSERT_TRUE(mlir::succeeded(multiPieceDag)) << failureReason;
    auto multiPieceChain = buildChain(*multiPieceDag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(multiPieceChain)) << failureReason;
    expectClosedIdentityChain(*multiPieceChain);
    ASSERT_FALSE(multiPieceChain->movements.plan.discards.empty());
    for (const ResultDiscardPlan &discard :
         multiPieceChain->movements.plan.discards) {
      EXPECT_EQ(discard.source, PhysicalVersionId{discard.id.source});
      StorageObjectId object{StorageObjectOrigin{discard.source}};
      auto lifetime =
          llvm::find_if(multiPieceChain->storage.lifetimes,
                        [&](const StorageLifetimeDescription &candidate) {
                          return candidate.object == object;
                        });
      ASSERT_NE(lifetime, multiPieceChain->storage.lifetimes.end());
      ASSERT_EQ(lifetime->uses.size(), 1u);
      EXPECT_EQ(lifetime->definition, lifetime->uses.front());
      EXPECT_EQ(lifetime->definition,
                StorageAccessSite{discard.id.source.execution});
    }
    EXPECT_EQ(print(multiPiece->getOperation()), multiPieceBefore);

    auto multiProducer = parse(multiProducerSource(extent));
    ASSERT_TRUE(multiProducer);
    const std::string multiProducerBefore =
        print(multiProducer->getOperation());
    failureReason.clear();
    auto multiProducerDag =
        StructuredDAGAnalysis::create(function(*multiProducer), &failureReason);
    ASSERT_TRUE(mlir::succeeded(multiProducerDag)) << failureReason;
    auto multiProducerChain = buildChain(*multiProducerDag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(multiProducerChain)) << failureReason;
    expectClosedIdentityChain(*multiProducerChain);

    for (const RepresentationResourceDescription &resource :
         multiProducerChain->representations.resources) {
      EXPECT_EQ(resource.exactDomain.getForm(), ExactIndexSetForm::BoxUnion);
      EXPECT_FALSE(resource.exactDomain.getBoxes().empty());
    }
    EXPECT_EQ(print(multiProducer->getOperation()), multiProducerBefore);
  }
}

TEST_F(CanonicalPlanningCoverageTest,
       AttentionModesKeepProofAndSelectedMappingsExactAcrossTheFullChain) {
  struct Case {
    std::string source;
    int64_t globalScoreElements;
    unsigned expectedRoots;
    wafer::AttentionAlgorithm algorithm;
  };
  std::vector<Case> cases;
  cases.push_back({wafer::test::buildFlashAttentionPlanningFixture(
                       /*queryExtent=*/1024, /*keyValueExtent=*/1024,
                       /*withMask=*/false),
                   int64_t{2} * 1024 * 1024, 1,
                   wafer::AttentionAlgorithm::FlashAttention});
  cases.push_back({wafer::test::buildFlashAttentionPlanningFixture(
                       /*queryExtent=*/1025, /*keyValueExtent=*/1031,
                       /*withMask=*/true),
                   int64_t{2} * 1025 * 1031, 1,
                   wafer::AttentionAlgorithm::FlashAttention});
  cases.push_back({wafer::test::buildFlashDecodingPlanningFixture(
                       /*queryExtent=*/1024, /*keyValueExtent=*/1024),
                   int64_t{2} * 1024 * 1024, 1,
                   wafer::AttentionAlgorithm::FlashDecoding});
  cases.push_back({wafer::test::buildFlashDecodingPlanningFixture(
                       /*queryExtent=*/1025, /*keyValueExtent=*/1031),
                   int64_t{2} * 1025 * 1031, 1,
                   wafer::AttentionAlgorithm::FlashDecoding});
  cases.push_back({wafer::test::buildMultiK2FlashDecodingPlanningFixture(),
                   int64_t{2} * 1025 * 33 * 31, 1,
                   wafer::AttentionAlgorithm::FlashDecoding});
  cases.push_back({wafer::test::buildTwoFlashAttentionPlanningFixture(),
                   int64_t{2} * 1025 * 1031, 2,
                   wafer::AttentionAlgorithm::FlashAttention});

  for (const auto &[caseIndex, testCase] : llvm::enumerate(cases)) {
    SCOPED_TRACE(caseIndex);
    auto module = parse(testCase.source);
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto chain = buildChain(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(chain)) << failureReason;
    ASSERT_EQ(chain->attention.roots.size(), testCase.expectedRoots);
    expectClosedIdentityChain(*chain);

    PreparedAttentionDecompositionOutcome preparedOutcome =
        prepareSelectedAttentionDecomposition(chain->attention);
    const auto *prepared =
        std::get_if<PreparedAttentionDecomposition>(&preparedOutcome);
    ASSERT_NE(prepared, nullptr);
    ASSERT_EQ(prepared->work.roots.size(), testCase.expectedRoots);

    mlir::OwningOpRef<mlir::ModuleOp> selected = module->clone();
    ASSERT_TRUE(selected);
    llvm::SmallVector<wafer::LinalgExtAttentionOp, 2> selectedOps;
    selected->walk(
        [&](wafer::LinalgExtAttentionOp op) { selectedOps.push_back(op); });
    ASSERT_EQ(selectedOps.size(), prepared->work.roots.size());
    mlir::IRRewriter rewriter(context.get());
    for (auto [op, description] :
         llvm::zip_equal(selectedOps, prepared->work.roots)) {
      EXPECT_EQ(description.algorithm, testCase.algorithm);
      auto materialized = emitSelectedAttentionDecomposition(
          rewriter, op, description, &failureReason);
      ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
      EXPECT_EQ(materialized->root, description.root);

      std::set<AttentionActionId> describedActions;
      for (const AttentionActionDescription &action : description.actions)
        EXPECT_TRUE(describedActions.insert(action.id).second);
      std::set<AttentionActionId> materializedActions;
      unsigned stateMerges = 0;
      unsigned finalizes = 0;
      for (const AttentionActionMaterialization &action :
           materialized->actions) {
        EXPECT_TRUE(materializedActions.insert(action.id).second);
        EXPECT_FALSE(action.operations.empty());
        stateMerges += action.id.kind == AttentionActionKind::StateMerge;
        finalizes += action.id.kind == AttentionActionKind::Finalize;
      }
      EXPECT_EQ(materializedActions, describedActions);

      std::set<AttentionValueId> describedValues;
      for (const AttentionValueDescription &value : description.values)
        EXPECT_TRUE(describedValues.insert(value.id).second);
      std::set<AttentionValueId> materializedValues;
      unsigned scoreBlocks = 0;
      for (const AttentionValueMaterialization &value : materialized->values) {
        EXPECT_TRUE(materializedValues.insert(value.id).second);
        if (value.id.kind != AttentionValueKind::ScoreBlock)
          continue;
        ++scoreBlocks;
        ASSERT_FALSE(value.occurrences.empty());
        for (mlir::Value occurrence : value.occurrences) {
          auto type =
              mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
          ASSERT_TRUE(type);
          EXPECT_LT(type.getNumElements(), testCase.globalScoreElements);
        }
      }
      EXPECT_EQ(materializedValues, describedValues);
      EXPECT_GT(scoreBlocks, 0u);
      if (description.algorithm == wafer::AttentionAlgorithm::FlashDecoding) {
        EXPECT_GT(stateMerges, 0u);
        EXPECT_EQ(finalizes, stateMerges);
      } else {
        EXPECT_EQ(stateMerges, 0u);
        EXPECT_GT(finalizes, 0u);
      }
    }
    EXPECT_EQ(count<wafer::LinalgExtAttentionOp>(*selected), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*selected)));
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalPlanningCoverageTest,
       BrokenAttentionWorkDoesNotMutateOrPoisonAValidRaggedChain) {
  auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
      /*queryExtent=*/1025, /*keyValueExtent=*/1031));
  ASSERT_TRUE(module);
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto chain = buildChain(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(chain)) << failureReason;

  CanonicalAttentionWorkCoordinate duplicated = chain->attention;
  ASSERT_FALSE(duplicated.roots.empty());
  ASSERT_FALSE(duplicated.roots.front().actions.empty());
  duplicated.roots.front().actions.push_back(
      duplicated.roots.front().actions.front());
  PreparedAttentionDecompositionOutcome rejected =
      prepareSelectedAttentionDecomposition(duplicated);
  const auto *failure = std::get_if<BrokenPreparedAttention>(&rejected);
  ASSERT_NE(failure, nullptr);
  EXPECT_EQ(failure->reason, BrokenPreparedAttentionReason::DuplicateIdentity);
  EXPECT_EQ(print(module->getOperation()), before);

  PreparedAttentionDecompositionOutcome retried =
      prepareSelectedAttentionDecomposition(chain->attention);
  EXPECT_TRUE(std::holds_alternative<PreparedAttentionDecomposition>(retried));
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
