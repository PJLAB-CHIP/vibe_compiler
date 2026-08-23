//===- EventGraphTest.cpp --------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet box(llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult result =
      IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(result.isExact()) << result.reason;
  StaticRectangularIndexSet rectangle{offsets,
                                      llvm::SmallVector<int64_t, 4>(sizes)};
  return ExactIndexSet(std::move(*result.set), ExactIndexSetForm::BoxUnion,
                       {rectangle});
}

struct EventInputs {
  RegionPlan regions;
  TemporalPlan temporal;
  SerializedExecutionPlan serialized;
  CanonicalMovementCoordinate movement;
  CanonicalStorageCoordinate storage;
  ExecutionInstanceId producer;
  ExecutionInstanceId consumer;
  PhysicalVersionId input;
  PhysicalVersionId producerOutput;
  PhysicalVersionId transferred;
  PhysicalVersionId consumerOutput;
  MovementActionId loadAction;
  MovementActionId peerAction;
};

EventInputs makeInputs(mlir::MLIRContext &context, int64_t extent,
                       uint32_t rootBase = 0) {
  SemanticRootKey producerRoot;
  producerRoot.anchorIndex = rootBase;
  SemanticRootKey consumerRoot;
  consumerRoot.anchorIndex = rootBase + 1;
  RootRegionWorkId producerWork{producerRoot, TileId(0)};
  RootRegionWorkId consumerWork{consumerRoot, TileId(1)};
  LogicalShardId producerShard{producerRoot, {0}};
  LogicalShardId consumerShard{consumerRoot, {1}};
  ExecutionInstanceId producer{
      RequiredRootExecution{producerWork, producerShard}};
  ExecutionInstanceId consumer{
      RequiredRootExecution{consumerWork, consumerShard}};

  DemandFragmentId inputFragment;
  inputFragment.source.kind = RootBoundaryKind::ProgramInput;
  inputFragment.source.semantic = producerRoot;
  inputFragment.use.destinationShard = producerShard;
  inputFragment.ownerTile = TileId(0);
  BoundaryRegionValueId inputValue{producerWork, inputFragment};
  DemandFragmentId transferFragment;
  transferFragment.source.kind = RootBoundaryKind::StructuredResult;
  transferFragment.source.semantic = producerRoot;
  transferFragment.use.destinationShard = consumerShard;
  transferFragment.ownerTile = TileId(0);
  BoundaryRegionValueId transferredValue{consumerWork, transferFragment};

  PhysicalVersionId input{inputValue};
  PhysicalVersionId producerOutput{ExecutionResultValueId{producer, 0}};
  PhysicalVersionId transferred{transferredValue};
  PhysicalVersionId consumerOutput{ExecutionResultValueId{consumer, 0}};
  ExternalLoadId loadId{inputValue};
  DDRBoundaryTransferId transferId{transferredValue};
  ResultPublicationId publicationId{
      std::get<ExecutionResultValueId>(consumerOutput.logicalValue)};

  EventInputs inputs;
  inputs.producer = producer;
  inputs.consumer = consumer;
  inputs.input = input;
  inputs.producerOutput = producerOutput;
  inputs.transferred = transferred;
  inputs.consumerOutput = consumerOutput;
  inputs.loadAction = MovementActionId{loadId};
  inputs.peerAction = MovementActionId{transferId};
  inputs.regions.groups = {
      RegionGroupPlan{TileId(0), {producerWork}, {{producer}}, {}, {}, {}},
      RegionGroupPlan{TileId(1), {consumerWork}, {{consumer}}, {}, {}, {}}};
  inputs.temporal.scopes = {
      {TraversalScopeId{RegionExecutionId{producer}, TopLevelWorkPieceId{0}},
       {2, extent, 128},
       {0, 1, 2}},
      {TraversalScopeId{RegionExecutionId{consumer}, TopLevelWorkPieceId{0}},
       {2, extent, 128},
       {0, 1, 2}}};
  inputs.serialized.executions = {producer, consumer};

  inputs.movement.plan.externalLoads = {{loadId, input}};
  inputs.movement.plan.ddrTransfers = {
      {transferId, producerOutput, transferred,
       MovementRealization{MovementRealizationKind::TargetRoutedPeer,
                           {{TileId(0), TileId(1)}}}}};
  inputs.movement.plan.publications = {{publicationId, consumerOutput}};
  ExactIndexSet domain = box({2, extent, 128});
  mlir::Type f16 = mlir::Float16Type::get(&context);
  inputs.movement.resources = {
      {MovementActionId{loadId}, domain, f16, std::nullopt, TileId(0)},
      {MovementActionId{transferId}, domain, f16, TileId(0), TileId(1)},
      {MovementActionId{publicationId}, domain, f16, TileId(1), std::nullopt}};

  std::vector<PhysicalVersionId> versions{input, producerOutput, transferred,
                                          consumerOutput};
  for (const PhysicalVersionId &version : versions) {
    StorageObjectId object{StorageObjectOrigin{version}};
    const TileId tile = version == transferred || version == consumerOutput
                            ? TileId(1)
                            : TileId(0);
    inputs.storage.plan.storageObjects.push_back({object, tile});
    inputs.storage.plan.versionBindings.push_back(
        {version, object, StorageBindingKind::Fresh});
    inputs.storage.resources.emplace_back(object, domain, f16,
                                          MemLayout::Tensor);
  }
  StorageObjectId inputObject{StorageObjectOrigin{input}};
  StorageObjectId producerObject{StorageObjectOrigin{producerOutput}};
  StorageObjectId transferredObject{StorageObjectOrigin{transferred}};
  StorageObjectId consumerObject{StorageObjectOrigin{consumerOutput}};
  inputs.storage.lifetimes = {
      {inputObject,
       StorageAccessSite{MovementActionId{loadId}},
       {StorageAccessSite{producer}}},
      {producerObject,
       StorageAccessSite{producer},
       {StorageAccessSite{MovementActionId{transferId}}}},
      {transferredObject,
       StorageAccessSite{MovementActionId{transferId}},
       {StorageAccessSite{consumer}}},
      {consumerObject,
       StorageAccessSite{consumer},
       {StorageAccessSite{MovementActionId{publicationId}}}}};
  return inputs;
}

void appendInputs(EventInputs &destination, EventInputs source) {
  destination.regions.groups.insert(destination.regions.groups.end(),
                                    source.regions.groups.begin(),
                                    source.regions.groups.end());
  destination.temporal.scopes.insert(destination.temporal.scopes.end(),
                                     source.temporal.scopes.begin(),
                                     source.temporal.scopes.end());
  destination.serialized.executions.insert(
      destination.serialized.executions.end(),
      source.serialized.executions.begin(), source.serialized.executions.end());
  destination.movement.plan.externalLoads.insert(
      destination.movement.plan.externalLoads.end(),
      source.movement.plan.externalLoads.begin(),
      source.movement.plan.externalLoads.end());
  destination.movement.plan.ddrTransfers.insert(
      destination.movement.plan.ddrTransfers.end(),
      source.movement.plan.ddrTransfers.begin(),
      source.movement.plan.ddrTransfers.end());
  destination.movement.plan.publications.insert(
      destination.movement.plan.publications.end(),
      source.movement.plan.publications.begin(),
      source.movement.plan.publications.end());
  destination.movement.resources.insert(destination.movement.resources.end(),
                                        source.movement.resources.begin(),
                                        source.movement.resources.end());
  destination.storage.plan.storageObjects.insert(
      destination.storage.plan.storageObjects.end(),
      source.storage.plan.storageObjects.begin(),
      source.storage.plan.storageObjects.end());
  destination.storage.plan.versionBindings.insert(
      destination.storage.plan.versionBindings.end(),
      source.storage.plan.versionBindings.begin(),
      source.storage.plan.versionBindings.end());
  destination.storage.resources.insert(destination.storage.resources.end(),
                                       source.storage.resources.begin(),
                                       source.storage.resources.end());
  destination.storage.lifetimes.insert(destination.storage.lifetimes.end(),
                                       source.storage.lifetimes.begin(),
                                       source.storage.lifetimes.end());
}

template <typename Resource> size_t countResources(const EventGraph &graph) {
  return llvm::count_if(graph.getResourceUses(), [](const auto &use) {
    return std::holds_alternative<Resource>(use.resource);
  });
}

TEST(EventGraphTest, AlignedAndRaggedChainHasAllAndOnlyTypedFacts) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    EventInputs inputs = makeInputs(context, extent);
    EventGraphBuildResult result = buildEventGraph(
        CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
        inputs.movement, inputs.storage, inputs.storage.plan);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    const EventGraph &graph = *result.graph;
    EXPECT_EQ(graph.getEvents().size(), 21u);
    EXPECT_EQ(graph.getHardDependencies().size(), 20u);
    EXPECT_EQ(graph.getCompletionObligations().size(), 6u);
    EXPECT_EQ(graph.getComponents().size(), 1u);
    EXPECT_EQ(countResources<SPMRangeResource>(graph), 8u);
    EXPECT_EQ(countResources<CardDDRResource>(graph), 2u);
    EXPECT_EQ(countResources<TileDTEEngineResource>(graph), 4u);
    EXPECT_EQ(countResources<OpaqueNoCTransferResource>(graph), 1u);
    EXPECT_EQ(countResources<DirectedNoCLinkResource>(graph), 0u);

    auto ready = graph.getReadyEvents({});
    ASSERT_TRUE(ready);
    ASSERT_EQ(ready->size(), 1u);
    const auto *action =
        std::get_if<MovementEventAction>(&ready->front().action);
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(std::holds_alternative<ExternalLoadId>(action->action));
    EXPECT_EQ(ready->front().kind, PlannedEventKind::MovementIssue);
  }
}

TEST(EventGraphTest, OpaqueAndExactRoutesKeepDifferentKnowledgeBoundaries) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs inputs = makeInputs(context, 1031);
  EventGraphBuildResult opaque = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan);
  ASSERT_TRUE(opaque.succeeded());
  EXPECT_EQ(countResources<DirectedNoCLinkResource>(*opaque.graph), 0u);

  ExactMovementRoute route{inputs.peerAction,
                           {{TileId(0), TileId(8)}, {TileId(8), TileId(1)}}};
  EventGraphBuildResult exact = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, {}, {route});
  ASSERT_TRUE(exact.succeeded())
      << (exact.failure ? exact.failure->detail : "");
  EXPECT_EQ(countResources<DirectedNoCLinkResource>(*exact.graph), 2u);
  EXPECT_EQ(countResources<OpaqueNoCTransferResource>(*exact.graph), 1u);
}

TEST(EventGraphTest, CompletionContractAndInputOrderAreDeterministic) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs inputs = makeInputs(context, 1025);
  ExecutionEventContract contract{inputs.producer,
                                  {NCCWorker::Worker2, NCCWorker::Worker0,
                                   NCCWorker::Worker1, NCCWorker::Worker0},
                                  NCCCompletionKind::OrderedAsynchronousIssue,
                                  0};
  EventGraphBuildResult original = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, {contract});
  ASSERT_TRUE(original.succeeded())
      << (original.failure ? original.failure->detail : "");
  EventId issue = {ExecutionEventAction{inputs.producer},
                   PlannedEventKind::ComputeIssue};
  auto event =
      llvm::find_if(original.graph->getEvents(),
                    [&](const auto &item) { return item.id == issue; });
  ASSERT_NE(event, original.graph->getEvents().end());
  EXPECT_EQ(event->workerDomain,
            (std::vector<NCCWorker>{NCCWorker::Worker0, NCCWorker::Worker1,
                                    NCCWorker::Worker2}));
  auto obligation =
      llvm::find_if(original.graph->getCompletionObligations(),
                    [&](const auto &item) { return item.issue == issue; });
  ASSERT_NE(obligation, original.graph->getCompletionObligations().end());
  EXPECT_EQ(obligation->protocol, CompletionProtocol::NCCParticipant);

  std::reverse(inputs.regions.groups.begin(), inputs.regions.groups.end());
  std::reverse(inputs.temporal.scopes.begin(), inputs.temporal.scopes.end());
  std::reverse(inputs.serialized.executions.begin(),
               inputs.serialized.executions.end());
  std::reverse(inputs.movement.resources.begin(),
               inputs.movement.resources.end());
  std::reverse(inputs.storage.plan.storageObjects.begin(),
               inputs.storage.plan.storageObjects.end());
  std::reverse(inputs.storage.plan.versionBindings.begin(),
               inputs.storage.plan.versionBindings.end());
  std::reverse(inputs.storage.resources.begin(),
               inputs.storage.resources.end());
  std::reverse(inputs.storage.lifetimes.begin(),
               inputs.storage.lifetimes.end());
  EventGraphBuildResult reversed = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, {contract});
  ASSERT_TRUE(reversed.succeeded())
      << (reversed.failure ? reversed.failure->detail : "");
  EXPECT_EQ(*original.graph, *reversed.graph);
}

TEST(EventGraphTest, IndependentBranchesExposeExactResourceReadySuccessors) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs first = makeInputs(context, 1025, 0);
  EventInputs second = makeInputs(context, 1025, 2);
  const EventId firstLoad = {
      MovementEventAction{first.loadAction, std::nullopt},
      PlannedEventKind::MovementIssue};
  const EventId secondLoad = {
      MovementEventAction{second.loadAction, std::nullopt},
      PlannedEventKind::MovementIssue};
  appendInputs(first, std::move(second));
  EventGraphBuildResult result = buildEventGraph(
      CardId(0), first.regions, first.temporal, first.serialized,
      first.movement, first.storage, first.storage.plan);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto ready = result.graph->getReadyEvents({});
  ASSERT_TRUE(ready);
  EXPECT_EQ(std::set<EventId>(ready->begin(), ready->end()),
            (std::set<EventId>{firstLoad, secondLoad}));

  EventDependency firstBeforeSecond{firstLoad, secondLoad,
                                    EventDependencyReason::EffectOrder};
  auto ordered = result.graph->getReadyEvents({}, {firstBeforeSecond});
  ASSERT_TRUE(ordered);
  EXPECT_EQ(*ordered, (std::vector<EventId>{firstLoad}));
  EventDependency secondBeforeFirst{secondLoad, firstLoad,
                                    EventDependencyReason::EffectOrder};
  EXPECT_FALSE(
      result.graph->getReadyEvents({}, {firstBeforeSecond, secondBeforeFirst})
          .has_value());

  EventDependency invalid{
      firstLoad,
      EventId{
          BufferEventAction{StorageObjectId{StorageObjectOrigin{first.input}},
                            StorageObjectId{StorageObjectOrigin{first.input}}},
          PlannedEventKind::BufferReady},
      EventDependencyReason::EffectOrder};
  EXPECT_FALSE(result.graph->getReadyEvents({}, {invalid}).has_value());
}

TEST(EventGraphTest, CycleWorkLimitAndEmptyWorkerDomainStayTyped) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs inputs = makeInputs(context, 1025);
  inputs.storage.plan.orderRequirements.push_back(
      {inputs.consumerOutput, inputs.input,
       BufferOrderKind::ReuseAfterCompletion});
  EventGraphBuildResult cycle = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan);
  ASSERT_FALSE(cycle.succeeded());
  ASSERT_TRUE(cycle.failure);
  EXPECT_EQ(cycle.failure->kind, EventGraphFailureKind::ExactRejection);
  EXPECT_EQ(cycle.failure->reason,
            EventGraphFailureReason::HardDependencyCycle);
  EXPECT_FALSE(cycle.failure->witness.empty());

  inputs.storage.plan.orderRequirements.clear();
  EventGraphLimits limits;
  limits.maxEvents = 4;
  EventGraphBuildResult limited = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, {}, {}, limits);
  ASSERT_FALSE(limited.succeeded());
  ASSERT_TRUE(limited.failure);
  EXPECT_EQ(limited.failure->kind, EventGraphFailureKind::Indeterminate);
  EXPECT_EQ(limited.failure->reason, EventGraphFailureReason::WorkLimit);

  ExecutionEventContract invalid{
      inputs.producer, {}, NCCCompletionKind::OrderedAsynchronousIssue, 0};
  EventGraphBuildResult emptyWorker = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, {invalid});
  ASSERT_FALSE(emptyWorker.succeeded());
  ASSERT_TRUE(emptyWorker.failure);
  EXPECT_EQ(emptyWorker.failure->kind, EventGraphFailureKind::ExactRejection);
  EXPECT_EQ(emptyWorker.failure->reason,
            EventGraphFailureReason::EmptyWorkerDomain);
}

} // namespace
