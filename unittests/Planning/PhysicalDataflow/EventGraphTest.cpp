//===- EventGraphTest.cpp --------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

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
  std::vector<ExecutionEventContract> contracts;
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
  inputs.contracts = {
      {producer,
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2},
       NCCCompletionKind::OrderedAsynchronousIssue,
       0},
      {consumer,
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2},
       NCCCompletionKind::OrderedAsynchronousIssue,
       0}};

  inputs.movement.plan.externalLoads = {{loadId, input}};
  inputs.movement.plan.ddrTransfers = {
      {transferId, producerOutput, transferred}};
  inputs.movement.plan.peerGraphs = {{PeerTransferGraphKind::TargetRoutedPeer,
                                      {{TileId(0), TileId(1)}},
                                      {MovementActionId(transferId)}}};
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
  destination.contracts.insert(destination.contracts.end(),
                               source.contracts.begin(),
                               source.contracts.end());
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
        inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    const EventGraph &graph = *result.graph;
    EXPECT_EQ(graph.getEvents().size(), 29u);
    EXPECT_EQ(graph.getHardDependencies().size(), 30u);
    EXPECT_EQ(graph.getCompletionObligations().size(), 7u);
    EXPECT_EQ(graph.getComponents().size(), 1u);
    EXPECT_EQ(countResources<SPMRangeResource>(graph), 8u);
    EXPECT_EQ(countResources<CardDDRResource>(graph), 2u);
    EXPECT_EQ(countResources<DirectDTESenderResource>(graph), 1u);
    EXPECT_EQ(countResources<DTEReceiverFSMResource>(graph), 1u);
    EXPECT_EQ(countResources<OpaqueNoCTransferResource>(graph), 1u);
    EXPECT_EQ(countResources<DirectedNoCLinkResource>(graph), 0u);
    EXPECT_TRUE(llvm::all_of(
        graph.getResourceUses(), [](const PlannedResourceUse &use) {
          const bool coarseEstimate =
              std::holds_alternative<CardDDRResource>(use.resource) ||
              std::holds_alternative<TileEngineResource>(use.resource);
          return !coarseEstimate ||
                 (use.knowledge == ResourceKnowledge::Estimate &&
                  use.mode == ResourceUseMode::CapacityUnits);
        }));
    EXPECT_TRUE(llvm::none_of(
        graph.getOrderChoices(), [](const DisjunctiveResourceOrder &choice) {
          return std::holds_alternative<CardDDRResource>(choice.resource) ||
                 std::holds_alternative<TileEngineResource>(choice.resource) ||
                 std::holds_alternative<SPMRangeResource>(choice.resource);
        }));
    EXPECT_TRUE(llvm::none_of(
        graph.getCompletionObligations(), [](const auto &obligation) {
          return obligation.protocol == CompletionProtocol::Unknown ||
                 obligation.protocol ==
                     CompletionProtocol::NoAsynchronousCompletion;
        }));

    auto ready = graph.getReadyEvents({});
    ASSERT_TRUE(ready);
    ASSERT_EQ(ready->size(), 1u);
    const auto *action =
        std::get_if<MovementEventAction>(&ready->front().action);
    ASSERT_NE(action, nullptr);
    EXPECT_TRUE(std::holds_alternative<ExternalLoadId>(action->action));
    EXPECT_EQ(ready->front().kind, PlannedEventKind::MovementIssue);

    MovementHop hop{TileId(0), TileId(1)};
    EventId receiveIssue{MovementEventAction{inputs.peerAction,
                                             MovementEventPhase::PeerReceive, 0,
                                             hop},
                         PlannedEventKind::MovementIssue};
    EventId sendIssue{MovementEventAction{inputs.peerAction,
                                          MovementEventPhase::PeerSend, 0, hop},
                      PlannedEventKind::MovementIssue};
    EventId receiveCompletion{
        MovementEventAction{inputs.peerAction, MovementEventPhase::PeerReceive,
                            0, hop},
        PlannedEventKind::Completion};
    EventId sendCompletion{MovementEventAction{inputs.peerAction,
                                               MovementEventPhase::PeerSend, 0,
                                               hop},
                           PlannedEventKind::Completion};
    EXPECT_FALSE(llvm::is_contained(
        graph.getHardDependencies(),
        EventDependency{receiveIssue, sendIssue,
                        EventDependencyReason::TransferReady}));
    for (const EventId &issue : {receiveIssue, sendIssue})
      for (const EventId &completion : {receiveCompletion, sendCompletion})
        EXPECT_TRUE(llvm::is_contained(
            graph.getHardDependencies(),
            EventDependency{issue, completion,
                            EventDependencyReason::Completion}));
    EventId combineIssue{
        MovementEventAction{inputs.peerAction,
                            MovementEventPhase::LocalCombine, 0},
        PlannedEventKind::LocalCombine};
    EventId combineCompletion{
        MovementEventAction{inputs.peerAction,
                            MovementEventPhase::LocalCombine, 0},
        PlannedEventKind::Completion};
    EXPECT_TRUE(llvm::is_contained(
        graph.getHardDependencies(),
        EventDependency{receiveCompletion, combineIssue,
                        EventDependencyReason::TransferReady}));
    EXPECT_TRUE(llvm::is_contained(
        graph.getCompletionObligations(),
        CompletionObligation{combineIssue, combineCompletion,
                             CompletionProtocol::NCCParticipant, 0}));
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
      inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
  ASSERT_TRUE(opaque.succeeded());
  EXPECT_EQ(countResources<DirectedNoCLinkResource>(*opaque.graph), 0u);

  ExactMovementRoute route{{inputs.peerAction},
                           {TileId(0), TileId(1)},
                           {{TileId(0), TileId(8)}, {TileId(8), TileId(1)}}};
  EventGraphBuildResult exact =
      buildEventGraph(CardId(0), inputs.regions, inputs.temporal,
                      inputs.serialized, inputs.movement, inputs.storage,
                      inputs.storage.plan, inputs.contracts, {route});
  ASSERT_TRUE(exact.succeeded())
      << (exact.failure ? exact.failure->detail : "");
  EXPECT_EQ(countResources<DirectedNoCLinkResource>(*exact.graph), 2u);
  EXPECT_EQ(countResources<OpaqueNoCTransferResource>(*exact.graph), 1u);
  EXPECT_TRUE(llvm::all_of(
      exact.graph->getResourceUses(), [](const PlannedResourceUse &use) {
        return !std::holds_alternative<DirectedNoCLinkResource>(use.resource) ||
               (use.knowledge == ResourceKnowledge::Exact &&
                use.mode == ResourceUseMode::CapacityUnits);
      }));
  EXPECT_TRUE(llvm::none_of(
      exact.graph->getOrderChoices(),
      [](const DisjunctiveResourceOrder &choice) {
        return std::holds_alternative<DirectedNoCLinkResource>(choice.resource);
      }));
}

TEST(EventGraphTest, RelayStorageUsesExactPerHopCompletionEvents) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs inputs = makeInputs(context, 1031);
  ASSERT_EQ(inputs.movement.plan.peerGraphs.size(), 1u);
  PeerTransferGraphPlan &graph = inputs.movement.plan.peerGraphs.front();
  graph.kind = PeerTransferGraphKind::SoftwareRelay;
  graph.hops = {{TileId(0), TileId(2)}, {TileId(2), TileId(1)}};
  PeerRelayStorageId relayId{graph.actions, 0, TileId(2)};
  StorageObjectId relay{StorageObjectOrigin(relayId)};
  inputs.storage.plan.storageObjects.push_back({relay, TileId(2)});
  inputs.storage.resources.emplace_back(relay, box({2, 1031, 128}),
                                        mlir::Float16Type::get(&context),
                                        MemLayout::Tensor);
  inputs.storage.lifetimes.push_back(
      {relay,
       StorageAccessSite{PeerTransferSiteId{
           graph.actions, 0, PeerTransferSiteId::Endpoint::Receive,
           MovementHop{TileId(0), TileId(2)}}},
       {StorageAccessSite{PeerTransferSiteId{
           graph.actions, 0, PeerTransferSiteId::Endpoint::Send,
           MovementHop{TileId(2), TileId(1)}}}}});
  EventGraphBuildResult result = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  EXPECT_EQ(llvm::count_if(result.graph->getCompletionObligations(),
                           [](const CompletionObligation &obligation) {
                             return obligation.protocol ==
                                    CompletionProtocol::DirectDTE;
                           }),
            4u);
  EventId incomingCompletion{
      MovementEventAction{graph.actions.front(),
                          MovementEventPhase::PeerReceive, 0,
                          MovementHop{TileId(0), TileId(2)}},
      PlannedEventKind::Completion};
  EventId outgoingIssue{MovementEventAction{graph.actions.front(),
                                            MovementEventPhase::PeerSend, 0,
                                            MovementHop{TileId(2), TileId(1)}},
                        PlannedEventKind::MovementIssue};
  EXPECT_TRUE(llvm::is_contained(
      result.graph->getHardDependencies(),
      EventDependency{incomingCompletion, outgoingIssue,
                      EventDependencyReason::TransferReady}));
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
  inputs.contracts.front() = contract;
  EventGraphBuildResult original = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
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

  std::vector<ExecutionEventContract> synchronousContracts = inputs.contracts;
  synchronousContracts.front().completion =
      NCCCompletionKind::SynchronousWriteback;
  synchronousContracts.front().participantMask = 1;
  EventGraphBuildResult synchronous =
      buildEventGraph(CardId(0), inputs.regions, inputs.temporal,
                      inputs.serialized, inputs.movement, inputs.storage,
                      inputs.storage.plan, synchronousContracts);
  ASSERT_TRUE(synchronous.succeeded())
      << (synchronous.failure ? synchronous.failure->detail : "");
  auto synchronousObligation =
      llvm::find_if(synchronous.graph->getCompletionObligations(),
                    [&](const auto &item) { return item.issue == issue; });
  ASSERT_NE(synchronousObligation,
            synchronous.graph->getCompletionObligations().end());
  EXPECT_EQ(synchronousObligation->protocol,
            CompletionProtocol::NCCSynchronousWriteback);

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
  std::reverse(inputs.contracts.begin(), inputs.contracts.end());
  EventGraphBuildResult reversed = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
  ASSERT_TRUE(reversed.succeeded())
      << (reversed.failure ? reversed.failure->detail : "");
  EXPECT_EQ(*original.graph, *reversed.graph);
}

TEST(EventGraphTest, IndependentBranchesStayReadyWithoutInventedCardDDROrder) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  EventInputs first = makeInputs(context, 1025, 0);
  EventInputs second = makeInputs(context, 1025, 2);
  const EventId firstLoad = {MovementEventAction{first.loadAction},
                             PlannedEventKind::MovementIssue};
  const EventId secondLoad = {MovementEventAction{second.loadAction},
                              PlannedEventKind::MovementIssue};
  const EventId firstLoadPhase = {
      MovementEventAction{first.loadAction, MovementEventPhase::DDRLoad, 0},
      PlannedEventKind::MovementIssue};
  const EventId secondLoadPhase = {
      MovementEventAction{second.loadAction, MovementEventPhase::DDRLoad, 0},
      PlannedEventKind::MovementIssue};
  appendInputs(first, std::move(second));
  EventGraphBuildResult result = buildEventGraph(
      CardId(0), first.regions, first.temporal, first.serialized,
      first.movement, first.storage, first.storage.plan, first.contracts);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto ready = result.graph->getReadyEvents({});
  ASSERT_TRUE(ready);
  EXPECT_EQ(std::set<EventId>(ready->begin(), ready->end()),
            (std::set<EventId>{firstLoad, secondLoad}));

  EventDependency firstBeforeSecond{firstLoadPhase, secondLoadPhase,
                                    EventDependencyReason::EffectOrder};
  auto phases = result.graph->getReadyEvents({firstLoad, secondLoad});
  ASSERT_TRUE(phases);
  EXPECT_TRUE(llvm::is_contained(*phases, firstLoadPhase));
  EXPECT_TRUE(llvm::is_contained(*phases, secondLoadPhase));
  EXPECT_FALSE(
      result.graph->getReadyEvents({firstLoad, secondLoad}, {firstBeforeSecond})
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
      inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
  ASSERT_FALSE(cycle.succeeded());
  ASSERT_TRUE(cycle.failure);
  EXPECT_EQ(cycle.failure->kind, EventGraphFailureKind::ExactRejection);
  EXPECT_EQ(cycle.failure->reason,
            EventGraphFailureReason::HardDependencyCycle);
  EXPECT_FALSE(cycle.failure->witness.empty());

  inputs.storage.plan.orderRequirements.clear();
  EventGraphLimits limits;
  limits.maxEvents = 4;
  EventGraphBuildResult limited =
      buildEventGraph(CardId(0), inputs.regions, inputs.temporal,
                      inputs.serialized, inputs.movement, inputs.storage,
                      inputs.storage.plan, inputs.contracts, {}, limits);
  ASSERT_FALSE(limited.succeeded());
  ASSERT_TRUE(limited.failure);
  EXPECT_EQ(limited.failure->kind, EventGraphFailureKind::Indeterminate);
  EXPECT_EQ(limited.failure->reason, EventGraphFailureReason::WorkLimit);

  ExecutionEventContract invalid{
      inputs.producer, {}, NCCCompletionKind::OrderedAsynchronousIssue, 0};
  std::vector<ExecutionEventContract> invalidContracts = inputs.contracts;
  invalidContracts.front() = invalid;
  EventGraphBuildResult emptyWorker = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan, invalidContracts);
  ASSERT_FALSE(emptyWorker.succeeded());
  ASSERT_TRUE(emptyWorker.failure);
  EXPECT_EQ(emptyWorker.failure->kind, EventGraphFailureKind::ExactRejection);
  EXPECT_EQ(emptyWorker.failure->reason,
            EventGraphFailureReason::EmptyWorkerDomain);

  EventGraphBuildResult missingContract = buildEventGraph(
      CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
      inputs.movement, inputs.storage, inputs.storage.plan,
      llvm::ArrayRef<ExecutionEventContract>(inputs.contracts).drop_back());
  ASSERT_FALSE(missingContract.succeeded());
  ASSERT_TRUE(missingContract.failure);
  EXPECT_EQ(missingContract.failure->kind, EventGraphFailureKind::Deferred);
}

TEST(EventGraphTest,
     ExecutionContractsUseTypedLinalgAndRejectUnmodeledEffects) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @contract() {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<2x1024x128xf16>
    %result = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2x1024x128xf16>) -> tensor<2x1024x128xf16>
    %buffer = memref.alloc() : memref<2x1024x128xf16>
    memref.dealloc %buffer : memref<2x1024x128xf16>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto fill = *function.getOps<mlir::linalg::FillOp>().begin();
  auto allocation = *function.getOps<mlir::memref::AllocOp>().begin();
  SemanticRootKey root;
  RootRegionWorkId workId{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  ExecutionInstanceId execution{RequiredRootExecution{workId, shard}};
  SerializedExecutionPlan serialized{{execution}};
  RootRegionWork work;
  work.id = workId;
  work.rootOperation = fill;
  ExecutionEventContractResult supported =
      deriveExecutionEventContracts(serialized, {work});
  ASSERT_TRUE(supported.succeeded());
  ASSERT_EQ(supported.contracts.size(), 1u);
  EXPECT_EQ(supported.contracts.front().completion,
            NCCCompletionKind::OrderedAsynchronousIssue);
  EXPECT_EQ(supported.contracts.front().workerDomain.size(), kNCCWorkerCount);

  work.rootOperation = allocation;
  ExecutionEventContractResult unsupported =
      deriveExecutionEventContracts(serialized, {work});
  ASSERT_FALSE(unsupported.succeeded());
  ASSERT_TRUE(unsupported.failure);
  EXPECT_EQ(unsupported.failure->kind, EventGraphFailureKind::Unsupported);
  EXPECT_EQ(unsupported.failure->reason,
            EventGraphFailureReason::UnsupportedExecutionContract);
}

TEST(EventGraphTest,
     ProductionStructureDerivationUsesTypedRootAndExactTemporalClasses) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t extent : {int64_t{1024}, int64_t{1025}, int64_t{1031}}) {
    SCOPED_TRACE(extent);
    EventInputs inputs = makeInputs(context, extent);
    inputs.regions.groups.resize(1);
    inputs.temporal.scopes.resize(1);
    inputs.temporal.scopes.front().iteratorTileSizes = {2, 128, 128};
    inputs.temporal.scopes.front().waveLoopOrder = {1};
    inputs.serialized.executions = {inputs.producer};
    inputs.contracts = {inputs.contracts.front()};
    inputs.movement.plan.ddrTransfers.clear();
    inputs.movement.plan.peerGraphs.clear();
    ResultPublicationId publication{
        std::get<ExecutionResultValueId>(inputs.producerOutput.logicalValue)};
    inputs.movement.plan.publications = {{publication, inputs.producerOutput}};
    ExactIndexSet domain = box({2, extent, 128});
    mlir::Type f16 = mlir::Float16Type::get(&context);
    inputs.movement.resources = {
        {inputs.loadAction, domain, f16, std::nullopt, TileId(0)},
        {MovementActionId(publication), domain, f16, TileId(0), std::nullopt}};
    StorageObjectId inputObject{StorageObjectOrigin{inputs.input}};
    StorageObjectId outputObject{StorageObjectOrigin{inputs.producerOutput}};
    inputs.storage.plan.storageObjects = {{inputObject, TileId(0)},
                                          {outputObject, TileId(0)}};
    inputs.storage.plan.versionBindings = {
        {inputs.input, inputObject, StorageBindingKind::Fresh},
        {inputs.producerOutput, outputObject, StorageBindingKind::Fresh}};
    inputs.storage.resources = {{inputObject, domain, f16, MemLayout::Tensor},
                                {outputObject, domain, f16, MemLayout::Tensor}};
    inputs.storage.lifetimes = {
        {inputObject,
         StorageAccessSite{inputs.loadAction},
         {StorageAccessSite{inputs.producer}}},
        {outputObject,
         StorageAccessSite{inputs.producer},
         {StorageAccessSite{MovementActionId(publication)}}}};
    EventGraphBuildResult graph = buildEventGraph(
        CardId(0), inputs.regions, inputs.temporal, inputs.serialized,
        inputs.movement, inputs.storage, inputs.storage.plan, inputs.contracts);
    ASSERT_TRUE(graph.succeeded())
        << (graph.failure ? graph.failure->detail : "");

    std::string source = R"mlir(
module {
  func.func @root() {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<2xEXTENTx128xf16>
    %result = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2xEXTENTx128xf16>)
        -> tensor<2xEXTENTx128xf16>
    return
  }
}
)mlir";
    for (size_t marker = source.find("EXTENT"); marker != std::string::npos;
         marker = source.find("EXTENT"))
      source.replace(marker, 6, std::to_string(extent));
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto fill = *function.getOps<mlir::linalg::FillOp>().begin();
    const auto &required =
        std::get<RequiredRootExecution>(inputs.producer.source);
    RootRegionWork work;
    work.id = required.work;
    work.rootOperation = fill;
    TemporalScopeDescriptor descriptor;
    descriptor.id = inputs.temporal.scopes.front().id;
    descriptor.iterationOffsets = {0, 0, 0};
    descriptor.iterationExtents = {2, extent, 128};
    descriptor.iteratorCapabilities.assign(3,
                                           IteratorTilingCapability::Tileable);
    ExecutionStructureDomainResult structures = buildExecutionStructureDomain(
        *graph.graph, {descriptor}, inputs.temporal, {work});
    ASSERT_TRUE(structures.succeeded())
        << (structures.failure ? structures.failure->detail : "");
    ExecutionStructureSuccessor serialized = structures.domain->getFirstPlan();
    ASSERT_TRUE(serialized.getCursor());
    ExecutionStructureSuccessor pipelined =
        structures.domain->getNextPlan(*serialized.getCursor());
    ASSERT_EQ(pipelined.getKind(), ExecutionStructureSuccessorKind::Plan)
        << pipelined.getDetail().str();
    ASSERT_TRUE(pipelined.getPlan());
    auto selected =
        llvm::find_if(pipelined.getPlan()->scopes, [](const auto &choice) {
          return std::holds_alternative<PipelinedExecutionStructure>(choice);
        });
    ASSERT_NE(selected, pipelined.getPlan()->scopes.end());
    const auto &pipeline = std::get<PipelinedExecutionStructure>(*selected);
    EXPECT_EQ(pipeline.iteration.recurrenceAxis, 1u);
    EXPECT_EQ(pipeline.iteration.prefixCount, 1u);
    EXPECT_EQ(pipeline.iteration.steadyTripCount, 7u);
    EXPECT_EQ(pipeline.iteration.tailCount, extent == 1024 ? 0u : 1u);
  }
}

} // namespace
