//===- CompleteCandidatePreparationTest.cpp --------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidatePreparation.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Transforms/MemoryPlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExecutionInstanceId makeExecution(uint32_t anchorIndex = 0, int64_t tile = 0) {
  SemanticRootKey root;
  root.anchorIndex = anchorIndex;
  RootRegionWorkId work{root, TileId(tile)};
  LogicalShardId shard{root, {0}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

ExternalLoadId makeExternalLoad(int64_t tile, uint64_t anchorIndex) {
  SemanticRootKey root;
  root.anchorIndex = anchorIndex;
  RootRegionWorkId work{root, TileId(tile)};
  LogicalShardId shard{root, {0}};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::ProgramInput;
  fragment.source.index = 0;
  fragment.use = {0, shard};
  return ExternalLoadId{BoundaryRegionValueId{work, fragment}};
}

ExactIndexSet makeDomain(int64_t extent = 1031) {
  llvm::SmallVector<int64_t, 4> offsets{0, 0, 0};
  llvm::SmallVector<int64_t, 4> sizes{2, extent, 128};
  IndexSetResult result =
      IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(result.isExact()) << result.reason;
  return ExactIndexSet(std::move(*result.set), ExactIndexSetForm::BoxUnion,
                       {StaticRectangularIndexSet{offsets, sizes}});
}

TEST(CompleteCandidatePreparationTest,
     PipelinedRotatingScheduleFeedsActualMiniMalloc) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "main",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  auto region = builder.create<TileRegionOp>(module.getLoc(), mlir::TypeRange{},
                                             mlir::ValueRange{});
  region.getBody().push_back(new mlir::Block());
  mlir::OpBuilder regionBuilder =
      mlir::OpBuilder::atBlockBegin(&region.getBody().front());
  auto type = mlir::MemRefType::get(
      {2, 1031, 128}, regionBuilder.getF16Type(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
  mlir::Value original =
      regionBuilder.create<mlir::memref::AllocOp>(module.getLoc(), type);
  auto lower =
      regionBuilder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 128);
  auto upper =
      regionBuilder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 1024);
  auto step =
      regionBuilder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 128);
  auto zero = regionBuilder.create<mlir::arith::ConstantOp>(
      module.getLoc(),
      regionBuilder.getFloatAttr(regionBuilder.getF16Type(), 0.0));
  auto loop = regionBuilder.create<mlir::scf::ForOp>(module.getLoc(), lower,
                                                     upper, step);
  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
  bodyBuilder.create<InstrFillOp>(module.getLoc(), original, zero,
                                  FillDomainAttr(), NCCWorker::Worker0);
  regionBuilder.setInsertionPointAfter(loop);
  regionBuilder.create<mlir::memref::DeallocOp>(module.getLoc(), original);
  regionBuilder.create<TileYieldOp>(module.getLoc());
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(module.getLoc());
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  ExecutionInstanceId execution = makeExecution();
  EventId issue{ExecutionEventAction{execution},
                PlannedEventKind::ComputeIssue};
  EventId completion{ExecutionEventAction{execution},
                     PlannedEventKind::Completion};
  OccurrenceRelationId recurrence{
      TraversalScopeId{RegionExecutionId{execution}, TopLevelWorkPieceId{0}},
      {1, 9, 1}};
  PipelineScopeId scope{{issue, completion}, {recurrence}};
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = recurrence;
  pipeline.iteration = {1, 1, 7, 1};
  pipeline.eventStages = {{issue, StageId(0)}, {completion, StageId(1)}};
  pipeline.dependences = {
      {issue, completion, 0, PipelineDependenceKind::AsyncCompletion}};
  pipeline.completionObligations = {
      {issue, completion, CompletionProtocol::NCCParticipant, 0}};
  ExecutionStructurePlan structure{{pipeline}};

  PhysicalVersionId version{ExecutionResultValueId{execution, 0}};
  StorageObjectId object{StorageObjectOrigin(version)};
  BufferPlan buffers;
  buffers.storageObjects.push_back({object, TileId(0)});
  buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  buffers.slotFamilies.push_back({SlotFamilyId{{object}}, recurrence, 2, {1}});
  ScheduleDomainInput input;
  input.structure = structure;
  input.buffers = buffers;
  input.events = {
      {issue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {completion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {issue, completion, EventDependencyReason::Completion}};
  input.completionObligations = pipeline.completionObligations;
  input.components = {{{issue, completion}}};
  input.slotLifetimes.push_back({buffers.slotFamilies.front().id,
                                 recurrence,
                                 pipeline.iteration,
                                 {issue},
                                 {completion},
                                 1,
                                 2,
                                 9});
  ScheduleDomainResult domain = buildScheduleDomain(input);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ScheduleSuccessor first = domain.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());

  RepresentationPlan representations;
  representations.logicalValues.push_back({version.logicalValue, version});
  representations.physicalVersions.push_back({version, MemLayout::Tensor});
  PreparedCandidateEvents events{input.events, input.hardDependencies,
                                 input.completionObligations};
  CompleteCandidatePreparation preparation(
      std::move(*domain.domain), std::move(events), representations, {}, {},
      buffers,
      {StorageResourceDescription{object, makeDomain(), builder.getF16Type(),
                                  MemLayout::Tensor}},
      structure, *first.getPlan(), {{RegionExecutionId(execution), 0}});

  mlir::OwningOpRef<mlir::ModuleOp> owned(module);
  StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back({0, 0, original});
  relations.operandBuffers.push_back({0, original});
  CandidateInstructionIR tile{CardId(0), TileId(0), &owned, &relations};
  CardExecutablePreparationFailure failure;
  mlir::LogicalResult prepared =
      preparation.prepareInstructionIR({tile}, failure);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failure.detail;
  ASSERT_TRUE(owned);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*owned)));
  EXPECT_TRUE(mlir::succeeded(
      wafer::planSPMMemoryModule(*owned, 0, 3 * 1024 * 1024, 16)));
  unsigned allocations = 0;
  owned->walk([&](mlir::memref::AllocOp) { ++allocations; });
  EXPECT_EQ(allocations, 2u);
}

TEST(CompleteCandidatePreparationTest,
     SelectedAliasReuseKeepsExactEventOwnersAndActualMiniMalloc) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    mlir::DialectRegistry registry;
    registerCompilationDialects(registry);
    auto context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto function = moduleBuilder.create<mlir::func::FuncOp>(
        module.getLoc(), "main",
        moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
    auto region = builder.create<TileRegionOp>(
        module.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
    region.getBody().push_back(new mlir::Block());
    mlir::OpBuilder regionBuilder =
        mlir::OpBuilder::atBlockBegin(&region.getBody().front());
    auto type = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
    mlir::Value first =
        regionBuilder.create<mlir::memref::AllocOp>(module.getLoc(), type);
    mlir::Value second =
        regionBuilder.create<mlir::memref::AllocOp>(module.getLoc(), type);
    auto zero = regionBuilder.create<mlir::arith::ConstantOp>(
        module.getLoc(),
        regionBuilder.getFloatAttr(regionBuilder.getF16Type(), 0.0));
    regionBuilder.create<InstrFillOp>(module.getLoc(), first, zero,
                                      FillDomainAttr(), NCCWorker::Worker0);
    regionBuilder.create<mlir::memref::DeallocOp>(module.getLoc(), first);
    regionBuilder.create<InstrFillOp>(module.getLoc(), second, zero,
                                      FillDomainAttr(), NCCWorker::Worker0);
    regionBuilder.create<mlir::memref::DeallocOp>(module.getLoc(), second);
    regionBuilder.create<TileYieldOp>(module.getLoc());
    builder.setInsertionPointAfter(region);
    builder.create<mlir::func::ReturnOp>(module.getLoc());
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    ExecutionInstanceId firstExecution = makeExecution(0);
    ExecutionInstanceId secondExecution = makeExecution(1);
    EventId firstIssue{ExecutionEventAction{firstExecution},
                       PlannedEventKind::ComputeIssue};
    EventId firstCompletion{ExecutionEventAction{firstExecution},
                            PlannedEventKind::Completion};
    EventId secondIssue{ExecutionEventAction{secondExecution},
                        PlannedEventKind::ComputeIssue};
    EventId secondCompletion{ExecutionEventAction{secondExecution},
                             PlannedEventKind::Completion};
    PhysicalVersionId firstVersion{ExecutionResultValueId{firstExecution, 0}};
    PhysicalVersionId secondVersion{ExecutionResultValueId{secondExecution, 0}};
    StorageObjectId object{StorageObjectOrigin(firstVersion)};
    BufferPlan buffers;
    buffers.storageObjects.push_back({object, TileId(0)});
    buffers.versionBindings = {
        {firstVersion, object, StorageBindingKind::Fresh},
        {secondVersion, object, StorageBindingKind::Reuse}};
    PipelineScopeId scope{
        {firstIssue, firstCompletion, secondIssue, secondCompletion}, {}};
    llvm::sort(scope.events);
    ExecutionStructurePlan structure{{SerializedExecutionStructure{scope}}};
    ScheduleDomainInput input;
    input.structure = structure;
    input.buffers = buffers;
    input.events = {
        {firstIssue,
         CardId(0),
         TileId(0),
         {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
        {firstCompletion, CardId(0), TileId(0), {}},
        {secondIssue,
         CardId(0),
         TileId(0),
         {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
        {secondCompletion, CardId(0), TileId(0), {}}};
    input.hardDependencies = {
        {firstIssue, firstCompletion, EventDependencyReason::Completion},
        {firstCompletion, secondIssue, EventDependencyReason::SSAValue},
        {secondIssue, secondCompletion, EventDependencyReason::Completion}};
    input.completionObligations = {
        {firstIssue, firstCompletion, CompletionProtocol::NCCParticipant, 0},
        {secondIssue, secondCompletion, CompletionProtocol::NCCParticipant, 0}};
    input.components = {
        {{firstIssue, firstCompletion, secondIssue, secondCompletion}}};
    ScheduleDomainResult domain = buildScheduleDomain(input);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    ScheduleSuccessor selected = domain.domain->getFirstPlan();
    ASSERT_EQ(selected.getKind(), ScheduleSuccessorKind::Plan);
    ASSERT_TRUE(selected.getPlan());

    RepresentationPlan representations;
    representations.logicalValues = {
        {firstVersion.logicalValue, firstVersion},
        {secondVersion.logicalValue, secondVersion}};
    representations.physicalVersions = {{firstVersion, MemLayout::Tensor},
                                        {secondVersion, MemLayout::Tensor}};
    PreparedCandidateEvents events{input.events, input.hardDependencies,
                                   input.completionObligations};
    CompleteCandidatePreparation preparation(
        std::move(*domain.domain), std::move(events), representations, {}, {},
        buffers,
        {StorageResourceDescription{object, makeDomain(extent),
                                    regionBuilder.getF16Type(),
                                    MemLayout::Tensor}},
        structure, *selected.getPlan(),
        {{RegionExecutionId(firstExecution), 0},
         {RegionExecutionId(secondExecution), 1}});

    mlir::OwningOpRef<mlir::ModuleOp> owned(module);
    StructuredMaterializationRelations relations;
    relations.operationResultBuffers = {{0, 0, first}, {1, 0, second}};
    relations.operandBuffers = {{0, first}, {1, second}};
    CandidateInstructionIR tile{CardId(0), TileId(0), &owned, &relations};
    CardExecutablePreparationFailure failure;
    ASSERT_TRUE(
        mlir::succeeded(preparation.prepareInstructionIR({tile}, failure)))
        << failure.detail;
    ASSERT_TRUE(owned);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*owned)));
    EXPECT_TRUE(mlir::succeeded(
        wafer::planSPMMemoryModule(*owned, 0, 3 * 1024 * 1024, 16)));
    unsigned allocations = 0;
    owned->walk([&](mlir::memref::AllocOp) { ++allocations; });
    EXPECT_EQ(allocations, 1u);
    ASSERT_EQ(relations.operationResultBuffers.size(), 2u);
    EXPECT_EQ(relations.operationResultBuffers[0].buffer,
              relations.operationResultBuffers[1].buffer);
  }
}

TEST(CompleteCandidatePreparationTest,
     ExternalLoadFanoutReplacesPerTileLoadsWithOneActualRootTransfer) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    mlir::DialectRegistry registry;
    registerCompilationDialects(registry);
    auto context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    auto ddrType = mlir::MemRefType::get(
        {2, extent, 128}, mlir::Float16Type::get(context.get()),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::DDR, MemLayout::Tensor));
    auto spmType = mlir::MemRefType::get(
        {2, extent, 128}, mlir::Float16Type::get(context.get()),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));

    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
    std::vector<StructuredMaterializationRelations> relations(3);
    std::vector<mlir::Value> buffers;
    modules.reserve(3);
    buffers.reserve(3);
    for (int64_t tileId : {1, 2, 3}) {
      auto module =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      mlir::OpBuilder moduleBuilder(module.getBodyRegion());
      auto function = moduleBuilder.create<mlir::func::FuncOp>(
          module.getLoc(), "tile",
          moduleBuilder.getFunctionType(mlir::TypeRange{ddrType},
                                        mlir::TypeRange{}));
      mlir::Block *entry = function.addEntryBlock();
      mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
      mlir::Value buffer =
          builder.create<mlir::memref::AllocOp>(module.getLoc(), spmType);
      builder.create<StorageLoadOp>(module.getLoc(), entry->getArgument(0),
                                    buffer);
      builder.create<mlir::func::ReturnOp>(module.getLoc());
      ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
      buffers.push_back(buffer);
      modules.emplace_back(module);
      (void)tileId;
    }

    ExternalLoadId rootLoad = makeExternalLoad(1, 1);
    ExternalLoadId secondLoad = makeExternalLoad(2, 2);
    ExternalLoadId thirdLoad = makeExternalLoad(3, 3);
    PeerTransferGraphPlan graph;
    graph.kind = PeerTransferGraphKind::ExternalLoadFanout;
    graph.actions = {MovementActionId(rootLoad), MovementActionId(secondLoad),
                     MovementActionId(thirdLoad)};
    llvm::sort(graph.actions);
    graph.hops = {{TileId(1), TileId(2)}, {TileId(1), TileId(3)}};
    graph.ddrRoot = rootLoad;
    MovementPlan movement;
    for (const MovementActionId &action : graph.actions) {
      ExternalLoadId load = std::get<ExternalLoadId>(action);
      movement.externalLoads.push_back(
          {load, PhysicalVersionId{RegionValueVersionId(load.destination)}});
    }
    llvm::sort(movement.externalLoads);
    movement.peerGraphs.push_back(graph);
    std::vector<MovementResourceDescription> movementResources;
    for (const MovementActionId &action : graph.actions)
      movementResources.push_back(
          {action, makeDomain(extent), mlir::Float16Type::get(context.get()),
           std::nullopt,
           std::get<ExternalLoadId>(action).destination.work.tile});

    ExecutionInstanceId execution = makeExecution(99, 1);
    EventId issue{ExecutionEventAction{execution},
                  PlannedEventKind::ComputeIssue};
    EventId completion{ExecutionEventAction{execution},
                       PlannedEventKind::Completion};
    PhysicalVersionId version{ExecutionResultValueId{execution, 0}};
    StorageObjectId object{StorageObjectOrigin(version)};
    BufferPlan bufferPlan;
    bufferPlan.storageObjects.push_back({object, TileId(1)});
    bufferPlan.versionBindings.push_back(
        {version, object, StorageBindingKind::Fresh});
    PipelineScopeId scope{{issue, completion}, {}};
    llvm::sort(scope.events);
    ExecutionStructurePlan structure{{SerializedExecutionStructure{scope}}};
    ScheduleDomainInput input;
    input.structure = structure;
    input.buffers = bufferPlan;
    input.events = {
        {issue,
         CardId(0),
         TileId(1),
         {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
        {completion, CardId(0), TileId(1), {}}};
    input.hardDependencies = {
        {issue, completion, EventDependencyReason::Completion}};
    input.completionObligations = {
        {issue, completion, CompletionProtocol::NCCParticipant, 0}};
    input.components = {{{issue, completion}}};
    ScheduleDomainResult scheduleDomain = buildScheduleDomain(input);
    ASSERT_TRUE(scheduleDomain.succeeded())
        << (scheduleDomain.failure ? scheduleDomain.failure->detail : "");
    ScheduleSuccessor schedule = scheduleDomain.domain->getFirstPlan();
    ASSERT_EQ(schedule.getKind(), ScheduleSuccessorKind::Plan);
    ASSERT_TRUE(schedule.getPlan());
    RepresentationPlan representations;
    representations.logicalValues.push_back({version.logicalValue, version});
    representations.physicalVersions.push_back({version, MemLayout::Tensor});
    relations.front().operationResultBuffers.push_back({0, 0, buffers[0]});
    CompleteCandidatePreparation preparation(
        std::move(*scheduleDomain.domain),
        PreparedCandidateEvents{input.events, input.hardDependencies,
                                input.completionObligations},
        representations, movement, movementResources, bufferPlan,
        {StorageResourceDescription{object, makeDomain(extent),
                                    mlir::Float16Type::get(context.get()),
                                    MemLayout::Tensor}},
        structure, *schedule.getPlan(), {{RegionExecutionId(execution), 0}});
    std::vector<CandidateTileDataflowIR> tiles;
    for (size_t index = 0; index < modules.size(); ++index)
      tiles.push_back({CardId(0), TileId(static_cast<int64_t>(index + 1)),
                       &modules[index], &relations[index]});

    CardExecutablePreparationFailure failure;
    ASSERT_TRUE(
        mlir::succeeded(preparation.prepareTileDataflow(tiles, failure)))
        << failure.detail;
    unsigned loads = 0;
    unsigned sends = 0;
    unsigned receives = 0;
    for (mlir::OwningOpRef<mlir::ModuleOp> &module : modules) {
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
      module->walk([&](StorageLoadOp) { ++loads; });
      module->walk([&](CommPeerSendOp) { ++sends; });
      module->walk([&](CommPeerRecvOp) { ++receives; });
    }
    EXPECT_EQ(loads, 1u);
    EXPECT_EQ(sends, 2u);
    EXPECT_EQ(receives, 2u);
  }
}

} // namespace
