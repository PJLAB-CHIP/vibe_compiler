//===- ScheduleMaterializationTest.cpp --------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ScheduleMaterialization.h"

#include "Wafer/Driver/CompilationInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <set>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

ExecutionInstanceId makeExecution(uint32_t anchor, TileId tile = TileId(0)) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  analysis::RootRegionWorkId work{root, tile};
  LogicalShardId shard{root, {anchor}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

EventId makeEvent(uint32_t anchor, PlannedEventKind kind,
                  TileId tile = TileId(0)) {
  return {ExecutionEventAction{makeExecution(anchor, tile)}, kind};
}

struct NCCProblem {
  ScheduleDomainInput input;
  EventId firstIssue;
  EventId firstCompletion;
  EventId handoffReady;
  EventId secondIssue;
  EventId secondCompletion;
};

NCCProblem makeNCCProblem() {
  NCCProblem problem;
  problem.firstIssue = makeEvent(0, PlannedEventKind::ComputeIssue);
  problem.firstCompletion = makeEvent(0, PlannedEventKind::Completion);
  problem.handoffReady = makeEvent(0, PlannedEventKind::BufferReady);
  problem.secondIssue = makeEvent(1, PlannedEventKind::ComputeIssue);
  problem.secondCompletion = makeEvent(1, PlannedEventKind::Completion);
  std::vector<EventId> events{problem.firstIssue, problem.firstCompletion,
                              problem.handoffReady, problem.secondIssue,
                              problem.secondCompletion};
  for (const EventId &event : events) {
    PlannedEvent planned{event, CardId(0), TileId(0), {}};
    if (event.kind == PlannedEventKind::ComputeIssue)
      planned.workerDomain = {NCCWorker::Worker0, NCCWorker::Worker1,
                              NCCWorker::Worker2};
    problem.input.events.push_back(std::move(planned));
  }
  problem.input.hardDependencies = {
      {problem.firstIssue, problem.firstCompletion,
       EventDependencyReason::Completion},
      {problem.firstCompletion, problem.handoffReady,
       EventDependencyReason::SSAValue},
      {problem.handoffReady, problem.secondIssue,
       EventDependencyReason::SSAValue},
      {problem.secondIssue, problem.secondCompletion,
       EventDependencyReason::Completion}};
  problem.input.completionObligations = {
      {problem.firstIssue, problem.firstCompletion,
       CompletionProtocol::NCCParticipant, 0},
      {problem.secondIssue, problem.secondCompletion,
       CompletionProtocol::NCCParticipant, 0}};
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(0), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  problem.input.buffers.storageObjects.push_back({object, TileId(0)});
  problem.input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  SPMRangeResource range{object, {{{0, 0, 0}, {2, 1025, 128}}}};
  problem.input.resourceUses = {
      {problem.firstIssue, range, ResourceUseMode::Write,
       ResourceIntervalKind::IssueToCompletion, problem.firstCompletion,
       ResourceKnowledge::Exact},
      {problem.secondIssue, range, ResourceUseMode::Read,
       ResourceIntervalKind::IssueToCompletion, problem.secondCompletion,
       ResourceKnowledge::Exact}};
  PipelineScopeId scope{events, {}};
  problem.input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  problem.input.components.push_back({events});
  return problem;
}

mlir::OwningOpRef<mlir::ModuleOp> parseNCCModule(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %buffer, %zero
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                 &context);
}

std::vector<ScheduleEventIRBinding> bindNCCModule(mlir::ModuleOp module,
                                                  const NCCProblem &problem) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  llvm::SmallVector<InstrFillOp, 2> fills;
  function.walk([&](InstrFillOp fill) { fills.push_back(fill); });
  EXPECT_EQ(fills.size(), 2u);
  mlir::Block *block = &function.getBody().front();
  return {{problem.firstIssue, TileId(0), block, {fills[0]}},
          {problem.firstCompletion, TileId(0), block, {}},
          {problem.handoffReady, TileId(0), block, {}},
          {problem.secondIssue, TileId(0), block, {fills[1]}},
          {problem.secondCompletion, TileId(0), block, {}}};
}

const CompletionPlacement *findPlacement(const ClosedSchedulePlan &plan,
                                         const EventId &issue) {
  auto placement =
      llvm::find_if(plan.completionPlacements, [&](const auto &candidate) {
        return candidate.issue == issue;
      });
  return placement == plan.completionPlacements.end() ? nullptr : &*placement;
}

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

TEST(ScheduleMaterializationTest,
     SameWorkerHandoffAvoidsIntermediateJoinAndCrossWorkerDoesNot) {
  auto context = createContext();
  NCCProblem problem = makeNCCProblem();
  ScheduleDomainResult result = buildScheduleDomain(problem.input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor sameWorker = result.domain->getFirstPlan();
  ASSERT_EQ(sameWorker.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(sameWorker.getPlan());
  const CompletionPlacement *first =
      findPlacement(*sameWorker.getPlan(), problem.firstIssue);
  const CompletionPlacement *second =
      findPlacement(*sameWorker.getPlan(), problem.secondIssue);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->participantMask, 0u);
  EXPECT_EQ(second->participantMask, 1u);

  auto module = parseNCCModule(*context);
  ASSERT_TRUE(module);
  std::vector<ScheduleEventIRBinding> bindings =
      bindNCCModule(*module, problem);
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*result.domain, *sameWorker.getPlan(),
                                     {ScheduleIRModule{TileId(0), *module}},
                                     bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  unsigned joins = 0;
  materialized.materialized->modules.front()->walk(
      [&](SyncNCCJoinOp) { ++joins; });
  EXPECT_EQ(joins, 1u);
  ASSERT_FALSE(materialized.materialized->completionGroups.empty());
  MaterializedCompletionGroup saved =
      std::move(materialized.materialized->completionGroups.back());
  materialized.materialized->completionGroups.pop_back();
  std::string verificationFailure;
  EXPECT_TRUE(mlir::failed(verifyMaterializedSchedule(
      *materialized.materialized, &verificationFailure)));
  EXPECT_FALSE(verificationFailure.empty());
  materialized.materialized->completionGroups.push_back(std::move(saved));

  ASSERT_TRUE(sameWorker.getCursor());
  ScheduleSuccessor crossWorker =
      result.domain->getNextPlan(*sameWorker.getCursor());
  ASSERT_EQ(crossWorker.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(crossWorker.getPlan());
  first = findPlacement(*crossWorker.getPlan(), problem.firstIssue);
  second = findPlacement(*crossWorker.getPlan(), problem.secondIssue);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->participantMask, 1u);
  EXPECT_EQ(second->participantMask, 2u);
  EXPECT_EQ(first->boundary.after, problem.handoffReady);

  auto crossModule = parseNCCModule(*context);
  ASSERT_TRUE(crossModule);
  bindings = bindNCCModule(*crossModule, problem);
  prepared = prepareScheduleMaterialization(
      *result.domain, *crossWorker.getPlan(),
      {ScheduleIRModule{TileId(0), *crossModule}}, bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  modules.clear();
  modules.push_back(std::move(crossModule));
  materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  joins = 0;
  materialized.materialized->modules.front()->walk(
      [&](SyncNCCJoinOp) { ++joins; });
  EXPECT_EQ(joins, 2u);
}

TEST(ScheduleMaterializationTest,
     SynchronousWritebackUsesSelectedWorkerWithoutDerivedJoin) {
  auto context = createContext();
  ScheduleDomainInput input;
  EventId issue = makeEvent(10, PlannedEventKind::ComputeIssue);
  EventId completion = makeEvent(10, PlannedEventKind::Completion);
  input.events = {
      {issue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {completion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {issue, completion, EventDependencyReason::Completion}};
  input.completionObligations = {
      {issue, completion, CompletionProtocol::NCCSynchronousWriteback, 1}};
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(10), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  input.buffers.storageObjects.push_back({object, TileId(0)});
  input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  PipelineScopeId scope{{issue, completion}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components.push_back({scope.events});

  ScheduleDomainResult domain = buildScheduleDomain(input);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ScheduleSuccessor worker0 = domain.domain->getFirstPlan();
  ASSERT_EQ(worker0.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(worker0.getCursor());
  ScheduleSuccessor worker1 = domain.domain->getNextPlan(*worker0.getCursor());
  ASSERT_EQ(worker1.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(worker1.getPlan());
  ASSERT_EQ(worker1.getPlan()->workerBindings.size(), 1u);
  EXPECT_EQ(worker1.getPlan()->workerBindings.front().worker,
            NCCWorker::Worker1);
  ASSERT_EQ(worker1.getPlan()->completionPlacements.size(), 1u);
  EXPECT_EQ(worker1.getPlan()->completionPlacements.front().participantMask,
            2u);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %input = memref.alloc()
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
    %value = memref.alloc()
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %index = memref.alloc()
        : memref<1xi32, #wafer.memory<spm, tensor>>
    wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
        %input into %value, %index {elem_count = 262400 : i64}
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
      into memref<1xf16, #wafer.memory<spm, tensor>>,
           memref<1xi32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto peripheral = *function.getOps<InstrPeripheralOp>().begin();
  mlir::Block *block = &function.getBody().front();
  std::vector<ScheduleEventIRBinding> bindings{
      {issue, TileId(0), block, {peripheral}},
      {completion, TileId(0), block, {}}};
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*domain.domain, *worker1.getPlan(),
                                     {ScheduleIRModule{TileId(0), *module}},
                                     bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  unsigned joins = 0;
  materialized.materialized->modules.front()->walk(
      [&](SyncNCCJoinOp) { ++joins; });
  EXPECT_EQ(joins, 0u);
  std::optional<NCCWorker> selectedWorker = getNCCIssueWorker(peripheral);
  ASSERT_TRUE(selectedWorker);
  EXPECT_EQ(*selectedWorker, NCCWorker::Worker1);
}

struct DTEProblem {
  ScheduleDomainInput input;
  EventId issue;
  EventId completion;
};

DTEProblem makeDTEProblem() {
  DTEProblem problem;
  problem.issue = makeEvent(20, PlannedEventKind::MovementIssue);
  problem.completion = makeEvent(20, PlannedEventKind::Completion);
  problem.input.events = {{problem.issue, CardId(0), TileId(0), {}},
                          {problem.completion, CardId(0), TileId(0), {}}};
  problem.input.hardDependencies = {
      {problem.issue, problem.completion, EventDependencyReason::Completion}};
  problem.input.completionObligations = {
      {problem.issue, problem.completion, CompletionProtocol::DirectDTE, 0}};
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(20), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  problem.input.buffers.storageObjects.push_back({object, TileId(0)});
  problem.input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  problem.input.resourceUses.push_back(
      {problem.issue, DTEReceiverFSMResource{TileId(0)},
       ResourceUseMode::CapacityUnits, ResourceIntervalKind::IssueToCompletion,
       problem.completion, ResourceKnowledge::Exact});
  PipelineScopeId scope{{problem.issue, problem.completion}, {}};
  problem.input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  problem.input.components.push_back({scope.events});
  return problem;
}

TEST(ScheduleMaterializationTest,
     DirectDTEUsesExactWaitAndCanonicalReceiverFSM) {
  auto context = createContext();
  DTEProblem problem = makeDTEProblem();
  ScheduleDomainResult result = buildScheduleDomain(problem.input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());
  EXPECT_TRUE(llvm::is_contained(
      first.getPlan()->resourceBindings,
      EventResourceBinding{
          problem.issue,
          ResourceInstanceId{DTEReceiverFSMResource{TileId(0)}, 0}}));
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 1 : i64, bytes = 524800 : i64,
         message = #wafer.dte_message<communication = 3, round = 0, slice = 0>}
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto receive = *function.getOps<InstrDTERecvOp>().begin();
  mlir::Block *block = &function.getBody().front();
  std::vector<ScheduleEventIRBinding> bindings{
      {problem.issue, TileId(0), block, {receive}},
      {problem.completion, TileId(0), block, {}}};
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*result.domain, *first.getPlan(),
                                     {ScheduleIRModule{TileId(0), *module}},
                                     bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  unsigned waits = 0;
  materialized.materialized->modules.front()->walk(
      [&](InstrDTEWaitOp) { ++waits; });
  EXPECT_EQ(waits, 1u);
}

TEST(ScheduleMaterializationTest,
     PrepareRejectsUnboundIssueAndPreexistingCompletionWithoutMutation) {
  auto context = createContext();
  NCCProblem problem = makeNCCProblem();
  ScheduleDomainResult result = buildScheduleDomain(problem.input);
  ASSERT_TRUE(result.succeeded());
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());

  auto unbound = parseNCCModule(*context);
  ASSERT_TRUE(unbound);
  std::vector<ScheduleEventIRBinding> bindings =
      bindNCCModule(*unbound, problem);
  bindings[3].operations.clear();
  const std::string unboundBefore = print(unbound->getOperation());
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*result.domain, *first.getPlan(),
                                     {ScheduleIRModule{TileId(0), *unbound}},
                                     bindings);
  ASSERT_FALSE(prepared.succeeded());
  EXPECT_EQ(print(unbound->getOperation()), unboundBefore);

  auto completed = parseNCCModule(*context);
  ASSERT_TRUE(completed);
  auto function = *completed->getOps<mlir::func::FuncOp>().begin();
  mlir::OpBuilder builder(function.getBody().front().getTerminator());
  builder.create<SyncNCCJoinOp>(function.getLoc(), NCCWorker::Worker0);
  bindings = bindNCCModule(*completed, problem);
  const std::string completedBefore = print(completed->getOperation());
  prepared = prepareScheduleMaterialization(
      *result.domain, *first.getPlan(),
      {ScheduleIRModule{TileId(0), *completed}}, bindings);
  ASSERT_FALSE(prepared.succeeded());
  EXPECT_EQ(print(completed->getOperation()), completedBefore);
}

TEST(ScheduleMaterializationTest,
     IndependentNCCAndDTECompletionsShareOneLatestBoundaryWithoutCrossClear) {
  auto context = createContext();
  ScheduleDomainInput input;
  EventId nccIssue = makeEvent(30, PlannedEventKind::ComputeIssue);
  EventId nccCompletion = makeEvent(30, PlannedEventKind::Completion);
  EventId dteIssue = makeEvent(31, PlannedEventKind::MovementIssue);
  EventId dteCompletion = makeEvent(31, PlannedEventKind::Completion);
  input.events = {
      {nccIssue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {nccCompletion, CardId(0), TileId(0), {}},
      {dteIssue, CardId(0), TileId(0), {}},
      {dteCompletion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {nccIssue, nccCompletion, EventDependencyReason::Completion},
      {dteIssue, dteCompletion, EventDependencyReason::Completion}};
  input.completionObligations = {
      {nccIssue, nccCompletion, CompletionProtocol::NCCParticipant, 0},
      {dteIssue, dteCompletion, CompletionProtocol::DirectDTE, 0}};
  input.resourceUses.push_back({dteIssue, DTEReceiverFSMResource{TileId(0)},
                                ResourceUseMode::CapacityUnits,
                                ResourceIntervalKind::IssueToCompletion,
                                dteCompletion, ResourceKnowledge::Exact});
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(30), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  input.buffers.storageObjects.push_back({object, TileId(0)});
  input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  PipelineScopeId scope{{nccIssue, nccCompletion, dteIssue, dteCompletion}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components.push_back({scope.events});
  ScheduleDomainResult domain = buildScheduleDomain(input);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ScheduleSuccessor first = domain.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());
  ASSERT_EQ(first.getPlan()->completionPlacements.size(), 2u);
  EXPECT_EQ(first.getPlan()->completionPlacements[0].boundary,
            first.getPlan()->completionPlacements[1].boundary);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>, f16
    %token = wafer.instr.dte_recv %buffer
        {peer = 1 : i64, bytes = 524800 : i64,
         message = #wafer.dte_message<communication = 4, round = 0, slice = 0>}
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto fill = *function.getOps<InstrFillOp>().begin();
  auto receive = *function.getOps<InstrDTERecvOp>().begin();
  mlir::Block *block = &function.getBody().front();
  std::vector<ScheduleEventIRBinding> bindings{
      {nccIssue, TileId(0), block, {fill}},
      {nccCompletion, TileId(0), block, {}},
      {dteIssue, TileId(0), block, {receive}},
      {dteCompletion, TileId(0), block, {}}};
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*domain.domain, *first.getPlan(),
                                     {ScheduleIRModule{TileId(0), *module}},
                                     bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  ASSERT_EQ(materialized.materialized->completionGroups.size(), 1u);
  EXPECT_EQ(
      materialized.materialized->completionGroups.front().operations.size(),
      2u);
}

TEST(ScheduleMaterializationTest,
     OverlappingReceivesUseDifferentFSMsAndOneExactGroupedWait) {
  auto context = createContext();
  ScheduleDomainInput input;
  EventId firstIssue = makeEvent(40, PlannedEventKind::MovementIssue);
  EventId firstCompletion = makeEvent(40, PlannedEventKind::Completion);
  EventId secondIssue = makeEvent(41, PlannedEventKind::MovementIssue);
  EventId secondCompletion = makeEvent(41, PlannedEventKind::Completion);
  input.events = {{firstIssue, CardId(0), TileId(0), {}},
                  {firstCompletion, CardId(0), TileId(0), {}},
                  {secondIssue, CardId(0), TileId(0), {}},
                  {secondCompletion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {firstIssue, firstCompletion, EventDependencyReason::Completion},
      {secondIssue, secondCompletion, EventDependencyReason::Completion},
      {secondIssue, firstCompletion, EventDependencyReason::TransferReady}};
  input.completionObligations = {
      {firstIssue, firstCompletion, CompletionProtocol::DirectDTE, 0},
      {secondIssue, secondCompletion, CompletionProtocol::DirectDTE, 0}};
  for (auto [issue, completion] : {std::pair{firstIssue, firstCompletion},
                                   std::pair{secondIssue, secondCompletion}})
    input.resourceUses.push_back({issue, DTEReceiverFSMResource{TileId(0)},
                                  ResourceUseMode::CapacityUnits,
                                  ResourceIntervalKind::IssueToCompletion,
                                  completion, ResourceKnowledge::Exact});
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(40), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  input.buffers.storageObjects.push_back({object, TileId(0)});
  input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  PipelineScopeId scope{
      {firstIssue, firstCompletion, secondIssue, secondCompletion}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components.push_back({scope.events});

  ScheduleDomainResult domain = buildScheduleDomain(input);
  ASSERT_TRUE(domain.succeeded())
      << (domain.failure ? domain.failure->detail : "");
  ScheduleSuccessor first = domain.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());
  std::set<uint32_t> lanes;
  for (const EventResourceBinding &binding : first.getPlan()->resourceBindings)
    if (std::holds_alternative<DTEReceiverFSMResource>(
            binding.instance.resource))
      lanes.insert(binding.instance.lane);
  EXPECT_EQ(lanes, (std::set<uint32_t>{0, 1}));
  ASSERT_EQ(first.getPlan()->completionPlacements.size(), 2u);
  EXPECT_EQ(first.getPlan()->completionPlacements[0].boundary,
            first.getPlan()->completionPlacements[1].boundary);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %first = memref.alloc()
        : memref<2x1031x128xf16, #wafer.memory<spm, tensor>>
    %second = memref.alloc()
        : memref<2x1031x128xf16, #wafer.memory<spm, tensor>>
    %first_token = wafer.instr.dte_recv %first
        {peer = 1 : i64, bytes = 527872 : i64,
         message = #wafer.dte_message<communication = 5, round = 0, slice = 0>}
        : memref<2x1031x128xf16, #wafer.memory<spm, tensor>> -> !async.token
    %second_token = wafer.instr.dte_recv %second
        {peer = 2 : i64, bytes = 527872 : i64,
         message = #wafer.dte_message<communication = 6, round = 0, slice = 0>}
        : memref<2x1031x128xf16, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  llvm::SmallVector<InstrDTERecvOp, 2> receives;
  function.walk([&](InstrDTERecvOp receive) { receives.push_back(receive); });
  ASSERT_EQ(receives.size(), 2u);
  mlir::Block *block = &function.getBody().front();
  std::vector<ScheduleEventIRBinding> bindings{
      {firstIssue, TileId(0), block, {receives[0]}},
      {firstCompletion, TileId(0), block, {}},
      {secondIssue, TileId(0), block, {receives[1]}},
      {secondCompletion, TileId(0), block, {}}};
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(*domain.domain, *first.getPlan(),
                                     {ScheduleIRModule{TileId(0), *module}},
                                     bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(modules), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  ASSERT_EQ(materialized.materialized->completionGroups.size(), 1u);
  ASSERT_EQ(
      materialized.materialized->completionGroups.front().operations.size(),
      1u);
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(
      materialized.materialized->completionGroups.front().operations.front());
  ASSERT_TRUE(wait);
  EXPECT_EQ(wait.getTokens().size(), 2u);
}

} // namespace
