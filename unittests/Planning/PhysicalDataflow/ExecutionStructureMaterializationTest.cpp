//===- ExecutionStructureMaterializationTest.cpp ---------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureMaterialization.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

ExecutionInstanceId makeExecution(uint32_t anchor) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  analysis::RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

EventId makeEvent(uint32_t anchor) {
  return {ExecutionEventAction{makeExecution(anchor)},
          PlannedEventKind::ComputeIssue};
}

struct MaterializationInput {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ExecutionStructurePlan plan;
  ExecutionStructureLoopBinding binding;
};

MaterializationInput makeDistanceOneInput(mlir::MLIRContext &context,
                                          uint64_t extent) {
  const uint64_t tail = extent % 128 == 0 ? 0 : 1;
  std::string text = R"mlir(
module {
  func.func @main(%unused: tensor<2xEXTENTx128xf16>) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %seed = arith.constant 0 : i64
    %result = scf.for %iv = %c0 to %c7 step %c1
        iter_args(%carry = %seed) -> i64 {
      %index = arith.index_cast %iv : index to i64
      %offset = arith.addi %index, %index : i64
      %next = arith.addi %carry, %offset : i64
      %scaled = arith.muli %next, %next : i64
      scf.yield %scaled : i64
    }
    return %result : i64
  }
}
)mlir";
  const size_t marker = text.find("EXTENT");
  text.replace(marker, 6, std::to_string(extent));
  MaterializationInput input;
  input.module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!input.module)
    return input;
  mlir::scf::ForOp loop;
  mlir::arith::IndexCastOp index;
  mlir::arith::AddIOp offset;
  mlir::arith::AddIOp add;
  mlir::arith::MulIOp multiply;
  input.module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  input.module->walk(
      [&](mlir::arith::IndexCastOp operation) { index = operation; });
  input.module->walk([&](mlir::arith::AddIOp operation) {
    if (mlir::isa<mlir::BlockArgument>(operation.getLhs()))
      add = operation;
    else
      offset = operation;
  });
  input.module->walk(
      [&](mlir::arith::MulIOp operation) { multiply = operation; });

  EventId convertEvent = makeEvent(0);
  EventId addEvent = makeEvent(1);
  EventId multiplyEvent = makeEvent(2);
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(0)},
                                      TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, 1 + 7 + tail, 1};
  PipelineScopeId scope{{convertEvent, addEvent, multiplyEvent}, {recurrence}};
  llvm::sort(scope.events);
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = recurrence;
  pipeline.iteration = {/*recurrenceAxis=*/1, /*prefixCount=*/1,
                        /*steadyTripCount=*/7, tail};
  pipeline.eventStages = {{convertEvent, StageId(0)},
                          {addEvent, StageId(1)},
                          {multiplyEvent, StageId(2)}};
  llvm::sort(pipeline.eventStages);
  pipeline.dependences = {
      {convertEvent, addEvent, 0, PipelineDependenceKind::DataReady},
      {addEvent, multiplyEvent, 0, PipelineDependenceKind::DataReady},
      {multiplyEvent, multiplyEvent, 1, PipelineDependenceKind::DataReady}};
  llvm::sort(pipeline.dependences);
  pipeline.lowering = ExecutionStructureLowering::SCFDistanceOne;
  input.plan.scopes.push_back(pipeline);
  input.binding.scope = scope;
  input.binding.steadyLoop = loop;
  input.binding.eventOrder = {{convertEvent, {index}},
                              {addEvent, {offset}},
                              {multiplyEvent, {add, multiply}}};
  return input;
}

MaterializationInput makeFiniteUnrolledInput(mlir::MLIRContext &context) {
  MaterializationInput input;
  input.module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%unused: tensor<2x1025x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    scf.for %iv = %c0 to %c7 step %c1 {
      %token = async.execute {
        async.yield
      }
      async.await %token : !async.token
    }
    return
  }
}
)mlir",
                                                         &context);
  if (!input.module)
    return input;
  mlir::scf::ForOp loop;
  mlir::async::ExecuteOp issue;
  mlir::async::AwaitOp wait;
  input.module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  input.module->walk(
      [&](mlir::async::ExecuteOp operation) { issue = operation; });
  input.module->walk([&](mlir::async::AwaitOp operation) { wait = operation; });
  EventId issueEvent = makeEvent(10);
  EventId completionEvent = makeEvent(11);
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(10)},
                                      TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, 9, 1};
  PipelineScopeId scope{{issueEvent, completionEvent}, {recurrence}};
  llvm::sort(scope.events);
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = recurrence;
  pipeline.iteration = {/*recurrenceAxis=*/1, /*prefixCount=*/1,
                        /*steadyTripCount=*/7, /*tailCount=*/1};
  pipeline.eventStages = {{issueEvent, StageId(0)},
                          {completionEvent, StageId(1)}};
  llvm::sort(pipeline.eventStages);
  pipeline.dependences = {{issueEvent, completionEvent, 0,
                           PipelineDependenceKind::AsyncCompletion}};
  pipeline.completionObligations = {
      {issueEvent, completionEvent, CompletionProtocol::DirectDTE, 0}};
  pipeline.lowering = ExecutionStructureLowering::FiniteUnrolled;
  input.plan.scopes.push_back(pipeline);
  input.binding.scope = scope;
  input.binding.steadyLoop = loop;
  input.binding.eventOrder = {{issueEvent, {issue}}, {completionEvent, {wait}}};
  return input;
}

unsigned countLoops(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::scf::ForOp) { ++count; });
  return count;
}

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

TEST(ExecutionStructureMaterializationTest,
     AlignedAndRaggedDistanceOneProduceExactSCFPhases) {
  for (uint64_t extent : {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    MaterializationInput input = makeDistanceOneInput(*context, extent);
    ASSERT_TRUE(input.module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*input.module)));
    PreparedExecutionStructureResult prepared =
        prepareExecutionStructureMaterialization(*input.module, input.plan,
                                                 BufferPlan{}, {input.binding});
    ASSERT_TRUE(prepared.succeeded())
        << (prepared.failure ? prepared.failure->detail : "");
    MaterializedExecutionStructureResult materialized =
        materializeExecutionStructure(std::move(input.module),
                                      std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    ASSERT_EQ(materialized.materialized->scopes.size(), 1u);
    const MaterializedExecutionStructureScope &scope =
        materialized.materialized->scopes.front();
    EXPECT_EQ(scope.stageCount, 3u);
    EXPECT_EQ(scope.kernelDynamicTripCount, 5u);
    EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()),
              1u);
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(verifyMaterializedExecutionStructure(
        *materialized.materialized, &failureReason)))
        << failureReason;
  }
}

TEST(ExecutionStructureMaterializationTest,
     CrossStageAsyncCompletionUsesBoundedFiniteUnrollWithoutTokenBackedge) {
  auto context = createContext();
  MaterializationInput input = makeFiniteUnrolledInput(*context);
  ASSERT_TRUE(input.module);
  PreparedExecutionStructureResult prepared =
      prepareExecutionStructureMaterialization(*input.module, input.plan,
                                               BufferPlan{}, {input.binding});
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  MaterializedExecutionStructureResult materialized =
      materializeExecutionStructure(std::move(input.module),
                                    std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()), 0u);
  unsigned tokenBlockArguments = 0;
  materialized.materialized->module->walk([&](mlir::Block *block) {
    for (mlir::BlockArgument argument : block->getArguments())
      tokenBlockArguments +=
          mlir::isa<mlir::async::TokenType>(argument.getType());
  });
  EXPECT_EQ(tokenBlockArguments, 0u);
  ASSERT_EQ(materialized.materialized->scopes.size(), 1u);
  EXPECT_EQ(materialized.materialized->scopes.front().events.size(), 14u);
}

TEST(ExecutionStructureMaterializationTest,
     PrepareRejectsMissingMappingStageContradictionAndUnprovedExternalWrite) {
  auto context = createContext();
  MaterializationInput missing = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(missing.module);
  missing.binding.eventOrder.pop_back();
  PreparedExecutionStructureResult missingResult =
      prepareExecutionStructureMaterialization(*missing.module, missing.plan,
                                               BufferPlan{}, {missing.binding});
  ASSERT_FALSE(missingResult.succeeded());
  ASSERT_TRUE(missingResult.failure);
  EXPECT_EQ(missingResult.failure->kind,
            ExecutionStructureMaterializationFailureKind::BrokenContract);

  MaterializationInput contradictory = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(contradictory.module);
  auto &pipeline =
      std::get<PipelinedExecutionStructure>(contradictory.plan.scopes.front());
  pipeline.eventStages.front().stage = StageId(2);
  llvm::sort(pipeline.eventStages);
  PreparedExecutionStructureResult contradiction =
      prepareExecutionStructureMaterialization(*contradictory.module,
                                               contradictory.plan, BufferPlan{},
                                               {contradictory.binding});
  ASSERT_FALSE(contradiction.succeeded());
  ASSERT_TRUE(contradiction.failure);
  EXPECT_EQ(contradiction.failure->kind,
            ExecutionStructureMaterializationFailureKind::BrokenContract);

  auto external = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%buffer: memref<2x1025x128xf16>, %value: f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    scf.for %iv = %c0 to %c7 step %c1 {
      memref.store %value, %buffer[%c0, %iv, %c0]
          : memref<2x1025x128xf16>
      %loaded = memref.load %buffer[%c0, %iv, %c0]
          : memref<2x1025x128xf16>
    }
    return
  }
}
)mlir",
                                                          context.get());
  ASSERT_TRUE(external);
  mlir::scf::ForOp loop;
  mlir::memref::StoreOp store;
  mlir::memref::LoadOp load;
  external->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  external->walk([&](mlir::memref::StoreOp operation) { store = operation; });
  external->walk([&](mlir::memref::LoadOp operation) { load = operation; });
  EventId storeEvent = makeEvent(20);
  EventId loadEvent = makeEvent(21);
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(20)},
                                      TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, 9, 1};
  PipelineScopeId scope{{storeEvent, loadEvent}, {recurrence}};
  llvm::sort(scope.events);
  PipelinedExecutionStructure externalPipeline;
  externalPipeline.scope = scope;
  externalPipeline.recurrence = recurrence;
  externalPipeline.iteration = {1, 1, 7, 1};
  externalPipeline.eventStages = {{storeEvent, StageId(0)},
                                  {loadEvent, StageId(1)}};
  llvm::sort(externalPipeline.eventStages);
  externalPipeline.dependences = {
      {storeEvent, loadEvent, 0, PipelineDependenceKind::Effect}};
  ExecutionStructurePlan externalPlan{{externalPipeline}};
  ExecutionStructureLoopBinding externalBinding{
      scope, loop, {{storeEvent, {store}}, {loadEvent, {load}}}, {}};
  PreparedExecutionStructureResult externalResult =
      prepareExecutionStructureMaterialization(*external, externalPlan,
                                               BufferPlan{}, {externalBinding});
  ASSERT_FALSE(externalResult.succeeded());
  ASSERT_TRUE(externalResult.failure);
  EXPECT_EQ(externalResult.failure->kind,
            ExecutionStructureMaterializationFailureKind::Unsupported);
}

TEST(ExecutionStructureMaterializationTest,
     MultipleScopesPrepareAtomicallyAndMaterializeIndependently) {
  auto context = createContext();
  MaterializationInput input = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(input.module);
  mlir::func::FuncOp first =
      input.module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(first);
  mlir::OpBuilder builder(input.module->getContext());
  builder.setInsertionPointToEnd(input.module->getBody());
  mlir::IRMapping mapping;
  auto second = mlir::cast<mlir::func::FuncOp>(builder.clone(*first, mapping));
  second.setName("second");
  mlir::scf::ForOp secondLoop;
  mlir::arith::IndexCastOp secondIndex;
  mlir::arith::AddIOp secondOffset;
  mlir::arith::AddIOp secondAdd;
  mlir::arith::MulIOp secondMultiply;
  second.walk([&](mlir::scf::ForOp operation) { secondLoop = operation; });
  second.walk(
      [&](mlir::arith::IndexCastOp operation) { secondIndex = operation; });
  second.walk([&](mlir::arith::AddIOp operation) {
    if (mlir::isa<mlir::BlockArgument>(operation.getLhs()))
      secondAdd = operation;
    else
      secondOffset = operation;
  });
  second.walk(
      [&](mlir::arith::MulIOp operation) { secondMultiply = operation; });

  EventId convertEvent = makeEvent(100);
  EventId addEvent = makeEvent(101);
  EventId multiplyEvent = makeEvent(102);
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(100)},
                                      TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, 9, 1};
  PipelineScopeId scope{{convertEvent, addEvent, multiplyEvent}, {recurrence}};
  llvm::sort(scope.events);
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = recurrence;
  pipeline.iteration = {1, 1, 7, 1};
  pipeline.eventStages = {{convertEvent, StageId(0)},
                          {addEvent, StageId(1)},
                          {multiplyEvent, StageId(2)}};
  llvm::sort(pipeline.eventStages);
  pipeline.dependences = {
      {convertEvent, addEvent, 0, PipelineDependenceKind::DataReady},
      {addEvent, multiplyEvent, 0, PipelineDependenceKind::DataReady},
      {multiplyEvent, multiplyEvent, 1, PipelineDependenceKind::DataReady}};
  llvm::sort(pipeline.dependences);
  ExecutionStructureLoopBinding secondBinding{
      scope,
      secondLoop,
      {{convertEvent, {secondIndex}},
       {addEvent, {secondOffset}},
       {multiplyEvent, {secondAdd, secondMultiply}}},
      {}};
  input.plan.scopes.push_back(pipeline);
  std::vector<ExecutionStructureLoopBinding> bindings{input.binding,
                                                      secondBinding};
  const std::string before = print(input.module->getOperation());
  bindings.back().eventOrder.back().operations.pop_back();
  PreparedExecutionStructureResult broken =
      prepareExecutionStructureMaterialization(*input.module, input.plan,
                                               BufferPlan{}, bindings);
  ASSERT_FALSE(broken.succeeded());
  EXPECT_EQ(print(input.module->getOperation()), before);

  bindings.back() = secondBinding;
  PreparedExecutionStructureResult prepared =
      prepareExecutionStructureMaterialization(*input.module, input.plan,
                                               BufferPlan{}, bindings);
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  MaterializedExecutionStructureResult materialized =
      materializeExecutionStructure(std::move(input.module),
                                    std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  EXPECT_EQ(materialized.materialized->scopes.size(), 2u);
  EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()), 2u);
}

TEST(ExecutionStructureMaterializationTest,
     VerifierRejectsStaleMaterializedRelations) {
  auto context = createContext();
  MaterializationInput input = makeDistanceOneInput(*context, 1024);
  ASSERT_TRUE(input.module);
  PreparedExecutionStructureResult prepared =
      prepareExecutionStructureMaterialization(*input.module, input.plan,
                                               BufferPlan{}, {input.binding});
  ASSERT_TRUE(prepared.succeeded());
  MaterializedExecutionStructureResult materialized =
      materializeExecutionStructure(std::move(input.module),
                                    std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  ASSERT_FALSE(materialized.materialized->scopes.front().events.empty());
  materialized.materialized->scopes.front().events.front().operation = nullptr;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(verifyMaterializedExecutionStructure(
      *materialized.materialized, &failureReason)));
  EXPECT_FALSE(failureReason.empty());
}

} // namespace
