//===- SelectedBufferMaterializationTest.cpp ------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer::compiler::testing;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

struct ExactBufferInput {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  wafer::StructuredMaterializationRelations relations;
  wafer::compiler::detail::SelectedBufferRequest request;
};

ExactBufferInput makeLocalInput(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @pipeline(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.for %iv = %c0 to %c5 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir",
                                                        &context);
  ExactBufferInput result;
  result.module = std::move(module);
  if (!result.module)
    return result;
  mlir::func::FuncOp function =
      result.module->lookupSymbol<mlir::func::FuncOp>("pipeline");
  mlir::memref::AllocOp slot;
  function.walk([&](mlir::memref::AllocOp allocation) { slot = allocation; });
  result.relations.operandBuffers.push_back(
      {/*structuredNodeId=*/1, function.getArgument(0)});
  result.relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/2, slot.getResult()});
  result.request.producerNode = 1;
  result.request.consumerNode = 2;
  result.request.requireLocalDataflow = true;
  return result;
}

TEST(SelectedBufferMaterializationTest,
     MaterializesSelectedDoubleTripleAndFourSlotRotation) {
  auto context = createContext();
  for (uint32_t slotCount : {2U, 3U, 4U}) {
    ExactBufferInput input = makeLocalInput(*context);
    ASSERT_TRUE(input.module);
    input.request.bufferCount = slotCount;
    wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
    auto materialized = wafer::compiler::detail::materializeSelectedBuffering(
        std::move(input.module), llvm::ArrayRef{input.request},
        std::move(input.relations), &failure);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
    EXPECT_EQ(materialized->slotAllocationCount, slotCount);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized->module)));
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
            materialized->module->getOperation(),
            materialized->materializationRelations)));
  }
}

TEST(SelectedBufferMaterializationTest,
     RejectsMultiplicityLargerThanTheActualStaticLoop) {
  auto context = createContext();
  ExactBufferInput input = makeLocalInput(*context);
  ASSERT_TRUE(input.module);
  input.request.bufferCount = 6;
  wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::materializeSelectedBuffering(
          std::move(input.module), llvm::ArrayRef{input.request},
          std::move(input.relations), &failure)));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::SelectedBufferMaterializationFailureKind::
                TripCountTooSmall);
}

TEST(SelectedBufferMaterializationTest,
     RejectsLoopExternalCrossStageWriteWithoutAliasProof) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @pipeline(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %slot = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    scf.for %iv = %c0 to %c4 step %c1 {
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    memref.dealloc %slot : memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::func::FuncOp function =
      module->lookupSymbol<mlir::func::FuncOp>("pipeline");
  mlir::memref::AllocOp slot;
  function.walk([&](mlir::memref::AllocOp allocation) { slot = allocation; });
  wafer::StructuredMaterializationRelations relations;
  relations.operandBuffers.push_back({1, function.getArgument(0)});
  relations.operationResultBuffers.push_back({2, slot.getResult()});
  wafer::compiler::detail::SelectedBufferRequest request;
  request.producerNode = 1;
  request.consumerNode = 2;
  request.bufferCount = 2;
  request.requireLocalDataflow = true;
  wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::materializeSelectedBuffering(
          std::move(module), llvm::ArrayRef{request}, std::move(relations),
          &failure)));
  EXPECT_NE(failure.detail.find("cross-stage write hazard"), std::string::npos);
}

TEST(SelectedBufferMaterializationTest,
     PreservesDirectDTEIssueWaitAndLastConsumerRelease) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @pipeline() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      %sent = wafer.instr.dte_send %slot
          {peer = 0 : i64, bytes = 8 : i64,
           message = #wafer.dte_message<communication = 30, round = 0, slice = 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.dte_wait %sent : !async.token
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::memref::AllocOp slot;
  module->walk([&](mlir::memref::AllocOp allocation) { slot = allocation; });
  ASSERT_TRUE(slot);
  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back({1, slot.getResult()});
  wafer::compiler::detail::SelectedBufferRequest request;
  request.producerNode = 1;
  request.consumerNode = 2;
  request.bufferCount = 3;
  request.messages.push_back(
      {wafer::compiler::detail::SelectedBufferMessageDirection::Send,
       /*communicationId=*/30, /*payloadSlice=*/0});
  wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
  auto materialized = wafer::compiler::detail::materializeSelectedBuffering(
      std::move(module), llvm::ArrayRef{request}, std::move(relations),
      &failure);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
  EXPECT_EQ(materialized->slotAllocationCount, 3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized->module)));
  wafer::InstrDTESendOp send;
  wafer::InstrDTEWaitOp wait;
  materialized->module->walk(
      [&](wafer::InstrDTESendOp operation) { send = operation; });
  materialized->module->walk(
      [&](wafer::InstrDTEWaitOp operation) { wait = operation; });
  ASSERT_TRUE(send);
  ASSERT_TRUE(wait);
  ASSERT_EQ(wait.getTokens().size(), 1u);
  EXPECT_EQ(wait.getTokens().front().getDefiningOp(), send.getOperation());
  EXPECT_TRUE(send->isBeforeInBlock(wait));
  unsigned allocations = 0;
  unsigned releases = 0;
  materialized->module->walk([&](mlir::memref::AllocOp) { ++allocations; });
  materialized->module->walk([&](mlir::memref::DeallocOp) { ++releases; });
  EXPECT_EQ(allocations, 3u);
  EXPECT_EQ(releases, 3u);
}

TEST(SelectedBufferMaterializationTest,
     RejectsLogicalEndpointsInDifferentStaticLoops) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @pipeline() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %producer = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %producer, %producer into %producer
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      memref.dealloc %producer : memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    scf.for %iv = %c0 to %c3 step %c1 {
      %consumer = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %consumer, %consumer into %consumer
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      memref.dealloc %consumer : memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::memref::AllocOp, 2> allocations;
  module->walk([&](mlir::memref::AllocOp allocation) {
    allocations.push_back(allocation);
  });
  ASSERT_EQ(allocations.size(), 2u);
  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back({1, allocations[0].getResult()});
  relations.operandBuffers.push_back({2, allocations[1].getResult()});
  wafer::compiler::detail::SelectedBufferRequest request;
  request.producerNode = 1;
  request.consumerNode = 2;
  request.bufferCount = 2;
  request.requireLocalDataflow = true;
  wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::materializeSelectedBuffering(
          std::move(module), llvm::ArrayRef{request}, std::move(relations),
          &failure)));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::SelectedBufferMaterializationFailureKind::
                NoExactLoop);
  EXPECT_NE(failure.detail.find("no static loop containing every exact"),
            std::string::npos);
}

} // namespace
