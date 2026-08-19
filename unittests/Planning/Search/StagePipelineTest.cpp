//===- StagePipelineTest.cpp -----------------------------------------===//

#include "Wafer/Planning/Search/StagePipeline.h"

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Planning/Search/InstructionSchedule.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

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

struct PipelineInput {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
  SelectedBufferingScope scope;
};

PipelineInput makeInput(mlir::MLIRContext &context) {
  PipelineInput result;
  result.module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
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
  if (!result.module)
    return result;
  auto function = result.module->lookupSymbol<mlir::func::FuncOp>("main");
  mlir::memref::AllocOp slot;
  function.walk([&](mlir::memref::AllocOp allocation) { slot = allocation; });
  result.relations.operandBuffers.push_back({1, function.getArgument(0)});
  result.relations.operationResultBuffers.push_back({2, slot.getResult()});
  result.scope.requests.push_back(
      {/*producerNode=*/1, /*consumerNode=*/2, /*bufferCount=*/3,
       /*requireLocalDataflow=*/true, /*messages=*/{}});
  return result;
}

template <typename OpT> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpT) { ++count; });
  return count;
}

TEST(StagePipelineTest, EmptyScopeIsSerializedIdentity) {
  auto context = createContext();
  PipelineInput input = makeInput(*context);
  ASSERT_TRUE(input.module);
  mlir::Operation *original = input.module->getOperation();
  auto materialized = materializeStagePipelines(
      std::move(input.module), /*scopes=*/{}, std::move(input.relations));
  ASSERT_TRUE(mlir::succeeded(materialized));
  EXPECT_EQ(materialized->module->getOperation(), original);
  EXPECT_TRUE(materialized->pipelines.empty());
  EXPECT_EQ(materialized->slotAllocationCount, 0u);
}

TEST(StagePipelineTest,
     SelectedScopeMaterializesStagesAndInvalidatesOldScheduleDomain) {
  auto context = createContext();
  PipelineInput input = makeInput(*context);
  ASSERT_TRUE(input.module);
  std::string failureReason;
  auto oldSchedule = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *input.module}},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(oldSchedule)) << failureReason;
  CardInstructionScheduleAssignment oldAssignment =
      oldSchedule->getFirstAssignment();
  const unsigned originalRDMA =
      countOps<InstrRDMAOp>(input.module->getOperation());

  SelectedBufferMaterializationFailure failure;
  auto materialized = materializeStagePipelines(
      std::move(input.module), llvm::ArrayRef{input.scope},
      std::move(input.relations), &failure);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
  ASSERT_EQ(materialized->pipelines.size(), 1u);
  EXPECT_EQ(materialized->pipelines.front().slotCount, 3u);
  EXPECT_GE(materialized->pipelines.front().stageCount, 2u);
  EXPECT_EQ(materialized->pipelines.front().slotAllocationCount, 3u);
  EXPECT_EQ(materialized->slotAllocationCount, 3u);
  EXPECT_GT(countOps<InstrRDMAOp>(materialized->module->getOperation()),
            originalRDMA);
  EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
      materialized->module->getOperation(),
      materialized->materializationRelations)));
  EXPECT_FALSE(oldSchedule->contains(oldAssignment));

  auto newSchedule = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *materialized->module}},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(newSchedule)) << failureReason;
  EXPECT_TRUE(newSchedule->contains(newSchedule->getFirstAssignment()));
}

} // namespace
