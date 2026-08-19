//===- NCCCompletionAnalysisTest.cpp ---------------------------------===//

#include "Wafer/Analysis/Scheduling/NCCCompletionAnalysis.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class NCCCompletionAnalysisTest : public ::testing::Test {
protected:
  NCCCompletionAnalysisTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    wafer::registerWaferCoreDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NCCCompletionAnalysisTest,
       ExpandsDirectCallsAndRecordsPendingStateAtEachOperation) {
  auto module = parse(R"mlir(
module {
  func.func private @worker0(%buffer: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    func.call @worker0(%buffer)
        : (memref<4xf16, #wafer.memory<spm, tensor>>) -> ()
    wafer.instr.ncc_join [0, 1]
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto analysis =
      wafer::analysis::NCCCompletionAnalysis::create(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(analysis)) << failureReason;
  EXPECT_EQ(analysis->getSummary().issuedWorkerMask, UINT32_C(0x3));
  EXPECT_TRUE(analysis->getSummary().hasCrossWorkerWindow);
  mlir::func::CallOp call;
  wafer::SyncNCCJoinOp join;
  module->walk([&](mlir::func::CallOp operation) { call = operation; });
  module->walk([&](wafer::SyncNCCJoinOp operation) { join = operation; });
  auto callState = analysis->getOperationState(call);
  auto joinState = analysis->getOperationState(join);
  ASSERT_TRUE(callState);
  ASSERT_TRUE(joinState);
  EXPECT_EQ(callState->pendingBefore, UINT32_C(0x2));
  EXPECT_EQ(callState->pendingAfter, UINT32_C(0x3));
  EXPECT_EQ(joinState->pendingBefore, UINT32_C(0x3));
  EXPECT_EQ(joinState->pendingAfter, 0u);
}

TEST_F(NCCCompletionAnalysisTest, RecomputesWorkerFactsAfterTypedMutation) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto before = wafer::analysis::NCCCompletionAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(before));
  EXPECT_EQ(before->getSummary().issuedWorkerMask, UINT32_C(0x1));
  wafer::InstrFillOp fill;
  module->walk([&](wafer::InstrFillOp operation) { fill = operation; });
  ASSERT_TRUE(fill);
  ASSERT_TRUE(mlir::succeeded(
      wafer::setNCCIssueWorker(fill, wafer::NCCWorker::Worker1)));
  wafer::SyncNCCJoinOp join;
  module->walk([&](wafer::SyncNCCJoinOp operation) { join = operation; });
  ASSERT_TRUE(join);
  join.setParticipantsAttr(
      mlir::DenseI64ArrayAttr::get(context.get(), llvm::ArrayRef<int64_t>{1}));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto after = wafer::analysis::NCCCompletionAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(after));
  EXPECT_EQ(after->getSummary().issuedWorkerMask, UINT32_C(0x2));
}

TEST_F(NCCCompletionAnalysisTest,
       JoinsOnEveryIfAndForPathLeaveNoPendingWorker) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    %condition = arith.constant true
    scf.if %condition {
      wafer.instr.fill %buffer, %zero
          {worker = #wafer.ncc_worker<worker0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
    } else {
      wafer.instr.fill %buffer, %zero
          {worker = #wafer.ncc_worker<worker1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [1]
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      wafer.instr.fill %buffer, %zero
          {worker = #wafer.ncc_worker<worker2>}
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [2]
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto analysis = wafer::analysis::NCCCompletionAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(analysis));
  EXPECT_EQ(analysis->getSummary().issuedWorkerMask, UINT32_C(0x7));
  EXPECT_FALSE(analysis->getSummary().hasCrossWorkerWindow);
  mlir::scf::IfOp branch;
  mlir::scf::ForOp loop;
  module->walk([&](mlir::scf::IfOp operation) { branch = operation; });
  module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  auto branchState = analysis->getOperationState(branch);
  auto loopState = analysis->getOperationState(loop);
  ASSERT_TRUE(branchState);
  ASSERT_TRUE(loopState);
  EXPECT_EQ(branchState->pendingAfter, 0u);
  EXPECT_EQ(loopState->pendingAfter, 0u);
}

TEST_F(NCCCompletionAnalysisTest, RecursiveCallGraphFailsClosed) {
  auto module = parse(R"mlir(
module {
  func.func private @recursive() {
    func.call @recursive() : () -> ()
    return
  }
  func.func @main() {
    func.call @recursive() : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(
      wafer::analysis::NCCCompletionAnalysis::create(*module, &failureReason)));
  EXPECT_NE(failureReason.find("recursive"), std::string::npos);
}

} // namespace
