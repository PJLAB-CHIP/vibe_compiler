//===- RequiredNCCJoinPlacementTest.cpp
//------------------------------------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

static constexpr llvm::StringLiteral kUnsupportedDynamicLoop = R"mlir(
module {
  func.func @dynamic_loop(%upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [2]
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    scf.for %index = %c0 to %upper step %c1 {
      wafer.instr.fill %buffer, %zero
          {worker = #wafer.ncc_worker<worker1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    }
    return
  }
}
)mlir";

struct MarkAfterFailedNCCPass
    : public mlir::PassWrapper<MarkAfterFailedNCCPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() final {
    getOperation()->setAttr("test.after_failed_ncc",
                            mlir::UnitAttr::get(getOperation().getContext()));
  }
};

TEST(RequiredNCCJoinPlacementTest,
     DynamicLoopRebuildClosesEntryBackedgeAndReturn) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kUnsupportedDynamicLoop,
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
  unsigned joins = 0;
  module->walk([&](wafer::SyncNCCJoinOp) { ++joins; });
  EXPECT_EQ(joins, 1u);
}

TEST(RequiredNCCJoinPlacementTest,
     SuccessfulDynamicLoopRebuildContinuesNestedPipeline) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kUnsupportedDynamicLoop,
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  mlir::PassManager manager(&context);
  manager.addNestedPass<mlir::func::FuncOp>(
      wafer::createRebuildRequiredNCCJoinsPass());
  manager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<MarkAfterFailedNCCPass>());
  EXPECT_TRUE(mlir::succeeded(manager.run(*module)));
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  EXPECT_TRUE(function->hasAttr("test.after_failed_ncc"));
}

TEST(RequiredNCCJoinPlacementTest,
     RankThreeRaggedCrossWorkerConflictUsesExactParticipants) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @cross_worker(%zero: f16) {
    %buffer = memref.alloc()
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.elementwise <add> %buffer, %buffer into %buffer
        {worker = #wafer.ncc_worker<worker1>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>,
          memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      into memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  module->walk([&](wafer::SyncNCCJoinOp join) { joins.push_back(join); });
  ASSERT_EQ(joins.size(), 2u);
  EXPECT_EQ(joins[0].getParticipants(), llvm::ArrayRef<int64_t>({0}));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(joins[0]->getNextNode()));
  EXPECT_EQ(joins[1].getParticipants(), llvm::ArrayRef<int64_t>({1}));
  EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(joins[1]->getNextNode()));
}

TEST(RequiredNCCJoinPlacementTest,
     RankThreeRaggedDistinctDTEBufferClosesPendingNCCDomain) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @ncc_to_dte(%zero: f16) {
    %computed = memref.alloc()
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    %transport = memref.alloc()
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %computed, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
    %token = wafer.instr.dte_send %transport
        {peer = 1 : i64, bytes = 262400 : i64,
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  module->walk([&](wafer::SyncNCCJoinOp join) { joins.push_back(join); });
  ASSERT_EQ(joins.size(), 1u);
  EXPECT_EQ(joins.front().getParticipants(), llvm::ArrayRef<int64_t>({0}));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(joins.front()->getNextNode()));
}

} // namespace
