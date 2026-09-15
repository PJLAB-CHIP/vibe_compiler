//===- RequiredNCCJoinPlacementTest.cpp
//------------------------------------===//

#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
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
// Tiny payload isolates dynamic control flow; realistic entry hazards are
// covered by EntryOnlyWaitsStayOutsideTheSteadyLoop below.
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
  llvm::SmallVector<wafer::SyncNCCJoinOp> joins;
  module->walk([&](wafer::SyncNCCJoinOp join) { joins.push_back(join); });
  ASSERT_EQ(joins.size(), 2u);
  EXPECT_EQ(joins.front().getParticipants(), llvm::ArrayRef<int64_t>{0});
  auto guard = joins.front()->getParentOfType<mlir::scf::IfOp>();
  ASSERT_TRUE(guard);
  auto comparison = guard.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  ASSERT_TRUE(comparison);
  EXPECT_EQ(comparison.getPredicate(), mlir::arith::CmpIPredicate::eq);
  EXPECT_EQ(joins.back().getParticipants(), llvm::ArrayRef<int64_t>({0, 1}));
  EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(joins.back()->getNextNode()));
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

TEST(RequiredNCCJoinPlacementTest, EntryOnlyWaitsStayOutsideTheSteadyLoop) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t length : {1024, 1025, 1031})
    for (unsigned kind = 0; kind < 4; ++kind) {
      SCOPED_TRACE(length);
      SCOPED_TRACE(kind);
      std::string source;
      llvm::raw_string_ostream os(source);
      std::string type = "memref<1x" + std::to_string(length) + "x64xf16, #wafer.memory<spm, tensor>>";
      os << "module { func.func @entry(%zero: f16)"
         << (kind < 2 ? " -> f16" : "") << " {\n"
         << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n"
         << "%end = arith.constant " << length << " : index\n"
         << "%a = memref.alloc() : " << type << "\n"
         << "%b = memref.alloc() : " << type << "\n"
         << "wafer.instr.fill %a, %zero : " << type << ", f16\n"
         << (kind < 2 ? "%sum = " : "")
         << "scf.for %i = %c0 to %end step %c1"
         << (kind < 2 ? " iter_args(%acc = %zero) -> (f16)" : "") << " {\n";
      if (kind < 2) {
        os << "%v = memref.load %a[%c0, %i, %c0] : " << type << "\n"
           << "%next = arith.addf %acc, %v : f16\n";
        if (kind == 1)
          os << "wafer.instr.fill %b, %zero : " << type << ", f16\n";
      } else {
        os << "wafer.instr.fill %a, %zero {worker = #wafer.ncc_worker<worker"
           << (kind == 2 ? 1 : 0) << ">} : " << type << ", f16\n";
      }
      if (kind < 2)
        os << "scf.yield %next : f16\n";
      os << "}\nreturn" << (kind < 2 ? " %sum : f16" : "") << "\n}}";
      auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      llvm::SmallVector<wafer::SyncNCCJoinOp> joins;
      module->walk([&](wafer::SyncNCCJoinOp join) {
        EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
        joins.push_back(join);
      });
      ASSERT_EQ(joins.size(), kind == 0 || kind == 3 ? 1u : 2u);
      EXPECT_EQ(joins.front().getParticipants(), llvm::ArrayRef<int64_t>{0});
      if (kind != 3) {
        EXPECT_TRUE(mlir::isa<mlir::scf::ForOp>(joins.front()->getNextNode()));
      }
      if (kind == 2) {
        EXPECT_EQ(joins.back().getParticipants(), llvm::ArrayRef<int64_t>{1});
      }
      std::string before;
      llvm::raw_string_ostream beforeStream(before);
      module->print(beforeStream);
      ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
      std::string after;
      llvm::raw_string_ostream afterStream(after);
      module->print(afterStream);
      EXPECT_EQ(before, after);
    }
}

TEST(RequiredNCCJoinPlacementTest, KcoreStoresReleaseOnceBeforeTheirDMAConsumer) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t length : {1024, 1025, 1031}) {
    std::string spm = "memref<1x" + std::to_string(length) + "x1xi64, #wafer.memory<spm, tensor>>";
    std::string ddr = "memref<1x" + std::to_string(length) + "x1xi64, #wafer.memory<ddr, tensor>>";
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module { func.func @entry(%out: " << ddr << ") {\n"
       << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n"
       << "%end = arith.constant " << length << " : index\n"
       << "%a = memref.alloc() : " << spm << "\n"
       << "scf.for %i = %c0 to %end step %c1 {\n"
       << "%v = arith.index_cast %i : index to i64\n"
       << "memref.store %v, %a[%c0, %i, %c0] : " << spm << "\n}\n"
       << "wafer.instr.wdma %a to %out {byte_count = " << length * 8
       << " : i64, inner_bytes = " << length * 8
       << " : i64, dst_strides = array<i64: 0, 0, 0>, dst_iterations = array<i64: 1, 1, 1>} : "
       << spm << " to " << ddr << "\nreturn\n}}";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
    unsigned releases = 0;
    module->walk([&](mlir::LLVM::FenceOp fence) {
      EXPECT_FALSE(fence->getParentOfType<mlir::scf::ForOp>());
      EXPECT_TRUE(mlir::isa<wafer::InstrWDMAOp>(fence->getNextNode()));
      ++releases;
    });
    EXPECT_EQ(releases, 1u);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

} // namespace
