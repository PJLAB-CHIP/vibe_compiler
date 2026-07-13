//===- DirectDTETransportTest.cpp - All-rank DTE acceptance tests --------===//

#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

class DirectDTETransportTest : public ::testing::Test {
protected:
  DirectDTETransportTest() {
    registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
    wafer::registerAllDialects(registry);
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

constexpr llvm::StringLiteral kSendRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kRecvRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kOverlappingSendRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %first = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 10, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %second = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 11, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %first, %second : !async.token, !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kFiveLiveReceiversRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %r0 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 20, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r1 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 21, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r2 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 22, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r3 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 23, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r4 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 24, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %r0, %r1, %r2, %r3, %r4 : !async.token, !async.token, !async.token, !async.token, !async.token
    return
  }
})mlir";

TEST_F(DirectDTETransportTest, MatchesCompleteDomainAndAttachesTypedBinding) {
  auto sendModule = parse(kSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);

  wafer::InstrDTESendOp send;
  sendModule->walk([&](wafer::InstrDTESendOp operation) { send = operation; });
  wafer::InstrDTERecvOp recv;
  recvModule->walk([&](wafer::InstrDTERecvOp operation) { recv = operation; });
  ASSERT_TRUE(send);
  ASSERT_TRUE(recv);
  ASSERT_TRUE(send.getBinding());
  ASSERT_TRUE(recv.getBinding());
  EXPECT_EQ(send.getBinding(), recv.getBinding());
  EXPECT_EQ(send.getBinding()->getAllocationProfile(),
            wafer::DTEAllocationProfile::Normal);
  EXPECT_EQ(send.getBinding()->getReceiverFsmId(), 0);
  EXPECT_EQ(send.getBinding()->getRemoteReceiverOffset(), 65792);
  EXPECT_EQ(send.getBinding()->getCompletionProfile(),
            wafer::DTECompletionProfile::SenderWaitReceiverFSM);
}

TEST_F(DirectDTETransportTest, MissingPeerFailsWithoutPublishingBinding) {
  auto sendModule = parse(kSendRank);
  ASSERT_TRUE(sendModule);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*sendModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([&](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, ByteMismatchFailsWithoutPublishingBinding) {
  auto sendModule = parse(kSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    operation.setBytesAttr(mlir::IntegerAttr::get(
        mlir::IntegerType::get(operation.getContext(), 64), 8));
  });
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([&](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
  recvModule->walk([&](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, UnplannedSPMRangeFailsClosed) {
  auto sendModule = parse(kSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  sendModule->walk([](mlir::memref::AllocOp operation) {
    operation->removeAttr(wafer::kWaferSPMOffsetAttrName);
  });
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  recvModule->walk([&](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, OverlappingNormalSendersFailClosed) {
  auto module = parse(kOverlappingSendRank);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  module->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, FifthOverlappingReceiverFailsClosed) {
  auto module = parse(kFiveLiveReceiversRank);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  module->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

} // namespace
