//===- DirectDTETransportTest.cpp - All-rank DTE acceptance tests --------===//

#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class DirectDTETransportTest : public ::testing::Test {
protected:
  DirectDTETransportTest() {
    registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
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

static std::string makeControlledRank(bool isSend, int64_t peer,
                                      int64_t spmOffset,
                                      llvm::StringRef functionArguments,
                                      llvm::StringRef controlPrefix,
                                      llvm::StringRef controlSuffix,
                                      bool addStaticTail) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main"
     << functionArguments
     << " {\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c2 = arith.constant 2 : index\n"
        "    %c4 = arith.constant 4 : index\n"
        "    %c8 = arith.constant 8 : index\n"
        "    %buffer = memref.alloc() {wafer.spm.offset = "
     << "#wafer.spm_offset<" << spmOffset
     << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
     << controlPrefix << "    %token0 = wafer.instr.dte_"
     << (isSend ? "send" : "recv") << " %buffer"
     << " {peer = " << peer
     << " : i64, bytes = 16 : i64, message = "
        "#wafer.dte_message<communication = 9, phase = collective_permute, "
        "round = 2, slice = 0>} : "
        "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
        "    wafer.instr.dte_wait %token0 : !async.token\n"
     << controlSuffix;
  if (addStaticTail) {
    os << "    %token1 = wafer.instr.dte_" << (isSend ? "send" : "recv")
       << " %buffer"
       << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 9, phase = collective_permute, "
          "round = 2, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %token1 : !async.token\n";
  }
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

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

constexpr llvm::StringLiteral kLoopEscapingTokenSendRank = R"mlir(
module {
  func.func @main(%initial: !async.token) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %escaped = scf.for %index = %c0 to %c4 step %c1
        iter_args(%carried = %initial) -> !async.token {
      %token = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 16 : i64,
           message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
          : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
      scf.yield %token : !async.token
    }
    wafer.instr.dte_wait %escaped : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kInterveningBufferAccessSendRank = R"mlir(
module {
  func.func @main() {
    %index = arith.constant 0 : index
    %value = arith.constant 0.0 : f32
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    memref.store %value, %buffer[%index]
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
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

TEST_F(DirectDTETransportTest,
       MatchesNestedStaticLoopInstancesAndSeparateStaticTail) {
  constexpr llvm::StringLiteral kNestedPrefix = R"mlir(
    scf.for %outer = %c0 to %c8 step %c2 {
      scf.for %inner = %c0 to %c4 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kNestedSuffix = R"mlir(
      }
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);

  unsigned sendCount = 0;
  unsigned recvCount = 0;
  sendModule->walk([&](wafer::InstrDTESendOp operation) {
    ++sendCount;
    EXPECT_TRUE(operation.getBinding());
  });
  recvModule->walk([&](wafer::InstrDTERecvOp operation) {
    ++recvCount;
    EXPECT_TRUE(operation.getBinding());
  });
  EXPECT_EQ(sendCount, 2u);
  EXPECT_EQ(recvCount, 2u);
}

TEST_F(DirectDTETransportTest, ReorderedLoopAndStaticTailFailClosed) {
  constexpr llvm::StringLiteral kNestedPrefix = R"mlir(
    scf.for %outer = %c0 to %c8 step %c2 {
      scf.for %inner = %c0 to %c4 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kNestedSuffix = R"mlir(
      }
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);

  mlir::scf::ForOp outerLoop;
  wafer::InstrDTERecvOp staticTail;
  recvModule->walk([&](mlir::scf::ForOp loop) {
    if (!loop->getParentOfType<mlir::scf::ForOp>())
      outerLoop = loop;
  });
  recvModule->walk([&](wafer::InstrDTERecvOp recv) {
    if (!recv->getParentOfType<mlir::scf::ForOp>())
      staticTail = recv;
  });
  ASSERT_TRUE(outerLoop);
  ASSERT_TRUE(staticTail);
  mlir::Operation *tailWait = staticTail.getToken().use_begin()->getOwner();
  staticTail->moveBefore(outerLoop);
  tailWait->moveBefore(outerLoop);

  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, MismatchedStaticLoopBoundsFailClosed) {
  constexpr llvm::StringLiteral kSendPrefix = R"mlir(
    scf.for %outer = %c0 to %c8 step %c2 {
      scf.for %inner = %c0 to %c8 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kRecvPrefix = R"mlir(
    scf.for %outer = %c0 to %c8 step %c2 {
      scf.for %inner = %c0 to %c4 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kSuffix = R"mlir(
      }
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kSendPrefix, kSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kRecvPrefix, kSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, DynamicLoopControlFailsClosed) {
  constexpr llvm::StringLiteral kDynamicPrefix = R"mlir(
    scf.for %outer = %c0 to %bound step %c2 {
)mlir";
  constexpr llvm::StringLiteral kLoopSuffix = R"mlir(
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"(%bound: index)", kDynamicPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"(%bound: index)", kDynamicPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
}

TEST_F(DirectDTETransportTest, ConditionalControlInstanceFailsClosed) {
  constexpr llvm::StringLiteral kConditionalPrefix = R"mlir(
    scf.if %condition {
)mlir";
  constexpr llvm::StringLiteral kConditionalSuffix = R"mlir(
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"(%condition: i1)", kConditionalPrefix,
      kConditionalSuffix, /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"(%condition: i1)", kConditionalPrefix,
      kConditionalSuffix, /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
}

TEST_F(DirectDTETransportTest, MismatchedStructuredLoopFamilyFailsClosed) {
  constexpr llvm::StringLiteral kSendPrefix = R"mlir(
    scf.for %tile = %c0 to %c8 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kRecvPrefix = R"mlir(
    scf.for %unrelated = %c0 to %c8 step %c2 {
    }
    scf.for %tile = %c0 to %c8 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kLoopSuffix = R"mlir(
    }
)mlir";
  auto sendModule = parse(makeControlledRank(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kSendPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledRank(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kRecvPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
}

TEST_F(DirectDTETransportTest, LoopEscapingIssueTokenFailsClosed) {
  auto sendModule = parse(kLoopEscapingTokenSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, InterveningIssueBufferAccessFailsClosed) {
  auto sendModule = parse(kInterveningBufferAccessSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
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
