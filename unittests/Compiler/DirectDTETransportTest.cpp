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

struct LinearTransportSite {
  bool isSend = false;
  int64_t peer = -1;
  int64_t spmOffset = -1;
  int64_t communication = -1;
};

static std::string
makeLinearTransportRank(llvm::ArrayRef<LinearTransportSite> sites,
                        bool waitImmediately) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n";
  for (auto [index, site] : llvm::enumerate(sites))
    os << "    %buffer" << index
       << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
       << site.spmOffset << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n";
  for (auto [index, site] : llvm::enumerate(sites)) {
    os << "    %token" << index << " = wafer.instr.dte_"
       << (site.isSend ? "send" : "recv") << " %buffer" << index
       << " {peer = " << site.peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << site.communication
       << ", phase = collective_permute, round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n";
    if (waitImmediately)
      os << "    wafer.instr.dte_wait %token" << index << " : !async.token\n";
  }
  if (!waitImmediately) {
    os << "    wafer.instr.dte_wait ";
    for (size_t index = 0; index < sites.size(); ++index) {
      if (index != 0)
        os << ", ";
      os << "%token" << index;
    }
    os << " : ";
    for (size_t index = 0; index < sites.size(); ++index) {
      if (index != 0)
        os << ", ";
      os << "!async.token";
    }
    os << "\n";
  }
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

static std::string makeTwoHelperRank(bool isSend, int64_t peer,
                                     int64_t baseOffset,
                                     bool reverseDefinitions, bool reverseCalls,
                                     bool reuseMessageIdentity = false) {
  auto emitHelper = [&](llvm::raw_ostream &os, llvm::StringRef name,
                        int64_t offset, int64_t communication) {
    os << "  func.func private @" << name
       << "() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<"
       << offset
       << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
          "    %token = wafer.instr.dte_"
       << (isSend ? "send" : "recv") << " %buffer {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << communication
       << ", phase = collective_permute, round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %token : !async.token\n"
          "    return\n"
          "  }\n";
  };

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n";
  os << "    func.call @" << (reverseCalls ? "second" : "first")
     << "() : () -> ()\n";
  os << "    func.call @" << (reverseCalls ? "first" : "second")
     << "() : () -> ()\n";
  os << "    return\n"
        "  }\n";
  if (reverseDefinitions) {
    emitHelper(os, "second", baseOffset + 256, reuseMessageIdentity ? 50 : 51);
    emitHelper(os, "first", baseOffset, 50);
  } else {
    emitHelper(os, "first", baseOffset, 50);
    emitHelper(os, "second", baseOffset + 256, reuseMessageIdentity ? 50 : 51);
  }
  os << "}\n";
  return source;
}

static std::string makeStructuredPhaseRank(bool prologueIsSend,
                                           bool steadyIsSend,
                                           bool epilogueIsSend, int64_t peer,
                                           int64_t baseOffset) {
  auto emitIssue = [&](llvm::raw_ostream &os, llvm::StringRef indent,
                       llvm::StringRef token, llvm::StringRef buffer,
                       bool isSend, int64_t communication) {
    os << indent << token << " = wafer.instr.dte_" << (isSend ? "send" : "recv")
       << " " << buffer << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << communication
       << ", phase = collective_permute, round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n";
    os << indent << "wafer.instr.dte_wait " << token << " : !async.token\n";
  };

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        "    %c2 = arith.constant 2 : index\n"
        "    %c4 = arith.constant 4 : index\n";
  for (int64_t index = 0; index < 3; ++index)
    os << "    %buffer" << index
       << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
       << baseOffset + index * 256
       << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n";
  emitIssue(os, "    ", "%prologue", "%buffer0", prologueIsSend, 40);
  os << "    scf.for %outer = %c0 to %c4 step %c1 {\n"
        "      scf.for %inner = %c0 to %c2 step %c1 {\n";
  emitIssue(os, "        ", "%steady", "%buffer1", steadyIsSend, 41);
  os << "      }\n"
        "    }\n";
  emitIssue(os, "    ", "%epilogue", "%buffer2", epilogueIsSend, 42);
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

static std::string makeSiblingLoopRank(bool firstIsSend, bool secondIsSend,
                                       int64_t peer, int64_t baseOffset) {
  auto emitLoop = [&](llvm::raw_ostream &os, llvm::StringRef induction,
                      llvm::StringRef token, llvm::StringRef buffer,
                      bool isSend, int64_t communication) {
    os << "    scf.for " << induction << " = %c0 to %c4 step %c1 {\n"
       << "      " << token << " = wafer.instr.dte_"
       << (isSend ? "send" : "recv") << " " << buffer << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << communication
       << ", phase = collective_permute, round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
       << "      wafer.instr.dte_wait " << token << " : !async.token\n"
       << "    }\n";
  };

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        "    %c4 = arith.constant 4 : index\n"
        "    %buffer0 = memref.alloc() {wafer.spm.offset = "
        "#wafer.spm_offset<"
     << baseOffset
     << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
        "    %buffer1 = memref.alloc() {wafer.spm.offset = "
        "#wafer.spm_offset<"
     << baseOffset + 256 << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n";
  emitLoop(os, "%first_index", "%first", "%buffer0", firstIsSend, 80);
  emitLoop(os, "%second_index", "%second", "%buffer1", secondIsSend, 81);
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

static std::string makeTileRegionSiblingRank(unsigned rank) {
  auto emitRegion = [&](llvm::raw_ostream &os, unsigned region,
                        llvm::StringRef input, bool hasTransport, bool isSend,
                        int64_t peer, int64_t offset, int64_t communication) {
    os << "    %tile" << region << " = wafer.tile.region(\n"
       << "        " << input
       << " : memref<4xf32, #wafer.memory<ddr, tensor>>)\n"
          "        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {\n"
          "    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>):\n";
    if (hasTransport) {
      os << "      %buffer = memref.alloc() {wafer.spm.offset = "
            "#wafer.spm_offset<"
         << offset
         << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
            "      %token = wafer.instr.dte_"
         << (isSend ? "send" : "recv") << " %buffer {peer = " << peer
         << " : i64, bytes = 16 : i64, message = "
            "#wafer.dte_message<communication = "
         << communication
         << ", phase = collective_permute, round = 0, slice = 0>} : "
            "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
            "      wafer.instr.dte_wait %token : !async.token\n";
    }
    os << "      wafer.tile.yield %source "
          ": memref<4xf32, #wafer.memory<ddr, tensor>>\n"
          "    }\n";
  };

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main("
        "%input: memref<4xf32, #wafer.memory<ddr, tensor>>) {\n";
  if (rank == 0)
    emitRegion(os, 0, "%input", /*hasTransport=*/true, /*isSend=*/true,
               /*peer=*/1, /*offset=*/65536, /*communication=*/90);
  else if (rank == 1)
    emitRegion(os, 0, "%input", /*hasTransport=*/true, /*isSend=*/false,
               /*peer=*/0, /*offset=*/66048, /*communication=*/90);
  else
    emitRegion(os, 0, "%input", /*hasTransport=*/false, /*isSend=*/false,
               /*peer=*/-1, /*offset=*/-1, /*communication=*/-1);

  if (rank == 0)
    emitRegion(os, 1, "%tile0", /*hasTransport=*/true, /*isSend=*/true,
               /*peer=*/2, /*offset=*/65792, /*communication=*/91);
  else if (rank == 2)
    emitRegion(os, 1, "%tile0", /*hasTransport=*/true, /*isSend=*/false,
               /*peer=*/0, /*offset=*/66304, /*communication=*/91);
  else
    emitRegion(os, 1, "%tile0", /*hasTransport=*/false, /*isSend=*/false,
               /*peer=*/-1, /*offset=*/-1, /*communication=*/-1);
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

static std::string makeCrossBlockCycleRank(bool firstRank, int64_t baseOffset) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        "    %c4 = arith.constant 4 : index\n"
        "    %buffer0 = memref.alloc() {wafer.spm.offset = "
        "#wafer.spm_offset<"
     << baseOffset
     << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
        "    %buffer1 = memref.alloc() {wafer.spm.offset = "
        "#wafer.spm_offset<"
     << baseOffset + 256 << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n";
  if (firstRank) {
    os << "    %first = wafer.instr.dte_send %buffer0 {peer = 1 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 60, phase = collective_permute, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %first : !async.token\n"
          "    scf.for %index = %c0 to %c4 step %c1 {\n"
          "      %second = wafer.instr.dte_recv %buffer1 {peer = 1 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 61, phase = collective_permute, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "      wafer.instr.dte_wait %second : !async.token\n"
          "    }\n";
  } else {
    os << "    scf.for %index = %c0 to %c4 step %c1 {\n"
          "      %first = wafer.instr.dte_send %buffer0 {peer = 0 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 61, phase = collective_permute, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "      wafer.instr.dte_wait %first : !async.token\n"
          "    }\n"
          "    %second = wafer.instr.dte_recv %buffer1 {peer = 0 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 60, phase = collective_permute, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %second : !async.token\n";
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

constexpr llvm::StringLiteral kInterveningBufferReadSendRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <neg> %buffer into %dest
        : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kInterveningBufferReadRecvRank = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <neg> %buffer into %dest
        : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<4xf32, #wafer.memory<spm, tensor>>
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
  EXPECT_EQ(send.getBinding()->getRemoteAddressMode(),
            wafer::DTERemoteAddressMode::Absolute);
  EXPECT_EQ(send.getBinding()->getRemoteReceiverAddress(), 65792);
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

TEST_F(DirectDTETransportTest, ReceivePreparationBreaksCrossRankSendWaitCycle) {
  std::string rank0 =
      makeLinearTransportRank({{false, 1, 65536, 31}, {true, 1, 65792, 30}},
                              /*waitImmediately=*/false);
  std::string rank1 =
      makeLinearTransportRank({{false, 0, 66048, 30}, {true, 0, 66304, 31}},
                              /*waitImmediately=*/false);
  auto rank0Module = parse(rank0);
  auto rank1Module = parse(rank1);
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*rank0Module, *rank1Module};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, MutualSendBeforeReceiveWaitCycleFailsClosed) {
  std::string rank0 =
      makeLinearTransportRank({{true, 1, 65536, 30}, {false, 1, 65792, 31}},
                              /*waitImmediately=*/true);
  std::string rank1 =
      makeLinearTransportRank({{true, 0, 66048, 31}, {false, 0, 66304, 30}},
                              /*waitImmediately=*/true);
  auto rank0Module = parse(rank0);
  auto rank1Module = parse(rank1);
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*rank0Module, *rank1Module};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("wait graph contains a cyclic dependency"),
            std::string::npos);
  for (mlir::ModuleOp module : modules) {
    module.walk([](wafer::InstrDTESendOp operation) {
      EXPECT_FALSE(operation.getBinding());
    });
    module.walk([](wafer::InstrDTERecvOp operation) {
      EXPECT_FALSE(operation.getBinding());
    });
  }
}

TEST_F(DirectDTETransportTest,
       AcceptsNestedSteadyStateWithPrologueAndEpilogueBlocks) {
  auto rank0Module =
      parse(makeStructuredPhaseRank(/*prologueIsSend=*/false,
                                    /*steadyIsSend=*/false,
                                    /*epilogueIsSend=*/true, /*peer=*/1,
                                    /*baseOffset=*/65536));
  auto rank1Module =
      parse(makeStructuredPhaseRank(/*prologueIsSend=*/true,
                                    /*steadyIsSend=*/true,
                                    /*epilogueIsSend=*/false, /*peer=*/0,
                                    /*baseOffset=*/66560));
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*rank0Module, *rank1Module};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, AcceptsOrderedSiblingLoopOccurrences) {
  auto rank0Module =
      parse(makeSiblingLoopRank(/*firstIsSend=*/false,
                                /*secondIsSend=*/true, /*peer=*/1,
                                /*baseOffset=*/65536));
  auto rank1Module =
      parse(makeSiblingLoopRank(/*firstIsSend=*/true,
                                /*secondIsSend=*/false, /*peer=*/0,
                                /*baseOffset=*/66560));
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*rank0Module, *rank1Module};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest,
       CountsTileRegionSiblingsIndependentOfTransportContent) {
  auto rank0Module = parse(makeTileRegionSiblingRank(/*rank=*/0));
  auto rank1Module = parse(makeTileRegionSiblingRank(/*rank=*/1));
  auto rank2Module = parse(makeTileRegionSiblingRank(/*rank=*/2));
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  ASSERT_TRUE(rank2Module);
  llvm::SmallVector<mlir::ModuleOp, 3> modules{*rank0Module, *rank1Module,
                                               *rank2Module};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, CrossBlockWaitCycleFailsClosed) {
  auto rank0Module =
      parse(makeCrossBlockCycleRank(/*firstRank=*/true, /*baseOffset=*/65536));
  auto rank1Module =
      parse(makeCrossBlockCycleRank(/*firstRank=*/false, /*baseOffset=*/66560));
  ASSERT_TRUE(rank0Module);
  ASSERT_TRUE(rank1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*rank0Module, *rank1Module};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("wait graph contains a cyclic dependency"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest,
       HelperDefinitionOrderDoesNotDefineMessageOccurrence) {
  auto sendModule = parse(makeTwoHelperRank(/*isSend=*/true, /*peer=*/1,
                                            /*baseOffset=*/65536,
                                            /*reverseDefinitions=*/false,
                                            /*reverseCalls=*/false,
                                            /*reuseMessageIdentity=*/true));
  auto recvModule = parse(makeTwoHelperRank(/*isSend=*/false, /*peer=*/0,
                                            /*baseOffset=*/66560,
                                            /*reverseDefinitions=*/true,
                                            /*reverseCalls=*/false,
                                            /*reuseMessageIdentity=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, MismatchedHelperCallOccurrenceFailsClosed) {
  auto sendModule = parse(makeTwoHelperRank(/*isSend=*/true, /*peer=*/1,
                                            /*baseOffset=*/65536,
                                            /*reverseDefinitions=*/false,
                                            /*reverseCalls=*/false));
  auto recvModule = parse(makeTwoHelperRank(/*isSend=*/false, /*peer=*/0,
                                            /*baseOffset=*/66560,
                                            /*reverseDefinitions=*/true,
                                            /*reverseCalls=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("occurrence paths are not structurally "
                                "identical across ranks"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest,
       StaticSiteReusedAcrossDifferentCallBindingsFailsClosed) {
  constexpr llvm::StringLiteral kRepeatedSendSite = R"mlir(
module {
  func.func @main() {
    func.call @transport() : () -> ()
    func.call @transport() : () -> ()
    return
  }
  func.func private @transport() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 52, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  constexpr llvm::StringLiteral kDistinctReceiveSites = R"mlir(
module {
  func.func @main() {
    func.call @first() : () -> ()
    func.call @second() : () -> ()
    return
  }
  func.func private @first() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 52, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
  func.func private @second() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 52, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  auto sendModule = parse(kRepeatedSendSite);
  auto recvModule = parse(kDistinctReceiveSites);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("different physical bindings across call "
                                "occurrences"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest, UnusedDTEHelperFailsCallOccurrenceProof) {
  constexpr llvm::StringLiteral kUnusedSend = R"mlir(
module {
  func.func @main() {
    return
  }
  func.func private @unused_transport() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 70, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  constexpr llvm::StringLiteral kUsedRecv = R"mlir(
module {
  func.func @main() {
    func.call @transport() : () -> ()
    return
  }
  func.func private @transport() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 70, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  auto sendModule = parse(kUnusedSend);
  auto recvModule = parse(kUsedRecv);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("outside the entry call closure"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest,
       MatchesEquivalentDynamicStreamAcrossDifferentStaticPlacement) {
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
  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_TRUE(operation.getBinding());
  });
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_TRUE(operation.getBinding());
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

TEST_F(DirectDTETransportTest,
       IgnoresUnrelatedStaticControlWhenDynamicMessageStreamMatches) {
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
  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
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

TEST_F(DirectDTETransportTest, InterveningSendSourceReadIsAccepted) {
  auto sendModule = parse(kInterveningBufferReadSendRank);
  auto recvModule = parse(kRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::compiler::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, InterveningReceiveDestinationReadFailsClosed) {
  auto sendModule = parse(kSendRank);
  auto recvModule = parse(kInterveningBufferReadRecvRank);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> modules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::acceptDirectDTETransport(modules);
  EXPECT_TRUE(mlir::failed(contract));
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
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

TEST_F(DirectDTETransportTest,
       CollectiveReductionPhysicalLayoutMismatchFailsClosed) {
  std::string sendSource = kSendRank.str();
  std::string recvSource = kRecvRank.str();
  auto replaceAll = [](std::string &text, llvm::StringRef from,
                       llvm::StringRef to) {
    size_t position = 0;
    while ((position = text.find(from.str(), position)) != std::string::npos) {
      text.replace(position, from.size(), to.str());
      position += to.size();
    }
  };
  replaceAll(sendSource, "collective_permute", "all_reduce_tree_reduce");
  replaceAll(recvSource, "collective_permute", "all_reduce_tree_reduce");
  replaceAll(recvSource, "#wafer.memory<spm, tensor>",
             "#wafer.memory<spm, cx>");
  auto sendModule = parse(sendSource);
  auto recvModule = parse(recvSource);
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
