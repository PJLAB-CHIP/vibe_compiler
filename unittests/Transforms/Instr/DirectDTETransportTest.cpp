//===- DirectDTETransportTest.cpp - Tile DTE tests -------------===//

#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "TestSupport/Driver/CompilerTesting.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/TopologyIds.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/NativeDirectDTEMultiSend.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"

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

#include <algorithm>
#include <memory>
#include <string>

namespace {

class DirectDTETransportTest : public ::testing::Test {
protected:
  DirectDTETransportTest() {
    registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    wafer::registerWaferCoreDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    std::string tileSource = source.str();
    constexpr llvm::StringLiteral kModuleHeader = "module {";
    size_t module = tileSource.find(kModuleHeader.str());
    if (module == std::string::npos)
      return {};
    tileSource.insert(module + kModuleHeader.size(),
                      R"mlir(
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir");
    return mlir::parseSourceString<mlir::ModuleOp>(
        tileSource, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

constexpr llvm::StringLiteral kSendTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kRecvTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

static std::string makeControlledTileModule(bool isSend, int64_t peer,
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
        "#wafer.dte_message<communication = 9, "
        "round = 2, slice = 0>} : "
        "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
        "    wafer.instr.dte_wait %token0 : !async.token\n"
     << controlSuffix;
  if (addStaticTail) {
    os << "    %token1 = wafer.instr.dte_" << (isSend ? "send" : "recv")
       << " %buffer"
       << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 9, "
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
makeLinearTransportTileModule(llvm::ArrayRef<LinearTransportSite> sites,
                              bool waitImmediately, int64_t extent = 0) {
  const std::string tensorType =
      extent ? "1x1x" + std::to_string(extent) + "xf16" : "4xf32";
  const int64_t bytes = extent ? extent * 2 : 16;
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main() {\n";
  for (auto [index, site] : llvm::enumerate(sites))
    os << "    %buffer" << index
       << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
       << site.spmOffset << ">} : memref<" << tensorType
       << ", #wafer.memory<spm, tensor>>\n";
  for (auto [index, site] : llvm::enumerate(sites)) {
    os << "    %token" << index << " = wafer.instr.dte_"
       << (site.isSend ? "send" : "recv") << " %buffer" << index
       << " {peer = " << site.peer
       << " : i64, bytes = " << bytes << " : i64, message = "
          "#wafer.dte_message<communication = "
       << site.communication
       << ", round = 0, slice = 0>} : memref<" << tensorType
       << ", #wafer.memory<spm, tensor>> -> !async.token\n";
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

static std::string makeTwoHelperTileModule(bool isSend, int64_t peer,
                                           int64_t baseOffset,
                                           bool reverseDefinitions,
                                           bool reverseCalls,
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
       << ", round = 0, slice = 0>} : "
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

static std::string makeStructuredPhaseTileModule(bool prologueIsSend,
                                                 bool steadyIsSend,
                                                 bool epilogueIsSend,
                                                 int64_t peer,
                                                 int64_t baseOffset) {
  auto emitIssue = [&](llvm::raw_ostream &os, llvm::StringRef indent,
                       llvm::StringRef token, llvm::StringRef buffer,
                       bool isSend, int64_t communication) {
    os << indent << token << " = wafer.instr.dte_" << (isSend ? "send" : "recv")
       << " " << buffer << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << communication
       << ", round = 0, slice = 0>} : "
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

static std::string makeSiblingLoopTileModule(bool firstIsSend,
                                             bool secondIsSend, int64_t peer,
                                             int64_t baseOffset) {
  auto emitLoop = [&](llvm::raw_ostream &os, llvm::StringRef induction,
                      llvm::StringRef token, llvm::StringRef buffer,
                      bool isSend, int64_t communication) {
    os << "    scf.for " << induction << " = %c0 to %c4 step %c1 {\n"
       << "      " << token << " = wafer.instr.dte_"
       << (isSend ? "send" : "recv") << " " << buffer << " {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = "
       << communication
       << ", round = 0, slice = 0>} : "
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

static std::string makeTileRegionSiblingTileModule(wafer::TileId tileId) {
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
         << ", round = 0, slice = 0>} : "
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
  if (tileId == wafer::TileId(0))
    emitRegion(os, 0, "%input", /*hasTransport=*/true, /*isSend=*/true,
               /*peer=*/1, /*offset=*/65536, /*communication=*/90);
  else if (tileId == wafer::TileId(1))
    emitRegion(os, 0, "%input", /*hasTransport=*/true, /*isSend=*/false,
               /*peer=*/0, /*offset=*/66048, /*communication=*/90);
  else
    emitRegion(os, 0, "%input", /*hasTransport=*/false, /*isSend=*/false,
               /*peer=*/-1, /*offset=*/-1, /*communication=*/-1);

  if (tileId == wafer::TileId(0))
    emitRegion(os, 1, "%tile0", /*hasTransport=*/true, /*isSend=*/true,
               /*peer=*/2, /*offset=*/65792, /*communication=*/91);
  else if (tileId == wafer::TileId(2))
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

static std::string makeShiftedTileRegionTransportProgram(bool isSend,
                                                         unsigned region) {
  auto emitRegion = [&](llvm::raw_ostream &os, unsigned index,
                        llvm::StringRef input) {
    os << "    %tile" << index << " = wafer.tile.region(\n"
       << "        " << input
       << " : memref<4xf32, #wafer.memory<ddr, tensor>>)\n"
          "        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {\n"
          "    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>):\n";
    if (index == region)
      os << "      %buffer = memref.alloc() {wafer.spm.offset = "
            "#wafer.spm_offset<"
         << (isSend ? 65536 : 65792)
         << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
            "      %token = wafer.instr.dte_"
         << (isSend ? "send" : "recv")
         << " %buffer {peer = " << (isSend ? 1 : 0)
         << " : i64, bytes = 16 : i64, message = "
            "#wafer.dte_message<communication = 92, round = 0, slice = 0>} "
            ": memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
            "      wafer.instr.dte_wait %token : !async.token\n";
    os << "      wafer.tile.yield %source "
          ": memref<4xf32, #wafer.memory<ddr, tensor>>\n"
          "    }\n";
  };

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  func.func @main("
        "%input: memref<4xf32, #wafer.memory<ddr, tensor>>) {\n";
  emitRegion(os, 0, "%input");
  emitRegion(os, 1, "%tile0");
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

static std::string makeCrossBlockCycleTileModule(bool firstTile,
                                                 int64_t baseOffset) {
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
  if (firstTile) {
    os << "    %first = wafer.instr.dte_send %buffer0 {peer = 1 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 60, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %first : !async.token\n"
          "    scf.for %index = %c0 to %c4 step %c1 {\n"
          "      %second = wafer.instr.dte_recv %buffer1 {peer = 1 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 61, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "      wafer.instr.dte_wait %second : !async.token\n"
          "    }\n";
  } else {
    os << "    scf.for %index = %c0 to %c4 step %c1 {\n"
          "      %first = wafer.instr.dte_send %buffer0 {peer = 0 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 61, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "      wafer.instr.dte_wait %first : !async.token\n"
          "    }\n"
          "    %second = wafer.instr.dte_recv %buffer1 {peer = 0 : i64, "
          "bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 60, "
          "round = 0, slice = 0>} : "
          "memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token\n"
          "    wafer.instr.dte_wait %second : !async.token\n";
  }
  os << "    return\n"
        "  }\n"
        "}\n";
  return source;
}

constexpr llvm::StringLiteral kOverlappingSendTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %first = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 10, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %second = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 11, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %first, %second : !async.token, !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kFiveLiveReceiversTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %r0 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 20, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r1 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 21, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r2 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 22, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r3 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 23, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %r4 = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 24, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %r0, %r1, %r2, %r3, %r4 : !async.token, !async.token, !async.token, !async.token, !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kLoopEscapingTokenSendTileModule = R"mlir(
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
           message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
          : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
      scf.yield %token : !async.token
    }
    wafer.instr.dte_wait %escaped : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kInterveningBufferAccessSendTileModule = R"mlir(
module {
  func.func @main() {
    %index = arith.constant 0 : index
    %value = arith.constant 0.0 : f32
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    memref.store %value, %buffer[%index]
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kInterveningBufferReadSendTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <neg> %buffer into %dest
        : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

constexpr llvm::StringLiteral kInterveningBufferReadRecvTileModule = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <neg> %buffer into %dest
        : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";

TEST_F(DirectDTETransportTest,
       CompletionRebuildSerializesActualSenderSlotReuse) {
  auto module = parse(kOverlappingSendTileModule);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  auto result = wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.sends, 2u);
  EXPECT_EQ(result.statistics.waitsErased, 1u);
  EXPECT_EQ(result.statistics.waitsPlaced, 2u);
  EXPECT_EQ(result.statistics.senderSlotReuseWaits, 1u);

  llvm::SmallVector<mlir::Operation *, 4> transport;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTEWaitOp>(operation))
      transport.push_back(operation);
  });
  ASSERT_EQ(transport.size(), 4u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(transport[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(transport[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(transport[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(transport[3]));
}

TEST_F(DirectDTETransportTest,
       CompletionRebuildUsesReceiverFSMCapacityOnDistinctBuffers) {
  llvm::SmallVector<LinearTransportSite, 5> sites;
  for (int64_t index = 0; index < 5; ++index)
    sites.push_back(LinearTransportSite{/*isSend=*/false, /*peer=*/index,
                                        /*spmOffset=*/65536 + index * 4096,
                                        /*communication=*/20 + index});
  auto module =
      parse(makeLinearTransportTileModule(sites, /*waitImmediately=*/false,
                                          /*extent=*/1025));
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  auto result = wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.receives, 5u);
  EXPECT_EQ(result.statistics.waitsPlaced, 5u);
  EXPECT_EQ(result.statistics.receiverFSMReuseWaits, 1u);
  EXPECT_EQ(result.statistics.receiverReadySlotReuseWaits, 0u);
  llvm::SmallVector<wafer::InstrDTERecvOp, 5> receives;
  module->walk([&](wafer::InstrDTERecvOp recv) { receives.push_back(recv); });
  for (unsigned index = 1; index < 4; ++index)
    EXPECT_EQ(receives[index]->getPrevNode(), receives[index - 1].getOperation());
}

TEST_F(DirectDTETransportTest,
       CompletionRebuildWaitsBeforeSamePeerReadySlotReuse) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeLinearTransportTileModule(
        {{false, 0, 65536, 20}, {false, 0, 69632, 21}},
        /*waitImmediately=*/false, extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
    auto result = wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.waitsPlaced, 2u);
    EXPECT_EQ(result.statistics.receiverReadySlotReuseWaits, 1u);
    llvm::SmallVector<wafer::InstrDTERecvOp, 2> receives;
    module->walk([&](wafer::InstrDTERecvOp recv) { receives.push_back(recv); });
    ASSERT_EQ(receives.size(), 2u);
    auto wait = mlir::dyn_cast_or_null<wafer::InstrDTEWaitOp>(
        receives[1]->getPrevNode());
    ASSERT_TRUE(wait);
    EXPECT_EQ(wait.getTokens().front(), receives[0].getToken());
  }
}

TEST_F(DirectDTETransportTest, OverlappingSamePeerReadyPostsFailClosed) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto sender = parse(makeLinearTransportTileModule(
        {{true, 1, 65536, 20}, {true, 1, 69632, 21}},
        /*waitImmediately=*/true, extent));
    auto receiver = parse(makeLinearTransportTileModule(
        {{false, 0, 65536, 20}, {false, 0, 69632, 21}},
        /*waitImmediately=*/false, extent));
    ASSERT_TRUE(sender);
    ASSERT_TRUE(receiver);
    llvm::SmallVector<mlir::ModuleOp, 2> modules{*sender, *receiver};
    mlir::ScopedDiagnosticHandler suppress(
        context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
    EXPECT_TRUE(mlir::failed(
        wafer::compiler::testing::bindDirectDTETransport(modules)));
    receiver->walk([](wafer::InstrDTERecvOp recv) {
      EXPECT_FALSE(recv.getBinding());
    });
    auto rebuilt = wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
    ASSERT_TRUE(rebuilt.succeeded()) << rebuilt.detail;
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::testing::bindDirectDTETransport(modules)));
  }
}

TEST_F(DirectDTETransportTest,
       MutualSendIssueBeforeReceiveFailsWithDelayedWaits) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto tile0 = parse(makeLinearTransportTileModule(
        {{true, 1, 65536, 30}, {false, 1, 69632, 31}},
        /*waitImmediately=*/false, extent));
    auto tile1 = parse(makeLinearTransportTileModule(
        {{true, 0, 65536, 31}, {false, 0, 69632, 30}},
        /*waitImmediately=*/false, extent));
    ASSERT_TRUE(tile0);
    ASSERT_TRUE(tile1);
    llvm::SmallVector<mlir::ModuleOp, 2> modules{*tile0, *tile1};
    std::string diagnosticText;
    mlir::ScopedDiagnosticHandler handler(
        context.get(), [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream stream(diagnosticText);
          diagnostic.print(stream);
          return mlir::success();
        });
    EXPECT_TRUE(mlir::failed(
        wafer::compiler::testing::bindDirectDTETransport(modules)));
    EXPECT_NE(diagnosticText.find("wait graph contains a cyclic dependency"),
              std::string::npos);
  }
}

TEST_F(DirectDTETransportTest,
       CompletionRebuildWaitsForReceiveBeforeRelayAndKeepsSendThroughReads) {
  constexpr llvm::StringLiteral kRelay = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %recv = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %send = wafer.instr.dte_send %buffer
        {peer = 2 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 1, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <neg> %buffer into %dest
        : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %recv, %send : !async.token, !async.token
    return
  }
})mlir";
  auto module = parse(kRelay);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  auto result = wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(result.succeeded()) << result.detail;

  llvm::SmallVector<mlir::Operation *, 8> operations;
  mlir::func::FuncOp function;
  module->walk([&](mlir::func::FuncOp current) { function = current; });
  ASSERT_TRUE(function);
  for (mlir::Operation &operation : function.getBody().front())
    if (mlir::isa<wafer::InstrDTERecvOp, wafer::InstrDTESendOp,
                  wafer::InstrDTEWaitOp, wafer::InstrElementwiseOp>(operation))
      operations.push_back(&operation);
  ASSERT_EQ(operations.size(), 5u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTERecvOp>(operations[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(operations[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(operations[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(operations[3]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(operations[4]));
}

TEST_F(DirectDTETransportTest, MatchesCompleteDomainAndAttachesTypedBinding) {
  auto sendModule = parse(kSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);

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
       NativeBroadcastReplacesSourceSendsAndKeepsExactReceivers) {
  auto makeSource = [](int64_t extent, int64_t destinationCount) {
    std::string source;
    llvm::raw_string_ostream os(source);
    int64_t row = extent - 1;
    int64_t sourceOffset = row * 256;
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : memref<1x"
       << extent << "x128xf16, #wafer.memory<spm, tensor>>\n";
    for (int64_t index = 0; index < destinationCount; ++index) {
      os << "    %token" << index
         << " = wafer.instr.dte_send %buffer {buffer_offset = " << sourceOffset
         << " : i64, peer = " << index + 1
         << " : i64, bytes = 256 : i64, message = "
            "#wafer.dte_message<communication = 70, round = 0, slice = "
         << index << ">} : memref<1x" << extent
         << "x128xf16, #wafer.memory<spm, tensor>> -> !async.token\n";
    }
    os << "    return\n  }\n}\n";
    return source;
  };
  auto makeReceiver = [](int64_t offset, int64_t slice) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
       << "#wafer.spm_offset<" << offset
       << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>>\n"
          "    %token = wafer.instr.dte_recv %buffer {peer = 0 : i64, "
          "bytes = 256 : i64, message = "
          "#wafer.dte_message<communication = 70, round = 0, slice = "
       << slice
       << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> "
          "!async.token\n"
          "    return\n  }\n}\n";
    return source;
  };

  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t destinationCount : {2, 4, 8, 15}) {
      SCOPED_TRACE((llvm::Twine("extent=") + llvm::Twine(extent) +
                    ", destinations=" + llvm::Twine(destinationCount))
                       .str());
      auto source = parse(makeSource(extent, destinationCount));
      ASSERT_TRUE(source);
      llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 15> receiverOwners;
      llvm::SmallVector<mlir::ModuleOp, 16> modules{*source};
      for (int64_t index = 0; index < destinationCount; ++index) {
        receiverOwners.push_back(
            parse(makeReceiver(393216 + index * 256, index)));
        ASSERT_TRUE(receiverOwners.back());
        modules.push_back(*receiverOwners.back());
      }

      auto grouped =
          wafer::compiler::detail::materializeNativeDirectDTEMultiSends(
              modules);
      ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
      EXPECT_EQ(grouped.statistics.broadcastOperations, 1u);
      EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved,
                static_cast<unsigned>(destinationCount));
      unsigned unicastSends = 0;
      source->walk([&](wafer::InstrDTESendOp) { ++unicastSends; });
      EXPECT_EQ(unicastSends, 0u);
      wafer::InstrDTEBroadcastOp broadcast;
      source->walk(
          [&](wafer::InstrDTEBroadcastOp operation) { broadcast = operation; });
      ASSERT_TRUE(broadcast);
      EXPECT_EQ(broadcast.getSourceOffsetAttr().getInt(), (extent - 1) * 256);
      ASSERT_EQ(broadcast.getPeersAttr().size(), destinationCount);
      for (int64_t index = 0; index < destinationCount; ++index)
        EXPECT_EQ(broadcast.getPeersAttr().asArrayRef()[index], index + 1);

      auto completion =
          wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
      ASSERT_TRUE(mlir::succeeded(contract));
      EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
      auto bindings = broadcast.getBindings();
      ASSERT_TRUE(bindings);
      EXPECT_EQ(bindings->size(), static_cast<size_t>(destinationCount));
      for (mlir::Attribute attribute : *bindings)
        EXPECT_TRUE(mlir::isa<wafer::DirectDTEBindingAttr>(attribute));
    }
  }
}

TEST_F(DirectDTETransportTest,
       NativeScatterUsesOneContiguousSourceAndOrderedDestinations) {
  auto makeSource = [](int64_t extent, int64_t destinationCount) {
    std::string text;
    llvm::raw_string_ostream os(text);
    int64_t stride = extent * 128;
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : memref<1x"
       << extent << "x128xf16, #wafer.memory<spm, tensor>>\n";
    for (int64_t index = 0; index < destinationCount; ++index) {
      os << "    %slice" << index << " = memref.subview %buffer[0, " << index
         << ", 0] [1, 1, 128] [1, 1, 1] : memref<1x" << extent
         << "x128xf16, #wafer.memory<spm, tensor>> to "
            "memref<1x1x128xf16, strided<["
         << stride << ", 128, 1], offset: " << index * 128
         << ">, #wafer.memory<spm, tensor>>\n"
         << "    %token" << index << " = wafer.instr.dte_send %slice" << index
         << " {peer = " << index + 1
         << " : i64, bytes = 256 : i64, message = "
            "#wafer.dte_message<communication = 71, round = 0, slice = "
         << index << ">} : memref<1x1x128xf16, strided<[" << stride
         << ", 128, 1], offset: " << index * 128
         << ">, #wafer.memory<spm, tensor>> -> !async.token\n";
    }
    os << "    return\n  }\n}\n";
    return text;
  };

  auto makeReceiver = [](int64_t offset, int64_t slice) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
       << "#wafer.spm_offset<" << offset
       << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>>\n"
          "    %token = wafer.instr.dte_recv %buffer {peer = 0 : i64, "
          "bytes = 256 : i64, message = "
          "#wafer.dte_message<communication = 71, round = 0, slice = "
       << slice
       << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> "
          "!async.token\n"
          "    return\n  }\n}\n";
    return text;
  };

  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t destinationCount : {2, 4, 8, 15}) {
      SCOPED_TRACE((llvm::Twine("extent=") + llvm::Twine(extent) +
                    ", destinations=" + llvm::Twine(destinationCount))
                       .str());
      auto source = parse(makeSource(extent, destinationCount));
      ASSERT_TRUE(source);
      llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 15> receiverOwners;
      llvm::SmallVector<mlir::ModuleOp, 16> modules{*source};
      for (int64_t index = 0; index < destinationCount; ++index) {
        receiverOwners.push_back(
            parse(makeReceiver(393216 + index * 256, index)));
        ASSERT_TRUE(receiverOwners.back());
        modules.push_back(*receiverOwners.back());
      }

      auto grouped =
          wafer::compiler::detail::materializeNativeDirectDTEMultiSends(
              modules);
      ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
      EXPECT_EQ(grouped.statistics.scatterOperations, 1u);
      EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved,
                static_cast<unsigned>(destinationCount));
      wafer::InstrDTEScatterOp scatter;
      source->walk(
          [&](wafer::InstrDTEScatterOp operation) { scatter = operation; });
      ASSERT_TRUE(scatter);
      EXPECT_EQ(scatter.getSourceOffsetAttr().getInt(), 0);
      ASSERT_EQ(scatter.getPeersAttr().size(), destinationCount);

      auto completion =
          wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
      ASSERT_TRUE(mlir::succeeded(contract));
      EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
      ASSERT_TRUE(scatter.getBindings());
      EXPECT_EQ(scatter.getBindings()->size(),
                static_cast<size_t>(destinationCount));
    }
  }
}

TEST_F(DirectDTETransportTest,
       UnsupportedNativeParametersRemainOrdinaryUnicast) {
  auto makeSource = [](int64_t destinationCount, int64_t bytes) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : "
          "memref<1x1025x128xf16, #wafer.memory<spm, tensor>>\n";
    for (int64_t index = 0; index < destinationCount; ++index) {
      os << "    %token" << index
         << " = wafer.instr.dte_send %buffer {peer = " << index + 1
         << " : i64, bytes = " << bytes
         << " : i64, message = "
            "#wafer.dte_message<communication = 72, round = 0, slice = "
         << index
         << ">} : memref<1x1025x128xf16, "
            "#wafer.memory<spm, tensor>> -> !async.token\n";
    }
    os << "    return\n  }\n}\n";
    return text;
  };

  struct Case {
    int64_t destinations;
    int64_t bytes;
  };
  for (Case testCase : {Case{1, 256}, Case{3, 256}, Case{5, 256}, Case{16, 256},
                        Case{2, 255}, Case{2, 257}}) {
    SCOPED_TRACE((llvm::Twine("destinations=") +
                  llvm::Twine(testCase.destinations) +
                  ", bytes=" + llvm::Twine(testCase.bytes))
                     .str());
    auto source = parse(makeSource(testCase.destinations, testCase.bytes));
    ASSERT_TRUE(source);
    llvm::SmallVector<mlir::ModuleOp, 1> modules{*source};
    auto grouped =
        wafer::compiler::detail::materializeNativeDirectDTEMultiSends(modules);
    ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
    EXPECT_EQ(grouped.statistics.broadcastOperations, 0u);
    EXPECT_EQ(grouped.statistics.scatterOperations, 0u);
    unsigned unicastSends = 0;
    source->walk([&](wafer::InstrDTESendOp) { ++unicastSends; });
    EXPECT_EQ(unicastSends, static_cast<unsigned>(testCase.destinations));
  }
}

TEST_F(DirectDTETransportTest,
       ExactP2PCoalescingRequiresContiguousSourceAndDestinationRanges) {
  auto makeFragments = [](bool send, int64_t peer, int64_t allocationOffset,
                          int64_t extent, int64_t rowBase, int64_t rowStep = 1,
                          bool separateCommunications = false) {
    std::string text;
    llvm::raw_string_ostream os(text);
    int64_t stride = extent * 128;
    os << "module {\n  func.func @main() {\n"
          "    %buffer = memref.alloc() {wafer.spm.offset = "
       << "#wafer.spm_offset<" << allocationOffset << ">} : memref<1x" << extent
       << "x128xf16, #wafer.memory<spm, tensor>>\n";
    for (int64_t index = 0; index < 2; ++index) {
      int64_t row = rowBase + index * rowStep;
      os << "    %slice" << index << " = memref.subview %buffer[0, " << row
         << ", 0] [1, 1, 128] [1, 1, 1] : memref<1x" << extent
         << "x128xf16, #wafer.memory<spm, tensor>> to "
            "memref<1x1x128xf16, strided<["
         << stride << ", 128, 1], offset: " << row * 128
         << ">, #wafer.memory<spm, tensor>>\n"
         << "    %token" << index << " = wafer.instr.dte_"
         << (send ? "send" : "recv") << " %slice" << index
         << " {peer = " << peer
         << " : i64, bytes = 256 : i64, message = "
            "#wafer.dte_message<communication = "
         << 80 + (separateCommunications ? index : 0)
         << ", round = 0, slice = " << index
         << ">} : memref<1x1x128xf16, strided<[" << stride
         << ", 128, 1], offset: " << row * 128
         << ">, #wafer.memory<spm, tensor>> -> !async.token\n";
    }
    os << "    return\n  }\n}\n";
    return text;
  };

  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE((llvm::Twine("extent=") + llvm::Twine(extent)).str());
    auto source = parse(makeFragments(/*send=*/true, /*peer=*/1,
                                      /*allocationOffset=*/65536, extent,
                                      /*rowBase=*/0));
    auto destination = parse(makeFragments(/*send=*/false, /*peer=*/0,
                                           /*allocationOffset=*/393216, extent,
                                           /*rowBase=*/2));
    ASSERT_TRUE(source);
    ASSERT_TRUE(destination);
    llvm::SmallVector<mlir::ModuleOp, 2> modules{*source, *destination};
    auto coalesced =
        wafer::compiler::detail::coalesceExactDirectDTETransfers(modules);
    ASSERT_TRUE(coalesced.succeeded()) << coalesced.detail;
    EXPECT_EQ(coalesced.statistics.coalescedP2PTransfers, 1u);
    EXPECT_EQ(coalesced.statistics.unicastSendOperationsRemoved, 2u);
    EXPECT_EQ(coalesced.statistics.unicastReceiveOperationsRemoved, 2u);

    wafer::InstrDTESendOp send;
    wafer::InstrDTERecvOp receive;
    source->walk([&](wafer::InstrDTESendOp operation) { send = operation; });
    destination->walk(
        [&](wafer::InstrDTERecvOp operation) { receive = operation; });
    ASSERT_TRUE(send);
    ASSERT_TRUE(receive);
    EXPECT_EQ(send.getBytesAttr().getInt(), 512);
    EXPECT_EQ(receive.getBytesAttr().getInt(), 512);
    ASSERT_TRUE(send.getBufferOffset());
    ASSERT_TRUE(receive.getBufferOffset());
    EXPECT_EQ(*send.getBufferOffset(), 0);
    EXPECT_EQ(*receive.getBufferOffset(), 512);

    auto completion =
        wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
    ASSERT_TRUE(completion.succeeded()) << completion.detail;
    auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
    ASSERT_TRUE(mlir::succeeded(contract));
    EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
  }

  auto gapSource = parse(makeFragments(/*send=*/true, /*peer=*/1,
                                       /*allocationOffset=*/65536,
                                       /*extent=*/1025, /*rowBase=*/0,
                                       /*rowStep=*/2));
  auto gapDestination = parse(makeFragments(
      /*send=*/false, /*peer=*/0, /*allocationOffset=*/393216,
      /*extent=*/1025, /*rowBase=*/2, /*rowStep=*/2));
  ASSERT_TRUE(gapSource);
  ASSERT_TRUE(gapDestination);
  llvm::SmallVector<mlir::ModuleOp, 2> gapModules{*gapSource, *gapDestination};
  auto kept =
      wafer::compiler::detail::coalesceExactDirectDTETransfers(gapModules);
  ASSERT_TRUE(kept.succeeded()) << kept.detail;
  EXPECT_EQ(kept.statistics.coalescedP2PTransfers, 0u);
  unsigned remainingSends = 0;
  gapSource->walk([&](wafer::InstrDTESendOp) { ++remainingSends; });
  EXPECT_EQ(remainingSends, 2u);

  auto phaseSource = parse(makeFragments(
      /*send=*/true, /*peer=*/1, /*allocationOffset=*/65536,
      /*extent=*/1025, /*rowBase=*/0, /*rowStep=*/1,
      /*separateCommunications=*/true));
  auto phaseDestination = parse(makeFragments(
      /*send=*/false, /*peer=*/0, /*allocationOffset=*/393216,
      /*extent=*/1025, /*rowBase=*/2, /*rowStep=*/1,
      /*separateCommunications=*/true));
  ASSERT_TRUE(phaseSource);
  ASSERT_TRUE(phaseDestination);
  llvm::SmallVector<mlir::ModuleOp, 2> phaseModules{*phaseSource,
                                                    *phaseDestination};
  auto phasesKept =
      wafer::compiler::detail::coalesceExactDirectDTETransfers(phaseModules);
  ASSERT_TRUE(phasesKept.succeeded()) << phasesKept.detail;
  EXPECT_EQ(phasesKept.statistics.coalescedP2PTransfers, 0u);
}

TEST_F(DirectDTETransportTest,
       FullCardAllToAllCoalescesToOneNativeScatterPerSource) {
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module {\n  func.func @main() {\n"
          "    %send_buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : "
          "memref<1x1025x128xf16, #wafer.memory<spm, tensor>>\n";
    int64_t receiveOrdinal = 0;
    int64_t sendOrdinal = 0;
    for (int64_t round = 0; round < 4; ++round) {
      for (int64_t source = round * 4;
           source < std::min<int64_t>((round + 1) * 4, 16); ++source) {
        if (source == tile)
          continue;
        int64_t payloadSlice = tile < source ? tile : tile - 1;
        os << "    %receive_buffer" << receiveOrdinal
           << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
           << 393216 + receiveOrdinal * 512
           << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>>\n"
           << "    %receive" << receiveOrdinal
           << " = wafer.instr.dte_recv %receive_buffer" << receiveOrdinal
           << " {peer = " << source
           << " : i64, bytes = 256 : i64, message = "
              "#wafer.dte_message<communication = "
           << 100 + source << ", round = " << round
           << ", slice = " << payloadSlice
           << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> "
              "!async.token\n";
        ++receiveOrdinal;
      }
      if (tile / 4 != round)
        continue;
      for (int64_t destination = 0; destination < 16; ++destination) {
        if (destination == tile)
          continue;
        int64_t segment = destination < tile ? destination : destination - 1;
        os << "    %send_slice" << sendOrdinal
           << " = memref.subview %send_buffer[0, " << segment
           << ", 0] [1, 1, 128] [1, 1, 1] : "
              "memref<1x1025x128xf16, #wafer.memory<spm, tensor>> to "
              "memref<1x1x128xf16, strided<[131200, 128, 1], offset: "
           << segment * 128 << ">, #wafer.memory<spm, tensor>>\n"
           << "    %send" << sendOrdinal
           << " = wafer.instr.dte_send %send_slice" << sendOrdinal
           << " {peer = " << destination
           << " : i64, bytes = 256 : i64, message = "
              "#wafer.dte_message<communication = "
           << 100 + tile << ", round = " << round << ", slice = " << segment
           << ">} : memref<1x1x128xf16, "
              "strided<[131200, 128, 1], offset: "
           << segment * 128
           << ">, #wafer.memory<spm, tensor>> -> !async.token\n";
        ++sendOrdinal;
      }
    }
    os << "    return\n  }\n}\n";
    owners.push_back(parse(text));
    ASSERT_TRUE(owners.back()) << "tile=" << tile;
    modules.push_back(*owners.back());
  }

  auto grouped =
      wafer::compiler::detail::materializeNativeDirectDTEMultiSends(modules);
  ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
  EXPECT_EQ(grouped.statistics.scatterOperations, 16u);
  EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved, 240u);
  unsigned scatterCount = 0;
  unsigned sendCount = 0;
  for (mlir::ModuleOp module : modules) {
    module.walk([&](wafer::InstrDTEScatterOp) { ++scatterCount; });
    module.walk([&](wafer::InstrDTESendOp) { ++sendCount; });
  }
  EXPECT_EQ(scatterCount, 16u);
  EXPECT_EQ(sendCount, 0u);

  auto completion =
      wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(completion.succeeded()) << completion.detail;
  auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
  for (mlir::ModuleOp module : modules) {
    wafer::InstrDTEScatterOp scatter;
    module.walk(
        [&](wafer::InstrDTEScatterOp operation) { scatter = operation; });
    ASSERT_TRUE(scatter);
    ASSERT_TRUE(scatter.getBindings());
    EXPECT_EQ(scatter.getBindings()->size(), 15u);
  }
}

TEST_F(DirectDTETransportTest,
       FullCardAllGatherCoalescesToOneNativeBroadcastPerSource) {
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module {\n  func.func @main() {\n"
          "    %send_buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : "
          "memref<1x2x64xf16, #wafer.memory<spm, tensor>>\n";
    int64_t receiveOrdinal = 0;
    int64_t sendOrdinal = 0;
    for (int64_t round = 0; round < 4; ++round) {
      for (int64_t source = round * 4;
           source < std::min<int64_t>((round + 1) * 4, 16); ++source) {
        if (source == tile)
          continue;
        int64_t payloadSlice = tile < source ? tile : tile - 1;
        os << "    %receive_buffer" << receiveOrdinal
           << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
           << 393216 + receiveOrdinal * 512
           << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>>\n"
           << "    %receive" << receiveOrdinal
           << " = wafer.instr.dte_recv %receive_buffer" << receiveOrdinal
           << " {peer = " << source
           << " : i64, bytes = 256 : i64, message = "
              "#wafer.dte_message<communication = "
           << 200 + source << ", round = " << round
           << ", slice = " << payloadSlice
           << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> "
              "!async.token\n";
        ++receiveOrdinal;
      }
      if (tile / 4 != round)
        continue;
      for (int64_t destination = 0; destination < 16; ++destination) {
        if (destination == tile)
          continue;
        int64_t payloadSlice =
            destination < tile ? destination : destination - 1;
        os << "    %send" << sendOrdinal
           << " = wafer.instr.dte_send %send_buffer {peer = " << destination
           << " : i64, bytes = 256 : i64, message = "
              "#wafer.dte_message<communication = "
           << 200 + tile << ", round = " << round
           << ", slice = " << payloadSlice
           << ">} : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> "
              "!async.token\n";
        ++sendOrdinal;
      }
    }
    os << "    return\n  }\n}\n";
    owners.push_back(parse(text));
    ASSERT_TRUE(owners.back()) << "tile=" << tile;
    modules.push_back(*owners.back());
  }

  auto grouped =
      wafer::compiler::detail::materializeNativeDirectDTEMultiSends(modules);
  ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
  EXPECT_EQ(grouped.statistics.broadcastOperations, 16u);
  EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved, 240u);
  auto completion =
      wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(completion.succeeded()) << completion.detail;
  auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
  for (mlir::ModuleOp module : modules) {
    unsigned broadcasts = 0;
    module.walk([&](wafer::InstrDTEBroadcastOp operation) {
      ++broadcasts;
      ASSERT_TRUE(operation.getBindings());
      EXPECT_EQ(operation.getBindings()->size(), 15u);
    });
    EXPECT_EQ(broadcasts, 1u);
  }
}

TEST_F(DirectDTETransportTest,
       RecursiveDoublingUsesActualContiguousGatherRanges) {
  auto makeTile = [](int64_t participantCount, int64_t tile, int64_t extent) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module {\n"
          "  func.func @main(%boundary: memref<1xi8, "
          "#wafer.memory<ddr, tensor>>) {\n"
          "    %unused = wafer.tile.region(%boundary : memref<1xi8, "
          "#wafer.memory<ddr, tensor>>) -> "
          "(memref<1xi8, #wafer.memory<ddr, tensor>>) {\n"
          "    ^bb0(%ddr: memref<1xi8, #wafer.memory<ddr, tensor>>):\n"
          "    %gather = memref.alloc() : memref<"
       << participantCount << "x1x" << extent
       << "xf16, #wafer.memory<spm, tensor>>\n"
       << "    %local = memref.subview %gather[" << tile << ", 0, 0] "
       << "[1, 1, " << extent << "] [1, 1, 1] : memref<" << participantCount
       << "x1x" << extent << "xf16, #wafer.memory<spm, tensor>> to memref<1x1x"
       << extent << "xf16, strided<[" << extent << ", " << extent
       << ", 1], offset: " << tile * extent
       << ">, #wafer.memory<spm, tensor>>\n"
          "    %zero = arith.constant 0.000000e+00 : f16\n"
          "    wafer.instr.fill %local, %zero : memref<1x1x"
       << extent << "xf16, strided<[" << extent << ", " << extent
       << ", 1], offset: " << tile * extent
       << ">, #wafer.memory<spm, tensor>>, f16\n";
    for (int64_t half = 1, round = 0; half < participantCount;
         half *= 2, ++round) {
      int64_t groupBase = tile & ~(2 * half - 1);
      bool upper = (tile & half) != 0;
      int64_t sendStart = groupBase + (upper ? half : 0);
      int64_t receiveStart = groupBase + (upper ? 0 : half);
      int64_t peer = tile ^ half;
      int64_t bytes = half * extent * 2;
      os << "    %receive_view" << round << " = memref.subview %gather["
         << receiveStart << ", 0, 0] [" << half << ", 1, " << extent
         << "] [1, 1, 1] : memref<" << participantCount << "x1x" << extent
         << "xf16, #wafer.memory<spm, tensor>> to memref<" << half << "x1x"
         << extent << "xf16, strided<[" << extent << ", " << extent
         << ", 1], offset: " << receiveStart * extent
         << ">, #wafer.memory<spm, tensor>>\n"
         << "    %send_view" << round << " = memref.subview %gather["
         << sendStart << ", 0, 0] [" << half << ", 1, " << extent
         << "] [1, 1, 1] : memref<" << participantCount << "x1x" << extent
         << "xf16, #wafer.memory<spm, tensor>> to memref<" << half << "x1x"
         << extent << "xf16, strided<[" << extent << ", " << extent
         << ", 1], offset: " << sendStart * extent
         << ">, #wafer.memory<spm, tensor>>\n"
         << "    %receive" << round << " = wafer.instr.dte_recv %receive_view"
         << round << " {peer = " << peer << " : i64, bytes = " << bytes
         << " : i64, message = #wafer.dte_message<communication = 200, "
            "round = "
         << round << ", slice = " << receiveStart << ">} : memref<" << half
         << "x1x" << extent << "xf16, strided<[" << extent << ", " << extent
         << ", 1], offset: " << receiveStart * extent
         << ">, #wafer.memory<spm, tensor>> -> !async.token\n"
         << "    %send" << round << " = wafer.instr.dte_send %send_view"
         << round << " {peer = " << peer << " : i64, bytes = " << bytes
         << " : i64, message = #wafer.dte_message<communication = 200, "
            "round = "
         << round << ", slice = " << sendStart << ">} : memref<" << half
         << "x1x" << extent << "xf16, strided<[" << extent << ", " << extent
         << ", 1], offset: " << sendStart * extent
         << ">, #wafer.memory<spm, tensor>> -> !async.token\n";
    }
    os << "    wafer.tile.yield %ddr : "
          "memref<1xi8, #wafer.memory<ddr, tensor>>\n"
          "    }\n"
          "    return\n"
          "  }\n"
          "}\n";
    return text;
  };

  for (int64_t participantCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE((llvm::Twine("participants=") +
                    llvm::Twine(participantCount) +
                    ", extent=" + llvm::Twine(extent))
                       .str());
      llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
      llvm::SmallVector<mlir::ModuleOp, 16> modules;
      for (int64_t tile = 0; tile < participantCount; ++tile) {
        owners.push_back(parse(makeTile(participantCount, tile, extent)));
        ASSERT_TRUE(owners.back()) << "tile=" << tile;
        modules.push_back(*owners.back());
      }

      for (auto &owner : owners)
        ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*owner)));
      auto completion =
          wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      modules.clear();
      for (auto &owner : owners) {
        wafer::compiler::detail::TileMemoryPlanningFailure failure;
        auto planned = wafer::compiler::detail::planTileMemory(
            std::move(owner), &failure,
            /*materializationRelations=*/nullptr,
            /*emitSPMCapacityDiagnostics=*/false);
        ASSERT_TRUE(mlir::succeeded(planned));
        owner = std::move(*planned);
        modules.push_back(*owner);
      }
      auto contract = wafer::compiler::testing::bindDirectDTETransport(modules);
      ASSERT_TRUE(mlir::succeeded(contract));
      EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);

      int64_t expectedRounds = participantCount == 4 ? 2 : 4;
      int64_t expectedBytes = (participantCount - 1) * extent * 2;
      for (auto [tile, module] : llvm::enumerate(modules)) {
        int64_t sends = 0;
        int64_t receives = 0;
        int64_t sendBytes = 0;
        int64_t receiveBytes = 0;
        llvm::SmallVector<unsigned, 16> covered(participantCount, 0);
        covered[tile] = 1;
        module.walk([&](wafer::InstrDTESendOp send) {
          ++sends;
          sendBytes += send.getBytesAttr().getInt();
          int64_t round = send.getMessageAttr().getRound();
          EXPECT_EQ(send.getPeerAttr().getInt(),
                    static_cast<int64_t>(tile) ^ (int64_t{1} << round));
          EXPECT_EQ(send.getBytesAttr().getInt(),
                    (int64_t{1} << round) * extent * 2);
        });
        module.walk([&](wafer::InstrDTERecvOp receive) {
          ++receives;
          receiveBytes += receive.getBytesAttr().getInt();
          int64_t round = receive.getMessageAttr().getRound();
          EXPECT_EQ(receive.getPeerAttr().getInt(),
                    static_cast<int64_t>(tile) ^ (int64_t{1} << round));
          int64_t blockCount = int64_t{1} << round;
          EXPECT_EQ(receive.getBytesAttr().getInt(), blockCount * extent * 2);
          int64_t first = receive.getMessageAttr().getPayloadSlice();
          for (int64_t block = 0; block < blockCount; ++block) {
            ASSERT_GE(first + block, 0);
            ASSERT_LT(first + block, participantCount);
            ++covered[first + block];
          }
        });
        EXPECT_EQ(sends, expectedRounds);
        EXPECT_EQ(receives, expectedRounds);
        EXPECT_EQ(sendBytes, expectedBytes);
        EXPECT_EQ(receiveBytes, expectedBytes);
        EXPECT_TRUE(
            llvm::all_of(covered, [](unsigned count) { return count == 1; }));
        unsigned allocations = 0;
        module.walk([&](mlir::memref::AllocOp allocation) {
          ++allocations;
          EXPECT_TRUE(
              static_cast<bool>(allocation->getAttrOfType<wafer::SPMOffsetAttr>(
                  wafer::kWaferSPMOffsetAttrName)));
        });
        EXPECT_EQ(allocations, 1u);
      }
    }
  }

  auto oversized = parse(makeTile(/*participantCount=*/16, /*tile=*/0,
                                  /*extent=*/196608));
  ASSERT_TRUE(oversized);
  llvm::SmallVector<mlir::ModuleOp, 1> oversizedModules{*oversized};
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*oversized)));
  auto oversizedCompletion =
      wafer::compiler::detail::rebuildRequiredDirectDTEWaits(oversizedModules);
  ASSERT_TRUE(oversizedCompletion.succeeded()) << oversizedCompletion.detail;
  wafer::compiler::detail::TileMemoryPlanningFailure capacityFailure;
  auto rejected = wafer::compiler::detail::planTileMemory(
      std::move(oversized), &capacityFailure,
      /*materializationRelations=*/nullptr,
      /*emitSPMCapacityDiagnostics=*/false);
  EXPECT_TRUE(mlir::failed(rejected));
  EXPECT_EQ(
      capacityFailure.kind,
      wafer::compiler::detail::TileMemoryPlanningFailureKind::SPMAllocation);
  EXPECT_TRUE(capacityFailure.spmCapacityOverflow);
}

TEST_F(DirectDTETransportTest,
       NonContiguousSubviewsKeepRootLevelCompletionConflict) {
  constexpr llvm::StringLiteral kNonContiguous = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<1x1x1031xf16, #wafer.memory<spm, tensor>>
    %receive_view = memref.subview %buffer[0, 0, 0] [1, 1, 516] [1, 1, 2]
        : memref<1x1x1031xf16, #wafer.memory<spm, tensor>> to
          memref<1x1x516xf16, strided<[1031, 1031, 2]>,
                 #wafer.memory<spm, tensor>>
    %send_view = memref.subview %buffer[0, 0, 1] [1, 1, 515] [1, 1, 2]
        : memref<1x1x1031xf16, #wafer.memory<spm, tensor>> to
          memref<1x1x515xf16, strided<[1031, 1031, 2], offset: 1>,
                 #wafer.memory<spm, tensor>>
    %receive = wafer.instr.dte_recv %receive_view
        {peer = 1 : i64, bytes = 1032 : i64,
         message = #wafer.dte_message<communication = 201, round = 0, slice = 0>}
        : memref<1x1x516xf16, strided<[1031, 1031, 2]>,
                 #wafer.memory<spm, tensor>> -> !async.token
    %send = wafer.instr.dte_send %send_view
        {peer = 1 : i64, bytes = 1030 : i64,
         message = #wafer.dte_message<communication = 202, round = 0, slice = 0>}
        : memref<1x1x515xf16, strided<[1031, 1031, 2], offset: 1>,
                 #wafer.memory<spm, tensor>> -> !async.token
    return
  }
})mlir";
  auto module = parse(kNonContiguous);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  auto completion =
      wafer::compiler::detail::rebuildRequiredDirectDTEWaits(modules);
  ASSERT_TRUE(completion.succeeded()) << completion.detail;

  llvm::SmallVector<mlir::Operation *, 4> transport;
  mlir::func::FuncOp function;
  module->walk([&](mlir::func::FuncOp current) { function = current; });
  ASSERT_TRUE(function);
  for (mlir::Operation &operation : function.getBody().front())
    if (mlir::isa<wafer::InstrDTERecvOp, wafer::InstrDTESendOp,
                  wafer::InstrDTEWaitOp>(operation))
      transport.push_back(&operation);
  ASSERT_EQ(transport.size(), 4u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTERecvOp>(transport[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(transport[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(transport[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(transport[3]));
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
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);

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

TEST_F(DirectDTETransportTest,
       ResolvesStaticLoopBoundsThroughNestedTileRegions) {
  auto makeTileModule = [](bool isSend, int64_t peer, int64_t spmOffset) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module {\n"
          "  func.func @main() {\n"
          "    %c0 = arith.constant 0 : index\n"
          "    %c2 = arith.constant 2 : index\n"
          "    %c4 = arith.constant 4 : index\n"
          "    %bounds:3 = wafer.tile.region(%c0, %c4, %c2 : index, index, "
          "index) -> (index, index, index) {\n"
          "    ^bb0(%lower: index, %upper: index, %step: index):\n"
          "      wafer.tile.yield %lower, %upper, %step : index, index, "
          "index\n"
          "    }\n"
          "    %forwarded:3 = wafer.tile.region(%bounds#0, %bounds#1, "
          "%bounds#2 : index, index, index) -> (index, index, index) {\n"
          "    ^bb0(%lower: index, %upper: index, %step: index):\n"
          "      %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<"
       << spmOffset
       << ">} : memref<4xf32, #wafer.memory<spm, tensor>>\n"
          "      scf.for %iteration = %lower to %upper step %step {\n"
          "        %token = wafer.instr.dte_"
       << (isSend ? "send" : "recv") << " %buffer {peer = " << peer
       << " : i64, bytes = 16 : i64, message = "
          "#wafer.dte_message<communication = 9, "
          "round = 2, slice = 0>} : memref<4xf32, "
          "#wafer.memory<spm, tensor>> -> !async.token\n"
          "        wafer.instr.dte_wait %token : !async.token\n"
          "      }\n"
          "      wafer.tile.yield %lower, %upper, %step : index, index, "
          "index\n"
          "    }\n"
          "    return\n"
          "  }\n"
          "}\n";
    return source;
  };
  auto sendModule = parse(makeTileModule(/*isSend=*/true, /*peer=*/1,
                                         /*spmOffset=*/65536));
  auto recvModule = parse(makeTileModule(/*isSend=*/false, /*peer=*/0,
                                         /*spmOffset=*/65792));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_TRUE(operation.getBinding());
  });
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_TRUE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest,
       AcceptsInvariantEndpointsWithoutExpandingLargeTripCount) {
  constexpr llvm::StringLiteral kLargeLoopPrefix = R"mlir(
    %large = arith.constant 4294967296 : index
    scf.for %iteration = %c0 to %large step %c2 {
)mlir";
  constexpr llvm::StringLiteral kLoopSuffix = R"mlir(
    }
)mlir";
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kLargeLoopPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kLargeLoopPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_TRUE(operation.getBinding());
  });
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_TRUE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, ReceivePreparationBreaksCrossTileSendWaitCycle) {
  std::string tile0Source = makeLinearTransportTileModule(
      {{false, 1, 65536, 31}, {true, 1, 65792, 30}},
      /*waitImmediately=*/false);
  std::string tile1Source = makeLinearTransportTileModule(
      {{false, 0, 66048, 30}, {true, 0, 66304, 31}},
      /*waitImmediately=*/false);
  auto tile0Module = parse(tile0Source);
  auto tile1Module = parse(tile1Source);
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*tile0Module, *tile1Module};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest,
       ReceivePreparationMayOverlapDisjointEffectfulLoop) {
  auto sendModule = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 262400 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir");
  auto recvModule = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1025 = arith.constant 1025 : index
    %zero = arith.constant 0.0 : f16
    %receive = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    %other = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<328192>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %receive
        {peer = 0 : i64, bytes = 262400 : i64,
         message = #wafer.dte_message<communication = 9, round = 2, slice = 0>}
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>> -> !async.token
    scf.for %index = %c0 to %c1025 step %c1 {
      memref.store %zero, %other[%c0, %index, %c0]
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    }
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir");
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, MutualSendBeforeReceiveWaitCycleFailsClosed) {
  std::string tile0Source = makeLinearTransportTileModule(
      {{true, 1, 65536, 30}, {false, 1, 65792, 31}},
      /*waitImmediately=*/true);
  std::string tile1Source = makeLinearTransportTileModule(
      {{true, 0, 66048, 31}, {false, 0, 66304, 30}},
      /*waitImmediately=*/true);
  auto tile0Module = parse(tile0Source);
  auto tile1Module = parse(tile1Source);
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*tile0Module, *tile1Module};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("wait graph contains a cyclic dependency"),
            std::string::npos);
  for (mlir::ModuleOp module : tileModules) {
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
  auto tile0Module =
      parse(makeStructuredPhaseTileModule(/*prologueIsSend=*/false,
                                          /*steadyIsSend=*/false,
                                          /*epilogueIsSend=*/true, /*peer=*/1,
                                          /*baseOffset=*/65536));
  auto tile1Module =
      parse(makeStructuredPhaseTileModule(/*prologueIsSend=*/true,
                                          /*steadyIsSend=*/true,
                                          /*epilogueIsSend=*/false, /*peer=*/0,
                                          /*baseOffset=*/66560));
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*tile0Module, *tile1Module};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, AcceptsOrderedSiblingLoopOccurrences) {
  auto tile0Module =
      parse(makeSiblingLoopTileModule(/*firstIsSend=*/false,
                                      /*secondIsSend=*/true, /*peer=*/1,
                                      /*baseOffset=*/65536));
  auto tile1Module =
      parse(makeSiblingLoopTileModule(/*firstIsSend=*/true,
                                      /*secondIsSend=*/false, /*peer=*/0,
                                      /*baseOffset=*/66560));
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*tile0Module, *tile1Module};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest,
       CountsTileRegionSiblingsIndependentOfTransportContent) {
  auto tile0Module = parse(makeTileRegionSiblingTileModule(wafer::TileId(0)));
  auto tile1Module = parse(makeTileRegionSiblingTileModule(wafer::TileId(1)));
  auto tile2Module = parse(makeTileRegionSiblingTileModule(wafer::TileId(2)));
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  ASSERT_TRUE(tile2Module);
  llvm::SmallVector<mlir::ModuleOp, 3> tileModules{*tile0Module, *tile1Module,
                                                   *tile2Module};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest,
       MatchesMessagesAcrossDifferentSequentialTileRegionOrdinals) {
  auto sendModule = parse(makeShiftedTileRegionTransportProgram(
      /*isSend=*/true, /*region=*/0));
  auto recvModule = parse(makeShiftedTileRegionTransportProgram(
      /*isSend=*/false, /*region=*/1));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, CrossBlockWaitCycleFailsClosed) {
  auto tile0Module = parse(
      makeCrossBlockCycleTileModule(/*firstTile=*/true, /*baseOffset=*/65536));
  auto tile1Module = parse(makeCrossBlockCycleTileModule(
      /*firstTile=*/false, /*baseOffset=*/66560));
  ASSERT_TRUE(tile0Module);
  ASSERT_TRUE(tile1Module);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*tile0Module, *tile1Module};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("wait graph contains a cyclic dependency"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest,
       HelperDefinitionOrderDoesNotDefineMessageOccurrence) {
  auto sendModule =
      parse(makeTwoHelperTileModule(/*isSend=*/true, /*peer=*/1,
                                    /*baseOffset=*/65536,
                                    /*reverseDefinitions=*/false,
                                    /*reverseCalls=*/false,
                                    /*reuseMessageIdentity=*/true));
  auto recvModule =
      parse(makeTwoHelperTileModule(/*isSend=*/false, /*peer=*/0,
                                    /*baseOffset=*/66560,
                                    /*reverseDefinitions=*/true,
                                    /*reverseCalls=*/false,
                                    /*reuseMessageIdentity=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, MismatchedHelperCallOccurrenceFailsClosed) {
  auto sendModule = parse(makeTwoHelperTileModule(/*isSend=*/true, /*peer=*/1,
                                                  /*baseOffset=*/65536,
                                                  /*reverseDefinitions=*/false,
                                                  /*reverseCalls=*/false));
  auto recvModule = parse(makeTwoHelperTileModule(/*isSend=*/false, /*peer=*/0,
                                                  /*baseOffset=*/66560,
                                                  /*reverseDefinitions=*/true,
                                                  /*reverseCalls=*/true));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("occurrence paths are not structurally "
                                "identical across physical Tiles"),
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
         message = #wafer.dte_message<communication = 52, round = 0, slice = 0>}
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
         message = #wafer.dte_message<communication = 52, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
  func.func private @second() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 52, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  auto sendModule = parse(kRepeatedSendSite);
  auto recvModule = parse(kDistinctReceiveSites);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
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
         message = #wafer.dte_message<communication = 70, round = 0, slice = 0>}
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
         message = #wafer.dte_message<communication = 70, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    return
  }
})mlir";
  auto sendModule = parse(kUnusedSend);
  auto recvModule = parse(kUsedRecv);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("outside the entry call closure"),
            std::string::npos);
}

TEST_F(DirectDTETransportTest,
       ReceiveHoistedAcrossSamePeerLoopFailsReadySlotReuse) {
  constexpr llvm::StringLiteral kNestedPrefix = R"mlir(
    scf.for %outer = %c0 to %c8 step %c2 {
      scf.for %inner = %c0 to %c4 step %c2 {
)mlir";
  constexpr llvm::StringLiteral kNestedSuffix = R"mlir(
      }
    }
)mlir";
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kNestedPrefix, kNestedSuffix,
      /*addStaticTail=*/true));
  auto recvModule = parse(makeControlledTileModule(
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
  // This bounded negative isolates a formerly accepted hoist: distinct ranges
  // and an acyclic token-completion graph do not protect the peer ready slot.
  // The hoisted receive and the first loop receive overwrite one notification.
  mlir::OpBuilder builder(outerLoop);
  auto tailBuffer = builder.create<mlir::memref::AllocOp>(
      staticTail.getLoc(),
      mlir::cast<mlir::MemRefType>(staticTail.getBuffer().getType()));
  tailBuffer->setAttr(
      wafer::kWaferSPMOffsetAttrName,
      wafer::SPMOffsetAttr::get(staticTail.getContext(), /*offset=*/66048));
  staticTail->setOperand(0, tailBuffer);
  staticTail->moveBefore(outerLoop);

  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  std::string diagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnosticText);
        diagnostic.print(stream);
        return mlir::success();
      });
  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  EXPECT_NE(diagnosticText.find("reuses a peer ready slot"), std::string::npos);
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
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kSendPrefix, kSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kRecvPrefix, kSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
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
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"(%bound: index)", kDynamicPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"(%bound: index)", kDynamicPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
}

TEST_F(DirectDTETransportTest, ConditionalControlInstanceFailsClosed) {
  constexpr llvm::StringLiteral kConditionalPrefix = R"mlir(
    scf.if %condition {
)mlir";
  constexpr llvm::StringLiteral kConditionalSuffix = R"mlir(
    }
)mlir";
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"(%condition: i1)", kConditionalPrefix,
      kConditionalSuffix, /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"(%condition: i1)", kConditionalPrefix,
      kConditionalSuffix, /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
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
  auto sendModule = parse(makeControlledTileModule(
      /*isSend=*/true, /*peer=*/1, /*spmOffset=*/65536,
      /*functionArguments=*/"()", kSendPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  auto recvModule = parse(makeControlledTileModule(
      /*isSend=*/false, /*peer=*/0, /*spmOffset=*/65792,
      /*functionArguments=*/"()", kRecvPrefix, kLoopSuffix,
      /*addStaticTail=*/false));
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, LoopEscapingIssueTokenFailsClosed) {
  auto sendModule = parse(kLoopEscapingTokenSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, InterveningIssueBufferAccessFailsClosed) {
  auto sendModule = parse(kInterveningBufferAccessSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, InterveningSendSourceReadIsAccepted) {
  auto sendModule = parse(kInterveningBufferReadSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  ASSERT_TRUE(mlir::succeeded(contract));
  EXPECT_EQ(*contract, wafer::TransportContract::DirectDTE);
}

TEST_F(DirectDTETransportTest, InterveningReceiveDestinationReadFailsClosed) {
  auto sendModule = parse(kSendTileModule);
  auto recvModule = parse(kInterveningBufferReadRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, MissingPeerLeavesBindingsUnset) {
  auto sendModule = parse(kSendTileModule);
  ASSERT_TRUE(sendModule);
  llvm::SmallVector<mlir::ModuleOp, 1> tileModules{*sendModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([&](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, ByteMismatchLeavesBindingsUnset) {
  auto sendModule = parse(kSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  recvModule->walk([](wafer::InstrDTERecvOp operation) {
    operation.setBytesAttr(mlir::IntegerAttr::get(
        mlir::IntegerType::get(operation.getContext(), 64), 8));
  });
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  sendModule->walk([&](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
  recvModule->walk([&](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, UnplannedSPMRangeFailsClosed) {
  auto sendModule = parse(kSendTileModule);
  auto recvModule = parse(kRecvTileModule);
  ASSERT_TRUE(sendModule);
  ASSERT_TRUE(recvModule);
  sendModule->walk([](mlir::memref::AllocOp operation) {
    operation->removeAttr(wafer::kWaferSPMOffsetAttrName);
  });
  llvm::SmallVector<mlir::ModuleOp, 2> tileModules{*sendModule, *recvModule};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  recvModule->walk([&](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, OverlappingNormalSendersFailClosed) {
  auto module = parse(kOverlappingSendTileModule);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> tileModules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  module->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(DirectDTETransportTest, FifthOverlappingReceiverFailsClosed) {
  auto module = parse(kFiveLiveReceiversTileModule);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::ModuleOp, 1> tileModules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto contract = wafer::compiler::testing::bindDirectDTETransport(tileModules);
  EXPECT_TRUE(mlir::failed(contract));
  module->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

} // namespace
