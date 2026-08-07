//===- CoordinatedExecutableFinalizationTest.cpp
//----------------------------===//

#include "../../lib/Wafer/Compiler/CoordinatedExecutableFinalization.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/CoordinatedVariantSelection.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace {

class MarkerCommunicationPoint final
    : public wafer::compiler::detail::CoordinatedCommunicationActionPoint {
public:
  MarkerCommunicationPoint()
      : CoordinatedCommunicationActionPoint({"test-communication-action", 7}) {}

  mlir::LogicalResult materialize(
      llvm::MutableArrayRef<mlir::ModuleOp> isolatedCanonicalInstrModules,
      const wafer::frontend::FrontendProgramVerificationResult &,
      std::string *failureReason) const final {
    for (mlir::ModuleOp module : isolatedCanonicalInstrModules)
      module->setAttr("wafer.test_communication_marker",
                      mlir::UnitAttr::get(module.getContext()));
    if (failureReason)
      failureReason->clear();
    return mlir::success();
  }
};

class MarkerCommunicationProvider final
    : public wafer::compiler::detail::CoordinatedCommunicationActionProvider {
public:
  llvm::StringRef getStableKey() const final {
    return "test-communication-action";
  }

  mlir::LogicalResult
  query(llvm::ArrayRef<mlir::ModuleOp>,
        const wafer::frontend::FrontendProgramVerificationResult &,
        wafer::compiler::detail::CoordinatedCommunicationActionPoints &points,
        std::string *failureReason) const final {
    points.push_back(std::make_unique<MarkerCommunicationPoint>());
    if (failureReason)
      failureReason->clear();
    return mlir::success();
  }
};

class CoordinatedExecutableFinalizationTest : public ::testing::Test {
protected:
  CoordinatedExecutableFinalizationTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeProgram(int64_t rankCount) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: )mlir"
       << rankCount << R"mlir(>, policy = "all_available",
       endpoints = array<i64>}
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeChainProgram(int64_t rankCount) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: )mlir"
       << rankCount << R"mlir(>, policy = "all_available",
       endpoints = array<i64>}
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %consumer_out = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>) outs(%consumer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %squared = arith.mulf %value, %value : f16
      linalg.yield %squared : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeLocalReductionAllReduceProgram(int64_t rankCount) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: )mlir"
       << rankCount << R"mlir(>, policy = "all_available",
       endpoints = array<i64>}
  func.func @main(%input: tensor<4x8xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.0 : f16
    %local_out = tensor.empty() : tensor<4xf16>
    %local_init = linalg.fill ins(%zero : f16)
        outs(%local_out : tensor<4xf16>) -> tensor<4xf16>
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<4x8xf16>) outs(%local_init : tensor<4xf16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %sum = arith.addf %value, %accumulator : f16
      linalg.yield %sum : f16
    } -> tensor<4xf16>
    %global_out = tensor.empty() : tensor<4xf16>
    %global = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xf16>) outs(%global_out : tensor<4xf16>) {
    ^bb0(%left: f16, %right: f16):
      %sum = arith.addf %left, %right : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 31 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15>}
        -> tensor<4xf16>
    return %global : tensor<4xf16>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeOversizedTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%boundary: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(%boundary
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr into %spm
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %spm, %zero
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.yield %ddr
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeReadyOrderInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %compute_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute_source = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %ddr = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [1]
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeRepairableSPMPressureTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(%input: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>)
      -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
          memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %long_lived = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %long_lived
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      %intervening = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %intervening
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %intervening, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.store %long_lived, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeDDRBoundaryRepairableSPMPressureTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(%input: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>)
      -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
          memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %first = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %first
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %first, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      %second = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %second
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %second, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeFixedSlotInstrProgram(int64_t upperBound = 3) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_upper = arith.constant )mlir"
       << upperBound << R"mlir( : index
    scf.for %iv = %c0 to %c_upper step %c1 {
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
)mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeManyFixedSlotInstrProgram(bool firstLoopMaterializes) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
)mlir";
    for (unsigned index = 0; index < 8; ++index) {
      const int64_t upperBound = index == 0 ? 32 : 10 - index;
      os << "    %c_upper" << index << " = arith.constant " << upperBound
         << " : index\n"
         << "    scf.for %iv" << index << " = %c0 to %c_upper" << index
         << " step %c1 {\n";
      if (index != 0 || firstLoopMaterializes) {
        os << "      %slot" << index << " = memref.alloc()\n"
           << "          : memref<4xf16, #wafer.memory<spm, tensor>>\n"
           << "      wafer.instr.rdma %input to %slot" << index << "\n"
           << "          {byte_count = 8 : i64, inner_bytes = 8 : i64,\n"
           << "           src_iterations = array<i64: 1, 1, 1>,\n"
           << "           src_strides = array<i64: 0, 0, 0>}\n"
           << "          : memref<4xf16, #wafer.memory<ddr, tensor>>\n"
           << "         to memref<4xf16, #wafer.memory<spm, tensor>>\n"
           << "      wafer.instr.elementwise <add> %slot" << index << ", %slot"
           << index << " into %slot" << index << "\n"
           << "          : memref<4xf16, #wafer.memory<spm, tensor>>,\n"
           << "            memref<4xf16, #wafer.memory<spm, tensor>>\n"
           << "        into memref<4xf16, #wafer.memory<spm, tensor>>\n";
      }
      os << "      scf.yield\n"
         << "    }\n";
    }
    os << R"mlir(    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeFixedSlotDirectDTEComputeInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %dte = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %zero = arith.constant 0.000000e+00 : f16
      wafer.instr.fill %dte, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      %token = wafer.instr.dte_send %dte
          {peer = 1 : i64, bytes = 8 : i64,
           message = #wafer.dte_message<communication = 9, phase = peer_dataflow, round = 0, slice = 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
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
      wafer.instr.dte_wait %token : !async.token
      scf.yield
    }
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeWorkerInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %c, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0, 1, 2]
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeReadyOrderTileProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(
      %input: memref<8xf16, #wafer.memory<ddr, tensor>>)
      -> memref<8xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc()
        : memref<8xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<8xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<8xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<8xf16, #wafer.memory<ddr, tensor>>):
      %compute_source = memref.alloc()
          : memref<8xf16, #wafer.memory<spm, tensor>>
      %computed = wafer.tile.elementwise #wafer.elementwise_kind<neg>
          %compute_source
          : (memref<8xf16, #wafer.memory<spm, tensor>>)
         -> memref<8xf16, #wafer.memory<spm, tensor>>
      %loaded = memref.alloc()
          : memref<8xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %loaded
          : memref<8xf16, #wafer.memory<ddr, tensor>>
        into memref<8xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %computed, %out
          : memref<8xf16, #wafer.memory<spm, tensor>>
         -> memref<8xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<8xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<8xf16, #wafer.memory<ddr, tensor>>
  }
}
)mlir",
                                                   context.get());
  }

  static llvm::SmallVector<mlir::ModuleOp, 2>
  views(llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> owners) {
    llvm::SmallVector<mlir::ModuleOp, 2> result;
    for (const auto &owner : owners)
      result.push_back(*owner);
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult
  replicatedProgram(int64_t rankCount, int64_t elements = 8) {
    auto makeBoundary = [&](int64_t index) {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape = {elements};
      binding.localShape = {elements};
      binding.dtype = "f16";
      for (int64_t rank = 0; rank < rankCount; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets = {0};
        slice.sizes = {elements};
        slice.strides = {1};
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };
    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = rankCount;
    program.programUserInputCount = 1;
    program.distributedInputs.push_back(makeBoundary(0));
    program.distributedOutputs.push_back(makeBoundary(0));
    return program;
  }

  wafer::frontend::FrontendProgramVerificationResult
  replicatedReductionProgram(int64_t rankCount) {
    auto makeBoundary = [&](int64_t index, llvm::ArrayRef<int64_t> shape) {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape.assign(shape.begin(), shape.end());
      binding.localShape.assign(shape.begin(), shape.end());
      binding.dtype = "f16";
      for (int64_t rank = 0; rank < rankCount; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets.assign(shape.size(), 0);
        slice.sizes.assign(shape.begin(), shape.end());
        slice.strides.assign(shape.size(), 1);
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };
    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = rankCount;
    program.programUserInputCount = 1;
    program.distributedInputs.push_back(makeBoundary(0, {4, 8}));
    program.distributedOutputs.push_back(makeBoundary(0, {4}));
    return program;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST(CoordinatedExecutableFinalizationLaneCoordinatorTest,
     StartsWithSeedAndRotatesAfterRejectedAttempts) {
  using Coordinator = wafer::compiler::detail::
      CoordinatedExecutableFinalizationLaneCoordinator;
  using Lane =
      wafer::compiler::detail::CoordinatedExecutableFinalizationLane;
  Coordinator coordinator;
  EXPECT_FALSE(coordinator.chooseNextLane(/*seedAvailable=*/true,
                                          /*expansionAvailable=*/true));
  ASSERT_TRUE(
      mlir::succeeded(coordinator.recordMandatoryBaselineAccepted()));

  for (Lane expected : {Lane::Seed, Lane::Expansion, Lane::Seed,
                        Lane::Expansion}) {
    auto lane = coordinator.chooseNextLane(/*seedAvailable=*/true,
                                           /*expansionAvailable=*/true);
    ASSERT_TRUE(lane);
    EXPECT_EQ(*lane, expected);
    ASSERT_TRUE(mlir::succeeded(
        coordinator.recordAttempt(*lane, /*exactAccepted=*/false)));
  }
  EXPECT_EQ(coordinator.getAttemptCount(), 5u);
  EXPECT_EQ(coordinator.getExactAcceptedCount(), 1u);

  // Temporary lane unavailability does not erase the opposite pending turn.
  auto forcedExpansion = coordinator.chooseNextLane(
      /*seedAvailable=*/false, /*expansionAvailable=*/true);
  ASSERT_TRUE(forcedExpansion);
  EXPECT_EQ(*forcedExpansion, Lane::Expansion);
  ASSERT_TRUE(mlir::succeeded(coordinator.recordAttempt(
      *forcedExpansion, /*exactAccepted=*/false)));
  auto resumed = coordinator.chooseNextLane(/*seedAvailable=*/true,
                                            /*expansionAvailable=*/true);
  ASSERT_TRUE(resumed);
  EXPECT_EQ(*resumed, Lane::Seed);
}

TEST(CoordinatedExecutableFinalizationLaneCoordinatorTest,
     StopsAtEightInvocationWideExactAcceptances) {
  using Coordinator = wafer::compiler::detail::
      CoordinatedExecutableFinalizationLaneCoordinator;
  Coordinator coordinator;
  ASSERT_TRUE(
      mlir::succeeded(coordinator.recordMandatoryBaselineAccepted()));
  for (unsigned index = 1;
       index < wafer::compiler::detail::
                   kMaximumCoordinatedExactScheduleActions;
       ++index) {
    auto lane = coordinator.chooseNextLane(/*seedAvailable=*/true,
                                           /*expansionAvailable=*/true);
    ASSERT_TRUE(lane);
    ASSERT_TRUE(mlir::succeeded(
        coordinator.recordAttempt(*lane, /*exactAccepted=*/true)));
  }
  EXPECT_EQ(coordinator.getExactAcceptedCount(), 8u);
  EXPECT_EQ(coordinator.getAttemptCount(), 8u);
  EXPECT_FALSE(coordinator.canAttempt());
  EXPECT_FALSE(coordinator.chooseNextLane(/*seedAvailable=*/true,
                                          /*expansionAvailable=*/true));
}

TEST(CoordinatedExecutableFinalizationLaneCoordinatorTest,
     StopsAtSixteenTotalAttemptsWhenEveryRotatedAttemptRejects) {
  using Coordinator = wafer::compiler::detail::
      CoordinatedExecutableFinalizationLaneCoordinator;
  Coordinator coordinator;
  ASSERT_TRUE(
      mlir::succeeded(coordinator.recordMandatoryBaselineAccepted()));
  while (coordinator.canAttempt()) {
    auto lane = coordinator.chooseNextLane(/*seedAvailable=*/true,
                                           /*expansionAvailable=*/true);
    ASSERT_TRUE(lane);
    ASSERT_TRUE(mlir::succeeded(
        coordinator.recordAttempt(*lane, /*exactAccepted=*/false)));
  }
  EXPECT_EQ(coordinator.getAttemptCount(), 16u);
  EXPECT_EQ(coordinator.getExactAcceptedCount(), 1u);
  EXPECT_FALSE(coordinator.chooseNextLane(/*seedAvailable=*/true,
                                          /*expansionAvailable=*/true));
}

TEST(CoordinatedExecutableFinalizationLaneCoordinatorTest,
     SetupFailureClassificationKeepsCoordinatorInvariantsFatal) {
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  failure.kind = wafer::compiler::detail::
      CoordinatedExecutableAdmissionFailureKind::RankFinalization;
  EXPECT_TRUE(wafer::compiler::detail::
                  isRecoverableCoordinatedExecutableSetupFailure(failure));
  failure.kind = wafer::compiler::detail::
      CoordinatedExecutableAdmissionFailureKind::SPMAllocation;
  EXPECT_TRUE(wafer::compiler::detail::
                  isRecoverableCoordinatedExecutableSetupFailure(failure));
  failure.kind = wafer::compiler::detail::
      CoordinatedExecutableAdmissionFailureKind::RankDomain;
  EXPECT_FALSE(wafer::compiler::detail::
                   isRecoverableCoordinatedExecutableSetupFailure(failure));
  failure.kind = wafer::compiler::detail::
      CoordinatedExecutableAdmissionFailureKind::WorkLedger;
  EXPECT_FALSE(wafer::compiler::detail::
                   isRecoverableCoordinatedExecutableSetupFailure(failure));
}

TEST_F(CoordinatedExecutableFinalizationTest,
       PublishesReadyOrderOnlyAsACompleteRankAction) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(makeReadyOrderInstrProgram());
  parents.push_back(makeReadyOrderInstrProgram());
  ASSERT_TRUE(parents[0] && parents[1]);
  wafer::OptimizationConfig config = wafer::OptimizationConfig::production();
  uint64_t work = 0;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(parents), config, &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          ReadyOrderQualification);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(actions->size(), 2u);
  EXPECT_EQ(actions->front().stableOrdinal, 0u);
  EXPECT_EQ(actions->front().readyOrderKind,
            wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical);
  EXPECT_EQ(actions->back().stableOrdinal, 1u);
  EXPECT_EQ(actions->back().readyOrderKind,
            wafer::compiler::detail::CoordinatedReadyOrderKind::ReadyOrder);
  EXPECT_EQ(actions->back().rankModules.size(), 2u);
  EXPECT_EQ(work, 4u);
  for (const auto &action : *actions) {
    EXPECT_EQ(action.rankModules.size(), 2u);
    for (const auto &module : action.rankModules)
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }

  // A rank without a local move still receives an identity sibling under the
  // same all-rank action; it never disappears into a partial rank frontier.
  auto immovable = makeWorkerInstrProgram();
  ASSERT_TRUE(immovable);
  llvm::SmallVector<mlir::ModuleOp, 2> mismatched = {*parents.front(),
                                                     *immovable};
  actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      mismatched, config, &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          ReadyOrderQualification);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(actions->size(), 2u);
  EXPECT_EQ(actions->back().rankModules.size(), 2u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       MaterializesFixedSlotAndWorkerActionsWithoutARankProduct) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> fixedParents;
  fixedParents.push_back(makeFixedSlotInstrProgram());
  fixedParents.push_back(makeFixedSlotInstrProgram());
  ASSERT_TRUE(fixedParents[0] && fixedParents[1]);
  wafer::OptimizationConfig fixedConfig =
      wafer::OptimizationConfig::production();
  uint64_t fixedWork = 0;
  auto fixedActions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(fixedParents), fixedConfig, &fixedWork,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          StaticFixedSlotQualification);
  ASSERT_TRUE(mlir::succeeded(fixedActions));
  auto fixed = llvm::find_if(*fixedActions, [](const auto &action) {
    return action.bufferingKind ==
           wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot;
  });
  ASSERT_NE(fixed, fixedActions->end());
  EXPECT_EQ(fixed->bufferingPlanOrdinal, 1u);
  EXPECT_EQ(fixed->rankModules.size(), 2u);
  EXPECT_LE(fixedActions->size(),
            wafer::compiler::detail::kMaximumCoordinatedScheduleActions);

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> workerParents;
  workerParents.push_back(makeWorkerInstrProgram());
  workerParents.push_back(makeWorkerInstrProgram());
  ASSERT_TRUE(workerParents[0] && workerParents[1]);
  wafer::OptimizationConfig workerConfig =
      wafer::OptimizationConfig::production();
  uint64_t workerWork = 0;
  auto workerActions =
      wafer::compiler::detail::deriveCoordinatedScheduleActions(
          views(workerParents), workerConfig, &workerWork,
          wafer::compiler::detail::CoordinatedScheduleActionFamily::
              DisjointWorkerPlacementQualification);
  ASSERT_TRUE(mlir::succeeded(workerActions));
  auto placed = llvm::find_if(*workerActions, [](const auto &action) {
    return action.workerPlacementKind ==
           wafer::compiler::detail::CoordinatedWorkerPlacementKind::
               DisjointComponents;
  });
  ASSERT_NE(placed, workerActions->end());
  EXPECT_EQ(placed->workerPlacementPlanOrdinal, 1u);
  EXPECT_EQ(placed->rankModules.size(), 2u);
  EXPECT_EQ(workerActions->size(), 2u);
  EXPECT_EQ(workerWork, 4u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       FixedSlotCorrespondenceRequiresExactPathBoundsAndStep) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(makeFixedSlotInstrProgram(/*upperBound=*/3));
  parents.push_back(makeFixedSlotInstrProgram(/*upperBound=*/4));
  ASSERT_TRUE(parents[0] && parents[1]);

  wafer::OptimizationConfig config = wafer::OptimizationConfig::production();
  uint64_t work = 0;
  wafer::compiler::detail::CoordinatedScheduleRecipeStatistics statistics;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(parents), config, &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          StaticFixedSlotQualification,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(actions->size(), 1u);
  EXPECT_EQ(actions->front().stableOrdinal, 0u);
  EXPECT_EQ(actions->front().bufferingKind,
            wafer::compiler::detail::CoordinatedBufferingKind::Single);
  EXPECT_EQ(statistics.enumeratedRecipes, 1u);
  EXPECT_EQ(statistics.materializationAttempts, 1u);
  EXPECT_EQ(work, parents.size());
}

TEST_F(CoordinatedExecutableFinalizationTest,
       MaterializationFailureBackfillsWithoutCloningTheRecipeDomain) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/false));
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/false));
  ASSERT_TRUE(parents[0] && parents[1]);

  wafer::OptimizationConfig config = wafer::OptimizationConfig::production();
  uint64_t work = 0;
  wafer::compiler::detail::CoordinatedScheduleRecipeStatistics statistics;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(parents), config, &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          StaticFixedSlotQualification,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(statistics.enumeratedRecipes, 9u);
  ASSERT_EQ(statistics.coverageRetainedRecipes,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_EQ(statistics.materializationAttempts, 9u);
  EXPECT_EQ(statistics.materializationFailures, 1u);
  EXPECT_EQ(statistics.materializationBackfills, 1u);
  EXPECT_EQ(statistics.successfulActionClones,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_EQ(statistics.actualRankClones,
            statistics.materializationAttempts * parents.size());
  EXPECT_EQ(work, statistics.materializationAttempts * parents.size());
  EXPECT_LE(statistics.materializationAttempts,
            wafer::compiler::detail::
                kMaximumCoordinatedScheduleRecipeMaterializationAttempts);
  ASSERT_EQ(actions->size(),
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_FALSE(llvm::any_of(
      *actions, [](const auto &action) { return action.stableOrdinal == 1; }));
  EXPECT_TRUE(llvm::any_of(
      *actions, [](const auto &action) { return action.stableOrdinal == 8; }));
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ExactConsumerRejectionTriggersTheSameDeterministicBackfill) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/true));
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/true));
  ASSERT_TRUE(parents[0] && parents[1]);

  wafer::OptimizationConfig config = wafer::OptimizationConfig::production();
  uint64_t work = 0;
  wafer::compiler::detail::CoordinatedScheduleRecipeStatistics statistics;
  llvm::SmallVector<uint32_t, 8> acceptedOrdinals;
  mlir::LogicalResult result =
      wafer::compiler::detail::walkCoordinatedScheduleActions(
          views(parents), config,
          [&](wafer::compiler::detail::CoordinatedScheduleAction &&action)
              -> mlir::FailureOr<wafer::compiler::detail::
                                     CoordinatedScheduleActionConsumption> {
            if (action.stableOrdinal == 1)
              return wafer::compiler::detail::
                  CoordinatedScheduleActionConsumption::RecoverableRejection;
            acceptedOrdinals.push_back(action.stableOrdinal);
            return wafer::compiler::detail::
                CoordinatedScheduleActionConsumption::Accepted;
          },
          &work, /*baselineOnly=*/false,
          wafer::compiler::detail::CoordinatedScheduleActionFamily::
              StaticFixedSlotQualification,
          &statistics);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(statistics.enumeratedRecipes, 9u);
  EXPECT_EQ(statistics.coverageRetainedRecipes,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_EQ(statistics.materializationAttempts, 9u);
  EXPECT_EQ(statistics.materializationFailures, 0u);
  EXPECT_EQ(statistics.materializationBackfills, 1u);
  EXPECT_EQ(statistics.successfulActionClones,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_EQ(statistics.actualRankClones,
            statistics.materializationAttempts * parents.size());
  EXPECT_EQ(work, statistics.materializationAttempts * parents.size());
  EXPECT_EQ(acceptedOrdinals.size(),
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_FALSE(llvm::is_contained(acceptedOrdinals, 1u));
  EXPECT_TRUE(llvm::is_contained(acceptedOrdinals, 8u));
  EXPECT_LE(statistics.materializationAttempts,
            wafer::compiler::detail::
                kMaximumCoordinatedScheduleRecipeMaterializationAttempts);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       CommunicationPointsComposeWithCommonRecipesBeforeBoundedCloning) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/true));
  parents.push_back(
      makeManyFixedSlotInstrProgram(/*firstLoopMaterializes=*/true));
  ASSERT_TRUE(parents[0] && parents[1]);

  MarkerCommunicationProvider provider;
  llvm::SmallVector<
      const wafer::compiler::detail::CoordinatedCommunicationActionProvider *,
      1>
      providers{&provider};
  wafer::frontend::FrontendProgramVerificationResult program =
      replicatedProgram(/*rankCount=*/2, /*elements=*/4);
  wafer::compiler::detail::CoordinatedScheduleRecipeStatistics statistics;
  uint64_t work = 0;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(parents), wafer::OptimizationConfig::production(), &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::Production,
      &statistics, &program, providers);
  ASSERT_TRUE(mlir::succeeded(actions));

  bool sawCommunication = false;
  bool sawCommunicationFixedSlot = false;
  for (const auto &action : *actions) {
    if (!action.communicationPointIdentity)
      continue;
    sawCommunication = true;
    EXPECT_EQ(action.communicationPointIdentity->providerKey,
              "test-communication-action");
    EXPECT_EQ(action.communicationPointIdentity->stableOrdinal, 7u);
    sawCommunicationFixedSlot |=
        action.bufferingKind ==
        wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot;
    for (const auto &module : action.rankModules)
      EXPECT_TRUE(module.get()->hasAttr("wafer.test_communication_marker"));
  }
  EXPECT_TRUE(sawCommunication);
  EXPECT_TRUE(sawCommunicationFixedSlot);
  for (const auto &parent : parents)
    EXPECT_FALSE(parent.get()->hasAttr("wafer.test_communication_marker"));

  EXPECT_GT(statistics.enumeratedRecipes, statistics.materializationAttempts);
  EXPECT_LE(statistics.materializationAttempts,
            wafer::compiler::detail::
                kMaximumCoordinatedScheduleRecipeMaterializationAttempts);
  EXPECT_LE(statistics.successfulActionClones,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_EQ(statistics.actualRankClones,
            statistics.materializationAttempts * parents.size());
  EXPECT_EQ(work, statistics.actualRankClones);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       SerializedDirectDTEComputeIsATypedPreAdmissionActionSibling) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(makeFixedSlotDirectDTEComputeInstrProgram());
  parents.push_back(makeFixedSlotDirectDTEComputeInstrProgram());
  ASSERT_TRUE(parents[0] && parents[1]);
  wafer::OptimizationConfig config = wafer::OptimizationConfig::production();
  uint64_t work = 0;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      views(parents), config, &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::
          ProductionWithSerializedDirectDTECompute);
  ASSERT_TRUE(mlir::succeeded(actions));
  auto serialized = llvm::find_if(*actions, [](const auto &action) {
    return action.serializationKind ==
           wafer::compiler::detail::CoordinatedScheduleSerializationKind::
               DirectDTEComputeWindows;
  });
  std::string actionSummary;
  llvm::raw_string_ostream actionStream(actionSummary);
  for (const auto &action : *actions)
    actionStream << "ordinal=" << action.stableOrdinal
                 << " buffering=" << static_cast<unsigned>(action.bufferingKind)
                 << " serialization="
                 << static_cast<unsigned>(action.serializationKind) << '\n';
  ASSERT_NE(serialized, actions->end()) << actionSummary;
  EXPECT_EQ(serialized->bufferingKind,
            wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot);
  EXPECT_GT(serialized->bufferingPlanOrdinal, 0u);
  EXPECT_EQ(serialized->workerPlacementKind,
            wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced);
  ASSERT_EQ(serialized->rankModules.size(), 2u);
  for (auto &module : serialized->rankModules) {
    bool hasSerializedWindow = false;
    module->walk([&](mlir::Operation *issue) {
      mlir::Value token;
      if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(issue))
        token = send.getToken();
      else if (auto recv = mlir::dyn_cast<wafer::InstrDTERecvOp>(issue))
        token = recv.getToken();
      else
        return;
      if (!token.hasOneUse())
        return;
      auto wait =
          mlir::dyn_cast<wafer::InstrDTEWaitOp>(*token.getUsers().begin());
      if (!wait || wait->getBlock() != issue->getBlock())
        return;
      bool computeBetween = false;
      for (mlir::Operation *between = issue->getNextNode();
           between && between != wait.getOperation();
           between = between->getNextNode()) {
        auto instruction =
            mlir::dyn_cast<wafer::WaferInstructionOpInterface>(between);
        computeBetween |=
            instruction &&
            (instruction.getInstructionFamily() == wafer::InstrFamily::CT ||
             instruction.getInstructionFamily() == wafer::InstrFamily::NE);
      }
      hasSerializedWindow |= !computeBetween;
    });
    EXPECT_TRUE(hasSerializedWindow);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
  EXPECT_GT(work, parents.size() * 2u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       EstimateBeamRetainsBaselineAndEveryTypedActionShapeDeterministically) {
  using Dimension =
      wafer::compiler::detail::CoordinatedScheduleActionEstimateDimension;
  using Estimate = wafer::compiler::detail::CoordinatedScheduleActionEstimate;
  auto makeEstimate =
      [](uint32_t ordinal,
         wafer::compiler::detail::CoordinatedReadyOrderKind artifact,
         wafer::compiler::detail::CoordinatedBufferingKind buffering,
         wafer::compiler::detail::CoordinatedWorkerPlacementKind placement,
         uint64_t overlap, uint64_t instructionSites) {
        Estimate estimate;
        estimate.stableOrdinal = ordinal;
        estimate.readyOrderKind = artifact;
        estimate.bufferingKind = buffering;
        estimate.workerPlacementKind = placement;
        estimate
            .values[static_cast<size_t>(Dimension::QualifiedOverlapWindows)] =
            overlap;
        estimate.values[static_cast<size_t>(Dimension::InstructionSites)] =
            instructionSites;
        return estimate;
      };

  std::vector<Estimate> estimates;
  estimates.push_back(makeEstimate(
      0, wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical,
      wafer::compiler::detail::CoordinatedBufferingKind::Single,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced, 0,
      10));
  // Two fixed-slot loop identities occupy the same typed action shape. The
  // versioned estimate prefers the larger qualified overlap value.
  estimates.push_back(makeEstimate(
      1, wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical,
      wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced, 1,
      10));
  estimates.push_back(makeEstimate(
      2, wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical,
      wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced, 2,
      10));
  estimates.push_back(makeEstimate(
      3, wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical,
      wafer::compiler::detail::CoordinatedBufferingKind::Single,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::
          DisjointComponents,
      0, 9));
  estimates.push_back(makeEstimate(
      4, wafer::compiler::detail::CoordinatedReadyOrderKind::Canonical,
      wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::
          DisjointComponents,
      2, 9));
  estimates.push_back(makeEstimate(
      5, wafer::compiler::detail::CoordinatedReadyOrderKind::ReadyOrder,
      wafer::compiler::detail::CoordinatedBufferingKind::Single,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced, 0,
      10));
  estimates.push_back(makeEstimate(
      6, wafer::compiler::detail::CoordinatedReadyOrderKind::ReadyOrder,
      wafer::compiler::detail::CoordinatedBufferingKind::Single,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::
          DisjointComponents,
      0, 9));
  estimates.push_back(makeEstimate(
      7, wafer::compiler::detail::CoordinatedReadyOrderKind::ReadyOrder,
      wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::Unplaced, 2,
      10));
  estimates.push_back(makeEstimate(
      8, wafer::compiler::detail::CoordinatedReadyOrderKind::ReadyOrder,
      wafer::compiler::detail::CoordinatedBufferingKind::StaticFixedSlot,
      wafer::compiler::detail::CoordinatedWorkerPlacementKind::
          DisjointComponents,
      2, 9));

  auto first =
      wafer::compiler::detail::selectCoordinatedScheduleActionEstimateBeam(
          estimates);
  ASSERT_TRUE(mlir::succeeded(first));
  EXPECT_EQ(first->size(),
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_TRUE(llvm::is_contained(*first, 0u));
  EXPECT_TRUE(llvm::is_contained(*first, 2u));
  EXPECT_FALSE(llvm::is_contained(*first, 1u));

  std::reverse(estimates.begin(), estimates.end());
  auto replay =
      wafer::compiler::detail::selectCoordinatedScheduleActionEstimateBeam(
          estimates);
  ASSERT_TRUE(mlir::succeeded(replay));
  EXPECT_EQ(*first, *replay);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       LowersEachTileParentOnceAcrossMultipleExactScheduleActions) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.finalizationReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(
      0, std::move(tile),
      std::make_shared<const std::string>("ready-order Tile parent"));
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto workSession =
      std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation workActivation(
      workSession);
  auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
      variant, replicatedProgram(rankCount), *executionConfig,
      wafer::OptimizationConfig::production(), *ledger, diagnostics, failure,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(gated)) << diagnosticsText;
  ASSERT_FALSE(gated->empty());
  EXPECT_TRUE(llvm::any_of(*gated, [](const auto &survivor) {
    return survivor.reservedBaseline;
  }));
  llvm::SmallSet<uint32_t, 8> actionOrdinals;
  for (const auto &survivor : *gated)
    EXPECT_TRUE(actionOrdinals.insert(survivor.scheduleActionOrdinal).second);

  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(
      snapshot.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering)],
      1u);
  EXPECT_GT(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    ExecutableScheduleAction)],
            1u);
  EXPECT_GT(statistics.scheduleEstimatedActions, 0u);
  // Exact action attempts are deliberately distinct from the retained
  // production frontier: equivalent-cost actions may be deduplicated after
  // all common exact gates have accepted them.
  EXPECT_GT(statistics.scheduleMaterializationAttempts, 1u);
  EXPECT_GT(statistics.scheduleSuccessfulActionClones, 1u);
  EXPECT_GT(statistics.scheduleExactActions, 1u);
  EXPECT_GT(statistics.scheduleExpansionAttempts, 0u);
  EXPECT_GE(statistics.scheduleExactActions, gated->size());
  EXPECT_LE(statistics.scheduleExactActions,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  const wafer::support::CompileWorkStatistics work = workSession->snapshot();
  EXPECT_EQ(work.finalizationInstructionLowerings, 1u);
  EXPECT_GT(work.finalizationCandidateClones, 1u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       CursorReusesOneCanonicalParentForSeedAndExpansion) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.finalizationReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(0, std::move(tile), nullptr);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto cursor =
      wafer::compiler::detail::beginCoordinatedExecutableFinalization(
          variant, program, *executionConfig,
          wafer::OptimizationConfig::production(), *ledger, diagnostics,
          failure, &statistics);
  ASSERT_TRUE(mlir::succeeded(cursor)) << diagnosticsText;
  EXPECT_FALSE((*cursor)->seedAttempted());
  EXPECT_FALSE((*cursor)->exactSeedAccepted());
  EXPECT_FALSE((*cursor)->exhausted());
  EXPECT_EQ((*cursor)->getCanonicalParentCount(), 1u);

  auto seed = wafer::compiler::detail::advanceCoordinatedExecutableFinalization(
      **cursor, variant.finalizationReservation);
  ASSERT_TRUE(mlir::succeeded(seed)) << diagnosticsText;
  ASSERT_EQ(seed->kind, wafer::compiler::detail::
                            CoordinatedExecutableFinalizationStepKind::Accepted)
      << diagnosticsText;
  ASSERT_TRUE(seed->admitted);
  EXPECT_EQ(seed->admitted->scheduleActionOrdinal, 0u);
  EXPECT_TRUE((*cursor)->seedAttempted());
  EXPECT_TRUE((*cursor)->exactSeedAccepted());
  EXPECT_FALSE((*cursor)->exhausted());

  auto reservation = ledger->tryReserveExecutableScheduleAttempt();
  ASSERT_TRUE(reservation);
  auto expansion =
      wafer::compiler::detail::advanceCoordinatedExecutableFinalization(
          **cursor, *reservation);
  ASSERT_TRUE(mlir::succeeded(expansion)) << diagnosticsText;
  EXPECT_NE(expansion->kind,
            wafer::compiler::detail::
                CoordinatedExecutableFinalizationStepKind::Exhausted);

  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.scheduleAttemptsConsumed, 2u);
  EXPECT_EQ(snapshot.scheduleAttemptsReserved, 0u);
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    TileToInstrLowering)],
            1u);
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    ExecutableScheduleAction)],
            2u);
  EXPECT_EQ(statistics.scheduleSeedAttempts, 1u);
  EXPECT_EQ(statistics.scheduleExpansionAttempts, 1u);
  EXPECT_EQ(statistics.peakLiveScheduleActionClones, 1u);
  EXPECT_TRUE(wafer::containsTileDataflowOperations(
      variant.ranks.front().module.get().getOperation()));
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ExactRejectedCanonicalSeedCannotEnterExpansionLane) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.finalizationReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(0, std::move(tile), nullptr);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  // The canonical action can materialize, but this deliberately inconsistent
  // frontend rank contract must fail one of the common exact gates.
  auto inconsistentProgram = replicatedProgram(/*rankCount=*/2);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  auto cursor =
      wafer::compiler::detail::beginCoordinatedExecutableFinalization(
          variant, inconsistentProgram, *executionConfig,
          wafer::OptimizationConfig::production(), *ledger, diagnostics,
          failure);
  ASSERT_TRUE(mlir::succeeded(cursor)) << diagnosticsText;
  auto seed = wafer::compiler::detail::advanceCoordinatedExecutableFinalization(
      **cursor, variant.finalizationReservation);
  ASSERT_TRUE(mlir::succeeded(seed)) << diagnosticsText;
  EXPECT_EQ(seed->kind,
            wafer::compiler::detail::
                CoordinatedExecutableFinalizationStepKind::RecoverableRejected);
  EXPECT_TRUE((*cursor)->seedAttempted());
  EXPECT_FALSE((*cursor)->exactSeedAccepted());
  EXPECT_TRUE((*cursor)->exhausted());
  EXPECT_EQ(ledger->getSnapshot().scheduleAttemptsConsumed, 1u);
  EXPECT_EQ(ledger->getSnapshot().scheduleAttemptsReserved, 0u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ReservedModeMaterializesOnlyTheMandatoryBaselineAction) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.finalizationReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(0, std::move(tile), nullptr);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
      variant, replicatedProgram(rankCount), *executionConfig,
      wafer::OptimizationConfig::production(), *ledger, diagnostics, failure,
      /*statistics=*/nullptr,
      wafer::compiler::detail::WholeVariantSelectionMode::ReservedBaseline);
  ASSERT_TRUE(mlir::succeeded(gated)) << diagnosticsText;
  ASSERT_EQ(gated->size(), 1u);
  EXPECT_TRUE(gated->front().reservedBaseline);
  EXPECT_EQ(gated->front().scheduleActionOrdinal, 0u);
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    ExecutableScheduleAction)],
            1u);
  EXPECT_EQ(snapshot.scheduleAttemptsConsumed, 1u);
  EXPECT_EQ(snapshot.scheduleAttemptsReserved, 0u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ExternalBaselineReleasesEveryPermanentlyIneligibleAction) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));
  auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableFinalizationUpperBound(rankCount);
  ASSERT_TRUE(mlir::succeeded(finalization));
  auto reservation = ledger->tryReserveExecutableFinalization(*finalization);
  ASSERT_TRUE(reservation);

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.stableSemanticOrdinal = 1;
  variant.finalizationReservation = *reservation;
  variant.ranks.emplace_back(0, std::move(tile), nullptr);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  wafer::analysis::WholeCardInstructionProgramCost zeroBaseline;
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
      variant, replicatedProgram(rankCount), *executionConfig,
      wafer::OptimizationConfig::production(), *ledger, diagnostics, failure,
      &statistics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production,
      &zeroBaseline);
  EXPECT_TRUE(mlir::failed(gated));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::CoordinatedExecutableAdmissionFailureKind::
                WholeVariantExactGate);
  EXPECT_EQ(failure.gate, "no-admitted-executable");
  EXPECT_GT(statistics.scheduleExactActions, 0u);
  EXPECT_LE(statistics.scheduleExactActions,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  EXPECT_TRUE(wafer::containsTileDataflowOperations(
      variant.ranks.front().module.get().getOperation()));
}

TEST_F(CoordinatedExecutableFinalizationTest,
       FullyGatesAllRanksWithoutMutatingTileParents) {
  constexpr int64_t rankCount = 16;
  auto source = makeProgram(rankCount);
  ASSERT_TRUE(source);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig searchConfig;
  searchConfig.rankCount = rankCount;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, searchConfig, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  for (const auto &tileParent : *frontier) {
    ASSERT_TRUE(wafer::containsTileDataflowOperations(
        tileParent.ranks.front().module.get().getOperation()));
    wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
    auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
        tileParent, replicatedProgram(rankCount), *executionConfig,
        searchConfig.optimizations, *ledger, diagnostics, failure);
    ASSERT_TRUE(mlir::succeeded(gated))
        << "semantic ordinal " << tileParent.stableSemanticOrdinal << "\n"
        << diagnosticsText;
    EXPECT_EQ(failure.kind,
              wafer::compiler::detail::
                  CoordinatedExecutableAdmissionFailureKind::None);
    ASSERT_FALSE(gated->empty());
    for (const auto &survivor : *gated) {
      ASSERT_EQ(survivor.variant.ranks.size(), rankCount);
      for (const wafer::compiler::RankExecutable &rank :
           survivor.variant.ranks) {
        EXPECT_FALSE(wafer::containsTileDataflowOperations(
            rank.getModule().getOperation()));
        bool hasInstruction = false;
        rank.getModule().walk(
            [&](wafer::WaferInstructionOpInterface) { hasInstruction = true; });
        EXPECT_TRUE(hasInstruction);
      }
    }
    EXPECT_TRUE(wafer::containsTileDataflowOperations(
        tileParent.ranks.front().module.get().getOperation()));
  }
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.finalizationReserved, 0u);
  EXPECT_EQ(
      snapshot.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering)],
      rankCount * frontier->size());
  EXPECT_GE(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::ABIValidation)],
            rankCount * frontier->size());
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ExactAdmittedFrontierDigestIsIndependentOfRankParallelism) {
  constexpr int64_t rankCount = 16;
  auto serialSource = makeChainProgram(rankCount);
  auto parallelSource = makeChainProgram(rankCount);
  ASSERT_TRUE(serialSource);
  ASSERT_TRUE(parallelSource);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount);

  struct Result {
    std::string admittedDigest;
    std::string winnerDigest;
    uint64_t admittedCount = 0;
    uint64_t maximumWorkers = 1;
  };
  auto run = [&](mlir::ModuleOp source,
                 unsigned parallelism) -> std::optional<Result> {
    auto ledger =
        wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
    if (mlir::failed(ledger))
      return std::nullopt;
    wafer::compiler::detail::CoordinatedDataflowSearchConfig searchConfig;
    searchConfig.rankCount = rankCount;
    searchConfig.candidateParallelism = parallelism;
    auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
        source, searchConfig, *ledger);
    if (mlir::failed(frontier))
      return std::nullopt;

    std::vector<wafer::compiler::detail::AdmittedCoordinatedExecutable>
        admitted;
    wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    for (const auto &tileParent : *frontier) {
      const wafer::analysis::WholeCardInstructionProgramCost *baselineCost =
          nullptr;
      auto baseline = llvm::find_if(admitted, [](const auto &candidate) {
        return candidate.reservedBaseline;
      });
      if (baseline != admitted.end())
        baselineCost = &baseline->variant.resourceCost;
      wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
      auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
          tileParent, program, *executionConfig, searchConfig.optimizations,
          *ledger, diagnostics, failure, &statistics,
          wafer::compiler::detail::WholeVariantSelectionMode::Production,
          baselineCost, parallelism);
      if (mlir::failed(gated)) {
        if (tileParent.reservedBaseline) {
          ADD_FAILURE() << diagnosticText;
          return std::nullopt;
        }
        continue;
      }
      for (auto &executable : *gated)
        admitted.push_back(std::move(executable));
    }
    if (llvm::count_if(admitted, [](const auto &candidate) {
          return candidate.reservedBaseline;
        }) != 1)
      return std::nullopt;
    const auto beforeSelection = ledger->getSnapshot();
    if (beforeSelection.finalizationReserved != 0)
      return std::nullopt;
    Result result;
    result.admittedDigest = wafer::compiler::detail::
        computeAdmittedCoordinatedExecutableFrontierDigest(admitted);
    result.admittedCount = admitted.size();
    result.maximumWorkers = statistics.maximumRankPipelineWorkers;
    auto winner = wafer::compiler::detail::selectAdmittedCoordinatedExecutable(
        std::move(admitted),
        wafer::compiler::detail::WholeVariantSelectionMode::Production,
        diagnostics, &statistics);
    if (mlir::failed(winner))
      return std::nullopt;
    const auto afterSelection = ledger->getSnapshot();
    if (beforeSelection.capacity != afterSelection.capacity ||
        beforeSelection.consumed != afterSelection.consumed ||
        beforeSelection.mandatoryGenerationReserved !=
            afterSelection.mandatoryGenerationReserved ||
        beforeSelection.finalizationReserved !=
            afterSelection.finalizationReserved ||
        beforeSelection.repairReserved != afterSelection.repairReserved ||
        beforeSelection.scheduleAttemptCapacity !=
            afterSelection.scheduleAttemptCapacity ||
        beforeSelection.scheduleAttemptsReserved !=
            afterSelection.scheduleAttemptsReserved ||
        beforeSelection.scheduleAttemptsConsumed !=
            afterSelection.scheduleAttemptsConsumed ||
        beforeSelection.unreserved != afterSelection.unreserved ||
        beforeSelection.consumedByKind != afterSelection.consumedByKind)
      return std::nullopt;
    llvm::ArrayRef<wafer::compiler::detail::AdmittedCoordinatedExecutable>
        winnerView(&*winner, 1);
    result.winnerDigest = wafer::compiler::detail::
        computeAdmittedCoordinatedExecutableFrontierDigest(winnerView);
    return result;
  };

  std::optional<Result> serial = run(*serialSource, /*parallelism=*/1);
  std::optional<Result> parallel = run(*parallelSource, /*parallelism=*/8);
  ASSERT_TRUE(serial);
  ASSERT_TRUE(parallel);
  EXPECT_EQ(serial->maximumWorkers, 1u);
  EXPECT_GT(parallel->maximumWorkers, 1u);
  EXPECT_EQ(serial->admittedCount, parallel->admittedCount);
  EXPECT_EQ(serial->admittedDigest, parallel->admittedDigest);
  EXPECT_EQ(serial->winnerDigest, parallel->winnerDigest);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       PartialReductionCollectiveSiblingPassesTheSameExactAllRankGate) {
  constexpr int64_t rankCount = 16;
  auto source = makeLocalReductionAllReduceProgram(rankCount);
  ASSERT_TRUE(source);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig searchConfig;
  searchConfig.rankCount = rankCount;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, searchConfig, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());

  bool foundAdmittedPartial = false;
  unsigned partialTileVariants = 0;
  unsigned admittedPartialExecutables = 0;
  unsigned admittedPartialRanksWithWait = 0;
  std::string lastPartialFailureGate;
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  for (const auto &variant : *frontier) {
    unsigned tileReductions = 0;
    variant.ranks.front().module.get().walk(
        [&](wafer::ComputeReduceOp) { ++tileReductions; });
    partialTileVariants += tileReductions > 1;
    wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
    auto gated = wafer::compiler::detail::finalizeCoordinatedTileVariant(
        variant, replicatedReductionProgram(rankCount), *executionConfig,
        searchConfig.optimizations, *ledger, diagnostics, failure,
        /*statistics=*/nullptr,
        wafer::compiler::detail::WholeVariantSelectionMode::Production,
        /*productionBaselineCost=*/nullptr);
    if (mlir::failed(gated)) {
      if (tileReductions > 1)
        lastPartialFailureGate = failure.gate;
      ASSERT_FALSE(variant.reservedBaseline) << diagnosticsText;
      continue;
    }
    for (auto &candidate : *gated) {
      // Partial-reduction semantics are explicit in the immutable Tile parent.
      // Instruction lowering may legally realize each local reduction as an
      // elementwise tree, so do not recover the upper-level role from a
      // particular low-level instruction family.
      bool completePartial = tileReductions > 1;
      admittedPartialExecutables += tileReductions > 1;
      for (const wafer::compiler::RankExecutable &rank :
           candidate.variant.ranks) {
        unsigned waits = 0;
        rank.getModule().walk([&](wafer::InstrDTEWaitOp) { ++waits; });
        admittedPartialRanksWithWait += tileReductions > 1 && waits > 0;
        completePartial &= waits > 0;
      }
      foundAdmittedPartial |= completePartial;
    }
  }
  EXPECT_TRUE(foundAdmittedPartial)
      << "partial Tile variants: " << partialTileVariants
      << ", admitted partial executables: " << admittedPartialExecutables
      << ", admitted partial ranks with DTE wait: "
      << admittedPartialRanksWithWait
      << ", last partial failure gate: " << lastPartialFailureGate << '\n'
      << diagnosticsText;
  EXPECT_EQ(ledger->getSnapshot().finalizationReserved, 0u);
}

TEST_F(CoordinatedExecutableFinalizationTest,
       ExactOversizedSPMPreflightPreservesTileParentAndClosesAttempt) {
  constexpr int64_t rankCount = 16;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 rankCount);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  auto selectedTileIR = std::make_shared<const std::string>("actual Tile");
  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.finalizationReservation = ledger->getMandatoryBaselineReservation();
  for (int64_t rank = 0; rank < rankCount; ++rank) {
    auto module = makeOversizedTileProgram(/*elements=*/2'000'000);
    ASSERT_TRUE(module);
    variant.ranks.emplace_back(rank, std::move(module), selectedTileIR);
  }
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount);
  wafer::compiler::detail::CoordinatedExecutableAdmissionFailure failure;
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto workSession =
      std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation workActivation(
      workSession);
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::finalizeCoordinatedTileVariant(
          variant, program, *executionConfig,
          wafer::OptimizationConfig::production(), *ledger, diagnostics,
          failure)));
  diagnostics.flush();
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::CoordinatedExecutableAdmissionFailureKind::
                SPMAllocation)
      << diagnosticsText;
  EXPECT_EQ(failure.gate, "individual-spm-capacity");
  EXPECT_TRUE(wafer::containsTileDataflowOperations(
      variant.ranks.front().module.get().getOperation()));
  bool parentHasPhysicalOffset = false;
  variant.ranks.front().module.get().walk(
      [&](mlir::memref::AllocOp allocation) {
        parentHasPhysicalOffset |=
            allocation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
            allocation->hasAttr(wafer::kWaferDDROffsetAttrName);
      });
  EXPECT_FALSE(parentHasPhysicalOffset);
  const auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.finalizationReserved, 0u);
  EXPECT_EQ(snapshot.scheduleAttemptsConsumed, 0u);
  EXPECT_EQ(snapshot.scheduleAttemptsReserved, 0u);
  EXPECT_EQ(
      snapshot.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering)],
      0u);
  EXPECT_EQ(
      snapshot.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::SPMAllocationProblem)],
      rankCount);
  const wafer::support::CompileWorkStatistics work = workSession->snapshot();
  EXPECT_EQ(work.finalizationInstructionLowerings, 0u);
  EXPECT_EQ(work.spmPlanningInvocations, 0u);
}

} // namespace
