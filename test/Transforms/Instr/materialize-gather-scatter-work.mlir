// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %s 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-materialize-gather-scatter-work,wafer-rebuild-required-ncc-joins),wafer-plan-spm-memory)' %s | FileCheck %s --check-prefix=INSTR
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-materialize-gather-scatter-work,wafer-rebuild-required-ncc-joins),wafer-plan-spm-memory,wafer-lower-instr-to-target-llvm)' %s | FileCheck %s --check-prefix=LLVM

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @entry() {
    wafer.tile.region() -> () {
    %source = memref.alloc() : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() : memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 262400 : i64, inner_bytes = 4 : i64,
         src_strides = array<i64: 0, 4, 4100>, src_iterations = array<i64: 32, 1025, 2>,
         dst_strides = array<i64: 4, 128, 131200>, dst_iterations = array<i64: 32, 1025, 2>}
        : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>
       to memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %dest, %dest into %dest
        : memref<2x1025x32xf32, #wafer.memory<spm, tensor>>,
          memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
       into memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield
    }
    return
  }
}

// REJECT: unmaterialized_gather_scatter_work
// INSTR: scf.for
// INSTR: wafer.instr.gather_scatter
// INSTR-SAME: byte_count = 65536 : i64
// INSTR-NOT: wafer.instr.ncc_join
// INSTR: wafer.instr.gather_scatter
// INSTR-SAME: byte_count = 256 : i64
// INSTR: wafer.instr.elementwise <add>
// INSTR: wafer.instr.ncc_join [0]
// INSTR-NEXT: return
// LLVM: llvm.call @wafer_tx81_gather_scatter
// LLVM: llvm.call @wafer_tx81_elementwise_add
// LLVM: llvm.call @wafer_tx81_ncc_join
