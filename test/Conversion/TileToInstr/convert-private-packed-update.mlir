// RUN: wafer-opt --wafer-lower-tile-region-to-instr %s | FileCheck %s
// RUN: wafer-opt --wafer-lower-tile-region-to-instr --wafer-plan-spm-memory --wafer-plan-ddr-memory %s | wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{default-ddr-arena-argument-index=0})' | FileCheck %s --check-prefix=TARGET

// CHECK-LABEL: func.func @update
// CHECK: memref.reinterpret_cast
// CHECK: wafer.instr.rdma
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: dst_offset = 8 : i64
// CHECK: wafer.instr.elementwise <ne>
// CHECK-SAME: rhs_unit_elements = 1 : i64
// CHECK: wafer.instr.wdma
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET-COUNT-2: llvm.call @wafer_tx81_bit2fp
// TARGET: llvm.call @wafer_tx81_gather_scatter
// TARGET: llvm.call @wafer_tx81_elementwise_ne
// TARGET: llvm.call @wafer_tx81_wdma

func.func @update(%arena: i64) {
  %root = memref.alloc() : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%root : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%dest: memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>):
    %view = memref.subview %dest[0, 7, 3] [1, 1024, 32] [1, 1, 1]
      : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
    %input = memref.alloc() : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.store %input, %view
      : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
      -> memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield
  }
  return
}
