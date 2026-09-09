// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr,wafer-lower-instr-to-target-llvm)' %s | FileCheck %s --check-prefix=TARGET

func.func @fill_dynamic_base(%out: memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%out : memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>):
    %all = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    %one = arith.constant 1.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    wafer.tile.fill %all, %zero : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>, f16
    scf.for %batch = %c0 to %c2 step %c1 {
      %slice = memref.subview %all[%batch, 0, 3] [1, 1025, 64] [1, 1, 1]
          : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
         to memref<1x1025x64xf16, strided<[131200, 128, 1], offset: ?>, #wafer.memory<spm, tensor>>
      wafer.tile.fill %slice, %one
          : memref<1x1025x64xf16, strided<[131200, 128, 1], offset: ?>, #wafer.memory<spm, tensor>>, f16
    }
    wafer.tile.store %all, %ddr : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
        -> memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %ddr : memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @fill_dynamic_base
// CHECK: wafer.instr.fill
// CHECK: scf.for
// CHECK: %[[ROWS:.*]] = arith.constant 1025 : index
// CHECK: scf.for %[[ROW:.*]] = {{.*}} to %[[ROWS]] step
// CHECK: %[[SEGMENT:.*]] = memref.subview {{.*}}[0, %[[ROW]], 0] [1, 1, 64] [1, 1, 1]
// CHECK: wafer.instr.fill %[[SEGMENT]]
// CHECK-NOT: wafer.instr.ncc_join
// CHECK: wafer.instr.wdma
// CHECK: wafer.instr.ncc_join [0]
// TARGET-LABEL: llvm.func @fill_dynamic_base
// TARGET: llvm.mlir.constant(64 : i32)
// TARGET: llvm.call @wafer_tx81_memset
// TARGET-NOT: wafer.tile.fill
