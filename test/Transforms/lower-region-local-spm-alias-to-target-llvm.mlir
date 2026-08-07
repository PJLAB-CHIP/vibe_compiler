// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | wafer-opt --wafer-lower-instr-to-target-llvm | FileCheck --implicit-check-not=wafer.tile.region %s

module {
  func.func @tensor_layout_spm_alias_chain(
      %boundary: memref<1xf16, #wafer.memory<ddr, tensor>>) {
    %done = wafer.tile.region(
        %boundary : memref<1xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<1xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<1xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %produced = memref.alloc()
          : memref<1x2x4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %produced, %zero
          : memref<1x2x4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      %collapsed = memref.collapse_shape %produced [[0, 1], [2]]
          : memref<1x2x4xf16, #wafer.memory<spm, tensor>>
         into memref<2x4xf16, #wafer.memory<spm, tensor>>
      %cast = memref.cast %collapsed
          : memref<2x4xf16, #wafer.memory<spm, tensor>>
         to memref<2x4xf16, strided<[4, 1], offset: ?>,
                   #wafer.memory<spm, tensor>>
      %local = memref.alloc()
          : memref<2x4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.gather_scatter %cast to %local
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x4xf16, strided<[4, 1], offset: ?>,
                   #wafer.memory<spm, tensor>>
         to memref<2x4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %ddr
          : memref<1xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: llvm.func @tensor_layout_spm_alias_chain
// CHECK: %[[SOURCE:.+]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK: llvm.call @wafer_tx81_memset_v3(%[[SOURCE]],
// CHECK: %[[DESTINATION:.+]] = llvm.mlir.constant(65792 : i64) : i64
// CHECK: llvm.call @wafer_tx81_gather_scatter_v3
// CHECK-SAME: (%[[SOURCE]], %[[DESTINATION]],
// CHECK: llvm.call @wafer_tx81_ncc_join
// CHECK: llvm.return
