// RUN: wafer-opt --wafer-place-spm-buffers %s | FileCheck %s
// RUN: not wafer-opt --wafer-place-spm-buffers='spm-base=65536 spm-limit=65792' %s 2>&1 | FileCheck --check-prefix=OVERFLOW %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @place_instruction_spm(%input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
                                 %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %arg1: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %arg0 to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 6, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>

    %elementwise = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.elementwise_kind<add> %loaded, %loaded into %elementwise
        {indexing_maps = [#map, #map, #map]}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>,
          memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>

    %cx = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, cx>>
    wafer.instr.wdma %elementwise to %arg1
        {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<ddr, tensor>>

    wafer.tile.yield %arg1 : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @place_instruction_spm
// CHECK: %[[LOADED:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65536, 12, 256, 256, 257>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma {{.*}} to %[[LOADED]]
// CHECK: %[[ELEMENTWISE:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65792, 12, 256, 257, 258>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.elementwise <add> %[[LOADED]], %[[LOADED]] into %[[ELEMENTWISE]]
// CHECK: %[[CX:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<66048, 256, 256, 258, 259>} : memref<4x8xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.wdma %[[ELEMENTWISE]]

// OVERFLOW: capacity_overflow
