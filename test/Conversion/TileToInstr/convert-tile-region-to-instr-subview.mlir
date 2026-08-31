// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

// This rank-4 case pairs a 1024-element divisible prefix with the tail of a
// 1025-element sequence.  The sizes are intentionally representative: the
// test must exercise blocked NCx physical addressing rather than a tiny
// row-major example.
func.func @copy_into_static_ncx_subviews(%token: i1) {
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%token_arg: i1):
    %base = memref.alloc()
        : memref<1x4x1025x128xf16, #wafer.memory<spm, ncx>>
    %prefix_source = memref.alloc()
        : memref<1x4x1024x128xf16, #wafer.memory<spm, ncx>>
    %prefix = memref.subview %base[0, 0, 0, 0] [1, 4, 1024, 128]
        [1, 1, 1, 1]
        : memref<1x4x1025x128xf16, #wafer.memory<spm, ncx>>
       to memref<1x4x1024x128xf16,
                    strided<[524800, 131200, 128, 1]>,
                    #wafer.memory<spm, ncx>>
    wafer.tile.copy_into %prefix_source into %prefix
        : memref<1x4x1024x128xf16, #wafer.memory<spm, ncx>>
       into memref<1x4x1024x128xf16,
                       strided<[524800, 131200, 128, 1]>,
                       #wafer.memory<spm, ncx>>

    %tail_source = memref.alloc()
        : memref<1x4x1x128xf16, #wafer.memory<spm, ncx>>
    %tail = memref.subview %base[0, 0, 1024, 0] [1, 4, 1, 128]
        [1, 1, 1, 1]
        : memref<1x4x1025x128xf16, #wafer.memory<spm, ncx>>
       to memref<1x4x1x128xf16,
                    strided<[524800, 131200, 128, 1], offset: 131072>,
                    #wafer.memory<spm, ncx>>
    wafer.tile.copy_into %tail_source into %tail
        : memref<1x4x1x128xf16, #wafer.memory<spm, ncx>>
       into memref<1x4x1x128xf16,
                       strided<[524800, 131200, 128, 1], offset: 131072>,
                       #wafer.memory<spm, ncx>>
    wafer.tile.yield %token_arg : i1
  }
  return
}

// CHECK-LABEL: func.func @copy_into_static_ncx_subviews
// CHECK-NOT: wafer.tile.copy_into
// CHECK: %[[BASE:.+]] = memref.alloc() : memref<1x4x1025x128xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.gather_scatter %{{.+}} to %[[BASE]] {byte_count = 1048576 : i64
// CHECK-SAME: dst_iterations = array<i64: 8, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 131200, 0, 0>
// CHECK-SAME: inner_bytes = 131072 : i64
// CHECK: wafer.instr.gather_scatter %{{.+}} to %[[BASE]] {byte_count = 1024 : i64
// CHECK-SAME: dst_iterations = array<i64: 8, 1, 1>
// CHECK-SAME: dst_offset = 131072 : i64
// CHECK-SAME: dst_strides = array<i64: 131200, 0, 0>
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.tile.yield

func.func @store_static_ncx_source_subview(
    %output: memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%output
      : memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>):
    %base = memref.alloc()
        : memref<1x2x256x128xf16, #wafer.memory<spm, ncx>>
    %tail = memref.subview %base[0, 0, 255, 0] [1, 2, 1, 128]
        [1, 1, 1, 1]
        : memref<1x2x256x128xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1x128xf16,
                    strided<[65536, 32768, 128, 1], offset: 32640>,
                    #wafer.memory<spm, ncx>>
    wafer.tile.store %tail, %out
        : memref<1x2x1x128xf16,
                     strided<[65536, 32768, 128, 1], offset: 32640>,
                     #wafer.memory<spm, ncx>>
       -> memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<1x2x1x128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @store_static_ncx_source_subview
// CHECK: %[[STORE_BASE:.+]] = memref.alloc() : memref<1x2x256x128xf16, #wafer.memory<spm, ncx>>
// CHECK-NOT: memref.subview
// CHECK: wafer.instr.wdma %[[STORE_BASE]] to %{{.+}}
// CHECK-NOT: memref.subview
