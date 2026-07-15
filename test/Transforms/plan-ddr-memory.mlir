// RUN: wafer-opt --wafer-plan-ddr-memory %s | FileCheck %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=64' %s 2>&1 | FileCheck --check-prefix=CAPACITY %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-bandwidth-limit-bytes=20' %s 2>&1 | FileCheck --check-prefix=BANDWIDTH %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=-1' %s 2>&1 | FileCheck --check-prefix=BAD-LIMIT %s

func.func @plan_external_strided_views(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %output_tile = memref.subview %output[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input_tile, %output_tile
      : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %loaded to %out
        {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 16, 0, 0>, inner_bytes = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    wafer.tile.yield %out
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @plan_external_strided_views
// CHECK-NOT: {{wafer[.]ddr[.]access}}
// CHECK: %[[INPUT_TILE:.+]] = memref.subview
// CHECK: %[[OUTPUT_TILE:.+]] = memref.subview
// CHECK: wafer.instr.rdma
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: wafer.instr.wdma
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 16, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64

// CAPACITY: memory_capacity_overflow
// BANDWIDTH: bandwidth_pressure_too_high
// BAD-LIMIT: invalid_ddr_resource_limit
