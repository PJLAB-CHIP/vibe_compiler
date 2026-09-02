// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @pack_large_identity_tensor_to_cx_mapping(
    %input: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input
      : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>):
    %cx = memref.alloc()
        : memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %arg0 into %cx
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
      into memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @pack_large_identity_tensor_to_cx_mapping
// CHECK-COUNT-1: wafer.instr.rdma
// CHECK-SAME: byte_count = 8388608 : i64
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-SAME: src_iterations = array<i64: 1024, 64, 1>
// CHECK-SAME: src_strides = array<i64: 8192, 128, 0>

func.func @unpack_large_cx_to_tensor_mapping(
    %input: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>,
        memref<1024x4096xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>,
       %arg1: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>):
    %cx = memref.alloc() : memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %arg0 into %cx
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
      into memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.store %cx, %arg1
        : memref<1024x4096xf16, #wafer.memory<spm, cx>>
       -> memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %arg1
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @unpack_large_cx_to_tensor_mapping
// CHECK-COUNT-1: wafer.instr.wdma
// CHECK-SAME: dst_iterations = array<i64: 1024, 64, 1>
// CHECK-SAME: dst_strides = array<i64: 8192, 128, 0>

func.func @unpack_ragged_cx_to_tensor_mapping(
    %input: memref<1025x4096xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<1025x4096xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<1025x4096xf16, #wafer.memory<ddr, tensor>>,
        memref<1025x4096xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1025x4096xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<1025x4096xf16, #wafer.memory<ddr, tensor>>,
       %arg1: memref<1025x4096xf16, #wafer.memory<ddr, tensor>>):
    %cx = memref.alloc() : memref<1025x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %arg0 into %cx
        : memref<1025x4096xf16, #wafer.memory<ddr, tensor>>
      into memref<1025x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.store %cx, %arg1
        : memref<1025x4096xf16, #wafer.memory<spm, cx>>
       -> memref<1025x4096xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %arg1
        : memref<1025x4096xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @unpack_ragged_cx_to_tensor_mapping
// CHECK: wafer.instr.wdma
// CHECK-SAME: dst_iterations = array<i64: 1025, 64, 1>
// CHECK-SAME: dst_strides = array<i64: 8192, 128, 0>
