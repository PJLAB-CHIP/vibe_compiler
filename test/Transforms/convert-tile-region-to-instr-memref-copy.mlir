// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @aligned_bufferization_copy() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %source = memref.alloc()
        : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
    %target = memref.alloc()
        : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
    memref.copy %source, %target
        : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
          to memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @aligned_bufferization_copy
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: memref.copy

func.func @ragged_bufferization_copy() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %source = memref.alloc()
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    %target = memref.alloc()
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    memref.copy %source, %target
        : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
          to memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ragged_bufferization_copy
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: memref.copy

func.func @ddr_bufferization_copy() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %source = memref.alloc()
        : memref<1x1024x4xf32, #wafer.memory<ddr, tensor>>
    %target = memref.alloc()
        : memref<1x1024x4xf32, #wafer.memory<ddr, tensor>>
    memref.copy %source, %target
        : memref<1x1024x4xf32, #wafer.memory<ddr, tensor>>
          to memref<1x1024x4xf32, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ddr_bufferization_copy
// CHECK: %[[STAGING:.+]] = memref.alloc() : memref<1x1024x4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma %{{.+}} to %[[STAGING]]
// CHECK: wafer.instr.wdma %[[STAGING]] to %{{.+}}
// CHECK-NOT: memref.copy
