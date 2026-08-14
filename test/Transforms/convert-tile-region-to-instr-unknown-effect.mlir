// RUN: not wafer-opt --allow-unregistered-dialect \
// RUN:   --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 \
// RUN:   | FileCheck %s

func.func @unknown_effect() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %tile_slot = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  %tile_zero = arith.constant 0.000000e+00 : f32
  wafer.instr.fill %tile_slot, %tile_zero
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  "test.untracked_observer"() : () -> ()
  %index = arith.constant 0 : index
  %value = memref.load %tile_slot[%index]
      : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: required_ncc_join_failure: operation without a typed memory-effect contract may observe pending NCC work
