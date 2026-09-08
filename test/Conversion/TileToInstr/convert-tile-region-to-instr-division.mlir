// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

// CHECK-LABEL: func.func @divide_1024
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <abs>
// CHECK: wafer.instr.elementwise <lt>
// CHECK: wafer.instr.elementwise <ne>
// CHECK: wafer.instr.elementwise <logic_and>
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.mask_move
func.func @divide_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x2x1024xf32, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<1x2x1024xf32, #wafer.memory<spm, tensor>>
    %result = wafer.tile.elementwise #wafer.elementwise_kind<div> %lhs, %rhs
        : (memref<1x2x1024xf32, #wafer.memory<spm, tensor>>,
           memref<1x2x1024xf32, #wafer.memory<spm, tensor>>)
       -> memref<1x2x1024xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @divide_1025
// CHECK: memref.alloc() : memref<1x2x1032xf32, #wafer.memory<spm, tensor>>
// CHECK: memref.subview
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <abs>
// CHECK: wafer.instr.elementwise <lt>
// CHECK: wafer.instr.elementwise <ne>
// CHECK: wafer.instr.elementwise <logic_and>
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.mask_move
// CHECK: wafer.instr.gather_scatter
func.func @divide_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x2x1025xf32, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<1x2x1025xf32, #wafer.memory<spm, tensor>>
    %result = wafer.tile.elementwise #wafer.elementwise_kind<div> %lhs, %rhs
        : (memref<1x2x1025xf32, #wafer.memory<spm, tensor>>,
           memref<1x2x1025xf32, #wafer.memory<spm, tensor>>)
       -> memref<1x2x1025xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @divide_1031
// CHECK: memref.alloc() : memref<1x2x1032xf32, #wafer.memory<spm, tensor>>
// CHECK: memref.subview
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <mul>
// CHECK: wafer.instr.elementwise <sub>
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.elementwise <abs>
// CHECK: wafer.instr.elementwise <lt>
// CHECK: wafer.instr.elementwise <ne>
// CHECK: wafer.instr.elementwise <logic_and>
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.elementwise <div>
// CHECK: wafer.instr.mask_move
// CHECK: wafer.instr.gather_scatter
func.func @divide_1031() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x2x1031xf32, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<1x2x1031xf32, #wafer.memory<spm, tensor>>
    %result = wafer.tile.elementwise #wafer.elementwise_kind<div> %lhs, %rhs
        : (memref<1x2x1031xf32, #wafer.memory<spm, tensor>>,
           memref<1x2x1031xf32, #wafer.memory<spm, tensor>>)
       -> memref<1x2x1031xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @divide_blocked_1024
// CHECK-COUNT-2: wafer.instr.fill {{.*}}physical_footprint
// CHECK: wafer.instr.mask_move
func.func @divide_blocked_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x1024x1024xf32, #wafer.memory<spm, ncx>>
    %rhs = memref.alloc() : memref<1x1024x1024xf32, #wafer.memory<spm, ncx>>
    %result = wafer.tile.elementwise <div> %lhs, %rhs : (memref<1x1024x1024xf32, #wafer.memory<spm, ncx>>, memref<1x1024x1024xf32, #wafer.memory<spm, ncx>>) -> memref<1x1024x1024xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @divide_blocked_1025
// CHECK-COUNT-2: wafer.instr.fill {{.*}}physical_footprint
// CHECK: wafer.instr.mask_move
func.func @divide_blocked_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x1025x1024xf32, #wafer.memory<spm, ncx>>
    %rhs = memref.alloc() : memref<1x1025x1024xf32, #wafer.memory<spm, ncx>>
    %result = wafer.tile.elementwise <div> %lhs, %rhs : (memref<1x1025x1024xf32, #wafer.memory<spm, ncx>>, memref<1x1025x1024xf32, #wafer.memory<spm, ncx>>) -> memref<1x1025x1024xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @divide_blocked_1031
// CHECK-COUNT-2: wafer.instr.fill {{.*}}physical_footprint
// CHECK: wafer.instr.mask_move
func.func @divide_blocked_1031() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %lhs = memref.alloc() : memref<1x1031x1024xf32, #wafer.memory<spm, ncx>>
    %rhs = memref.alloc() : memref<1x1031x1024xf32, #wafer.memory<spm, ncx>>
    %result = wafer.tile.elementwise <div> %lhs, %rhs : (memref<1x1031x1024xf32, #wafer.memory<spm, ncx>>, memref<1x1031x1024xf32, #wafer.memory<spm, ncx>>) -> memref<1x1031x1024xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}
