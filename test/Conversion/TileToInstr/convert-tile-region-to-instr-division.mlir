// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s --implicit-check-not="wafer.instr.elementwise <div>" --implicit-check-not="wafer.instr.bit2fp" --implicit-check-not="wafer.instr.mask_move" --implicit-check-not="wafer.instr.fill"

// CHECK-LABEL: func.func @divide_1024
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
// CHECK: wafer.instr.elementwise <recip> %[[DEN:.*]] into %[[RECIP:.*]] :
// CHECK-NEXT: wafer.instr.elementwise <mul> %{{.*}}, %[[RECIP]] into %{{.*}} :
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
