// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @ordered_sum(
    %input: memref<3x2xf32, #wafer.memory<spm, cx>>)
    -> memref<2xf32, #wafer.memory<spm, cx>> {
  %init = arith.constant 0.000000e+00 : f32
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input, %init
      {dimensions = array<i64: 0>}
      : (memref<3x2xf32, #wafer.memory<spm, cx>>, f32)
     -> memref<2xf32, #wafer.memory<spm, cx>>
  return %result : memref<2xf32, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @ordered_sum
// CHECK: %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[A:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[B:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[SLICE:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[DEST:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.fill %[[A]], %[[INIT]]
// CHECK: wafer.instr.gather_scatter %arg0 to %[[SLICE]]
// CHECK-NOT: src_offset
// CHECK-NEXT: wafer.instr.elementwise <add> %[[A]], %[[SLICE]] into %[[B]]
// CHECK: wafer.instr.gather_scatter %arg0 to %[[SLICE]]
// CHECK-SAME: src_offset = 16
// CHECK-NEXT: wafer.instr.elementwise <add> %[[B]], %[[SLICE]] into %[[A]]
// CHECK: wafer.instr.gather_scatter %arg0 to %[[SLICE]]
// CHECK-SAME: src_offset = 32
// CHECK-NEXT: wafer.instr.elementwise <add> %[[A]], %[[SLICE]] into %[[B]]
// CHECK: wafer.instr.gather_scatter %[[B]] to %[[DEST]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: wafer.instr.reduce
// CHECK: return %[[DEST]]

func.func @sorted_multidim_lexicographic(
    %input: memref<2x1x2xf32, #wafer.memory<spm, ncx>>)
    -> memref<1xf32, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 0>, init_value = 0.000000e+00 : f32}
      : (memref<2x1x2xf32, #wafer.memory<spm, ncx>>)
     -> memref<1xf32, #wafer.memory<spm, cx>>
  return %result : memref<1xf32, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @sorted_multidim_lexicographic
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.gather_scatter
// CHECK-NOT: src_offset
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_offset = 4
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_offset = 256
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_offset = 260
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.ncc_join [0]
// CHECK: return

func.func @large_identity_sum_uses_native_reduce(
    %input: memref<1024x1xf32, #wafer.memory<spm, cx>>)
    -> memref<1xf32, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f32}
      : (memref<1024x1xf32, #wafer.memory<spm, cx>>)
     -> memref<1xf32, #wafer.memory<spm, cx>>
  return %result : memref<1xf32, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @large_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 1 : i64
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK: return
// CHECK-NOT: wafer.instr.local_fence

func.func @large_f16_identity_sum_uses_native_reduce(
    %input: memref<4096x1xf16, #wafer.memory<spm, cx>>)
    -> memref<1xf16, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f16}
      : (memref<4096x1xf16, #wafer.memory<spm, cx>>)
     -> memref<1xf16, #wafer.memory<spm, cx>>
  return %result : memref<1xf16, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @large_f16_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 1 : i64
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK: return
// CHECK-NOT: wafer.instr.local_fence

func.func @ordered_max(
    %input: memref<2x2xf16, #wafer.memory<spm, cx>>)
    -> memref<2xf16, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<max> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<2x2xf16, #wafer.memory<spm, cx>>)
     -> memref<2xf16, #wafer.memory<spm, cx>>
  return %result : memref<2xf16, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @ordered_max
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.elementwise <max>
// CHECK: wafer.instr.elementwise <max>
// CHECK: return

func.func @ordered_min(
    %input: memref<2x2xf16, #wafer.memory<spm, cx>>)
    -> memref<2xf16, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<min> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<2x2xf16, #wafer.memory<spm, cx>>)
     -> memref<2xf16, #wafer.memory<spm, cx>>
  return %result : memref<2xf16, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @ordered_min
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.elementwise <min>
// CHECK: wafer.instr.elementwise <min>
// CHECK: return
