// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @ordered_sum() {
  // Tiny shape isolates the exact non-identity-init ordered recurrence.
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<3x2xf32, #wafer.memory<spm, cx>>
  %init = arith.constant 1.000000e+00 : f32
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input, %init
      {dimensions = array<i64: 0>}
      : (memref<3x2xf32, #wafer.memory<spm, cx>>, f32)
     -> memref<2xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_sum
// CHECK: %[[INIT:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[A:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[B:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[SLICE:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[DEST:.*]] = memref.alloc() : memref<2xf32, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.fill %[[A]], %[[INIT]]
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-SAME: iter_args(%[[CURRENT:.*]] = %[[A]], %[[NEXT:.*]] = %[[B]])
// CHECK: wafer.instr.gather_scatter %{{.+}} to %[[SLICE]] src_offset_value(%{{.+}})
// CHECK-NEXT: wafer.instr.elementwise <add> %[[CURRENT]], %[[SLICE]] into %[[NEXT]]
// CHECK: scf.yield %[[NEXT]], %[[CURRENT]]
// CHECK: wafer.instr.gather_scatter %[[LOOP]]#0 to %[[DEST]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: wafer.instr.reduce
// CHECK: return

func.func @sorted_multidim_lexicographic() {
  // Tiny shape isolates ordered multi-axis traversal with non-identity init.
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x1x2xf32, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 0>, init_value = 1.000000e+00 : f32}
      : (memref<2x1x2xf32, #wafer.memory<spm, ncx>>)
     -> memref<1xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @sorted_multidim_lexicographic
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK-COUNT-2: scf.for
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_offset_value
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.ncc_join [0]
// CHECK: return

func.func @large_identity_sum_uses_native_reduce() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1024x1xf32, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f32}
      : (memref<1024x1xf32, #wafer.memory<spm, cx>>)
     -> memref<1xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @large_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 1 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return
// CHECK-NOT: wafer.instr.ncc_join [0]

func.func @large_f16_identity_sum_uses_native_reduce() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<4096x1xf16, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f16}
      : (memref<4096x1xf16, #wafer.memory<spm, cx>>)
     -> memref<1xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @large_f16_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 1 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return
// CHECK-NOT: wafer.instr.ncc_join [0]

func.func @large_multidim_identity_sum_uses_native_chain() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 3>, init_value = 0.000000e+00 : f16}
      : (memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @large_multidim_identity_sum_uses_native_chain
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: scf.for
// CHECK: %[[TMP:.*]] = memref.alloc() : memref<1x24x32x1xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.reduce <sum> %{{.*}} into %[[TMP]]
// CHECK-SAME: dim = 0 : i64
// CHECK: %[[DEST:.*]] = memref.alloc() : memref<1x24x1x1xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.reduce <sum> %[[TMP]] into %[[DEST]]
// CHECK-SAME: dim = 1 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: byte_count = 48 : i64
// CHECK-SAME: dst_strides = array<i64: 2, 0, 0>
// CHECK-SAME: src_strides = array<i64: 8, 0, 0>
// CHECK-NEXT: wafer.tile.yield

func.func @ragged_multidim_identity_sum_uses_native_chain() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 3>, init_value = 0.000000e+00 : f16}
      : (memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ragged_multidim_identity_sum_uses_native_chain
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: scf.for
// CHECK-COUNT-2: wafer.instr.reduce <sum>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: byte_count = 48 : i64
// CHECK-SAME: dst_strides = array<i64: 2, 0, 0>
// CHECK-SAME: src_strides = array<i64: 8, 0, 0>
// CHECK-NEXT: wafer.tile.yield

func.func @large_single_axis_identity_sum_uses_native_reduce() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 3>, init_value = 0.000000e+00 : f16}
      : (memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24x32xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @large_single_axis_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: scf.for
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 0 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: byte_count = 1536 : i64
// CHECK-SAME: inner_bytes = 2 : i64
// CHECK-SAME: src_strides = array<i64: 8, 0, 0>
// CHECK-NEXT: wafer.tile.yield

func.func @ragged_single_axis_identity_sum_uses_native_reduce() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 3>, init_value = 0.000000e+00 : f16}
      : (memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24x32xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ragged_single_axis_identity_sum_uses_native_reduce
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: scf.for
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 0 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: byte_count = 1536 : i64
// CHECK-SAME: inner_bytes = 2 : i64
// CHECK-SAME: src_strides = array<i64: 8, 0, 0>
// CHECK-NEXT: wafer.tile.yield

func.func @large_multidim_sum_keeps_inner_affine_runs() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 3>, init_value = -0.000000e+00 : f16}
      : (memref<1x24x32x1024xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24x1x1xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @large_multidim_sum_keeps_inner_affine_runs
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK-COUNT-3: scf.for
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.ncc_join [0]
// CHECK: return

func.func @ragged_multidim_sum_keeps_piece_loop_and_tail() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2, 3>, init_value = -0.000000e+00 : f16}
      : (memref<1x24x32x1025xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x24x1x1xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ragged_multidim_sum_keeps_piece_loop_and_tail
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK-COUNT-3: scf.for
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_strides = array<i64: 256, 0, 0>
// CHECK-NEXT: wafer.instr.elementwise <add>
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.ncc_join [0]
// CHECK: return

func.func @ordered_max() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x2xf16, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<max> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<2x2xf16, #wafer.memory<spm, cx>>)
     -> memref<2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_max
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: scf.for
// CHECK: wafer.instr.elementwise <max>
// CHECK: scf.yield
// CHECK: return

func.func @ordered_min() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x2xf16, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<min> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<2x2xf16, #wafer.memory<spm, cx>>)
     -> memref<2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_min
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: scf.for
// CHECK: wafer.instr.elementwise <min>
// CHECK: scf.yield
// CHECK: return
