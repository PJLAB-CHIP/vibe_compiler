// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @ordered_trailing_tensor_sum() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %tile_input = memref.alloc()
      : memref<2x2x512xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %tile_input
      {dimensions = array<i64: 2>, init_value = 0.000000e+00 : f16}
      : (memref<2x2x512xf16, #wafer.memory<spm, ncx>>)
     -> memref<2x2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_trailing_tensor_sum
// Eight exact 64-channel physical runs cover all 512 ordered tuples. The
// body remains one gather followed by one accumulation, with both
// accumulators carried and swapped across each loop iteration.
// CHECK: scf.for
// CHECK: src_offset_value(%{{.+}})
// CHECK: scf.yield %{{.+}}, %{{.+}}
// CHECK-COUNT-7: scf.for
// CHECK: wafer.instr.gather_scatter %{{.+}} to %{{.+}}
// CHECK-NOT: wafer.tile.reduce
// CHECK: return

func.func @ordered_aligned_workload_reduction() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc()
        : memref<2x2x1024xf16, #wafer.memory<spm, ncx>>
    // Zero is intentionally not max's identity. This keeps the source-ordered
    // route without introducing a numeric-policy choice into the test.
    %result = wafer.tile.reduce #wafer.reduce_kind<max> %input
        {dimensions = array<i64: 2>, init_value = 0.000000e+00 : f16}
        : (memref<2x2x1024xf16, #wafer.memory<spm, ncx>>)
       -> memref<2x2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_aligned_workload_reduction
// CHECK: scf.for
// CHECK: src_offset_value(%{{.+}})
// CHECK: scf.yield %{{.+}}, %{{.+}}
// CHECK-COUNT-15: scf.for
// CHECK-NOT: wafer.tile.reduce
// CHECK: return

func.func @ordered_ragged_workload_reduction() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc()
        : memref<2x2x1031xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.reduce #wafer.reduce_kind<max> %input
        {dimensions = array<i64: 2>, init_value = 0.000000e+00 : f16}
        : (memref<2x2x1031xf16, #wafer.memory<spm, ncx>>)
       -> memref<2x2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_ragged_workload_reduction
// Sixteen full physical blocks plus the seven-channel tail are separate exact
// affine descriptor classes.
// CHECK: scf.for
// CHECK: src_offset_value(%{{.+}})
// CHECK: scf.yield %{{.+}}, %{{.+}}
// CHECK-COUNT-16: scf.for
// CHECK-NOT: wafer.tile.reduce
// CHECK: return
