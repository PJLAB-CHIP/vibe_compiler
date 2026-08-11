// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @cross_worker_alias(%zero: f32) {
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  %result = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %shared, %zero
      {worker = #wafer.ncc_worker<worker0>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.elementwise <add> %shared, %shared into %result
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>,
        memref<4xf32, #wafer.memory<spm, tensor>>
    into memref<4xf32, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @cross_worker_alias
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.instr.elementwise
// CHECK-SAME: worker = #wafer.ncc_worker<worker1>
// CHECK-NEXT: wafer.instr.ncc_join [1]
// CHECK-NEXT: return

func.func @mixed_dte_ncc_backedge(%zero: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1 {
    wafer.instr.fill %shared, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    %token = wafer.instr.dte_send %shared
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    wafer.instr.elementwise <add> %shared, %shared into %shared
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @mixed_dte_ncc_backedge
// CHECK: scf.for
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: %[[TOKEN:.*]] = wafer.instr.dte_send
// CHECK-NEXT: wafer.instr.dte_wait %[[TOKEN]]
// CHECK-NEXT: wafer.instr.elementwise
// CHECK-NEXT: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return

func.func @cross_worker_disjoint(%zero: f32)
    -> (memref<4xf32, #wafer.memory<spm, tensor>>,
        memref<4xf32, #wafer.memory<spm, tensor>>) {
  %lhs = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  %rhs = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %lhs, %zero
      {worker = #wafer.ncc_worker<worker0>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.fill %rhs, %zero
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  return %lhs, %rhs
      : memref<4xf32, #wafer.memory<spm, tensor>>,
        memref<4xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @cross_worker_disjoint
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.fill
// CHECK-SAME: worker = #wafer.ncc_worker<worker1>
// CHECK-NEXT: wafer.instr.ncc_join [0, 1]
// CHECK-NEXT: return

func.func @unknown_alias_is_conservative(
    %unknown: memref<4xf32, #wafer.memory<spm, tensor>>, %zero: f32)
    -> memref<4xf32, #wafer.memory<spm, tensor>> {
  %local = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %local, %zero
      {worker = #wafer.ncc_worker<worker0>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.fill %unknown, %zero
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  return %local : memref<4xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @unknown_alias_is_conservative
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.instr.fill
// CHECK-SAME: worker = #wafer.ncc_worker<worker1>
// CHECK-NEXT: wafer.instr.ncc_join [1]
// CHECK-NEXT: return

func.func @same_worker_alias(%zero: f32) {
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %shared, %zero
      {worker = #wafer.ncc_worker<worker0>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.elementwise <add> %shared, %shared into %shared
      {worker = #wafer.ncc_worker<worker0>}
      : memref<4xf32, #wafer.memory<spm, tensor>>,
        memref<4xf32, #wafer.memory<spm, tensor>>
    into memref<4xf32, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @same_worker_alias
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.elementwise
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return

func.func @branch_only_issue(%condition: i1, %zero: f32)
    -> memref<4xf32, #wafer.memory<spm, tensor>> {
  %slot = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.if %condition {
    wafer.instr.fill %slot, %zero
        {worker = #wafer.ncc_worker<worker2>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  }
  return %slot : memref<4xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @branch_only_issue
// CHECK: scf.if
// CHECK: wafer.instr.fill
// CHECK-SAME: worker = #wafer.ncc_worker<worker2>
// CHECK: wafer.instr.ncc_join [2]
// CHECK-NEXT: return

func.func @single_trip_cross_worker(%zero: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c1 step %c1 {
    wafer.instr.fill %shared, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise <add> %shared, %shared into %shared
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @single_trip_cross_worker
// CHECK: scf.for
// CHECK-NOT: wafer.instr.ncc_join [1]
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.instr.elementwise
// CHECK: wafer.instr.ncc_join [1]
// CHECK-NEXT: return

func.func @static_loop_bounds_through_residency_inputs(%zero: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %ignored = wafer.tile.region(%c0, %c4, %c1, %zero
      : index, index, index, f32) -> (f32) {
  ^bb0(%lower: index, %upper: index, %step: index, %value: f32):
    %buffer = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
    scf.for %iv = %lower to %upper step %step {
      wafer.instr.fill %buffer, %value
          {worker = #wafer.ncc_worker<worker0>}
          : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    }
    wafer.instr.elementwise <add> %buffer, %buffer into %buffer
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %value : f32
  }
  return
}

// CHECK-LABEL: func.func @static_loop_bounds_through_residency_inputs
// CHECK: wafer.tile.region
// CHECK: scf.for
// CHECK: wafer.instr.fill
// CHECK: }
// CHECK-NEXT: wafer.instr.elementwise
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: return

func.func @cross_worker_backedge(%zero: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1 {
    wafer.instr.fill %shared, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise <add> %shared, %shared into %shared
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @cross_worker_backedge
// CHECK: scf.for
// CHECK: wafer.instr.ncc_join [1]
// CHECK-NEXT: wafer.instr.fill
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.instr.elementwise
// CHECK-SAME: worker = #wafer.ncc_worker<worker1>
// CHECK: wafer.instr.ncc_join [1]
// CHECK-NEXT: return

func.func @same_worker_backedge(%zero: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1 {
    wafer.instr.fill %shared, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise <add> %shared, %shared into %shared
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @same_worker_backedge
// CHECK: scf.for
// CHECK-NOT: wafer.instr.ncc_join
// CHECK: wafer.instr.fill
// CHECK-NEXT: wafer.instr.elementwise
// CHECK: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: return

func.func @conditional_same_worker_backedge(%condition: i1, %zero: f32)
    -> memref<4xf32, #wafer.memory<spm, tensor>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1 {
    scf.if %condition {
      wafer.instr.fill %shared, %zero
          {worker = #wafer.ncc_worker<worker0>}
          : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    }
  }
  return %shared : memref<4xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @conditional_same_worker_backedge
// CHECK: scf.for
// CHECK: scf.if
// CHECK: wafer.instr.fill
// CHECK: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: }
// CHECK-NOT: wafer.instr.ncc_join
// CHECK-NEXT: return %{{.*}} : memref<4xf32, #wafer.memory<spm, tensor>>

func.func @region_exit_completes_only_region_local_roots(%zero: f32) {
  %outer = memref.alloc()
      : memref<4xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %outer, %zero
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  %ignored0 = wafer.tile.region(%zero : f32) -> (f32) {
  ^bb0(%value: f32):
    %local0 = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.fill %local0, %value
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise <add> %local0, %local0 into %local0
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %value : f32
  }
  %ignored1 = wafer.tile.region(%zero : f32) -> (f32) {
  ^bb0(%value: f32):
    %local1 = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.fill %local1, %value
        {worker = #wafer.ncc_worker<worker2>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise <add> %local1, %local1 into %local1
        {worker = #wafer.ncc_worker<worker2>}
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %value : f32
  }
  wafer.instr.elementwise <add> %outer, %outer into %outer
      {worker = #wafer.ncc_worker<worker1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>,
        memref<4xf32, #wafer.memory<spm, tensor>>
    into memref<4xf32, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @region_exit_completes_only_region_local_roots
// CHECK: wafer.instr.fill %[[OUTER:[^,]+]],
// CHECK-SAME: worker = #wafer.ncc_worker<worker1>
// CHECK: wafer.tile.region
// CHECK: %[[LOCAL0:[^ ]+]] = memref.alloc()
// CHECK: wafer.instr.fill %[[LOCAL0]]
// CHECK: wafer.instr.elementwise <add> %[[LOCAL0]], %[[LOCAL0]] into %[[LOCAL0]]
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: %{{.*}} = wafer.tile.region
// CHECK: %[[LOCAL1:[^ ]+]] = memref.alloc()
// CHECK: wafer.instr.fill %[[LOCAL1]]
// CHECK: wafer.instr.elementwise <add> %[[LOCAL1]], %[[LOCAL1]] into %[[LOCAL1]]
// CHECK-NEXT: wafer.instr.ncc_join [2]
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: wafer.instr.elementwise <add> %[[OUTER]], %[[OUTER]] into %[[OUTER]]
// CHECK-NEXT: wafer.instr.ncc_join [1]
// CHECK-NEXT: return

func.func @managed_materialization_releases_each_spm_root(
    %input: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%input, %output
      : memref<128xf16, #wafer.memory<ddr, tensor>>,
        memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %first = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %first
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       into memref<128xf16, #wafer.memory<spm, tensor>>
    %spill = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.store %first, %spill
        : memref<128xf16, #wafer.memory<spm, tensor>>
       -> memref<128xf16, #wafer.memory<ddr, tensor>>
    %middle = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %middle
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       into memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %middle, %out
        : memref<128xf16, #wafer.memory<spm, tensor>>
       -> memref<128xf16, #wafer.memory<ddr, tensor>>
    %reloaded = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %spill into %reloaded
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       into memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %reloaded, %out
        : memref<128xf16, #wafer.memory<spm, tensor>>
       -> memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @managed_materialization_releases_each_spm_root
// CHECK: wafer.instr.rdma %{{.+}} to %[[FIRST:[^ ]+]]
// CHECK: wafer.instr.wdma %[[FIRST]] to %[[SPILL:[^ ]+]]
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK: wafer.instr.rdma %{{.+}} to %[[MIDDLE:[^ ]+]]
// CHECK-NEXT: wafer.instr.wdma %[[MIDDLE]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NEXT: %[[RELOADED:[^ ]+]] = memref.alloc()
// CHECK-NEXT: wafer.instr.rdma %[[SPILL]] to %[[RELOADED]]
// CHECK-NEXT: wafer.instr.wdma %[[RELOADED]]
// CHECK-NEXT: wafer.instr.ncc_join [0]
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: return
