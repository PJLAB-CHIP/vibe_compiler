// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

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
// CHECK-NEXT: wafer.tile.yield
// CHECK: }
// CHECK-NEXT: wafer.instr.ncc_join [0]
