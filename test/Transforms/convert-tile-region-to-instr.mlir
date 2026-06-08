// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @load_compute_store(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %fill_value: f16) {
  %region = wafer.tile.region(%input, %output, %fill_value
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>,
        memref<4x8xf16, #wafer.memory<ddr, tensor>>, f16)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %fill: f16):
    %loaded = wafer.tile.load %in
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.fill %loaded, %fill
        : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16
    %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %loaded, %loaded
        : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
           memref<4x8xf16, #wafer.memory<spm, tensor>>)
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %sum, %out
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @load_compute_store
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.fill
// CHECK-NOT: wafer.tile.elementwise
// CHECK-NOT: wafer.tile.store
// CHECK: %[[LOAD_DST:.+]] = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma %{{.+}} to %[[LOAD_DST]]
// CHECK-SAME: byte_count = 64 : i64
// CHECK: wafer.instr.fill %[[LOAD_DST]]
// CHECK: %[[SUM:.+]] = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.elementwise <add> %[[LOAD_DST]], %[[LOAD_DST]] into %[[SUM]]
// CHECK: wafer.instr.wdma %[[SUM]] to %{{.+}}

func.func @gemm_reduce_and_reshape(
    %lhs_ddr: memref<4x64xf16, #wafer.memory<ddr, tensor>>,
    %rhs_ddr: memref<64x64xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<64xf16, #wafer.memory<ddr, tensor>>,
    %zero: f16) {
  %region = wafer.tile.region(%lhs_ddr, %rhs_ddr, %output, %zero
      : memref<4x64xf16, #wafer.memory<ddr, tensor>>,
        memref<64x64xf16, #wafer.memory<ddr, tensor>>,
        memref<64xf16, #wafer.memory<ddr, tensor>>, f16)
      -> (memref<64xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%lhs_boundary: memref<4x64xf16, #wafer.memory<ddr, tensor>>,
       %rhs_boundary: memref<64x64xf16, #wafer.memory<ddr, tensor>>,
       %out_boundary: memref<64xf16, #wafer.memory<ddr, tensor>>,
       %zero_arg: f16):
    %lhs_tensor = wafer.tile.load %lhs_boundary
        : memref<4x64xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x64xf16, #wafer.memory<spm, tensor>>
    %rhs_tensor = wafer.tile.load %rhs_boundary
        : memref<64x64xf16, #wafer.memory<ddr, tensor>>
       -> memref<64x64xf16, #wafer.memory<spm, tensor>>
    %lhs_cx = wafer.tile.materialize_layout %lhs_tensor
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
       -> memref<4x64xf16, #wafer.memory<spm, cx>>
    %rhs_cx = wafer.tile.materialize_layout %rhs_tensor
        : memref<64x64xf16, #wafer.memory<spm, tensor>>
       -> memref<64x64xf16, #wafer.memory<spm, cx>>
    %gemm = wafer.tile.gemm %lhs_cx, %rhs_cx
        : (memref<4x64xf16, #wafer.memory<spm, cx>>,
           memref<64x64xf16, #wafer.memory<spm, cx>>)
       -> memref<4x64xf16, #wafer.memory<spm, cx>>
    %reduced = wafer.tile.reduce #wafer.reduce_kind<sum> %gemm, %zero_arg
        {dimensions = array<i64: 0>}
        : (memref<4x64xf16, #wafer.memory<spm, cx>>, f16)
       -> memref<64xf16, #wafer.memory<spm, cx>>
    %reshaped = wafer.tile.reshape %reduced
        : memref<64xf16, #wafer.memory<spm, cx>>
       -> memref<64xf16, #wafer.memory<spm, cx>>
    %writeback = wafer.tile.materialize_layout %reshaped
        : memref<64xf16, #wafer.memory<spm, cx>>
       -> memref<64xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %writeback, %out_boundary
        : memref<64xf16, #wafer.memory<spm, tensor>>
       -> memref<64xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out_boundary
        : memref<64xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @gemm_reduce_and_reshape
// CHECK-NOT: wafer.tile.materialize_layout
// CHECK-NOT: wafer.tile.gemm
// CHECK-NOT: wafer.tile.reduce
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.gemm
// CHECK-SAME: k = 64 : i64
// CHECK-SAME: m = 4 : i64
// CHECK-SAME: n = 64 : i64
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 0>
// CHECK-NOT: wafer.tile.reshape
// CHECK: wafer.instr.wdma

func.func @nested_control_flow(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %cond: i1) {
  %region = wafer.tile.region(%input, %output, %cond
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>,
        memref<4x8xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %cond_arg: i1):
    %loaded = wafer.tile.load %in
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    %selected = scf.if %cond_arg -> (memref<4x8xf16, #wafer.memory<spm, tensor>>) {
      %copy = wafer.tile.copy %loaded
          : memref<4x8xf16, #wafer.memory<spm, tensor>>
         -> memref<4x8xf16, #wafer.memory<spm, tensor>>
      scf.yield %copy : memref<4x8xf16, #wafer.memory<spm, tensor>>
    } else {
      scf.yield %loaded : memref<4x8xf16, #wafer.memory<spm, tensor>>
    }
    wafer.tile.store %selected, %out
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @nested_control_flow
// CHECK: scf.if
// CHECK-NOT: wafer.tile.copy
// CHECK: wafer.instr.gather_scatter
// CHECK: scf.yield

func.func @nested_loop(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>,
        memref<4x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<4x8xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loaded = wafer.tile.load %in
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    %looped = scf.for %i = %c0 to %c1 step %c1
        iter_args(%iter = %loaded)
        -> (memref<4x8xf16, #wafer.memory<spm, tensor>>) {
      %copy = wafer.tile.copy %iter
          : memref<4x8xf16, #wafer.memory<spm, tensor>>
         -> memref<4x8xf16, #wafer.memory<spm, tensor>>
      scf.yield %copy : memref<4x8xf16, #wafer.memory<spm, tensor>>
    }
    wafer.tile.store %looped, %out
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @nested_loop
// CHECK: scf.for
// CHECK-NOT: wafer.tile.copy
// CHECK: wafer.instr.gather_scatter
// CHECK: scf.yield

func.func @reshape_view(
    %input: memref<4x16xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<8x8xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<4x16xf16, #wafer.memory<ddr, tensor>>,
        memref<8x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<8x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x16xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<8x8xf16, #wafer.memory<ddr, tensor>>):
    %loaded = wafer.tile.load %in
        : memref<4x16xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x16xf16, #wafer.memory<spm, tensor>>
    %view = wafer.tile.reshape %loaded
        : memref<4x16xf16, #wafer.memory<spm, tensor>>
       -> memref<8x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %view, %out
        : memref<8x8xf16, #wafer.memory<spm, tensor>>
       -> memref<8x8xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<8x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @reshape_view
// CHECK-NOT: wafer.tile.reshape
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [8, 8]
// CHECK-SAME: strides: [8, 1]
// CHECK: wafer.instr.wdma
