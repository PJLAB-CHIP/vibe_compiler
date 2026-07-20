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
    %loaded = memref.alloc()
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %loaded
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
      into memref<4x8xf16, #wafer.memory<spm, tensor>>
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
// CHECK: wafer.instr.rdma %{{[a-zA-Z0-9_]+}} to %[[LOAD_DST]]
// CHECK-SAME: byte_count = 64 : i64
// CHECK: wafer.instr.fill %[[LOAD_DST]]
// CHECK: %[[SUM:.+]] = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.elementwise <add> %[[LOAD_DST]], %[[LOAD_DST]] into %[[SUM]]
// CHECK: wafer.instr.wdma %[[SUM]] to %{{.+}}
// CHECK: wafer.instr.local_fence
// CHECK-NEXT: wafer.tile.yield

func.func @strided_ddr_tile_load_store(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %output_tile = memref.subview %output[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input_tile, %output_tile
      : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %loaded
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %loaded, %out
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @strided_ddr_tile_load_store
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.store
// CHECK: %[[INPUT_TILE:.+]] = memref.subview
// CHECK: %[[OUTPUT_TILE:.+]] = memref.subview
// CHECK: %[[LOAD_DST:.+]] = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma %{{[a-zA-Z0-9_]+}} to %[[LOAD_DST]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.wdma %[[LOAD_DST]] to %{{[a-zA-Z0-9_]+}}
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 16, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>

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
    %lhs_tensor = memref.alloc()
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %lhs_boundary into %lhs_tensor
        : memref<4x64xf16, #wafer.memory<ddr, tensor>>
      into memref<4x64xf16, #wafer.memory<spm, tensor>>
    %rhs_tensor = memref.alloc()
        : memref<64x64xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %rhs_boundary into %rhs_tensor
        : memref<64x64xf16, #wafer.memory<ddr, tensor>>
      into memref<64x64xf16, #wafer.memory<spm, tensor>>
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
    %zero_const = arith.constant 0.0 : f16
    %reduced = wafer.tile.reduce #wafer.reduce_kind<sum> %gemm, %zero_const
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
// CHECK-NOT: wafer.instr.reduce
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.local_fence
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.local_fence
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.local_fence
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.local_fence
// CHECK-NOT: wafer.instr.reduce
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
    %loaded = memref.alloc()
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %loaded
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
      into memref<4x8xf16, #wafer.memory<spm, tensor>>
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
    %loaded = memref.alloc()
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %loaded
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
      into memref<4x8xf16, #wafer.memory<spm, tensor>>
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

func.func @movement_extract_insert_broadcast_transpose(%zero: f16) {
  %region = wafer.tile.region(%zero : f16) -> (f16) {
  ^bb0(%fill: f16):
    %wide = memref.alloc()
        : memref<8xf16, #wafer.memory<spm, tensor>>
    %slice = wafer.tile.extract_slice %wide
        {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
        : memref<8xf16, #wafer.memory<spm, tensor>>
       -> memref<4xf16, #wafer.memory<spm, tensor>>
    %inserted = wafer.tile.insert_slice %slice into %wide
        {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
         into memref<8xf16, #wafer.memory<spm, tensor>>
       -> memref<8xf16, #wafer.memory<spm, tensor>>
    %matrix = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %transposed = wafer.tile.transpose %matrix
        {permutation = array<i64: 1, 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<3x2xf16, #wafer.memory<spm, tensor>>
    %vector = memref.alloc()
        : memref<3xf16, #wafer.memory<spm, tensor>>
    %broadcast = wafer.tile.broadcast %vector
        {dimensions = array<i64: 1>}
        : memref<3xf16, #wafer.memory<spm, tensor>>
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %fill : f16
  }
  return
}

// CHECK-LABEL: func.func @movement_extract_insert_broadcast_transpose
// CHECK-NOT: wafer.tile.extract_slice
// CHECK-NOT: wafer.tile.insert_slice
// CHECK-NOT: wafer.tile.transpose
// CHECK-NOT: wafer.tile.broadcast
// CHECK: %[[WIDE:.+]] = memref.alloc() : memref<8xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[SLICE:.+]] = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[WIDE]] to %[[SLICE]]
// CHECK-SAME: byte_count = 8 : i64
// CHECK-SAME: src_offset = 4 : i64
// CHECK: %[[INSERTED:.+]] = memref.alloc() : memref<8xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[WIDE]] to %[[INSERTED]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK: wafer.instr.gather_scatter %[[SLICE]] to %[[INSERTED]]
// CHECK-SAME: byte_count = 8 : i64
// CHECK-SAME: dst_offset = 4 : i64
// CHECK: %[[MATRIX:.+]] = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[TRANSPOSED:.+]] = memref.alloc() : memref<3x2xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[MATRIX]] to %[[TRANSPOSED]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 3, 1>
// CHECK-SAME: dst_strides = array<i64: 2, 4, 0>
// CHECK-SAME: inner_bytes = 2 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 3, 1>
// CHECK-SAME: src_strides = array<i64: 6, 2, 0>
// CHECK-NOT: wafer.instr.gather_scatter %[[MATRIX]] to %[[TRANSPOSED]]
// CHECK: %[[VECTOR:.+]] = memref.alloc() : memref<3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[BROADCAST:.+]] = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[VECTOR]] to %[[BROADCAST]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 6, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 0, 0, 0>
// CHECK-NOT: wafer.instr.gather_scatter %[[VECTOR]] to %[[BROADCAST]]

func.func @movement_cx_ncx_block_major_offsets(%zero: f16) {
  %region = wafer.tile.region(%zero : f16) -> (f16) {
  ^bb0(%fill: f16):
    %cx_source = memref.alloc()
        : memref<2x128xf16, #wafer.memory<spm, cx>>
    %cx_slice = wafer.tile.extract_slice %cx_source
        {offsets = array<i64: 1, 0>, sizes = array<i64: 1, 128>, strides = array<i64: 1, 1>}
        : memref<2x128xf16, #wafer.memory<spm, cx>>
       -> memref<1x128xf16, #wafer.memory<spm, cx>>
    %ncx_source = memref.alloc()
        : memref<2x2x128xf16, #wafer.memory<spm, ncx>>
    %ncx_slice = wafer.tile.extract_slice %ncx_source
        {offsets = array<i64: 1, 0, 0>, sizes = array<i64: 1, 2, 128>, strides = array<i64: 1, 1, 1>}
        : memref<2x2x128xf16, #wafer.memory<spm, ncx>>
       -> memref<1x2x128xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %fill : f16
  }
  return
}

// CHECK-LABEL: func.func @movement_cx_ncx_block_major_offsets
// CHECK-NOT: wafer.tile.extract_slice
// CHECK: %[[CX_SOURCE:.+]] = memref.alloc() : memref<2x128xf16, #wafer.memory<spm, cx>>
// CHECK: %[[CX_SLICE:.+]] = memref.alloc() : memref<1x128xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.gather_scatter %[[CX_SOURCE]] to %[[CX_SLICE]]
// CHECK-SAME: byte_count = 256 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 128, 0, 0>
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_offset = 128 : i64
// CHECK-SAME: src_strides = array<i64: 256, 0, 0>
// CHECK-NOT: wafer.instr.gather_scatter %[[CX_SOURCE]] to %[[CX_SLICE]]
// CHECK: %[[NCX_SOURCE:.+]] = memref.alloc() : memref<2x2x128xf16, #wafer.memory<spm, ncx>>
// CHECK: %[[NCX_SLICE:.+]] = memref.alloc() : memref<1x2x128xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.gather_scatter %[[NCX_SOURCE]] to %[[NCX_SLICE]]
// CHECK-SAME: byte_count = 512 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: dst_strides = array<i64: 256, 128, 0>
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: src_offset = 512 : i64
// CHECK-SAME: src_strides = array<i64: 256, 128, 0>
// CHECK-NOT: wafer.instr.gather_scatter %[[NCX_SOURCE]] to %[[NCX_SLICE]]

func.func @reshape_view(
    %input: memref<4x16xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<8x8xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<4x16xf16, #wafer.memory<ddr, tensor>>,
        memref<8x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<8x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x16xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<8x8xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc()
        : memref<4x16xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %in into %loaded
        : memref<4x16xf16, #wafer.memory<ddr, tensor>>
      into memref<4x16xf16, #wafer.memory<spm, tensor>>
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

func.func @reshape_then_materialize_cx_uses_reshaped_logical_order(%zero: f16) {
  %region = wafer.tile.region(%zero : f16) -> (f16) {
  ^bb0(%fill: f16):
    %source = memref.alloc()
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
    %reshaped = wafer.tile.reshape %source
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
       -> memref<2x128xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %reshaped
        : memref<2x128xf16, #wafer.memory<spm, tensor>>
       -> memref<2x128xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %fill : f16
  }
  return
}

// CHECK-LABEL: func.func @reshape_then_materialize_cx_uses_reshaped_logical_order
// CHECK-NOT: wafer.tile.reshape
// CHECK-NOT: wafer.tile.materialize_layout
// CHECK: %[[SOURCE:.+]] = memref.alloc() : memref<4x64xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[RESHAPED:.+]] = memref.reinterpret_cast %[[SOURCE]]
// CHECK-SAME: sizes: [2, 128]
// CHECK-SAME: strides: [128, 1]
// CHECK: %[[CX:.+]] = memref.alloc() : memref<2x128xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.gather_scatter %[[RESHAPED]] to %[[CX]]
// CHECK-SAME: byte_count = 512 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: dst_strides = array<i64: 256, 128, 0>
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: src_strides = array<i64: 128, 256, 0>
// CHECK-NOT: wafer.instr.gather_scatter %[[RESHAPED]] to %[[CX]]

func.func @reshape_cx_tail_only_metadata_view(%zero: f16) {
  %region = wafer.tile.region(%zero : f16) -> (f16) {
  ^bb0(%fill: f16):
    %source = memref.alloc()
        : memref<4x16xf16, #wafer.memory<spm, cx>>
    %reshaped = wafer.tile.reshape %source
        : memref<4x16xf16, #wafer.memory<spm, cx>>
       -> memref<8x8xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %fill : f16
  }
  return
}

// CHECK-LABEL: func.func @reshape_cx_tail_only_metadata_view
// CHECK-NOT: wafer.tile.reshape
// CHECK: memref.alloc() : memref<4x16xf16, #wafer.memory<spm, cx>>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [8, 8]
// CHECK-SAME: strides: [8, 1]
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: return

func.func @reshape_cx_block_major_materializes(%zero: f16) {
  %region = wafer.tile.region(%zero : f16) -> (f16) {
  ^bb0(%fill: f16):
    %source = memref.alloc()
        : memref<2x128xf16, #wafer.memory<spm, cx>>
    %reshaped = wafer.tile.reshape %source
        : memref<2x128xf16, #wafer.memory<spm, cx>>
       -> memref<4x64xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %fill : f16
  }
  return
}

// CHECK-LABEL: func.func @reshape_cx_block_major_materializes
// CHECK-NOT: wafer.tile.reshape
// CHECK: memref.alloc() : memref<2x128xf16, #wafer.memory<spm, cx>>
// CHECK: %[[RESHAPED:.+]] = memref.alloc() : memref<4x64xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.gather_scatter %{{.+}} to %[[RESHAPED]]
// CHECK-SAME: byte_count = 512 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: dst_strides = array<i64: 128, 256, 0>
// CHECK-SAME: inner_bytes = 128 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 2, 1>
// CHECK-SAME: src_strides = array<i64: 256, 128, 0>
// CHECK-NOT: wafer.instr.gather_scatter %{{.+}} to %[[RESHAPED]]
