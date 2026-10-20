module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 0, 2048] [1, 1024, 1024] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>):
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        scf.for %arg6 = %c0 to %c1024 step %c256 {
          %alloc_0 = memref.alloc() {alignment = 256 : i64, wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          wafer.instr.fill %alloc_0, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>, f32
          %c768 = arith.constant 768 : index
          %0 = scf.for %arg7 = %c0 to %c768 step %c256 iter_args(%arg8 = %alloc_0) -> (memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {
            %subview_10 = memref.subview %arg2[0, %arg5, %arg7] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_11 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<720896>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            wafer.instr.rdma %subview_10 to %alloc_11 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 2048, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %c2048_12 = arith.constant 2048 : index
            %5 = arith.addi %arg6, %c2048_12 : index
            %subview_13 = memref.subview %arg3[0, %arg7, %5] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_14 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<327680>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            wafer.instr.rdma %subview_13 to %alloc_14 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 8192, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %alloc_15 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            wafer.instr.gather_scatter %alloc_11 to %alloc_15 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %alloc_16 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<589824>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            wafer.instr.gather_scatter %alloc_14 to %alloc_16 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %alloc_17 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<720896>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.gemm %alloc_15, %alloc_16 psum(%arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) into %alloc_17 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 256 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.gather_scatter %alloc_17 to %arg8 {byte_count = 262144 : i64, dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>, inner_bytes = 262144 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>> to memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.ncc_join [0]
            %6 = wafer.instr.dte_send %alloc_14 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 589824, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 5, round = 0, slice = 0>, peer = 6 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            wafer.instr.dte_wait %6 : !async.token
            %7 = wafer.instr.dte_send %alloc_14 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 589824, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 5, round = 0, slice = 1>, peer = 10 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            wafer.instr.dte_wait %7 : !async.token
            %8 = wafer.instr.dte_send %alloc_14 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 589824, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 5, round = 0, slice = 2>, peer = 14 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            wafer.instr.dte_wait %8 : !async.token
            scf.yield %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          }
          %subview_1 = memref.subview %arg2[0, %arg5, %c768] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_2 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<720896>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.rdma %subview_1 to %alloc_2 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 2048, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %c2048 = arith.constant 2048 : index
          %1 = arith.addi %arg6, %c2048 : index
          %subview_3 = memref.subview %arg3[0, %c768, %1] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_4 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<327680>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.rdma %subview_3 to %alloc_4 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 8192, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %alloc_5 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_2 to %alloc_5 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_6 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<589824>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_4 to %alloc_6 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_7 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<720896>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gemm %alloc_5, %alloc_6 psum(%0 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) into %alloc_7 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 256 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_8 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<851968>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.gather_scatter %alloc_7 to %alloc_8 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 512, 128, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 128, 32768, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %subview_9 = memref.subview %arg4[0, %arg5, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.instr.wdma %alloc_8 to %subview_9 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 1, 1>, dst_strides = array<i64: 8192, 0, 0>, inner_bytes = 512 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.instr.ncc_join [0]
          %2 = wafer.instr.dte_send %alloc_4 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 720896, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 6, round = 0, slice = 0>, peer = 6 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          wafer.instr.dte_wait %2 : !async.token
          %3 = wafer.instr.dte_send %alloc_4 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 720896, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 6, round = 0, slice = 1>, peer = 10 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          wafer.instr.dte_wait %3 : !async.token
          %4 = wafer.instr.dte_send %alloc_4 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 720896, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 6, round = 0, slice = 2>, peer = 14 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          wafer.instr.dte_wait %4 : !async.token
        }
      }
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

