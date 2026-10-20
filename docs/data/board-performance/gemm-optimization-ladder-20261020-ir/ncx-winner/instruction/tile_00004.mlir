module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>> {
    %alloc = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>
    wafer.tile.region(%arg1, %arg0, %alloc : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>>, memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>>, %arg4: memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>):
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg5 = %c0 to %c4096 step %c1024 {
        %alloc_0 = memref.alloc() {alignment = 256 : i64, wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
        wafer.instr.fill %alloc_0, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>, f32
        %c768 = arith.constant 768 : index
        %0 = scf.for %arg6 = %c0 to %c768 step %c256 iter_args(%arg7 = %alloc_0) -> (memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) {
          %alloc_5 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2686976>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %c64_6 = arith.constant 64 : index
          %6 = arith.divui %arg6, %c64_6 : index
          %c524288_7 = arith.constant 524288 : index
          %7 = arith.muli %6, %c524288_7 : index
          %c131072_8 = arith.constant 131072 : index
          %8 = arith.addi %7, %c131072_8 : index
          wafer.instr.rdma %arg2 to %alloc_5 src_offset_value(%8) {byte_count = 131072 : i64, dst_offset = 0 : i64, inner_bytes = 32768 : i64, src_iterations = array<i64: 4, 1, 1>, src_strides = array<i64: 524288, 0, 0>} : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_9 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2162688>} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
          %alloc_10 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1114112>} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
          wafer.instr.ncc_join [0]
          %9 = wafer.instr.dte_recv %alloc_9 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 2162688, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 524288 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 6>, peer = 5 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
          wafer.instr.dte_wait %9 : !async.token
          wafer.instr.gemm %alloc_5, %alloc_9 psum(%arg7 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) into %alloc_10 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 1024 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> into memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_10 to %arg7 {byte_count = 1048576 : i64, dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>, inner_bytes = 1048576 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>> to memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
          wafer.instr.ncc_join [0]
          %10 = wafer.instr.dte_send %alloc_9 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 2162688, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 524288 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 7>, peer = 8 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
          wafer.instr.dte_wait %10 : !async.token
          scf.yield %arg7 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
        }
        %alloc_1 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2686976>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
        wafer.instr.rdma %arg2 to %alloc_1 {byte_count = 131072 : i64, dst_offset = 0 : i64, inner_bytes = 32768 : i64, src_iterations = array<i64: 4, 1, 1>, src_offset = 6422528 : i64, src_strides = array<i64: 524288, 0, 0>} : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
        %alloc_2 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1114112>} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
        %alloc_3 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1638400>} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
        wafer.instr.ncc_join [0]
        %1 = wafer.instr.dte_recv %alloc_2 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 1114112, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 524288 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 6>, peer = 5 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
        wafer.instr.dte_wait %1 : !async.token
        wafer.instr.gemm %alloc_1, %alloc_2 psum(%0 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) into %alloc_3 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 1024 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> into memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
        %alloc_4 = memref.alloc() {alignment = 256 : i64, wafer.spm.offset = #wafer.spm_offset<2162688>} : memref<1x256x1024xbf16, #wafer.memory<spm, cx>>
        wafer.instr.gather_scatter %alloc_3 to %alloc_4 {byte_count = 524288 : i64, dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>, inner_bytes = 524288 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> to memref<1x256x1024xbf16, #wafer.memory<spm, cx>>
        %c64 = arith.constant 64 : index
        %2 = arith.divui %arg5, %c64 : index
        %c524288 = arith.constant 524288 : index
        %3 = arith.muli %2, %c524288 : index
        %c131072 = arith.constant 131072 : index
        %4 = arith.addi %3, %c131072 : index
        wafer.instr.wdma %alloc_4 to %arg4 dst_offset_value(%4) {byte_count = 524288 : i64, dst_iterations = array<i64: 16, 1, 1>, dst_strides = array<i64: 524288, 0, 0>, inner_bytes = 32768 : i64, src_offset = 0 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, cx>> to memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>
        wafer.instr.ncc_join [0]
        %5 = wafer.instr.dte_send %alloc_2 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 1114112, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 524288 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 7>, peer = 8 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
        wafer.instr.dte_wait %5 : !async.token
      }
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>
  }
}

