module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 2048, 0] [1, 1024, 1024] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 8388608>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 8388608>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 8388608>, #wafer.memory<ddr, tensor>>):
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_0 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>>
      %0 = wafer.instr.dte_recv %alloc_0 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65536, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 2097152 : i64, message = #wafer.dte_message<communication = 9, round = 0, slice = 1>, peer = 0 : i64} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.dte_wait %0 : !async.token
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        scf.for %arg6 = %c0 to %c1024 step %c256 {
          %alloc_1 = memref.alloc() {alignment = 256 : i64, wafer.spm.offset = #wafer.spm_offset<2162688>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          wafer.instr.fill %alloc_1, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>, f32
          %c768 = arith.constant 768 : index
          %1 = scf.for %arg7 = %c0 to %c768 step %c256 iter_args(%arg8 = %alloc_1) -> (memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {
            %c2048_9 = arith.constant 2048 : index
            %6 = arith.addi %arg5, %c2048_9 : index
            %subview_10 = memref.subview %arg2[0, %6, %arg7] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_11 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2686976>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            wafer.instr.rdma %subview_10 to %alloc_11 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 2048, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %alloc_12 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2818048>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            wafer.instr.gather_scatter %alloc_11 to %alloc_12 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %alloc_13 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2949120>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %c0_14 = arith.constant 0 : index
            %c2048_15 = arith.constant 2048 : index
            %7 = arith.muli %arg7, %c2048_15 : index
            %8 = arith.addi %c0_14, %7 : index
            %c2_16 = arith.constant 2 : index
            %9 = arith.muli %arg6, %c2_16 : index
            %10 = arith.addi %8, %9 : index
            wafer.instr.gather_scatter %alloc_0 to %alloc_13 src_offset_value(%10) {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 2048, 128, 0>} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %alloc_17 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2424832>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.gemm %alloc_12, %alloc_13 psum(%arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) into %alloc_17 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 256 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.gather_scatter %alloc_17 to %arg8 {byte_count = 262144 : i64, dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>, inner_bytes = 262144 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>> to memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.instr.ncc_join [0]
            %11 = wafer.instr.dte_send %alloc_11 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 2686976, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 5, round = 0, slice = 0>, peer = 9 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            wafer.instr.dte_wait %11 : !async.token
            scf.yield %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          }
          %c2048 = arith.constant 2048 : index
          %2 = arith.addi %arg5, %c2048 : index
          %subview_2 = memref.subview %arg2[0, %2, %c768] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_3 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2424832>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.rdma %subview_2 to %alloc_3 {byte_count = 131072 : i64, inner_bytes = 512 : i64, src_iterations = array<i64: 256, 1, 1>, src_strides = array<i64: 2048, 0, 0>} : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %alloc_4 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2555904>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_3 to %alloc_4 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 512, 128, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_5 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2686976>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %c1572864 = arith.constant 1572864 : index
          %c2 = arith.constant 2 : index
          %3 = arith.muli %arg6, %c2 : index
          %4 = arith.addi %c1572864, %3 : index
          wafer.instr.gather_scatter %alloc_0 to %alloc_5 src_offset_value(%4) {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 2048, 128, 0>} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_6 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2818048>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gemm %alloc_4, %alloc_5 psum(%1 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) into %alloc_6 {batch_count = 1 : i64, k = 256 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 256 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_7 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<2949120>} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.gather_scatter %alloc_6 to %alloc_7 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 4, 1>, dst_strides = array<i64: 512, 128, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 4, 1>, src_strides = array<i64: 128, 32768, 0>} : memref<1x256x256xbf16, #wafer.memory<spm, ncx>> to memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %subview_8 = memref.subview %arg4[0, %arg5, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 8388608>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.instr.wdma %alloc_7 to %subview_8 {byte_count = 131072 : i64, dst_iterations = array<i64: 256, 1, 1>, dst_strides = array<i64: 8192, 0, 0>, inner_bytes = 512 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.instr.ncc_join [0]
          %5 = wafer.instr.dte_send %alloc_3 {binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 2424832, route_bindings = [], completion = sender_wait_receiver_fsm>, bytes = 131072 : i64, message = #wafer.dte_message<communication = 6, round = 0, slice = 0>, peer = 9 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          wafer.instr.dte_wait %5 : !async.token
        }
      }
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

