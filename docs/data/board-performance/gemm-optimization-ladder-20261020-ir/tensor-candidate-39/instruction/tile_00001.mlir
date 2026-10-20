module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 256, 0] [1, 256, 4096] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 1048576>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 1048576>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 1048576>, #wafer.memory<ddr, tensor>>):
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_0 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>>
      %subview_1 = memref.subview %arg2[0, 256, 0] [1, 256, 1024] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x1024xbf16, strided<[4194304, 1024, 1], offset: 262144>, #wafer.memory<ddr, tensor>>
      wafer.instr.rdma %subview_1 to %alloc_0 {byte_count = 524288 : i64, inner_bytes = 524288 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x1024xbf16, strided<[4194304, 1024, 1], offset: 262144>, #wafer.memory<ddr, tensor>> to memref<1x256x1024xbf16, #wafer.memory<spm, tensor>>
      scf.for %arg5 = %c0 to %c4096 step %c512 {
        %alloc_2 = memref.alloc() {alignment = 256 : i64, wafer.spm.offset = #wafer.spm_offset<589824>} : memref<1x256x512xf32, #wafer.memory<spm, ncx>>
        wafer.instr.fill %alloc_2, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x512xf32, #wafer.memory<spm, ncx>>, f32
        %c896 = arith.constant 896 : index
        %0 = scf.for %arg6 = %c0 to %c896 step %c128 iter_args(%arg7 = %alloc_2) -> (memref<1x256x512xf32, #wafer.memory<spm, ncx>>) {
          %subview_10 = memref.subview %arg3[0, %arg6, %arg5] [1, 128, 512] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_11 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1638400>} : memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
          wafer.instr.rdma %subview_10 to %alloc_11 {byte_count = 131072 : i64, inner_bytes = 1024 : i64, src_iterations = array<i64: 128, 1, 1>, src_strides = array<i64: 8192, 0, 0>} : memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
          %alloc_12 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1900544>} : memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
          %c0_13 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %1 = arith.muli %arg6, %c2 : index
          %2 = arith.addi %c0_13, %1 : index
          wafer.instr.gather_scatter %alloc_0 to %alloc_12 src_offset_value(%2) {byte_count = 65536 : i64, dst_iterations = array<i64: 256, 2, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 2, 1>, src_strides = array<i64: 2048, 128, 0>} : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
          %alloc_14 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1769472>} : memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_11 to %alloc_14 {byte_count = 131072 : i64, dst_iterations = array<i64: 128, 8, 1>, dst_strides = array<i64: 128, 16384, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 128, 8, 1>, src_strides = array<i64: 1024, 128, 0>} : memref<1x128x512xbf16, #wafer.memory<spm, tensor>> to memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
          %alloc_15 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1114112>} : memref<1x256x512xf32, #wafer.memory<spm, ncx>>
          wafer.instr.gemm %alloc_12, %alloc_14 psum(%arg7 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>) into %alloc_15 {batch_count = 1 : i64, k = 128 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 512 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x128xbf16, #wafer.memory<spm, ncx>>, memref<1x128x512xbf16, #wafer.memory<spm, ncx>> into memref<1x256x512xf32, #wafer.memory<spm, ncx>>
          wafer.instr.gather_scatter %alloc_15 to %arg7 {byte_count = 524288 : i64, dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>, inner_bytes = 524288 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>} : memref<1x256x512xf32, #wafer.memory<spm, ncx>> to memref<1x256x512xf32, #wafer.memory<spm, ncx>>
          scf.yield %arg7 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>
        }
        %subview_3 = memref.subview %arg3[0, %c896, %arg5] [1, 128, 512] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        %alloc_4 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1638400>} : memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
        wafer.instr.rdma %subview_3 to %alloc_4 {byte_count = 131072 : i64, inner_bytes = 1024 : i64, src_iterations = array<i64: 128, 1, 1>, src_strides = array<i64: 8192, 0, 0>} : memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
        %alloc_5 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1507328>} : memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
        %c1792 = arith.constant 1792 : index
        wafer.instr.gather_scatter %alloc_0 to %alloc_5 src_offset_value(%c1792) {byte_count = 65536 : i64, dst_iterations = array<i64: 256, 2, 1>, dst_strides = array<i64: 128, 32768, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 2, 1>, src_strides = array<i64: 2048, 128, 0>} : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
        %alloc_6 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1376256>} : memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
        wafer.instr.gather_scatter %alloc_4 to %alloc_6 {byte_count = 131072 : i64, dst_iterations = array<i64: 128, 8, 1>, dst_strides = array<i64: 128, 16384, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 128, 8, 1>, src_strides = array<i64: 1024, 128, 0>} : memref<1x128x512xbf16, #wafer.memory<spm, tensor>> to memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
        %alloc_7 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1114112>} : memref<1x256x512xbf16, #wafer.memory<spm, ncx>>
        wafer.instr.gemm %alloc_5, %alloc_6 psum(%0 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>) into %alloc_7 {batch_count = 1 : i64, k = 128 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, m = 256 : i64, n = 512 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : memref<1x256x128xbf16, #wafer.memory<spm, ncx>>, memref<1x128x512xbf16, #wafer.memory<spm, ncx>> into memref<1x256x512xbf16, #wafer.memory<spm, ncx>>
        %alloc_8 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<1572864>} : memref<1x256x512xbf16, #wafer.memory<spm, tensor>>
        wafer.instr.gather_scatter %alloc_7 to %alloc_8 {byte_count = 262144 : i64, dst_iterations = array<i64: 256, 8, 1>, dst_strides = array<i64: 1024, 128, 0>, inner_bytes = 128 : i64, src_iterations = array<i64: 256, 8, 1>, src_strides = array<i64: 128, 32768, 0>} : memref<1x256x512xbf16, #wafer.memory<spm, ncx>> to memref<1x256x512xbf16, #wafer.memory<spm, tensor>>
        %subview_9 = memref.subview %arg4[0, 0, %arg5] [1, 256, 512] [1, 1, 1] : memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 1048576>, #wafer.memory<ddr, tensor>> to memref<1x256x512xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        wafer.instr.wdma %alloc_8 to %subview_9 {byte_count = 262144 : i64, dst_iterations = array<i64: 256, 1, 1>, dst_strides = array<i64: 8192, 0, 0>, inner_bytes = 1024 : i64} : memref<1x256x512xbf16, #wafer.memory<spm, tensor>> to memref<1x256x512xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      }
    }
    wafer.instr.ncc_join [0]
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

