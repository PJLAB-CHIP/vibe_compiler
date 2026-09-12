// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
       {axes = ["card_partition"],
       shape = array<i64: 1>}

  func.func @target_instr_kernel(
      %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
    %input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
    %output_tile = memref.subview %output[1, 2] [2, 3] [1, 1]
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>

    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %cx = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %gemm_rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<3x4xf16, #wafer.memory<spm, cx>>
    %gemm_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<2x4xf16, #wafer.memory<spm, cx>>
    %i32_fill = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<16xi32, #wafer.memory<spm, tensor>>
    %mask_i1 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<2x3xi1, #wafer.memory<spm, tensor>>
    %mask_fp = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67072>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %convert_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67328>}
        : memref<2x3xf32, #wafer.memory<spm, tensor>>
    %reduce_input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<524288>}
        : memref<1x1x2x3xf16, #wafer.memory<spm, ncx>>
    %reduce_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<1x1x2x1xf16, #wafer.memory<spm, ncx>>
    %pad_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67840>}
        : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
    %arg_value = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<68096>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %arg_index = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<68352>}
        : memref<1xi32, #wafer.memory<spm, tensor>>
    %conv_act = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<68608>}
        : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    %conv_input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<98304>}
        : memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>
    %conv_weight = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<131072>}
        : memref<2x3x7x5xf16, #wafer.memory<spm, cx>>
    %conv_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<262144>}
        : memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
    %pool_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<327680>}
        : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
    %pool_idx = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<344064>}
        : memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
    %img2col_src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<352256>}
        : memref<1x4x5x3xf16, #wafer.memory<spm, tensor>>
    %img2col_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<360448>}
        : memref<1x6x12x3xf16, #wafer.memory<spm, tensor>>
    %fill_value = arith.constant 7 : i32

    wafer.instr.rdma %input_tile to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_strides = array<i64: 16, 0, 0>,
         src_iterations = array<i64: 2, 1, 1>}
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %loaded to %cx
        {byte_count = 8 : i64, inner_bytes = 4 : i64,
         src_offset = 2 : i64, dst_offset = 4 : i64,
         src_strides = array<i64: 4, 0, 0>,
         src_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 4, 0, 0>,
         dst_iterations = array<i64: 2, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.instr.fill %i32_fill, %fill_value
        : memref<16xi32, #wafer.memory<spm, tensor>>, i32
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %loaded, %loaded into %loaded
        : memref<2x3xf16, #wafer.memory<spm, tensor>>,
          memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.bit2fp %mask_i1 into %mask_fp
        : memref<2x3xi1, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.mask_move %loaded, %mask_fp into %loaded
        : memref<2x3xf16, #wafer.memory<spm, tensor>>,
          memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.reduce #wafer.instr_reduce_kind<sum> %reduce_input into %reduce_out
        {dim = 0 : i64}
        : memref<1x1x2x3xf16, #wafer.memory<spm, ncx>>
      into memref<1x1x2x1xf16, #wafer.memory<spm, ncx>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %loaded into %convert_out
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.instr.gemm %cx, %gemm_rhs into %gemm_out
        {m = 2 : i64, k = 3 : i64, n = 4 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>,
          memref<3x4xf16, #wafer.memory<spm, cx>>
      into memref<2x4xf16, #wafer.memory<spm, cx>>
    wafer.instr.conv #wafer.instr_conv_kind<conv> %conv_input, %conv_weight into %conv_out
        {input_shape = array<i64: 1, 7, 11, 5>,
         weight_shape = array<i64: 2, 3, 7, 5>,
         output_shape = array<i64: 1, 3, 5, 7>,
         pads = array<i64: 1, 0, 2, 1>,
         unpads = array<i64: 0, 0, 0, 0>,
         kernel_strides = array<i64: 3, 2, 2, 3>,
         dilations = array<i64: 2, 1>}
        : memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
          memref<2x3x7x5xf16, #wafer.memory<spm, cx>>
      into memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
    wafer.instr.pool #wafer.instr_pool_kind<indexedmax> %conv_act into %pool_out, %pool_idx
        {source_shape = array<i64: 1, 8, 8, 64>,
         dest_shape = array<i64: 1, 4, 4, 64>,
         pads = array<i64: 0, 0, 0, 0>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
      into memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
           memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
    wafer.instr.unpool #wafer.instr_unpool_kind<mask> %pool_out, %pool_idx into %conv_act
        {source_shape = array<i64: 1, 4, 4, 64>,
         dest_shape = array<i64: 1, 8, 8, 64>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
          memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
      into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    wafer.instr.unpool #wafer.instr_unpool_kind<avg> %pool_out into %conv_act
        {source_shape = array<i64: 1, 4, 4, 64>,
         dest_shape = array<i64: 1, 8, 8, 64>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
      into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %loaded into %pad_out
        {source_shape = array<i64: 1, 1, 2, 3>,
         dest_shape = array<i64: 1, 1, 2, 3>,
         pads = array<i64: 0, 0, 0, 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.tdma_data_move #wafer.instr_data_move_kind<img2col> %img2col_src into %img2col_out
        {source_shape = array<i64: 1, 4, 5, 3>,
         dest_shape = array<i64: 1, 6, 12, 3>,
         pads = array<i64: 1, 0, 2, 1>,
         kernel_strides = array<i64: 2, 3, 2, 1>}
        : memref<1x4x5x3xf16, #wafer.memory<spm, tensor>>
       to memref<1x6x12x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax> %loaded into %arg_value, %arg_index
        {elem_count = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<1xf16, #wafer.memory<spm, tensor>>,
           memref<1xi32, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %loaded to %output_tile
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         dst_strides = array<i64: 16, 0, 0>,
         dst_iterations = array<i64: 2, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
    return
  }
}

// CHECK-LABEL: llvm.func @target_instr_kernel
// CHECK-SAME: (%[[DDR_IN:[a-zA-Z0-9_]+]]: i64, %[[DDR_OUT:[a-zA-Z0-9_]+]]: i64)
// CHECK-NOT: wafer.
// CHECK-NOT: memref.
// CHECK-NOT: func.func
// CHECK: %[[IN_OFF:.+]] = llvm.mlir.constant(20 : i64) : i64
// CHECK: %[[IN_ADDR:.+]] = llvm.add %[[DDR_IN]], %[[IN_OFF]] : i64
// CHECK: %[[OUT_OFF:.+]] = llvm.mlir.constant(20 : i64) : i64
// CHECK: %[[OUT_ADDR:.+]] = llvm.add %[[DDR_OUT]], %[[OUT_OFF]] : i64
// CHECK: llvm.call @wafer_tx81_rdma(%[[IN_ADDR]]
// CHECK: llvm.call @wafer_tx81_gather_scatter
// CHECK: llvm.call @wafer_tx81_memset
// CHECK: llvm.call @wafer_tx81_elementwise_add
// CHECK: llvm.call @wafer_tx81_bit2fp
// CHECK: llvm.call @wafer_tx81_mask_move
// CHECK: llvm.call @wafer_tx81_reduce_sum
// CHECK: llvm.call @wafer_tx81_convert_fp16_fp32
// CHECK: llvm.call @wafer_tx81_gemm
// CHECK: llvm.call @wafer_tx81_conv
// CHECK: llvm.call @wafer_tx81_pool_indexedmax
// CHECK: llvm.call @wafer_tx81_unpool_mask
// CHECK: llvm.call @wafer_tx81_tdma_pad
// CHECK: llvm.call @wafer_tx81_tdma_img2col
// CHECK: llvm.call @wafer_tx81_peripheral_argmax
// CHECK: llvm.call @wafer_tx81_wdma({{.*}}%[[OUT_ADDR]]
// CHECK: llvm.call @wafer_tx81_ncc_join
// CHECK: llvm.return

// LLVMIR-DAG: declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_gemm(i64, i64, i64, i32, i32, i32, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_mask_move(i64, i32, i64, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_conv(i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_tdma_img2col(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVMIR-DAG: declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVMIR-LABEL: define void @target_instr_kernel(i64 %{{.*}}, i64 %{{.*}})
// LLVMIR-NOT: call void (...)
// LLVMIR: call void @wafer_tx81_rdma(i64 %{{.*}}, i64 65536, i32 12, i32 6, i32 16, i32 0, i32 0, i32 2, i32 1, i32 1, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_gather_scatter(i64 65538, i64 65796, i32 8, i32 4, i32 4, i32 0, i32 0, i32 2, i32 1, i32 1, i32 4, i32 0, i32 0, i32 2, i32 1, i32 1, i32 0)
// LLVMIR: call void @wafer_tx81_gemm(i64 65792, i64 66048, i64 66304, i32 2, i32 3, i32 4, i32 1, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_conv(i64 98304, i64 131072, i64 262144,
// LLVMIR-SAME: i32 0, i32 1, i32 7, i32 11, i32 5,
// LLVMIR-SAME: i32 2, i32 3, i32 7, i32 5,
// LLVMIR-SAME: i32 1, i32 3, i32 5, i32 7,
// LLVMIR-SAME: i32 1, i32 0, i32 2, i32 1,
// LLVMIR-SAME: i32 0, i32 0, i32 0, i32 0,
// LLVMIR-SAME: i32 3, i32 2, i32 2, i32 3,
// LLVMIR-SAME: i32 2, i32 1, i32 2, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_pool_indexedmax(i64 68608, i64 327680,
// LLVMIR-SAME: i64 344064, i32 118,
// LLVMIR-SAME: i32 1, i32 8, i32 8, i32 64,
// LLVMIR-SAME: i32 1, i32 4, i32 4, i32 64,
// LLVMIR-SAME: i32 0, i32 0, i32 0, i32 0,
// LLVMIR-SAME: i32 2, i32 2, i32 2, i32 2, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_unpool_mask(i64 327680, i64 68608,
// LLVMIR-SAME: i32 123, i32 344064,
// LLVMIR-SAME: i32 1, i32 4, i32 4, i32 64,
// LLVMIR-SAME: i32 1, i32 8, i32 8, i32 64,
// LLVMIR-SAME: i32 2, i32 2, i32 2, i32 2, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_unpool_avg(i64 327680, i64 68608,
// LLVMIR-SAME: i32 122, i32 0,
// LLVMIR-SAME: i32 1, i32 4, i32 4, i32 64,
// LLVMIR-SAME: i32 1, i32 8, i32 8, i32 64,
// LLVMIR-SAME: i32 2, i32 2, i32 2, i32 2, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_tdma_img2col(i64 352256, i64 360448,
// LLVMIR-SAME: i32 1, i32 4, i32 5, i32 3,
// LLVMIR-SAME: i32 1, i32 6, i32 12, i32 3,
// LLVMIR-SAME: i32 1, i32 0, i32 2, i32 1,
// LLVMIR-SAME: i32 2, i32 3, i32 2, i32 1, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_wdma(i64 65536, i64 %{{.*}}, i32 12, i32 6, i32 16, i32 0, i32 0, i32 2, i32 1, i32 1, i32 2, i32 0)
