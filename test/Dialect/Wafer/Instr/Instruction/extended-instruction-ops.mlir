// RUN: wafer-opt %s | FileCheck %s

module {
  %act = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %conv_act = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x2x7x5xf16, #wafer.memory<spm, ncx>>
  %conv_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
  %pool_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %pool_idx = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
  %tensor = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
  %moved = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
  %img2col_src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x5x3xf16, #wafer.memory<spm, tensor>>
  %img2col_dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x6x12x3xf16, #wafer.memory<spm, tensor>>
  %arg_value = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %arg_index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xi32, #wafer.memory<spm, tensor>>

  wafer.instr.conv #wafer.instr_conv_kind<conv> %conv_act, %weight into %conv_out
      {input_shape = array<i64: 1, 7, 11, 5>,
       weight_shape = array<i64: 3, 2, 7, 5>,
       output_shape = array<i64: 1, 3, 5, 7>,
       pads = array<i64: 1, 0, 2, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 3, 2, 2, 3>,
       dilations = array<i64: 2, 1>}
      : memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
        memref<3x2x7x5xf16, #wafer.memory<spm, ncx>>
    into memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>

  wafer.instr.pool #wafer.instr_pool_kind<indexedmax> %act into %pool_out, %pool_idx
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 4, 4, 64>,
       pads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
         memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>

  wafer.instr.unpool #wafer.instr_unpool_kind<mask> %pool_out, %pool_idx into %act
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
        memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %tensor into %moved
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       pads = array<i64: 0, 0, 0, 0>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
     to memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>

  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<img2col> %img2col_src into %img2col_dst
      {source_shape = array<i64: 1, 4, 5, 3>,
       dest_shape = array<i64: 1, 6, 12, 3>,
       pads = array<i64: 1, 0, 2, 1>,
       kernel_strides = array<i64: 2, 3, 2, 1>}
      : memref<1x4x5x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x6x12x3xf16, #wafer.memory<spm, tensor>>

  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax> %tensor into %arg_value, %arg_index
      {elem_count = 4096 : i64}
      : memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
}

// CHECK: wafer.instr.conv <conv>
// CHECK-SAME: dilations = array<i64: 2, 1>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 2, 3>
// CHECK-SAME: weight_shape = array<i64: 3, 2, 7, 5>
// CHECK: wafer.instr.pool <indexedmax>
// CHECK: wafer.instr.unpool <mask>
// CHECK-SAME: %{{.*}}, %{{.*}} into %{{.*}}
// CHECK: wafer.instr.tdma_data_move <pad>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK: wafer.instr.tdma_data_move <img2col>
// CHECK-SAME: dest_shape = array<i64: 1, 6, 12, 3>
// CHECK-SAME: kernel_strides = array<i64: 2, 3, 2, 1>
// CHECK: wafer.instr.peripheral <argmax>
