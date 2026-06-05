// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  %buf = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %send0 = wafer.abi.dte_send %buf
      {bytes = 16 : i64, peer = 1 : i64, fsm_id = 0 : i64, packet_id = 0 : i64, stream_id = 0 : i64}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
  wafer.abi.dte_wait %send0 : !async.token
  %send1 = wafer.abi.dte_send %buf
      {bytes = 16 : i64, peer = 1 : i64, fsm_id = 0 : i64, packet_id = 0 : i64, stream_id = 0 : i64}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
  wafer.abi.dte_wait %send1 : !async.token
}

// CHECK: wafer.abi.dte_send
// CHECK: wafer.abi.dte_wait
// CHECK: wafer.abi.dte_send
// CHECK: wafer.abi.dte_wait
