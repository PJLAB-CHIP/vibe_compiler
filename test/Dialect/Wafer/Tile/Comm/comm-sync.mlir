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
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.instr.local_drain
    %send = wafer.tile.send %buf {peer = 1 : i64, bytes = 16 : i64}
        : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
    %recv = wafer.tile.recv %buf {peer = 0 : i64, bytes = 16 : i64}
        : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
    wafer.tile.wait %send, %recv : !async.token, !async.token
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: wafer.instr.local_drain
// CHECK: wafer.tile.send %{{.+}} {bytes = 16 : i64, peer = 1 : i64}
// CHECK: wafer.tile.recv %{{.+}} {bytes = 16 : i64, peer = 0 : i64}
// CHECK: wafer.tile.wait %{{.+}}, %{{.+}} : !async.token, !async.token
