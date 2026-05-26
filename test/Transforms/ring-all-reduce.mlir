// RUN: wafer-opt --wafer-lower-ring-reduce-collectives %s | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 4 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1,
                                      0, 0, 0, 2,
                                      0, 0, 0, 3>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %recv = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %result = wafer.comm.all_reduce #wafer.reduce_kind<max> %input using %recv
      {bytes = 16 : i64, group_size = 4 : i64, local_rank = 2 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
      : (!wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK-LABEL: module
// CHECK-NOT: wafer.comm.all_reduce
// CHECK: wafer.comm.send %{{.*}} {bytes = 16 : i64, peer = 3 : i64, slot = 2 : i64}
// CHECK: wafer.comm.recv %{{.*}} {bytes = 16 : i64, peer = 1 : i64, slot = 2 : i64}
// CHECK: wafer.comm.wait %{{.*}}, %{{.*}} : !async.token, !async.token
// CHECK: wafer.compute.elementwise <max> %{{.*}}, %{{.*}}
// CHECK: wafer.compute.elementwise <max> %{{.*}}, %{{.*}}
// CHECK: wafer.compute.elementwise <max> %{{.*}}, %{{.*}}
