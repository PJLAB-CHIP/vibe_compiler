// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile_region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.sync.local_drain
    %send = wafer.comm.send %buf {peer = 1 : i64, bytes = 16 : i64}
        : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
    %recv = wafer.comm.recv %buf {peer = 0 : i64, bytes = 16 : i64}
        : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
    wafer.comm.wait %send, %recv : !async.token, !async.token
    wafer.tile_yield %arg0 : tensor<4xf32>
  }
}

// CHECK: wafer.sync.local_drain
// CHECK: wafer.comm.send %{{.+}} {bytes = 16 : i64, peer = 1 : i64}
// CHECK: wafer.comm.recv %{{.+}} {bytes = 16 : i64, peer = 0 : i64}
// CHECK: wafer.comm.wait %{{.+}}, %{{.+}} : !async.token, !async.token
