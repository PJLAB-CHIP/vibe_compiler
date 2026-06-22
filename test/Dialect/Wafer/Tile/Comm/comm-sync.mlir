// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.local_drain
    %send = wafer.tile.send %buf {peer = 1 : i64, bytes = 16 : i64}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %recv = wafer.tile.recv %buf {peer = 0 : i64, bytes = 16 : i64}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.tile.wait %send, %recv : !async.token, !async.token
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: wafer.instr.local_drain
// CHECK: wafer.tile.send %{{.+}} {bytes = 16 : i64, peer = 1 : i64}
// CHECK: wafer.tile.recv %{{.+}} {bytes = 16 : i64, peer = 0 : i64}
// CHECK: wafer.tile.wait %{{.+}}, %{{.+}} : !async.token, !async.token
