// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %send = wafer.instr.dte_send %buf {peer = 1 : i64, bytes = 16 : i64,
        message = #wafer.dte_message<communication = 7, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %recv = wafer.instr.dte_recv %buf {peer = 0 : i64, bytes = 16 : i64,
        message = #wafer.dte_message<communication = 7, phase = collective_permute, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: wafer.instr.local_fence
// CHECK: wafer.instr.dte_send %{{.+}} {bytes = 16 : i64, message = #wafer.dte_message<communication = 7, phase = collective_permute, round = 0, slice = 0>, peer = 1 : i64}
// CHECK: wafer.instr.dte_recv %{{.+}} {bytes = 16 : i64, message = #wafer.dte_message<communication = 7, phase = collective_permute, round = 0, slice = 0>, peer = 0 : i64}
// CHECK: wafer.instr.dte_wait %{{.+}}, %{{.+}} : !async.token, !async.token
