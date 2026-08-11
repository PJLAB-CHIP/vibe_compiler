// RUN: wafer-opt %s | FileCheck %s
// RUN: wafer-opt --canonicalize %s | FileCheck %s --check-prefix=CANONICAL

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.region(
      %source : memref<4xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    %send = wafer.instr.dte_send %buf {peer = 1 : i64, bytes = 16 : i64,
        message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %recv = wafer.instr.dte_recv %buf {peer = 0 : i64, bytes = 16 : i64,
        message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    wafer.tile.yield %arg0
        : memref<4xf32, #wafer.memory<ddr, tensor>>
  }
}

// CHECK: wafer.instr.ncc_join [0]
// CHECK: wafer.instr.dte_send %{{.+}} {bytes = 16 : i64, message = #wafer.dte_message<communication = 7, round = 0, slice = 0>, peer = 1 : i64}
// CHECK: wafer.instr.dte_recv %{{.+}} {bytes = 16 : i64, message = #wafer.dte_message<communication = 7, round = 0, slice = 0>, peer = 0 : i64}
// CHECK: wafer.instr.dte_wait %{{.+}}, %{{.+}} : !async.token, !async.token

// CANONICAL: wafer.instr.dte_send
// CANONICAL: wafer.instr.dte_recv
// CANONICAL: wafer.instr.dte_wait
