// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
  %send = wafer.comm.send %buf {peer = 1 : i64, bytes = 16 : i64}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>> -> !async.token
}

// CHECK: comm p2p buffer must use SPM memory space
