// RUN: wafer-opt %s | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967288xi1, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967288xi1, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %src to %dst
      {byte_count = 536870911 : i64, inner_bytes = 536870911 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4294967288xi1, #wafer.memory<ddr, tensor>>
     to memref<4294967288xi1, #wafer.memory<spm, tensor>>
}

// CHECK: wafer.instr.rdma
// CHECK-SAME: byte_count = 536870911 : i64
// CHECK-SAME: inner_bytes = 536870911 : i64
