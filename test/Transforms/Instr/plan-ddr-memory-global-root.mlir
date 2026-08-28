// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=512' %s | FileCheck %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=128 ddr-largest-contiguous-bytes=512' %s 2>&1 | FileCheck --check-prefix=CAPACITY %s

memref.global "private" constant @weights
    : memref<128xf16, #wafer.memory<ddr, tensor>> = dense<1.0>

func.func @read_static_constant_global() {
  %weights = memref.get_global @weights
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tile = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %weights to %tile
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK: memref.global "private" constant @weights
// CHECK-LABEL: func.func @read_static_constant_global
// CHECK: %[[WEIGHTS:.+]] = memref.get_global @weights
// CHECK: wafer.instr.rdma %[[WEIGHTS]]
// CHECK-NOT: wafer.ddr.offset

// CAPACITY: memory_capacity_overflow
