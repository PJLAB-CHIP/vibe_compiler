// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=1024' %s 2>&1 | FileCheck %s

func.func @overlapping_ddr_ranges_exceed_capacity() {
  %ddr0 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %ddr1 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm0 to %ddr0
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.wdma %spm1 to %ddr1
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// CHECK: memory_capacity_overflow
