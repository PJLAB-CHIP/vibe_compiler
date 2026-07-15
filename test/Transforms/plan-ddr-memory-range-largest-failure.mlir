// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=1024 ddr-largest-contiguous-bytes=128' %s 2>&1 | FileCheck %s

func.func @single_ddr_range_exceeds_largest_contiguous() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm to %ddr
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// CHECK: largest_contiguous_range_too_small
