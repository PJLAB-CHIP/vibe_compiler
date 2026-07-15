// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=67584 spm-alignment=384' %t/spm.mlir | FileCheck %s --check-prefix=SPM
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-alignment-bytes=384 ddr-capacity-bytes=2048 ddr-largest-contiguous-bytes=2048' %t/ddr.mlir | FileCheck %s --check-prefix=DDR

//--- spm.mlir
func.func @spm_non_dividing_alignment(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %aligned = memref.alloc() {alignment = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-LABEL: func.func @spm_non_dividing_alignment
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<66048>

//--- ddr.mlir
func.func @ddr_non_dividing_alignment() {
  %large = memref.alloc()
      : memref<512xf16, #wafer.memory<ddr, tensor>>
  %aligned = memref.alloc() {alignment = 256 : i64}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %large_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<512xf16, #wafer.memory<spm, tensor>>
  %aligned_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<66560>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %large_spm to %large
      {byte_count = 1024 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 1024 : i64}
      : memref<512xf16, #wafer.memory<spm, tensor>>
     to memref<512xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.wdma %aligned_spm to %aligned
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// DDR-LABEL: func.func @ddr_non_dividing_alignment
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<1536>
