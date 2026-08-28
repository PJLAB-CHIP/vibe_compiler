// RUN: split-file %s %t
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/ddr.mlir 2>&1 | FileCheck --check-prefix=DDR-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=256' %t/ddr.mlir | FileCheck --check-prefix=DDR %s
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/spm.mlir 2>&1 | FileCheck --check-prefix=SPM-CAPACITY %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %t/spm.mlir | FileCheck --check-prefix=SPM %s

// bufferization.to_tensor/to_memref may have structured tensor control flow
// between them. Provenance follows the tensor operand recursively rather than
// relying on an adjacent round-trip pattern.

//--- ddr.mlir
func.func @ddr_bufferization_alias_through_if(%condition: i1) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %a = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tensor = bufferization.to_tensor %a restrict writable
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %selected = scf.if %condition -> (tensor<128xf16>) {
    scf.yield %tensor : tensor<128xf16>
  } else {
    scf.yield %tensor : tensor<128xf16>
  }
  %alias = bufferization.to_memref %selected : memref<128xf16>
  %b = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %b[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %late = memref.load %alias[%c0] : memref<128xf16>
  return
}

// DDR-CAPACITY: memory_capacity_overflow
// DDR-LABEL: func.func @ddr_bufferization_alias_through_if
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR: bufferization.to_tensor
// DDR: scf.if
// DDR: bufferization.to_memref
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// DDR: memref.load

//--- spm.mlir
func.func @spm_bufferization_alias_through_if(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %condition: i1) {
  %result = wafer.tile.region(%boundary, %condition
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %a = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %tensor = bufferization.to_tensor %a restrict writable
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %selected = scf.if %cond -> (tensor<128xf16>) {
      scf.yield %tensor : tensor<128xf16>
    } else {
      scf.yield %tensor : tensor<128xf16>
    }
    %alias = bufferization.to_memref %selected : memref<128xf16>
    %b = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    memref.store %one, %b[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %late = memref.load %alias[%c0] : memref<128xf16>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-CAPACITY: capacity_overflow
// SPM-LABEL: func.func @spm_bufferization_alias_through_if
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65536>
// SPM: bufferization.to_tensor
// SPM: scf.if
// SPM: bufferization.to_memref
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>
// SPM: memref.load
