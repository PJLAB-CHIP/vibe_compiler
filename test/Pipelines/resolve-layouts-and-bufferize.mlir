// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-resolve-current-layouts-and-bufferize)' %s | FileCheck %s --check-prefix=LEAF
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-resolve-layouts-and-bufferize)' %s | FileCheck %s --check-prefix=COMPOSITE
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-plan-spm-memory)' %s | FileCheck %s --check-prefix=SPM
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-plan-ddr-memory)' %s | FileCheck %s --check-prefix=DDR

module {
  func.func @identity(%arg0: tensor<2x1024x64xf16>)
      -> tensor<2x1024x64xf16> {
    return %arg0 : tensor<2x1024x64xf16>
  }
}

// LEAF-LABEL: func.func @identity(%arg0: memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>) -> memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
// LEAF-NEXT: return %arg0
// COMPOSITE-LABEL: func.func @identity(%arg0: memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>) -> memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
// COMPOSITE-NEXT: return %arg0
// SPM-LABEL: func.func @identity(%arg0: tensor<2x1024x64xf16>) -> tensor<2x1024x64xf16>
// SPM-NEXT: return %arg0
// DDR-LABEL: func.func @identity(%arg0: tensor<2x1024x64xf16>) -> tensor<2x1024x64xf16>
// DDR-NEXT: return %arg0
