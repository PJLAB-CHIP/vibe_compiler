// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-bufferize-instr-function-boundaries)' %s | FileCheck %s --check-prefix=LEAF
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-bufferize-instr-functions)' %s | FileCheck %s --check-prefix=COMPOSITE
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-plan-spm-memory)' %s | FileCheck %s --check-prefix=SPM
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-plan-ddr-memory)' %s | FileCheck %s --check-prefix=DDR

module {
  func.func @identity(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}

// LEAF-LABEL: func.func @identity(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>) -> memref<4xf32, #wafer.memory<ddr, tensor>>
// LEAF-NEXT: return %arg0
// COMPOSITE-LABEL: func.func @identity(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>) {
// COMPOSITE-NEXT: return
// SPM-LABEL: func.func @identity(%arg0: tensor<4xf32>) -> tensor<4xf32>
// SPM-NEXT: return %arg0
// DDR-LABEL: func.func @identity(%arg0: tensor<4xf32>) -> tensor<4xf32>
// DDR-NEXT: return %arg0
