// RUN: split-file %s %t
// RUN: not wafer-opt --mlir-disable-threading --wafer-plan-spm-memory --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %t/spm.mlir 2>&1 | FileCheck %s --check-prefix=SPM --implicit-check-not=wafer.spm.offset
// RUN: not wafer-opt --mlir-disable-threading --wafer-plan-ddr-memory --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %t/ddr.mlir 2>&1 | FileCheck %s --check-prefix=DDR --implicit-check-not=wafer.ddr.offset
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-mixed-scope.mlir 2>&1 | FileCheck %s --check-prefix=DDR-MIXED

//--- spm.mlir
module {
  func.func @spm_early_success(
      %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<128xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
      %planned = memref.alloc()
          : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<128xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }

  func.func @spm_late_failure(
      %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %size: index) {
    %region = wafer.tile.region(%boundary, %size
        : memref<128xf16, #wafer.memory<ddr, tensor>>, index)
        -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %arg1: index):
      %invalid = memref.alloc(%arg1)
          : memref<?xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<128xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// SPM: unsupported_layout_conversion: SPM memory planning requires static memref.alloc sizes and symbols
// SPM: func.func @spm_early_success
// SPM: %[[PLANNED:.+]] = memref.alloc()
// SPM: func.func @spm_late_failure

//--- ddr.mlir
module {
  func.func @ddr_early_success() {
    %planned = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    return
  }

  func.func @ddr_late_failure(%size: index) {
    %invalid = memref.alloc(%size)
        : memref<?xf16, #wafer.memory<ddr, tensor>>
    return
  }
}

// DDR: unsupported_compiler_managed_ddr: DDR memory planning requires static memref.alloc sizes and symbols
// DDR: func.func @ddr_early_success
// DDR: %[[PLANNED:.+]] = memref.alloc()
// DDR: func.func @ddr_late_failure

//--- ddr-mixed-scope.mlir
module {
  %top_level = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>

  func.func @entry() {
    return
  }
}

// DDR-MIXED: unsupported_ddr_planning_scope
