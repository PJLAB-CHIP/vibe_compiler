// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66560' %t/spm-pressure.mlir | FileCheck %s --check-prefix=SPM-PRESSURE
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=67840' %t/spm-search.mlir | FileCheck %s --check-prefix=SPM-SEARCH
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=2304 ddr-largest-contiguous-bytes=1536' %t/ddr-search.mlir | FileCheck %s --check-prefix=DDR-SEARCH
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=67840' %t/spm-infeasible.mlir 2>&1 | FileCheck %s --check-prefix=SPM-INFEASIBLE
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=2304 ddr-largest-contiguous-bytes=1536' %t/ddr-infeasible.mlir 2>&1 | FileCheck %s --check-prefix=DDR-INFEASIBLE

// Keep the established pressure-aware packing coverage independently of the
// selected static-packing algorithm.

//--- spm-pressure.mlir
func.func @packing_avoids_fragmentation_under_pressure(
    %boundary: memref<256xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<256xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<256xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<256xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %early_dead = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %left = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %right = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %early_dead, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    %large = memref.alloc() : memref<256xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %left, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %right, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %large, %zero
        : memref<256xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<256xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-PRESSURE-LABEL: func.func @packing_avoids_fragmentation_under_pressure
// SPM-PRESSURE: %[[EARLY:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>} : memref<128xf16, #wafer.memory<spm, tensor>>
// SPM-PRESSURE: %[[LEFT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// SPM-PRESSURE: %[[RIGHT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// SPM-PRESSURE: wafer.instr.fill %[[EARLY]]
// SPM-PRESSURE: %[[LARGE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>} : memref<256xf16, #wafer.memory<spm, tensor>>
// SPM-PRESSURE: wafer.instr.fill %[[LEFT]]
// SPM-PRESSURE: wafer.instr.fill %[[RIGHT]]
// SPM-PRESSURE: wafer.instr.fill %[[LARGE]]

// The conflict graph below is a path D-B-C-A. A deterministic first-fit order
// D,A,C,B cannot fit B in nine 256-byte units, while the checked placement
// D@0, B@6, C@0, A@4 is legal and exactly fills the arena. The SPM and DDR
// cases deliberately carry the same sizes and use order so both planners must
// consume the shared static-packing solver rather than report capacity.

//--- spm-search.mlir
func.func @spm_static_search_recovers_legal_placement(
    %boundary: memref<1xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<1xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<1xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %d = memref.alloc() : memref<768xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<256xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %d, %zero
        : memref<768xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    %c = memref.alloc() : memref<512xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %b, %zero
        : memref<256xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    %a = memref.alloc() : memref<640xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %c, %zero
        : memref<512xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %a, %zero
        : memref<640xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %out : memref<1xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-SEARCH-LABEL: func.func @spm_static_search_recovers_legal_placement
// SPM-SEARCH: %[[SPM_D:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<768xf16, #wafer.memory<spm, tensor>>
// SPM-SEARCH: %[[SPM_B:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67072>} : memref<256xf16, #wafer.memory<spm, tensor>>
// SPM-SEARCH: wafer.instr.fill %[[SPM_D]]
// SPM-SEARCH: %[[SPM_C:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<512xf16, #wafer.memory<spm, tensor>>
// SPM-SEARCH: wafer.instr.fill %[[SPM_B]]
// SPM-SEARCH: %[[SPM_A:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>} : memref<640xf16, #wafer.memory<spm, tensor>>
// SPM-SEARCH: wafer.instr.fill %[[SPM_C]]
// SPM-SEARCH: wafer.instr.fill %[[SPM_A]]

//--- ddr-search.mlir
func.func @ddr_static_search_recovers_legal_placement() {
  %c0 = arith.constant 0 : index
  %d = memref.alloc() : memref<768xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<256xf16, #wafer.memory<ddr, tensor>>
  %d_value = memref.load %d[%c0]
      : memref<768xf16, #wafer.memory<ddr, tensor>>
  %c = memref.alloc() : memref<512xf16, #wafer.memory<ddr, tensor>>
  %b_value = memref.load %b[%c0]
      : memref<256xf16, #wafer.memory<ddr, tensor>>
  %a = memref.alloc() : memref<640xf16, #wafer.memory<ddr, tensor>>
  %c_value = memref.load %c[%c0]
      : memref<512xf16, #wafer.memory<ddr, tensor>>
  %a_value = memref.load %a[%c0]
      : memref<640xf16, #wafer.memory<ddr, tensor>>
  return
}

// DDR-SEARCH-LABEL: func.func @ddr_static_search_recovers_legal_placement
// DDR-SEARCH: %[[DDR_D:.+]] = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<768xf16, #wafer.memory<ddr, tensor>>
// DDR-SEARCH: %[[DDR_B:.+]] = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<1536>} : memref<256xf16, #wafer.memory<ddr, tensor>>
// DDR-SEARCH: memref.load %[[DDR_D]]
// DDR-SEARCH: %[[DDR_C:.+]] = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<512xf16, #wafer.memory<ddr, tensor>>
// DDR-SEARCH: memref.load %[[DDR_B]]
// DDR-SEARCH: %[[DDR_A:.+]] = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<1024>} : memref<640xf16, #wafer.memory<ddr, tensor>>
// DDR-SEARCH: memref.load %[[DDR_C]]
// DDR-SEARCH: memref.load %[[DDR_A]]

// Two simultaneously live 1536-byte buffers have a 3072-byte lower bound, so
// a complete solver must still diagnose the 2304-byte arena as infeasible.

//--- spm-infeasible.mlir
func.func @spm_proven_capacity_infeasible(
    %boundary: memref<1xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<1xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<1xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %lhs = memref.alloc() : memref<768xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<768xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %lhs, %zero
        : memref<768xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %rhs, %zero
        : memref<768xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %out : memref<1xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-INFEASIBLE: capacity_overflow

//--- ddr-infeasible.mlir
func.func @ddr_proven_capacity_infeasible() {
  %c0 = arith.constant 0 : index
  %lhs = memref.alloc() : memref<768xf16, #wafer.memory<ddr, tensor>>
  %rhs = memref.alloc() : memref<768xf16, #wafer.memory<ddr, tensor>>
  %lhs_value = memref.load %lhs[%c0]
      : memref<768xf16, #wafer.memory<ddr, tensor>>
  %rhs_value = memref.load %rhs[%c0]
      : memref<768xf16, #wafer.memory<ddr, tensor>>
  return
}

// DDR-INFEASIBLE: memory_capacity_overflow
