// RUN: wafer-opt %s -o %t.first
// RUN: wafer-opt %t.first -o %t.second
// RUN: diff %t.first %t.second
// RUN: FileCheck %s < %t.second

module {
  func.func @no_inputs() {
    wafer.tile.region() -> () {
      %local = memref.alloc() : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield
    }
    return
  }
  func.func @no_results(%source: memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>) {
    wafer.tile.region(%source : memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%input: memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>):
      %local = memref.alloc() : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %input into %local : memref<2x1025x64xf16, #wafer.memory<ddr, tensor>> into memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @no_inputs
// CHECK: wafer.tile.region() -> () {
// CHECK: memref.alloc() : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
// CHECK: }
// CHECK-LABEL: func.func @no_results
// CHECK: wafer.tile.region(%{{.*}} : memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>) -> () {
// CHECK: wafer.tile.load
// CHECK: }
