// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @region_boundary(
      %source: memref<4xf32, #wafer.memory<ddr, tensor>>, %flag: i1)
      -> memref<4xf32, #wafer.memory<ddr, tensor>> {
    %forwarded, %condition = wafer.tile.region(%source, %flag
        : memref<4xf32, #wafer.memory<ddr, tensor>>, i1) ->
        (memref<4xf32, #wafer.memory<ddr, tensor>>, i1) {
    ^bb0(%boundary: memref<4xf32, #wafer.memory<ddr, tensor>>,
         %control: i1):
      // SPM storage is owned and consumed entirely inside this residency
      // domain; only DDR data and typed control cross its boundary.
    %local = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %boundary, %control
          : memref<4xf32, #wafer.memory<ddr, tensor>>, i1
    }
    return %forwarded : memref<4xf32, #wafer.memory<ddr, tensor>>
  }
}

// CHECK: func.func @region_boundary
// CHECK: %[[FORWARDED:.+]]:2 = wafer.tile.region(%{{.+}}, %{{.+}} : memref<4xf32, #wafer.memory<ddr, tensor>>, i1) -> (memref<4xf32, #wafer.memory<ddr, tensor>>, i1) {
// CHECK: %[[LOCAL:.+]] = memref.alloc() : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.yield %{{.+}}, %{{.+}} : memref<4xf32, #wafer.memory<ddr, tensor>>, i1
