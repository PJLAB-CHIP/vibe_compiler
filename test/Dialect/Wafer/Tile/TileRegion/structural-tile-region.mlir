// RUN: wafer-opt %s | FileCheck %s
// RUN: wafer-opt %s --mlir-print-op-generic | wafer-opt | FileCheck %s --check-prefix=ROUNDTRIP

module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @structural(%input: tensor<2x1025x128xf16>)
        -> tensor<2x1025x128xf16> {
      %result = wafer.tile.region(
          %input : tensor<2x1025x128xf16>)
          -> (tensor<2x1025x128xf16>) {
      ^bb0(%arg0: tensor<2x1025x128xf16>):
        %empty = tensor.empty() : tensor<2x1025x128xf16>
        %mapped = linalg.map ins(%arg0 : tensor<2x1025x128xf16>)
            outs(%empty : tensor<2x1025x128xf16>) (%value: f16) {
          linalg.yield %value : f16
        }
        wafer.tile.yield %mapped : tensor<2x1025x128xf16>
      }
      return %result : tensor<2x1025x128xf16>
    }
  }
}

// CHECK: wafer.tile.module card_id = 0 tile_id = 0
// CHECK: %[[RESULT:.+]] = wafer.tile.region(%{{.+}} : tensor<2x1025x128xf16>) -> (tensor<2x1025x128xf16>)
// CHECK: linalg.map
// CHECK: wafer.tile.yield %{{.+}} : tensor<2x1025x128xf16>

// ROUNDTRIP: wafer.tile.region(%{{.+}} : tensor<2x1025x128xf16>) -> (tensor<2x1025x128xf16>)
// ROUNDTRIP: wafer.tile.yield %{{.+}} : tensor<2x1025x128xf16>
