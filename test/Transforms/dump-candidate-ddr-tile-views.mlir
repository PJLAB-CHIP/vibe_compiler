// RUN: wafer-opt --wafer-dump-candidate-ddr-tile-views='candidate-tile-offsets=1,2 candidate-tile-sizes=2,3' %s 2>&1 | FileCheck %s

module {
  func.func @candidate_tiled_matmul(%lhs: tensor<4x8xf16>,
                                    %rhs: tensor<8x8xf16>,
                                    %out: tensor<4x8xf16>)
      -> tensor<4x8xf16> {
    %group = wafer.group ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x8xf16>)
        outs(%out : tensor<4x8xf16>) {
    ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x8xf16>,
         %arg2: tensor<4x8xf16>):
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<8x8xf16>)
          outs(%arg2 : tensor<4x8xf16>) -> tensor<4x8xf16>
      wafer.group.yield %mm : tensor<4x8xf16>
    } : tensor<4x8xf16>
    return %group : tensor<4x8xf16>
  }
}

// CHECK-LABEL: wafer.candidate_ddr_tile_views group @candidate_tiled_matmul#0
// CHECK: offsets=[1,2] sizes=[2,3]
// CHECK: %[[LHS_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 0] [2, 8] [1, 1]
// CHECK-SAME: memref<2x8xf16, strided<[8, 1], offset: 8>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load %[[LHS_TILE]]
// CHECK: %[[RHS_TILE:.+]] = memref.subview
// CHECK-SAME: [0, 2] [8, 3] [1, 1]
// CHECK-SAME: memref<8x3xf16, strided<[8, 1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load %[[RHS_TILE]]
// CHECK: %[[OUT_INIT_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 2] [2, 3] [1, 1]
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load %[[OUT_INIT_TILE]]
// CHECK: %[[OUT_STORE_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 2] [2, 3] [1, 1]
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.store {{%.*}}, %[[OUT_STORE_TILE]]
