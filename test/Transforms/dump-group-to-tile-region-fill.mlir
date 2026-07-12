// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @fill_preserves_old_init(%out0: tensor<4xi32>,
                                   %out1: tensor<4xi32>)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %group:2 = wafer.group ins() outs(%out0, %out1
      : tensor<4xi32>, tensor<4xi32>) {
  ^bb0(%out0_arg: tensor<4xi32>, %out1_arg: tensor<4xi32>):
    %zero = arith.constant 0 : i32
    %filled = linalg.fill ins(%zero : i32)
        outs(%out0_arg : tensor<4xi32>) -> tensor<4xi32>
    %copy = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%out0_arg : tensor<4xi32>)
        outs(%out1_arg : tensor<4xi32>) {
      ^bb0(%value: i32, %init: i32):
        linalg.yield %value : i32
      } -> tensor<4xi32>
    wafer.group.yield %filled, %copy : tensor<4xi32>, tensor<4xi32>
  } : tensor<4xi32>, tensor<4xi32>
  return %group#0, %group#1 : tensor<4xi32>, tensor<4xi32>
}

func.func @insert_preserves_old_dest(%patch: tensor<2xi32>,
                                     %out0: tensor<4xi32>,
                                     %out1: tensor<4xi32>)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %group:2 = wafer.group ins(%patch : tensor<2xi32>)
      outs(%out0, %out1 : tensor<4xi32>, tensor<4xi32>) {
  ^bb0(%patch_arg: tensor<2xi32>, %out0_arg: tensor<4xi32>,
       %out1_arg: tensor<4xi32>):
    %updated = tensor.insert_slice %patch_arg into %out0_arg[0] [2] [1]
        : tensor<2xi32> into tensor<4xi32>
    %copy = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%out0_arg : tensor<4xi32>)
        outs(%out1_arg : tensor<4xi32>) {
      ^bb0(%value: i32, %init: i32):
        linalg.yield %value : i32
      } -> tensor<4xi32>
    wafer.group.yield %updated, %copy : tensor<4xi32>, tensor<4xi32>
  } : tensor<4xi32>, tensor<4xi32>
  return %group#0, %group#1 : tensor<4xi32>, tensor<4xi32>
}

// CHECK: wafer.group_to_tile_region group @fill_preserves_old_init#0
// CHECK: %[[FILLED:.+]] = memref.alloc
// CHECK: wafer.tile.fill %[[FILLED]],
// CHECK: %[[ORIGINAL:.+]] = wafer.tile.load
// CHECK: %[[COPY:.+]] = wafer.tile.copy %[[ORIGINAL]]
// CHECK-NOT: wafer.tile.copy %[[FILLED]]
// CHECK: wafer.tile.store %[[FILLED]],
// CHECK: wafer.tile.store %[[COPY]],
// CHECK: wafer.group_to_tile_region group @insert_preserves_old_dest#0
// CHECK: %[[ORIGINAL:.+]] = wafer.tile.load {{%.*}} : memref<4xi32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.tile.store
// CHECK: %[[UPDATED:.+]] = wafer.tile.insert_slice {{%.*}} into %[[ORIGINAL]]
// CHECK: %[[OLD_COPY:.+]] = wafer.tile.copy %[[ORIGINAL]]
// CHECK: wafer.tile.store %[[UPDATED]],
// CHECK: wafer.tile.store %[[OLD_COPY]],
