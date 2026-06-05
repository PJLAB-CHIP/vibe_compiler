// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-tile-region)' %s | FileCheck %s

func.func @add_group(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                     %out: tensor<4xf32>) -> tensor<4xf32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
        outs(%arg2 : tensor<4xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<4xf32>
    wafer.group.yield %sum : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @two_chained_groups(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                              %bias: tensor<4xf32>, %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %tmp = tensor.empty() : tensor<4xf32>
  %first = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%tmp : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
        outs(%arg2 : tensor<4xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<4xf32>
    wafer.group.yield %sum : tensor<4xf32>
  } : tensor<4xf32>

  %second = wafer.group ins(%first, %bias : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
        outs(%arg2 : tensor<4xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<4xf32>
    wafer.group.yield %sum : tensor<4xf32>
  } : tensor<4xf32>
  return %second : tensor<4xf32>
}

func.func @if_group(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                    %out: tensor<4xf32>, %cond: i1) -> tensor<4xf32> {
  %group = wafer.group ins(%lhs, %rhs, %cond : tensor<4xf32>, tensor<4xf32>, i1)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: i1,
       %arg3: tensor<4xf32>):
    %selected = scf.if %arg2 -> tensor<4xf32> {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
          outs(%arg3 : tensor<4xf32>) {
        ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
          %add = arith.addf %lhs_el, %rhs_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      scf.yield %sum : tensor<4xf32>
    } else {
      scf.yield %arg1 : tensor<4xf32>
    }
    wafer.group.yield %selected : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @loop_group(%input: tensor<4xf32>, %out: tensor<4xf32>,
                      %lb: index, %ub: index, %step: index)
    -> tensor<4xf32> {
  %group = wafer.group ins(%input, %lb, %ub, %step : tensor<4xf32>, index, index, index)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: index, %arg2: index, %arg3: index,
       %arg4: tensor<4xf32>):
    %loop_result = scf.for %i = %arg1 to %arg2 step %arg3
        iter_args(%acc = %arg4) -> (tensor<4xf32>) {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%acc, %arg0 : tensor<4xf32>, tensor<4xf32>)
          outs(%acc : tensor<4xf32>) {
        ^bb0(%acc_el: f32, %input_el: f32, %out_el: f32):
          %add = arith.addf %acc_el, %input_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      scf.yield %sum : tensor<4xf32>
    }
    wafer.group.yield %loop_result : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK-LABEL: func.func @add_group
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: bufferization.to_tensor
// CHECK: wafer.tile.region
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.elementwise <add>
// CHECK: wafer.tile.store
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>

// CHECK-LABEL: func.func @two_chained_groups
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: bufferization.to_tensor
// CHECK: memref.alloc() : memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: [[FIRST:%[0-9]+]] = wafer.tile.region
// CHECK: wafer.tile.elementwise <add>
// CHECK: wafer.tile.store
// CHECK: [[SECOND:%[0-9]+]] = wafer.tile.region([[FIRST]],
// CHECK: wafer.tile.elementwise <add>
// CHECK: wafer.tile.store
// CHECK: return [[SECOND]] : memref<4xf32, #wafer.memory<ddr, tensor>>

// CHECK-LABEL: func.func @if_group
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: i1)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK: wafer.tile.region
// CHECK: scf.if {{%.*}} -> (memref<4xf32, #wafer.memory<spm, tensor>>)
// CHECK: wafer.tile.elementwise <add>
// CHECK: scf.yield {{%.*}} : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: else
// CHECK: scf.yield {{%.*}} : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.store

// CHECK-LABEL: func.func @loop_group
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: index, %{{[^:]+}}: index, %{{[^:]+}}: index)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK: wafer.tile.region
// CHECK: scf.for {{%.*}} = {{%.*}} to {{%.*}} step {{%.*}} iter_args({{%.*}} = {{%.*}}) -> (memref<4xf32, #wafer.memory<spm, tensor>>)
// CHECK: wafer.tile.elementwise <add>
// CHECK: scf.yield {{%.*}} : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.store
