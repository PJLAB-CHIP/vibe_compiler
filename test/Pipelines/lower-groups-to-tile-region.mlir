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

func.func @nested_control_flow_group(%input: tensor<8xf32>,
                                     %bias: tensor<4xf32>,
                                     %out: tensor<8xf32>, %cond: i1,
                                     %lb: index, %ub: index, %step: index)
    -> tensor<8xf32> {
  %group = wafer.group ins(%input, %bias, %cond, %lb, %ub, %step
      : tensor<8xf32>, tensor<4xf32>, i1, index, index, index)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>, %arg2: i1,
       %arg3: index, %arg4: index, %arg5: index, %arg6: tensor<8xf32>):
    %loop_result = scf.for %i = %arg3 to %arg4 step %arg5
        iter_args(%acc = %arg6) -> (tensor<8xf32>) {
      %next = scf.if %arg2 -> tensor<8xf32> {
        %acc_lo = tensor.extract_slice %acc[0] [4] [1]
            : tensor<8xf32> to tensor<4xf32>
        %input_lo = tensor.extract_slice %arg0[0] [4] [1]
            : tensor<8xf32> to tensor<4xf32>
        %scaled = linalg.generic {
            indexing_maps = [
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>
            ],
            iterator_types = ["parallel"]
          } ins(%acc_lo, %input_lo : tensor<4xf32>, tensor<4xf32>)
            outs(%acc_lo : tensor<4xf32>) {
          ^bb0(%acc_el: f32, %input_el: f32, %out_el: f32):
            %mul = arith.mulf %acc_el, %input_el : f32
            linalg.yield %mul : f32
          } -> tensor<4xf32>
        %expanded = tensor.expand_shape %scaled [[0, 1]]
            output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
        %collapsed = tensor.collapse_shape %expanded [[0, 1]]
            : tensor<2x2xf32> into tensor<4xf32>
        %updated = tensor.insert_slice %collapsed into %acc[0] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
        scf.yield %updated : tensor<8xf32>
      } else {
        %acc_hi = tensor.extract_slice %acc[4] [4] [1]
            : tensor<8xf32> to tensor<4xf32>
        %clamped = linalg.generic {
            indexing_maps = [
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>
            ],
            iterator_types = ["parallel"]
          } ins(%acc_hi, %arg1 : tensor<4xf32>, tensor<4xf32>)
            outs(%acc_hi : tensor<4xf32>) {
          ^bb0(%acc_el: f32, %bias_el: f32, %out_el: f32):
            %max = arith.maximumf %acc_el, %bias_el : f32
            linalg.yield %max : f32
          } -> tensor<4xf32>
        %updated = tensor.insert_slice %clamped into %acc[4] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
        scf.yield %updated : tensor<8xf32>
      }
      scf.yield %next : tensor<8xf32>
    }
    wafer.group.yield %loop_result : tensor<8xf32>
  } : tensor<8xf32>
  return %group : tensor<8xf32>
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

// CHECK-LABEL: func.func @nested_control_flow_group
// CHECK-SAME: (%{{[^:]+}}: memref<8xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<8xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: i1, %{{[^:]+}}: index, %{{[^:]+}}: index, %{{[^:]+}}: index)
// CHECK-SAME: -> memref<8xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK: wafer.tile.region
// CHECK: scf.for {{%.*}} = {{%.*}} to {{%.*}} step {{%.*}} iter_args({{%.*}} = {{%.*}}) -> (memref<8xf32, #wafer.memory<spm, tensor>>)
// CHECK: scf.if {{%.*}} -> (memref<8xf32, #wafer.memory<spm, tensor>>)
// CHECK: wafer.tile.extract_slice
// CHECK: wafer.tile.elementwise <mul>
// CHECK: wafer.tile.reshape
// CHECK: wafer.tile.reshape
// CHECK: wafer.tile.insert_slice
// CHECK: scf.yield {{%.*}} : memref<8xf32, #wafer.memory<spm, tensor>>
// CHECK: else
// CHECK: wafer.tile.extract_slice
// CHECK: wafer.tile.elementwise <max>
// CHECK: wafer.tile.insert_slice
// CHECK: scf.yield {{%.*}} : memref<8xf32, #wafer.memory<spm, tensor>>
// CHECK: scf.yield {{%.*}} : memref<8xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.store
