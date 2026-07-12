// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-selected-instr)' %s | FileCheck --implicit-check-not=selected_group --implicit-check-not='tensor<' --implicit-check-not=bufferization.to_ %s

func.func @preserved_helper(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

func.func @pipeline_elementwise(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                                %out: tensor<8xf32>) -> tensor<8xf32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
        outs(%arg2 : tensor<8xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<8xf32>
    wafer.group.yield %sum : tensor<8xf32>
  } : tensor<8xf32>
  return %group : tensor<8xf32>
}

func.func @pipeline_two_groups(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                               %tmp: tensor<8xf32>, %out: tensor<8xf32>)
    -> tensor<8xf32> {
  %first = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%tmp : tensor<8xf32>) {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
        outs(%arg2 : tensor<8xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<8xf32>
    wafer.group.yield %sum : tensor<8xf32>
  } : tensor<8xf32>
  %second = wafer.group ins(%first, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
        outs(%arg2 : tensor<8xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<8xf32>
    wafer.group.yield %sum : tensor<8xf32>
  } : tensor<8xf32>
  return %second : tensor<8xf32>
}

// CHECK-LABEL: func.func @preserved_helper
// CHECK-SAME: memref<8xf32, #wafer.memory<ddr, tensor>>
// CHECK: return

// CHECK-LABEL: func.func @pipeline_elementwise
// CHECK-NOT: selected_group
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK: wafer.tile.region
// CHECK: wafer.spm.offset
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.wdma

// CHECK-LABEL: func.func @pipeline_two_groups
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK: wafer.tile.region
// CHECK: wafer.tile.region
// CHECK: return
