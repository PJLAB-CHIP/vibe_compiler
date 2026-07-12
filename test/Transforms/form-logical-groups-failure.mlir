// RUN: not wafer-opt --wafer-form-logical-groups %s 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>

module {
  func.func @duplicate_destination_boundary(
      %input: tensor<4xf32>,
      %out: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %results:2 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>)
        outs(%out, %out : tensor<4xf32>, tensor<4xf32>) {
      ^bb0(%in: f32, %old0: f32, %old1: f32):
        linalg.yield %in, %in : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
    return %results#0, %results#1 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK: error: 'linalg.generic' op cannot form required logical wafer.group: computed logical-group boundary contains duplicate SSA values
