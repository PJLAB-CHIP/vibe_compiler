// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s | FileCheck %s

module {
  func.func @different_traversal_shapes_do_not_share_scope(
      %activation: tensor<4x8xf16>,
      %wide_weight: tensor<8x16xf16>,
      %narrow_weight: tensor<8x12xf16>)
      -> (tensor<4x16xf16>, tensor<4x12xf16>) {
    %zero = arith.constant 0.0 : f16
    %wide_empty = tensor.empty() : tensor<4x16xf16>
    %wide_init = linalg.fill ins(%zero : f16)
        outs(%wide_empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    %wide = linalg.matmul
        ins(%activation, %wide_weight : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%wide_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %narrow_empty = tensor.empty() : tensor<4x12xf16>
    %narrow_init = linalg.fill ins(%zero : f16)
        outs(%narrow_empty : tensor<4x12xf16>) -> tensor<4x12xf16>
    %narrow = linalg.matmul
        ins(%activation, %narrow_weight : tensor<4x8xf16>, tensor<8x12xf16>)
        outs(%narrow_init : tensor<4x12xf16>) -> tensor<4x12xf16>
    return %wide, %narrow : tensor<4x16xf16>, tensor<4x12xf16>
  }

  func.func @mixed_task_families_do_not_share_scope(
      %input: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %local_empty = tensor.empty() : tensor<4xf32>
    %local = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%local_empty : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<4xf32>
    %collective_empty = tensor.empty() : tensor<4xf32>
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>)
        outs(%collective_empty : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 1 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    return %local, %collective : tensor<4xf32>, tensor<4xf32>
  }

  func.func @control_flow_roots_do_not_share_scope(
      %gate_input: tensor<4x16xf16>,
      %up_input: tensor<4x16xf16>)
      -> (tensor<4x16xf16>, tensor<4x16xf16>) {
    %gate_empty = tensor.empty() : tensor<4x16xf16>
    %up_empty = tensor.empty() : tensor<4x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %results:2 = scf.for %i = %c0 to %c4 step %c1
        iter_args(%gate_iter = %gate_empty, %up_iter = %up_empty)
        -> (tensor<4x16xf16>, tensor<4x16xf16>) {
      %gate = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%gate_input : tensor<4x16xf16>)
          outs(%gate_iter : tensor<4x16xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
      } -> tensor<4x16xf16>
      %up = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%up_input : tensor<4x16xf16>)
          outs(%up_iter : tensor<4x16xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
      } -> tensor<4x16xf16>
      scf.yield %gate, %up : tensor<4x16xf16>, tensor<4x16xf16>
    }
    return %results#0, %results#1 : tensor<4x16xf16>, tensor<4x16xf16>
  }
}

// CHECK-LABEL: func.func @different_traversal_shapes_do_not_share_scope
// CHECK-COUNT-2: wafer.tile.region
// CHECK: return

// CHECK-LABEL: func.func @mixed_task_families_do_not_share_scope
// CHECK-COUNT-2: wafer.tile.region
// CHECK: return

// CHECK-LABEL: func.func @control_flow_roots_do_not_share_scope
// CHECK: %[[GATE_EMPTY:[0-9A-Za-z_]+]] = tensor.empty
// CHECK-NEXT: %[[GATE_DDR:[0-9A-Za-z_]+]] = memref.alloc() {{.*}} : memref<4x16xf16, #wafer.memory<ddr, tensor>>
// CHECK: %[[UP_EMPTY:[0-9A-Za-z_]+]] = tensor.empty
// CHECK-NEXT: %[[UP_DDR:[0-9A-Za-z_]+]] = memref.alloc() {{.*}} : memref<4x16xf16, #wafer.memory<ddr, tensor>>
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[GATE_EMPTY]], {{.*}} = %[[UP_EMPTY]])
// CHECK: wafer.tile.region({{.*}}, %[[GATE_DDR]]
// CHECK: %[[GATE_TENSOR:[0-9A-Za-z_]+]] = bufferization.to_tensor
// CHECK: wafer.tile.region({{.*}}, %[[UP_DDR]]
// CHECK: %[[UP_TENSOR:[0-9A-Za-z_]+]] = bufferization.to_tensor
// CHECK: scf.yield %[[GATE_TENSOR]], %[[UP_TENSOR]]
