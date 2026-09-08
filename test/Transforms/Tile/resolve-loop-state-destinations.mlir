// RUN: wafer-opt %s --verify-each --pass-pipeline='builtin.module(wafer-resolve-layouts-and-bufferize)' | FileCheck %s
// A loop returns an updated tensor while another state still reads its old
// value. Bufferization must preserve both values and copy back only the update.
// Full source-to-SPM/package and multi-block/tail execution are covered by the
// registered PyTorch prefill none/search tests.

#id = affine_map<(b, m, n) -> (b, m, n)>

module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @old_state_1024(%input: tensor<2x1024x64xf32>, %other: tensor<2x1024x64xf32>) {
      %result:2 = wafer.tile.region(%input, %other : tensor<2x1024x64xf32>, tensor<2x1024x64xf32>) -> (tensor<2x1024x64xf32>, tensor<2x1024x64xf32>) {
      ^bb0(%initial: tensor<2x1024x64xf32>, %initial_other: tensor<2x1024x64xf32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %loop:2 = scf.for %iv = %c0 to %c16 step %c1
            iter_args(%state = %initial, %other_state = %initial_other) -> (tensor<2x1024x64xf32>, tensor<2x1024x64xf32>) {
          %next = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state : tensor<2x1024x64xf32>) outs(%state : tensor<2x1024x64xf32>) {
          ^bb1(%x: f32, %unused: f32):
            %twice = arith.addf %x, %x : f32
            linalg.yield %twice : f32
          } -> tensor<2x1024x64xf32>
          %read_old = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state, %next : tensor<2x1024x64xf32>, tensor<2x1024x64xf32>) outs(%other_state : tensor<2x1024x64xf32>) {
          ^bb1(%old: f32, %new: f32, %unused: f32):
            %difference = arith.subf %new, %old : f32
            linalg.yield %difference : f32
          } -> tensor<2x1024x64xf32>
          scf.yield %next, %read_old : tensor<2x1024x64xf32>, tensor<2x1024x64xf32>
        }
        wafer.tile.yield %loop#0, %loop#1 : tensor<2x1024x64xf32>, tensor<2x1024x64xf32>
      }
      return
    }
  }

// CHECK-LABEL: func.func @old_state_1024
// CHECK: scf.for {{.*}} iter_args(%[[STATE0:[^ ]+]] = %{{[^,]+}}, %[[OTHER0:[^ ]+]] = %{{[^)]+}})
// CHECK: %[[TEMP0:.+]] = memref.alloc()
// CHECK: linalg.generic {{.*}} ins(%[[STATE0]] {{.*}}) outs(%[[TEMP0]]
// CHECK: arith.addf
// CHECK: linalg.generic {{.*}} ins(%[[STATE0]], %[[TEMP0]] {{.*}}) outs(%[[OTHER0]]
// CHECK: arith.subf
// CHECK: memref.copy %[[TEMP0]], %[[STATE0]]
// CHECK-NOT: memref.copy
// CHECK: scf.yield %[[STATE0]], %[[OTHER0]]
// CHECK-NOT: bufferization.materialize_in_destination
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @old_state_1025(%input: tensor<2x1025x64xf32>, %other: tensor<2x1025x64xf32>) {
      %result:2 = wafer.tile.region(%input, %other : tensor<2x1025x64xf32>, tensor<2x1025x64xf32>) -> (tensor<2x1025x64xf32>, tensor<2x1025x64xf32>) {
      ^bb0(%initial: tensor<2x1025x64xf32>, %initial_other: tensor<2x1025x64xf32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %loop:2 = scf.for %iv = %c0 to %c16 step %c1
            iter_args(%state = %initial, %other_state = %initial_other) -> (tensor<2x1025x64xf32>, tensor<2x1025x64xf32>) {
          %next = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state : tensor<2x1025x64xf32>) outs(%state : tensor<2x1025x64xf32>) {
          ^bb1(%x: f32, %unused: f32):
            %twice = arith.addf %x, %x : f32
            linalg.yield %twice : f32
          } -> tensor<2x1025x64xf32>
          %read_old = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state, %next : tensor<2x1025x64xf32>, tensor<2x1025x64xf32>) outs(%other_state : tensor<2x1025x64xf32>) {
          ^bb1(%old: f32, %new: f32, %unused: f32):
            %difference = arith.subf %new, %old : f32
            linalg.yield %difference : f32
          } -> tensor<2x1025x64xf32>
          scf.yield %next, %read_old : tensor<2x1025x64xf32>, tensor<2x1025x64xf32>
        }
        wafer.tile.yield %loop#0, %loop#1 : tensor<2x1025x64xf32>, tensor<2x1025x64xf32>
      }
      return
    }
  }

// CHECK-LABEL: func.func @old_state_1025
// CHECK: scf.for {{.*}} iter_args(%[[STATE1:[^ ]+]] = %{{[^,]+}}, %[[OTHER1:[^ ]+]] = %{{[^)]+}})
// CHECK: %[[TEMP1:.+]] = memref.alloc()
// CHECK: linalg.generic {{.*}} ins(%[[STATE1]] {{.*}}) outs(%[[TEMP1]]
// CHECK: arith.addf
// CHECK: linalg.generic {{.*}} ins(%[[STATE1]], %[[TEMP1]] {{.*}}) outs(%[[OTHER1]]
// CHECK: arith.subf
// CHECK: memref.copy %[[TEMP1]], %[[STATE1]]
// CHECK-NOT: memref.copy
// CHECK: scf.yield %[[STATE1]], %[[OTHER1]]
// CHECK-NOT: bufferization.materialize_in_destination
  wafer.tile.module card_id = 0 tile_id = 2 {
    func.func @old_state_1031(%input: tensor<2x1031x64xf32>, %other: tensor<2x1031x64xf32>) {
      %result:2 = wafer.tile.region(%input, %other : tensor<2x1031x64xf32>, tensor<2x1031x64xf32>) -> (tensor<2x1031x64xf32>, tensor<2x1031x64xf32>) {
      ^bb0(%initial: tensor<2x1031x64xf32>, %initial_other: tensor<2x1031x64xf32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %loop:2 = scf.for %iv = %c0 to %c16 step %c1
            iter_args(%state = %initial, %other_state = %initial_other) -> (tensor<2x1031x64xf32>, tensor<2x1031x64xf32>) {
          %next = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state : tensor<2x1031x64xf32>) outs(%state : tensor<2x1031x64xf32>) {
          ^bb1(%x: f32, %unused: f32):
            %twice = arith.addf %x, %x : f32
            linalg.yield %twice : f32
          } -> tensor<2x1031x64xf32>
          %read_old = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state, %next : tensor<2x1031x64xf32>, tensor<2x1031x64xf32>) outs(%other_state : tensor<2x1031x64xf32>) {
          ^bb1(%old: f32, %new: f32, %unused: f32):
            %difference = arith.subf %new, %old : f32
            linalg.yield %difference : f32
          } -> tensor<2x1031x64xf32>
          scf.yield %next, %read_old : tensor<2x1031x64xf32>, tensor<2x1031x64xf32>
        }
        wafer.tile.yield %loop#0, %loop#1 : tensor<2x1031x64xf32>, tensor<2x1031x64xf32>
      }
      return
    }
  }

// CHECK-LABEL: func.func @old_state_1031
// CHECK: scf.for {{.*}} iter_args(%[[STATE2:[^ ]+]] = %{{[^,]+}}, %[[OTHER2:[^ ]+]] = %{{[^)]+}})
// CHECK: %[[TEMP2:.+]] = memref.alloc()
// CHECK: linalg.generic {{.*}} ins(%[[STATE2]] {{.*}}) outs(%[[TEMP2]]
// CHECK: arith.addf
// CHECK: linalg.generic {{.*}} ins(%[[STATE2]], %[[TEMP2]] {{.*}}) outs(%[[OTHER2]]
// CHECK: arith.subf
// CHECK: memref.copy %[[TEMP2]], %[[STATE2]]
// CHECK-NOT: memref.copy
// CHECK: scf.yield %[[STATE2]], %[[OTHER2]]
// CHECK-NOT: bufferization.materialize_in_destination

  // Nested in-place state and a scalar counter require no state copy. The
  // dynamic trip count also admits zero iterations without changing the init.
  // CHECK-LABEL: func.func @nested_in_place
  // CHECK: scf.for {{.*}} iter_args(%[[OUTER:[^ ]+]] = %{{[^,]+}}, %[[COUNT:[^ ]+]] =
  // CHECK-NOT: memref.copy
  // CHECK: scf.for {{.*}} iter_args(%[[INNER:[^ ]+]] = %[[OUTER]])
  // CHECK-NOT: memref.copy
  // CHECK: linalg.generic {{.*}} ins(%[[INNER]] {{.*}}) outs(%[[INNER]]
  // CHECK-NOT: memref.copy
  // CHECK: scf.yield %[[INNER]]
  // CHECK-NOT: memref.copy
  // CHECK: scf.yield
  // CHECK-NOT: memref.copy
  wafer.tile.module card_id = 0 tile_id = 3 {
    func.func @nested_in_place(%input: tensor<2x1031x64xf32>, %trips: index) {
      %result = wafer.tile.region(%input, %trips : tensor<2x1031x64xf32>, index)
          -> (tensor<2x1031x64xf32>) {
      ^bb0(%initial: tensor<2x1031x64xf32>, %trip_count: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %loop:2 = scf.for %i = %c0 to %trip_count step %c1
            iter_args(%state = %initial, %count = %c0)
            -> (tensor<2x1031x64xf32>, index) {
          %inner = scf.for %j = %c0 to %trip_count step %c1
              iter_args(%nested = %state) -> tensor<2x1031x64xf32> {
            %next = linalg.generic {
                indexing_maps = [#id, #id],
                iterator_types = ["parallel", "parallel", "parallel"]}
                ins(%nested : tensor<2x1031x64xf32>)
                outs(%nested : tensor<2x1031x64xf32>) {
            ^bb1(%x: f32, %unused: f32):
              %twice = arith.addf %x, %x : f32
              linalg.yield %twice : f32
            } -> tensor<2x1031x64xf32>
            scf.yield %next : tensor<2x1031x64xf32>
          }
          %incremented = arith.addi %count, %c1 : index
          scf.yield %inner, %incremented : tensor<2x1031x64xf32>, index
        }
        wafer.tile.yield %loop#0 : tensor<2x1031x64xf32>
      }
      return
    }
  }

}
