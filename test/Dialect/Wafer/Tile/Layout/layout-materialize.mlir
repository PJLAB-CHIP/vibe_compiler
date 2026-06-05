// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %src = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %dst = wafer.tile.materialize_layout %src
        : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: %[[SRC:.+]] = {{.*}}unrealized_conversion_cast to !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
// CHECK: wafer.tile.materialize_layout %[[SRC]] : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
