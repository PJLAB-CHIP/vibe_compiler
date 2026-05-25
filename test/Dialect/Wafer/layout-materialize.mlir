// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile_region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %src = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %dst = wafer.layout.materialize %src
        : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    wafer.tile_yield %arg0 : tensor<4xf32>
  }
}

// CHECK: %[[SRC:.+]] = {{.*}}unrealized_conversion_cast to !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
// CHECK: wafer.layout.materialize %[[SRC]] : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
