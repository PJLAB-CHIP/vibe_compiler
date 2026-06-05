// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>)
      -> (!wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.tile.yield %local
        : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  }
}

// CHECK: SPM storage values cannot cross wafer.tile.region boundaries
