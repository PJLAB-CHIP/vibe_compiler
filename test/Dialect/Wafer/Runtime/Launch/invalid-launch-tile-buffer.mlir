// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %output = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.launch @compiled_kernel
      inputs(%input : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
      outputs(%output : tensor<4xf32>)
      {package_ref = "single_tile.pkg", spm_bytes = 4096 : i64, ddr_bytes = 8192 : i64}
      : tensor<4xf32>
}

// CHECK: launch boundary values must be ranked tensors
