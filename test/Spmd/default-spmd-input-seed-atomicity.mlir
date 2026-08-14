// REQUIRES: shardy
// RUN: not wafer-opt --mlir-disable-threading --wafer-apply-default-spmd-sharding \
// RUN:   --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=sdy.mesh --implicit-check-not=sdy.sharding_constraint

module {
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 2>}

  func.func @invalid_second_custom_call(
      %x: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %0 = stablehlo.custom_call @Sharding(%x)
        {backend_config = "",
         mhlo.sharding = "{devices=[2,1]0,1}"}
        : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %1 = stablehlo.custom_call @Sharding(%0)
        {backend_config = ""}
        : (tensor<32x16xf32>) -> tensor<32x16xf32>
    return %1 : tensor<32x16xf32>
  }
}

// CHECK: expected frontend sharding custom call to carry mhlo.sharding or stablehlo.sharding
// CHECK: IR Dump After ApplyDefaultSpmdShardingPass Failed
// CHECK: stablehlo.custom_call @Sharding
// CHECK: mhlo.sharding = "{devices=[2,1]0,1}"
// CHECK: stablehlo.custom_call @Sharding
