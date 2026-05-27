// REQUIRES: shardy
// RUN: wafer-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: wafer-import-model --verify-import-result %s | FileCheck %s --check-prefix=VERIFY

sdy.mesh @mesh = <["tp"=2]>

func.func @partitioned_all_gather(
    %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tp"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"tp"}>}) {
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 0 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// PARSE: sdy.mesh @mesh = <["tp"=2]>
// PARSE: func.func @partitioned_all_gather
// PARSE-SAME: sdy.sharding

// VERIFY: wafer-import-model: verified frontend artifact
