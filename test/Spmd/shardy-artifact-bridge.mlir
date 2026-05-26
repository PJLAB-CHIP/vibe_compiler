// REQUIRES: shardy
// RUN: wafer-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: wafer-opt --wafer-lower-stablehlo-collectives-to-comm='local-rank=1' %s | FileCheck %s --check-prefix=COMM
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

// COMM: sdy.mesh @mesh = <["tp"=2]>
// COMM-LABEL: func.func @partitioned_all_gather
// COMM-NOT: stablehlo.all_gather
// COMM: wafer.comm.all_gather %{{.*}} into %{{.*}} {bytes = 16 : i64, group_size = 2 : i64, local_rank = 1 : i64, rank_group = array<i64: 0, 1>}

// VERIFY: wafer-import-model: verified frontend artifact
