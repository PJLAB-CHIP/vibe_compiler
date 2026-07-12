// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 print-candidate-summary tile-search-effort=quick' %s 2>&1 | FileCheck --check-prefixes=FIRST,IR %s
// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 tile-search=min-estimated-time print-candidate-summary tile-search-effort=quick max-search-candidates=4' %s 2>&1 | FileCheck --check-prefixes=MIN,IR %s

func.func @non_divisible_two_by_two_grid(
    %lhs: tensor<2x100001xi64>, %rhs: tensor<2x100001xi64>,
    %extra: tensor<2x100001xi64>,
    %out: tensor<2x100001xi64>) -> tensor<2x100001xi64> {
  %group = wafer.group
      ins(%lhs, %rhs, %extra
          : tensor<2x100001xi64>, tensor<2x100001xi64>,
            tensor<2x100001xi64>)
      outs(%out : tensor<2x100001xi64>) {
  ^bb0(%lhs_arg: tensor<2x100001xi64>, %rhs_arg: tensor<2x100001xi64>,
       %extra_arg: tensor<2x100001xi64>,
       %out_arg: tensor<2x100001xi64>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs_arg, %rhs_arg, %extra_arg
            : tensor<2x100001xi64>, tensor<2x100001xi64>,
              tensor<2x100001xi64>)
        outs(%out_arg : tensor<2x100001xi64>) {
    ^bb0(%left: i64, %right: i64, %extra_value: i64, %old: i64):
      %partial = arith.addi %left, %right : i64
      %value = arith.addi %partial, %extra_value : i64
      linalg.yield %value : i64
    } -> tensor<2x100001xi64>
    wafer.group.yield %sum : tensor<2x100001xi64>
  } : tensor<2x100001xi64>
  return %group : tensor<2x100001xi64>
}

// FIRST: wafer.select_group_tile selected group @non_divisible_two_by_two_grid#0
// FIRST-SAME: mode=first-legal
// FIRST-SAME: tile=[1,50001]

// MIN: wafer.select_group_tile selected group @non_divisible_two_by_two_grid#0
// MIN-SAME: mode=min-estimated-time
// MIN-SAME: tile=[1,50001]

// IR-LABEL: func.func @non_divisible_two_by_two_grid
// IR-NOT: wafer.group

// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.elementwise <add>
// IR: %[[STORE_00:.+]] = memref.subview {{%.*}}[0, 0] [1, 50001] [1, 1]
// IR-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_00]]

// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.elementwise <add>
// IR: %[[STORE_01:.+]] = memref.subview {{%.*}}[0, 50001] [1, 50000] [1, 1]
// IR-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_01]]

// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.elementwise <add>
// IR: %[[STORE_10:.+]] = memref.subview {{%.*}}[1, 0] [1, 50001] [1, 1]
// IR-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_10]]

// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.rdma
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.elementwise <add>
// IR: %[[STORE_11:.+]] = memref.subview {{%.*}}[1, 50001] [1, 50000] [1, 1]
// IR-NEXT: wafer.instr.wdma {{%.*}} to %[[STORE_11]]
// IR-NOT: wafer.instr.elementwise
// IR-NOT: wafer.instr.wdma
// IR: wafer.tile.yield
