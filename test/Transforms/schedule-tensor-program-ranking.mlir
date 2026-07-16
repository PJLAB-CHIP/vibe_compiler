// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick print-candidate-summary})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal tile-search-effort=quick print-candidate-summary})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

func.func @elementwise_last_dim_65(
    %lhs: tensor<2x65xf16>, %rhs: tensor<2x65xf16>,
    %out: tensor<2x65xf16>) -> tensor<2x65xf16> {
  %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<2x65xf16>, tensor<2x65xf16>)
      outs(%out : tensor<2x65xf16>) {
    ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
      %add = arith.addf %lhs_el, %rhs_el : f16
      linalg.yield %add : f16
    } -> tensor<2x65xf16>
  return %sum : tensor<2x65xf16>
}

// DEFAULT: wafer.schedule_tensor_program selected task @elementwise_last_dim_65#0
// DEFAULT-SAME: mode=min-estimated-time
// DEFAULT-SAME: tile=[2,65]
// DEFAULT-SAME: estimated_time_ps={{[1-9][0-9]*}}
// DEFAULT-SAME: candidates={{[2-9][0-9]*|[2-9]}}

// DEBUG: wafer.schedule_tensor_program selected task @elementwise_last_dim_65#0
// DEBUG-SAME: mode=first-legal
// DEBUG-SAME: tile=[2,65]
// DEBUG-SAME: estimated_time_ps={{[1-9][0-9]*}}
// DEBUG-SAME: candidates=1
