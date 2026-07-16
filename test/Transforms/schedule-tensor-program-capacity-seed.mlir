// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal tile-search-effort=quick max-search-candidates=1 print-candidate-summary})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SEED --implicit-check-not='rejected task @capacity_seed#0' --implicit-check-not='rejected task @fused_transpose_capacity_seed#0'
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick max-search-candidates=2 candidate-parallelism=1 print-candidate-summary})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BOUND

module {
  func.func @capacity_seed(%lhs: tensor<16x4096xf16>,
                           %rhs: tensor<4096x688xf16>,
                           %out: tensor<16x688xf16>)
      -> tensor<16x688xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x688xf16>) -> tensor<16x688xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x4096xf16>, tensor<4096x688xf16>)
        outs(%init : tensor<16x688xf16>) -> tensor<16x688xf16>
    return %result : tensor<16x688xf16>
  }

  func.func @fused_transpose_capacity_seed(
      %lhs: tensor<16x4096xf16>, %weight: tensor<688x4096xf16>,
      %out: tensor<16x688xf16>) -> tensor<16x688xf16> {
    %transposed_out = tensor.empty() : tensor<4096x688xf16>
    %rhs = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<688x4096xf16>)
        outs(%transposed_out : tensor<4096x688xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<4096x688xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x688xf16>) -> tensor<16x688xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x4096xf16>, tensor<4096x688xf16>)
        outs(%init : tensor<16x688xf16>) -> tensor<16x688xf16>
    return %result : tensor<16x688xf16>
  }
}

// The target-physical estimate accounts for Tensor and Cx roots. In
// particular, N=172 pads its Cx tail to 192 and remains over capacity, so the
// deterministic highest-pressure walk reaches the existing N=86 menu entry.
// The exactly modeled seed is visited before the known-over-capacity full
// candidate, so one hard-cap slot is enough; the complete artifact gates still
// decide acceptance.
// SEED: wafer.schedule_tensor_program selected task @capacity_seed#0 mode=first-legal tile=[16,86] split=[]
// SEED-SAME: candidates=1
// SEED: wafer.schedule_tensor_program selected task @fused_transpose_capacity_seed#0 mode=first-legal tile=[16,86] split=[]
// SEED-SAME: candidates=1

// Once min-estimated-time continues past the passing seed, the required-live
// six-root bound rejects both known-impossible full tiles without running
// complete materialization/SPM planning. The fused estimator separately
// inventories its seventh (source Tensor) root for seed direction.
// BOUND: wafer.schedule_tensor_program rejected task @capacity_seed#0 tile=[16,688] split=[] reason=target_spm_bound: required modeled live roots exceed planning window
// BOUND: wafer.schedule_tensor_program rejected task @fused_transpose_capacity_seed#0 tile=[16,688] split=[] reason=target_spm_bound: required modeled live roots exceed planning window
