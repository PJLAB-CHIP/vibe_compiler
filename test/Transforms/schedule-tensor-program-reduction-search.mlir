// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time print-candidate-summary=true preferred-tile-sizes=50000,25000 max-search-candidates=8})' %t/relaxed-generic.mlir 2>&1 | FileCheck %s --check-prefix=RELAXED
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time preferred-tile-sizes=50000,25000 max-search-candidates=8})' %t/strict-generic.mlir 2>&1 | FileCheck %s --check-prefix=STRICT-REDUCE
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time print-candidate-summary=true preferred-tile-sizes=25000 max-search-candidates=8})' %t/unsplit-matmul.mlir 2>&1 | FileCheck %s --check-prefix=UNSPLIT-MATMUL
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time preferred-tile-sizes=50000 max-search-candidates=8})' %t/strict-matmul.mlir 2>&1 | FileCheck %s --check-prefix=STRICT-MATMUL
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time preferred-tile-sizes=50000 max-search-candidates=1})' %t/relaxed-generic.mlir 2>&1 | FileCheck %s --check-prefix=HARD-CAP
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal preferred-tile-sizes=50000 max-search-candidates=1})' %t/relaxed-generic.mlir 2>&1 | FileCheck %s --check-prefix=HARD-CAP-FIRST
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time preferred-tile-sizes=50000 max-search-candidates=1})' %t/strict-matmul.mlir 2>&1 | FileCheck %s --check-prefix=HARD-CAP-ALL-FAIL
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time preferred-tile-sizes=50000 max-search-candidates=2 candidate-parallelism=4})' %t/strict-matmul.mlir 2>&1 | FileCheck %s --check-prefix=HARD-CAP-PARALLEL

//--- relaxed-generic.mlir
module {
  func.func @relaxed_sum(%input: tensor<1x100000xf32>) -> tensor<1xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<1x100000xf32>)
        outs(%init : tensor<1xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc
          fastmath<reassoc,nnan,ninf,nsz> : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// The source combiner explicitly permits changed grouping, exceptional-value
// assumptions and signed-zero folding. Search may therefore refine the single
// reduction axis. Chunk construction and the final combine remain in ascending
// logical-index order.
// RELAXED: wafer.schedule_tensor_program selected task @relaxed_sum#0
// RELAXED-SAME: split=[50000]
// RELAXED-COUNT-2: wafer.instr.reduce <sum>
// RELAXED: wafer.instr.elementwise <add>

// A positive cap is a hard resource boundary even before any candidate passes.
// HARD-CAP: no_candidate: tile selection found no passing task candidate after 1 candidates
// HARD-CAP-FIRST: no_candidate: tile selection found no passing task candidate after 1 candidates

//--- strict-generic.mlir
module {
  func.func @strict_sum(%input: tensor<1x100000xf32>) -> tensor<1xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<1x100000xf32>)
        outs(%init : tensor<1xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// The full reduction is outside target command geometry, but a partial-sum
// tree would change rounding, NaN/infinity and signed-zero behavior. It is not
// an admissible recovery candidate without explicit source facts.
// STRICT-REDUCE: reduction refinement unavailable: candidate reduction split changes floating-point grouping
// STRICT-REDUCE-SAME: requires explicit fastmath<reassoc,nnan,ninf,nsz> source legality

//--- unsplit-matmul.mlir
module {
  func.func @unsplit_matmul(%lhs: tensor<1x50000xf16>,
                           %rhs: tensor<50000x1xf16>) -> tensor<1x1xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<1x1xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<1x1xf16>) -> tensor<1x1xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<1x50000xf16>, tensor<50000x1xf16>)
        outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
    return %result : tensor<1x1xf16>
  }
}

// A representable named floating-point matmul stays one increasing-K GEMM.
// The search must not invent a partial-GEMM tree merely because smaller sizes
// occur in its generic tile menu.
// UNSPLIT-MATMUL: wafer.schedule_tensor_program selected task @unsplit_matmul#0
// UNSPLIT-MATMUL-SAME: split=[]
// UNSPLIT-MATMUL-COUNT-1: wafer.instr.gemm
// UNSPLIT-MATMUL-NOT: wafer.instr.elementwise <add>

//--- strict-matmul.mlir
module {
  func.func @strict_matmul(%lhs: tensor<100000x100000xf16>,
                          %rhs: tensor<100000x100000xf16>)
      -> tensor<100000x100000xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<100000x100000xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<100000x100000xf16>) -> tensor<100000x100000xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<100000x100000xf16>, tensor<100000x100000xf16>)
        outs(%init : tensor<100000x100000xf16>) -> tensor<100000x100000xf16>
    return %result : tensor<100000x100000xf16>
  }
}

// Named linalg.matmul has no typed reassociation contract. Computing two
// independently rounded GEMMs and adding them is therefore not equivalent to
// its increasing-K accumulator semantics.
// STRICT-MATMUL: reduction refinement unavailable: candidate floating-point matmul reduction split has no explicit source-IR reassociation legality fact

// Every refinement still has an unrepresentable unsplit K. With no passing
// candidate, the positive cap must remain a hard boundary rather than letting
// the search drain its traversal-size queue.
// HARD-CAP-ALL-FAIL: no_candidate: tile selection found no passing task candidate after 1 candidates

// A parallel batch consumes only the remaining positive budget.
// HARD-CAP-PARALLEL: no_candidate: tile selection found no passing task candidate after 2 candidates
