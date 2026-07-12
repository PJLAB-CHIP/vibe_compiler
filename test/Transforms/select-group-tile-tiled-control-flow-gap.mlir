// RUN: not wafer-opt --wafer-select-group-tile='logical-rank=0 tile-search-effort=quick' --mlir-print-ir-after-failure %s 2>&1 | FileCheck --implicit-check-not=wafer.tile.region --implicit-check-not=wafer.instr --check-prefixes=DIAG,IR %s

func.func @unsupported_nested_tensor_generate(
    %out: tensor<4x4xf32>, %cond: i1) -> tensor<4x4xf32> {
  %group = wafer.group ins(%cond : i1) outs(%out : tensor<4x4xf32>) {
  ^bb0(%arg0: i1, %arg1: tensor<4x4xf32>):
    %selected = scf.if %arg0 -> tensor<4x4xf32> {
      %generated = tensor.generate {
      ^bb0(%i: index, %j: index):
        %one = arith.constant 1.0 : f32
        tensor.yield %one : f32
      } : tensor<4x4xf32>
      scf.yield %generated : tensor<4x4xf32>
    } else {
      scf.yield %arg1 : tensor<4x4xf32>
    }
    wafer.group.yield %selected : tensor<4x4xf32>
  } : tensor<4x4xf32>
  return %group : tensor<4x4xf32>
}

// DIAG: no_candidate: tile selection found no passing candidate
// DIAG-SAME: unsupported op inside structured control-flow tensor.generate

// IR: IR Dump After SelectGroupTilePass Failed
// IR-LABEL: func.func @unsupported_nested_tensor_generate
// IR: wafer.group
// IR: tensor.generate
