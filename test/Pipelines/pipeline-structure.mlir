// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=STABLEHLO
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-bufferize-instr-functions)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BUFFERIZE
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-prepare-instr-for-memory-planning)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MEMORY-PREP

module {
  func.func @main() {
    return
  }
}

// TILE: builtin.module(func.func(wafer.tile.region(wafer-convert-tile-region-to-instr),wafer-place-required-ncc-joins))
// STABLEHLO: builtin.module(wafer-normalize-stablehlo-collectives,wafer-fold-default-stablehlo-execution-ids,wafer-fold-constant-integer-tensor-casts,wafer-lower-static-stablehlo-concatenate,stablehlo-legalize-to-linalg{{.*}},wafer-fold-static-tensor-ops,canonicalize{{.*}})
// BUFFERIZE: builtin.module(canonicalize{{.*}},wafer-bufferize-instr-function-boundaries,canonicalize{{.*}})
// MEMORY-PREP: builtin.module(canonicalize{{.*}},wafer-bufferize-instr-function-boundaries,canonicalize{{.*}},func.func(wafer-rebuild-required-ncc-joins))
