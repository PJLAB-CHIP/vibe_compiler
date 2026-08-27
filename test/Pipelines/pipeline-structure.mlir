// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: wafer-opt --verify-each=true --pass-pipeline='builtin.module(wafer-bufferize-instr-functions)' --dump-pass-pipeline -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BUFFERIZE

module {
  func.func @main() {
    return
  }
}

// TILE: builtin.module(func.func(wafer.tile.region(wafer-convert-tile-region-to-instr)),wafer-convert-bufferization-copies-to-instr,func.func(wafer-rebuild-required-ncc-joins))
// BUFFERIZE: builtin.module(wafer-bufferize-instr-function-boundaries,canonicalize{{.*}},cse,canonicalize{{.*}},drop-equivalent-buffer-results)
