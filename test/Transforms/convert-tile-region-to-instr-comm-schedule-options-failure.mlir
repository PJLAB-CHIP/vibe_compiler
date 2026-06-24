// RUN: not wafer-opt --wafer-convert-tile-region-to-instr='all-gather-schedule=tree' %s 2>&1 | FileCheck --check-prefix=AG %s
// RUN: not wafer-opt --wafer-convert-tile-region-to-instr='all-reduce-schedule=direct' %s 2>&1 | FileCheck --check-prefix=AR %s
// RUN: not wafer-opt --wafer-convert-tile-region-to-instr='reduce-scatter-schedule=ring' %s 2>&1 | FileCheck --check-prefix=RS %s

module {}

// AG: unsupported all_gather schedule: tree
// AR: unsupported all_reduce schedule: direct
// RS: unsupported reduce_scatter schedule: ring
