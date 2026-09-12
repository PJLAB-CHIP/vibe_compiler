// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, ntensor>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<ddr, ntensor>>
  wafer.tile.store %source, %dest
      : memref<4x16xf16, #wafer.memory<spm, ntensor>>
     -> memref<4x16xf16, #wafer.memory<ddr, ntensor>>
}

// CHECK: tile.store dest must use tensor layout
