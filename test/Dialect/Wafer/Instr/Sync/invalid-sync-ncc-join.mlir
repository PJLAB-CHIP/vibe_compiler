// RUN: split-file %s %t
// RUN: wafer-opt -verify-diagnostics %t/empty.mlir
// RUN: wafer-opt -verify-diagnostics %t/out-of-range.mlir
// RUN: wafer-opt -verify-diagnostics %t/duplicate.mlir
// RUN: wafer-opt -verify-diagnostics %t/noncanonical.mlir

//--- empty.mlir

module {
  // expected-error @below {{completion_participants_empty: NCC join requires at least one worker}}
  wafer.instr.ncc_join []
}

//--- out-of-range.mlir

module {
  // expected-error @below {{completion_participant_out_of_range: worker 3 is outside [0, 3)}}
  wafer.instr.ncc_join [0, 3]
}

//--- duplicate.mlir

module {
  // expected-error @below {{completion_participants_not_canonical: workers must be strictly increasing and unique}}
  wafer.instr.ncc_join [1, 1]
}

//--- noncanonical.mlir

module {
  // expected-error @below {{completion_participants_not_canonical: workers must be strictly increasing and unique}}
  wafer.instr.ncc_join [2, 0]
}
