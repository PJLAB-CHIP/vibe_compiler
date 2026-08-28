// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-ddr-memory -split-input-file -verify-diagnostics %t/ddr.mlir
// RUN: wafer-opt --wafer-plan-spm-memory -split-input-file -verify-diagnostics %t/spm.mlir

//--- ddr.mlir
// expected-error @below {{unsupported_lifetime_control_flow: DDR memory planning only supports single-block wafer.tile.region, scf.if and scf.for structured regions}}
func.func @multi_block_cfg(%condition: i1) {
  cf.cond_br %condition, ^bb1, ^bb2
^bb1:
  cf.br ^bb2
^bb2:
  return
}

// -----

func.func @unsupported_parallel_ddr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{unsupported_lifetime_control_flow: DDR memory planning only supports single-block wafer.tile.region, scf.if and scf.for structured regions}}
  scf.parallel (%index) = (%c0) to (%c1) step (%c1) {
    scf.reduce
  }
  return
}

// -----

func.func @unsupported_ddr_root_producer() {
  // expected-error @below {{unsupported_lifetime_alias: DDR memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
  %ddr = memref.alloca()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// -----

func.func @unsupported_loop_carried_ddr_allocation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    // expected-error @below {{unsupported_lifetime_alias: loop-body DDR allocation cannot be loop-carried without multi-instance placement}}
    %next = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    scf.yield %next : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_ddr_raw_pointer_escape() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  // expected-error @below {{unsupported_lifetime_alias: tracked DDR storage cannot escape through raw metadata or an operation without supported alias/effect semantics}}
  %pointer = memref.extract_aligned_pointer_as_index %ddr
      : memref<128xf16, #wafer.memory<ddr, tensor>> -> index
  return
}

// -----

func.func @unsupported_ddr_to_memref_without_origin() {
  %tensor = tensor.empty() : tensor<128xf16>
  // expected-error @below {{unsupported_lifetime_alias: DDR memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
  %unknown = bufferization.to_memref %tensor
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// -----

func.func @unsupported_generic_to_ddr_space_cast(%generic: memref<128xf16>) {
  // expected-error @below {{unsupported_lifetime_alias: DDR memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
  %ddr = memref.memory_space_cast %generic
      : memref<128xf16>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// -----

func.func private @mixed_tensor_result(%arg: tensor<128xf16>, %cond: i1)
    -> tensor<128xf16> {
  %independent = tensor.empty() : tensor<128xf16>
  %selected = arith.select %cond, %arg, %independent : tensor<128xf16>
  return %selected : tensor<128xf16>
}

func.func @unsupported_type_erased_ddr_actual(%cond: i1) {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tensor = bufferization.to_tensor %ddr restrict writable
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  // expected-error @below {{unsupported_lifetime_alias: tracked DDR storage cannot escape through raw metadata or an operation without supported alias/effect semantics}}
  %mixed = func.call @mixed_tensor_result(%tensor, %cond)
      : (tensor<128xf16>, i1) -> tensor<128xf16>
  return
}

//--- spm.mlir
func.func @unsupported_parallel_spm(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // Shared structured lifetime legality owns unsupported control flow before
    // any owner-specific completion tracker can consume the timeline.
    // expected-error @below {{unsupported_lifetime_control_flow: SPM memory planning only supports single-block func.func, non-nested wafer.tile.region, scf.if and scf.for structured regions}}
    scf.parallel (%index) = (%c0) to (%c1) step (%c1) {
      scf.reduce
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_spm_root_producer(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    // expected-error @below {{unsupported_lifetime_alias: SPM memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
    %spm = memref.alloca()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_loop_carried_spm_allocation(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init)
        -> (memref<128xf16, #wafer.memory<spm, tensor>>) {
      // expected-error @below {{unsupported_lifetime_alias: loop-body SPM allocation cannot be loop-carried without multi-instance placement}}
      %next = memref.alloc()
          : memref<128xf16, #wafer.memory<spm, tensor>>
      scf.yield %next : memref<128xf16, #wafer.memory<spm, tensor>>
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_spm_raw_pointer_escape(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{unsupported_lifetime_alias: tracked SPM storage cannot escape through raw metadata or an operation without supported alias/effect semantics}}
    %pointer = memref.extract_aligned_pointer_as_index %spm
        : memref<128xf16, #wafer.memory<spm, tensor>> -> index
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_spm_to_memref_without_origin(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %tensor = tensor.empty() : tensor<128xf16>
    // expected-error @below {{unsupported_lifetime_alias: SPM memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
    %unknown = bufferization.to_memref %tensor
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unsupported_generic_to_spm_space_cast(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %generic = memref.alloc() : memref<128xf16>
    // expected-error @below {{unsupported_lifetime_alias: SPM memref producers must be memref.alloc or implement a supported alias/control-flow interface}}
    %spm = memref.memory_space_cast %generic
        : memref<128xf16>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func private @erase_tensor_storage_handle(%arg: tensor<128xf16>)
    -> index {
  %generic = bufferization.to_memref %arg : memref<128xf16>
  %pointer = builtin.unrealized_conversion_cast %generic
      : memref<128xf16> to index
  return %pointer : index
}

func.func @unsupported_type_erased_spm_pointer_escape(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %tensor = bufferization.to_tensor %spm restrict writable
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{unsupported_lifetime_alias: tracked SPM storage cannot escape through raw metadata or an operation without supported alias/effect semantics}}
    %pointer = func.call @erase_tensor_storage_handle(%tensor)
        : (tensor<128xf16>) -> index
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
