# CMake shim for the pinned Shardy checkout.
#
# Upstream Shardy only ships Bazel BUILD files at the pinned revisions used by
# the PyTorch/XLA-rooted dependency stack. This file mirrors the public SDY
# dialect/pass targets that Wafer needs for compile gating, while reusing the
# top-level pinned LLVM/MLIR and embedded StableHLO targets.

function(wafer_shardy_set_common_target_properties target)
  target_include_directories(${target}
    PUBLIC
      "${WAFER_SHARDY_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}")
  target_compile_features(${target} PUBLIC cxx_std_17)
  llvm_update_compile_flags(${target})
endfunction()

function(wafer_add_shardy_targets)
  if(TARGET ShardySdyDialect)
    return()
  endif()

  if(NOT TARGET StablehloOps)
    message(FATAL_ERROR
      "Shardy CMake targets require embedded StableHLO targets. "
      "Enable WAFER_ENABLE_IMPORTER_DEPS or WAFER_ENABLE_SPMD_PARTITIONER_DEPS.")
  endif()

  set(_wafer_sdy_ir_dir "${WAFER_SHARDY_SOURCE_DIR}/shardy/dialect/sdy/ir")
  set(_wafer_sdy_transforms_dir "${WAFER_SHARDY_SOURCE_DIR}/shardy/dialect/sdy/transforms")
  set(_wafer_sdy_common_dir "${WAFER_SHARDY_SOURCE_DIR}/shardy/common")
  set(_wafer_sdy_tools_dir "${WAFER_SHARDY_SOURCE_DIR}/shardy/tools")

  include_directories("${WAFER_SHARDY_SOURCE_DIR}")
  file(MAKE_DIRECTORY
    "${CMAKE_BINARY_DIR}/shardy/dialect/sdy/ir"
    "${CMAKE_BINARY_DIR}/shardy/dialect/sdy/transforms/import"
    "${CMAKE_BINARY_DIR}/shardy/dialect/sdy/transforms/export"
    "${CMAKE_BINARY_DIR}/shardy/dialect/sdy/transforms/propagation")

  set(TABLEGEN_OUTPUT)
  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/op_interface.td")
  mlir_tablegen(shardy/dialect/sdy/ir/op_interface.h.inc -gen-op-interface-decls)
  mlir_tablegen(shardy/dialect/sdy/ir/op_interface.cc.inc -gen-op-interface-defs)

  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/dialect.td")
  mlir_tablegen(shardy/dialect/sdy/ir/dialect.h.inc -gen-dialect-decls)
  mlir_tablegen(shardy/dialect/sdy/ir/dialect.cc.inc -gen-dialect-defs)

  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/ops.td")
  mlir_tablegen(shardy/dialect/sdy/ir/ops.h.inc -gen-op-decls)
  mlir_tablegen(shardy/dialect/sdy/ir/ops.cc.inc -gen-op-defs)

  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/attrs.td")
  mlir_tablegen(shardy/dialect/sdy/ir/attrs.h.inc -gen-attrdef-decls)
  mlir_tablegen(shardy/dialect/sdy/ir/attrs.cc.inc -gen-attrdef-defs)

  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/enums.td")
  mlir_tablegen(shardy/dialect/sdy/ir/enums.h.inc -gen-enum-decls)
  mlir_tablegen(shardy/dialect/sdy/ir/enums.cc.inc -gen-enum-defs)

  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_ir_dir}/canonicalization.td")
  mlir_tablegen(shardy/dialect/sdy/ir/canonicalization.cc.inc -gen-rewriters)
  add_public_tablegen_target(ShardySdyDialectIncGen)

  set(TABLEGEN_OUTPUT)
  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_transforms_dir}/import/passes.td")
  mlir_tablegen(shardy/dialect/sdy/transforms/import/passes.h.inc
    -gen-pass-decls -name=SdyImport)
  add_public_tablegen_target(ShardySdyImportPassesIncGen)

  set(TABLEGEN_OUTPUT)
  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_transforms_dir}/export/passes.td")
  mlir_tablegen(shardy/dialect/sdy/transforms/export/passes.h.inc
    -gen-pass-decls -name=SdyExport)
  add_public_tablegen_target(ShardySdyExportPassesIncGen)

  set(TABLEGEN_OUTPUT)
  set(LLVM_TARGET_DEFINITIONS "${_wafer_sdy_transforms_dir}/propagation/passes.td")
  mlir_tablegen(shardy/dialect/sdy/transforms/propagation/passes.h.inc
    -gen-pass-decls -name=SdyPropagation)
  add_public_tablegen_target(ShardySdyPropagationPassesIncGen)

  add_library(ShardySdyDialect STATIC
    "${_wafer_sdy_ir_dir}/canonicalization.cc"
    "${_wafer_sdy_ir_dir}/data_flow_utils.cc"
    "${_wafer_sdy_ir_dir}/dialect.cc"
    "${_wafer_sdy_ir_dir}/parsers.cc"
    "${_wafer_sdy_ir_dir}/printers.cc"
    "${_wafer_sdy_ir_dir}/utils.cc"
    "${_wafer_sdy_ir_dir}/verifiers.cc")
  wafer_shardy_set_common_target_properties(ShardySdyDialect)
  add_dependencies(ShardySdyDialect ShardySdyDialectIncGen)
  target_link_libraries(ShardySdyDialect
    PUBLIC
      LLVMSupport
      MLIRBytecodeOpInterface
      MLIRFuncDialect
      MLIRIR
      MLIRInferTypeOpInterface
      MLIRSideEffectInterfaces
      MLIRSupport
      StablehloAssemblyFormat
      StablehloOps
      StablehloTypeInference)

  add_library(ShardySdyRegister STATIC
    "${_wafer_sdy_ir_dir}/register.cc")
  wafer_shardy_set_common_target_properties(ShardySdyRegister)
  target_link_libraries(ShardySdyRegister
    PUBLIC
      ShardySdyDialect
      MLIRFuncAllExtensions
      MLIRFuncDialect
      MLIRIR
      StablehloOps)

  add_library(ShardyCommon STATIC
    "${_wafer_sdy_common_dir}/file_utils.cc"
    "${_wafer_sdy_common_dir}/save_module_op.cc")
  wafer_shardy_set_common_target_properties(ShardyCommon)
  target_link_libraries(ShardyCommon
    PUBLIC
      ShardySdyDialect
      LLVMSupport
      MLIRFuncDialect
      MLIRIR
      MLIRPass
      MLIRSupport)

  add_library(ShardySdyTransformCommon STATIC
    "${_wafer_sdy_transforms_dir}/common/op_properties.cc")
  wafer_shardy_set_common_target_properties(ShardySdyTransformCommon)
  target_link_libraries(ShardySdyTransformCommon
    PUBLIC
      ShardySdyDialect
      MLIRIR
      MLIRSupport
      StablehloBase
      StablehloOps)

  add_library(ShardySdyImportPasses STATIC
    "${_wafer_sdy_transforms_dir}/import/add_data_flow_edges.cc"
    "${_wafer_sdy_transforms_dir}/import/apply_sharding_constraints.cc"
    "${_wafer_sdy_transforms_dir}/import/constant_splitter.cc"
    "${_wafer_sdy_transforms_dir}/import/import_maximal_sharding.cc"
    "${_wafer_sdy_transforms_dir}/import/import_pipeline.cc"
    "${_wafer_sdy_transforms_dir}/import/sharding_group_unification.cc")
  wafer_shardy_set_common_target_properties(ShardySdyImportPasses)
  add_dependencies(ShardySdyImportPasses ShardySdyImportPassesIncGen)
  target_link_libraries(ShardySdyImportPasses
    PUBLIC
      ShardyCommon
      ShardySdyDialect
      ShardySdyTransformCommon
      LLVMSupport
      MLIRFuncDialect
      MLIRIR
      MLIRPass
      MLIRRewrite
      MLIRSideEffectInterfaces
      MLIRSupport
      MLIRTransformUtils
      MLIRTransforms
      StablehloOps)

  add_library(ShardySdyExportPasses STATIC
    "${_wafer_sdy_transforms_dir}/export/export_pipeline.cc"
    "${_wafer_sdy_transforms_dir}/export/sharding_constraint_to_reshard.cc"
    "${_wafer_sdy_transforms_dir}/export/sink_data_flow_edges.cc"
    "${_wafer_sdy_transforms_dir}/export/update_non_divisible_input_output_shardings.cc")
  wafer_shardy_set_common_target_properties(ShardySdyExportPasses)
  add_dependencies(ShardySdyExportPasses ShardySdyExportPassesIncGen)
  target_link_libraries(ShardySdyExportPasses
    PUBLIC
      ShardyCommon
      ShardySdyDialect
      ShardySdyTransformCommon
      LLVMSupport
      MLIRFuncDialect
      MLIRIR
      MLIRPass
      MLIRRewrite
      MLIRSideEffectInterfaces
      MLIRSupport
      MLIRTransformUtils
      StablehloOps)

  add_library(ShardySdyPropagationSupport STATIC
    "${_wafer_sdy_transforms_dir}/propagation/aggressive_factor_propagation.cc"
    "${_wafer_sdy_transforms_dir}/propagation/auto_partitioner_registry.cc"
    "${_wafer_sdy_transforms_dir}/propagation/basic_factor_propagation.cc"
    "${_wafer_sdy_transforms_dir}/propagation/op_sharding_rule_builder.cc"
    "${_wafer_sdy_transforms_dir}/propagation/op_sharding_rule_registry.cc"
    "${_wafer_sdy_transforms_dir}/propagation/sharding_projection.cc"
    "${_wafer_sdy_transforms_dir}/propagation/utils.cc")
  wafer_shardy_set_common_target_properties(ShardySdyPropagationSupport)
  target_link_libraries(ShardySdyPropagationSupport
    PUBLIC
      ShardySdyDialect
      ShardySdyTransformCommon
      LLVMSupport
      MLIRFuncDialect
      MLIRIR
      MLIRPass
      MLIRSupport
      StablehloOps)

  add_library(ShardySdyPropagationPasses STATIC
    "${_wafer_sdy_transforms_dir}/propagation/aggressive_propagation.cc"
    "${_wafer_sdy_transforms_dir}/propagation/basic_propagation.cc"
    "${_wafer_sdy_transforms_dir}/propagation/op_priority_propagation.cc"
    "${_wafer_sdy_transforms_dir}/propagation/populate_op_sharding_rules.cc"
    "${_wafer_sdy_transforms_dir}/propagation/propagation_pipeline.cc"
    "${_wafer_sdy_transforms_dir}/propagation/user_priority_propagation.cc")
  wafer_shardy_set_common_target_properties(ShardySdyPropagationPasses)
  add_dependencies(ShardySdyPropagationPasses ShardySdyPropagationPassesIncGen)
  target_link_libraries(ShardySdyPropagationPasses
    PUBLIC
      ShardyCommon
      ShardySdyDialect
      ShardySdyExportPasses
      ShardySdyImportPasses
      ShardySdyPropagationSupport
      ShardySdyTransformCommon
      LLVMSupport
      MLIRBufferizationDialect
      MLIRFuncDialect
      MLIRIR
      MLIRPass
      MLIRRewrite
      MLIRSideEffectInterfaces
      MLIRSupport
      MLIRTransformUtils
      StablehloOps)

  add_library(ShardySdyTransforms STATIC
    "${_wafer_sdy_transforms_dir}/passes.cc")
  wafer_shardy_set_common_target_properties(ShardySdyTransforms)
  target_link_libraries(ShardySdyTransforms
    PUBLIC
      ShardySdyExportPasses
      ShardySdyImportPasses
      ShardySdyPropagationPasses
      MLIRPass)

  add_executable(shardy-sdy-opt
    "${_wafer_sdy_tools_dir}/sdy_opt_main.cc")
  wafer_shardy_set_common_target_properties(shardy-sdy-opt)
  target_link_libraries(shardy-sdy-opt
    PRIVATE
      ShardySdyDialect
      ShardySdyTransforms
      StablehloOps
      MLIRFuncAllExtensions
      MLIRFuncDialect
      MLIRIR
      MLIRMlirOptMain
      MLIRQuantDialect)

  add_custom_target(wafer-shardy-cmake-gate
    DEPENDS
      ShardySdyDialect
      ShardySdyRegister
      ShardySdyTransforms
      shardy-sdy-opt)
endfunction()
