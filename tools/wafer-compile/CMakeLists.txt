set(_wafer_compile_sources
  CompilerIRDump.cpp
  DriverOptions.cpp
  wafer-compile.cpp
)
set(_wafer_compile_test_sources
  TargetModelGate.cpp
)

# Installed/build-tree resources discovered relative to the executable:
# the device linker script (share/wafer) and the pinned XLA SPMD partitioner
# helper (libexec/wafer). No source or build tree absolute path is compiled
# into the production binary.
set(_wafer_compile_resource_directory
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../share/wafer")
set(_wafer_compile_libexec_directory
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../libexec/wafer")
set(_wafer_device_linker_resource
  "${_wafer_compile_resource_directory}/wafer_device_link.py")
set(_wafer_spmd_helper_resource
  "${_wafer_compile_libexec_directory}/wafer_xla_spmd_partitioner")
set(_wafer_compile_crt_resource_directory
  "${_wafer_compile_resource_directory}/crt")
set(_wafer_compile_crt_source_resource
  "${_wafer_compile_crt_resource_directory}/src/wafer_tx81_crt.c")
set(_wafer_compile_crt_profile_impl_resource
  "${_wafer_compile_crt_resource_directory}/src/wafer_tx81_profiler_impl.inc")
set(_wafer_compile_crt_include_resource
  "${_wafer_compile_crt_resource_directory}/include/wafer_tx81_crt.h")
set(_wafer_compile_crt_profiler_include_resource
  "${_wafer_compile_crt_resource_directory}/include/wafer_tx81_profiler.h")
set(_wafer_compile_abi_resource_directory
  "${_wafer_compile_resource_directory}/include/Wafer/ABI")
set(_wafer_compile_abi_direct_dte_resource
  "${_wafer_compile_abi_resource_directory}/Tx81DirectDTEStatusABI.h")
set(_wafer_compile_abi_ncc_resource
  "${_wafer_compile_abi_resource_directory}/Tx81NCCABI.h")
set(_wafer_compile_abi_profiler_resource
  "${_wafer_compile_abi_resource_directory}/Tx81ProfilerABI.h")

set(_wafer_compile_resource_outputs
  "${_wafer_device_linker_resource}"
  "${_wafer_compile_crt_source_resource}"
  "${_wafer_compile_crt_profile_impl_resource}"
  "${_wafer_compile_crt_include_resource}"
  "${_wafer_compile_crt_profiler_include_resource}"
  "${_wafer_compile_abi_direct_dte_resource}"
  "${_wafer_compile_abi_ncc_resource}"
  "${_wafer_compile_abi_profiler_resource}")
add_custom_command(
  OUTPUT
    "${_wafer_device_linker_resource}"
    "${_wafer_compile_crt_source_resource}"
    "${_wafer_compile_crt_profile_impl_resource}"
    "${_wafer_compile_crt_include_resource}"
    "${_wafer_compile_crt_profiler_include_resource}"
    "${_wafer_compile_abi_direct_dte_resource}"
    "${_wafer_compile_abi_ncc_resource}"
    "${_wafer_compile_abi_profiler_resource}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "${_wafer_compile_resource_directory}"
    "${_wafer_compile_crt_resource_directory}/src"
    "${_wafer_compile_crt_resource_directory}/include"
    "${_wafer_compile_abi_resource_directory}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
    "${_wafer_device_linker_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_crt.c"
    "${_wafer_compile_crt_source_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_profiler_impl.inc"
    "${_wafer_compile_crt_profile_impl_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_crt.h"
    "${_wafer_compile_crt_include_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_profiler.h"
    "${_wafer_compile_crt_profiler_include_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81DirectDTEStatusABI.h"
    "${_wafer_compile_abi_direct_dte_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81NCCABI.h"
    "${_wafer_compile_abi_ncc_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81ProfilerABI.h"
    "${_wafer_compile_abi_profiler_resource}"
  DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_crt.c"
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_profiler_impl.inc"
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_crt.h"
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_profiler.h"
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81DirectDTEStatusABI.h"
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81NCCABI.h"
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81ProfilerABI.h"
  VERBATIM
)
if(WAFER_XLA_SPMD_PARTITIONER_HELPER)
  list(APPEND _wafer_compile_resource_outputs "${_wafer_spmd_helper_resource}")
  add_custom_command(
    OUTPUT "${_wafer_spmd_helper_resource}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${_wafer_compile_libexec_directory}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
      "${_wafer_spmd_helper_resource}"
    DEPENDS "${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
    VERBATIM
  )
endif()
add_custom_target(wafer-compile-resources
  DEPENDS ${_wafer_compile_resource_outputs}
)

add_executable(wafer-compile ${_wafer_compile_sources})
llvm_update_compile_flags(wafer-compile)
add_dependencies(wafer-compile wafer-compile-resources)
target_link_libraries(wafer-compile PRIVATE LLVMSupport WaferCompiler)

add_executable(wafer-compile-test ${_wafer_compile_sources}
  ${_wafer_compile_test_sources})
llvm_update_compile_flags(wafer-compile-test)
add_dependencies(wafer-compile-test wafer-compile-resources)
target_link_libraries(wafer-compile-test PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile-test PRIVATE
  WAFER_ENABLE_TEST_HELPER_OVERRIDE=1)
if(WAFER_ENABLE_SYSTEMC_MODEL)
  target_compile_definitions(wafer-compile-test PRIVATE
    WAFER_ENABLE_SYSTEMC_MODEL=1)
  target_link_libraries(wafer-compile-test PRIVATE WaferSystemCSimulator)
  if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
    target_compile_definitions(wafer-compile-test PRIVATE
      WAFER_ENABLE_TARGET_NUMERIC_BACKEND=1)
    target_link_libraries(wafer-compile-test PRIVATE WaferTargetNumericBackend)
  endif()
endif()

# Product install is registered only by the canonical build. The
# top-level contract already verified its importer, SPMD and helper closure.
if(WAFER_ENABLE_PRODUCT_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS wafer-compile
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Compiler)
  install(PROGRAMS "${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/wafer"
    COMPONENT Compiler)
  install(FILES
    "${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/wafer"
    COMPONENT Compiler)
  install(FILES
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_crt.c"
    "${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_profiler_impl.inc"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/wafer/crt/src"
    COMPONENT Compiler)
  install(FILES
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_crt.h"
    "${CMAKE_SOURCE_DIR}/runtime/crt/include/wafer_tx81_profiler.h"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/wafer/crt/include"
    COMPONENT Compiler)
  install(FILES
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81DirectDTEStatusABI.h"
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81NCCABI.h"
    "${CMAKE_SOURCE_DIR}/include/Wafer/ABI/Tx81ProfilerABI.h"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/wafer/include/Wafer/ABI"
    COMPONENT Compiler)
endif()
unset(_wafer_compile_resource_directory)
unset(_wafer_compile_libexec_directory)
unset(_wafer_device_linker_resource)
unset(_wafer_spmd_helper_resource)
unset(_wafer_compile_resource_outputs)
