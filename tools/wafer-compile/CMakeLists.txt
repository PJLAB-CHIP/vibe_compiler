set(_wafer_compile_sources
  DriverOptions.cpp
  TargetModelGate.cpp
  wafer-compile.cpp
)
file(REAL_PATH "${Python3_EXECUTABLE}" _wafer_python_executable)
file(REAL_PATH "${LLVM_TOOLS_BINARY_DIR}/clang++" _wafer_device_clangxx)
if(NOT EXISTS "${_wafer_device_clangxx}")
  message(FATAL_ERROR "LLVM clang++ is required for TX device code")
endif()

add_executable(wafer-compile ${_wafer_compile_sources})
llvm_update_compile_flags(wafer-compile)
target_link_libraries(wafer-compile PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${_wafer_python_executable}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
  WAFER_DEVICE_CLANGXX="${_wafer_device_clangxx}"
)
if(WAFER_ENABLE_SYSTEMC_MODEL)
  target_compile_definitions(wafer-compile PRIVATE
    WAFER_ENABLE_SYSTEMC_MODEL=1)
  target_link_libraries(wafer-compile PRIVATE WaferSystemCModel)
  if(WAFER_ENABLE_BULK_MODEL_DEPS)
    target_compile_definitions(wafer-compile PRIVATE
      WAFER_ENABLE_TARGET_BULK_MODEL=1)
    target_link_libraries(wafer-compile PRIVATE WaferTargetBulkModel)
  endif()
endif()

add_executable(wafer-compile-test ${_wafer_compile_sources})
llvm_update_compile_flags(wafer-compile-test)
target_link_libraries(wafer-compile-test PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile-test PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${_wafer_python_executable}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
  WAFER_DEVICE_CLANGXX="${_wafer_device_clangxx}"
  WAFER_ENABLE_TEST_HELPER_OVERRIDE=1
)
if(WAFER_ENABLE_SYSTEMC_MODEL)
  target_compile_definitions(wafer-compile-test PRIVATE
    WAFER_ENABLE_SYSTEMC_MODEL=1)
  target_link_libraries(wafer-compile-test PRIVATE WaferSystemCModel)
  if(WAFER_ENABLE_BULK_MODEL_DEPS)
    target_compile_definitions(wafer-compile-test PRIVATE
      WAFER_ENABLE_TARGET_BULK_MODEL=1)
    target_link_libraries(wafer-compile-test PRIVATE WaferTargetBulkModel)
  endif()
endif()
unset(_wafer_python_executable)
unset(_wafer_device_clangxx)
