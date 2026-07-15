add_executable(wafer-compile wafer-compile.cpp)
llvm_update_compile_flags(wafer-compile)
target_link_libraries(wafer-compile PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
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

add_executable(wafer-compile-test wafer-compile.cpp)
llvm_update_compile_flags(wafer-compile-test)
target_link_libraries(wafer-compile-test PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile-test PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
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
