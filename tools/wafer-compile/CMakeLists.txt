add_executable(wafer-compile wafer-compile.cpp)
target_link_libraries(wafer-compile PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
)

add_executable(wafer-compile-test wafer-compile.cpp)
target_link_libraries(wafer-compile-test PRIVATE LLVMSupport WaferCompiler)
target_compile_definitions(wafer-compile-test PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
  WAFER_ENABLE_TEST_HELPER_OVERRIDE=1
)
