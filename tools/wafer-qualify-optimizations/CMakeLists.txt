add_executable(wafer-qualify-optimizations
  wafer-qualify-optimizations.cpp
)
llvm_update_compile_flags(wafer-qualify-optimizations)
target_include_directories(wafer-qualify-optimizations PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Wafer
)
target_link_libraries(wafer-qualify-optimizations PRIVATE
  LLVMSupport
  WaferCompiler
  WaferSupport
)
target_compile_definitions(wafer-qualify-optimizations PRIVATE
  WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  WAFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
)
if(WAFER_ENABLE_SYSTEMC_MODEL)
  target_compile_definitions(wafer-qualify-optimizations PRIVATE
    WAFER_ENABLE_SYSTEMC_MODEL=1)
  target_link_libraries(wafer-qualify-optimizations PRIVATE WaferSystemCModel)
  if(WAFER_ENABLE_BULK_MODEL_DEPS)
    target_compile_definitions(wafer-qualify-optimizations PRIVATE
      WAFER_ENABLE_TARGET_BULK_MODEL=1)
    target_link_libraries(wafer-qualify-optimizations PRIVATE
      WaferTargetBulkModel)
  endif()
endif()
