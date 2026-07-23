add_executable(wafer-run
  WaferRunBoardIO.cpp
  wafer-run.cpp
)
llvm_update_compile_flags(wafer-run)
target_link_libraries(wafer-run
  PRIVATE
  LLVMSupport
  WaferRuntime
)
if(WAFER_ENABLE_BOARD_RUNTIME)
  target_sources(wafer-run PRIVATE TxBoardRuntime.cpp)
  target_compile_definitions(wafer-run PRIVATE
    WAFER_ENABLE_BOARD_RUNTIME=1
    WAFER_TX_RUNTIME_LIBRARY_PATH="${WAFER_TX_RUNTIME_LIBRARY_REAL_PATH}"
  )
  target_include_directories(wafer-run PRIVATE
    "${WAFER_TX_RUNTIME_INCLUDE_DIR}"
  )
  target_link_libraries(wafer-run PRIVATE ${CMAKE_DL_LIBS})
endif()
