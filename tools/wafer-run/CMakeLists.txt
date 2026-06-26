add_executable(wafer-run wafer-run.cpp)
target_link_libraries(wafer-run
  PRIVATE
  LLVMSupport
  WaferRuntime
)
