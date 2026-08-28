add_executable(wafer-cmodel-qualify-onednn wafer-cmodel-qualify-onednn.cpp)
llvm_update_compile_flags(wafer-cmodel-qualify-onednn)
target_link_libraries(wafer-cmodel-qualify-onednn
  PRIVATE
  LLVMSupport
  WaferOneDNNBackend
  WaferCompiler
  WaferSimulatorInvocation
)
