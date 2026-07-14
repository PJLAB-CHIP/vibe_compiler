add_executable(wafer-cmodel-qualify-bulk wafer-cmodel-qualify-bulk.cpp)
llvm_update_compile_flags(wafer-cmodel-qualify-bulk)
target_link_libraries(wafer-cmodel-qualify-bulk
  PRIVATE
  LLVMSupport
  WaferBulkModel
)
