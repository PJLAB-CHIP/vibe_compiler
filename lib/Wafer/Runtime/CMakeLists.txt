add_mlir_library(WaferRuntime
  HostRuntime.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
)

target_link_libraries(WaferRuntime PUBLIC ${CMAKE_DL_LIBS})
