add_mlir_library(WaferRuntime
  Board/BoardRuntime.cpp
  Profile/ProfileInstrumentation.cpp
  Profile/ProfilerRecord.cpp
  Invocation/RuntimeInvocationPlanning.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
  WaferPackage
)
