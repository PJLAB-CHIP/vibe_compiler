add_mlir_library(WaferRuntime
  PARTIAL_SOURCES_INTENDED

  Board/BoardRuntime.cpp
  Profile/ProfileInstrumentation.cpp
  Profile/ProfilerRecord.cpp
  Invocation/RuntimeInvocationPlanning.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
  WaferPackageSupport
)
