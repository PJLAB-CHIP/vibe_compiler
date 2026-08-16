add_mlir_library(WaferRuntime
  PARTIAL_SOURCES_INTENDED

  BoardRuntime.cpp
  ProfileInstrumentation.cpp
  ProfilerRecord.cpp
  RuntimeInvocationPlanning.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
  WaferPackageSupport
)
