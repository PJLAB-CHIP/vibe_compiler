add_mlir_library(WaferSimulatorInvocation
  ProgramInvocation.cpp
  ProgramTensorComparison.cpp
  TargetCallExecution.cpp
  TargetModelInvocation.cpp

  LINK_LIBS PUBLIC
  WaferCompiler
  WaferPhysicalTensor
  WaferTarget
  WaferSimulatorMemory
  LLVMSupport
)
