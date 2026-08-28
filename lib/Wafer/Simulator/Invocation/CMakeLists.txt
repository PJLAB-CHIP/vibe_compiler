add_mlir_library(WaferSimulatorInvocation
  ProgramInvocation.cpp
  ProgramTensorComparison.cpp
  TargetCallExecution.cpp
  TargetModelInvocation.cpp

  LINK_LIBS PUBLIC
  LLVMCore
  LLVMExecutionEngine
  LLVMJITLink
  LLVMOrcJIT
  LLVMRuntimeDyld
  WaferCodeGen
  WaferFrontend
  WaferProgramData
  WaferPhysicalTensor
  WaferTarget
  WaferSimulatorMemory
  LLVMSupport
  ${WAFER_NATIVE_JIT_LIBS}
)
