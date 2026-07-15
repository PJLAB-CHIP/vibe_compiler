add_mlir_library(WaferFrontend
  DistributedBoundary.cpp
  DistributedSupport.cpp
  FunctionBoundary.cpp
  NpyPayload.cpp
  ParameterShards.cpp
  Program.cpp
  ProgramMetadata.cpp
  ProgramSupport.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  WaferIR
)
