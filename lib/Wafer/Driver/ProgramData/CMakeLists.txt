add_mlir_library(WaferProgramData
  ProgramData.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferFrontend
)
