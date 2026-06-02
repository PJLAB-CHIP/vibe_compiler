add_mlir_library(WaferFrontend
  Program.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
