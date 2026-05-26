add_mlir_library(WaferFrontend
  Artifact.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
