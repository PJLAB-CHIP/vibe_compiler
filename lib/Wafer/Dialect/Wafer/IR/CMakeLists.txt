add_mlir_dialect_library(WaferIR
  WaferDialect.cpp

  DEPENDS
  WaferAttrsIncGen
  WaferOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
