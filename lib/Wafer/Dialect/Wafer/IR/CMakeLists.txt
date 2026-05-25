add_mlir_dialect_library(WaferIR
  WaferDialect.cpp
  WaferInterfaces.cpp

  DEPENDS
  WaferAttrsIncGen
  WaferInterfacesIncGen
  WaferOpsIncGen

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
