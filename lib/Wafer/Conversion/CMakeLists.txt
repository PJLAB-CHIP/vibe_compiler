add_mlir_library(WaferConversion
  Passes.cpp
  WaferToCABI/LowerToCAbiSkeleton.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRPass
  MLIRSupport
)
