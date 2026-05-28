add_mlir_library(WaferConversion
  Passes.cpp
  WaferToCABI/LowerTileRegionToCAbi.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRPass
  MLIRSupport
)
