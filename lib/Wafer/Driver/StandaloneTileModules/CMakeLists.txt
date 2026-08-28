add_mlir_library(WaferStandaloneTileModules
  StandaloneTileModules.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Driver/StandaloneTileModules

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRSupport
)
