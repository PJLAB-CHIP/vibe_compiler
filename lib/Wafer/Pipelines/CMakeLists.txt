add_mlir_library(WaferPipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferConversion
  WaferIR
  WaferTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
)
