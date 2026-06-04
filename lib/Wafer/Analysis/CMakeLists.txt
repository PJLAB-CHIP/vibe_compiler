add_mlir_library(WaferAnalysis
  Group/LayoutPlanningAnalysis.cpp
  Group/TilingDemandAnalysis.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRSupport
  MLIRTensorDialect
)
