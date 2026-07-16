add_mlir_library(WaferAnalysis
  Scheduling/LayoutPlanningAnalysis.cpp
  Scheduling/TilingDemandAnalysis.cpp
  ScheduleCost/ExecutionCost.cpp
  ScheduleCost/SPMHighWater.cpp
  ScheduleCostAnalysis.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
)
