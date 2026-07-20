add_mlir_library(WaferAnalysis
  PhysicalDataflow/IndexRelation.cpp
  PhysicalDataflow/TransferRealizability.cpp
  Scheduling/LayoutPlanningAnalysis.cpp
  Scheduling/TilingDemandAnalysis.cpp
  ScheduleCost/ExecutionCost.cpp
  ScheduleCost/SPMHighWater.cpp
  ScheduleCostAnalysis.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRAffineAnalysis
  MLIRAsyncDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPresburger
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
)
