add_mlir_library(WaferAnalysis
  CollectiveTopologyAnalysis.cpp
  ExecutionTopologyAnalysis.cpp
  PhysicalDataflow/IndexRelation.cpp
  PhysicalDataflow/PhysicalAccessRelation.cpp
  PhysicalDataflow/TransferRealizability.cpp
  NoCProfitabilityAnalysis.cpp
  ScheduleCost/ExecutionCost.cpp
  ScheduleCost/SPMHighWater.cpp
  ScheduleCostAnalysis.cpp
  StaticBufferRange.cpp

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
