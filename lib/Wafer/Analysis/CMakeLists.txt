set(LLVM_OPTIONAL_SOURCES
  CollectiveTopologyAnalysis.cpp
  ExecutionTopologyAnalysis.cpp
)

add_mlir_library(WaferAnalysis
  DirectCallGraphAnalysis.cpp
  PhysicalDataflow/IndexRelation.cpp
  PhysicalDataflow/PhysicalAccessRelation.cpp
  PhysicalDataflow/PhysicalLayoutRelation.cpp
  PhysicalDataflow/TransferRealizability.cpp
  TheoreticalScheduleCostAnalysis.cpp
  ScheduleCost/ExecutionCost.cpp
  ScheduleCost/DDRHighWater.cpp
  ScheduleCost/SPMHighWater.cpp
  ScheduleCostAnalysis.cpp
  StaticBufferRange.cpp
  TargetSchedulingAnalysis.cpp

  LINK_LIBS PUBLIC
  WaferIR
  WaferTarget
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
