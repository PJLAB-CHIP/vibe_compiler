add_mlir_library(WaferAnalysis
  Scheduling/LayoutPlanningAnalysis.cpp
  Scheduling/TilingDemandAnalysis.cpp
  StructuredNormalization/IdentityViewProof.cpp
  StructuredNormalization/StaticTripCountProof.cpp
  StructuredNormalization/DPSInitAnalysis.cpp
  StructuredNormalization/SchedulableCallClosure.cpp
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
  MLIRDialectUtils
  MLIRSupport
  MLIRTensorDialect
)
