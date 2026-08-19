set(LLVM_OPTIONAL_SOURCES
  Topology/CollectiveTopologyAnalysis.cpp
  Topology/ExecutionTopologyAnalysis.cpp
)

add_mlir_library(WaferAnalysis
  CallGraph/DirectCallGraphAnalysis.cpp
  PhysicalDataflow/IndexRelation.cpp
  PhysicalDataflow/ExactDemand.cpp
  PhysicalDataflow/PhysicalAccessRelation.cpp
  PhysicalDataflow/PhysicalLayoutRelation.cpp
  PhysicalDataflow/TransferRealizability.cpp
  ScheduleCost/TheoreticalScheduleCostAnalysis.cpp
  ScheduleCost/ExecutionCost.cpp
  ScheduleCost/DDRHighWater.cpp
  ScheduleCost/SPMHighWater.cpp
  ScheduleCost/ScheduleCostAnalysis.cpp
  Memory/StaticBufferRange.cpp
  Scheduling/TargetSchedulingAnalysis.cpp

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

set(WAFER_COMPILATION_ANALYSIS_SOURCES
  Analysis/Executable/ExecutableCallClosure.cpp
  Analysis/Executable/ProgramResourceVerification.cpp
  Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.cpp
  Analysis/Structured/CardProgramAnalysis.cpp
  Analysis/Structured/StructuredBufferRelations.cpp
  Analysis/Structured/StructuredDAGAnalysis.cpp
  Analysis/Structured/StructuredNodeUseIndex.cpp
  Analysis/Structured/StructuredOperationTileFootprint.cpp
  PARENT_SCOPE)
