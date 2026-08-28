add_mlir_library(WaferPlanning
  PhysicalDataflow/AttentionSpatialConstraints.cpp
  PhysicalDataflow/CanonicalSpatialAssignment.cpp
  PhysicalDataflow/CanonicalRegionPlan.cpp
  PhysicalDataflow/ExactDemand.cpp
  PhysicalDataflow/RootRegionWorkAnalysis.cpp
  PhysicalDataflow/RootWorkDomain.cpp
  PhysicalDataflow/PlanningProblem.cpp
  PhysicalDataflow/PlanningMemo.cpp
  PhysicalDataflow/PlanningProfile.cpp
  PhysicalDataflow/PlanningSession.cpp
  PhysicalDataflow/PlanningState.cpp
  PhysicalDataflow/RegionPlan.cpp
  PhysicalDataflow/RegionDomain.cpp
  PhysicalDataflow/ExactPBQPSolver.cpp
  PhysicalDataflow/TemporalDomain.cpp
  PhysicalDataflow/StructuredDemandAnalysis.cpp
  PhysicalDataflow/SpatialDomain.cpp
  PhysicalDataflow/SpatialPlan.cpp
  PhysicalDataflow/StructuredDemandView.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferAnalysis
  WaferIR
  WaferTarget
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
)

target_include_directories(obj.WaferPlanning PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferPlanning PRIVATE ${PROJECT_SOURCE_DIR}/lib)
