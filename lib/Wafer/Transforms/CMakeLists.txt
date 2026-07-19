add_mlir_library(WaferTransforms
  Scheduling/CandidateAnalysis.cpp
  Scheduling/CandidateCommit.cpp
  Scheduling/CandidateEvaluation.cpp
  Scheduling/CandidateSelection.cpp
  Scheduling/FullBufferHandoff.cpp
  Scheduling/ScheduleTensorProgram.cpp
  Scheduling/StructuredSchedulingScope.cpp
  StructuredOptimization/QualificationTelemetry.cpp
  StructuredOptimization/DebugQualificationInstrumentation.cpp
  StructuredOptimization/RequiredTensorNormalization.cpp
  StructuredOptimization/TensorNormalFormVerifier.cpp
  StructuredOptimization/SelectedPayloadNormalization.cpp
  DDR/PlanDDRMemory.cpp
  MemoryPlanning/LifetimeAnalysis.cpp
  MemoryPlanning/MiniMallocPacking.cpp
  MemoryPlanning/StaticMemoryPacking.cpp
  SPM/PlanSPMMemory.cpp
  SPMD/ApplyDefaultSpmdSharding.cpp
  Target/MaterializeExecutionMesh.cpp
  Target/MaterializeTargetTopology.cpp
  Target/LowerInstrToTargetLLVM.cpp
  Target/TargetCallPreflight.cpp
  Target/TargetCallLoweringSupport.cpp
  Target/MovementTargetCallLowering.cpp
  Target/ComputeTargetCallLowering.cpp
  Target/DirectDTETargetCallLowering.cpp
  Target/PeripheralTargetCallLowering.cpp
  Target/SyncTargetCallLowering.cpp
  Target/InstructionTargetCallLowering.cpp
  Target/TargetLLVMStructure.cpp
  Target/TargetLLVMConversionPatterns.cpp
  Target/TargetLLVMConversion.cpp
  Target/StripTargetMetadata.cpp
  Passes.cpp

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferSupport
  WaferTensorProgramToTileRegion
  WaferIR
  WaferStableHLOToLinalg
  WaferTarget
  WaferTileRegionToInstr
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

target_include_directories(obj.WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(obj.WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)
target_link_libraries(WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferTransforms PRIVATE ShardySdyDialect)
  target_link_libraries(WaferTransforms PRIVATE ShardySdyDialect)
endif()
