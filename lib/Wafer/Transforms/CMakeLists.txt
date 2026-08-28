add_mlir_library(WaferTransforms
  FrontendVerification.cpp
  Bufferization/BufferizeInstrFunctionBoundaries.cpp
  Execution/ExecutionStructure.cpp
  Scheduling/RedundantTransferElimination.cpp
  DDR/PlanDDRMemory.cpp
  MemoryPlanning/LifetimeAnalysis.cpp
  MemoryPlanning/MiniMallocPacking.cpp
  MemoryPlanning/Pipelines.cpp
  MemoryPlanning/StaticIndexRange.cpp
  MemoryPlanning/StaticMemoryPacking.cpp
  SPM/PlanSPMMemory.cpp
  SPMD/ApplyDefaultSpmdSharding.cpp
  SPMD/Pipelines.cpp
  Target/MaterializeExecutionMesh.cpp
  Target/MaterializeTargetTopology.cpp
  Target/LowerInstrToTargetLLVM.cpp
  Target/TargetLoweringVerification.cpp
  Target/TargetCallIRAdapter.cpp
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
  WaferFrontend
  WaferIR
  WaferStableHLOToLinalg
  WaferTarget
  WaferTileRegionToInstr
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
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
  MLIRSCFTransforms
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
)

set(WAFER_COMPILATION_TRANSFORM_SOURCES
  Transforms/Tile/StructuredBufferRelations.cpp
  Transforms/Tile/StructuredNodeUseIndex.cpp
  Transforms/Transport/DirectDTETransport.cpp
  PARENT_SCOPE)

target_include_directories(obj.WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(obj.WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)
target_link_libraries(WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferTransforms PRIVATE
    ShardySdyDialect ShardySdyTransforms)
  target_link_libraries(WaferTransforms PRIVATE
    ShardySdyDialect ShardySdyTransforms)
endif()
