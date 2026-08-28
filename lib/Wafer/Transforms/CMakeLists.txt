add_subdirectory(Linalg)

add_mlir_library(WaferTransforms
  StableHLO/FrontendVerification.cpp
  StableHLO/SPMD/ApplyDefaultSpmdSharding.cpp
  StableHLO/SPMD/Pipelines.cpp
  Module/MaterializeExecutionMesh.cpp
  Module/MaterializeTargetTopology.cpp
  Module/StripTargetMetadata.cpp
  Tile/ExecutionStructure.cpp
  Instr/BufferizeInstrFunctionBoundaries.cpp
  Instr/LifetimeAnalysis.cpp
  Instr/MiniMallocPacking.cpp
  Instr/NCCJoinPlacement.cpp
  Instr/Pipelines.cpp
  Instr/PlanDDRMemory.cpp
  Instr/PlanSPMMemory.cpp
  Instr/RedundantTransferElimination.cpp
  Instr/StaticIndexRange.cpp
  Instr/StaticMemoryPacking.cpp
  Passes.cpp

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferFrontend
  WaferIR
  WaferInstrToLLVM
  WaferLinalgTransforms
  WaferStableHLOToLinalg
  WaferTarget
  WaferTileToInstr
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
)

set(WAFER_COMPILATION_TRANSFORM_SOURCES
  Transforms/Tile/CurrentIRLayoutOptimization.cpp
  Transforms/Tile/StructuredBufferRelations.cpp
  Transforms/Tile/StructuredNodeUseIndex.cpp
  Transforms/Instr/DirectDTETransport.cpp
  PARENT_SCOPE)

target_include_directories(obj.WaferTransforms PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTransforms PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
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
