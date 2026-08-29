add_subdirectory(Linalg)
add_subdirectory(StableHLO)

add_mlir_library(WaferTransforms
  Module/MaterializeExecutionMesh.cpp
  Module/MaterializeTargetTopology.cpp
  Module/StripTargetMetadata.cpp
  Tile/ExecutionStructure.cpp
  Instr/BufferizeInstrFunctionBoundaries.cpp
  Instr/DirectDTETransport.cpp
  Instr/LifetimeAnalysis.cpp
  Instr/MiniMallocPacking.cpp
  Instr/NCCJoinPlacement.cpp
  Instr/Pipelines.cpp
  Instr/PlanDDRMemory.cpp
  Instr/PlanSPMMemory.cpp
  Instr/RedundantTransferElimination.cpp
  Instr/StaticMemoryPacking.cpp
  Instr/TileMemoryPlanning.cpp
  Passes.cpp
  Tile/LayoutOptimization.cpp
  Tile/StructuredBufferRelations.cpp
  Tile/StructuredNodeUseIndex.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  WaferLinalgTransforms
  WaferPlanning
  WaferStableHLOTransforms
  WaferTarget
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

target_include_directories(obj.WaferTransforms PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTransforms PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(obj.WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)
target_link_libraries(WaferTransforms PRIVATE WaferThirdPartyMiniMalloc)
