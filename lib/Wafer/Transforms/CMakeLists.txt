add_subdirectory(Tile)
add_subdirectory(Linalg)
add_subdirectory(StableHLO)

add_mlir_library(WaferTransforms
  Module/MaterializeExecutionMesh.cpp
  Module/MaterializeTargetTopology.cpp
  Module/StripTargetMetadata.cpp
  Instr/DirectDTETransport.cpp
  Instr/LifetimeAnalysis.cpp
  Instr/MiniMallocPacking.cpp
  Instr/NativeDirectDTEMultiSend.cpp
  Instr/NCCJoinPlacement.cpp
  Instr/PlanDDRMemory.cpp
  Instr/PlanSPMMemory.cpp
  Instr/RedundantTransferElimination.cpp
  Instr/StaticMemoryPacking.cpp
  Instr/TileMemoryPlanning.cpp
  Passes.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  WaferLinalgTransforms
  WaferPlanning
  WaferStableHLOTransforms
  WaferTileTransforms
  WaferTarget
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
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
