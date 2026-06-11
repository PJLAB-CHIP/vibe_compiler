add_mlir_library(WaferTransforms
  Group/DumpCandidateDdrTileViews.cpp
  Group/DumpLayoutPlan.cpp
  Group/DumpTilingDemand.cpp
  Group/DumpGroupToTileRegion.cpp
  Group/FormLogicalGroups.cpp
  DDR/AcceptDDRMemoryPlan.cpp
  SPM/PlanSPMMemory.cpp
  SPMD/ApplyDefaultSpmdSharding.cpp
  Passes.cpp

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferGroupToTileRegion
  WaferIR
  WaferStableHLOToLinalg
  WaferTileRegionToInstr
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

target_include_directories(obj.WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferTransforms PRIVATE ShardySdyDialect)
  target_link_libraries(WaferTransforms PRIVATE ShardySdyDialect)
endif()
