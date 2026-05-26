add_mlir_library(WaferTransforms
  Support/AttentionGemmUtils.cpp
  StableHLOToLinalg/NormalizeConstants.cpp
  StableHLOToLinalg/LowerStablehloDot.cpp
  StableHLOToLinalg/LowerStablehloElementwise.cpp
  StableHLOToLinalg/LowerStablehloCollectivesToComm.cpp
  StableHLOToLinalg/LowerStablehloReduce.cpp
  StableHLOToLinalg/LowerStablehloShape.cpp
  GroupFormation/FormGroups.cpp
  GroupScheduling/CheckMlpSchedule.cpp
  GroupScheduling/CheckNormSchedule.cpp
  GroupScheduling/CheckProjectionResidualSchedule.cpp
  GroupScheduling/CheckRootTileCandidates.cpp
  GroupScheduling/CheckSoftmaxSchedule.cpp
  TileRegionMaterialization/MaterializeMultiTileNoComm.cpp
  TileRegionMaterialization/MaterializeSingleTile.cpp
  SPMBufferize/CheckSPMAllocation.cpp
  LayoutMaterialization/CompactLayoutAssignment.cpp
  LaunchOutlining/MaterializeDDRExternalBindings.cpp
  Communication/LowerRingCollectives.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)

target_include_directories(obj.WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferTransforms PRIVATE StablehloOps)
  target_link_libraries(WaferTransforms PRIVATE StablehloOps)
endif()
