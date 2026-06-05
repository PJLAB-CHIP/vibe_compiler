add_mlir_conversion_library(WaferGroupToTileRegion
  WaferGroupToTileRegion/WaferGroupToTileRegion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferGroupToTileRegion

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

add_mlir_conversion_library(WaferStableHLOToLinalg
  StableHLOToLinalg/LegalizeStablehloToLinalg.cpp
  StableHLOToLinalg/NormalizeStablehloCollectives.cpp

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferStableHLOToLinalg PRIVATE
    WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferStableHLOToLinalg PRIVATE
    WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferStableHLOToLinalg PRIVATE StablehloOps)
  target_link_libraries(obj.WaferStableHLOToLinalg PRIVATE StablehloLinalgTransforms)
  target_link_libraries(WaferStableHLOToLinalg PRIVATE StablehloOps)
  target_link_libraries(WaferStableHLOToLinalg PRIVATE StablehloLinalgTransforms)
endif()
