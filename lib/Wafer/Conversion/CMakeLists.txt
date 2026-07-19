add_mlir_conversion_library(WaferTensorProgramToTileRegion
  WaferTensorProgramToTileRegion/BodyEmitter.cpp
  WaferTensorProgramToTileRegion/CandidateMaterialization.cpp
  WaferTensorProgramToTileRegion/CandidateSupport.cpp
  WaferTensorProgramToTileRegion/CollectiveLowering.cpp
  WaferTensorProgramToTileRegion/CompleteTraversal.cpp
  WaferTensorProgramToTileRegion/GenericLowering.cpp
  WaferTensorProgramToTileRegion/NamedComputeLowering.cpp
  WaferTensorProgramToTileRegion/TensorControlFlowLowering.cpp
  WaferTensorProgramToTileRegion/TileMaterialization.cpp
  WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTensorProgramToTileRegion

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  WaferSupport
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRDialectUtils
)

add_mlir_conversion_library(WaferTileRegionToInstr
  WaferTileRegionToInstr/CollectiveLowering.cpp
  WaferTileRegionToInstr/ComputeLowering.cpp
  WaferTileRegionToInstr/MovementLowering.cpp
  WaferTileRegionToInstr/MovementSupport.cpp
  WaferTileRegionToInstr/WaferTileRegionToInstr.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTileRegionToInstr

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
)

add_mlir_conversion_library(WaferStableHLOToLinalg
  StableHLOToLinalg/ConstantTensorFolding.cpp
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
