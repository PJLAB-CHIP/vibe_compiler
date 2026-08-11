add_mlir_conversion_library(WaferCardProgramToTileModules
  WaferCardProgramToTileModules/WaferCardProgramToTileModules.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferCardProgramToTileModules

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(WaferTensorProgramToCardProgram
  WaferTensorProgramToCardProgram/WaferTensorProgramToCardProgram.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTensorProgramToCardProgram

  LINK_LIBS PUBLIC
  WaferIR
  WaferTensorProgramToTileRegion
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(WaferTensorProgramToTileRegion
  WaferTensorProgramToTileRegion/BidirectionalTiling.cpp
  WaferTensorProgramToTileRegion/BodyEmitter.cpp
  WaferTensorProgramToTileRegion/CandidateMaterialization.cpp
  WaferTensorProgramToTileRegion/CandidateSupport.cpp
  WaferTensorProgramToTileRegion/CollectiveLowering.cpp
  WaferTensorProgramToTileRegion/DependentDataflow.cpp
  WaferTensorProgramToTileRegion/GenericLowering.cpp
  WaferTensorProgramToTileRegion/NamedComputeLowering.cpp
  WaferTensorProgramToTileRegion/TensorControlFlowLowering.cpp
  WaferTensorProgramToTileRegion/TileMaterialization.cpp
  WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTensorProgramToTileRegion

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  MLIRAnalysis
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRDialectUtils
)

add_mlir_conversion_library(WaferTileRegionToInstr
  WaferTileRegionToInstr/ComputeLowering.cpp
  WaferTileRegionToInstr/MovementLowering.cpp
  WaferTileRegionToInstr/MovementSupport.cpp
  WaferTileRegionToInstr/PeerLowering.cpp
  WaferTileRegionToInstr/WaferTileRegionToInstr.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTileRegionToInstr

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
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
  StableHLOToLinalg/LowerStaticStablehloConcatenate.cpp
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
