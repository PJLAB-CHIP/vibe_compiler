add_mlir_conversion_library(WaferTileModuleFanout
  WaferTileModuleFanout/WaferTileModuleFanout.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTileModuleFanout

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(WaferStructuredTiling
  StructuredTiling/StructuredTiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
)

set(LLVM_TARGET_DEFINITIONS WaferTileRegionToInstr/Patterns.td)
mlir_tablegen(WaferTileRegionToInstr/Patterns.inc -gen-rewriters)
add_public_tablegen_target(WaferTileRegionToInstrPatternsIncGen)

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
  WaferTileRegionToInstrPatternsIncGen

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

target_include_directories(obj.WaferTileRegionToInstr PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(WaferStableHLOToLinalg
  StableHLOToLinalg/AttentionMatching.cpp
  StableHLOToLinalg/ConstantTensorFolding.cpp
  StableHLOToLinalg/LegalizeStablehloToLinalg.cpp
  StableHLOToLinalg/LowerStaticStablehloConcatenate.cpp
  StableHLOToLinalg/NormalizeAttention.cpp
  StableHLOToLinalg/NormalizeStructuredGraph.cpp
  StableHLOToLinalg/NormalizeStablehloCollectives.cpp
  StableHLOToLinalg/StructuredGraphEGraph.cpp

  DEPENDS
  WaferPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
)

target_link_libraries(WaferStableHLOToLinalg PRIVATE
  WaferThirdPartyStructuredEGraph)
add_dependencies(WaferStableHLOToLinalg WaferStructuredEGraphBuild)

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

add_mlir_library(WaferStableHLOPipelines
  StableHLOToLinalg/Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferTransforms
  MLIRPass
  MLIRTransforms
)

add_mlir_library(WaferTileRegionPipelines
  WaferTileRegionToInstr/Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferIR
  WaferTransforms
  MLIRFuncDialect
  MLIRPass
)
