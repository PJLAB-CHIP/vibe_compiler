add_mlir_conversion_library(WaferCardModuleToTileModules
  WaferCardModuleToTileModules/WaferCardModuleToTileModules.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferCardModuleToTileModules

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(WaferTensorProgramToCardModule
  WaferTensorProgramToCardModule/CardModuleAssembly.cpp
  WaferTensorProgramToCardModule/CardModuleCommon.cpp
  WaferTensorProgramToCardModule/MappingValidation.cpp
  WaferTensorProgramToCardModule/SourceValidation.cpp
  WaferTensorProgramToCardModule/TileEntryMaterialization.cpp
  WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/WaferTensorProgramToCardModule

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
  WaferTensorProgramToTileRegion/CoupledRegionFunction.cpp
  WaferTensorProgramToTileRegion/ConfiguredReductionTraversal.cpp
  WaferTensorProgramToTileRegion/ConfiguredStructuredTraversal.cpp
  WaferTensorProgramToTileRegion/DependentDataflow.cpp
  WaferTensorProgramToTileRegion/EdgeDomain.cpp
  WaferTensorProgramToTileRegion/EdgeFragmentPlanning.cpp
  WaferTensorProgramToTileRegion/GenericLowering.cpp
  WaferTensorProgramToTileRegion/NamedComputeLowering.cpp
  WaferTensorProgramToTileRegion/PartialReductionRegion.cpp
  WaferTensorProgramToTileRegion/ProducerTileFusion.cpp
  WaferTensorProgramToTileRegion/ConsumerInputReconstruction.cpp
  WaferTensorProgramToTileRegion/RootRegionEmitter.cpp
  WaferTensorProgramToTileRegion/SingleRootTileRegion.cpp
  WaferTensorProgramToTileRegion/SingleRootRegionFunction.cpp
  WaferTensorProgramToTileRegion/SelectedEdgeActions.cpp
  WaferTensorProgramToTileRegion/SelectedEdgeFinalization.cpp
  WaferTensorProgramToTileRegion/SelectedEdgeMaterialization.cpp
  WaferTensorProgramToTileRegion/SelectedEdgeMapping.cpp
  WaferTensorProgramToTileRegion/SelectedPeerMaterialization.cpp
  WaferTensorProgramToTileRegion/StructuredIterationTile.cpp
  WaferTensorProgramToTileRegion/TensorControlFlowLowering.cpp
  WaferTensorProgramToTileRegion/TemporalTileMaterialization.cpp
  WaferTensorProgramToTileRegion/TemporalRegionTraversal.cpp
  WaferTensorProgramToTileRegion/TemporalPartialReductionTraversal.cpp
  WaferTensorProgramToTileRegion/TemporalWaveLoop.cpp
  WaferTensorProgramToTileRegion/TensorDependencyChain.cpp
  WaferTensorProgramToTileRegion/TileMaterialization.cpp
  WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.cpp
  WaferTensorProgramToTileRegion/WindowedInsertSlice.cpp

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
