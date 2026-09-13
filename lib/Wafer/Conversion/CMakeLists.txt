add_mlir_conversion_library(WaferTileToInstr
  TileToInstr/ComputeLowering.cpp
  TileToInstr/MovementLowering.cpp
  TileToInstr/MovementSupport.cpp
  TileToInstr/PeerLowering.cpp
  TileToInstr/TileToInstr.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/TileToInstr

  DEPENDS
  WaferConversionPassesIncGen

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

target_include_directories(obj.WaferTileToInstr PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(WaferInstrToLLVM
  InstrToLLVM/ComputeTargetCallLowering.cpp
  InstrToLLVM/DirectDTETargetCallLowering.cpp
  InstrToLLVM/InstructionTargetCallLowering.cpp
  InstrToLLVM/LowerInstrToTargetLLVM.cpp
  InstrToLLVM/MovementTargetCallLowering.cpp
  InstrToLLVM/PeripheralTargetCallLowering.cpp
  InstrToLLVM/SyncTargetCallLowering.cpp
  InstrToLLVM/TargetCallIRAdapter.cpp
  InstrToLLVM/TargetCallLoweringSupport.cpp
  InstrToLLVM/TargetLLVMConversion.cpp
  InstrToLLVM/TargetLLVMConversionPatterns.cpp
  InstrToLLVM/TargetLLVMStructure.cpp
  InstrToLLVM/TargetLoweringVerification.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Wafer/Conversion/InstrToLLVM

  DEPENDS
  WaferConversionPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  WaferTarget
  WaferTileToInstr
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
)

target_include_directories(obj.WaferInstrToLLVM PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferInstrToLLVM PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)

add_mlir_conversion_library(WaferStableHLOToLinalg
  StableHLOToLinalg/FoldConvolutionInputCasts.cpp
  StableHLOToLinalg/LegalizeStablehloToLinalg.cpp
  StableHLOToLinalg/LowerStaticStablehloConcatenate.cpp
  StableHLOToLinalg/NormalizeStablehloCollectives.cpp
  StableHLOToLinalg/PromoteStablehloLogistic.cpp

  DEPENDS
  WaferConversionPassesIncGen

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

add_library(WaferConversionPasses INTERFACE)
target_link_libraries(WaferConversionPasses INTERFACE
  WaferInstrToLLVM
  WaferStableHLOToLinalg
  WaferTileToInstr
)

add_mlir_library(WaferStableHLOPipelines
  StableHLOToLinalg/Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferLinalgTransforms
  WaferStableHLOToLinalg
  WaferStableHLOTransforms
  WaferTransforms
  MLIRPass
  MLIRTransforms
)

add_mlir_library(WaferTileToInstrPipelines
  TileToInstr/Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferIR
  WaferTileToInstr
  WaferTransforms
  MLIRFuncDialect
  MLIRPass
)
