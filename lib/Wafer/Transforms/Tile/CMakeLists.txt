add_mlir_library(WaferTileTransforms
  ExecutionStructure.cpp
  LayoutOptimization.cpp
  StructuredBufferRelations.cpp
  StructuredNodeUseIndex.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferAnalysis
  WaferIR
  WaferPlanning
  MLIRAnalysis
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRViewLikeInterface
)

target_include_directories(obj.WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
