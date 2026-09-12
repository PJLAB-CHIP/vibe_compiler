add_mlir_library(WaferTileTransforms
  BoundaryMovement.cpp
  DistributedCollectiveMovement.cpp
  ExecutionStructure.cpp
  GemmFinalization.cpp
  LayoutOptimization.cpp
  PhysicalMovementPlacement.cpp
  Pipelines.cpp
  ReadOnlyInputSharing.cpp
  StructuredToTile.cpp
  StructuredBufferRelations.cpp
  TiledOutputStores.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferAnalysis
  WaferIR
  WaferPlanning
  MLIRAnalysis
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)

target_include_directories(obj.WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
