add_mlir_library(WaferTileTransforms
  BoundaryMovement.cpp
  BooleanReduction.cpp
  DistributedCollectiveMovement.cpp
  ElementwisePayloads.cpp
  ExecutionStructure.cpp
  GemmFinalization.cpp
  GatherLowering.cpp
  LayoutOptimization.cpp
  LoopSubsetState.cpp
  PhysicalMovementPlacement.cpp
  ScalarExecution.cpp
  StorageInitialization.cpp
  Pipelines.cpp
  AccessReuse.cpp
  StructuredToTile.cpp
  StructuredBufferRelations.cpp
  TiledOutputStores.cpp
  TensorInitialization.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferAnalysis
  WaferIR
  WaferPlanning
  MLIRAnalysis
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLoopLikeInterface
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSubsetOpInterface
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)

target_include_directories(obj.WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferTileTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
