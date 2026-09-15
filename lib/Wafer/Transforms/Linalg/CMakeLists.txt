add_mlir_library(WaferLinalgTransforms
  AttentionMath.cpp
  AttentionMatching.cpp
  CommunicationRegionClosure.cpp
  ContractionAccumulation.cpp
  ConstantTensorFolding.cpp
  FuseBatchNormInference.cpp
  NormalizeAttention.cpp
  NormalizeStructuredGraph.cpp
  NormalizeTensorReads.cpp
  OnlineAttentionDecomposition.cpp
  OnlineAttentionMaterialization.cpp
  SpatialRegionMaterialization.cpp
  StructuredGraphEGraph.cpp
  StructuredTiling.cpp
  TemporalTiling.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  WaferAnalysis
  WaferIR
  WaferPlanning
  WaferTileTransforms
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithUtils
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTilingInterface
  MLIRTransforms
)

target_include_directories(obj.WaferLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(WaferLinalgTransforms PRIVATE
  WaferStructuredEGraph)
add_dependencies(WaferLinalgTransforms WaferStructuredEGraphBuild)
