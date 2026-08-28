add_mlir_library(WaferLinalgTransforms
  AttentionMatching.cpp
  ConstantTensorFolding.cpp
  NormalizeAttention.cpp
  NormalizeStructuredGraph.cpp
  StructuredGraphEGraph.cpp
  StructuredTiling.cpp

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
  MLIRTilingInterface
)

target_include_directories(obj.WaferLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(WaferLinalgTransforms PRIVATE
  WaferThirdPartyStructuredEGraph)
add_dependencies(WaferLinalgTransforms WaferStructuredEGraphBuild)
