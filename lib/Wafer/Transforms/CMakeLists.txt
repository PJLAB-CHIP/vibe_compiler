add_mlir_library(WaferTransforms
  CheckRootTileCandidates.cpp
  FormGroups.cpp
  LowerStablehloDot.cpp
  LowerStablehloElementwise.cpp
  LowerStablehloReduce.cpp
  LowerStablehloShape.cpp
  NormalizeConstants.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(WaferTransforms PUBLIC StablehloOps)
endif()
