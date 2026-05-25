add_mlir_library(WaferTransforms
  LowerStablehloDot.cpp
  LowerStablehloElementwise.cpp
  LowerStablehloShape.cpp
  NormalizeConstants.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
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
