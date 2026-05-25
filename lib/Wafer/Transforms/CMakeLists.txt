add_mlir_library(WaferTransforms
  NormalizeConstants.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(WaferTransforms PUBLIC StablehloOps)
endif()
