add_mlir_library(WaferPipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  WaferIR
  WaferTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
)

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferPipelines PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferPipelines PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(WaferPipelines PUBLIC ShardySdyTransforms)
endif()

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferPipelines PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferPipelines PRIVATE WAFER_ENABLE_STABLEHLO=1)
endif()
