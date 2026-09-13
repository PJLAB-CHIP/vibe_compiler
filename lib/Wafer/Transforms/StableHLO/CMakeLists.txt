add_mlir_library(WaferStableHLOTransforms
  ExpandStablehloBatchNormInference.cpp
  FrontendVerification.cpp
  FoldStablehloConstants.cpp
  SPMD/ApplyDefaultSpmdSharding.cpp
  SPMD/Pipelines.cpp

  DEPENDS
  WaferTransformPassesIncGen

  LINK_LIBS PUBLIC
  WaferFrontend
  WaferIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferStableHLOTransforms PRIVATE
    WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferStableHLOTransforms PRIVATE
    WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferStableHLOTransforms PRIVATE StablehloOps)
  target_link_libraries(WaferStableHLOTransforms PRIVATE StablehloOps)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferStableHLOTransforms PRIVATE
    WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferStableHLOTransforms PRIVATE
    WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferStableHLOTransforms PRIVATE
    ShardySdyDialect ShardySdyTransforms)
  target_link_libraries(WaferStableHLOTransforms PRIVATE
    ShardySdyDialect ShardySdyTransforms)
endif()
