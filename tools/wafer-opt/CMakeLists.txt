add_executable(wafer-opt
  PipelineRegistration.cpp
  wafer-opt.cpp)
target_link_libraries(wafer-opt
  PRIVATE
  WaferConversionPasses
  WaferIR
  WaferStableHLOPipelines
  WaferTileToInstrPipelines
  WaferTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRArithTransforms
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransforms
  MLIRTransforms
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-opt PRIVATE StablehloRegister)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(wafer-opt PRIVATE ShardySdyRegister ShardySdyTransforms)
endif()
