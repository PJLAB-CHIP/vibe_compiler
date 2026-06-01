add_executable(wafer-opt wafer-opt.cpp)
target_link_libraries(wafer-opt
  PRIVATE
  WaferConversion
  WaferIR
  WaferPipelines
  WaferTransforms
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMlirOptMain
  MLIRSCFDialect
  MLIRTensorDialect
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-opt PRIVATE StablehloRegister)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(wafer-opt PRIVATE ShardySdyRegister ShardySdyTransforms)
endif()
