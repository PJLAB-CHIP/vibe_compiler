add_executable(wafer-opt wafer-opt.cpp)
target_link_libraries(wafer-opt
  PRIVATE
  LLVMSupport
  WaferFrontend
  WaferIR
  WaferPipelines
  WaferTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-opt PRIVATE StablehloRegister)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(wafer-opt PRIVATE
    WAFER_ENABLE_SHARDY=1
    WAFER_XLA_SPMD_PARTITIONER_HELPER="${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
  )
  target_link_libraries(wafer-opt PRIVATE ShardySdyRegister ShardySdyTransforms)
endif()
