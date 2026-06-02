add_executable(wafer-compile-stablehlo wafer-compile-stablehlo.cpp)
target_link_libraries(wafer-compile-stablehlo PRIVATE LLVMSupport)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-compile-stablehlo PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-compile-stablehlo
    PRIVATE
    WaferFrontend
    WaferPipelines
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTensorDialect
    StablehloRegister
  )
endif()

if(WAFER_ENABLE_IMPORTER_DEPS AND WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(wafer-compile-stablehlo PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(wafer-compile-stablehlo PRIVATE ShardySdyRegister)
endif()
