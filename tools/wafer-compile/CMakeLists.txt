add_executable(wafer-compile wafer-compile.cpp)
target_link_libraries(wafer-compile PRIVATE LLVMSupport)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-compile PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-compile
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
  target_compile_definitions(wafer-compile PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(wafer-compile PRIVATE ShardySdyRegister)
endif()
