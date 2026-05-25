add_executable(wafer-import-model wafer-import-model.cpp)
target_link_libraries(wafer-import-model PRIVATE LLVMSupport)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(wafer-import-model PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(wafer-import-model
    PRIVATE
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    StablehloRegister
  )
endif()
