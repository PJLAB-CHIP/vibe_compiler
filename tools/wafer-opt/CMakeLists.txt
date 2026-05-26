add_executable(wafer-opt wafer-opt.cpp)
target_link_libraries(wafer-opt
  PRIVATE
  WaferConversion
  WaferIR
  WaferTransforms
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRFuncDialect
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
