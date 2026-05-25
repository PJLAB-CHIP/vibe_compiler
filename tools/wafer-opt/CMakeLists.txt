add_executable(wafer-opt wafer-opt.cpp)
target_link_libraries(wafer-opt
  PRIVATE
  WaferIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMlirOptMain
  MLIRSCFDialect
  MLIRTensorDialect
)
