wafer_require_gtest()

add_executable(WaferUnitTests
  ABI/TileAbiTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferABI
  WaferIR
  WaferPipelines
  WaferTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  GTest::gtest_main
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(WaferUnitTests PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(WaferUnitTests PRIVATE StablehloRegister)
endif()

add_test(NAME WaferUnitTests COMMAND WaferUnitTests)
