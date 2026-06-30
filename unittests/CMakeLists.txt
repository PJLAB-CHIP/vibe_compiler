wafer_require_gtest()

add_executable(WaferUnitTests
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Runtime/HostRuntimeTest.cpp
)
add_library(WaferFakeTxRuntime SHARED Runtime/FakeTxRuntime.cpp)
add_dependencies(WaferUnitTests WaferFakeTxRuntime)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferIR
  WaferRuntime
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
target_compile_definitions(WaferUnitTests
  PRIVATE
  WAFER_FAKE_TX_RUNTIME_PATH="$<TARGET_FILE:WaferFakeTxRuntime>"
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(WaferUnitTests PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(WaferUnitTests PRIVATE StablehloRegister)
endif()

add_test(NAME WaferUnitTests COMMAND WaferUnitTests)
