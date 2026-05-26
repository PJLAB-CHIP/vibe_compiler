wafer_require_gtest()

add_executable(WaferUnitTests
  ABI/M0AbiTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferABI
  WaferIR
  MLIRIR
  MLIRParser
  GTest::gtest_main
)

add_test(NAME WaferUnitTests COMMAND WaferUnitTests)
