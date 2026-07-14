wafer_require_gtest()

add_executable(WaferUnitTests
  Compiler/CompilationTest.cpp
  Compiler/DirectDTETransportTest.cpp
  Compiler/ReferenceExecutorTest.cpp
  Compiler/TargetArtifactTest.cpp
  Compiler/TargetProfileTest.cpp
  Conversion/WaferGroupToTileRegionTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Runtime/PackageManifestTest.cpp
  Target/FormalNumericTest.cpp
  Target/NumericCodecTest.cpp
  Target/NumericDependencyConformanceTest.cpp
  Target/NumericRawRounderTest.cpp
  Target/NumericSemanticsTest.cpp
  Target/TargetFormatTest.cpp
)
llvm_update_compile_flags(WaferUnitTests)
find_program(WAFER_TEST_LLVM_CLANGXX NAMES clang++ REQUIRED)
target_compile_definitions(WaferUnitTests PRIVATE
  WAFER_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  WAFER_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_TEST_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
  WAFER_TEST_LLVM_CLANGXX="${WAFER_TEST_LLVM_CLANGXX}"
  WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR="${CMAKE_SOURCE_DIR}/test/Tools/Inputs/minimal-target-data-entry.ll"
)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferCompiler
  WaferIR
  WaferRuntime
  WaferPipelines
  WaferTransforms
  MLIRAsyncDialect
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

add_custom_target(check-wafer-unit
  COMMAND $<TARGET_FILE:WaferUnitTests>
  DEPENDS WaferUnitTests
  USES_TERMINAL
)

if(NOT WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  if(NOT CMAKE_READELF OR NOT EXISTS "${CMAKE_READELF}")
    message(FATAL_ERROR
      "feature-off numeric link closure requires the configured readelf tool")
  endif()
  if(NOT CMAKE_NM OR NOT EXISTS "${CMAKE_NM}")
    message(FATAL_ERROR
      "feature-off numeric link closure requires the configured nm tool")
  endif()
  add_test(NAME wafer-numeric-feature-off-link-closure
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/check_numeric_link_closure.py
      --binary $<TARGET_FILE:WaferUnitTests>
      --readelf ${CMAKE_READELF}
      --nm ${CMAKE_NM}
  )
  set_tests_properties(wafer-numeric-feature-off-link-closure PROPERTIES
    LABELS "numeric-model;link-closure"
  )
endif()

if(NOT WAFER_ENABLE_BULK_MODEL_DEPS)
  if(NOT CMAKE_READELF OR NOT EXISTS "${CMAKE_READELF}")
    message(FATAL_ERROR
      "feature-off bulk link closure requires the configured readelf tool")
  endif()
  if(NOT CMAKE_NM OR NOT EXISTS "${CMAKE_NM}")
    message(FATAL_ERROR
      "feature-off bulk link closure requires the configured nm tool")
  endif()
  add_test(NAME wafer-bulk-feature-off-link-closure
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/check_bulk_link_closure.py
      --binary $<TARGET_FILE:WaferUnitTests>
      --readelf ${CMAKE_READELF}
      --nm ${CMAKE_NM}
  )
  set_tests_properties(wafer-bulk-feature-off-link-closure PROPERTIES
    LABELS "bulk-model;link-closure"
  )
endif()

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_executable(WaferNumericModelUnitTests
    Target/FormalTensorNumericTest.cpp
    Target/MPFRNumericTest.cpp
    Target/NumericManagedDependencyIntegrationTest.cpp
    Target/NumericOracleDifferentialTest.cpp
    Target/SoftFloatOracleTest.cpp
  )
  llvm_update_compile_flags(WaferNumericModelUnitTests)
  target_compile_definitions(WaferNumericModelUnitTests PRIVATE
    WAFER_NUMERIC_MODEL_DEPS_ROOT="${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}"
    WAFER_NUMERIC_MODEL_DEPS_RECORD="${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD}"
    WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256="${WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256}"
  )
  target_link_libraries(WaferNumericModelUnitTests
    PRIVATE
    WaferNumericModel
    GTest::gtest_main
  )
  add_test(NAME WaferNumericModelUnitTests COMMAND WaferNumericModelUnitTests)
  add_test(NAME wafer-numeric-managed-load-integration
    COMMAND WaferNumericModelUnitTests
      --gtest_filter=NumericManagedDependencyIntegrationTest.*
  )
  set_tests_properties(
    WaferNumericModelUnitTests
    wafer-numeric-managed-load-integration
    PROPERTIES
    ENVIRONMENT
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
    LABELS "numeric-model;managed-dependencies"
  )
  add_custom_target(check-wafer-numeric-model-unit
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferNumericModelUnitTests>
    DEPENDS WaferNumericModelUnitTests
    USES_TERMINAL
  )
endif()

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  add_executable(WaferBulkModelUnitTests
    Target/BulkQualificationTest.cpp
  )
  llvm_update_compile_flags(WaferBulkModelUnitTests)
  target_link_libraries(WaferBulkModelUnitTests
    PRIVATE
    WaferBulkModel
    GTest::gtest_main
  )
  add_test(NAME WaferBulkModelUnitTests COMMAND WaferBulkModelUnitTests)
  set_tests_properties(WaferBulkModelUnitTests PROPERTIES
    LABELS "bulk-model;managed-dependencies"
  )
  add_custom_target(check-wafer-bulk-model-unit
    COMMAND $<TARGET_FILE:WaferBulkModelUnitTests>
    DEPENDS WaferBulkModelUnitTests
    USES_TERMINAL
  )
endif()
