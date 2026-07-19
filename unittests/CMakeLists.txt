wafer_require_gtest()

add_executable(WaferUnitTests
  Analysis/ScheduleCostAnalysisTest.cpp
  Compiler/CompilationTest.cpp
  Compiler/DirectDTETransportTest.cpp
  Compiler/ProgramTensorComparisonTest.cpp
  Compiler/ScheduledRankFinalizationTest.cpp
  Compiler/TargetABIPreparationTest.cpp
  Compiler/TargetArtifactTest.cpp
  Compiler/TargetCallFrontendTest.cpp
  Compiler/TargetProfileTest.cpp
  Compiler/WholeVariantCoordinatorTest.cpp
  Compiler/WholeVariantResourceAcceptanceTest.cpp
  Conversion/WaferTensorProgramToTileRegionTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Frontend/NpyPayloadTest.cpp
  Model/TargetModelMemoryTest.cpp
  Pipelines/PipelinesTest.cpp
  Runtime/PackageManifestTest.cpp
  Target/FormalNumericTest.cpp
  Target/NumericCodecTest.cpp
  Target/NumericDependencyConformanceTest.cpp
  Target/NumericRawRounderTest.cpp
  Target/NumericSemanticsTest.cpp
  Target/PhysicalTensorCodecTest.cpp
  Target/TargetFormatTest.cpp
  Transforms/MemoryPlanning/LifetimeAnalysisTest.cpp
  Transforms/MemoryPlanning/MiniMallocPackingTest.cpp
  Transforms/MemoryPlanning/StaticMemoryPackingTest.cpp
  Transforms/Scheduling/CandidateSelectionTest.cpp
  Transforms/Scheduling/FullBufferHandoffTest.cpp
  Transforms/Scheduling/StructuredSchedulingScopeTest.cpp
  Transforms/Target/LowerInstrToTargetLLVMTest.cpp
)
llvm_update_compile_flags(WaferUnitTests)
target_include_directories(WaferUnitTests PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Wafer/Transforms
)
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
  WaferAnalysis
  WaferCompiler
  WaferIR
  WaferTargetModelCore
  WaferRuntime
  WaferPipelines
  WaferTensorProgramToTileRegion
  WaferTransforms
  MLIRAsyncDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
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

if(NOT WAFER_ENABLE_SYSTEMC_MODEL)
  if(NOT CMAKE_READELF OR NOT EXISTS "${CMAKE_READELF}")
    message(FATAL_ERROR
      "feature-off SystemC link closure requires the configured readelf tool")
  endif()
  if(NOT CMAKE_NM OR NOT EXISTS "${CMAKE_NM}")
    message(FATAL_ERROR
      "feature-off SystemC link closure requires the configured nm tool")
  endif()
  add_test(NAME wafer-systemc-feature-off-link-closure
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/check_systemc_link_closure.py
      --binary $<TARGET_FILE:WaferUnitTests>
      --readelf ${CMAKE_READELF}
      --nm ${CMAKE_NM}
  )
  set_tests_properties(wafer-systemc-feature-off-link-closure PROPERTIES
    LABELS "systemc-model;link-closure"
  )
endif()

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_executable(WaferNumericModelUnitTests
    Model/TargetModelKernelTest.cpp
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
    WaferTargetFunctionalModel
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
    Model/ManagedReferenceTargetModelTest.cpp
    Target/BulkQualificationTest.cpp
  )
  llvm_update_compile_flags(WaferBulkModelUnitTests)
  target_link_libraries(WaferBulkModelUnitTests
    PRIVATE
    WaferBulkModel
    WaferTargetBulkModel
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

if(WAFER_ENABLE_SYSTEMC_MODEL)
  add_library(WaferSystemCModelTestSupport STATIC
    Model/SystemCTargetModelTestSupport.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelTestSupport)
  target_link_libraries(WaferSystemCModelTestSupport
    PUBLIC
    WaferCompiler
    WaferIR
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
  )

  add_executable(WaferSystemCModelIntegrationTest
    Model/SystemCTargetModelIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelIntegrationTest)
  target_link_libraries(WaferSystemCModelIntegrationTest
    PRIVATE
    WaferSystemCModel
    WaferCompiler
    WaferIR
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelIntegrationTest
    COMMAND WaferSystemCModelIntegrationTest)

  add_executable(WaferSystemCModelBatchedGemmIntegrationTest
    Model/SystemCTargetModelBatchedGemmIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelBatchedGemmIntegrationTest)
  target_link_libraries(WaferSystemCModelBatchedGemmIntegrationTest
    PRIVATE
    WaferSystemCModel
    WaferCompiler
    WaferIR
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelBatchedGemmIntegrationTest
    COMMAND WaferSystemCModelBatchedGemmIntegrationTest)

  add_executable(WaferSystemCModelDTEIntegrationTest
    Model/SystemCTargetModelDTEIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelDTEIntegrationTest)
  target_link_libraries(WaferSystemCModelDTEIntegrationTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelDTEIntegrationTest
    COMMAND WaferSystemCModelDTEIntegrationTest)

  add_executable(WaferSystemCModelDTEMismatchTest
    Model/SystemCTargetModelDTEMismatchTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelDTEMismatchTest)
  target_link_libraries(WaferSystemCModelDTEMismatchTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelDTEMismatchTest
    COMMAND WaferSystemCModelDTEMismatchTest)

  add_executable(WaferSystemCModelNoProgressTest
    Model/SystemCTargetModelNoProgressTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelNoProgressTest)
  target_link_libraries(WaferSystemCModelNoProgressTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelNoProgressTest
    COMMAND WaferSystemCModelNoProgressTest)

  add_executable(WaferSystemCModelEventFailureTest
    Model/SystemCTargetModelEventFailureTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelEventFailureTest)
  target_link_libraries(WaferSystemCModelEventFailureTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelEventFailureTest
    COMMAND WaferSystemCModelEventFailureTest)

  set_tests_properties(
    WaferSystemCModelIntegrationTest
    WaferSystemCModelBatchedGemmIntegrationTest
    WaferSystemCModelDTEIntegrationTest
    WaferSystemCModelDTEMismatchTest
    WaferSystemCModelNoProgressTest
    WaferSystemCModelEventFailureTest
    PROPERTIES
    LABELS "systemc-model;managed-dependencies;component"
    ENVIRONMENT
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
  )
  add_custom_target(check-wafer-systemc-model
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelBatchedGemmIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEMismatchTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelNoProgressTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelEventFailureTest>
    DEPENDS
      WaferSystemCModelIntegrationTest
      WaferSystemCModelBatchedGemmIntegrationTest
      WaferSystemCModelDTEIntegrationTest
      WaferSystemCModelDTEMismatchTest
      WaferSystemCModelNoProgressTest
      WaferSystemCModelEventFailureTest
    USES_TERMINAL
  )
endif()
