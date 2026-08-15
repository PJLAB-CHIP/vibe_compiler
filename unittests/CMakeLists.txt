wafer_require_gtest()

add_executable(WaferUnitTests
  Analysis/DirectCallGraphAnalysisTest.cpp
  Analysis/TheoreticalScheduleCostAnalysisTest.cpp
  Analysis/PhysicalDataflow/IndexRelationTest.cpp
  Analysis/ScheduleCostAnalysisTest.cpp
  Analysis/StaticBufferRangeTest.cpp
  Compiler/CompilationOutputTest.cpp
  Compiler/CompilationTest.cpp
  Compiler/CardExecutableCompilationTest.cpp
  Compiler/WholeCardExecutableLoweringTest.cpp
  Compiler/DirectDTETransportTest.cpp
  Compiler/ProgramTensorComparisonTest.cpp
  Compiler/PhysicalTileMemoryPlanningTest.cpp
  Compiler/StructuredBufferRelationsTest.cpp
  Compiler/TargetABIPreparationTest.cpp
  Compiler/TargetCodeGenTest.cpp
  Compiler/TargetCallFrontendTest.cpp
  Compiler/TargetIdentityTest.cpp
  Compiler/TileRegionSPMCapacityEvaluationTest.cpp
  Compiler/WholeDAGScheduleTest.cpp
  Compiler/WholeDAGEdgeStrategyPlanTest.cpp
  Compiler/WholeDAGPlacementEnumerationTest.cpp
  Compiler/WholeCardResourceVerificationTest.cpp
  Compiler/WholeCardExecutableSynthesisTest.cpp
  Conversion/BidirectionalTilingTest.cpp
  Conversion/RequiredNCCJoinPlacementTest.cpp
  Conversion/WaferCardProgramToTileModulesTest.cpp
  Conversion/WaferTensorProgramToCardProgramTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Dialect/Wafer/PhysicalTopologyTest.cpp
  Frontend/NpyPayloadTest.cpp
  Model/TargetModelMemoryTest.cpp
  Pipelines/PipelinesTest.cpp
  Runtime/BoardRuntimeTest.cpp
  Runtime/PackageManifestTest.cpp
  Runtime/ProfileInstrumentationTest.cpp
  Runtime/ProfilerRecordTest.cpp
  Runtime/Tx81ModelABITest.cpp
  Target/FormalNumericTest.cpp
  Target/NumericCodecTest.cpp
  Target/NumericDependencyConformanceTest.cpp
  Target/NumericRawRounderTest.cpp
  Target/NumericSemanticsTest.cpp
  Target/PhysicalTensorCodecTest.cpp
  Target/TargetFormatTest.cpp
  Target/TargetSchedulingCapabilityTest.cpp
  Transforms/MemoryPlanning/LifetimeAnalysisTest.cpp
  Transforms/MemoryPlanning/MiniMallocPackingTest.cpp
  Transforms/MemoryPlanning/StaticMemoryPackingTest.cpp
  Transforms/Scheduling/RedundantTransferEliminationTest.cpp
  Transforms/Target/LowerInstrToTargetLLVMTest.cpp
)
llvm_update_compile_flags(WaferUnitTests)
target_include_directories(WaferUnitTests PRIVATE
  ${CMAKE_SOURCE_DIR}/lib
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
  WaferCardProgramToTileModules
  WaferTensorProgramToCardProgram
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
add_test(NAME WaferUnitTests
  COMMAND WaferUnitTests
)

add_executable(WaferRunBoardIOUnitTests
  Runtime/WaferProfileCampaignTest.cpp
  Runtime/WaferRunBoardIOTest.cpp
  ${CMAKE_SOURCE_DIR}/tools/wafer-run/WaferProfileCampaign.cpp
  ${CMAKE_SOURCE_DIR}/tools/wafer-run/WaferRunBoardIO.cpp
)
llvm_update_compile_flags(WaferRunBoardIOUnitTests)
target_compile_definitions(WaferRunBoardIOUnitTests PRIVATE
  WAFER_PROFILE_CAMPAIGN_TESTING=1
)
target_include_directories(WaferRunBoardIOUnitTests PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/wafer-run
)
target_link_libraries(WaferRunBoardIOUnitTests
  PRIVATE
  WaferRuntime
  GTest::gtest_main
)
add_test(NAME WaferRunBoardIOUnitTests COMMAND WaferRunBoardIOUnitTests)

add_executable(WaferTargetPublicLinkSmoke
  LinkClosure/TargetPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferTargetPublicLinkSmoke)
target_link_libraries(WaferTargetPublicLinkSmoke PRIVATE WaferTarget)

add_executable(WaferRuntimePublicLinkSmoke
  LinkClosure/RuntimePublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferRuntimePublicLinkSmoke)
target_link_libraries(WaferRuntimePublicLinkSmoke PRIVATE WaferRuntime)

add_executable(WaferCompilerPublicLinkSmoke
  LinkClosure/CompilerPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferCompilerPublicLinkSmoke)
target_link_libraries(WaferCompilerPublicLinkSmoke PRIVATE WaferCompiler)

add_executable(WaferModelPublicLinkSmoke
  LinkClosure/ModelPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferModelPublicLinkSmoke)
target_link_libraries(WaferModelPublicLinkSmoke PRIVATE WaferTargetModelCore)

foreach(_wafer_public_link_smoke IN ITEMS
    WaferTargetPublicLinkSmoke
    WaferRuntimePublicLinkSmoke
    WaferCompilerPublicLinkSmoke
    WaferModelPublicLinkSmoke)
  add_test(NAME ${_wafer_public_link_smoke}
    COMMAND ${_wafer_public_link_smoke})
  set_tests_properties(${_wafer_public_link_smoke} PROPERTIES
    LABELS "public-header;link-closure")
endforeach()

add_custom_target(check-wafer-unit
  COMMAND $<TARGET_FILE:WaferUnitTests>
  COMMAND $<TARGET_FILE:WaferRunBoardIOUnitTests>
  COMMAND $<TARGET_FILE:WaferTargetPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferRuntimePublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferCompilerPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferModelPublicLinkSmoke>
  DEPENDS
    WaferUnitTests
    WaferRunBoardIOUnitTests
    WaferTargetPublicLinkSmoke
    WaferRuntimePublicLinkSmoke
    WaferCompilerPublicLinkSmoke
    WaferModelPublicLinkSmoke
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
  set_tests_properties(
    WaferNumericModelUnitTests PROPERTIES
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
  add_executable(WaferSystemCModelIntegrationTest
    Model/SystemCTargetModelIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelIntegrationTest)
  target_include_directories(WaferSystemCModelIntegrationTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
  )
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
  target_include_directories(WaferSystemCModelBatchedGemmIntegrationTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
  )
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

  add_executable(WaferSystemCModelOrientedGemmIntegrationTest
    Model/SystemCTargetModelOrientedGemmIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelOrientedGemmIntegrationTest)
  target_include_directories(WaferSystemCModelOrientedGemmIntegrationTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
  )
  target_link_libraries(WaferSystemCModelOrientedGemmIntegrationTest
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
  add_test(NAME WaferSystemCModelOrientedGemmIntegrationTest
    COMMAND WaferSystemCModelOrientedGemmIntegrationTest)

  set_tests_properties(
    WaferSystemCModelIntegrationTest
    WaferSystemCModelBatchedGemmIntegrationTest
    WaferSystemCModelOrientedGemmIntegrationTest
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
      $<TARGET_FILE:WaferSystemCModelOrientedGemmIntegrationTest>
    DEPENDS
      WaferSystemCModelIntegrationTest
      WaferSystemCModelBatchedGemmIntegrationTest
      WaferSystemCModelOrientedGemmIntegrationTest
    USES_TERMINAL
  )
endif()
