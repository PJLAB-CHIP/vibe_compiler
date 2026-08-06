wafer_require_gtest()

add_executable(WaferUnitTests
  Analysis/ExecutionTopologyAnalysisTest.cpp
  Analysis/NoCProfitabilityAnalysisTest.cpp
  Analysis/PhysicalDataflow/IndexRelationTest.cpp
  Analysis/ScheduleCostAnalysisTest.cpp
  Analysis/StaticBufferRangeTest.cpp
  Compiler/CompilationPublicationTest.cpp
  Compiler/CompilationTest.cpp
  Compiler/CoordinatedDataflowSearchTest.cpp
  Compiler/CoordinatedTerminalEvaluationTest.cpp
  Compiler/CoordinatedVariantSelectionTest.cpp
  Compiler/DirectDTETransportTest.cpp
  Compiler/GlobalTileRelationTest.cpp
  Compiler/NoCIntermediateDataflowTest.cpp
  Compiler/NoCPartialDataflowTest.cpp
  Compiler/NoCResidentDataflowTest.cpp
  Compiler/NoCTiledBoundaryDataflowTest.cpp
  Compiler/ProgramTensorComparisonTest.cpp
  Compiler/ScheduledRankFinalizationTest.cpp
  Compiler/TargetABIPreparationTest.cpp
  Compiler/TargetArtifactTest.cpp
  Compiler/TargetCallFrontendTest.cpp
  Compiler/TargetIdentityTest.cpp
  Compiler/WholeVariantAttemptPlanTest.cpp
  Compiler/WholeVariantCoordinatorTest.cpp
  Compiler/WholeVariantResourceAcceptanceTest.cpp
  Conversion/CollectiveCompletionTest.cpp
  Conversion/CollectiveTopologyTest.cpp
  Conversion/BidirectionalTilingTest.cpp
  Conversion/WaferTensorProgramToTileRegionTest.cpp
  Conversion/CommunicationAlternativesTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Frontend/NpyPayloadTest.cpp
  Model/TargetModelMemoryTest.cpp
  Pipelines/PipelinesTest.cpp
  Runtime/BoardRuntimeTest.cpp
  Runtime/PackageManifestTest.cpp
  Runtime/ProfileCompanionTest.cpp
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
  Transforms/PhysicalDataflow/CandidateRewritesTest.cpp
  Transforms/PhysicalDataflow/ReadyOrderTest.cpp
  Transforms/Scheduling/CandidateSelectionTest.cpp
  Transforms/Scheduling/FixedSlotPipelineTest.cpp
  Transforms/Scheduling/FullBufferHandoffTest.cpp
  Transforms/Scheduling/RankTileMaterializationTest.cpp
  Transforms/Scheduling/RankCandidateFrontierTest.cpp
  Transforms/Scheduling/RedundantTransferEliminationTest.cpp
  Transforms/Scheduling/StructuredSchedulingScopeTest.cpp
  Transforms/Scheduling/WorkerPlacementTest.cpp
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

set(WAFER_DEFAULT_UNIT_TEST_FILTER
  "--gtest_filter=-NoCIntermediateDataflowTest.*:NoCPartialDataflowTest.*:NoCResidentDataflowTest.*:NoCTiledBoundaryDataflowTest.*:WholeVariantCoordinatorTest.*:RankCandidateFrontierTest.*"
)
set(WAFER_COMPILER_INTEGRATION_TEST_FILTER
  "--gtest_filter=NoCIntermediateDataflowTest.*:NoCPartialDataflowTest.*:NoCResidentDataflowTest.*:NoCTiledBoundaryDataflowTest.*:WholeVariantCoordinatorTest.*-WholeVariantCoordinatorTest.PrefersShorterHopRingAllGatherWhenNCCDrainsAreEqual:WholeVariantCoordinatorTest.RejectsI8AllReduceWithoutNumericAdmission:WholeVariantCoordinatorTest.SelectsLargeDDRBoundDistributedContractionWithEstimatedModel"
)

add_test(NAME WaferUnitTests
  COMMAND WaferUnitTests ${WAFER_DEFAULT_UNIT_TEST_FILTER}
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

add_custom_target(check-wafer-unit
  COMMAND $<TARGET_FILE:WaferUnitTests> ${WAFER_DEFAULT_UNIT_TEST_FILTER}
  COMMAND $<TARGET_FILE:WaferRunBoardIOUnitTests>
  DEPENDS WaferUnitTests WaferRunBoardIOUnitTests
  USES_TERMINAL
)

add_custom_target(check-wafer-compiler-integration
  COMMAND $<TARGET_FILE:WaferUnitTests>
    ${WAFER_COMPILER_INTEGRATION_TEST_FILTER}
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
  add_library(WaferSystemCModelTestSupport STATIC
    Model/SystemCTargetModelTestSupport.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelTestSupport)
  target_include_directories(WaferSystemCModelTestSupport PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
  )
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

  add_executable(WaferSystemCModelFixedSlotIntegrationTest
    Model/SystemCTargetModelFixedSlotIntegrationTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelFixedSlotIntegrationTest)
  target_include_directories(WaferSystemCModelFixedSlotIntegrationTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
  )
  target_link_libraries(WaferSystemCModelFixedSlotIntegrationTest
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
  add_test(NAME WaferSystemCModelFixedSlotIntegrationTest
    COMMAND WaferSystemCModelFixedSlotIntegrationTest)

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

  add_executable(WaferSystemCModelDTEPreIssueTest
    Model/SystemCTargetModelDTEPreIssueTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelDTEPreIssueTest)
  target_link_libraries(WaferSystemCModelDTEPreIssueTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelDTEPreIssueTest
    COMMAND WaferSystemCModelDTEPreIssueTest)

  add_executable(WaferSystemCModelDTEReadinessTest
    Model/SystemCTargetModelDTEReadinessTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelDTEReadinessTest)
  target_link_libraries(WaferSystemCModelDTEReadinessTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelDTEReadinessTest
    COMMAND WaferSystemCModelDTEReadinessTest)

  function(add_wafer_systemc_dte_compute_access_test target mode)
    add_executable(${target}
      Model/SystemCTargetModelDTEComputeAccessTest.cpp
    )
    llvm_update_compile_flags(${target})
    target_compile_definitions(${target}
      PRIVATE WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=${mode})
    target_link_libraries(${target}
      PRIVATE
      WaferSystemCModel
      WaferSystemCModelTestSupport
      SystemC::systemc
      GTest::gtest
    )
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  add_wafer_systemc_dte_compute_access_test(
    WaferSystemCModelDTEDisjointComputeAccessTest 0)
  add_wafer_systemc_dte_compute_access_test(
    WaferSystemCModelDTEOverlappingComputeAccessTest 1)
  add_wafer_systemc_dte_compute_access_test(
    WaferSystemCModelDTEJoinedComputeAccessTest 2)

  function(add_wafer_systemc_dte_late_join_test target mode)
    add_executable(${target}
      Model/SystemCTargetModelDTELateJoinTest.cpp
    )
    llvm_update_compile_flags(${target})
    target_compile_definitions(${target}
      PRIVATE WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=${mode})
    target_link_libraries(${target}
      PRIVATE
      WaferSystemCModel
      WaferSystemCModelTestSupport
      SystemC::systemc
      GTest::gtest
    )
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  add_wafer_systemc_dte_late_join_test(
    WaferSystemCModelDTELateJoinSourceWriteTest 0)
  add_wafer_systemc_dte_late_join_test(
    WaferSystemCModelDTELateJoinDestinationReadTest 1)
  add_wafer_systemc_dte_late_join_test(
    WaferSystemCModelDTELateJoinDestinationWriteTest 2)

  add_executable(WaferSystemCModelNCCVisibilityTest
    Model/SystemCTargetModelNCCVisibilityTest.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelNCCVisibilityTest)
  target_link_libraries(WaferSystemCModelNCCVisibilityTest
    PRIVATE
    WaferSystemCModel
    WaferSystemCModelTestSupport
    SystemC::systemc
    GTest::gtest
  )
  add_test(NAME WaferSystemCModelNCCVisibilityTest
    COMMAND WaferSystemCModelNCCVisibilityTest)

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
    WaferSystemCModelFixedSlotIntegrationTest
    WaferSystemCModelBatchedGemmIntegrationTest
    WaferSystemCModelOrientedGemmIntegrationTest
    WaferSystemCModelDTEIntegrationTest
    WaferSystemCModelDTEPreIssueTest
    WaferSystemCModelDTEReadinessTest
    WaferSystemCModelDTEDisjointComputeAccessTest
    WaferSystemCModelDTEOverlappingComputeAccessTest
    WaferSystemCModelDTEJoinedComputeAccessTest
    WaferSystemCModelDTELateJoinSourceWriteTest
    WaferSystemCModelDTELateJoinDestinationReadTest
    WaferSystemCModelDTELateJoinDestinationWriteTest
    WaferSystemCModelNCCVisibilityTest
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
      $<TARGET_FILE:WaferSystemCModelFixedSlotIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelBatchedGemmIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelOrientedGemmIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEIntegrationTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEPreIssueTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEReadinessTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEDisjointComputeAccessTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEOverlappingComputeAccessTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTEJoinedComputeAccessTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTELateJoinSourceWriteTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTELateJoinDestinationReadTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelDTELateJoinDestinationWriteTest>
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferSystemCModelNCCVisibilityTest>
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
      WaferSystemCModelFixedSlotIntegrationTest
      WaferSystemCModelBatchedGemmIntegrationTest
      WaferSystemCModelOrientedGemmIntegrationTest
      WaferSystemCModelDTEIntegrationTest
      WaferSystemCModelDTEPreIssueTest
      WaferSystemCModelDTEReadinessTest
      WaferSystemCModelDTEDisjointComputeAccessTest
      WaferSystemCModelDTEOverlappingComputeAccessTest
      WaferSystemCModelDTEJoinedComputeAccessTest
      WaferSystemCModelDTELateJoinSourceWriteTest
      WaferSystemCModelDTELateJoinDestinationReadTest
      WaferSystemCModelDTELateJoinDestinationWriteTest
      WaferSystemCModelNCCVisibilityTest
      WaferSystemCModelDTEMismatchTest
      WaferSystemCModelNoProgressTest
      WaferSystemCModelEventFailureTest
    USES_TERMINAL
  )
endif()
