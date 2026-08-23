wafer_require_gtest()

add_executable(WaferUnitTests
  Analysis/CallGraph/DirectCallGraphAnalysisTest.cpp
  Analysis/ScheduleCost/TheoreticalScheduleCostAnalysisTest.cpp
  Analysis/Scheduling/NCCCompletionAnalysisTest.cpp
  Analysis/PhysicalDataflow/IndexRelationTest.cpp
  Analysis/PhysicalDataflow/ExactDemandTest.cpp
  Analysis/PhysicalDataflow/StructuredDemandAnalysisTest.cpp
  Analysis/PhysicalDataflow/AttentionDemandIntegrationTest.cpp
  Analysis/PhysicalDataflow/RootRegionWorkAnalysisTest.cpp
  Analysis/ScheduleCost/ScheduleCostAnalysisTest.cpp
  Analysis/Memory/StaticBufferRangeTest.cpp
  Driver/CompilationOutputTest.cpp
  Driver/CompilationTest.cpp
  TestSupport/CodeGen/CardExecutableTestSupport.cpp
  TestSupport/Planning/SpatialDemandTestSupport.cpp
  TestSupport/Planning/CanonicalPlanningTestSupport.cpp
  TestSupport/Planning/SpatialPlanReference.cpp
  Planning/Baseline/CardBaselineCompilationTest.cpp
  Planning/Baseline/CanonicalBaselinePlanTest.cpp
  CodeGen/Executable/CardExecutableCompilationTest.cpp
  CodeGen/Executable/CardExecutableLoweringTest.cpp
  CodeGen/Executable/SearchRoutingTest.cpp
  Transforms/Bufferization/SelectedBufferMaterializationTest.cpp
  Transforms/Transport/DirectDTETransportTest.cpp
  Program/ProgramTensorComparisonTest.cpp
  CodeGen/Executable/TileMemoryPlanningTest.cpp
  Analysis/Structured/StructuredBufferRelationsTest.cpp
  CodeGen/Target/TargetABIPreparationTest.cpp
  CodeGen/Target/TargetCodeGenTest.cpp
  Target/Execution/TargetCallExecutionTest.cpp
  CodeGen/Target/TargetIdentityTest.cpp
  Planning/PhysicalDataflow/AttentionSpatialConstraintsTest.cpp
  Planning/PhysicalDataflow/CanonicalAttentionWorkProjectionTest.cpp
  Planning/PhysicalDataflow/CanonicalSpatialAssignmentTest.cpp
  Planning/PhysicalDataflow/CanonicalRegionPlanTest.cpp
  Planning/PhysicalDataflow/CanonicalTemporalPlanTest.cpp
  Planning/PhysicalDataflow/CanonicalRepresentationPlanTest.cpp
  Planning/PhysicalDataflow/CanonicalMovementPlanTest.cpp
  Planning/PhysicalDataflow/CanonicalSerializedExecutionPlanTest.cpp
  Planning/PhysicalDataflow/CanonicalStoragePlanTest.cpp
  Planning/PhysicalDataflow/CanonicalSchedulePlanTest.cpp
  Planning/PhysicalDataflow/CanonicalPlanningCoverageTest.cpp
  Planning/PhysicalDataflow/SelectedAttentionDecompositionTest.cpp
  Planning/PhysicalDataflow/Search/PlanningSessionTest.cpp
  Planning/PhysicalDataflow/RootWorkDomainTest.cpp
  Planning/PhysicalDataflow/RegionDomainTest.cpp
  Planning/PhysicalDataflow/MovementDomainTest.cpp
  Planning/PhysicalDataflow/MovementTransferBuilderTest.cpp
  Planning/PhysicalDataflow/PhysicalVersionBuilderTest.cpp
  Planning/PhysicalDataflow/RepresentationDomainTest.cpp
  Planning/PhysicalDataflow/StructuralReadinessTest.cpp
  Planning/PhysicalDataflow/TemporalDomainTest.cpp
  Planning/PhysicalDataflow/SpatialDomainTest.cpp
  Planning/PhysicalDataflow/SpatialPlanTest.cpp
  Analysis/Executable/ProgramResourceVerificationTest.cpp
  Planning/Search/BufferingTest.cpp
  Planning/Search/CoupledRegionTest.cpp
  Planning/Search/ComputeImplementationTest.cpp
  Planning/Search/DataMovementTest.cpp
  Planning/Search/InstructionScheduleTest.cpp
  Planning/Search/PhysicalRepresentationTest.cpp
  Planning/Search/StagePipelineTest.cpp
  Planning/Search/TensorProgramAlternativeTest.cpp
  Planning/Search/UnifiedPhysicalDataflowTest.cpp
  Planning/Search/TemporalTilingTest.cpp
  Program/ProgramDataTest.cpp
  Conversion/AttentionNormalizationTest.cpp
  Conversion/BidirectionalTilingTest.cpp
  Conversion/CoupledTileRegionTest.cpp
  Conversion/RequiredNCCJoinPlacementTest.cpp
  Conversion/SingleRootTileRegionTest.cpp
  Conversion/WaferCardModuleToTileModulesTest.cpp
  Conversion/WaferTensorProgramToCardModuleTest.cpp
  Dialect/Wafer/WaferDialectTest.cpp
  Dialect/Wafer/WaferInterfacesTest.cpp
  Dialect/Wafer/TargetTopologyTest.cpp
  Frontend/Program/NpyPayloadTest.cpp
  Model/Core/TargetModelMemoryTest.cpp
  Pipelines/PipelinesTest.cpp
  Runtime/Board/BoardRuntimeTest.cpp
  Package/Manifest/PackageManifestTest.cpp
  Runtime/Profile/ProfileInstrumentationTest.cpp
  Runtime/Profile/ProfilerRecordTest.cpp
  Target/Numeric/Formal/FormalNumericTest.cpp
  Target/Numeric/NumericCodecTest.cpp
  Target/Numeric/NumericRawRounderTest.cpp
  Target/Layout/PhysicalTensorCodecTest.cpp
  Target/Layout/TargetTensorMaterializationTest.cpp
  Target/Core/TargetFormatTest.cpp
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
  ${CMAKE_SOURCE_DIR}/unittests
)
find_program(WAFER_TEST_LLVM_CLANGXX NAMES clang++ REQUIRED)
target_compile_definitions(WaferUnitTests PRIVATE
  WAFER_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  WAFER_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
  WAFER_TEST_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
  WAFER_TEST_LLVM_CLANGXX="${WAFER_TEST_LLVM_CLANGXX}"
  WAFER_TEST_TX8_DEPS_ROOT="${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
  WAFER_TEST_WAFER_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include"
  WAFER_TEST_WAFER_CRT_SOURCE="${CMAKE_SOURCE_DIR}/runtime/wafer_crt/src/wafer_tx81_crt.c"
  WAFER_TEST_WAFER_CRT_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/runtime/wafer_crt/include"
  WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR="${CMAKE_SOURCE_DIR}/test/Tools/Inputs/minimal-target-data-entry.ll"
)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferAnalysis
  WaferCompiler
  WaferIR
  WaferFormalNumeric
  WaferTargetModelCore
  WaferTargetTensorMaterialization
  WaferRuntime
  WaferStableHLOPipelines
  WaferTileRegionPipelines
  WaferTensorProgramToTileRegion
  WaferCardModuleToTileModules
  WaferTensorProgramToCardModule
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
  Runtime/WaferProfileCollectionTest.cpp
  Runtime/WaferRunBoardIOTest.cpp
  ${CMAKE_SOURCE_DIR}/tools/wafer-run/WaferProfileCollection.cpp
  ${CMAKE_SOURCE_DIR}/tools/wafer-run/WaferRunBoardIO.cpp
)
llvm_update_compile_flags(WaferRunBoardIOUnitTests)
target_compile_definitions(WaferRunBoardIOUnitTests PRIVATE
  WAFER_PROFILE_COLLECTION_TESTING=1
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
    Model/Core/TargetModelKernelTest.cpp
    Target/Numeric/Formal/FormalTensorNumericTest.cpp
    Target/Numeric/Formal/MPFRNumericTest.cpp
    Target/Numeric/NumericOracleDifferentialTest.cpp
    Target/Numeric/Formal/SoftFloatOracleTest.cpp
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
    Model/Reference/ManagedReferenceTargetModelTest.cpp
    Target/Numeric/Qualification/BulkQualificationTest.cpp
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
    Model/SystemC/SystemCTargetModelIntegrationTest.cpp
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

  add_library(WaferSystemCModelTestSupport STATIC
    Model/SystemC/SystemCTargetModelTestSupport.cpp
  )
  llvm_update_compile_flags(WaferSystemCModelTestSupport)
  target_include_directories(WaferSystemCModelTestSupport PUBLIC
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/unittests/Model/SystemC
  )
  target_link_libraries(WaferSystemCModelTestSupport
    PUBLIC
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
  )

  set(WAFER_SYSTEMC_MODEL_TEST_TARGETS WaferSystemCModelIntegrationTest)
  function(add_wafer_systemc_transport_test target source)
    add_executable(${target} ${source})
    llvm_update_compile_flags(${target})
    target_link_libraries(${target} PRIVATE
      WaferSystemCModelTestSupport
      GTest::gtest
    )
    add_test(NAME ${target} COMMAND ${target})
    set(WAFER_SYSTEMC_MODEL_TEST_TARGETS
      ${WAFER_SYSTEMC_MODEL_TEST_TARGETS} ${target}
      PARENT_SCOPE)
  endfunction()

  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEIntegrationTest
    Model/SystemC/SystemCTargetModelDTEIntegrationTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEComputeDisjointTest
    Model/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(WaferSystemCModelDTEComputeDisjointTest PRIVATE
    WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=0)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEComputeOverlapWithoutJoinTest
    Model/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(
    WaferSystemCModelDTEComputeOverlapWithoutJoinTest PRIVATE
    WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=1)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEComputeOverlapWithJoinTest
    Model/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(WaferSystemCModelDTEComputeOverlapWithJoinTest
    PRIVATE WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=2)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTELateSourceWriteTest
    Model/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCModelDTELateSourceWriteTest PRIVATE
    WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=0)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTELateDestinationReadTest
    Model/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCModelDTELateDestinationReadTest PRIVATE
    WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=1)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTELateDestinationWriteTest
    Model/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCModelDTELateDestinationWriteTest
    PRIVATE WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=2)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEMismatchTest
    Model/SystemC/SystemCTargetModelDTEMismatchTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEPreIssueTest
    Model/SystemC/SystemCTargetModelDTEPreIssueTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelDTEReadinessTest
    Model/SystemC/SystemCTargetModelDTEReadinessTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelEventFailureTest
    Model/SystemC/SystemCTargetModelEventFailureTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelNCCVisibilityTest
    Model/SystemC/SystemCTargetModelNCCVisibilityTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCModelNoProgressTest
    Model/SystemC/SystemCTargetModelNoProgressTest.cpp)

  add_executable(WaferSystemCModelBatchedGemmIntegrationTest
    Model/SystemC/SystemCTargetModelBatchedGemmIntegrationTest.cpp
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
    Model/SystemC/SystemCTargetModelOrientedGemmIntegrationTest.cpp
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

  list(APPEND WAFER_SYSTEMC_MODEL_TEST_TARGETS
    WaferSystemCModelBatchedGemmIntegrationTest
    WaferSystemCModelOrientedGemmIntegrationTest)
  set_tests_properties(
    ${WAFER_SYSTEMC_MODEL_TEST_TARGETS}
    PROPERTIES
    LABELS "systemc-model;managed-dependencies;component"
    ENVIRONMENT
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
  )
  add_custom_target(check-wafer-systemc-model
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      -R "^WaferSystemCModel"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${WAFER_SYSTEMC_MODEL_TEST_TARGETS}
    USES_TERMINAL
  )
endif()
