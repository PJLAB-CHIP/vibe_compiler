wafer_require_gtest()

find_program(WAFER_TEST_LLVM_CLANGXX NAMES clang++ REQUIRED)

set(WAFER_COMPONENT_UNIT_TARGETS)
function(wafer_add_component_unit_test target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBS" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  llvm_update_compile_flags(${target})
  target_include_directories(${target} PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/lib/Wafer/Transforms
    ${CMAKE_SOURCE_DIR}/unittests)
  target_compile_definitions(${target} PRIVATE
    WAFER_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    WAFER_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    WAFER_TEST_DEVICE_LINKER_SCRIPT="${CMAKE_SOURCE_DIR}/tools/wafer_device_link.py"
    WAFER_TEST_LLVM_CLANGXX="${WAFER_TEST_LLVM_CLANGXX}"
    WAFER_TEST_TX8_DEPS_ROOT="${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    WAFER_TEST_WAFER_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include"
    WAFER_TEST_WAFER_CRT_SOURCE="${CMAKE_SOURCE_DIR}/runtime/crt/src/wafer_tx81_crt.c"
    WAFER_TEST_WAFER_CRT_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/runtime/crt/include"
    WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR="${CMAKE_SOURCE_DIR}/test/Tools/Inputs/minimal-target-data-entry.ll")
  target_link_libraries(${target} PRIVATE
    ${ARG_LINK_LIBS}
    MLIRAsyncDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    GTest::gtest_main)
if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(${target} PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(${target} PRIVATE StablehloRegister)
  endif()
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
  LABELS "unit;component")
  set(WAFER_COMPONENT_UNIT_TARGETS
    ${WAFER_COMPONENT_UNIT_TARGETS} ${target} PARENT_SCOPE)
endfunction()

add_library(WaferExecutableTestSupport STATIC
  TestSupport/CodeGen/ExecutableTestSupport.cpp)
llvm_update_compile_flags(WaferExecutableTestSupport)
target_include_directories(WaferExecutableTestSupport PUBLIC
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/lib/Wafer/Transforms
  ${CMAKE_SOURCE_DIR}/unittests)
target_link_libraries(WaferExecutableTestSupport PUBLIC
  WaferCodeGen
  WaferCompiler
  WaferProgramData
  MLIRIR
  MLIRParser
  GTest::gtest)

wafer_add_component_unit_test(WaferIRUnitTests
  SOURCES
    Dialect/Wafer/WaferDialectTest.cpp
    Dialect/Wafer/WaferInterfacesTest.cpp
    Dialect/Wafer/TargetTopologyTest.cpp
  LINK_LIBS WaferIR WaferPhysicalTensor WaferStableHLOPipelines)

wafer_add_component_unit_test(WaferAnalysisUnitTests
  SOURCES
    Analysis/Module/DirectCallGraphAnalysisTest.cpp
    Analysis/Instr/NCCCompletionAnalysisTest.cpp
    Analysis/Linalg/IndexRelationTest.cpp
    Analysis/Linalg/TensorResultIndexingTest.cpp
    Analysis/Instr/ScheduleCostAnalysisTest.cpp
    Analysis/Instr/CostModelTest.cpp
  LINK_LIBS WaferAnalysis WaferIR)

wafer_add_component_unit_test(WaferPlanningUnitTests
  SOURCES
    Planning/PhysicalDataflow/ExactDemandTest.cpp
    Planning/PhysicalDataflow/StructuredDemandAnalysisTest.cpp
    Planning/PhysicalDataflow/AttentionDemandIntegrationTest.cpp
    Planning/PhysicalDataflow/RootRegionWorkAnalysisTest.cpp
    Planning/PhysicalDataflow/AttentionSpatialConstraintsTest.cpp
    Planning/PhysicalDataflow/CanonicalSpatialAssignmentTest.cpp
    Planning/PhysicalDataflow/CanonicalRegionPlanTest.cpp
    Planning/PhysicalDataflow/PlanningSessionTest.cpp
    Planning/PhysicalDataflow/RootWorkDomainTest.cpp
    Planning/PhysicalDataflow/RegionDomainTest.cpp
    Planning/PhysicalDataflow/ExactPBQPSolverTest.cpp
    Planning/PhysicalDataflow/TemporalDomainTest.cpp
    Planning/PhysicalDataflow/SpatialDomainTest.cpp
    Planning/PhysicalDataflow/SpatialPlanTest.cpp
    TestSupport/Planning/SpatialDemandTestSupport.cpp
    TestSupport/Planning/SpatialPlanReference.cpp
  LINK_LIBS WaferAnalysis WaferExecutableTestSupport WaferIR WaferPlanning)

wafer_add_component_unit_test(WaferDriverUnitTests
  SOURCES
    Driver/CompilationOutputTest.cpp
    Driver/CompilationTest.cpp
    Driver/ExecutableCompilationTest.cpp
    Driver/ExecutableLoweringTest.cpp
    Driver/PhysicalDataflow/SearchRoutingTest.cpp
    Driver/PhysicalDataflow/ActualResultControllerTest.cpp
    Driver/PhysicalDataflow/UnifiedSearchTest.cpp
    Driver/ProgramResourceVerificationTest.cpp
    Driver/ProgramDataTest.cpp
    Driver/StandaloneTileModulesTest.cpp
  LINK_LIBS
    WaferCompiler
    WaferCompilerTestSupport
    WaferExecutableTestSupport
    WaferSimulatorInvocation
    WaferStandaloneTileModules)

wafer_add_component_unit_test(WaferTransformsUnitTests
  SOURCES
    Transforms/Instr/DirectDTETransportTest.cpp
    Transforms/Instr/TileMemoryPlanningTest.cpp
    Transforms/Tile/StructuredBufferRelationsTest.cpp
    Transforms/Tile/LayoutOptimizationTest.cpp
    Transforms/Tile/StructuredToTileTest.cpp
    Transforms/Tile/ExecutionStructureMaterializationTest.cpp
    Transforms/Linalg/AttentionNormalizationTest.cpp
    Transforms/Linalg/OnlineAttentionDecompositionTest.cpp
    Transforms/Linalg/StructuredGraphEGraphTest.cpp
    Transforms/Linalg/StructuredGraphNormalizationTest.cpp
    Transforms/Linalg/SpatialRegionMaterializationTest.cpp
    Transforms/Linalg/StructuredTilingTest.cpp
    Transforms/Linalg/TemporalTilingTest.cpp
    Transforms/Instr/RequiredNCCJoinPlacementTest.cpp
    Transforms/Instr/LifetimeAnalysisTest.cpp
    Transforms/Instr/MiniMallocPackingTest.cpp
    Transforms/Instr/StaticMemoryPackingTest.cpp
    Transforms/Instr/RedundantTransferEliminationTest.cpp
  LINK_LIBS
    WaferCompiler
    WaferCompilerTestSupport
    WaferExecutableTestSupport
    WaferLinalgTransforms
    WaferTileToInstrPipelines
    WaferTransforms)

wafer_add_component_unit_test(WaferCodeGenUnitTests
  SOURCES
    CodeGen/LLVM/TargetABIPreparationTest.cpp
    CodeGen/LLVM/TargetCodeGenTest.cpp
    CodeGen/LLVM/TargetIdentityTest.cpp
  LINK_LIBS
    WaferCodeGen
    WaferExecutableTestSupport
    WaferPackageWriter
    WaferRuntime)

wafer_add_component_unit_test(WaferConversionUnitTests
  SOURCES Conversion/InstrToLLVM/LowerInstrToLLVMTest.cpp
  LINK_LIBS WaferIR WaferInstrToLLVM WaferTileToInstr)

wafer_add_component_unit_test(WaferFrontendUnitTests
  SOURCES Frontend/NpyPayloadTest.cpp
  LINK_LIBS WaferFrontend WaferIR)

if(WAFER_ENABLE_IMPORTER_DEPS)
  wafer_add_component_unit_test(WaferStableHLOFrontendUnitTests
    SOURCES Frontend/ProgramIngestionTest.cpp
    LINK_LIBS WaferFrontend WaferStableHLOProgram WaferIR)
endif()

wafer_add_component_unit_test(WaferSimulatorUnitTests
  SOURCES
    Simulator/Invocation/ProgramTensorComparisonTest.cpp
    Simulator/Invocation/TargetCallExecutionTest.cpp
    Simulator/Memory/TargetModelMemoryTest.cpp
    Simulator/Reference/FormalNumericTest.cpp
  LINK_LIBS
    WaferFormalNumeric
    WaferSimulatorInvocation
    WaferSimulatorMemory)

wafer_add_component_unit_test(WaferPipelineUnitTests
  SOURCES Pipelines/PipelinesTest.cpp
  LINK_LIBS
    WaferCompiler
    WaferStableHLOPipelines
    WaferTileToInstrPipelines
    WaferTransforms)

wafer_add_component_unit_test(WaferRuntimeUnitTests
  SOURCES
    Runtime/Board/BoardRuntimeTest.cpp
    Runtime/Profile/ProfileInstrumentationTest.cpp
    Runtime/Profile/ProfilerRecordTest.cpp
  LINK_LIBS WaferRuntime)

wafer_add_component_unit_test(WaferPackageUnitTests
  SOURCES Package/Manifest/PackageManifestTest.cpp
  LINK_LIBS WaferPackage WaferRuntime)

wafer_add_component_unit_test(WaferTargetUnitTests
  SOURCES
    Target/PhysicalTensor/NumericCodecTest.cpp
    Target/PhysicalTensor/NumericRawRounderTest.cpp
    Target/PhysicalTensor/PhysicalTensorCodecTest.cpp
    Target/PhysicalTensor/TargetTensorMaterializationTest.cpp
    Target/TargetFormatTest.cpp
  LINK_LIBS
    WaferFormalNumeric
    WaferIR
    WaferInstrToLLVM
    WaferPhysicalTensor
    WaferTarget)

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

add_executable(WaferPhysicalTensorPublicLinkSmoke
  LinkClosure/PhysicalTensorPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferPhysicalTensorPublicLinkSmoke)
target_link_libraries(WaferPhysicalTensorPublicLinkSmoke PRIVATE
  WaferPhysicalTensor)

add_executable(WaferRuntimePublicLinkSmoke
  LinkClosure/RuntimePublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferRuntimePublicLinkSmoke)
target_link_libraries(WaferRuntimePublicLinkSmoke PRIVATE WaferRuntime)

add_executable(WaferCompilerPublicLinkSmoke
  LinkClosure/CompilerPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferCompilerPublicLinkSmoke)
target_link_libraries(WaferCompilerPublicLinkSmoke PRIVATE WaferCompiler)

add_executable(WaferCodeGenPublicLinkSmoke
  LinkClosure/CodeGenPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferCodeGenPublicLinkSmoke)
target_link_libraries(WaferCodeGenPublicLinkSmoke PRIVATE WaferCodeGen)

add_executable(WaferPlanningPublicLinkSmoke
  LinkClosure/PlanningPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferPlanningPublicLinkSmoke)
target_link_libraries(WaferPlanningPublicLinkSmoke PRIVATE WaferPlanning)

add_executable(WaferProgramDataPublicLinkSmoke
  LinkClosure/ProgramDataPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferProgramDataPublicLinkSmoke)
target_link_libraries(WaferProgramDataPublicLinkSmoke PRIVATE WaferProgramData)

add_executable(WaferSimulatorPublicLinkSmoke
  LinkClosure/SimulatorPublicLinkSmoke.cpp)
llvm_update_compile_flags(WaferSimulatorPublicLinkSmoke)
target_link_libraries(WaferSimulatorPublicLinkSmoke PRIVATE WaferSimulatorMemory)

foreach(_wafer_public_link_smoke IN ITEMS
    WaferTargetPublicLinkSmoke
    WaferPhysicalTensorPublicLinkSmoke
    WaferRuntimePublicLinkSmoke
    WaferCompilerPublicLinkSmoke
    WaferCodeGenPublicLinkSmoke
    WaferPlanningPublicLinkSmoke
    WaferProgramDataPublicLinkSmoke
    WaferSimulatorPublicLinkSmoke)
  add_test(NAME ${_wafer_public_link_smoke}
    COMMAND ${_wafer_public_link_smoke})
  set_tests_properties(${_wafer_public_link_smoke} PROPERTIES
    LABELS "public-header;link-closure")
endforeach()

include(ProcessorCount)
ProcessorCount(WAFER_UNIT_TEST_JOBS)
if(NOT WAFER_UNIT_TEST_JOBS)
  set(WAFER_UNIT_TEST_JOBS 1)
endif()

add_custom_target(check-wafer-unit
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    --label-regex "^unit$" --parallel ${WAFER_UNIT_TEST_JOBS}
  COMMAND $<TARGET_FILE:WaferRunBoardIOUnitTests>
  COMMAND $<TARGET_FILE:WaferTargetPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferPhysicalTensorPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferRuntimePublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferCompilerPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferCodeGenPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferPlanningPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferProgramDataPublicLinkSmoke>
  COMMAND $<TARGET_FILE:WaferSimulatorPublicLinkSmoke>
  DEPENDS
    ${WAFER_COMPONENT_UNIT_TARGETS}
    WaferRunBoardIOUnitTests
    WaferTargetPublicLinkSmoke
    WaferPhysicalTensorPublicLinkSmoke
    WaferRuntimePublicLinkSmoke
    WaferCompilerPublicLinkSmoke
    WaferCodeGenPublicLinkSmoke
    WaferPlanningPublicLinkSmoke
    WaferProgramDataPublicLinkSmoke
    WaferSimulatorPublicLinkSmoke
  USES_TERMINAL
)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_executable(WaferReferenceNumericUnitTests
    Simulator/Kernel/TargetModelKernelTest.cpp
    Simulator/Reference/FormalTensorNumericTest.cpp
    Simulator/Reference/MPFRNumericTest.cpp
    Simulator/Reference/NumericOracleDifferentialTest.cpp
    Simulator/Reference/SoftFloatOracleTest.cpp
  )
  llvm_update_compile_flags(WaferReferenceNumericUnitTests)
  target_include_directories(WaferReferenceNumericUnitTests PRIVATE
    ${CMAKE_SOURCE_DIR}/lib)
  target_compile_definitions(WaferReferenceNumericUnitTests PRIVATE
    WAFER_NUMERIC_MODEL_DEPS_ROOT="${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}"
    WAFER_NUMERIC_MODEL_DEPS_RECORD="${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD}"
    WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256="${WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256}"
  )
  target_link_libraries(WaferReferenceNumericUnitTests
    PRIVATE
    WaferReferenceNumeric
    WaferSimulatorKernel
    GTest::gtest_main
  )
  add_test(NAME WaferReferenceNumericUnitTests COMMAND WaferReferenceNumericUnitTests)
  set_tests_properties(
    WaferReferenceNumericUnitTests PROPERTIES
    ENVIRONMENT
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
    LABELS "numeric-model;managed-dependencies"
  )
  add_custom_target(check-wafer-numeric-model-unit
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
      $<TARGET_FILE:WaferReferenceNumericUnitTests>
    DEPENDS WaferReferenceNumericUnitTests
    USES_TERMINAL
  )
endif()

if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  add_executable(WaferTargetNumericBackendUnitTests
    Simulator/Reference/ManagedReferenceTargetModelTest.cpp
    Simulator/OneDNN/OneDNNQualificationTest.cpp
  )
  llvm_update_compile_flags(WaferTargetNumericBackendUnitTests)
  target_link_libraries(WaferTargetNumericBackendUnitTests
    PRIVATE
    WaferOneDNNBackend
    WaferTargetNumericBackend
    GTest::gtest_main
  )
  add_test(NAME WaferTargetNumericBackendUnitTests
    COMMAND WaferTargetNumericBackendUnitTests)
  set_tests_properties(WaferTargetNumericBackendUnitTests PROPERTIES
    LABELS "target-numeric-backend;managed-dependencies"
  )
  add_custom_target(check-wafer-target-numeric-backend-unit
    COMMAND $<TARGET_FILE:WaferTargetNumericBackendUnitTests>
    DEPENDS WaferTargetNumericBackendUnitTests
    USES_TERMINAL
  )
endif()

if(WAFER_ENABLE_SYSTEMC_MODEL)
  add_library(WaferSystemCSimulatorTestSupport STATIC
    Simulator/SystemC/SystemCTargetModelTestSupport.cpp
  )
  llvm_update_compile_flags(WaferSystemCSimulatorTestSupport)
  target_include_directories(WaferSystemCSimulatorTestSupport PUBLIC
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/unittests/Simulator/SystemC
  )
  target_link_libraries(WaferSystemCSimulatorTestSupport
    PUBLIC
    WaferSystemCSimulator
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

  set(WAFER_SYSTEMC_MODEL_TEST_TARGETS)
  function(add_wafer_systemc_transport_test target source)
    add_executable(${target} ${source})
    llvm_update_compile_flags(${target})
    target_link_libraries(${target} PRIVATE
      WaferSystemCSimulatorTestSupport
      GTest::gtest
    )
    add_test(NAME ${target} COMMAND ${target})
    set(WAFER_SYSTEMC_MODEL_TEST_TARGETS
      ${WAFER_SYSTEMC_MODEL_TEST_TARGETS} ${target}
      PARENT_SCOPE)
  endfunction()

  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEIntegrationTest
    Simulator/SystemC/SystemCTargetModelDTEIntegrationTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEBroadcastTest
    Simulator/SystemC/SystemCTargetModelDTEMultiSendTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTEBroadcastTest PRIVATE
    WAFER_TEST_DTE_MULTISEND_KIND=0)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEScatterTest
    Simulator/SystemC/SystemCTargetModelDTEMultiSendTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTEScatterTest PRIVATE
    WAFER_TEST_DTE_MULTISEND_KIND=1)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEComputeDisjointTest
    Simulator/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTEComputeDisjointTest PRIVATE
    WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=0)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEComputeOverlapWithoutJoinTest
    Simulator/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(
    WaferSystemCSimulatorDTEComputeOverlapWithoutJoinTest PRIVATE
    WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=1)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEComputeOverlapWithJoinTest
    Simulator/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTEComputeOverlapWithJoinTest
    PRIVATE WAFER_TEST_DTE_COMPUTE_ACCESS_MODE=2)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTELateSourceWriteTest
    Simulator/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTELateSourceWriteTest PRIVATE
    WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=0)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTELateDestinationReadTest
    Simulator/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTELateDestinationReadTest PRIVATE
    WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=1)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTELateDestinationWriteTest
    Simulator/SystemC/SystemCTargetModelDTELateJoinTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDTELateDestinationWriteTest
    PRIVATE WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE=2)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEMismatchTest
    Simulator/SystemC/SystemCTargetModelDTEMismatchTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEPreIssueTest
    Simulator/SystemC/SystemCTargetModelDTEPreIssueTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDTEReadinessTest
    Simulator/SystemC/SystemCTargetModelDTEReadinessTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorEventFailureTest
    Simulator/SystemC/SystemCTargetModelEventFailureTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorNCCVisibilityTest
    Simulator/SystemC/SystemCTargetModelNCCVisibilityTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorNoProgressTest
    Simulator/SystemC/SystemCTargetModelNoProgressTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDDRPublicationTest
    Simulator/SystemC/SystemCTargetModelDDRPublicationTest.cpp)
  add_wafer_systemc_transport_test(
    WaferSystemCSimulatorDDRLateJoinTest
    Simulator/SystemC/SystemCTargetModelDDRPublicationTest.cpp)
  target_compile_definitions(WaferSystemCSimulatorDDRLateJoinTest PRIVATE
    WAFER_TEST_DDR_LATE_JOIN=1)

  set_tests_properties(
    ${WAFER_SYSTEMC_MODEL_TEST_TARGETS}
    PROPERTIES
    LABELS "systemc-model;managed-dependencies;component"
    ENVIRONMENT
      "LD_LIBRARY_PATH=${WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR}:${WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR}"
  )
  add_custom_target(check-wafer-systemc-model
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      -R "^WaferSystemCSimulator"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${WAFER_SYSTEMC_MODEL_TEST_TARGETS}
    USES_TERMINAL
  )
endif()
