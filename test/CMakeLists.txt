execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import lit"
  RESULT_VARIABLE WAFER_LIT_IMPORT_RESULT
)
if(NOT WAFER_LIT_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Python package 'lit' is required for FileCheck tests. "
    "Run tools/bootstrap_deps.py --python and configure with that Python, or install requirements-dev.txt.")
endif()

get_filename_component(WAFER_PYTHON_BIN_DIR "${Python3_EXECUTABLE}" DIRECTORY)
find_program(WAFER_LIT_EXECUTABLE
  NAMES lit llvm-lit
  HINTS "${WAFER_PYTHON_BIN_DIR}" "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

set(WAFER_FILECHECK_EXECUTABLE "")
set(_wafer_filecheck_candidates
  "${LLVM_TOOLS_BINARY_DIR}/FileCheck"
  "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/bin/FileCheck"
)
foreach(_candidate IN LISTS _wafer_filecheck_candidates)
  if(EXISTS "${_candidate}" AND NOT IS_DIRECTORY "${_candidate}")
    execute_process(
      COMMAND "${_candidate}" --version
      RESULT_VARIABLE _candidate_result
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(_candidate_result EQUAL 0)
      set(WAFER_FILECHECK_EXECUTABLE "${_candidate}")
      break()
    endif()
  endif()
endforeach()
if(NOT WAFER_FILECHECK_EXECUTABLE)
  find_program(WAFER_FILECHECK_EXECUTABLE NAMES FileCheck REQUIRED)
endif()
execute_process(
  COMMAND "${WAFER_FILECHECK_EXECUTABLE}" --version
  RESULT_VARIABLE WAFER_FILECHECK_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT WAFER_FILECHECK_RESULT EQUAL 0)
  message(FATAL_ERROR "FileCheck exists but is not executable: ${WAFER_FILECHECK_EXECUTABLE}")
endif()
get_filename_component(WAFER_FILECHECK_DIR "${WAFER_FILECHECK_EXECUTABLE}" DIRECTORY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

set(WAFER_LIT_ARGS "-sv" CACHE STRING "Arguments passed to lit")
separate_arguments(WAFER_LIT_ARGS)

set(_wafer_lit_dependencies
  wafer-compile
  wafer-compile-test
  wafer-compile-stablehlo
  wafer-opt
  wafer-run
)
if(TARGET wafer-cmodel-qualify-bulk)
  list(APPEND _wafer_lit_dependencies wafer-cmodel-qualify-bulk)
endif()

add_custom_target(check-wafer-lit
  COMMAND ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${_wafer_lit_dependencies}
  USES_TERMINAL
)
unset(_wafer_lit_dependencies)

add_test(NAME wafer-lit
  COMMAND ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME wafer-runtime-adapter-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/wafer_runtime_adapter_test.py
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-run $<TARGET_FILE:wafer-run>
)

add_test(NAME wafer-board-direct-dte-outer-deadline
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_timeout_test.py
)
set_tests_properties(wafer-board-direct-dte-outer-deadline PROPERTIES
  LABELS "runtime;no-card;board;direct-dte;safety"
  TIMEOUT 30
)

add_executable(wafer-ncc-probe-plan-test
  Board/Inputs/wafer_ncc_probe_plan.c
  Board/wafer_ncc_probe_plan_test.c
)
target_include_directories(wafer-ncc-probe-plan-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Board/Inputs
)
set_target_properties(wafer-ncc-probe-plan-test PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO
)
target_compile_options(wafer-ncc-probe-plan-test PRIVATE
  -Wall
  -Wextra
  -Werror
  -pedantic
)
add_test(NAME wafer-ncc-probe-plan
  COMMAND $<TARGET_FILE:wafer-ncc-probe-plan-test>
)
set_tests_properties(wafer-ncc-probe-plan PROPERTIES
  LABELS "runtime;no-card;ncc;probe-protocol"
  TIMEOUT 30
)

add_executable(wafer-ncc-hazard-relation-test
  Board/Inputs/wafer_ncc_hazard_relation.c
  Board/wafer_ncc_hazard_relation_test.c
)
target_include_directories(wafer-ncc-hazard-relation-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Board/Inputs
)
set_target_properties(wafer-ncc-hazard-relation-test PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO
)
target_compile_options(wafer-ncc-hazard-relation-test PRIVATE
  -Wall
  -Wextra
  -Werror
  -pedantic
)
add_test(NAME wafer-ncc-hazard-relation
  COMMAND $<TARGET_FILE:wafer-ncc-hazard-relation-test>
)
set_tests_properties(wafer-ncc-hazard-relation PROPERTIES
  LABELS "runtime;no-card;ncc;probe-protocol"
  TIMEOUT 30
)

add_test(NAME wafer-ncc-probe-protocol-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ncc_probe_protocol_test.py
)
set_tests_properties(wafer-ncc-probe-protocol-python PROPERTIES
  LABELS "runtime;no-card;ncc;probe-protocol"
  TIMEOUT 30
)

add_test(NAME wafer-instruction-family-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_instruction_family_catalog_test.py
)
set_tests_properties(wafer-instruction-family-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;qualification;oracle"
  TIMEOUT 30
)

add_test(NAME wafer-ct-vector-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ct_vector_calibration_catalog_test.py
)
set_tests_properties(wafer-ct-vector-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;calibration;oracle"
  TIMEOUT 60
)

add_test(NAME wafer-ct-convert-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ct_convert_calibration_catalog_test.py
)
set_tests_properties(wafer-ct-convert-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;convert;calibration;oracle"
  TIMEOUT 60
)

add_test(NAME wafer-datamove-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_datamove_calibration_catalog_test.py
)
set_tests_properties(wafer-datamove-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;datamove;layout;calibration;oracle"
  TIMEOUT 60
)

add_test(NAME wafer-ct-opcode-disposition-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ct_opcode_disposition_catalog_test.py
)
set_tests_properties(wafer-ct-opcode-disposition-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;calibration;coverage;negative"
  TIMEOUT 60
)

add_test(NAME wafer-hardware-calibration-matrix-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_hardware_calibration_matrix_test.py
)
set_tests_properties(wafer-hardware-calibration-matrix-python PROPERTIES
  LABELS "runtime;no-card;hardware;calibration;coverage"
  TIMEOUT 30
)

add_test(NAME wafer-physical-tensor-codec-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_physical_tensor_codec_test.py
)
set_tests_properties(wafer-physical-tensor-codec-python PROPERTIES
  LABELS "runtime;no-card;layout;calibration;oracle"
  TIMEOUT 30
)

add_test(NAME wafer-ne-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ne_calibration_catalog_test.py
)
set_tests_properties(wafer-ne-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;ne;layout;calibration;oracle"
  TIMEOUT 60
)

add_test(NAME wafer-spm-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_spm_calibration_catalog_test.py
)
set_tests_properties(wafer-spm-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;spm;calibration;oracle"
  TIMEOUT 30
)

add_test(NAME wafer-cache-coherence-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_cache_coherence_calibration_catalog_test.py
)
set_tests_properties(wafer-cache-coherence-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;cache;coherence;calibration;oracle"
  TIMEOUT 30
)

add_test(NAME wafer-transport-pmu-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_transport_pmu_calibration_catalog_test.py
)
set_tests_properties(wafer-transport-pmu-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;direct-dte;spm;tmnoc;pmu;calibration;negative"
  TIMEOUT 30
)

add_test(NAME wafer-full-card-barrier-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_full_card_barrier_contract_test.py
)
set_tests_properties(wafer-full-card-barrier-contract-python PROPERTIES
  LABELS "runtime;no-card;cluster;barrier;calibration;negative"
  TIMEOUT 30
)

if(WAFER_ENABLE_IMPORTER_DEPS AND
   WAFER_ENABLE_SPMD_PARTITIONER_DEPS AND
   WAFER_XLA_SPMD_PARTITIONER_HELPER)
  foreach(_wafer_no_card_launch_test IN ITEMS kernel-grid model)
    if(_wafer_no_card_launch_test STREQUAL "kernel-grid")
      set(_wafer_no_card_launch_abi
        "tx81-kernel-grid-pointer-table-v1")
    else()
      set(_wafer_no_card_launch_abi "tx81-model-bootparam-v1")
    endif()
    add_test(NAME wafer-runtime-${_wafer_no_card_launch_test}-add-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/runtime-${_wafer_no_card_launch_test}-add
        --launch-abi ${_wafer_no_card_launch_abi}
        --no-card
    )
    set_tests_properties(
      wafer-runtime-${_wafer_no_card_launch_test}-add-no-card PROPERTIES
      LABELS "runtime;no-card;all-rank;single-op;${_wafer_no_card_launch_test}"
      TIMEOUT 300
    )
  endforeach()
  unset(_wafer_no_card_launch_abi)
  unset(_wafer_no_card_launch_test)
  add_test(NAME wafer-runtime-cluster-direct-dte-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_collective_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/runtime-cluster-direct-dte
      --no-card
  )
  set_tests_properties(wafer-runtime-cluster-direct-dte-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;direct-dte"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-dte-ncc-execution-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_dte_ncc_execution_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-dte-ncc-execution-probe
      --total-timeout-seconds 300
      --no-card
  )
  set_tests_properties(
    wafer-runtime-dte-ncc-execution-probe-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;direct-dte;ncc;probe-build"
    TIMEOUT 330
  )
  add_test(NAME wafer-runtime-full-card-barrier-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_full_card_barrier_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-full-card-barrier-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-full-card-barrier-probe-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;barrier;probe-build"
    TIMEOUT 330
  )
  add_test(NAME wafer-runtime-k-sharded-gemm-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_k_sharded_gemm_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-k-sharded-gemm-no-card
      --no-card
  )
  set_tests_properties(wafer-runtime-k-sharded-gemm-no-card PROPERTIES
    LABELS "runtime;no-card;heavy;all-rank;gemm;direct-dte"
    TIMEOUT 1800
  )
  add_test(NAME wafer-runtime-single-gemm-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_single_gemm_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/runtime-single-gemm-no-card
      --no-card
  )
  set_tests_properties(wafer-runtime-single-gemm-no-card PROPERTIES
    LABELS "runtime;no-card;single-op;gemm"
    TIMEOUT 600
  )
  add_test(NAME wafer-runtime-ncc-pmu-readonly-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_pmu_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ncc-pmu-readonly-probe-no-card
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ncc-pmu-readonly-probe-no-card PROPERTIES
    LABELS "runtime;no-card;board;pmu;static-audit"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ncc-execution-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_execution_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ncc-execution-probe-no-card
      --suite build-smoke
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ncc-execution-probe-no-card PROPERTIES
    LABELS "runtime;no-card;ncc;execution-model;build-smoke"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-instruction-family-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_instruction_family_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-instruction-family-probe
      --suite safe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-instruction-family-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;qualification;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ct-vector-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_vector_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ct-vector-calibration-probe
      --suite exact
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ct-vector-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ct-convert-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_convert_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ct-convert-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ct-convert-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;convert;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-datamove-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_datamove_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-datamove-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-datamove-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;datamove;layout;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ne-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ne_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ne-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ne-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;ne;layout;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-spm-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-spm-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-spm-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;spm;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-cache-coherence-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_cache_coherence_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-cache-coherence-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-cache-coherence-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;cache;coherence;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-mn-tiled-gemm-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_single_gemm_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/runtime-mn-tiled-gemm-no-card
      --gemm-case mn-tiled
      --no-card
  )
  set_tests_properties(wafer-runtime-mn-tiled-gemm-no-card PROPERTIES
    LABELS "runtime;no-card;single-op;gemm;tiling"
    TIMEOUT 600
  )
endif()

if(WAFER_ENABLE_BOARD_TEST_EXECUTION)
  if(NOT WAFER_ENABLE_IMPORTER_DEPS OR
     NOT WAFER_ENABLE_SPMD_PARTITIONER_DEPS OR
     NOT WAFER_XLA_SPMD_PARTITIONER_HELPER)
    message(FATAL_ERROR
      "WAFER_ENABLE_BOARD_TEST_EXECUTION=ON requires importer and SPMD "
      "dependencies plus WAFER_XLA_SPMD_PARTITIONER_HELPER")
  endif()
  set(WAFER_BOARD_TEST_DEVICE_ID "0" CACHE STRING
    "TX device used by configured board execution tests")
  set(WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION "" CACHE STRING
    "Expected decimal TX runtime API version for board tests")
  set(WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME "" CACHE STRING
    "Expected exact TX device name for board tests")
  set(WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID "" CACHE STRING
    "Expected exact TX PCI bus ID for board tests")
  set(WAFER_BOARD_TEST_EXPECTED_TILE_COUNT "" CACHE STRING
    "Expected TX tile count for board tests")
  set(WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256 "" CACHE STRING
    "Expected SHA-256 of the public TX runtime library for board tests")
  foreach(_wafer_board_qualification_variable IN ITEMS
      WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION
      WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME
      WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID
      WAFER_BOARD_TEST_EXPECTED_TILE_COUNT
      WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256)
    if("${${_wafer_board_qualification_variable}}" STREQUAL "")
      message(FATAL_ERROR
        "WAFER_ENABLE_BOARD_TEST_EXECUTION=ON requires explicit "
        "${_wafer_board_qualification_variable}")
    endif()
  endforeach()
  string(LENGTH "${WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256}"
    _wafer_expected_tx_runtime_library_sha256_length)
  if(NOT WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION MATCHES "^[1-9][0-9]*$" OR
     NOT WAFER_BOARD_TEST_EXPECTED_TILE_COUNT MATCHES "^[1-9][0-9]*$" OR
     NOT WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256 MATCHES
       "^[0-9a-fA-F]+$" OR
     NOT _wafer_expected_tx_runtime_library_sha256_length EQUAL 64)
    message(FATAL_ERROR "Invalid TX board-test qualification value")
  endif()
  file(SHA256 "${WAFER_TX_RUNTIME_LIBRARY_REAL_PATH}"
    _wafer_tx_runtime_library_sha256)
  string(TOLOWER "${WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256}"
    _wafer_expected_tx_runtime_library_sha256)
  if(NOT _wafer_tx_runtime_library_sha256 STREQUAL
     _wafer_expected_tx_runtime_library_sha256)
    message(FATAL_ERROR
      "Configured TX runtime library does not match the explicitly expected "
      "board-test SHA-256")
  endif()
  add_test(NAME wafer-board-ncc-pmu-readonly-probe
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_pmu_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ncc-pmu-readonly-probe
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 10000
  )
  set_tests_properties(wafer-board-ncc-pmu-readonly-probe PROPERTIES
    LABELS "board;hardware;pmu;register-read-only;safety"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 60
  )
  add_test(NAME wafer-board-instruction-family-safe
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_instruction_family_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-instruction-family-safe
      --suite safe
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-instruction-family-safe PROPERTIES
    LABELS "board;hardware;instruction;qualification;safe"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )
  add_test(NAME wafer-board-ct-vector-calibration-exact
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_vector_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ct-vector-calibration-exact
      --suite exact
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-ct-vector-calibration-exact PROPERTIES
    LABELS "board;hardware;instruction;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 21600
  )
  add_test(NAME wafer-board-ct-convert-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_convert_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-ct-convert-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-ct-convert-calibration PROPERTIES
    LABELS "board;hardware;instruction;ct;convert;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )
  add_test(NAME wafer-board-datamove-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_datamove_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-datamove-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(wafer-board-datamove-calibration PROPERTIES
    LABELS "board;hardware;instruction;datamove;layout;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )
  add_test(NAME wafer-board-ne-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ne_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-ne-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(wafer-board-ne-calibration PROPERTIES
    LABELS "board;hardware;ne;layout;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )
  add_test(NAME wafer-board-spm-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-spm-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-spm-calibration PROPERTIES
    LABELS "board;hardware;spm;capacity;alignment;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )
  add_test(NAME wafer-board-cache-coherence-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_cache_coherence_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-cache-coherence-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(wafer-board-cache-coherence-calibration PROPERTIES
    LABELS "board;hardware;cache;coherence;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )
  add_test(NAME wafer-board-single-op-add
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_single_op_add_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-single-op-add
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --repeat 2
  )
  set_tests_properties(wafer-board-single-op-add PROPERTIES
    LABELS "board;hardware;single-op"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 300
  )
  add_test(NAME wafer-board-single-gemm
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_single_gemm_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-single-gemm
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
      --repeat 1
  )
  set_tests_properties(wafer-board-single-gemm PROPERTIES
    LABELS "board;hardware;single-op;gemm"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 600
  )
  add_test(NAME wafer-board-mn-tiled-gemm
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_single_gemm_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-mn-tiled-gemm
      --gemm-case mn-tiled
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
      --repeat 1
  )
  set_tests_properties(wafer-board-mn-tiled-gemm PROPERTIES
    LABELS "board;hardware;single-op;gemm;tiling"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 600
  )
  if(WAFER_BOARD_TEST_EXPECTED_TILE_COUNT EQUAL 16)
    add_test(NAME wafer-board-per-rank-add-smoke
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-per-rank-add-smoke
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --launch-abi per-rank-pointer-block-v1
        --completion-timeout-ms 60000
        --repeat 2
    )
    set_tests_properties(wafer-board-per-rank-add-smoke PROPERTIES
      LABELS "board;hardware;multi-launch-smoke;single-op"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
    add_test(NAME wafer-board-kernel-grid-add
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-kernel-grid-add
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --launch-abi tx81-kernel-grid-pointer-table-v1
        --completion-timeout-ms 60000
        --repeat 2
    )
    set_tests_properties(wafer-board-kernel-grid-add PROPERTIES
      LABELS "board;hardware;kernel-grid;all-rank;single-op"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
    add_test(NAME wafer-board-model-add
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-model-add
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --launch-abi tx81-model-bootparam-v1
        --completion-timeout-ms 60000
        --repeat 2
    )
    set_tests_properties(wafer-board-model-add PROPERTIES
      LABELS "board;hardware;model;all-rank;single-op"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
    add_test(NAME wafer-board-cluster-direct-dte
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_collective_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-cluster-direct-dte
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
        --repeat 2
    )
    set_tests_properties(wafer-board-cluster-direct-dte PROPERTIES
      LABELS "board;hardware;cluster;all-rank;direct-dte"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
    add_test(NAME wafer-board-dte-ncc-execution-probe
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_dte_ncc_execution_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-dte-ncc-execution-probe
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
        --total-timeout-seconds 1200
    )
    set_tests_properties(wafer-board-dte-ncc-execution-probe PROPERTIES
      LABELS "board;hardware;cluster;all-rank;direct-dte;ncc;execution-probe"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1230
    )
    add_test(NAME wafer-board-full-card-barrier-probe
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_full_card_barrier_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-full-card-barrier-probe
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
    )
    set_tests_properties(wafer-board-full-card-barrier-probe PROPERTIES
      LABELS "board;hardware;cluster;all-rank;barrier;execution-probe"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 630
    )
    add_test(NAME wafer-board-k-sharded-gemm
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_k_sharded_gemm_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-k-sharded-gemm
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 600000
        --repeat 2
    )
    set_tests_properties(wafer-board-k-sharded-gemm PROPERTIES
      LABELS "board;hardware;heavy;cluster;all-rank;gemm;direct-dte"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  else()
    message(STATUS
      "Not registering the all-rank board test: it requires exactly 16 TX tiles")
  endif()
  unset(_wafer_board_qualification_variable)
  unset(_wafer_expected_tx_runtime_library_sha256)
  unset(_wafer_expected_tx_runtime_library_sha256_length)
  unset(_wafer_tx_runtime_library_sha256)
endif()

add_test(NAME wafer-numeric-dependency-record-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_numeric_deps.py
)

add_test(NAME wafer-pytorch-xla-runtime-build-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_pytorch_xla_runtime_build.py
)
set_tests_properties(wafer-pytorch-xla-runtime-build-python PROPERTIES
  LABELS "framework-importer;managed-dependencies"
)

add_test(NAME wafer-bulk-dependency-record-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_bulk_deps.py
)
set_tests_properties(wafer-bulk-dependency-record-python PROPERTIES
  LABELS "bulk-model;managed-dependencies"
)

add_test(NAME wafer-systemc-dependency-record-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_systemc_deps.py
)
set_tests_properties(wafer-systemc-dependency-record-python PROPERTIES
  LABELS "systemc-model;managed-dependencies"
)

set(_wafer_numeric_cmake_gate_args
  --cmake ${CMAKE_COMMAND}
  --source-dir ${CMAKE_SOURCE_DIR}
  --work-root ${CMAKE_CURRENT_BINARY_DIR}/numeric-cmake-gates
  --generator ${CMAKE_GENERATOR}
  --c-compiler ${CMAKE_C_COMPILER}
  --cxx-compiler ${CMAKE_CXX_COMPILER}
  --llvm-dir ${LLVM_DIR}
  --mlir-dir ${MLIR_DIR}
  --python ${Python3_EXECUTABLE}
)
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND _wafer_numeric_cmake_gate_args
    --generator-platform ${CMAKE_GENERATOR_PLATFORM})
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND _wafer_numeric_cmake_gate_args
    --generator-toolset ${CMAKE_GENERATOR_TOOLSET})
endif()
if(CMAKE_MAKE_PROGRAM)
  list(APPEND _wafer_numeric_cmake_gate_args
    --make-program ${CMAKE_MAKE_PROGRAM})
endif()
if(CMAKE_BUILD_TYPE)
  list(APPEND _wafer_numeric_cmake_gate_args
    --build-type ${CMAKE_BUILD_TYPE})
endif()
add_test(NAME wafer-numeric-cmake-configuration-gates
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_numeric_cmake_gates.py
    ${_wafer_numeric_cmake_gate_args}
)
set_tests_properties(wafer-numeric-cmake-configuration-gates PROPERTIES
  LABELS "numeric-model;configuration"
  TIMEOUT 180
)

set(_wafer_bulk_cmake_gate_args
  --cmake ${CMAKE_COMMAND}
  --source-dir ${CMAKE_SOURCE_DIR}
  --work-root ${CMAKE_CURRENT_BINARY_DIR}/bulk-cmake-gates
  --generator ${CMAKE_GENERATOR}
  --c-compiler ${CMAKE_C_COMPILER}
  --cxx-compiler ${CMAKE_CXX_COMPILER}
  --llvm-dir ${LLVM_DIR}
  --mlir-dir ${MLIR_DIR}
  --python ${Python3_EXECUTABLE}
)
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND _wafer_bulk_cmake_gate_args
    --generator-platform ${CMAKE_GENERATOR_PLATFORM})
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND _wafer_bulk_cmake_gate_args
    --generator-toolset ${CMAKE_GENERATOR_TOOLSET})
endif()
if(CMAKE_MAKE_PROGRAM)
  list(APPEND _wafer_bulk_cmake_gate_args
    --make-program ${CMAKE_MAKE_PROGRAM})
endif()
if(CMAKE_BUILD_TYPE)
  list(APPEND _wafer_bulk_cmake_gate_args
    --build-type ${CMAKE_BUILD_TYPE})
endif()
if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  list(APPEND _wafer_bulk_cmake_gate_args
    --numeric-root ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}
    --numeric-record ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD})
endif()
add_test(NAME wafer-bulk-cmake-configuration-gates
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_bulk_cmake_gates.py
    ${_wafer_bulk_cmake_gate_args}
)
set_tests_properties(wafer-bulk-cmake-configuration-gates PROPERTIES
  LABELS "bulk-model;configuration"
  TIMEOUT 240
)

set(_wafer_systemc_cmake_gate_args
  --cmake ${CMAKE_COMMAND}
  --source-dir ${CMAKE_SOURCE_DIR}
  --work-root ${CMAKE_CURRENT_BINARY_DIR}/systemc-cmake-gates
  --generator ${CMAKE_GENERATOR}
  --c-compiler ${CMAKE_C_COMPILER}
  --cxx-compiler ${CMAKE_CXX_COMPILER}
  --llvm-dir ${LLVM_DIR}
  --mlir-dir ${MLIR_DIR}
  --python ${Python3_EXECUTABLE}
)
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND _wafer_systemc_cmake_gate_args
    --generator-platform ${CMAKE_GENERATOR_PLATFORM})
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND _wafer_systemc_cmake_gate_args
    --generator-toolset ${CMAKE_GENERATOR_TOOLSET})
endif()
if(CMAKE_MAKE_PROGRAM)
  list(APPEND _wafer_systemc_cmake_gate_args
    --make-program ${CMAKE_MAKE_PROGRAM})
endif()
if(CMAKE_BUILD_TYPE)
  list(APPEND _wafer_systemc_cmake_gate_args
    --build-type ${CMAKE_BUILD_TYPE})
endif()
if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  list(APPEND _wafer_systemc_cmake_gate_args
    --numeric-root ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}
    --numeric-record ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD})
endif()
if(WAFER_ENABLE_SYSTEMC_MODEL)
  list(APPEND _wafer_systemc_cmake_gate_args
    --systemc-root ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_ROOT}
    --systemc-record ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_RECORD})
endif()
add_test(NAME wafer-systemc-cmake-configuration-gates
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_systemc_cmake_gates.py
    ${_wafer_systemc_cmake_gate_args}
)
set_tests_properties(wafer-systemc-cmake-configuration-gates PROPERTIES
  LABELS "systemc-model;configuration"
  TIMEOUT 240
)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_test(NAME wafer-numeric-dependency-configured-record-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/check_deps.py
      --numeric-only
      --numeric-root ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}
      --numeric-record ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD}
      --emit-numeric-cmake-snapshot
  )
  set_tests_properties(wafer-numeric-dependency-configured-record-python
    PROPERTIES PASS_REGULAR_EXPRESSION
      "wafer-numeric-model-canonical-snapshot")
endif()

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  if(NOT CMAKE_READELF OR NOT EXISTS "${CMAKE_READELF}")
    message(FATAL_ERROR
      "feature-on bulk link closure requires the configured readelf tool")
  endif()
  if(NOT CMAKE_NM OR NOT EXISTS "${CMAKE_NM}")
    message(FATAL_ERROR
      "feature-on bulk link closure requires the configured nm tool")
  endif()
  add_test(NAME wafer-bulk-dependency-configured-record-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/bulk_deps.py
      --root ${WAFER_BULK_MODEL_DEPS_CANONICAL_ROOT}
      --record ${WAFER_BULK_MODEL_DEPS_CANONICAL_RECORD}
      --emit-cmake-snapshot
  )
  set_tests_properties(wafer-bulk-dependency-configured-record-python
    PROPERTIES
      PASS_REGULAR_EXPRESSION "wafer-bulk-model-canonical-snapshot"
      LABELS "bulk-model;managed-dependencies")
  add_test(NAME wafer-bulk-feature-on-link-closure
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/check_bulk_enabled_link_closure.py
      --binary $<TARGET_FILE:wafer-compile>
      --readelf ${CMAKE_READELF}
      --nm ${CMAKE_NM}
  )
  set_tests_properties(wafer-bulk-feature-on-link-closure PROPERTIES
    LABELS "bulk-model;link-closure"
  )
  add_test(NAME wafer-bulk-qualification-cli
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/test_bulk_qualification_cli.py
      --tool $<TARGET_FILE:wafer-cmodel-qualify-bulk>
      --work-root ${CMAKE_CURRENT_BINARY_DIR}/bulk-qualification-cli
  )
  set_tests_properties(wafer-bulk-qualification-cli PROPERTIES
    LABELS "bulk-model;qualification"
    TIMEOUT 180
  )
endif()

if(WAFER_ENABLE_SYSTEMC_MODEL)
  add_test(NAME wafer-systemc-dependency-configured-record-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tools/systemc_deps.py
      --root ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_ROOT}
      --record ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_RECORD}
      --emit-cmake-snapshot
  )
  set_tests_properties(wafer-systemc-dependency-configured-record-python
    PROPERTIES
      PASS_REGULAR_EXPRESSION "wafer-systemc-model-canonical-snapshot"
      LABELS "systemc-model;managed-dependencies")
endif()
