find_program(WAFER_BOARD_CLANGXX
  NAMES clang++
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

function(wafer_add_board_python_contract test_name script)
  add_test(NAME ${test_name}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
  )
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    LABELS "board;host;calibration;contract"
    TIMEOUT 60
  )
endfunction()

set(_wafer_board_python_contracts
  wafer-cache-coherence-catalog=wafer_cache_coherence_calibration_catalog_test.py
  wafer-collective-algorithm-source-contract=wafer_collective_hardware_characterization_catalog_test.py
  wafer-collective-traffic-source-contract=wafer_collective_traffic_behavior_catalog_test.py
  wafer-ct-convert-catalog=wafer_ct_convert_calibration_catalog_test.py
  wafer-ct-opcode-disposition-catalog=wafer_ct_opcode_disposition_catalog_test.py
  wafer-ct-reduce-pool-capability-catalog=wafer_ct_reduce_pool_capability_catalog_test.py
  wafer-ct-vector-catalog=wafer_ct_vector_calibration_catalog_test.py
  wafer-ct-vuvloop-catalog=wafer_ct_vuvloop_semantics_catalog_test.py
  wafer-datamove-catalog=wafer_datamove_calibration_catalog_test.py
  wafer-datamove-extended-catalog=wafer_datamove_extended_calibration_catalog_test.py
  wafer-direct-dte-evidence-contract=wafer_direct_dte_board_evidence_test.py
  wafer-engine-pipeline-catalog=wafer_engine_pipeline_characterization_catalog_test.py
  wafer-complete-tile-add-profile-contract=wafer_board_complete_tile_add_profile_test.py
  wafer-complete-tile-barrier-contract=wafer_complete_tile_barrier_contract_test.py
  wafer-hardware-calibration-runner-contract=wafer_hardware_calibration_batch_runner_test.py
  wafer-hardware-calibration-matrix=wafer_hardware_calibration_matrix_test.py
  wafer-instruction-family-catalog=wafer_instruction_family_catalog_test.py
  wafer-memory-descriptor-catalog=wafer_memory_descriptor_calibration_catalog_test.py
  wafer-ncc-protocol-contract=wafer_ncc_probe_protocol_test.py
  wafer-ne-catalog=wafer_ne_calibration_catalog_test.py
  wafer-ne-tail-throughput-catalog=wafer_ne_tail_throughput_catalog_test.py
  wafer-calibration-inventory=wafer_pending_hardware_calibration_inventory_test.py
  wafer-spm-catalog=wafer_spm_calibration_catalog_test.py
  wafer-spm-cross-tile-contract=wafer_spm_cross_tile_conflict_probe_contract_test.py
  wafer-spm-sustained-contract=wafer_spm_sustained_conflict_probe_contract_test.py
  wafer-transport-pmu-catalog=wafer_transport_pmu_calibration_catalog_test.py
  wafer-unrepresentable-hardware-catalog=wafer_unrepresentable_hardware_behavior_catalog_test.py
  wafer-worker-memory-contention-contract=wafer_worker_memory_contention_characterization_test.py
  wafer-worker-placement-contract=wafer_worker_placement_characterization_test.py
)
foreach(_wafer_board_contract IN LISTS _wafer_board_python_contracts)
  string(REPLACE "=" ";" _wafer_board_contract_fields
    "${_wafer_board_contract}")
  list(GET _wafer_board_contract_fields 0 _wafer_board_contract_name)
  list(GET _wafer_board_contract_fields 1 _wafer_board_contract_script)
  wafer_add_board_python_contract(
    ${_wafer_board_contract_name} ${_wafer_board_contract_script})
endforeach()

add_executable(wafer-ncc-probe-plan-test
  wafer_ncc_probe_plan_test.c
  Inputs/wafer_ncc_probe_plan.c)
target_include_directories(wafer-ncc-probe-plan-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs)
add_test(NAME wafer-ncc-probe-plan-contract
  COMMAND wafer-ncc-probe-plan-test)
set_tests_properties(wafer-ncc-probe-plan-contract PROPERTIES
  LABELS "board;host;ncc;contract"
  TIMEOUT 30)

add_executable(wafer-ncc-hazard-relation-test
  wafer_ncc_hazard_relation_test.c
  Inputs/wafer_ncc_hazard_relation.c)
target_include_directories(wafer-ncc-hazard-relation-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs)
add_test(NAME wafer-ncc-hazard-relation-contract
  COMMAND wafer-ncc-hazard-relation-test)
set_tests_properties(wafer-ncc-hazard-relation-contract PROPERTIES
  LABELS "board;host;ncc;contract"
  TIMEOUT 30)

function(wafer_add_raw_probe_no_card test_name script)
  add_test(NAME ${test_name}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${WAFER_BOARD_CLANGXX}
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
      ${ARGN}
      --no-card
  )
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    LABELS "board;no-card;calibration;current-interface"
    TIMEOUT 600
  )
endfunction()

wafer_add_raw_probe_no_card(
  wafer-runtime-dte-ncc-execution-probe-no-card
  wafer_board_dte_ncc_execution_probe_test.py
  --mode 1 --payload-bytes 16 --total-timeout-seconds 420)
wafer_add_raw_probe_no_card(
  wafer-runtime-complete-tile-barrier-probe-no-card
  wafer_board_complete_tile_barrier_probe_test.py --participants 16)
wafer_add_raw_probe_no_card(
  wafer-runtime-ddr-tile-offset-probe-no-card
  wafer_board_ddr_tile_offset_probe_test.py)
wafer_add_raw_probe_no_card(
  wafer-runtime-worker-placement-probe-no-card
  wafer_board_worker_placement_characterization_test.py
  --group worker-outstanding-ct-w0)
wafer_add_raw_probe_no_card(
  wafer-runtime-ddr-sparse-high-offset-probe-no-card
  wafer_board_ddr_sparse_high_offset_probe_test.py --workspace-gib 36)
set_tests_properties(wafer-runtime-ddr-sparse-high-offset-probe-no-card
  PROPERTIES LABELS "board;no-card;calibration;current-interface;heavy")
wafer_add_raw_probe_no_card(
  wafer-runtime-ncc-pmu-readonly-probe-no-card
  wafer_board_ncc_pmu_probe_test.py)
wafer_add_raw_probe_no_card(
  wafer-runtime-ncc-execution-probe-no-card
  wafer_board_ncc_execution_probe_test.py --suite build-smoke)
wafer_add_raw_probe_no_card(
  wafer-runtime-instruction-family-probe-no-card
  wafer_board_instruction_family_probe_test.py --case ct-neg-f16)
wafer_add_raw_probe_no_card(
  wafer-runtime-ct-vector-probe-no-card
  wafer_board_ct_vector_calibration_probe_test.py
  --case ct-op000-arithop_v_v_abs-f16-main)
wafer_add_raw_probe_no_card(
  wafer-runtime-ct-convert-probe-no-card
  wafer_board_ct_convert_calibration_probe_test.py
  --case ct-op139-convert-int8-fp16-main)
wafer_add_raw_probe_no_card(
  wafer-runtime-ct-vuvloop-probe-no-card
  wafer_board_ct_vuvloop_semantics_probe_test.py
  --case ct-vuvloop-rectangular-control)
wafer_add_raw_probe_no_card(
  wafer-runtime-datamove-probe-no-card
  wafer_board_datamove_calibration_probe_test.py
  --case datamove-transpose-37x53)
wafer_add_raw_probe_no_card(
  wafer-runtime-datamove-extended-probe-no-card
  wafer_board_datamove_extended_calibration_probe_test.py
  --case datamove-raw-concat-c-n2h7w9-c33-c32)
wafer_add_raw_probe_no_card(
  wafer-runtime-ne-probe-no-card
  wafer_board_ne_calibration_probe_test.py --case ne-f16-large-nn)
wafer_add_raw_probe_no_card(
  wafer-runtime-spm-probe-no-card
  wafer_board_spm_calibration_probe_test.py --case spm-lower-bound-256)
wafer_add_raw_probe_no_card(
  wafer-runtime-cache-coherence-probe-no-card
  wafer_board_cache_coherence_calibration_probe_test.py
  --case cache-host-h2d-to-kcore-read)
wafer_add_raw_probe_no_card(
  wafer-runtime-memory-descriptor-probe-no-card
  wafer_board_memory_descriptor_calibration_probe_test.py
  --case dma-src-offset256-dst-offset4096-tail4097-u8)
wafer_add_raw_probe_no_card(
  wafer-runtime-spm-sustained-probe-no-card
  wafer_board_spm_sustained_conflict_probe_test.py
  --case spm-conflict-pilot-4k-a-b)
wafer_add_raw_probe_no_card(
  wafer-runtime-spm-cross-tile-conflict-probe-no-card
  wafer_board_spm_cross_tile_conflict_probe_test.py
  --case spm-cross-tile-ct-rdma-translation-0-phase-0-bytes-256-a-b)
wafer_add_raw_probe_no_card(
  wafer-runtime-ne-tail-throughput-no-card
  wafer_board_ne_tail_throughput_test.py --group single-ne-tail)

function(wafer_add_current_source_no_card test_name script)
  add_test(NAME ${test_name}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
      ${ARGN}
      --no-card
  )
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    LABELS "board;no-card;calibration;current-interface"
    TIMEOUT 600
  )
endfunction()

wafer_add_current_source_no_card(
  wafer-runtime-complete-tile-add-no-card
  wafer_board_complete_tile_add_test.py)
wafer_add_current_source_no_card(
  wafer-runtime-complete-tile-add-profile-no-card
  wafer_board_complete_tile_add_test.py --profile)

add_test(NAME wafer-runtime-engine-pipeline-characterization-no-card
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/wafer_board_engine_pipeline_characterization_test.py
    --mode no-card
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-compile $<TARGET_FILE:wafer-compile>
    --wafer-run $<TARGET_FILE:wafer-run>
    --llvm-clangxx ${WAFER_BOARD_CLANGXX}
    --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/wafer-runtime-engine-pipeline-characterization-no-card
    --cell single/ct/small)
set_tests_properties(wafer-runtime-engine-pipeline-characterization-no-card
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    LABELS "board;no-card;calibration;current-interface"
    TIMEOUT 600)

add_test(NAME wafer-runtime-ddr-active-tile-contention-no-card
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/wafer_worker_memory_contention_characterization_driver.py
    --mode no-card
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-compile $<TARGET_FILE:wafer-compile>
    --wafer-run $<TARGET_FILE:wafer-run>
    --llvm-clangxx ${WAFER_BOARD_CLANGXX}
    --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/wafer-runtime-ddr-active-tile-contention-no-card
    --case ddr-contention-rdma-only-4k-contiguous-active-1)
set_tests_properties(wafer-runtime-ddr-active-tile-contention-no-card
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    LABELS "board;no-card;calibration;current-interface"
    TIMEOUT 600)

if(WAFER_ENABLE_BOARD_TEST_EXECUTION)
  function(wafer_add_current_board_probe test_name script)
    add_test(NAME ${test_name}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${WAFER_BOARD_CLANGXX}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        ${ARGN}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256}
    )
    set_tests_properties(${test_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps;WAFER_EXECUTE_HARDWARE_TESTS=1"
      LABELS "board;hardware;calibration;current-interface;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endfunction()

  function(wafer_add_current_source_board_case test_name script)
    add_test(NAME ${test_name}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        ${ARGN}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_LIBRARY_SHA256}
    )
    set_tests_properties(${test_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps;WAFER_EXECUTE_HARDWARE_TESTS=1"
      LABELS "board;hardware;calibration;current-interface;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endfunction()

  wafer_add_current_board_probe(
    wafer-board-dte-ncc-execution-probe
    wafer_board_dte_ncc_execution_probe_test.py
    --mode 1 --payload-bytes 16 --total-timeout-seconds 420)
  wafer_add_current_board_probe(
    wafer-board-complete-tile-barrier-probe
    wafer_board_complete_tile_barrier_probe_test.py --participants 16)
  wafer_add_current_source_board_case(
    wafer-board-complete-tile-add
    wafer_board_complete_tile_add_test.py --repeat 2)
  wafer_add_current_source_board_case(
    wafer-board-complete-tile-add-profile
    wafer_board_complete_tile_add_test.py --profile --repeat 1)
  wafer_add_current_source_board_case(
    wafer-board-program-data-add
    wafer_board_single_op_add_test.py --repeat 2)
  wafer_add_current_board_probe(
    wafer-board-ddr-tile-offset-probe
    wafer_board_ddr_tile_offset_probe_test.py)
  wafer_add_current_board_probe(
    wafer-board-worker-placement-probe
    wafer_board_worker_placement_characterization_test.py
    --group worker-outstanding-ct-w0)
  wafer_add_current_board_probe(
    wafer-board-spm-cross-tile-conflict-probe
    wafer_board_spm_cross_tile_conflict_probe_test.py
    --case spm-cross-tile-ct-rdma-translation-0-phase-0-bytes-256-a-b)
  wafer_add_current_board_probe(
    wafer-board-ne-tail-throughput-probe
    wafer_board_ne_tail_throughput_test.py --group single-ne-tail)
  wafer_add_current_board_probe(
    wafer-board-engine-pipeline-characterization-probe
    wafer_board_engine_pipeline_characterization_test.py
    --mode board --cell single/ct/small)
  wafer_add_current_board_probe(
    wafer-board-ddr-active-tile-contention-probe
    wafer_worker_memory_contention_characterization_driver.py
    --mode board
    --case ddr-contention-rdma-only-4k-contiguous-active-1)
endif()
