find_program(WAFER_BOARD_CLANGXX
  NAMES clang++
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

function(wafer_add_board_python_contract test_name script)
  add_test(NAME ${test_name}
    COMMAND ${WAFER_PYTHON_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
  )
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Support:${CMAKE_CURRENT_SOURCE_DIR}/Cases"
    LABELS "board;host;calibration;contract"
    TIMEOUT 60
  )
endfunction()

set(_wafer_board_python_contracts
  wafer-cache-coherence-catalog=Cases/wafer_cache_coherence_calibration_catalog_test.py
  wafer-collective-algorithm-source-contract=Cases/wafer_collective_hardware_characterization_catalog_test.py
  wafer-collective-traffic-source-contract=Cases/wafer_collective_traffic_behavior_catalog_test.py
  wafer-ct-convert-catalog=Cases/wafer_ct_convert_calibration_catalog_test.py
  wafer-ct-opcode-disposition-catalog=Cases/wafer_ct_opcode_disposition_catalog_test.py
  wafer-ct-reduce-pool-capability-catalog=Cases/wafer_ct_reduce_pool_capability_catalog_test.py
  wafer-ct-vector-catalog=Cases/wafer_ct_vector_calibration_catalog_test.py
  wafer-ct-vuvloop-catalog=Cases/wafer_ct_vuvloop_semantics_catalog_test.py
  wafer-datamove-catalog=Cases/wafer_datamove_calibration_catalog_test.py
  wafer-datamove-extended-catalog=Cases/wafer_datamove_extended_calibration_catalog_test.py
  wafer-direct-dte-evidence-contract=Cases/wafer_direct_dte_board_evidence_test.py
  wafer-direct-dte-profile-contract=Cases/wafer_board_direct_dte_profile_test.py
  wafer-board-direct-dte-outer-deadline=Cases/wafer_board_direct_dte_timeout_test.py
  wafer-engine-pipeline-catalog=Cases/wafer_engine_pipeline_characterization_catalog_test.py
  wafer-complete-tile-add-profile-contract=Cases/wafer_board_complete_tile_add_profile_test.py
  wafer-complete-tile-barrier-contract=Cases/wafer_complete_tile_barrier_contract_test.py
  wafer-hardware-calibration-runner-contract=Cases/wafer_hardware_calibration_batch_runner_test.py
  wafer-hardware-calibration-matrix=Cases/wafer_hardware_calibration_matrix_test.py
  wafer-instruction-family-catalog=Cases/wafer_instruction_family_catalog_test.py
  wafer-memory-descriptor-catalog=Cases/wafer_memory_descriptor_calibration_catalog_test.py
  wafer-ncc-protocol-contract=Cases/wafer_ncc_probe_protocol_test.py
  wafer-ne-catalog=Cases/wafer_ne_calibration_catalog_test.py
  wafer-ne-tail-throughput-catalog=Cases/wafer_ne_tail_throughput_catalog_test.py
  wafer-calibration-inventory=Cases/wafer_pending_hardware_calibration_inventory_test.py
  wafer-spm-catalog=Cases/wafer_spm_calibration_catalog_test.py
  wafer-spm-cross-tile-contract=Cases/wafer_spm_cross_tile_conflict_probe_contract_test.py
  wafer-spm-sustained-contract=Cases/wafer_spm_sustained_conflict_probe_contract_test.py
  wafer-transport-pmu-catalog=Cases/wafer_transport_pmu_calibration_catalog_test.py
  wafer-unrepresentable-hardware-catalog=Cases/wafer_unrepresentable_hardware_behavior_catalog_test.py
  wafer-worker-memory-contention-contract=Cases/wafer_worker_memory_contention_characterization_test.py
  wafer-worker-placement-contract=Cases/wafer_worker_placement_characterization_test.py
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
  Cases/wafer_ncc_probe_plan_test.c
  Inputs/wafer_ncc_probe_plan.c)
target_include_directories(wafer-ncc-probe-plan-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs)
add_test(NAME wafer-ncc-probe-plan-contract
  COMMAND wafer-ncc-probe-plan-test)
set_tests_properties(wafer-ncc-probe-plan-contract PROPERTIES
  LABELS "board;host;ncc;contract"
  TIMEOUT 30)

add_executable(wafer-ncc-hazard-relation-test
  Cases/wafer_ncc_hazard_relation_test.c
  Inputs/wafer_ncc_hazard_relation.c)
target_include_directories(wafer-ncc-hazard-relation-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs)
add_test(NAME wafer-ncc-hazard-relation-contract
  COMMAND wafer-ncc-hazard-relation-test)
set_tests_properties(wafer-ncc-hazard-relation-contract PROPERTIES
  LABELS "board;host;ncc;contract"
  TIMEOUT 30)

if(WAFER_ENABLE_BOARD_TEST_EXECUTION)
  function(wafer_add_current_board_probe test_name script)
    add_test(NAME ${test_name}
      COMMAND ${WAFER_PYTHON_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
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
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Support:${CMAKE_CURRENT_SOURCE_DIR}/Cases;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps;WAFER_EXECUTE_HARDWARE_TESTS=1"
      LABELS "board;hardware;calibration;current-interface;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endfunction()

  function(wafer_add_current_source_board_case test_name script)
    add_test(NAME ${test_name}
      COMMAND ${WAFER_PYTHON_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
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
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Support:${CMAKE_CURRENT_SOURCE_DIR}/Cases;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps;WAFER_EXECUTE_HARDWARE_TESTS=1"
      LABELS "board;hardware;calibration;current-interface;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endfunction()

  wafer_add_current_board_probe(
    wafer-board-dte-ncc-execution-probe
    Cases/wafer_board_dte_ncc_execution_probe_test.py
    --mode 1 --payload-bytes 16 --total-timeout-seconds 420)
  wafer_add_current_board_probe(
    wafer-board-complete-tile-barrier-probe
    Support/wafer_board_complete_tile_barrier_probe_runner.py --participants 16)
  wafer_add_current_source_board_case(
    wafer-board-complete-tile-add
    Support/wafer_board_complete_tile_add_runner.py --repeat 2)
  wafer_add_current_source_board_case(
    wafer-board-complete-tile-add-profile
    Support/wafer_board_complete_tile_add_runner.py --profile --repeat 1)
  wafer_add_current_source_board_case(
    wafer-board-program-data-add
    Cases/wafer_board_single_op_add_test.py --repeat 2)
  wafer_add_current_board_probe(
    wafer-board-ddr-tile-offset-probe
    Support/wafer_board_ddr_tile_offset_probe_runner.py)
  wafer_add_current_board_probe(
    wafer-board-worker-placement-probe
    Support/wafer_board_worker_placement_characterization_runner.py
    --group worker-outstanding-ct-w0)
  wafer_add_current_board_probe(
    wafer-board-spm-cross-tile-conflict-probe
    Support/wafer_board_spm_cross_tile_conflict_probe_runner.py
    --case spm-cross-tile-ct-rdma-translation-0-phase-0-bytes-256-a-b)
  wafer_add_current_board_probe(
    wafer-board-ne-tail-throughput-probe
    Support/wafer_board_ne_tail_throughput_runner.py --group single-ne-tail)
  wafer_add_current_board_probe(
    wafer-board-engine-pipeline-characterization-probe
    Support/wafer_board_engine_pipeline_characterization_runner.py
    --mode board --cell single/ct/small)
  wafer_add_current_board_probe(
    wafer-board-ddr-active-tile-contention-probe
    Support/wafer_worker_memory_contention_characterization_driver.py
    --mode board
    --case ddr-contention-rdma-only-4k-contiguous-active-1)
endif()
