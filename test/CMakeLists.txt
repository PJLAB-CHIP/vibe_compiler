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

add_test(NAME wafer-board-all-rank-add-profile-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_profile_test.py
)
set_tests_properties(
  wafer-board-all-rank-add-profile-contract-python PROPERTIES
  LABELS "runtime;no-card;board;all-rank;profiler;safety;evidence"
  TIMEOUT 30
)

add_test(NAME wafer-board-direct-dte-profile-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_profile_test.py
)
set_tests_properties(
  wafer-board-direct-dte-profile-contract-python PROPERTIES
  LABELS "runtime;no-card;board;all-rank;direct-dte;profiler;safety;evidence"
  TIMEOUT 30
)

add_test(NAME wafer-direct-dte-board-evidence-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_direct_dte_board_evidence_test.py
)
set_tests_properties(wafer-direct-dte-board-evidence-python PROPERTIES
  LABELS "runtime;no-card;board;direct-dte;evidence"
  TIMEOUT 30
)

add_test(NAME wafer-hardware-calibration-batch-runner-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_hardware_calibration_batch_runner_test.py
)
set_tests_properties(
  wafer-hardware-calibration-batch-runner-python PROPERTIES
  LABELS "runtime;no-card;hardware-calibration;orchestration"
  TIMEOUT 60
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

add_test(NAME wafer-ct-vuvloop-semantics-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ct_vuvloop_semantics_catalog_test.py
)
set_tests_properties(wafer-ct-vuvloop-semantics-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;calibration;observation"
  TIMEOUT 30
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

add_test(NAME wafer-datamove-extended-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_datamove_extended_calibration_catalog_test.py
)
set_tests_properties(
  wafer-datamove-extended-calibration-catalog-python PROPERTIES
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

add_test(NAME wafer-ct-reduce-pool-capability-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ct_reduce_pool_capability_catalog_test.py
)
set_tests_properties(
  wafer-ct-reduce-pool-capability-catalog-python PROPERTIES
  LABELS "runtime;no-card;instruction;ct;calibration;coverage;negative"
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

add_test(NAME wafer-pending-hardware-calibration-inventory-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_pending_hardware_calibration_inventory_test.py
)
set_tests_properties(
  wafer-pending-hardware-calibration-inventory-python PROPERTIES
  LABELS "runtime;no-card;hardware;calibration;pending;coverage"
  TIMEOUT 60
)

add_test(NAME wafer-ddr-active-rank-contention-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_worker_memory_contention_characterization_test.py
)
set_tests_properties(
  wafer-ddr-active-rank-contention-contract-python PROPERTIES
  LABELS "runtime;no-card;hardware;ddr;contention;calibration;pending"
  TIMEOUT 120
)

add_test(NAME wafer-worker-placement-characterization-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_worker_placement_characterization_test.py
)
set_tests_properties(
  wafer-worker-placement-characterization-python PROPERTIES
  LABELS "runtime;no-card;hardware;worker;placement;calibration;pending"
  TIMEOUT 120
)

add_test(NAME wafer-unrepresentable-hardware-behavior-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_unrepresentable_hardware_behavior_catalog_test.py
)
set_tests_properties(
  wafer-unrepresentable-hardware-behavior-python PROPERTIES
  LABELS "runtime;no-card;hardware;calibration;fail-closed"
  TIMEOUT 30
)

add_test(NAME wafer-engine-pipeline-characterization-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_engine_pipeline_characterization_catalog_test.py
)
set_tests_properties(
  wafer-engine-pipeline-characterization-catalog-python PROPERTIES
  LABELS "runtime;no-card;hardware;calibration;engine;pipeline;pending"
  TIMEOUT 120
)

add_test(NAME wafer-collective-traffic-behavior-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_collective_traffic_behavior_catalog_test.py
)
set_tests_properties(
  wafer-collective-traffic-behavior-catalog-python PROPERTIES
  LABELS "runtime;no-card;hardware;calibration;collective-traffic;pending"
  TIMEOUT 120
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
  TIMEOUT 120
)

add_test(NAME wafer-ne-tail-throughput-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_ne_tail_throughput_catalog_test.py
)
set_tests_properties(wafer-ne-tail-throughput-catalog-python PROPERTIES
  LABELS "runtime;no-card;ne;performance;calibration;pending"
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

add_test(NAME wafer-memory-descriptor-calibration-catalog-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_memory_descriptor_calibration_catalog_test.py
)
set_tests_properties(
  wafer-memory-descriptor-calibration-catalog-python PROPERTIES
  LABELS "runtime;no-card;memory;dma;ddr;spm;calibration;oracle"
  TIMEOUT 60
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

add_test(NAME wafer-ddr-tile-offset-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_tile_offset_probe_test.py
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-compile $<TARGET_FILE:wafer-compile>
    --wafer-run $<TARGET_FILE:wafer-run>
    --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
    --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-tile-offset-contract
    --list-cases
)
set_tests_properties(wafer-ddr-tile-offset-contract-python PROPERTIES
  LABELS "runtime;no-card;all-rank;ddr;allocation;offset;calibration"
  TIMEOUT 30
)

add_test(NAME wafer-spm-cross-tile-conflict-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_spm_cross_tile_conflict_probe_contract_test.py
)
set_tests_properties(
  wafer-spm-cross-tile-conflict-contract-python PROPERTIES
  LABELS "runtime;no-card;all-rank;spm;conflict-equivalence;calibration"
  TIMEOUT 60
)

add_test(NAME wafer-spm-sustained-conflict-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_spm_sustained_conflict_probe_contract_test.py
)
set_tests_properties(
  wafer-spm-sustained-conflict-contract-python PROPERTIES
  LABELS "runtime;no-card;spm;conflict-equivalence;calibration;pending"
  TIMEOUT 120
)

add_test(NAME wafer-ddr-sparse-high-offset-contract-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_sparse_high_offset_probe_test.py
    --workspace-gib 40
    --list-cases
)
set_tests_properties(wafer-ddr-sparse-high-offset-contract-python PROPERTIES
  LABELS "runtime;no-card;ddr;allocation;offset;calibration"
  TIMEOUT 30
)

if(WAFER_ENABLE_IMPORTER_DEPS AND
   WAFER_ENABLE_SPMD_PARTITIONER_DEPS AND
   WAFER_XLA_SPMD_PARTITIONER_HELPER)
  set(_wafer_compiler_optimization_rank_one_cases
    reciprocal-implementation
    f16-common-factor
    resident-fanout-share
    consumer-local-recompute
    ready-order-movement-first
    gemm-aligned-physical-route
    gemm-tail-physical-route
  )
  set(_wafer_compiler_optimization_collective_cases
    tree-all-reduce
  )
  set(_wafer_collective_characterization_cases
    all-gather-direct-vs-ring-256b
    all-gather-direct-vs-ring-4096b
    all-gather-direct-vs-ring-65536b
    reduce-scatter-direct-vs-ring-256b
    reduce-scatter-direct-vs-ring-4096b
    reduce-scatter-direct-vs-ring-65536b
    all-reduce-ring-vs-tree-256b
    all-reduce-ring-vs-tree-4096b
    all-reduce-ring-vs-tree-65536b
  )
  set(_wafer_collective_traffic_behavior_cases
    all-to-all-src-dst-lane-256b
    all-to-all-src-dst-lane-4096b
    all-to-all-src-dst-lane-65536b
    collective-permute-cycle-forward-256b
    collective-permute-cycle-forward-4096b
    collective-permute-cycle-forward-65536b
    collective-permute-cycle-reverse-4096b
    collective-permute-opposite-pairs-4096b
    collective-permute-disjoint-adjacent-pairs-4096b
    collective-permute-sparse-roles-4096b
    collective-permute-two-epoch-chain-4096b
  )
  set(_wafer_pending_dte_case_specs
    "14:dte-four-source-fanin-correctness"
    "15:dte-raw-broadcast-fanout2-adjacent"
    "16:dte-raw-broadcast-fanout2-interleaved"
    "17:dte-raw-broadcast-fanout4-adjacent"
    "18:dte-raw-broadcast-fanout4-interleaved"
    "19:dte-raw-broadcast-fanout8-adjacent"
    "20:dte-raw-broadcast-fanout8-interleaved"
    "21:dte-raw-broadcast-fanout15-adjacent"
    "22:dte-raw-broadcast-fanout15-interleaved"
    "23:dte-raw-scatter-fanout2-adjacent"
    "24:dte-raw-scatter-fanout2-interleaved"
    "25:dte-raw-scatter-fanout4-adjacent"
    "26:dte-raw-scatter-fanout4-interleaved"
    "27:dte-raw-scatter-fanout8-adjacent"
    "28:dte-raw-scatter-fanout8-interleaved"
    "29:dte-raw-scatter-fanout15-adjacent"
    "30:dte-raw-scatter-fanout15-interleaved"
    "31:dte-raw-shuffle-source1d-sections2-target1"
    "32:dte-raw-shuffle-source1d-sections2-target8"
    "33:dte-raw-shuffle-source1d-sections4-target1"
    "34:dte-raw-shuffle-source1d-sections4-target8"
    "35:dte-raw-shuffle-source1d-sections8-target1"
    "36:dte-raw-shuffle-source1d-sections8-target8"
    "37:dte-raw-shuffle-source1d-sections15-target1"
    "38:dte-raw-shuffle-source1d-sections15-target8"
  )
  set(_wafer_tx81_objdump
    ${CMAKE_SOURCE_DIR}/third_party/tx8_deps/Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2/bin/riscv64-unknown-elf-objdump
  )
  if(NOT EXISTS "${_wafer_tx81_objdump}")
    message(FATAL_ERROR
      "Compiler optimization package tests require the vendored TX81 objdump")
  endif()
  add_test(NAME wafer-compiler-optimization-campaign-catalog-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_compiler_optimization_campaign_catalog_test.py
  )
  set_tests_properties(
    wafer-compiler-optimization-campaign-catalog-python PROPERTIES
    LABELS "runtime;no-card;compiler-optimization;catalog"
    TIMEOUT 60
  )
  add_test(NAME wafer-collective-characterization-catalog-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_collective_hardware_characterization_catalog_test.py
  )
  set_tests_properties(
    wafer-collective-characterization-catalog-python PROPERTIES
    LABELS "runtime;no-card;collective-characterization;catalog;pending"
    TIMEOUT 60
  )
  foreach(_wafer_compiler_optimization_case IN LISTS
      _wafer_compiler_optimization_rank_one_cases
      _wafer_compiler_optimization_collective_cases)
    add_test(
      NAME
        wafer-runtime-compiler-optimization-${_wafer_compiler_optimization_case}-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_compiler_optimization_campaign_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
        --wafer-run $<TARGET_FILE:wafer-run>
        --tx8-objdump ${_wafer_tx81_objdump}
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/runtime-compiler-optimization-${_wafer_compiler_optimization_case}
        --case ${_wafer_compiler_optimization_case}
        --no-card
    )
    set_tests_properties(
      wafer-runtime-compiler-optimization-${_wafer_compiler_optimization_case}-no-card
      PROPERTIES
      LABELS "runtime;no-card;compiler-optimization;paired"
      TIMEOUT 1800
    )
  endforeach()
  foreach(_wafer_collective_characterization_case IN LISTS
      _wafer_collective_characterization_cases)
    add_test(
      NAME
        wafer-runtime-collective-characterization-${_wafer_collective_characterization_case}-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_collective_characterization_test.py
        --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
        --wafer-run $<TARGET_FILE:wafer-run>
        --tx8-objdump ${_wafer_tx81_objdump}
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/runtime-collective-characterization-${_wafer_collective_characterization_case}
        --case ${_wafer_collective_characterization_case}
        --no-card
    )
    set_tests_properties(
      wafer-runtime-collective-characterization-${_wafer_collective_characterization_case}-no-card
      PROPERTIES
      LABELS
        "runtime;no-card;collective-characterization;paired;all-rank;direct-dte;pending"
      TIMEOUT 3600
    )
  endforeach()
  foreach(_wafer_collective_traffic_case IN LISTS
      _wafer_collective_traffic_behavior_cases)
    add_test(
      NAME
        wafer-runtime-collective-traffic-behavior-${_wafer_collective_traffic_case}-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_collective_traffic_behavior_test.py
        --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
        --wafer-run $<TARGET_FILE:wafer-run>
        --tx8-objdump ${_wafer_tx81_objdump}
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/runtime-collective-traffic-${_wafer_collective_traffic_case}
        --case ${_wafer_collective_traffic_case}
        --no-card
    )
    set_tests_properties(
      wafer-runtime-collective-traffic-behavior-${_wafer_collective_traffic_case}-no-card
      PROPERTIES
      LABELS
        "runtime;no-card;collective-traffic;all-rank;direct-dte;pending"
      TIMEOUT 3600
    )
  endforeach()
  add_test(NAME wafer-runtime-kernel-grid-add-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/runtime-kernel-grid-add
      --launch-kind kernel
      --no-card
  )
  set_tests_properties(
    wafer-runtime-kernel-grid-add-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;single-op;kernel-grid"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-kernel-grid-add-profile-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-kernel-grid-add-profile
      --launch-kind kernel
      --profile
      --no-card
  )
  set_tests_properties(
    wafer-runtime-kernel-grid-add-profile-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;single-op;kernel-grid;profiler"
    TIMEOUT 600
  )
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
  add_test(NAME wafer-runtime-cluster-direct-dte-profile-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_collective_test.py
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-cluster-direct-dte-profile
      --profile
      --repeat 1
      --no-card
  )
  set_tests_properties(
    wafer-runtime-cluster-direct-dte-profile-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;direct-dte;profiler"
    TIMEOUT 600
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
      --total-timeout-seconds 420
      --no-card
  )
  set_tests_properties(
    wafer-runtime-dte-ncc-execution-probe-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;direct-dte;ncc;probe-build"
    TIMEOUT 450
  )
  foreach(_wafer_pending_dte_case_spec IN LISTS
      _wafer_pending_dte_case_specs)
    string(REPLACE ":" ";" _wafer_pending_dte_case_parts
      "${_wafer_pending_dte_case_spec}")
    list(GET _wafer_pending_dte_case_parts 0 _wafer_pending_dte_mode)
    list(GET _wafer_pending_dte_case_parts 1 _wafer_pending_dte_case)
    add_test(
      NAME wafer-runtime-${_wafer_pending_dte_case}-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_dte_ncc_execution_probe_test.py
        --host-contract-only
        --mode ${_wafer_pending_dte_mode}
        --payload-bytes 4096
    )
    set_tests_properties(
      wafer-runtime-${_wafer_pending_dte_case}-no-card PROPERTIES
      LABELS
        "runtime;no-card;all-rank;direct-dte;host-contract;pending"
      TIMEOUT 30
    )
  endforeach()
  unset(_wafer_pending_dte_case_spec)
  unset(_wafer_pending_dte_case_parts)
  unset(_wafer_pending_dte_mode)
  unset(_wafer_pending_dte_case)
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
  add_test(NAME wafer-runtime-ddr-tile-offset-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_tile_offset_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-tile-offset-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ddr-tile-offset-probe-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;ddr;allocation;offset;probe-build"
    TIMEOUT 630
  )
  add_test(NAME wafer-runtime-ddr-cross-tile-conflict-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_tile_offset_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-cross-tile-conflict-probe
      --conflict-equivalence
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ddr-cross-tile-conflict-probe-no-card PROPERTIES
    LABELS
      "runtime;no-card;all-rank;cluster;ddr;conflict-equivalence;preflight"
    TIMEOUT 630
  )
  add_test(NAME wafer-runtime-ddr-active-rank-contention-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_worker_memory_contention_characterization_driver.py
      --mode no-card
      --group ddr-contention-rdma-only-4k-contiguous
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-active-rank-contention
      --repeat 4
  )
  set_tests_properties(
    wafer-runtime-ddr-active-rank-contention-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;cluster;ddr;contention;preflight"
    TIMEOUT 900
  )
  add_test(NAME wafer-runtime-worker-placement-characterization-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_worker_placement_characterization_test.py
      --group worker-placement-ct
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-worker-placement-characterization
      --repeat 3
      --no-card
  )
  set_tests_properties(
    wafer-runtime-worker-placement-characterization-no-card PROPERTIES
    LABELS "runtime;no-card;ncc;worker;placement;calibration;preflight"
    TIMEOUT 900
  )
  add_test(NAME wafer-runtime-ddr-sparse-high-offset-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_sparse_high_offset_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-sparse-high-offset-probe
      --workspace-gib 40
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ddr-sparse-high-offset-probe-no-card PROPERTIES
    LABELS "runtime;no-card;ddr;allocation;offset;probe-build"
    TIMEOUT 630
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
  add_test(NAME wafer-runtime-ncc-safe-observations-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_execution_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ncc-safe-observations-no-card
      --suite board-all-preflight
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ncc-safe-observations-no-card PROPERTIES
    LABELS "runtime;no-card;ncc;execution-model;calibration;preflight"
    TIMEOUT 600
  )
  add_test(NAME wafer-runtime-engine-pipeline-characterization-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_engine_pipeline_characterization_test.py
      --mode no-card
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-engine-pipeline-characterization
      --repeat 3
  )
  set_tests_properties(
    wafer-runtime-engine-pipeline-characterization-no-card PROPERTIES
    LABELS "runtime;no-card;ncc;engine;pipeline;calibration;pending"
    TIMEOUT 1800
  )
  add_test(NAME wafer-runtime-spm-sustained-conflict-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_sustained_conflict_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-spm-sustained-conflict-probe
      --repeat 4
      --no-card
  )
  set_tests_properties(
    wafer-runtime-spm-sustained-conflict-probe-no-card PROPERTIES
    LABELS "runtime;no-card;spm;engine-pair;calibration;pending"
    TIMEOUT 900
  )
  add_test(NAME wafer-engine-pipeline-production-gate-python
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_engine_pipeline_characterization_test.py
      --mode production-gate
  )
  set_tests_properties(
    wafer-engine-pipeline-production-gate-python PROPERTIES
    LABELS "runtime;no-card;engine;pipeline;production;fail-closed"
    TIMEOUT 60
  )
  function(wafer_add_ncc_calibration_preflight test_suffix suite)
    add_test(NAME wafer-runtime-ncc-${test_suffix}-no-card
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_execution_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/runtime-ncc-${test_suffix}-no-card
        --suite ${suite}
        --no-card
    )
    set_tests_properties(
      wafer-runtime-ncc-${test_suffix}-no-card PROPERTIES
      LABELS "runtime;no-card;ncc;execution-model;calibration;preflight"
      TIMEOUT 600
    )
  endfunction()
  wafer_add_ncc_calibration_preflight(
    queue-saturation
    queue-saturation-manual
  )
  wafer_add_ncc_calibration_preflight(
    worker-wait-scope
    worker-wait-scope-manual
  )
  wafer_add_ncc_calibration_preflight(
    worker-subset-scope
    worker-subset-scope-manual
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
      --suite all
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ct-vector-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ct-vuvloop-semantics-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_vuvloop_semantics_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ct-vuvloop-semantics-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ct-vuvloop-semantics-probe-no-card PROPERTIES
    LABELS "runtime;no-card;instruction;ct;calibration;probe-build"
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
      --suite all
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
  add_test(NAME wafer-runtime-datamove-extended-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_datamove_extended_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-datamove-extended-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-datamove-extended-calibration-probe-no-card PROPERTIES
    LABELS
      "runtime;no-card;instruction;datamove;layout;calibration;probe-build"
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
      --suite all
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ne-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;ne;layout;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-ne-tail-throughput-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ne_tail_throughput_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ne-tail-throughput
      --group single-ne-tail
      --repeat 3
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ne-tail-throughput-no-card PROPERTIES
    LABELS "runtime;no-card;ne;performance;calibration;pending"
    TIMEOUT 900
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
  add_test(NAME wafer-runtime-ddr-conflict-equivalence-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_cache_coherence_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-ddr-conflict-equivalence-probe
      --ddr-conflict-equivalence
      --no-card
  )
  set_tests_properties(
    wafer-runtime-ddr-conflict-equivalence-probe-no-card PROPERTIES
    LABELS "runtime;no-card;ddr;conflict-equivalence;calibration;preflight"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-memory-descriptor-calibration-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-memory-descriptor-calibration-probe
      --no-card
  )
  set_tests_properties(
    wafer-runtime-memory-descriptor-calibration-probe-no-card PROPERTIES
    LABELS "runtime;no-card;memory;dma;ddr;spm;calibration;probe-build"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-spm-conflict-equivalence-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-spm-conflict-equivalence-probe
      --conflict-equivalence
      --no-card
  )
  set_tests_properties(
    wafer-runtime-spm-conflict-equivalence-probe-no-card PROPERTIES
    LABELS "runtime;no-card;memory;spm;calibration;preflight"
    TIMEOUT 300
  )
  add_test(NAME wafer-runtime-spm-cross-tile-conflict-probe-no-card
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_cross_tile_conflict_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/runtime-spm-cross-tile-conflict-probe
      --case
        spm-physical-tile-ct-rdma-translation-0-phase-0-bytes-256-a-b
      --no-card
  )
  set_tests_properties(
    wafer-runtime-spm-cross-tile-conflict-probe-no-card PROPERTIES
    LABELS "runtime;no-card;all-rank;spm;conflict-equivalence;preflight"
    TIMEOUT 600
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
  add_test(NAME wafer-board-ncc-all-safe-observations
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_execution_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ncc-all-safe-observations
      --suite board-all-safe
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
      --repeat 3
  )
  set_tests_properties(wafer-board-ncc-all-safe-observations PROPERTIES
    LABELS "board;hardware;ncc;execution;calibration;safe"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 32400
  )

  function(wafer_add_isolated_ncc_observation test_suffix suite case_name)
    set(_wafer_ncc_observation_repeat 1)
    if(ARGC GREATER 3)
      set(_wafer_ncc_observation_repeat ${ARGV3})
    endif()
    add_test(NAME wafer-board-ncc-${test_suffix}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ncc_execution_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-ncc-${test_suffix}
        --suite ${suite}
        --case ${case_name}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --repeat ${_wafer_ncc_observation_repeat}
    )
    set_tests_properties(wafer-board-ncc-${test_suffix} PROPERTIES
      LABELS "board;hardware;ncc;execution;calibration;isolated"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
  endfunction()

  wafer_add_isolated_ncc_observation(
    constructor-return-address
    constructor-observation
    ct-constructor-return-address-nonnull
  )
  wafer_add_isolated_ncc_observation(
    mapped-spm-ne-depth4-no-local-wait
    mapped-spm-boundary-observation
    mapped-spm-ncc-ne-depth4-to-kcore-a-no-local-wait
  )
  wafer_add_isolated_ncc_observation(
    mapped-spm-ne-depth4-local-wait
    mapped-spm-boundary-observation
    mapped-spm-ncc-ne-depth4-to-kcore-b-local-wait
  )
  set(_wafer_ncc_strided_dependency_cases
    dependency-strided-1d-serial-observation
    dependency-strided-1d-window-observation
    dependency-strided-2d-serial-observation
    dependency-strided-2d-window-observation
    dependency-strided-3d-serial-observation
    dependency-strided-3d-window-observation
    dependency-strided-1d-war-serial-observation
    dependency-strided-1d-war-window-observation
    dependency-strided-2d-war-serial-observation
    dependency-strided-2d-war-window-observation
    dependency-strided-3d-war-serial-observation
    dependency-strided-3d-war-window-observation
    dependency-strided-1d-rar-serial-observation
    dependency-strided-1d-rar-window-observation
    dependency-strided-2d-rar-serial-observation
    dependency-strided-2d-rar-window-observation
    dependency-strided-3d-rar-serial-observation
    dependency-strided-3d-rar-window-observation
    dependency-strided-1d-waw-exact-serial-observation
    dependency-strided-1d-waw-exact-window-observation
    dependency-strided-1d-waw-partial-serial-observation
    dependency-strided-1d-waw-partial-window-observation
    dependency-strided-1d-waw-adjacent-serial-observation
    dependency-strided-1d-waw-adjacent-window-observation
    dependency-strided-2d-waw-exact-serial-observation
    dependency-strided-2d-waw-exact-window-observation
    dependency-strided-2d-waw-partial-serial-observation
    dependency-strided-2d-waw-partial-window-observation
    dependency-strided-2d-waw-adjacent-serial-observation
    dependency-strided-2d-waw-adjacent-window-observation
    dependency-strided-3d-waw-exact-serial-observation
    dependency-strided-3d-waw-exact-window-observation
    dependency-strided-3d-waw-partial-serial-observation
    dependency-strided-3d-waw-partial-window-observation
    dependency-strided-3d-waw-adjacent-serial-observation
    dependency-strided-3d-waw-adjacent-window-observation
  )
  foreach(_wafer_ncc_strided_case IN LISTS
      _wafer_ncc_strided_dependency_cases)
    wafer_add_isolated_ncc_observation(
      ${_wafer_ncc_strided_case}
      strided-dependency-observation
      ${_wafer_ncc_strided_case}
    )
  endforeach()
  wafer_add_isolated_ncc_observation(
    default-wait-worker1-boundary
    completion-scope-manual
    ne-worker1-depth6-default-wait-window
  )
  wafer_add_isolated_ncc_observation(
    byworker1-completion-control
    completion-scope-manual
    ne-worker1-depth6-byworker-wait-window
  )
  wafer_add_isolated_ncc_observation(
    local-fence-worker1-boundary
    completion-scope-manual
    ne-worker1-depth6-local-fence-wait-window
  )
  wafer_add_isolated_ncc_observation(
    ct-active-occupancy
    active-occupancy-manual
    ct-large-depth6-plus1-tight-occupancy-observation
    3
  )
  wafer_add_isolated_ncc_observation(
    ne-active-occupancy
    active-occupancy-manual
    ne-large-depth6-plus1-tight-occupancy-observation
    3
  )
  wafer_add_isolated_ncc_observation(
    rdma-active-occupancy
    active-occupancy-manual
    rdma-large-depth6-plus1-tight-occupancy-observation
    3
  )
  wafer_add_isolated_ncc_observation(
    wdma-active-occupancy
    active-occupancy-manual
    wdma-large-depth6-plus1-tight-occupancy-observation
    3
  )
  wafer_add_isolated_ncc_observation(
    tdma-active-occupancy
    active-occupancy-manual
    tdma-large-depth4-plus1-tight-occupancy-observation
    3
  )
  foreach(_wafer_ncc_join_mask IN ITEMS 001 010 100 011 101 110)
    wafer_add_isolated_ncc_observation(
      join${_wafer_ncc_join_mask}-unjoined-boundary
      cross-worker-boundary-manual
      workers012-join${_wafer_ncc_join_mask}-observe-unjoined
    )
  endforeach()
  set(_wafer_pending_ncc_queue_saturation_cases
    queue-saturation-ct-short-depth5-tight-window
    queue-saturation-ct-short-depth6-tight-window
    queue-saturation-ct-short-depth7-tight-window
    queue-saturation-ct-sustained-depth5-tight-window
    queue-saturation-ct-sustained-depth6-tight-window
    queue-saturation-ct-sustained-depth7-tight-window
    queue-saturation-ne-short-depth5-tight-window
    queue-saturation-ne-short-depth6-tight-window
    queue-saturation-ne-short-depth7-tight-window
    queue-saturation-ne-sustained-depth5-tight-window
    queue-saturation-ne-sustained-depth6-tight-window
    queue-saturation-ne-sustained-depth7-tight-window
    queue-saturation-rdma-short-depth5-tight-window
    queue-saturation-rdma-short-depth6-tight-window
    queue-saturation-rdma-short-depth7-tight-window
    queue-saturation-rdma-sustained-depth5-tight-window
    queue-saturation-rdma-sustained-depth6-tight-window
    queue-saturation-rdma-sustained-depth7-tight-window
    queue-saturation-wdma-short-depth5-tight-window
    queue-saturation-wdma-short-depth6-tight-window
    queue-saturation-wdma-short-depth7-tight-window
    queue-saturation-wdma-sustained-depth5-tight-window
    queue-saturation-wdma-sustained-depth6-tight-window
    queue-saturation-wdma-sustained-depth7-tight-window
    queue-saturation-tdma-short-depth3-tight-window
    queue-saturation-tdma-short-depth4-tight-window
    queue-saturation-tdma-short-depth5-tight-window
    queue-saturation-tdma-sustained-depth3-tight-window
    queue-saturation-tdma-sustained-depth4-tight-window
    queue-saturation-tdma-sustained-depth5-tight-window
  )
  set(_wafer_pending_ncc_worker_wait_scope_cases
    worker-wait-scope-ne-worker0-default-tight-window
    worker-wait-scope-ne-worker0-byworker-tight-window
    worker-wait-scope-ne-worker0-local-fence-tight-window
    worker-wait-scope-rdma-worker0-default-tight-window
    worker-wait-scope-rdma-worker0-byworker-tight-window
    worker-wait-scope-rdma-worker0-local-fence-tight-window
    worker-wait-scope-ne-worker1-default-tight-window
    worker-wait-scope-ne-worker1-byworker-tight-window
    worker-wait-scope-ne-worker1-local-fence-tight-window
    worker-wait-scope-rdma-worker1-default-tight-window
    worker-wait-scope-rdma-worker1-byworker-tight-window
    worker-wait-scope-rdma-worker1-local-fence-tight-window
    worker-wait-scope-ne-worker2-default-tight-window
    worker-wait-scope-ne-worker2-byworker-tight-window
    worker-wait-scope-ne-worker2-local-fence-tight-window
    worker-wait-scope-rdma-worker2-default-tight-window
    worker-wait-scope-rdma-worker2-byworker-tight-window
    worker-wait-scope-rdma-worker2-local-fence-tight-window
  )
  set(_wafer_pending_ncc_worker_subset_scope_cases
    worker-subset-ne-target0-exclude-tight-window
    worker-subset-ne-target0-include-tight-window
    worker-subset-rdma-target0-exclude-tight-window
    worker-subset-rdma-target0-include-tight-window
    worker-subset-ne-target1-exclude-tight-window
    worker-subset-ne-target1-include-tight-window
    worker-subset-rdma-target1-exclude-tight-window
    worker-subset-rdma-target1-include-tight-window
    worker-subset-ne-target2-exclude-tight-window
    worker-subset-ne-target2-include-tight-window
    worker-subset-rdma-target2-exclude-tight-window
    worker-subset-rdma-target2-include-tight-window
  )
  foreach(_wafer_ncc_case IN LISTS
      _wafer_pending_ncc_queue_saturation_cases)
    wafer_add_isolated_ncc_observation(
      ${_wafer_ncc_case}
      queue-saturation-manual
      ${_wafer_ncc_case}
      3
    )
  endforeach()
  foreach(_wafer_ncc_case IN LISTS
      _wafer_pending_ncc_worker_wait_scope_cases)
    wafer_add_isolated_ncc_observation(
      ${_wafer_ncc_case}
      worker-wait-scope-manual
      ${_wafer_ncc_case}
      3
    )
  endforeach()
  foreach(_wafer_ncc_case IN LISTS
      _wafer_pending_ncc_worker_subset_scope_cases)
    wafer_add_isolated_ncc_observation(
      ${_wafer_ncc_case}
      worker-subset-scope-manual
      ${_wafer_ncc_case}
      3
    )
  endforeach()
  unset(_wafer_ncc_case)
  unset(_wafer_pending_ncc_queue_saturation_cases)
  unset(_wafer_pending_ncc_worker_wait_scope_cases)
  unset(_wafer_pending_ncc_worker_subset_scope_cases)
  foreach(_wafer_instruction_suite IN ITEMS
      ct-capability
      qualification-regression)
    if(_wafer_instruction_suite STREQUAL "ct-capability")
      set(_wafer_instruction_timeout 10800)
    else()
      set(_wafer_instruction_timeout 1800)
    endif()
    add_test(
      NAME wafer-board-instruction-family-${_wafer_instruction_suite}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_instruction_family_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-instruction-family-${_wafer_instruction_suite}
        --suite ${_wafer_instruction_suite}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --observation-samples 3
    )
    set_tests_properties(
      wafer-board-instruction-family-${_wafer_instruction_suite} PROPERTIES
      LABELS "board;hardware;instruction;qualification;focused"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT ${_wafer_instruction_timeout}
    )
  endforeach()
  set(_wafer_pending_argmin_domain_cases
    peripheral-argmin-tie-f16-observed
    peripheral-argmin-nan-f16-observed
  )
  foreach(_wafer_argmin_case IN LISTS _wafer_pending_argmin_domain_cases)
    add_test(
      NAME wafer-board-instruction-family-${_wafer_argmin_case}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_instruction_family_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-instruction-family-${_wafer_argmin_case}
        --case ${_wafer_argmin_case}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --observation-samples 3
    )
    set_tests_properties(
      wafer-board-instruction-family-${_wafer_argmin_case} PROPERTIES
      LABELS
        "board;hardware;instruction;ct;argmin;numeric-domain;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endforeach()
  unset(_wafer_argmin_case)
  unset(_wafer_pending_argmin_domain_cases)
  set(_wafer_pending_unpool_collision_cases
    unpool-index-f16-repeated-overlap-observed
    unpool-mask-f16-repeated-overlap-observed
  )
  foreach(_wafer_unpool_case IN LISTS
      _wafer_pending_unpool_collision_cases)
    add_test(
      NAME wafer-board-instruction-family-${_wafer_unpool_case}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_instruction_family_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-instruction-family-${_wafer_unpool_case}
        --case ${_wafer_unpool_case}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --observation-samples 3
    )
    set_tests_properties(
      wafer-board-instruction-family-${_wafer_unpool_case} PROPERTIES
      LABELS
        "board;hardware;instruction;ct;unpool;collision;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endforeach()
  unset(_wafer_unpool_case)
  unset(_wafer_pending_unpool_collision_cases)
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_engine_pipeline_characterization_test.py
      --emit-board-group-keys
    RESULT_VARIABLE _wafer_engine_pipeline_groups_result
    OUTPUT_VARIABLE _wafer_engine_pipeline_groups_output
    ERROR_VARIABLE _wafer_engine_pipeline_groups_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _wafer_engine_pipeline_groups_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to enumerate engine/pipeline board activation groups: "
      "${_wafer_engine_pipeline_groups_error}")
  endif()
  string(REPLACE "\n" ";"
    _wafer_engine_pipeline_board_groups
    "${_wafer_engine_pipeline_groups_output}")
  foreach(_wafer_engine_pipeline_group IN LISTS
      _wafer_engine_pipeline_board_groups)
    add_test(
      NAME wafer-board-engine-pipeline-${_wafer_engine_pipeline_group}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_engine_pipeline_characterization_test.py
        --mode board
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-engine-pipeline-${_wafer_engine_pipeline_group}
        --group ${_wafer_engine_pipeline_group}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --repeat 3
    )
    set_tests_properties(
      wafer-board-engine-pipeline-${_wafer_engine_pipeline_group}
      PROPERTIES
      LABELS "board;hardware;ncc;engine;pipeline;performance;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endforeach()
  unset(_wafer_engine_pipeline_group)
  unset(_wafer_engine_pipeline_board_groups)
  unset(_wafer_engine_pipeline_groups_output)
  unset(_wafer_engine_pipeline_groups_error)
  unset(_wafer_engine_pipeline_groups_result)
  add_test(NAME wafer-board-ne-tail-throughput-single-ne-tail
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ne_tail_throughput_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ne-tail-throughput-single-ne-tail
      --group single-ne-tail
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
      --repeat 3
  )
  set_tests_properties(
    wafer-board-ne-tail-throughput-single-ne-tail PROPERTIES
    LABELS "board;hardware;ne;performance;calibration;pending"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_sustained_conflict_probe_test.py
      --emit-board-case-keys
    RESULT_VARIABLE _wafer_spm_sustained_groups_result
    OUTPUT_VARIABLE _wafer_spm_sustained_groups_output
    ERROR_VARIABLE _wafer_spm_sustained_groups_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _wafer_spm_sustained_groups_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to enumerate sustained SPM conflict groups: "
      "${_wafer_spm_sustained_groups_error}")
  endif()
  string(REPLACE "\n" ";"
    _wafer_spm_sustained_board_groups
    "${_wafer_spm_sustained_groups_output}")
  foreach(_wafer_spm_sustained_group IN LISTS
      _wafer_spm_sustained_board_groups)
    add_test(
      NAME wafer-board-spm-sustained-${_wafer_spm_sustained_group}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_sustained_conflict_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-spm-sustained-${_wafer_spm_sustained_group}
        --case ${_wafer_spm_sustained_group}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
        --repeat 4
    )
    set_tests_properties(
      wafer-board-spm-sustained-${_wafer_spm_sustained_group}
      PROPERTIES
      LABELS "board;hardware;spm;engine-pair;performance;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 900
    )
  endforeach()
  unset(_wafer_spm_sustained_group)
  unset(_wafer_spm_sustained_board_groups)
  unset(_wafer_spm_sustained_groups_output)
  unset(_wafer_spm_sustained_groups_error)
  unset(_wafer_spm_sustained_groups_result)
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_worker_memory_contention_characterization_driver.py
      --emit-board-group-keys
    RESULT_VARIABLE _wafer_ddr_active_rank_groups_result
    OUTPUT_VARIABLE _wafer_ddr_active_rank_groups_output
    ERROR_VARIABLE _wafer_ddr_active_rank_groups_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _wafer_ddr_active_rank_groups_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to enumerate DDR active-rank groups: "
      "${_wafer_ddr_active_rank_groups_error}")
  endif()
  string(REPLACE "\n" ";"
    _wafer_ddr_active_rank_board_groups
    "${_wafer_ddr_active_rank_groups_output}")
  foreach(_wafer_ddr_active_rank_group IN LISTS
      _wafer_ddr_active_rank_board_groups)
    add_test(
      NAME wafer-board-ddr-active-rank-${_wafer_ddr_active_rank_group}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_worker_memory_contention_characterization_driver.py
        --mode board
        --group ${_wafer_ddr_active_rank_group}
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-ddr-active-rank-${_wafer_ddr_active_rank_group}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 180000
        --repeat 4
    )
    set_tests_properties(
      wafer-board-ddr-active-rank-${_wafer_ddr_active_rank_group}
      PROPERTIES
      LABELS "board;hardware;cluster;all-rank;ddr;contention;performance;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 4500
    )
  endforeach()
  unset(_wafer_ddr_active_rank_group)
  unset(_wafer_ddr_active_rank_board_groups)
  unset(_wafer_ddr_active_rank_groups_output)
  unset(_wafer_ddr_active_rank_groups_error)
  unset(_wafer_ddr_active_rank_groups_result)
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_worker_placement_characterization_test.py
      --emit-board-group-keys
    RESULT_VARIABLE _wafer_worker_placement_groups_result
    OUTPUT_VARIABLE _wafer_worker_placement_groups_output
    ERROR_VARIABLE _wafer_worker_placement_groups_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _wafer_worker_placement_groups_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to enumerate worker placement groups: "
      "${_wafer_worker_placement_groups_error}")
  endif()
  string(REPLACE "\n" ";"
    _wafer_worker_placement_board_groups
    "${_wafer_worker_placement_groups_output}")
  foreach(_wafer_worker_placement_group IN LISTS
      _wafer_worker_placement_board_groups)
    add_test(
      NAME wafer-board-worker-placement-${_wafer_worker_placement_group}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_worker_placement_characterization_test.py
        --group ${_wafer_worker_placement_group}
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-worker-placement-${_wafer_worker_placement_group}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 30000
        --repeat 3
    )
    set_tests_properties(
      wafer-board-worker-placement-${_wafer_worker_placement_group}
      PROPERTIES
      LABELS "board;hardware;ncc;worker;placement;progress;performance;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endforeach()
  unset(_wafer_worker_placement_group)
  unset(_wafer_worker_placement_board_groups)
  unset(_wafer_worker_placement_groups_output)
  unset(_wafer_worker_placement_groups_error)
  unset(_wafer_worker_placement_groups_result)
  add_test(NAME wafer-board-ct-vector-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_vector_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ct-vector-calibration
      --suite all
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-ct-vector-calibration PROPERTIES
    LABELS "board;hardware;instruction;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 43200
  )
  add_test(NAME wafer-board-ct-vuvloop-semantics
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_vuvloop_semantics_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ct-vuvloop-semantics
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
      --observation-samples 3
  )
  set_tests_properties(wafer-board-ct-vuvloop-semantics PROPERTIES
    LABELS "board;hardware;instruction;ct;calibration;observation;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )
  add_test(NAME wafer-board-ct-convert-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ct_convert_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-ct-convert-calibration
      --suite all
      --stochastic-samples 3
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
    TIMEOUT 21600
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
    TIMEOUT 4500
  )
  add_test(NAME wafer-board-datamove-extended-calibration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_datamove_extended_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-datamove-extended-calibration
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
      --observation-samples 3
  )
  set_tests_properties(
    wafer-board-datamove-extended-calibration PROPERTIES
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
      --suite all
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
      --observation-samples 3
  )
  set_tests_properties(wafer-board-ne-calibration PROPERTIES
    LABELS "board;hardware;ne;layout;calibration;manual"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 21600
  )
  add_test(NAME wafer-board-spm-non-preferred-geometry
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-spm-non-preferred-geometry
      --case spm-base-offset-64-observation
      --case spm-base-offset-128-observation
      --case spm-base-offset-192-observation
      --case spm-length-128-observation
      --case spm-length-384-observation
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 30000
  )
  set_tests_properties(wafer-board-spm-non-preferred-geometry PROPERTIES
    LABELS "board;hardware;spm;alignment;calibration;focused"
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
    TIMEOUT 16200
  )
  add_test(NAME wafer-board-spm-conflict-equivalence-rank-one
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-spm-conflict-equivalence-rank-one
      --conflict-equivalence
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(
    wafer-board-spm-conflict-equivalence-rank-one PROPERTIES
    LABELS "board;hardware;memory;spm;conflict-equivalence;pending"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 21600
  )
  add_test(NAME wafer-board-ddr-conflict-equivalence-rank-one
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_cache_coherence_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-ddr-conflict-equivalence-rank-one
      --ddr-conflict-equivalence
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(
    wafer-board-ddr-conflict-equivalence-rank-one PROPERTIES
    LABELS "board;hardware;memory;ddr;conflict-equivalence;pending"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 21600
  )
  add_test(NAME wafer-board-memory-descriptor-parallel-expanded
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-memory-descriptor-parallel-expanded
      --parallel-pair-expanded
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(
    wafer-board-memory-descriptor-parallel-expanded PROPERTIES
    LABELS "board;hardware;memory;parallel;calibration;focused"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 7200
  )
  add_test(NAME wafer-board-memory-descriptor-strided-dma-focused
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-memory-descriptor-strided-dma-focused
      --case ddr-large-1d-stride-holes
      --case ddr-large-2d-stride-holes
      --case ddr-large-3d-stride-holes
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(
    wafer-board-memory-descriptor-strided-dma-focused PROPERTIES
    LABELS "board;hardware;memory;dma;ddr;strided;calibration;focused"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )
  add_test(NAME wafer-board-memory-engine-pair-new-offsets
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_memory_descriptor_calibration_probe_test.py
      --repo-root ${CMAKE_SOURCE_DIR}
      --wafer-compile $<TARGET_FILE:wafer-compile>
      --wafer-run $<TARGET_FILE:wafer-run>
      --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
      --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/board-memory-engine-pair-new-offsets
      --domain spm-bank-engine-pair
      --relative-spm-offset 4352
      --relative-spm-offset 65536
      --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
      --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
      --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
      --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
      --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
      --expected-runtime-library-sha256
        ${_wafer_expected_tx_runtime_library_sha256}
      --completion-timeout-ms 60000
  )
  set_tests_properties(
    wafer-board-memory-engine-pair-new-offsets PROPERTIES
    LABELS "board;hardware;memory;spm;engine-pair;calibration;focused"
    RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
    SKIP_RETURN_CODE 77
    TIMEOUT 10800
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
      --repeat 1
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
  foreach(_wafer_compiler_optimization_case IN LISTS
      _wafer_compiler_optimization_rank_one_cases)
    add_test(
      NAME
        wafer-board-compiler-optimization-${_wafer_compiler_optimization_case}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_compiler_optimization_campaign_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
        --wafer-run $<TARGET_FILE:wafer-run>
        --tx8-objdump ${_wafer_tx81_objdump}
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-compiler-optimization-${_wafer_compiler_optimization_case}
        --case ${_wafer_compiler_optimization_case}
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
        --repeat 3
    )
    set_tests_properties(
      wafer-board-compiler-optimization-${_wafer_compiler_optimization_case}
      PROPERTIES
      LABELS "board;hardware;compiler-optimization;paired;rank-one"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endforeach()
  if(WAFER_BOARD_TEST_EXPECTED_TILE_COUNT EQUAL 16)
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
        --launch-kind kernel
        --completion-timeout-ms 60000
        --repeat 2
    )
    set_tests_properties(wafer-board-kernel-grid-add PROPERTIES
      LABELS "board;hardware;kernel-grid;all-rank;single-op"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
    )
    add_test(NAME wafer-board-kernel-grid-add-profile
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_all_rank_add_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-kernel-grid-add-profile
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --launch-kind kernel
        --completion-timeout-ms 60000
        --profile
        --repeat 1
    )
    set_tests_properties(wafer-board-kernel-grid-add-profile PROPERTIES
      LABELS "board;hardware;kernel-grid;all-rank;single-op;profiler"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
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
    add_test(NAME wafer-board-cluster-direct-dte-profile
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_direct_dte_collective_test.py
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-cluster-direct-dte-profile
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 60000
        --profile
        --repeat 1
    )
    set_tests_properties(wafer-board-cluster-direct-dte-profile PROPERTIES
      LABELS "board;hardware;cluster;all-rank;direct-dte;profiler"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
    foreach(_wafer_compiler_optimization_case IN LISTS
        _wafer_compiler_optimization_collective_cases)
      add_test(
        NAME
          wafer-board-compiler-optimization-${_wafer_compiler_optimization_case}
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_compiler_optimization_campaign_test.py
          --wafer-compile $<TARGET_FILE:wafer-compile>
          --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
          --wafer-run $<TARGET_FILE:wafer-run>
          --tx8-objdump ${_wafer_tx81_objdump}
          --work-dir
            ${CMAKE_CURRENT_BINARY_DIR}/board-compiler-optimization-${_wafer_compiler_optimization_case}
          --case ${_wafer_compiler_optimization_case}
          --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
          --expected-runtime-version
            ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
          --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
          --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
          --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
          --expected-runtime-library-sha256
            ${_wafer_expected_tx_runtime_library_sha256}
          --completion-timeout-ms 60000
          --repeat 3
      )
      set_tests_properties(
        wafer-board-compiler-optimization-${_wafer_compiler_optimization_case}
        PROPERTIES
        LABELS
          "board;hardware;compiler-optimization;paired;all-rank;direct-dte"
        RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
        SKIP_RETURN_CODE 77
        TIMEOUT 3600
      )
    endforeach()
    foreach(_wafer_collective_characterization_case IN LISTS
        _wafer_collective_characterization_cases)
      add_test(
        NAME
          wafer-board-collective-characterization-${_wafer_collective_characterization_case}
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_collective_characterization_test.py
          --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
          --wafer-run $<TARGET_FILE:wafer-run>
          --tx8-objdump ${_wafer_tx81_objdump}
          --work-dir
            ${CMAKE_CURRENT_BINARY_DIR}/board-collective-characterization-${_wafer_collective_characterization_case}
          --case ${_wafer_collective_characterization_case}
          --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
          --expected-runtime-version
            ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
          --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
          --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
          --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
          --expected-runtime-library-sha256
            ${_wafer_expected_tx_runtime_library_sha256}
          --completion-timeout-ms 60000
          --repeat 3
      )
      set_tests_properties(
        wafer-board-collective-characterization-${_wafer_collective_characterization_case}
        PROPERTIES
        LABELS
          "board;hardware;collective-characterization;paired;all-rank;direct-dte;pending"
        RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
        SKIP_RETURN_CODE 77
        TIMEOUT 3600
      )
    endforeach()
    foreach(_wafer_collective_traffic_case IN LISTS
        _wafer_collective_traffic_behavior_cases)
      add_test(
        NAME
          wafer-board-collective-traffic-behavior-${_wafer_collective_traffic_case}
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_collective_traffic_behavior_test.py
          --wafer-compile-test $<TARGET_FILE:wafer-compile-test>
          --wafer-run $<TARGET_FILE:wafer-run>
          --tx8-objdump ${_wafer_tx81_objdump}
          --work-dir
            ${CMAKE_CURRENT_BINARY_DIR}/board-collective-traffic-${_wafer_collective_traffic_case}
          --case ${_wafer_collective_traffic_case}
          --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
          --expected-runtime-version
            ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
          --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
          --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
          --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
          --expected-runtime-library-sha256
            ${_wafer_expected_tx_runtime_library_sha256}
          --completion-timeout-ms 60000
          --repeat 3
      )
      set_tests_properties(
        wafer-board-collective-traffic-behavior-${_wafer_collective_traffic_case}
        PROPERTIES
        LABELS
          "board;hardware;collective-traffic;all-rank;direct-dte;pending"
        RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
        SKIP_RETURN_CODE 77
        TIMEOUT 3600
      )
    endforeach()
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
        --total-timeout-seconds 1800
    )
    set_tests_properties(wafer-board-dte-ncc-execution-probe PROPERTIES
      LABELS "board;hardware;cluster;all-rank;direct-dte;ncc;execution-probe"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1830
    )
    foreach(_wafer_pending_dte_case_spec IN LISTS
        _wafer_pending_dte_case_specs)
      string(REPLACE ":" ";" _wafer_pending_dte_case_parts
        "${_wafer_pending_dte_case_spec}")
      list(GET _wafer_pending_dte_case_parts 0 _wafer_pending_dte_mode)
      list(GET _wafer_pending_dte_case_parts 1 _wafer_pending_dte_case)
      add_test(
        NAME wafer-board-${_wafer_pending_dte_case}
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_dte_ncc_execution_probe_test.py
          --repo-root ${CMAKE_SOURCE_DIR}
          --wafer-compile $<TARGET_FILE:wafer-compile>
          --wafer-run $<TARGET_FILE:wafer-run>
          --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
          --work-dir
            ${CMAKE_CURRENT_BINARY_DIR}/board-${_wafer_pending_dte_case}
          --mode ${_wafer_pending_dte_mode}
          --payload-bytes 4096
          --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
          --expected-runtime-version
            ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
          --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
          --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
          --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
          --expected-runtime-library-sha256
            ${_wafer_expected_tx_runtime_library_sha256}
          --completion-timeout-ms 60000
          --total-timeout-seconds 420
      )
      set_tests_properties(
        wafer-board-${_wafer_pending_dte_case} PROPERTIES
        LABELS
          "board;hardware;cluster;all-rank;direct-dte;single-case;pending"
        RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
        SKIP_RETURN_CODE 77
        TIMEOUT 450
      )
    endforeach()
    unset(_wafer_pending_dte_case_spec)
    unset(_wafer_pending_dte_case_parts)
    unset(_wafer_pending_dte_mode)
    unset(_wafer_pending_dte_case)
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
    add_test(NAME wafer-board-ddr-tile-offset-probe
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_tile_offset_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-ddr-tile-offset-probe
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 600000
        --repeat 1
    )
    set_tests_properties(wafer-board-ddr-tile-offset-probe PROPERTIES
      LABELS
        "board;hardware;cluster;all-rank;ddr;allocation;offset;calibration;manual"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1500
    )
    add_test(NAME wafer-board-ddr-conflict-equivalence-cross-tile
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_tile_offset_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-ddr-conflict-equivalence-cross-tile
        --conflict-equivalence
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
    set_tests_properties(
      wafer-board-ddr-conflict-equivalence-cross-tile PROPERTIES
      LABELS
        "board;hardware;cluster;all-rank;ddr;conflict-equivalence;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 7200
    )
    add_test(NAME wafer-board-spm-conflict-equivalence-cross-tile
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_spm_cross_tile_conflict_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-spm-conflict-equivalence-cross-tile
        --case
          spm-physical-tile-ct-rdma-translation-0-phase-0-bytes-256-a-b
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 180000
        --repeat 4
    )
    set_tests_properties(
      wafer-board-spm-conflict-equivalence-cross-tile PROPERTIES
      LABELS
        "board;hardware;cluster;all-rank;spm;conflict-equivalence;pending"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
    add_test(NAME wafer-board-ddr-sparse-high-offset-probe
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/wafer_board_ddr_sparse_high_offset_probe_test.py
        --repo-root ${CMAKE_SOURCE_DIR}
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --llvm-clangxx ${LLVM_TOOLS_BINARY_DIR}/clang++
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-ddr-sparse-high-offset-probe
        --workspace-gib 40
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 600000
        --repeat 1
    )
    set_tests_properties(
      wafer-board-ddr-sparse-high-offset-probe PROPERTIES
      LABELS "board;hardware;ddr;allocation;offset;calibration;manual"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1500
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
unset(_wafer_compiler_optimization_case)
unset(_wafer_compiler_optimization_collective_cases)
unset(_wafer_compiler_optimization_rank_one_cases)
unset(_wafer_collective_characterization_case)
unset(_wafer_collective_characterization_cases)
unset(_wafer_collective_traffic_case)
unset(_wafer_collective_traffic_behavior_cases)
unset(_wafer_tx81_objdump)

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
