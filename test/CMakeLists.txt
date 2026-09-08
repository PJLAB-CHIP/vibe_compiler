set(WAFER_PYTHON_COMMAND
  ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
  ${Python3_EXECUTABLE} -B)
set(WAFER_IMPORTER_PYTHON_COMMAND
  ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
  ${WAFER_IMPORTER_PYTHON_EXECUTABLE} -B)

execute_process(
  COMMAND ${WAFER_PYTHON_COMMAND} -c "import lit"
  RESULT_VARIABLE WAFER_LIT_IMPORT_RESULT
)
if(NOT WAFER_LIT_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Python package 'lit' is required for FileCheck tests. "
    "Run utils/deps/bootstrap_deps.py --python and configure with that Python, or install requirements-dev.txt.")
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

set(_wafer_default_lit_paths
  ${CMAKE_CURRENT_BINARY_DIR}/Dialect
  ${CMAKE_CURRENT_BINARY_DIR}/Conversion
  ${CMAKE_CURRENT_BINARY_DIR}/Frontend
  ${CMAKE_CURRENT_BINARY_DIR}/Pipelines
  ${CMAKE_CURRENT_BINARY_DIR}/Runtime
  ${CMAKE_CURRENT_BINARY_DIR}/Spmd
  ${CMAKE_CURRENT_BINARY_DIR}/Transforms
  ${CMAKE_CURRENT_BINARY_DIR}/Tools
)

set(_wafer_lit_dependencies
  wafer-compile
  wafer-compile-test
  wafer-opt
  wafer-run
)
if(TARGET wafer-verify-program)
  list(APPEND _wafer_lit_dependencies wafer-verify-program stablehlo-translate)
endif()
if(TARGET wafer-cmodel-qualify-onednn)
  list(APPEND _wafer_lit_dependencies wafer-cmodel-qualify-onednn)
endif()

add_custom_target(wafer-test-tools ALL
  DEPENDS ${_wafer_lit_dependencies})

add_custom_target(check-wafer-lit
  COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
    ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS}
    ${_wafer_default_lit_paths}
  DEPENDS ${_wafer_lit_dependencies}
  USES_TERMINAL
)
unset(_wafer_lit_dependencies)

add_test(NAME wafer-lit
  COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
    ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS}
    ${_wafer_default_lit_paths}
)

add_test(NAME wafer-runtime-adapter-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/wafer_runtime_adapter_test.py
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-run $<TARGET_FILE:wafer-run>
)
set_tests_properties(wafer-runtime-adapter-python PROPERTIES
  LABELS "runtime;no-card;adapter"
  TIMEOUT 30
)
add_test(NAME wafer-physical-tensor-codec-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_CURRENT_SOURCE_DIR}/Board/Cases/wafer_physical_tensor_codec_test.py
)
set_tests_properties(wafer-physical-tensor-codec-python PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Board/Support"
  LABELS "runtime;no-card;layout;calibration;oracle"
  TIMEOUT 30
)

if(WAFER_ENABLE_PYTORCH_XLA_IMPORTER)
  add_test(NAME wafer-pytorch-board-tensor-reference-python
    COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_pytorch_board_common_test.py
  )
  set_tests_properties(
    wafer-pytorch-board-tensor-reference-python PROPERTIES
    LABELS "runtime;no-card;pytorch;reference;fault-injection"
    TIMEOUT 30
  )
  add_test(NAME wafer-pytorch-board-cases-python
    COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
      ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_pytorch_board_cases_test.py
  )
  set_tests_properties(wafer-pytorch-board-cases-python PROPERTIES
    LABELS "runtime;no-card;pytorch;source;huggingface;card-local"
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;WAFER_STABLEHLO_TRANSLATE=$<TARGET_FILE:stablehlo-translate>"
    TIMEOUT 30
  )

  function(wafer_add_pytorch_source_no_card_test
      case_name dtype_name dtype_label optimization_policy)
    set(_wafer_test_name
      "wafer-runtime-pytorch-${case_name}-${dtype_label}-optimization-${optimization_policy}-no-card")
    add_test(NAME ${_wafer_test_name}
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case ${case_name}
        --dtype ${dtype_name}
        --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/pytorch-product/${_wafer_test_name}
        --optimization-policy ${optimization_policy}
        --no-card
    )
    set_tests_properties(${_wafer_test_name} PROPERTIES
      LABELS "runtime;no-card;product;pytorch;source;board-ready;${dtype_label};optimization-${optimization_policy}"
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
      PROCESSORS 16
      TIMEOUT 5400
    )
  endfunction()

  function(wafer_add_pytorch_communication_no_card_test case_name qualification)
    set(_wafer_test_name
      "wafer-runtime-pytorch-${case_name}-communication-no-card")
    add_test(NAME ${_wafer_test_name}
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case ${case_name} --dtype float16 --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile-test>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${_wafer_test_name}
        --optimization-policy none --qualify-communication ${qualification}
        --no-card)
    set_tests_properties(${_wafer_test_name} PROPERTIES
      LABELS "runtime;no-card;pytorch;source;communication;qualification;fp16"
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
      PROCESSORS 16 TIMEOUT 5400)
  endfunction()

  foreach(_wafer_allgather_case IN ITEMS
      allgather-add allgather-add-tail-1025 allgather-add-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_allgather_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_allgather_case} float16 fp16 search)
    wafer_add_pytorch_communication_no_card_test(
      ${_wafer_allgather_case} ring-allgather)
  endforeach()
  unset(_wafer_allgather_case)

  foreach(_wafer_gemm_case IN ITEMS
      single-card-gemm single-card-gemm-tail-1025 single-card-gemm-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_gemm_case} float16 fp16 none)
  endforeach()
  unset(_wafer_gemm_case)

  foreach(_wafer_alltoall_case IN ITEMS
      alltoall-transpose alltoall-transpose-tail-1025 alltoall-transpose-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_alltoall_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_alltoall_case} float16 fp16 search)
    wafer_add_pytorch_communication_no_card_test(
      ${_wafer_alltoall_case} direct-alltoall)
  endforeach()
  unset(_wafer_alltoall_case)

  foreach(_wafer_reduce_scatter_case IN ITEMS
      reduce-scatter-sum reduce-scatter-sum-tail-1025 reduce-scatter-sum-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_reduce_scatter_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_reduce_scatter_case} float16 fp16 search)
    wafer_add_pytorch_communication_no_card_test(
      ${_wafer_reduce_scatter_case} direct-reduce-scatter)
  endforeach()
  unset(_wafer_reduce_scatter_case)

  foreach(_wafer_all_reduce_case IN ITEMS
      all-reduce-sum all-reduce-sum-tail-1025 all-reduce-sum-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_all_reduce_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_all_reduce_case} float16 fp16 search)
    wafer_add_pytorch_communication_no_card_test(
      ${_wafer_all_reduce_case} all-reduce)
  endforeach()
  unset(_wafer_all_reduce_case)

  foreach(_wafer_local_reduce_case IN ITEMS
      local-reduce local-reduce-tail-1025 local-reduce-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_local_reduce_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_local_reduce_case} float16 fp16 search)
  endforeach()
  unset(_wafer_local_reduce_case)

  foreach(_wafer_local_conv_case IN ITEMS
      local-conv local-conv-tail-1025 local-conv-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_local_conv_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_local_conv_case} float16 fp16 search)
  endforeach()
  unset(_wafer_local_conv_case)

  foreach(_wafer_precision_case IN ITEMS
      biased-conv biased-conv-tail-1025 biased-conv-tail-1031
      sigmoid sigmoid-tail-1025 sigmoid-tail-1031)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_precision_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(
      ${_wafer_precision_case} float16 fp16 search)
  endforeach()
  unset(_wafer_precision_case)

  foreach(_wafer_division_case IN ITEMS division division-tail-1025 division-tail-1031)
    wafer_add_pytorch_source_no_card_test(${_wafer_division_case} float32 fp32 none)
    wafer_add_pytorch_source_no_card_test(${_wafer_division_case} float32 fp32 search)
  endforeach()
  unset(_wafer_division_case)

  foreach(_wafer_prefill_case IN ITEMS
      attention-prefill-tail-1025 attention-prefill-tail-1031)
    wafer_add_pytorch_source_no_card_test(${_wafer_prefill_case} float16 fp16 none)
    wafer_add_pytorch_source_no_card_test(${_wafer_prefill_case} float16 fp16 search)
  endforeach()
  unset(_wafer_prefill_case)

  foreach(_wafer_product_case IN ITEMS
      conv-mixed-dag
      attention-prefill
      attention-decode-kv-cache
      llama-2-7b-block)
    foreach(_wafer_product_dtype IN ITEMS
        "float16;fp16"
        "bfloat16;bf16")
      list(GET _wafer_product_dtype 0 _wafer_product_dtype_name)
      list(GET _wafer_product_dtype 1 _wafer_product_dtype_label)
      foreach(_wafer_product_policy IN ITEMS search none)
        wafer_add_pytorch_source_no_card_test(
          ${_wafer_product_case}
          ${_wafer_product_dtype_name}
          ${_wafer_product_dtype_label}
          ${_wafer_product_policy})
      endforeach()
    endforeach()
  endforeach()
  unset(_wafer_product_case)
  unset(_wafer_product_dtype)
  unset(_wafer_product_dtype_name)
  unset(_wafer_product_dtype_label)
  unset(_wafer_product_policy)
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
  if(WAFER_ENABLE_PYTORCH_XLA_IMPORTER)
    add_test(NAME wafer-board-pytorch-single-card-gemm
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case single-card-gemm
        --dtype float16
        --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/board-pytorch-single-card-gemm
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
    set_tests_properties(wafer-board-pytorch-single-card-gemm PROPERTIES
      LABELS "board;hardware;pytorch;source;single-op;gemm"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    )
    add_test(NAME wafer-board-pytorch-attention-prefill-optimization-none
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case attention-prefill
        --dtype float16
        --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-pytorch-attention-prefill-optimization-none
        --optimization-policy none
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
      wafer-board-pytorch-attention-prefill-optimization-none PROPERTIES
      LABELS "board;hardware;pytorch;source;huggingface;attention;prefill;fp16;baseline;optimization-none"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    )
    add_test(NAME wafer-board-pytorch-attention-decode-kv-cache-optimization-none
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case attention-decode-kv-cache
        --dtype float16
        --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-pytorch-attention-decode-kv-cache-optimization-none
        --optimization-policy none
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
      wafer-board-pytorch-attention-decode-kv-cache-optimization-none PROPERTIES
      LABELS "board;hardware;pytorch;source;huggingface;attention;decode;kv-cache;fp16;baseline;optimization-none"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps"
    )
    add_test(NAME wafer-board-pytorch-llama-2-7b-block-optimization-none
      COMMAND ${WAFER_IMPORTER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/PyTorch/wafer_board_pytorch_test.py
        --case llama-2-7b-block
        --dtype float16
        --seed 20260803
        --wafer-compile $<TARGET_FILE:wafer-compile>
        --wafer-run $<TARGET_FILE:wafer-run>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/board-pytorch-llama-2-7b-block-optimization-none
        --optimization-policy none
        --device-id ${WAFER_BOARD_TEST_DEVICE_ID}
        --expected-runtime-version ${WAFER_BOARD_TEST_EXPECTED_RUNTIME_VERSION}
        --expected-device-name ${WAFER_BOARD_TEST_EXPECTED_DEVICE_NAME}
        --expected-pci-bus-id ${WAFER_BOARD_TEST_EXPECTED_PCI_BUS_ID}
        --expected-tile-count ${WAFER_BOARD_TEST_EXPECTED_TILE_COUNT}
        --expected-runtime-library-sha256
          ${_wafer_expected_tx_runtime_library_sha256}
        --completion-timeout-ms 600000
        --repeat 3
    )
    set_tests_properties(
      wafer-board-pytorch-llama-2-7b-block-optimization-none PROPERTIES
      LABELS "board;hardware;heavy;pytorch;source;huggingface;llama;block;fp16;baseline;optimization-none"
      RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"
      SKIP_RETURN_CODE 77
      TIMEOUT 9000
    )
  endif()
  unset(_wafer_board_qualification_variable)
  unset(_wafer_expected_tx_runtime_library_sha256)
  unset(_wafer_expected_tx_runtime_library_sha256_length)
  unset(_wafer_tx_runtime_library_sha256)
endif()

add_subdirectory(Board)

add_test(NAME wafer-numeric-dependency-record-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_numeric_deps.py
)

add_test(NAME wafer-pytorch-xla-runtime-build-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_pytorch_xla_runtime_build.py
)
set_tests_properties(wafer-pytorch-xla-runtime-build-python PROPERTIES
  LABELS "framework-importer;managed-dependencies"
)

add_test(NAME wafer-onednn-dependency-record-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_onednn_deps.py
)
set_tests_properties(wafer-onednn-dependency-record-python PROPERTIES
  LABELS "onednn;managed-dependencies"
)

add_test(NAME wafer-systemc-dependency-record-python
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_systemc_deps.py
)
set_tests_properties(wafer-systemc-dependency-record-python PROPERTIES
  LABELS "systemc-model;managed-dependencies"
)

set_tests_properties(
  wafer-numeric-dependency-record-python
  wafer-pytorch-xla-runtime-build-python
  wafer-onednn-dependency-record-python
  wafer-systemc-dependency-record-python
  PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/utils/deps"
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
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_numeric_cmake_gates.py
    ${_wafer_numeric_cmake_gate_args}
)
set_tests_properties(wafer-numeric-cmake-configuration-gates PROPERTIES
  LABELS "numeric-model;configuration"
  TIMEOUT 180
)

set(_wafer_onednn_cmake_gate_args
  --cmake ${CMAKE_COMMAND}
  --source-dir ${CMAKE_SOURCE_DIR}
  --work-root ${CMAKE_CURRENT_BINARY_DIR}/onednn-cmake-gates
  --generator ${CMAKE_GENERATOR}
  --c-compiler ${CMAKE_C_COMPILER}
  --cxx-compiler ${CMAKE_CXX_COMPILER}
  --llvm-dir ${LLVM_DIR}
  --mlir-dir ${MLIR_DIR}
  --python ${Python3_EXECUTABLE}
)
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND _wafer_onednn_cmake_gate_args
    --generator-platform ${CMAKE_GENERATOR_PLATFORM})
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND _wafer_onednn_cmake_gate_args
    --generator-toolset ${CMAKE_GENERATOR_TOOLSET})
endif()
if(CMAKE_MAKE_PROGRAM)
  list(APPEND _wafer_onednn_cmake_gate_args
    --make-program ${CMAKE_MAKE_PROGRAM})
endif()
if(CMAKE_BUILD_TYPE)
  list(APPEND _wafer_onednn_cmake_gate_args
    --build-type ${CMAKE_BUILD_TYPE})
endif()
if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  list(APPEND _wafer_onednn_cmake_gate_args
    --numeric-root ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}
    --numeric-record ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD})
endif()
add_test(NAME wafer-onednn-cmake-configuration-gates
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_onednn_cmake_gates.py
    ${_wafer_onednn_cmake_gate_args}
)
set_tests_properties(wafer-onednn-cmake-configuration-gates PROPERTIES
  LABELS "onednn;configuration"
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
  COMMAND ${WAFER_PYTHON_COMMAND}
    ${CMAKE_SOURCE_DIR}/test/Tools/test_systemc_cmake_gates.py
    ${_wafer_systemc_cmake_gate_args}
)
set_tests_properties(wafer-systemc-cmake-configuration-gates PROPERTIES
  LABELS "systemc-model;configuration"
  TIMEOUT 240
)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_test(NAME wafer-numeric-dependency-configured-record-python
    COMMAND ${WAFER_PYTHON_COMMAND}
      ${CMAKE_SOURCE_DIR}/utils/checks/check_deps.py
      --numeric-only
      --numeric-root ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT}
      --numeric-record ${WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD}
      --emit-numeric-cmake-snapshot
  )
  set_tests_properties(wafer-numeric-dependency-configured-record-python
    PROPERTIES PASS_REGULAR_EXPRESSION
      "wafer-numeric-model-canonical-snapshot")
endif()

if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  if(NOT CMAKE_READELF OR NOT EXISTS "${CMAKE_READELF}")
    message(FATAL_ERROR
      "feature-on onednn link closure requires the configured readelf tool")
  endif()
  if(NOT CMAKE_NM OR NOT EXISTS "${CMAKE_NM}")
    message(FATAL_ERROR
      "feature-on onednn link closure requires the configured nm tool")
  endif()
  add_test(NAME wafer-onednn-dependency-configured-record-python
    COMMAND ${WAFER_PYTHON_COMMAND}
      ${CMAKE_SOURCE_DIR}/utils/deps/onednn_deps.py
      --root ${WAFER_ONEDNN_DEPS_CANONICAL_ROOT}
      --record ${WAFER_ONEDNN_DEPS_CANONICAL_RECORD}
      --emit-cmake-snapshot
  )
  set_tests_properties(wafer-onednn-dependency-configured-record-python
    PROPERTIES
      PASS_REGULAR_EXPRESSION "wafer-onednn-canonical-snapshot"
      LABELS "onednn;managed-dependencies")
  add_test(NAME wafer-onednn-feature-on-link-closure
    COMMAND ${WAFER_PYTHON_COMMAND}
      ${CMAKE_SOURCE_DIR}/utils/checks/check_onednn_enabled_link_closure.py
      --binary $<TARGET_FILE:wafer-cmodel-qualify-onednn>
      --readelf ${CMAKE_READELF}
      --nm ${CMAKE_NM}
  )
  set_tests_properties(wafer-onednn-feature-on-link-closure PROPERTIES
    LABELS "onednn;link-closure"
  )
  add_test(NAME wafer-onednn-qualification-cli
    COMMAND ${WAFER_PYTHON_COMMAND}
      ${CMAKE_SOURCE_DIR}/test/Tools/test_onednn_qualification_cli.py
      --tool $<TARGET_FILE:wafer-cmodel-qualify-onednn>
      --work-root ${CMAKE_CURRENT_BINARY_DIR}/onednn-qualification-cli
  )
  set_tests_properties(wafer-onednn-qualification-cli PROPERTIES
    LABELS "onednn;qualification"
    TIMEOUT 180
  )
endif()

if(WAFER_ENABLE_SYSTEMC_MODEL)
  add_test(NAME wafer-systemc-dependency-configured-record-python
    COMMAND ${WAFER_PYTHON_COMMAND}
      ${CMAKE_SOURCE_DIR}/utils/deps/systemc_deps.py
      --root ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_ROOT}
      --record ${WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_RECORD}
      --emit-cmake-snapshot
  )
  set_tests_properties(wafer-systemc-dependency-configured-record-python
    PROPERTIES
      PASS_REGULAR_EXPRESSION "wafer-systemc-model-canonical-snapshot"
      LABELS "systemc-model;managed-dependencies")

  if(WAFER_ENABLE_UNIT_TESTS)
    add_test(NAME wafer-runtime-target-model-source-vertical-no-card
      COMMAND ${WAFER_PYTHON_COMMAND}
        ${CMAKE_CURRENT_SOURCE_DIR}/Board/Support/wafer_target_model_source_runner.py
        --wafer-compile $<TARGET_FILE:wafer-compile-test>
        --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/wafer-runtime-target-model-source-vertical-no-card)
    set_tests_properties(wafer-runtime-target-model-source-vertical-no-card
      PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Board/Support;TX8_DEPS_ROOT=${CMAKE_SOURCE_DIR}/third_party/tx8_deps;WAFER_STABLEHLO_TRANSLATE=$<TARGET_FILE:stablehlo-translate>"
        LABELS "runtime;no-card;source;target-model;board-ready"
        PROCESSORS 4
        TIMEOUT 900)
  endif()
endif()
