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
  unset(_wafer_board_qualification_variable)
  unset(_wafer_expected_tx_runtime_library_sha256)
  unset(_wafer_expected_tx_runtime_library_sha256_length)
  unset(_wafer_tx_runtime_library_sha256)
endif()

add_test(NAME wafer-numeric-dependency-record-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_numeric_deps.py
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
