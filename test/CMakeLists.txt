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

add_custom_target(check-wafer-lit
  COMMAND ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS wafer-compile wafer-compile-test wafer-compile-stablehlo wafer-opt wafer-run
  USES_TERMINAL
)

add_test(NAME wafer-lit
  COMMAND ${WAFER_LIT_EXECUTABLE} ${WAFER_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME wafer-runtime-adapter-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/wafer_runtime_adapter_test.py
    --repo-root ${CMAKE_SOURCE_DIR}
    --wafer-run $<TARGET_FILE:wafer-run>
)

add_test(NAME wafer-numeric-dependency-record-python
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/test_numeric_deps.py
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
