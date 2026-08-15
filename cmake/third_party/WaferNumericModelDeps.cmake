# Managed functional-numeric dependency identity and imported targets.
#
# This module is intentionally offline.  The only producer is
# tools/bootstrap_deps.py --numeric-model-deps; configure merely validates the
# complete record and imports the exact recorded files.

function(_wafer_numeric_file_path output_variable file_name)
  string(JSON _wafer_numeric_canonical_path GET
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}" artifacts "${file_name}" path)
  if(NOT IS_ABSOLUTE "${_wafer_numeric_canonical_path}")
    message(FATAL_ERROR
      "Validated numeric file path is not canonical: ${file_name}")
  endif()
  set(${output_variable} "${_wafer_numeric_canonical_path}" PARENT_SCOPE)
endfunction()

function(wafer_enable_numeric_model_deps)
  if(NOT IS_ABSOLUTE "${WAFER_NUMERIC_MODEL_DEPS_ROOT}")
    message(FATAL_ERROR
      "WAFER_NUMERIC_MODEL_DEPS_ROOT must be an absolute managed path")
  endif()
  if(NOT EXISTS "${WAFER_NUMERIC_MODEL_DEPS_ROOT}" OR
     NOT IS_DIRECTORY "${WAFER_NUMERIC_MODEL_DEPS_ROOT}")
    message(FATAL_ERROR
      "WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON requires a managed dependency root. "
      "Run tools/bootstrap_deps.py --numeric-model-deps or set "
      "WAFER_NUMERIC_MODEL_DEPS_ROOT explicitly.")
  endif()
  if(NOT EXISTS "${WAFER_NUMERIC_MODEL_DEPS_RECORD}" OR
     IS_DIRECTORY "${WAFER_NUMERIC_MODEL_DEPS_RECORD}")
    message(FATAL_ERROR
      "WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON requires the completed managed "
      "numeric-model conformance record. Run "
      "tools/bootstrap_deps.py --numeric-model-deps.")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/check_deps.py"
            --numeric-only
            --numeric-root "${WAFER_NUMERIC_MODEL_DEPS_ROOT}"
            --numeric-record "${WAFER_NUMERIC_MODEL_DEPS_RECORD}"
            --emit-numeric-cmake-snapshot
    RESULT_VARIABLE _wafer_numeric_check_result
    OUTPUT_VARIABLE _WAFER_NUMERIC_SNAPSHOT_JSON
    ERROR_VARIABLE _wafer_numeric_check_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _wafer_numeric_check_result EQUAL 0)
    message(FATAL_ERROR
      "Managed numeric-model dependency identity/conformance check failed:\n"
      "${_wafer_numeric_check_error}")
  endif()

  string(JSON _wafer_numeric_snapshot_kind GET
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}" kind)
  if(NOT _wafer_numeric_snapshot_kind STREQUAL
       "wafer-numeric-model-canonical-snapshot")
    message(FATAL_ERROR "Managed numeric-model canonical snapshot kind mismatch")
  endif()
  string(JSON _wafer_numeric_record_sha256 GET
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}" record_sha256)
  string(JSON _wafer_numeric_canonical_root GET
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}" root)
  string(JSON _wafer_numeric_canonical_record GET
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}" record)
  set(_wafer_numeric_snapshot_dir
    "${CMAKE_BINARY_DIR}/generated/numeric-model")
  file(MAKE_DIRECTORY "${_wafer_numeric_snapshot_dir}")
  set(_wafer_numeric_snapshot_path
    "${_wafer_numeric_snapshot_dir}/numeric-model-deps.snapshot.json")
  file(WRITE "${_wafer_numeric_snapshot_path}"
    "${_WAFER_NUMERIC_SNAPSHOT_JSON}\n")
  set(WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256
    "${_wafer_numeric_record_sha256}" CACHE INTERNAL
    "Validated numeric-model dependency conformance identity")
  set(WAFER_NUMERIC_MODEL_DEPS_CANONICAL_ROOT
    "${_wafer_numeric_canonical_root}" CACHE INTERNAL
    "Canonical managed numeric-model dependency root")
  set(WAFER_NUMERIC_MODEL_DEPS_CANONICAL_RECORD
    "${_wafer_numeric_canonical_record}" CACHE INTERNAL
    "Canonical managed numeric-model dependency conformance record")
  set(WAFER_NUMERIC_MODEL_DEPS_SNAPSHOT
    "${_wafer_numeric_snapshot_path}" CACHE INTERNAL
    "Validated canonical numeric-model dependency snapshot")

  _wafer_numeric_file_path(_wafer_softfloat_library softfloat)
  _wafer_numeric_file_path(_wafer_softfloat_header softfloat-header)
  _wafer_numeric_file_path(_wafer_gmp_library gmp)
  _wafer_numeric_file_path(_wafer_gmp_header gmp-header)
  _wafer_numeric_file_path(_wafer_mpfr_library mpfr)
  _wafer_numeric_file_path(_wafer_mpfr_header mpfr-header)
  _wafer_numeric_file_path(_wafer_testsoftfloat testsoftfloat)

  get_filename_component(_wafer_softfloat_include
    "${_wafer_softfloat_header}" DIRECTORY)
  get_filename_component(_wafer_gmp_include "${_wafer_gmp_header}" DIRECTORY)
  get_filename_component(_wafer_mpfr_include "${_wafer_mpfr_header}" DIRECTORY)
  get_filename_component(_wafer_gmp_library_dir "${_wafer_gmp_library}" DIRECTORY)
  get_filename_component(_wafer_mpfr_library_dir "${_wafer_mpfr_library}" DIRECTORY)

  if(TARGET WaferNumeric::SoftFloat OR TARGET WaferNumeric::GMP OR
     TARGET WaferNumeric::MPFR OR TARGET WaferNumeric::TestFloat)
    message(FATAL_ERROR "managed numeric-model imported target already exists")
  endif()

  add_library(WaferNumeric::SoftFloat STATIC IMPORTED GLOBAL)
  set_target_properties(WaferNumeric::SoftFloat PROPERTIES
    IMPORTED_LOCATION "${_wafer_softfloat_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wafer_softfloat_include}"
    INTERFACE_COMPILE_DEFINITIONS
      "$<$<COMPILE_LANGUAGE:C>:THREAD_LOCAL=_Thread_local>;$<$<COMPILE_LANGUAGE:CXX>:THREAD_LOCAL=thread_local>")

  add_library(WaferNumeric::GMP SHARED IMPORTED GLOBAL)
  set_target_properties(WaferNumeric::GMP PROPERTIES
    IMPORTED_LOCATION "${_wafer_gmp_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wafer_gmp_include}")

  add_library(WaferNumeric::MPFR SHARED IMPORTED GLOBAL)
  set_target_properties(WaferNumeric::MPFR PROPERTIES
    IMPORTED_LOCATION "${_wafer_mpfr_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wafer_mpfr_include}"
    INTERFACE_LINK_LIBRARIES WaferNumeric::GMP)

  add_executable(WaferNumeric::TestFloat IMPORTED GLOBAL)
  set_target_properties(WaferNumeric::TestFloat PROPERTIES
    IMPORTED_LOCATION "${_wafer_testsoftfloat}")

  set(WAFER_NUMERIC_MODEL_GMP_LIBRARY_DIR
    "${_wafer_gmp_library_dir}" CACHE INTERNAL
    "Managed numeric-model GMP runtime directory")
  set(WAFER_NUMERIC_MODEL_MPFR_LIBRARY_DIR
    "${_wafer_mpfr_library_dir}" CACHE INTERNAL
    "Managed numeric-model MPFR runtime directory")
  message(STATUS
    "Enabled managed numeric-model dependencies: ${_wafer_numeric_record_sha256}")
endfunction()
