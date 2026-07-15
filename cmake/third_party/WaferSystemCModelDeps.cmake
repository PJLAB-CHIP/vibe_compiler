# Managed Accellera SystemC identity and official imported target for the
# functional-event model. This module is offline; tools/bootstrap_deps.py is
# the only supported source/build producer.

function(wafer_enable_systemc_model_deps)
  if(NOT WAFER_ENABLE_NUMERIC_MODEL_DEPS)
    message(FATAL_ERROR
      "WAFER_ENABLE_SYSTEMC_MODEL=ON requires "
      "WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON")
  endif()
  if(NOT IS_ABSOLUTE "${WAFER_SYSTEMC_MODEL_DEPS_ROOT}" OR
     NOT EXISTS "${WAFER_SYSTEMC_MODEL_DEPS_ROOT}" OR
     NOT IS_DIRECTORY "${WAFER_SYSTEMC_MODEL_DEPS_ROOT}")
    message(FATAL_ERROR
      "WAFER_ENABLE_SYSTEMC_MODEL=ON requires a managed SystemC-model root. "
      "Run tools/bootstrap_deps.py --systemc-model-deps or set "
      "WAFER_SYSTEMC_MODEL_DEPS_ROOT explicitly.")
  endif()
  if(NOT EXISTS "${WAFER_SYSTEMC_MODEL_DEPS_RECORD}" OR
     IS_DIRECTORY "${WAFER_SYSTEMC_MODEL_DEPS_RECORD}")
    message(FATAL_ERROR
      "WAFER_ENABLE_SYSTEMC_MODEL=ON requires the completed managed "
      "SystemC-model dependency record. Run "
      "tools/bootstrap_deps.py --systemc-model-deps.")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/systemc_deps.py"
            --root "${WAFER_SYSTEMC_MODEL_DEPS_ROOT}"
            --record "${WAFER_SYSTEMC_MODEL_DEPS_RECORD}"
            --emit-cmake-snapshot
    RESULT_VARIABLE _wafer_systemc_check_result
    OUTPUT_VARIABLE _WAFER_SYSTEMC_SNAPSHOT_JSON
    ERROR_VARIABLE _wafer_systemc_check_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _wafer_systemc_check_result EQUAL 0)
    message(FATAL_ERROR
      "Managed SystemC-model dependency identity/conformance check failed:\n"
      "${_wafer_systemc_check_error}")
  endif()

  string(JSON _wafer_systemc_schema GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" schema_version)
  string(JSON _wafer_systemc_kind GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" kind)
  if(NOT _wafer_systemc_schema EQUAL 1 OR
     NOT _wafer_systemc_kind STREQUAL
         "wafer-systemc-model-canonical-snapshot")
    message(FATAL_ERROR "Managed SystemC canonical snapshot schema mismatch")
  endif()
  string(JSON _wafer_systemc_root GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" root)
  string(JSON _wafer_systemc_record GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" record)
  string(JSON _wafer_systemc_record_sha256 GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" record_sha256)
  string(JSON _wafer_systemc_version GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" version)
  string(JSON _wafer_systemc_commit GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" commit)
  string(JSON _wafer_systemc_library GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" artifacts systemc-library path)
  string(JSON _wafer_systemc_library_sha256 GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" artifacts systemc-library sha256)
  string(JSON _wafer_systemc_config GET
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}" artifacts cmake-config path)
  get_filename_component(_wafer_systemc_config_dir
    "${_wafer_systemc_config}" DIRECTORY)

  if(TARGET SystemC::systemc)
    message(FATAL_ERROR
      "SystemC::systemc already exists before managed SystemC discovery")
  endif()
  # find_package caches <Package>_DIR.  A managed-root switch in an existing
  # build directory must not silently keep the package exported by the old
  # validated root.
  set(SystemCLanguage_DIR "${_wafer_systemc_config_dir}" CACHE PATH
    "Validated managed SystemC package directory" FORCE)
  find_package(SystemCLanguage 3.0.2 CONFIG REQUIRED
    PATHS "${_wafer_systemc_config_dir}"
    NO_DEFAULT_PATH)
  if(NOT TARGET SystemC::systemc)
    message(FATAL_ERROR
      "Managed SystemC package did not export SystemC::systemc")
  endif()
  if(NOT SystemCLanguage_VERSION MATCHES "^3\\.0\\.2($|\\.)")
    message(FATAL_ERROR
      "Managed SystemC package reports unexpected version "
      "${SystemCLanguage_VERSION}")
  endif()
  get_target_property(_wafer_systemc_imported_location
    SystemC::systemc IMPORTED_LOCATION_RELEASE)
  if(NOT _wafer_systemc_imported_location)
    get_target_property(_wafer_systemc_imported_location
      SystemC::systemc IMPORTED_LOCATION)
  endif()
  if(NOT _wafer_systemc_imported_location)
    message(FATAL_ERROR
      "Managed SystemC::systemc has no imported library location")
  endif()
  file(REAL_PATH "${_wafer_systemc_imported_location}"
    _wafer_systemc_imported_real)
  file(REAL_PATH "${_wafer_systemc_library}"
    _wafer_systemc_recorded_real)
  if(NOT _wafer_systemc_imported_real STREQUAL _wafer_systemc_recorded_real)
    message(FATAL_ERROR
      "Managed SystemC package target does not reference the recorded library")
  endif()

  set(_wafer_systemc_snapshot_dir
    "${CMAKE_BINARY_DIR}/generated/systemc-model")
  file(MAKE_DIRECTORY "${_wafer_systemc_snapshot_dir}")
  set(_wafer_systemc_snapshot_path
    "${_wafer_systemc_snapshot_dir}/systemc-model-deps.snapshot.json")
  file(WRITE "${_wafer_systemc_snapshot_path}"
    "${_WAFER_SYSTEMC_SNAPSHOT_JSON}\n")
  set(WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_ROOT "${_wafer_systemc_root}"
    CACHE INTERNAL "Canonical managed SystemC-model dependency root")
  set(WAFER_SYSTEMC_MODEL_DEPS_CANONICAL_RECORD "${_wafer_systemc_record}"
    CACHE INTERNAL "Canonical managed SystemC-model dependency record")
  set(WAFER_SYSTEMC_MODEL_DEPS_RECORD_SHA256
    "${_wafer_systemc_record_sha256}"
    CACHE INTERNAL "Managed SystemC-model dependency record identity")
  set(WAFER_SYSTEMC_MODEL_VERSION "${_wafer_systemc_version}"
    CACHE INTERNAL "Managed SystemC semantic version")
  set(WAFER_SYSTEMC_MODEL_COMMIT "${_wafer_systemc_commit}"
    CACHE INTERNAL "Managed SystemC source commit")
  set(WAFER_SYSTEMC_MODEL_LIBRARY_SHA256
    "${_wafer_systemc_library_sha256}"
    CACHE INTERNAL "Managed SystemC static library identity")
  set(WAFER_SYSTEMC_MODEL_DEPS_SNAPSHOT "${_wafer_systemc_snapshot_path}"
    CACHE INTERNAL "Managed SystemC-model canonical snapshot")
  message(STATUS
    "Enabled managed SystemC-model dependency: "
    "${_wafer_systemc_record_sha256}")
endfunction()
