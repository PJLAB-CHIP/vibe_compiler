# Managed oneDNN identity and imported target for the bulk CModel lane.
# This module is deliberately offline; tools/bootstrap_deps.py is the only
# supported source/build producer.

function(wafer_enable_bulk_model_deps)
  if(NOT WAFER_ENABLE_NUMERIC_MODEL_DEPS)
    message(FATAL_ERROR
      "WAFER_ENABLE_BULK_MODEL_DEPS=ON requires "
      "WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON")
  endif()
  if(NOT IS_ABSOLUTE "${WAFER_BULK_MODEL_DEPS_ROOT}" OR
     NOT EXISTS "${WAFER_BULK_MODEL_DEPS_ROOT}" OR
     NOT IS_DIRECTORY "${WAFER_BULK_MODEL_DEPS_ROOT}")
    message(FATAL_ERROR
      "WAFER_ENABLE_BULK_MODEL_DEPS=ON requires a managed bulk-model root. "
      "Run tools/bootstrap_deps.py --bulk-model-deps or set "
      "WAFER_BULK_MODEL_DEPS_ROOT explicitly.")
  endif()
  if(NOT EXISTS "${WAFER_BULK_MODEL_DEPS_RECORD}" OR
     IS_DIRECTORY "${WAFER_BULK_MODEL_DEPS_RECORD}")
    message(FATAL_ERROR
      "WAFER_ENABLE_BULK_MODEL_DEPS=ON requires the completed managed "
      "bulk-model dependency record. Run "
      "tools/bootstrap_deps.py --bulk-model-deps.")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/bulk_deps.py"
            --root "${WAFER_BULK_MODEL_DEPS_ROOT}"
            --record "${WAFER_BULK_MODEL_DEPS_RECORD}"
            --emit-cmake-snapshot
    RESULT_VARIABLE _wafer_bulk_check_result
    OUTPUT_VARIABLE _WAFER_BULK_SNAPSHOT_JSON
    ERROR_VARIABLE _wafer_bulk_check_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _wafer_bulk_check_result EQUAL 0)
    message(FATAL_ERROR
      "Managed bulk-model dependency identity/conformance check failed:\n"
      "${_wafer_bulk_check_error}")
  endif()

  string(JSON _wafer_bulk_kind GET
    "${_WAFER_BULK_SNAPSHOT_JSON}" kind)
  if(NOT _wafer_bulk_kind STREQUAL "wafer-bulk-model-canonical-snapshot")
    message(FATAL_ERROR "Managed bulk-model canonical snapshot kind mismatch")
  endif()
  string(JSON _wafer_bulk_root GET "${_WAFER_BULK_SNAPSHOT_JSON}" root)
  string(JSON _wafer_bulk_record GET "${_WAFER_BULK_SNAPSHOT_JSON}" record)
  string(JSON _wafer_bulk_record_sha256 GET
    "${_WAFER_BULK_SNAPSHOT_JSON}" record_sha256)
  string(JSON _wafer_bulk_version GET "${_WAFER_BULK_SNAPSHOT_JSON}" version)
  string(JSON _wafer_bulk_commit GET "${_WAFER_BULK_SNAPSHOT_JSON}" commit)
  string(JSON _wafer_bulk_library GET
    "${_WAFER_BULK_SNAPSHOT_JSON}" artifacts onednn-library path)
  string(JSON _wafer_bulk_library_sha256 GET
    "${_WAFER_BULK_SNAPSHOT_JSON}" artifacts onednn-library sha256)
  string(JSON _wafer_bulk_cxx_header GET
    "${_WAFER_BULK_SNAPSHOT_JSON}" artifacts onednn-cxx-header path)
  get_filename_component(_wafer_bulk_dnnl_dir
    "${_wafer_bulk_cxx_header}" DIRECTORY)
  get_filename_component(_wafer_bulk_oneapi_dir
    "${_wafer_bulk_dnnl_dir}" DIRECTORY)
  get_filename_component(_wafer_bulk_include_dir
    "${_wafer_bulk_oneapi_dir}" DIRECTORY)

  if(TARGET WaferBulk::oneDNN)
    message(FATAL_ERROR "managed bulk-model imported target already exists")
  endif()
  find_package(Threads REQUIRED)
  add_library(WaferBulk::oneDNN STATIC IMPORTED GLOBAL)
  set_target_properties(WaferBulk::oneDNN PROPERTIES
    IMPORTED_LOCATION "${_wafer_bulk_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wafer_bulk_include_dir}"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")

  set(_wafer_bulk_snapshot_dir "${CMAKE_BINARY_DIR}/generated/bulk-model")
  file(MAKE_DIRECTORY "${_wafer_bulk_snapshot_dir}")
  set(_wafer_bulk_snapshot_path
    "${_wafer_bulk_snapshot_dir}/bulk-model-deps.snapshot.json")
  file(WRITE "${_wafer_bulk_snapshot_path}" "${_WAFER_BULK_SNAPSHOT_JSON}\n")
  set(WAFER_BULK_MODEL_DEPS_CANONICAL_ROOT "${_wafer_bulk_root}"
    CACHE INTERNAL "Canonical managed bulk-model dependency root")
  set(WAFER_BULK_MODEL_DEPS_CANONICAL_RECORD "${_wafer_bulk_record}"
    CACHE INTERNAL "Canonical managed bulk-model dependency record")
  set(WAFER_BULK_MODEL_DEPS_RECORD_SHA256 "${_wafer_bulk_record_sha256}"
    CACHE INTERNAL "Managed bulk-model dependency record identity")
  set(WAFER_BULK_MODEL_ONEDNN_VERSION "${_wafer_bulk_version}"
    CACHE INTERNAL "Managed oneDNN semantic version")
  set(WAFER_BULK_MODEL_ONEDNN_COMMIT "${_wafer_bulk_commit}"
    CACHE INTERNAL "Managed oneDNN commit")
  set(WAFER_BULK_MODEL_ONEDNN_LIBRARY_SHA256 "${_wafer_bulk_library_sha256}"
    CACHE INTERNAL "Managed oneDNN static library identity")
  set(WAFER_BULK_MODEL_DEPS_SNAPSHOT "${_wafer_bulk_snapshot_path}"
    CACHE INTERNAL "Managed bulk-model canonical snapshot")
  message(STATUS
    "Enabled managed bulk-model dependency: ${_wafer_bulk_record_sha256}")
endfunction()
