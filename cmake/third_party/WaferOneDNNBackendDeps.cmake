# Managed oneDNN identity and imported target for WaferOneDNNBackend.
# This module is deliberately offline; tools/bootstrap_deps.py is the only
# supported source/build producer.

function(wafer_enable_onednn_backend)
  if(NOT WAFER_ENABLE_NUMERIC_MODEL_DEPS)
    message(FATAL_ERROR
      "WAFER_ENABLE_TARGET_NUMERIC_BACKEND=ON requires "
      "WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON")
  endif()
  if(NOT IS_ABSOLUTE "${WAFER_ONEDNN_DEPS_ROOT}" OR
     NOT EXISTS "${WAFER_ONEDNN_DEPS_ROOT}" OR
     NOT IS_DIRECTORY "${WAFER_ONEDNN_DEPS_ROOT}")
    message(FATAL_ERROR
      "WAFER_ENABLE_TARGET_NUMERIC_BACKEND=ON requires a managed oneDNN root. "
      "Run tools/bootstrap_deps.py --onednn-deps or set "
      "WAFER_ONEDNN_DEPS_ROOT explicitly.")
  endif()
  if(NOT EXISTS "${WAFER_ONEDNN_DEPS_RECORD}" OR
     IS_DIRECTORY "${WAFER_ONEDNN_DEPS_RECORD}")
    message(FATAL_ERROR
      "WAFER_ENABLE_TARGET_NUMERIC_BACKEND=ON requires the completed managed "
      "oneDNN dependency record. Run "
      "tools/bootstrap_deps.py --onednn-deps.")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/onednn_deps.py"
            --root "${WAFER_ONEDNN_DEPS_ROOT}"
            --record "${WAFER_ONEDNN_DEPS_RECORD}"
            --emit-cmake-snapshot
    RESULT_VARIABLE _wafer_onednn_check_result
    OUTPUT_VARIABLE _WAFER_ONEDNN_SNAPSHOT_JSON
    ERROR_VARIABLE _wafer_onednn_check_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _wafer_onednn_check_result EQUAL 0)
    message(FATAL_ERROR
      "Managed oneDNN dependency identity/conformance check failed:\n"
      "${_wafer_onednn_check_error}")
  endif()

  string(JSON _wafer_onednn_kind GET
    "${_WAFER_ONEDNN_SNAPSHOT_JSON}" kind)
  if(NOT _wafer_onednn_kind STREQUAL "wafer-onednn-canonical-snapshot")
    message(FATAL_ERROR "Managed oneDNN canonical snapshot kind mismatch")
  endif()
  string(JSON _wafer_onednn_root GET "${_WAFER_ONEDNN_SNAPSHOT_JSON}" root)
  string(JSON _wafer_onednn_record GET "${_WAFER_ONEDNN_SNAPSHOT_JSON}" record)
  string(JSON _wafer_onednn_record_sha256 GET
    "${_WAFER_ONEDNN_SNAPSHOT_JSON}" record_sha256)
  string(JSON _wafer_onednn_version GET "${_WAFER_ONEDNN_SNAPSHOT_JSON}" version)
  string(JSON _wafer_onednn_commit GET "${_WAFER_ONEDNN_SNAPSHOT_JSON}" commit)
  string(JSON _wafer_onednn_library GET
    "${_WAFER_ONEDNN_SNAPSHOT_JSON}" artifacts onednn-library path)
  string(JSON _wafer_onednn_library_sha256 GET
    "${_WAFER_ONEDNN_SNAPSHOT_JSON}" artifacts onednn-library sha256)
  string(JSON _wafer_onednn_cxx_header GET
    "${_WAFER_ONEDNN_SNAPSHOT_JSON}" artifacts onednn-cxx-header path)
  get_filename_component(_wafer_onednn_dnnl_dir
    "${_wafer_onednn_cxx_header}" DIRECTORY)
  get_filename_component(_wafer_onednn_oneapi_dir
    "${_wafer_onednn_dnnl_dir}" DIRECTORY)
  get_filename_component(_wafer_onednn_include_dir
    "${_wafer_onednn_oneapi_dir}" DIRECTORY)

  if(TARGET WaferOneDNN::oneDNN)
    message(FATAL_ERROR "managed oneDNN imported target already exists")
  endif()
  find_package(Threads REQUIRED)
  add_library(WaferOneDNN::oneDNN STATIC IMPORTED GLOBAL)
  set_target_properties(WaferOneDNN::oneDNN PROPERTIES
    IMPORTED_LOCATION "${_wafer_onednn_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wafer_onednn_include_dir}"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")

  set(_wafer_onednn_snapshot_dir "${CMAKE_BINARY_DIR}/generated/onednn")
  file(MAKE_DIRECTORY "${_wafer_onednn_snapshot_dir}")
  set(_wafer_onednn_snapshot_path
    "${_wafer_onednn_snapshot_dir}/onednn-deps.snapshot.json")
  file(WRITE "${_wafer_onednn_snapshot_path}" "${_WAFER_ONEDNN_SNAPSHOT_JSON}\n")
  set(WAFER_ONEDNN_DEPS_CANONICAL_ROOT "${_wafer_onednn_root}"
    CACHE INTERNAL "Canonical managed oneDNN dependency root")
  set(WAFER_ONEDNN_DEPS_CANONICAL_RECORD "${_wafer_onednn_record}"
    CACHE INTERNAL "Canonical managed oneDNN dependency record")
  set(WAFER_ONEDNN_DEPS_RECORD_SHA256 "${_wafer_onednn_record_sha256}"
    CACHE INTERNAL "Managed oneDNN dependency record identity")
  set(WAFER_ONEDNN_BACKEND_VERSION "${_wafer_onednn_version}"
    CACHE INTERNAL "Managed oneDNN semantic version")
  set(WAFER_ONEDNN_BACKEND_COMMIT "${_wafer_onednn_commit}"
    CACHE INTERNAL "Managed oneDNN commit")
  set(WAFER_ONEDNN_BACKEND_LIBRARY_SHA256 "${_wafer_onednn_library_sha256}"
    CACHE INTERNAL "Managed oneDNN static library identity")
  set(WAFER_ONEDNN_DEPS_SNAPSHOT "${_wafer_onednn_snapshot_path}"
    CACHE INTERNAL "Managed oneDNN canonical snapshot")
  message(STATUS
    "Enabled managed oneDNN dependency: ${_wafer_onednn_record_sha256}")
endfunction()
