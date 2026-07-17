cmake_minimum_required(VERSION 3.24)

project(WaferCompiler LANGUAGES CXX C)

include(cmake/third_party/WaferDependencyVersions.cmake)
include(cmake/third_party/WaferThirdParty.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WAFER_ENABLE_UNIT_TESTS "Build Wafer C++ unit tests" ON)
set(WAFER_XLA_SPMD_PARTITIONER_HELPER "" CACHE FILEPATH
  "Path to the pinned-XLA SPMD partitioner helper used by wafer-compile")
if(WAFER_XLA_SPMD_PARTITIONER_HELPER)
  if(NOT IS_ABSOLUTE "${WAFER_XLA_SPMD_PARTITIONER_HELPER}")
    message(FATAL_ERROR
      "WAFER_XLA_SPMD_PARTITIONER_HELPER must be an absolute path")
  endif()
  if(NOT EXISTS "${WAFER_XLA_SPMD_PARTITIONER_HELPER}" OR
     IS_DIRECTORY "${WAFER_XLA_SPMD_PARTITIONER_HELPER}")
    message(FATAL_ERROR
      "WAFER_XLA_SPMD_PARTITIONER_HELPER must name an existing regular executable")
  endif()
  get_filename_component(_wafer_spmd_helper_name
    "${WAFER_XLA_SPMD_PARTITIONER_HELPER}" NAME)
  get_filename_component(_wafer_spmd_helper_dir
    "${WAFER_XLA_SPMD_PARTITIONER_HELPER}" DIRECTORY)
  find_program(_wafer_spmd_helper_executable
    NAMES "${_wafer_spmd_helper_name}"
    PATHS "${_wafer_spmd_helper_dir}"
    NO_DEFAULT_PATH
    NO_CACHE)
  if(NOT _wafer_spmd_helper_executable)
    message(FATAL_ERROR
      "WAFER_XLA_SPMD_PARTITIONER_HELPER is not executable")
  endif()
  file(REAL_PATH "${WAFER_XLA_SPMD_PARTITIONER_HELPER}"
    _wafer_spmd_helper_real_path)
  set(WAFER_XLA_SPMD_PARTITIONER_HELPER
    "${_wafer_spmd_helper_real_path}" CACHE FILEPATH
    "Path to the pinned-XLA SPMD partitioner helper used by wafer-compile" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

wafer_add_minimalloc()

add_subdirectory(include/Wafer/IR)
add_subdirectory(include/Wafer/Transforms)
add_subdirectory(lib/Wafer)
add_subdirectory(tools/wafer-opt)
add_subdirectory(tools/wafer-compile)
add_subdirectory(tools/wafer-compile-stablehlo)
add_subdirectory(tools/wafer-run)
if(WAFER_ENABLE_BULK_MODEL_DEPS)
  add_subdirectory(tools/wafer-cmodel-qualify-bulk)
endif()

enable_testing()
add_subdirectory(test)

if(WAFER_ENABLE_UNIT_TESTS)
  add_subdirectory(unittests)
endif()

set(_wafer_check_targets check-wafer-lit)
foreach(_wafer_optional_check_target IN ITEMS
    check-wafer-unit
    check-wafer-numeric-model-unit
    check-wafer-bulk-model-unit
    check-wafer-systemc-model)
  if(TARGET ${_wafer_optional_check_target})
    list(APPEND _wafer_check_targets ${_wafer_optional_check_target})
  endif()
endforeach()
add_custom_target(check-wafer DEPENDS ${_wafer_check_targets})
unset(_wafer_check_targets)
unset(_wafer_optional_check_target)

if(TARGET wafer-shardy-cmake-gate)
  add_dependencies(check-wafer wafer-shardy-cmake-gate)
endif()
