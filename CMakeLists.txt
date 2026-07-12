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

add_subdirectory(include/Wafer/IR)
add_subdirectory(include/Wafer/Transforms)
add_subdirectory(lib/Wafer)
add_subdirectory(tools/wafer-opt)
add_subdirectory(tools/wafer-compile)
add_subdirectory(tools/wafer-compile-stablehlo)
add_subdirectory(tools/wafer-run)

enable_testing()
add_subdirectory(test)

if(WAFER_ENABLE_UNIT_TESTS)
  add_subdirectory(unittests)
endif()

if(TARGET WaferUnitTests)
  add_custom_target(check-wafer DEPENDS check-wafer-lit check-wafer-unit)
else()
  add_custom_target(check-wafer DEPENDS check-wafer-lit)
endif()

if(TARGET wafer-shardy-cmake-gate)
  add_dependencies(check-wafer wafer-shardy-cmake-gate)
endif()
