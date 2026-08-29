cmake_minimum_required(VERSION 3.24)

project(WaferCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/third_party/WaferDependencyVersions.cmake)
include(cmake/third_party/WaferThirdParty.cmake)

option(WAFER_ENABLE_UNIT_TESTS "Build Wafer C++ unit tests" ON)
option(WAFER_ENABLE_BOARD_RUNTIME
  "Build the TX board RuntimeProvider capability" OFF)
option(WAFER_ENABLE_BOARD_TEST_EXECUTION
  "Register explicitly qualified tests that execute TX hardware" OFF)
option(WAFER_ENABLE_PRODUCT_INSTALL
  "Register Compiler and Runtime install components" OFF)
option(WAFER_ENABLE_WARNINGS_AS_ERRORS
  "Treat diagnostics from Wafer-owned targets as errors" OFF)
if(WAFER_ENABLE_WARNINGS_AS_ERRORS)
  set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-overloaded-virtual>)
  endif()
endif()
if(WAFER_ENABLE_BOARD_TEST_EXECUTION AND NOT WAFER_ENABLE_BOARD_RUNTIME)
  message(FATAL_ERROR
    "WAFER_ENABLE_BOARD_TEST_EXECUTION=ON requires "
    "WAFER_ENABLE_BOARD_RUNTIME=ON")
endif()
if(WAFER_ENABLE_BOARD_RUNTIME)
  if(NOT IS_ABSOLUTE "${WAFER_TX_RUNTIME_ROOT}" OR
     NOT EXISTS "${WAFER_TX_RUNTIME_ROOT}" OR
     NOT IS_DIRECTORY "${WAFER_TX_RUNTIME_ROOT}")
    message(FATAL_ERROR
      "WAFER_ENABLE_BOARD_RUNTIME=ON requires an existing absolute "
      "WAFER_TX_RUNTIME_ROOT")
  endif()
  file(REAL_PATH "${WAFER_TX_RUNTIME_ROOT}" WAFER_TX_RUNTIME_ROOT_REAL_PATH)
  unset(WAFER_TX_RUNTIME_INCLUDE_DIR CACHE)
  unset(WAFER_TX_RUNTIME_INCLUDE_DIR)
  unset(WAFER_TX_RUNTIME_LIBRARY CACHE)
  unset(WAFER_TX_RUNTIME_LIBRARY)
  find_path(WAFER_TX_RUNTIME_INCLUDE_DIR
    NAMES tx_runtime.h
    PATHS "${WAFER_TX_RUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
    NO_CACHE
    REQUIRED)
  find_library(WAFER_TX_RUNTIME_LIBRARY
    NAMES hpgr
    PATHS "${WAFER_TX_RUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    NO_CACHE
    REQUIRED)
  file(REAL_PATH "${WAFER_TX_RUNTIME_INCLUDE_DIR}"
    WAFER_TX_RUNTIME_INCLUDE_DIR_REAL_PATH)
  file(REAL_PATH "${WAFER_TX_RUNTIME_LIBRARY}"
    WAFER_TX_RUNTIME_LIBRARY_REAL_PATH)
  cmake_path(IS_PREFIX WAFER_TX_RUNTIME_ROOT_REAL_PATH
    "${WAFER_TX_RUNTIME_INCLUDE_DIR_REAL_PATH}" NORMALIZE
    WAFER_TX_RUNTIME_INCLUDE_IS_CONTAINED)
  cmake_path(IS_PREFIX WAFER_TX_RUNTIME_ROOT_REAL_PATH
    "${WAFER_TX_RUNTIME_LIBRARY_REAL_PATH}" NORMALIZE
    WAFER_TX_RUNTIME_LIBRARY_IS_CONTAINED)
  if(NOT WAFER_TX_RUNTIME_INCLUDE_IS_CONTAINED OR
     NOT WAFER_TX_RUNTIME_LIBRARY_IS_CONTAINED)
    message(FATAL_ERROR
      "TX runtime header/library resolved outside WAFER_TX_RUNTIME_ROOT")
  endif()
  set(WAFER_TX_RUNTIME_INCLUDE_DIR
    "${WAFER_TX_RUNTIME_INCLUDE_DIR_REAL_PATH}")

  include(CMakePushCheckState)
  include(CheckCXXSourceCompiles)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${WAFER_TX_RUNTIME_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES "${WAFER_TX_RUNTIME_LIBRARY_REAL_PATH}")
  unset(WAFER_TX_RUNTIME_HAS_REQUIRED_SYMBOLS CACHE)
  check_cxx_source_compiles([[
    #include "tx_runtime.h"
    int main() {
      volatile auto p0 = &txGetDeviceCount;
      volatile auto p1 = &txSetDevice;
      volatile auto p2 = &txGetDeviceProperty;
      volatile auto p3 = &txGetDeviceAllTileInfo;
      volatile auto p4 = &txMemGetInfo;
      volatile auto p5 = &txRuntimeGetVersion;
      volatile auto p6 = &txGetDevicePCIBusId;
      volatile auto p7 = &txMalloc;
      volatile auto p8 = &txFree;
      volatile auto p9 = &txMemcpy;
      volatile auto p10 = &txModuleLoad;
      volatile auto p11 = &txModuleUnload;
      volatile auto p12 = &txModuleGetFunction;
      volatile auto p13 = &txLaunchKernel;
      volatile auto p14 = &txStreamCreate;
      volatile auto p15 = &txStreamDestroy;
      volatile auto p16 = &txStreamQuery;
      // Event timing is part of the exact target-provider ABI baseline.
      // Per-invocation Disabled policy suppresses event use, not symbol
      // qualification of the configured digest-pinned runtime.
      volatile auto p17 = &txEventCreate;
      volatile auto p18 = &txEventDestroy;
      volatile auto p19 = &txEventRecord;
      volatile auto p20 = &txEventQuery;
      volatile auto p21 = &txEventElapsedTime;
      return !(p0 && p1 && p2 && p3 && p4 && p5 && p6 && p7 && p8 && p9 &&
               p10 && p11 && p12 && p13 && p14 && p15 && p16 && p17 && p18 &&
               p19 && p20 && p21);
    }
  ]] WAFER_TX_RUNTIME_HAS_REQUIRED_SYMBOLS)
  cmake_pop_check_state()
  if(NOT WAFER_TX_RUNTIME_HAS_REQUIRED_SYMBOLS)
    message(FATAL_ERROR
      "The configured TX runtime library does not provide the complete "
      "public symbol set required by wafer-run")
  endif()
endif()
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

if(WAFER_ENABLE_PRODUCT_INSTALL)
  if(NOT WAFER_ENABLE_IMPORTER_DEPS OR
     NOT WAFER_ENABLE_SPMD_PARTITIONER_DEPS OR
     NOT WAFER_ENABLE_PYTORCH_XLA_IMPORTER OR
     NOT WAFER_XLA_SPMD_PARTITIONER_HELPER)
    message(FATAL_ERROR
      "WAFER_ENABLE_PRODUCT_INSTALL=ON requires the importer, framework "
      "adapter, SPMD dependencies, and configured partitioner helper")
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

wafer_add_minimalloc()
wafer_add_structured_egraph()

add_subdirectory(include/Wafer/IR)
add_subdirectory(include/Wafer/Transforms)
add_subdirectory(include/Wafer/Conversion)
add_subdirectory(lib/Wafer)
add_subdirectory(tools/wafer-opt)
add_subdirectory(tools/wafer-compile)
add_subdirectory(tools/wafer-verify-program)
add_subdirectory(tools/wafer-run)
if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  add_subdirectory(tools/wafer-cmodel-qualify-onednn)
endif()

if(WAFER_ENABLE_PYTORCH_XLA_IMPORTER)
  set(_wafer_python_build_dir "${CMAKE_BINARY_DIR}/python")
  add_custom_command(
    OUTPUT "${_wafer_python_build_dir}/wafer/frontend/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_wafer_python_build_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/python/wafer" "${_wafer_python_build_dir}/wafer"
    DEPENDS
      "${CMAKE_SOURCE_DIR}/python/wafer/__init__.py"
      "${CMAKE_SOURCE_DIR}/python/wafer/frontend/__init__.py"
    VERBATIM)
  add_custom_target(wafer-python ALL
    DEPENDS "${_wafer_python_build_dir}/wafer/frontend/__init__.py")
  if(WAFER_ENABLE_PRODUCT_INSTALL)
    include(GNUInstallDirs)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/wafer"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/python"
      COMPONENT Compiler)
  endif()
endif()

enable_testing()
add_subdirectory(test)

if(WAFER_ENABLE_UNIT_TESTS)
  add_subdirectory(unittests)
endif()

set(_wafer_check_targets
  check-wafer-lit
)
foreach(_wafer_optional_check_target IN ITEMS
    check-wafer-unit
    check-wafer-numeric-model-unit
    check-wafer-target-numeric-backend-unit
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
