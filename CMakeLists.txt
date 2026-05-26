cmake_minimum_required(VERSION 3.24)

project(WaferCompiler LANGUAGES CXX C)

include(cmake/third_party/WaferDependencyVersions.cmake)
include(cmake/third_party/WaferThirdParty.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WAFER_ENABLE_UNIT_TESTS "Build Wafer C++ unit tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(include/Wafer/IR)
add_subdirectory(lib/Wafer)
add_subdirectory(tools/wafer-opt)
add_subdirectory(tools/wafer-import-model)

enable_testing()
add_subdirectory(test)

if(WAFER_ENABLE_UNIT_TESTS)
  add_subdirectory(unittests)
endif()

if(TARGET WaferUnitTests)
  add_custom_target(check-wafer DEPENDS check-wafer-lit WaferUnitTests)
else()
  add_custom_target(check-wafer DEPENDS check-wafer-lit)
endif()

if(TARGET wafer-shardy-cmake-gate)
  add_dependencies(check-wafer wafer-shardy-cmake-gate)
endif()
