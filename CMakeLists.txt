cmake_minimum_required(VERSION 3.24)

project(WaferCompiler LANGUAGES CXX C)

include(cmake/WaferDependencyVersions.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WAFER_DEPS_ROOT "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH "Root for pinned third-party dependencies fetched by tools/bootstrap_deps.py")
option(WAFER_ALLOW_UNPINNED_LLVM "Allow an LLVM/MLIR package with the right major version but not the exact pinned release" OFF)
option(WAFER_ENABLE_IMPORTER_DEPS "Enable StableHLO/Shardy importer dependency discovery" OFF)
option(WAFER_ENABLE_UNIT_TESTS "Build Wafer C++ unit tests" ON)
option(WAFER_FETCH_GTEST "Fetch googletest when a system package is not available" ON)

if(NOT MLIR_DIR AND EXISTS "${WAFER_DEPS_ROOT}/llvm/${WAFER_LLVM_VERSION}/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "${WAFER_DEPS_ROOT}/llvm/${WAFER_LLVM_VERSION}/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake" FORCE)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL WAFER_LLVM_VERSION)
  if(WAFER_ALLOW_UNPINNED_LLVM AND LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21" AND LLVM_PACKAGE_VERSION VERSION_LESS "22")
    message(WARNING "Using LLVM ${LLVM_PACKAGE_VERSION} instead of pinned ${WAFER_LLVM_VERSION}")
  else()
    message(FATAL_ERROR
      "Wafer pins LLVM/MLIR ${WAFER_LLVM_VERSION}, but CMake found ${LLVM_PACKAGE_VERSION}. "
      "Run tools/bootstrap_deps.py --llvm or configure with -DWAFER_ALLOW_UNPINNED_LLVM=ON for a local override.")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(TableGen)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(WAFER_ENABLE_IMPORTER_DEPS)
  set(WAFER_STABLEHLO_SOURCE_DIR "${WAFER_DEPS_ROOT}/src/stablehlo" CACHE PATH "Pinned StableHLO checkout")
  set(WAFER_SHARDY_SOURCE_DIR "${WAFER_DEPS_ROOT}/src/shardy" CACHE PATH "Pinned Shardy checkout")
  if(NOT EXISTS "${WAFER_STABLEHLO_SOURCE_DIR}" OR NOT EXISTS "${WAFER_SHARDY_SOURCE_DIR}")
    message(FATAL_ERROR
      "StableHLO/Shardy sources are enabled but missing. "
      "Run tools/bootstrap_deps.py --importer-sources or set WAFER_STABLEHLO_SOURCE_DIR/WAFER_SHARDY_SOURCE_DIR.")
  endif()
endif()

add_subdirectory(include/Wafer/Dialect/Wafer/IR)
add_subdirectory(lib/Wafer)
add_subdirectory(tools/wafer-opt)

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
