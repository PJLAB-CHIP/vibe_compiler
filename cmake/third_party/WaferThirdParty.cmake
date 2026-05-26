# Central third-party dependency discovery for the Wafer compiler.

set(WAFER_DEPS_ROOT "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH
  "Root for pinned third-party dependencies fetched by tools/bootstrap_deps.py")
option(WAFER_ALLOW_UNPINNED_LLVM
  "Allow an LLVM/MLIR package with the right major version but not the exact pinned release" OFF)
option(WAFER_ENABLE_IMPORTER_DEPS
  "Enable StableHLO/Shardy frontend and SPMD dependency discovery" OFF)
option(WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS
  "Enable future framework importer adapter dependency roots" OFF)
option(WAFER_ENABLE_SPMD_PARTITIONER_DEPS
  "Enable future Shardy/GSPMD partitioner dependency roots" OFF)
option(WAFER_ENABLE_RUNTIME_DEPS
  "Enable future runtime/driver SDK dependency roots" OFF)
option(WAFER_FETCH_GTEST
  "Fetch googletest when a system package is not available" ON)

set(WAFER_TORCH_MLIR_SOURCE_DIR "${WAFER_DEPS_ROOT}/torch-mlir" CACHE PATH
  "Pinned torch-mlir checkout for future frontend importer adapters")
set(WAFER_PYTORCH_XLA_SOURCE_DIR "${WAFER_DEPS_ROOT}/pytorch-xla" CACHE PATH
  "Pinned PyTorch/XLA checkout for future torch.export to StableHLO importers")
set(WAFER_IMPORTER_PYTHON_VENV "${WAFER_DEPS_ROOT}/python-importer" CACHE PATH
  "Python venv containing pinned torch/torchvision/torch_xla importer wheels")
set(WAFER_STABLEHLO_SOURCE_DIR "${WAFER_DEPS_ROOT}/stablehlo" CACHE PATH
  "Pinned StableHLO checkout")
set(WAFER_SHARDY_SOURCE_DIR "${WAFER_DEPS_ROOT}/shardy" CACHE PATH
  "Pinned Shardy checkout")
set(WAFER_OPENXLA_XLA_SOURCE_DIR "${WAFER_DEPS_ROOT}/xla" CACHE PATH
  "Pinned OpenXLA/XLA checkout for future GSPMD partitioner integration")

set(WAFER_HPGR_SDK_ROOT "" CACHE PATH
  "Optional HPGR runtime SDK root containing tx_runtime headers/libs")
set(WAFER_KMD_UAPI_ROOT "" CACHE PATH
  "Optional KMD/UAPI headers root for future runtime adapter work")
set(WAFER_LEGACY_TSM_SDK_ROOT "" CACHE PATH
  "Optional legacy Tsm/VS runtime SDK root for fallback adapter work")

if(WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS)
  foreach(_wafer_importer_root
          WAFER_TORCH_MLIR_SOURCE_DIR
          WAFER_PYTORCH_XLA_SOURCE_DIR)
    if(NOT EXISTS "${${_wafer_importer_root}}")
      message(FATAL_ERROR
        "${_wafer_importer_root} must point at an existing checkout when "
        "WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON")
    endif()
  endforeach()
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  foreach(_wafer_spmd_root
          WAFER_SHARDY_SOURCE_DIR
          WAFER_OPENXLA_XLA_SOURCE_DIR)
    if(NOT EXISTS "${${_wafer_spmd_root}}")
      message(FATAL_ERROR
        "${_wafer_spmd_root} must point at an existing checkout when "
        "WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON")
    endif()
  endforeach()
endif()

if(WAFER_ENABLE_RUNTIME_DEPS)
  foreach(_wafer_runtime_root
          WAFER_HPGR_SDK_ROOT
          WAFER_KMD_UAPI_ROOT
          WAFER_LEGACY_TSM_SDK_ROOT)
    if(NOT EXISTS "${${_wafer_runtime_root}}")
      message(FATAL_ERROR
        "${_wafer_runtime_root} must point at an existing SDK root when "
        "WAFER_ENABLE_RUNTIME_DEPS=ON")
    endif()
  endforeach()
endif()

if(NOT MLIR_DIR AND EXISTS "${WAFER_DEPS_ROOT}/llvm/${WAFER_LLVM_VERSION}/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "${WAFER_DEPS_ROOT}/llvm/${WAFER_LLVM_VERSION}/lib/cmake/mlir"
    CACHE PATH "Path to MLIRConfig.cmake" FORCE)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL WAFER_LLVM_VERSION)
  if(WAFER_ALLOW_UNPINNED_LLVM
      AND LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21"
      AND LLVM_PACKAGE_VERSION VERSION_LESS "22")
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

if(WAFER_ENABLE_IMPORTER_DEPS)
  if(NOT EXISTS "${WAFER_STABLEHLO_SOURCE_DIR}" OR NOT EXISTS "${WAFER_SHARDY_SOURCE_DIR}")
    message(FATAL_ERROR
      "StableHLO/Shardy sources are enabled but missing. "
      "Run tools/bootstrap_deps.py --importer-sources or set WAFER_STABLEHLO_SOURCE_DIR/WAFER_SHARDY_SOURCE_DIR.")
  endif()

  set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO embedded in Wafer" FORCE)
  set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable StableHLO Python bindings" FORCE)
  get_filename_component(_wafer_python_bin_dir "${Python3_EXECUTABLE}" DIRECTORY)
  find_program(WAFER_STABLEHLO_LIT_EXECUTABLE
    NAMES lit llvm-lit
    HINTS "${WAFER_DEPS_ROOT}/python/bin" "${_wafer_python_bin_dir}" "${LLVM_TOOLS_BINARY_DIR}"
    REQUIRED
  )
  set(LLVM_EXTERNAL_LIT "${WAFER_STABLEHLO_LIT_EXECUTABLE}" CACHE STRING
      "Command used by embedded StableHLO lit targets" FORCE)

  foreach(_wafer_filecheck_candidate
          "${LLVM_TOOLS_BINARY_DIR}/FileCheck"
          "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/bin/FileCheck")
    if(EXISTS "${_wafer_filecheck_candidate}" AND NOT IS_DIRECTORY "${_wafer_filecheck_candidate}")
      execute_process(
        COMMAND "${_wafer_filecheck_candidate}" --version
        RESULT_VARIABLE _wafer_filecheck_candidate_result
        OUTPUT_QUIET
        ERROR_QUIET
      )
      if(_wafer_filecheck_candidate_result EQUAL 0)
        set(WAFER_STABLEHLO_FILECHECK_EXECUTABLE "${_wafer_filecheck_candidate}")
        break()
      endif()
    endif()
  endforeach()
  if(NOT WAFER_STABLEHLO_FILECHECK_EXECUTABLE)
    find_program(WAFER_STABLEHLO_FILECHECK_EXECUTABLE NAMES FileCheck REQUIRED)
  endif()

  find_program(WAFER_STABLEHLO_NOT_EXECUTABLE
    NAMES not
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/bin"
    REQUIRED
  )

  if(NOT TARGET FileCheck)
    add_executable(FileCheck IMPORTED GLOBAL)
    set_target_properties(FileCheck PROPERTIES
      IMPORTED_LOCATION "${WAFER_STABLEHLO_FILECHECK_EXECUTABLE}")
  endif()
  if(NOT TARGET not)
    add_executable(not IMPORTED GLOBAL)
    set_target_properties(not PROPERTIES
      IMPORTED_LOCATION "${WAFER_STABLEHLO_NOT_EXECUTABLE}")
  endif()

  add_subdirectory("${WAFER_STABLEHLO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/stablehlo" EXCLUDE_FROM_ALL)
endif()

function(wafer_require_gtest)
  if(TARGET GTest::gtest_main)
    return()
  endif()

  set(_wafer_googletest_source_dir "${WAFER_DEPS_ROOT}/googletest")
  if(EXISTS "${_wafer_googletest_source_dir}/CMakeLists.txt")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(
      "${_wafer_googletest_source_dir}"
      "${CMAKE_BINARY_DIR}/googletest"
      EXCLUDE_FROM_ALL)
    return()
  endif()

  find_package(GTest QUIET)
  if(GTest_FOUND)
    return()
  endif()

  if(NOT WAFER_FETCH_GTEST)
    message(FATAL_ERROR "GTest not found and WAFER_FETCH_GTEST=OFF")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY ${WAFER_GOOGLETEST_REPOSITORY}
    GIT_TAG ${WAFER_GOOGLETEST_COMMIT}
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endfunction()
