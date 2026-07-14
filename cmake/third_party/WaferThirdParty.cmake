# Central third-party dependency discovery for the Wafer compiler.

set(WAFER_DEPS_ROOT "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH
  "Root for pinned third-party dependencies fetched by tools/bootstrap_deps.py")
set(WAFER_LLVM_SOURCE_DIR "${WAFER_DEPS_ROOT}/llvm-project" CACHE PATH
  "Pinned llvm-project checkout matching the OpenXLA/XLA workspace")
set(WAFER_LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/llvm-install/${WAFER_LLVM_COMMIT}" CACHE PATH
  "Install prefix for a source-built pinned LLVM/MLIR package")
string(SUBSTRING "${WAFER_LLVM_COMMIT}" 0 4 _wafer_llvm_commit_short)
set(WAFER_LLVM_BUILD_DIR "${CMAKE_SOURCE_DIR}/build/third_party/llvm-project-${_wafer_llvm_commit_short}" CACHE PATH
  "Build directory for the pinned LLVM/MLIR source tree, used for tools not installed by LLVM")
option(WAFER_ALLOW_UNPINNED_LLVM
  "Allow an LLVM/MLIR package with the right major version but not the exact pinned package version" OFF)
option(WAFER_ENABLE_IMPORTER_DEPS
  "Enable StableHLO/Shardy frontend and SPMD dependency discovery" OFF)
option(WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS
  "Enable framework importer adapter dependency checks" OFF)
option(WAFER_ENABLE_SPMD_PARTITIONER_DEPS
  "Enable Shardy/GSPMD partitioner dependency roots and unified Shardy CMake gate" OFF)
option(WAFER_ENABLE_RUNTIME_DEPS
  "Enable future runtime/driver SDK dependency roots" OFF)
option(WAFER_ENABLE_NUMERIC_MODEL_DEPS
  "Enable exact managed SoftFloat/TestFloat/GMP/MPFR numeric-model artifacts" OFF)
option(WAFER_ENABLE_BULK_MODEL_DEPS
  "Enable the exact managed oneDNN bulk functional-model artifact" OFF)
option(WAFER_ENABLE_SYSTEMC_MODEL
  "Enable the exact managed SystemC functional-event model dependency" OFF)
option(WAFER_FETCH_GTEST
  "Fetch googletest when a system package is not available" ON)

set(WAFER_NUMERIC_MODEL_DEPS_ROOT
  "${WAFER_DEPS_ROOT}/numeric-model" CACHE PATH
  "Root produced by tools/bootstrap_deps.py --numeric-model-deps")
set(WAFER_NUMERIC_MODEL_DEPS_RECORD
  "${WAFER_NUMERIC_MODEL_DEPS_ROOT}/numeric-model-deps.json" CACHE FILEPATH
  "Completed managed numeric-model dependency conformance record")
set(WAFER_BULK_MODEL_DEPS_ROOT
  "${WAFER_DEPS_ROOT}/bulk-model" CACHE PATH
  "Root produced by tools/bootstrap_deps.py --bulk-model-deps")
set(WAFER_BULK_MODEL_DEPS_RECORD
  "${WAFER_BULK_MODEL_DEPS_ROOT}/bulk-model-deps.json" CACHE FILEPATH
  "Completed managed bulk-model dependency conformance record")
set(WAFER_SYSTEMC_MODEL_DEPS_ROOT
  "${WAFER_DEPS_ROOT}/systemc-model" CACHE PATH
  "Root produced by tools/bootstrap_deps.py --systemc-model-deps")
set(WAFER_SYSTEMC_MODEL_DEPS_RECORD
  "${WAFER_SYSTEMC_MODEL_DEPS_ROOT}/systemc-model-deps.json" CACHE FILEPATH
  "Completed managed SystemC-model dependency conformance record")

set(WAFER_IMPORTER_PYTHON_VENV "${WAFER_DEPS_ROOT}/python-importer" CACHE PATH
  "Python env containing torch and source-built torch_xla runtime for importer tools")
set(WAFER_IMPORTER_PYTHON_EXECUTABLE "${WAFER_IMPORTER_PYTHON_VENV}/bin/python" CACHE FILEPATH
  "Python executable used for framework importer adapter tests")
set(WAFER_STABLEHLO_SOURCE_DIR "${WAFER_DEPS_ROOT}/stablehlo" CACHE PATH
  "Pinned StableHLO checkout")
set(WAFER_SHARDY_SOURCE_DIR "${WAFER_DEPS_ROOT}/shardy" CACHE PATH
  "Pinned Shardy checkout")
set(WAFER_OPENXLA_XLA_SOURCE_DIR "${WAFER_DEPS_ROOT}/xla" CACHE PATH
  "Pinned OpenXLA/XLA checkout selected by the PyTorch/XLA importer baseline")
set(WAFER_PYTORCH_XLA_SOURCE_DIR "${WAFER_DEPS_ROOT}/pytorch-xla" CACHE PATH
  "Pinned PyTorch/XLA checkout used as the framework importer dependency baseline")

set(WAFER_TX_RUNTIME_ROOT "" CACHE PATH
  "Optional tx_runtime SDK root containing provider headers/libs")
set(WAFER_KMD_UAPI_ROOT "" CACHE PATH
  "Optional KMD/UAPI headers root for future runtime adapter work")
set(WAFER_LEGACY_TSM_RUNTIME_ROOT "" CACHE PATH
  "Optional legacy Tsm/VS runtime SDK root for fallback adapter work")
set(WAFER_ENABLE_PYTORCH_XLA_IMPORTER OFF CACHE BOOL
  "Pinned source-built PyTorch/XLA importer runtime is importable" FORCE)

if(WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS)
  if(NOT EXISTS "${WAFER_PYTORCH_XLA_SOURCE_DIR}/WORKSPACE")
    message(FATAL_ERROR
      "PyTorch/XLA source checkout is enabled but missing. "
      "Run tools/bootstrap_deps.py --importer-sources or set WAFER_PYTORCH_XLA_SOURCE_DIR.")
  endif()
  if(EXISTS "${WAFER_IMPORTER_PYTHON_EXECUTABLE}")
    execute_process(
      COMMAND "${WAFER_IMPORTER_PYTHON_EXECUTABLE}" -c
              "import torch; import torch_xla; from torch_xla.stablehlo import exported_program_to_stablehlo"
      RESULT_VARIABLE WAFER_PYTORCH_XLA_IMPORTER_RESULT
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(WAFER_PYTORCH_XLA_IMPORTER_RESULT EQUAL 0)
      set(WAFER_ENABLE_PYTORCH_XLA_IMPORTER ON CACHE BOOL
        "Pinned source-built PyTorch/XLA importer runtime is importable" FORCE)
    else()
      message(FATAL_ERROR
        "WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON requires an importer Python "
        "that can import source-built torch_xla. Run tools/build_pytorch_xla_runtime.py "
        "with WAFER_IMPORTER_PYTHON_EXECUTABLE pointing at that environment.")
    endif()
  else()
    message(FATAL_ERROR
      "WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON but WAFER_IMPORTER_PYTHON_EXECUTABLE "
      "does not exist. Build/install torch_xla from third_party/pytorch-xla source "
      "into the importer Python environment.")
  endif()
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  foreach(_wafer_spmd_root
          WAFER_STABLEHLO_SOURCE_DIR
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
          WAFER_TX_RUNTIME_ROOT
          WAFER_KMD_UAPI_ROOT
          WAFER_LEGACY_TSM_RUNTIME_ROOT)
    if(NOT EXISTS "${${_wafer_runtime_root}}")
      message(FATAL_ERROR
        "${_wafer_runtime_root} must point at an existing SDK root when "
        "WAFER_ENABLE_RUNTIME_DEPS=ON")
    endif()
  endforeach()
endif()

if(NOT MLIR_DIR AND EXISTS "${WAFER_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "${WAFER_LLVM_INSTALL_DIR}/lib/cmake/mlir"
    CACHE PATH "Path to MLIRConfig.cmake" FORCE)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  include("${CMAKE_CURRENT_LIST_DIR}/WaferNumericModelDeps.cmake")
  wafer_enable_numeric_model_deps()
endif()

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  include("${CMAKE_CURRENT_LIST_DIR}/WaferBulkModelDeps.cmake")
  wafer_enable_bulk_model_deps()
endif()

if(WAFER_ENABLE_SYSTEMC_MODEL)
  include("${CMAKE_CURRENT_LIST_DIR}/WaferSystemCModelDeps.cmake")
  wafer_enable_systemc_model_deps()
endif()

string(REGEX MATCH "^[0-9]+" WAFER_LLVM_PACKAGE_VERSION_MAJOR "${WAFER_LLVM_PACKAGE_VERSION}")
if(NOT LLVM_PACKAGE_VERSION STREQUAL WAFER_LLVM_PACKAGE_VERSION)
  if(WAFER_ALLOW_UNPINNED_LLVM
      AND LLVM_VERSION_MAJOR EQUAL WAFER_LLVM_PACKAGE_VERSION_MAJOR)
    message(WARNING "Using LLVM ${LLVM_PACKAGE_VERSION} instead of pinned ${WAFER_LLVM_PACKAGE_VERSION}")
  else()
    message(FATAL_ERROR
      "Wafer pins LLVM/MLIR ${WAFER_LLVM_PACKAGE_VERSION} at ${WAFER_LLVM_COMMIT}, "
      "but CMake found ${LLVM_PACKAGE_VERSION}. Build/install the pinned llvm-project source "
      "or configure MLIR_DIR/LLVM_DIR explicitly; use -DWAFER_ALLOW_UNPINNED_LLVM=ON only for a local override.")
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

if(WAFER_ENABLE_IMPORTER_DEPS OR WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  if(NOT EXISTS "${WAFER_STABLEHLO_SOURCE_DIR}" OR NOT EXISTS "${WAFER_SHARDY_SOURCE_DIR}")
    message(FATAL_ERROR
      "StableHLO/Shardy sources are enabled but missing. "
      "Run tools/bootstrap_deps.py --importer-sources or set WAFER_STABLEHLO_SOURCE_DIR/WAFER_SHARDY_SOURCE_DIR.")
  endif()

  set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO embedded in Wafer" FORCE)
  set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable StableHLO Python bindings" FORCE)
  get_filename_component(_wafer_python_bin_dir "${Python3_EXECUTABLE}" DIRECTORY)
  if(LLVM_EXTERNAL_LIT AND EXISTS "${LLVM_EXTERNAL_LIT}")
    set(WAFER_STABLEHLO_LIT_EXECUTABLE "${LLVM_EXTERNAL_LIT}" CACHE FILEPATH
      "Command used by embedded StableHLO lit targets" FORCE)
  else()
    find_program(WAFER_STABLEHLO_LIT_EXECUTABLE
      NAMES lit llvm-lit
      HINTS
        "${WAFER_DEPS_ROOT}/python/bin"
        "${_wafer_python_bin_dir}"
        "${LLVM_TOOLS_BINARY_DIR}"
        "${WAFER_LLVM_BUILD_DIR}/bin"
      REQUIRED
    )
  endif()
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

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  include("${CMAKE_CURRENT_LIST_DIR}/WaferShardyCMake.cmake")
  wafer_add_shardy_targets()
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
