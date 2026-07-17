cmake_minimum_required(VERSION 3.24)

project(WaferThirdPartyMiniMallocProject LANGUAGES CXX)

add_library(WaferThirdPartyMiniMalloc STATIC
  src/minimalloc.cc
  src/solver.cc
  src/sweeper.cc
)

target_include_directories(WaferThirdPartyMiniMalloc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(WaferThirdPartyMiniMalloc PUBLIC cxx_std_17)
set_target_properties(WaferThirdPartyMiniMalloc PROPERTIES
  CXX_EXTENSIONS NO
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(WaferThirdPartyMiniMalloc PRIVATE /EHs-c-)
else()
  target_compile_options(WaferThirdPartyMiniMalloc PRIVATE
    -fno-exceptions
    -fno-rtti
  )
endif()

option(WAFER_MINIMALLOC_BUILD_TESTS
  "Build the standalone tests for Wafer's curated MiniMalloc core"
  ${PROJECT_IS_TOP_LEVEL})
if(WAFER_MINIMALLOC_BUILD_TESTS)
  enable_testing()
  add_executable(WaferThirdPartyMiniMallocTest tests/minimalloc_test.cc)
  target_link_libraries(WaferThirdPartyMiniMallocTest
    PRIVATE WaferThirdPartyMiniMalloc)
  target_compile_features(WaferThirdPartyMiniMallocTest PRIVATE cxx_std_17)
  set_target_properties(WaferThirdPartyMiniMallocTest PROPERTIES
    CXX_EXTENSIONS NO)
  add_test(NAME WaferThirdPartyMiniMallocTest
    COMMAND WaferThirdPartyMiniMallocTest)
endif()
