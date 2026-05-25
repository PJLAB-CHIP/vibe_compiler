find_package(GTest QUIET)

if(NOT GTest_FOUND)
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
endif()

add_executable(WaferUnitTests
  Dialect/Wafer/WaferDialectTest.cpp
)
target_link_libraries(WaferUnitTests
  PRIVATE
  WaferIR
  MLIRIR
  MLIRParser
  GTest::gtest_main
)

add_test(NAME WaferUnitTests COMMAND WaferUnitTests)
