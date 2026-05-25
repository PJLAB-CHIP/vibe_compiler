add_library(WaferABI
  M0Abi.cpp
)

target_include_directories(WaferABI
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
