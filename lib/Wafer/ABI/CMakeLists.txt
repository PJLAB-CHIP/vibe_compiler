add_library(WaferABI
  TileAbi.cpp
)

target_include_directories(WaferABI
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
