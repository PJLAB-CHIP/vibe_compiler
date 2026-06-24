# SPDX-License-Identifier: Apache-2.0

# The purpose of this CMake file is to register the TX8FW-sdk package in:
#
#   Linux/macOS: ~/.cmake/packages/TX8FW-sdk
#       Windows: HKEY_CURRENT_USER
#
# By registering the TX8FW-sdk package, the TX8FW build system can locate the
# required version of TX8FW SDK through `find_package(TX8FW-sdk)` even when
# TX8FW_SDK_INSTALL_DIR is not explicitly set.
#
# Register the TX8FW-sdk package by running `cmake -P TX8FW_sdk_export.cmake`
# in this directory.

set(MD5_INFILE "current_path.txt")

# We write CMAKE_CURRENT_LIST_DIR into MD5_INFILE, as the content of that file
# will be used for MD5 calculation.  This means we effectively get the MD5 of
# CMAKE_CURRENT_LIST_DIR which must be used for CMake user package registry.
file(WRITE ${CMAKE_CURRENT_LIST_DIR}/${MD5_INFILE} ${CMAKE_CURRENT_LIST_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum ${CMAKE_CURRENT_LIST_DIR}/${MD5_INFILE}
                OUTPUT_VARIABLE MD5_SUM
)
string(SUBSTRING ${MD5_SUM} 0 32 MD5_SUM)
if(WIN32)
  execute_process(COMMAND ${CMAKE_COMMAND}
                  -E write_regv
                  "HKEY_CURRENT_USER\\Software\\Kitware\\CMake\\Packages\\TX8FW-sdk\;${MD5_SUM}"
                  "${CMAKE_CURRENT_LIST_DIR}"
  )
else()
  file(WRITE $ENV{HOME}/.cmake/packages/TX8FW-sdk/${MD5_SUM} ${CMAKE_CURRENT_LIST_DIR})
endif()

message("TX8FW-sdk (${CMAKE_CURRENT_LIST_DIR})")
message("has been added to the user package registry in:")
if(WIN32)
  message("HKEY_CURRENT_USER\\Software\\Kitware\\CMake\\Packages\\TX8FW-sdk\n")
else()
  message("~/.cmake/packages/TX8FW-sdk\n")
endif()

file(REMOVE ${CMAKE_CURRENT_LIST_DIR}/${MD5_INFILE})
