# SPDX-License-Identifier: Apache-2.0

# Purpose of this CMake file is to install a ZephyrConfig package reference in:
# Unix/Linux/MacOS: ~/.cmake/packages/TX8FW
# Windows         : HKEY_CURRENT_USER
#
# Having ZephyrConfig package allows for find_package(TX8FW) to work when TX8FW_BASE is not defined.
#
# Create the reference by running `cmake -P tx8fw_export.cmake` in this directory.

string(MD5 MD5_SUM ${CMAKE_CURRENT_LIST_DIR})
if(WIN32)
  execute_process(COMMAND ${CMAKE_COMMAND}
                  -E  write_regv
                 "HKEY_CURRENT_USER\\Software\\Kitware\\CMake\\Packages\\TX8FW\;${MD5_SUM}" "${CMAKE_CURRENT_LIST_DIR}"
)
else()
  file(WRITE $ENV{HOME}/.cmake/packages/TX8FW/${MD5_SUM} ${CMAKE_CURRENT_LIST_DIR})
endif()

message("TX8FW (${CMAKE_CURRENT_LIST_DIR})")
message("has been added to the user package registry in:")
if(WIN32)
  message("HKEY_CURRENT_USER\\Software\\Kitware\\CMake\\Packages\\TX8FW\n")
else()
  message("~/.cmake/packages/TX8FW\n")
endif()

file(REMOVE ${CMAKE_CURRENT_LIST_DIR}/${MD5_INFILE})
