# SPDX-License-Identifier: Apache-2.0

# This file provides TX8FW sdk config version package functionality.
#

# Those are TX8FW variables used.
#
include( ${CMAKE_CURRENT_LIST_DIR}/pre_verify.cmake)
if(NOT DEFINED __choose_tool_chain )
    return()
endif ()
set(SDK_VERSION ${TX8FW-sdk_VERSION})
set(SDK_MAJOR_MINOR ${TX8FW-sdk_VERSION_MAJOR}.${TX8FW-sdk_VERSION_MINOR})
set(SDK_VERSION ${TX8FW-sdk_VERSION_MAJOR}.${TX8FW-sdk_VERSION_MINOR}.${TX8FW-sdk_VERSION_PATCH})
set(SDK_MAJOR_MINOR_MICRO ${SDK_VERSION})

get_filename_component(TX8FW_SDK_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
set(TX8FW_SDK_INSTALL_DIR ${TX8FW_SDK_INSTALL_DIR})
set(TX8FW_TOOLCHAIN_VARIANT cross-compile)

if((DEFINED CROSS_COMPILE) OR (DEFINED ENV{CROSS_COMPILE}))
    if(NOT CROSS_COMPILE)
        set(CROSS_COMPILE $ENV{CROSS_COMPILE})
    endif ()
else ()
    set(CROSS_COMPILE  "${TX8FW_SDK_INSTALL_DIR}/${__choose_tool_chain}/bin/riscv64-unknown-elf-" )
endif()

set(TOOLCHAIN_HOME "${TX8FW_SDK_INSTALL_DIR}/${__choose_tool_chain}")

# Those are CMake package parameters.
set(TX8FW-sdk_FOUND True)
set(TX8FW-sdk_CONFIG_MODE_FOUND True)
set(TX8FW-sdk_DIR   ${TX8FW_SDK_INSTALL_DIR})
unset(__choose_tool_chain)