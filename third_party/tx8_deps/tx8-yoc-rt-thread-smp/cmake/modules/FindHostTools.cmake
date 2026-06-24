# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022, Nordic Semiconductor ASA

# FindHostTools module for locating a set of tools to use on the host for
# Zephyr development.
#
# This module will lookup the following tools for Zephyr development:
# +---------------------------------------------------------------+
# | Tool               | Required |  Notes:                       |
# +---------------------------------------------------------------+
# | Generic C-compiler | Yes      |  Pre-processing of devicetree |
# | TX8FW-sdk         |          |                               |
# | gperf              |          |                               |
# | openocd            |          |                               |
# | bossac             |          |                               |
# | imgtool            |          |                               |
# +---------------------------------------------------------------+
#
# The module defines the following variables:
#
# 'CMAKE_C_COMPILER'
# Path to C compiler.
# Set to 'CMAKE_C_COMPILER-NOTFOUND' if no C compiler was found.
#
# 'GPERF'
# Path to gperf.
# Set to 'GPERF-NOTFOUND' if gperf was not found.
#
# 'OPENOCD'
# Path to openocd.
# Set to 'OPENOCD-NOTFOUND' if openocd was not found.
#
# 'BOSSAC'
# Path to bossac.
# Set to 'BOSSAC-NOTFOUND' if bossac was not found.
#
# 'IMGTOOL'
# Path to imgtool.
# Set to 'IMGTOOL-NOTFOUND' if imgtool was not found.
#
# 'HostTools_FOUND', 'HOSTTOOLS_FOUND'
# True if all required host tools were found.

include(extensions)

if(HostTools_FOUND)
  return()
endif()
zephyr_get(TX8FW_SDK_PREFER_TOOLCHAIN)

if(NOT DEFINED CUSTOM_FIND_TX8FW_SDK)
  if(NOT DEFINED TX8FW_SDK_VERSION )
    #set(TX8FW_SDK_VERSION 0.15)
  endif ()
  find_package(TX8FW-sdk ${TX8FW_SDK_VERSION})
  message(NOTICE "TX8FW_SDK_INSTALL_DIR=${TX8FW_SDK_INSTALL_DIR}")
  if (NOT TX8FW-sdk_FOUND)
     MESSAGE(FATAL_ERROR "Cant find TX8FW SDK with TX8FW_SDK_VERSION: ${TX8FW_SDK_VERSION}  TX8FW_SDK_PREFER_TOOLCHAIN:${TX8FW_SDK_PREFER_TOOLCHAIN}" )
  endif ()

endif ()
# Default to the host system's toolchain if we are targeting a host based target
if((${BOARD_DIR} MATCHES "boards\/native") OR ("${ARCH}" STREQUAL "posix")
   OR ("${BOARD}" STREQUAL "unit_testing"))
  if(NOT "${TX8FW_TOOLCHAIN_VARIANT}" STREQUAL "llvm")
    set(TX8FW_TOOLCHAIN_VARIANT "host")
  endif()
endif()

# Prevent CMake from testing the toolchain
set(CMAKE_C_COMPILER_FORCED   1)
set(CMAKE_CXX_COMPILER_FORCED 1)

if(NOT TOOLCHAIN_ROOT)
  if(DEFINED ENV{TOOLCHAIN_ROOT})
    # Support for out-of-tree toolchain
    set(TOOLCHAIN_ROOT $ENV{TOOLCHAIN_ROOT})
  else()
    # Default toolchain cmake file
    set(TOOLCHAIN_ROOT ${TX8FW_BASE})
  endif()
endif()
zephyr_file(APPLICATION_ROOT TOOLCHAIN_ROOT)


set(TOOLCHAIN_ROOT ${TOOLCHAIN_ROOT} CACHE STRING "tx8fw toolchain root" FORCE)
assert(TOOLCHAIN_ROOT "tx8fw toolchain root path invalid: please set the TOOLCHAIN_ROOT-variable")

# Set cached TX8FW_TOOLCHAIN_VARIANT.
set(TX8FW_TOOLCHAIN_VARIANT ${TX8FW_TOOLCHAIN_VARIANT} CACHE STRING "tx8fw toolchain variant")

# Configure the toolchain based on what SDK/toolchain is in use.
include(${TOOLCHAIN_ROOT}/cmake/toolchain/${TX8FW_TOOLCHAIN_VARIANT}/generic.cmake)

# Configure the toolchain based on what toolchain technology is used
# (gcc, host-gcc etc.)
include(${TOOLCHAIN_ROOT}/cmake/compiler/${COMPILER}/generic.cmake OPTIONAL)

include(${TOOLCHAIN_ROOT}/cmake/linker/${LINKER}/generic.cmake OPTIONAL)
include(${TOOLCHAIN_ROOT}/cmake/bintools/${BINTOOLS}/generic.cmake OPTIONAL)

# Optional folder for toolchains with may provide a Kconfig file for capabilities settings.
set_ifndef(TOOLCHAIN_KCONFIG_DIR ${TOOLCHAIN_ROOT}/cmake/toolchain/${TX8FW_TOOLCHAIN_VARIANT})

set(HostTools_FOUND TRUE)
set(HOSTTOOLS_FOUND TRUE)
