# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022, Nordic Semiconductor ASA

# FindDeprecated module provides a single location for deprecated CMake build code.
# Whenever CMake code is deprecated it should be moved to this module and
# corresponding COMPONENTS should be created with name identifying the deprecated code.
#
# This makes it easier to maintain deprecated code and cleanup such code when it
# has been deprecated for two releases.
#
# Example:
# CMakeList.txt contains deprecated code, like:
# if(DEPRECATED_VAR)
#   deprecated()
# endif()
#
# such code can easily be around for a long time, so therefore such code should
# be moved to this module and can then be loaded as:
# FindDeprecated.cmake
# if(<deprecated_name> IN_LIST Deprecated_FIND_COMPONENTS)
#   # This code has been deprecated after Zephyr x.y
#   if(DEPRECATED_VAR)
#     deprecated()
#   endif()
# endif()
#
# and then in the original CMakeLists.txt, this code is inserted instead:
# find_package(Deprecated COMPONENTS <deprecated_name>)
#
# The module defines the following variables:
#
# 'Deprecated_FOUND', 'DEPRECATED_FOUND'
# True if the Deprecated component was found and loaded.

if("${Deprecated_FIND_COMPONENTS}" STREQUAL "")
  message(WARNING "find_package(Deprecated) missing required COMPONENTS keyword")
endif()

if("CROSS_COMPILE" IN_LIST Deprecated_FIND_COMPONENTS)
  list(REMOVE_ITEM Deprecated_FIND_COMPONENTS CROSS_COMPILE)
  if(NOT DEFINED TX8FW_TOOLCHAIN_VARIANT)
    set(TX8FW_TOOLCHAIN_VARIANT $ENV{TX8FW_TOOLCHAIN_VARIANT})
  endif()

  if(NOT TX8FW_TOOLCHAIN_VARIANT AND
     (CROSS_COMPILE OR (DEFINED ENV{CROSS_COMPILE})))
      set(TX8FW_TOOLCHAIN_VARIANT cross-compile CACHE STRING "Zephyr toolchain variant" FORCE)
      message(DEPRECATION  "Setting CROSS_COMPILE without setting TX8FW_TOOLCHAIN_VARIANT is deprecated."
                           "Please set TX8FW_TOOLCHAIN_VARIANT to 'cross-compile'"
      )
  endif()
endif()



if("SOURCES" IN_LIST Deprecated_FIND_COMPONENTS)
  list(REMOVE_ITEM Deprecated_FIND_COMPONENTS SOURCES)
  if(SOURCES)
    message(DEPRECATION
        "Setting SOURCES prior to calling find_package() for unit tests is deprecated."
        " To add sources after find_package() use:\n"
        "    target_sources(testbinary PRIVATE <source-file.c>)")
  endif()
endif()

if("PYTHON_PREFER" IN_LIST Deprecated_FIND_COMPONENTS)
  # This code was deprecated after Zephyr v3.4.0
  list(REMOVE_ITEM Deprecated_FIND_COMPONENTS PYTHON_PREFER)
  if(DEFINED PYTHON_PREFER)
    message(DEPRECATION "'PYTHON_PREFER' variable is deprecated. Please use "
                        "Python3_EXECUTABLE instead.")
    if(NOT DEFINED Python3_EXECUTABLE)
      set(Python3_EXECUTABLE ${PYTHON_PREFER})
    endif()
  endif()
endif()

if(NOT "${Deprecated_FIND_COMPONENTS}" STREQUAL "")
  message(STATUS "The following deprecated component(s) could not be found: "
                 "${Deprecated_FIND_COMPONENTS}")
endif()

set(Deprecated_FOUND True)
set(DEPRECATED_FOUND True)
