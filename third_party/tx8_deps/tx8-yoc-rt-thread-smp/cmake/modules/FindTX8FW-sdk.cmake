# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2023, Nordic Semiconductor ASA

# FindTX8FW-sdk module for supporting module search mode of tx8fw SDK.
#
# Its purpose is to allow the find_package basic signature mode to lookup Zephyr
# SDK and based on user / environment settings of selected toolchain decide if
# the tx8fw SDK CMake package should be loaded.
#
# It extends the TX8FW-sdk CMake package by providing more flexibility in when
# the tx8fw SDK is loaded and loads additional host tools from the tx8fw SDK.
#
# The module defines the following variables:
#
# 'TX8FW_SDK_INSTALL_DIR'
# Install location of the tx8fw SDK
#
# 'TX8FW_TOOLCHAIN_VARIANT'
# Zephyr toolchain variant to use if not defined already.
#
# 'TX8FW-sdk_FOUND'
# True if the tx8fw SDK was found.


set(FOR_SDK_VENDOR "tsingmicro")
set(FOR_SDK_PRODUCT_SERIES "tx8")

# Set internal variables if set in environment.
zephyr_get(TX8FW_TOOLCHAIN_VARIANT)

zephyr_get(TX8FW_SDK_INSTALL_DIR)

# Load tx8fw SDK Toolchain.
# There are three scenarios where tx8fw SDK should be looked up:
# 1) Zephyr specified as toolchain (TX8FW_SDK_INSTALL_DIR still used if defined)
# 2) No toolchain specified == Default to Zephyr toolchain
# Until we completely deprecate it
if(
(NOT  TX8FW_TOOLCHAIN_VARIANT) OR
( TX8FW_SDK_INSTALL_DIR) OR
(TX8FW-sdk_FIND_REQUIRED))

  if( (NOT  TX8FW_TOOLCHAIN_VARIANT) AND
  ( TX8FW_SDK_INSTALL_DIR))
    include(${TX8FW_SDK_INSTALL_DIR}/cmake/TX8FW_TOOLCHAIN_VARIANT.cmake OPTIONAL)
  endif ()

  # No toolchain was specified, so inform user that we will be searching.
  if (NOT  TX8FW_SDK_INSTALL_DIR AND
          NOT  TX8FW_TOOLCHAIN_VARIANT)
    message(STATUS "TX8FW_TOOLCHAIN_VARIANT not set, trying to locate tx8fw SDK")
  endif()

  # This ensure packages are sorted in descending order.
  SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION_CURRENT ${CMAKE_FIND_PACKAGE_SORT_DIRECTION})
  SET(CMAKE_FIND_PACKAGE_SORT_ORDER_CURRENT ${CMAKE_FIND_PACKAGE_SORT_ORDER})
  SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
  SET(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

  if( TX8FW_SDK_INSTALL_DIR)
    # The tx8fw SDK will automatically set the toolchain variant.
    # To support tx8fw SDK tools (DTC, and other tools) with 3rd party toolchains
    # then we keep track of current toolchain variant.
    set(ZEPHYR_CURRENT_TOOLCHAIN_VARIANT ${TX8FW_TOOLCHAIN_VARIANT})
    find_package(TX8FW-sdk ${TX8FW-sdk_FIND_VERSION}
            QUIET  CONFIG HINTS ${TX8FW_SDK_INSTALL_DIR}
    )
    if(DEFINED ZEPHYR_CURRENT_TOOLCHAIN_VARIANT)
      set(TX8FW_TOOLCHAIN_VARIANT ${ZEPHYR_CURRENT_TOOLCHAIN_VARIANT})
    endif()
  else()
    # Paths that are used to find installed tx8fw SDK versions
    SET(tx8fw-sdksearch_paths
        /usr
        /usr/local
        /opt
        $ENV{HOME}
        $ENV{HOME}/.local
        $ENV{HOME}/.local/opt
        $ENV{HOME}/bin)

    # Search for tx8fw SDK version 0.0.0 which does not exist, this is needed to
    # return a list of compatible versions and find the best suited version that
    # is available.
    find_package(TX8FW-sdk 0.0.0 EXACT QUIET CONFIG PATHS ${tx8fw-sdksearch_paths})

    # Remove duplicate entries and sort naturally in descending order.
    set(tx8fw-sdkfound_versions ${TX8FW-sdk_CONSIDERED_VERSIONS})
    set(tx8fw-sdkfound_configs ${TX8FW-sdk_CONSIDERED_CONFIGS})

    list(REMOVE_DUPLICATES TX8FW-sdk_CONSIDERED_VERSIONS)
    list(SORT TX8FW-sdk_CONSIDERED_VERSIONS COMPARE NATURAL ORDER DESCENDING)

    # Loop over each found TX8FW SDK version until one is found that is compatible.
    foreach(tx8fw-sdkcandidate ${TX8FW-sdk_CONSIDERED_VERSIONS})
      if("${tx8fw-sdkcandidate}" VERSION_GREATER_EQUAL "${TX8FW-sdk_FIND_VERSION}")
        # Find the path for the current version being checked and get the directory
        # of the tx8fw SDK so it can be checked.
        list(FIND tx8fw-sdkfound_versions ${tx8fw-sdkcandidate} tx8fw-sdkcurrent_index)
        list(GET tx8fw-sdkfound_configs ${tx8fw-sdkcurrent_index} tx8fw-sdkcurrent_check_path)
        get_filename_component(tx8fw-sdkcurrent_check_path ${tx8fw-sdkcurrent_check_path} DIRECTORY)

        # Then see if this version is compatible.
        if (${TX8FW-sdk_FIND_VERSION})
          find_package(TX8FW-sdk ${TX8FW-sdk_FIND_VERSION} QUIET CONFIG PATHS ${tx8fw-sdkcurrent_check_path} NO_DEFAULT_PATH)
        elseif (NOT ${tx8fw-sdkcandidate} STREQUAL "unknown")
          find_package(TX8FW-sdk ${tx8fw-sdkcandidate} QUIET CONFIG PATHS ${tx8fw-sdkcurrent_check_path} NO_DEFAULT_PATH)
        else ()
          find_package(TX8FW-sdk QUIET CONFIG PATHS ${tx8fw-sdkcurrent_check_path} NO_DEFAULT_PATH)
        endif ()
        if (${TX8FW-sdk_FOUND})
          # A compatible version of the tx8fw SDK has been found which is the highest
          # supported version, exit.
          break()
        endif()
      endif()
    endforeach()

    if (NOT ${TX8FW-sdk_FOUND})
      # This means no compatible tx8fw SDK versions were found, set the version
      # back to the minimum version so that it is displayed in the error text.
      find_package(TX8FW-sdk ${TX8FW-sdk_FIND_VERSION} QUIET  CONFIG PATHS ${tx8fw-sdksearch_paths})
    endif()
  endif()

  SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION ${CMAKE_FIND_PACKAGE_SORT_DIRECTION_CURRENT})
  SET(CMAKE_FIND_PACKAGE_SORT_ORDER ${CMAKE_FIND_PACKAGE_SORT_ORDER_CURRENT})
endif()

# Clean up temp variables
set(tx8fw-sdksearch_paths)
set(tx8fw-sdkfound_versions)
set(tx8fw-sdkfound_configs)
set(tx8fw-sdkcurrent_index)
set(tx8fw-sdkcurrent_check_path)

if(DEFINED TX8FW_SDK_INSTALL_DIR)
  # Cache the tx8fw SDK install dir.
  set(TX8FW_SDK_INSTALL_DIR ${TX8FW_SDK_INSTALL_DIR} CACHE PATH "tx8fw SDK install directory")
endif()

if(TX8FW-sdk_FOUND)
  include(${TX8FW_SDK_INSTALL_DIR}/cmake/tx8fw/host-tools.cmake OPTIONAL)

  message(STATUS "Found host-tools: tx8fw  ${SDK_VERSION} (${TX8FW_SDK_INSTALL_DIR})")
endif()
