# SPDX-License-Identifier: Apache-2.0

# This file provides TX8FW Config Package functionality.
#
# The purpose of this files is to allow users to decide if they want to:
# - Use TX8FW_BASE environment setting for explicitly set select a tx8fw installation
# - Support automatic TX8FW installation lookup through the use of find_package(ZEPHYR)

# First check to see if user has provided a TX8FW base manually.
# Set TX8FW base to environment setting.
# It will be empty if not set in environment.

# Internal TX8FW CMake package message macro.
#
# This macro is only intended to be used within the TX8FW CMake package.
# The function `find_package()` supports an optional QUIET argument, and to
# honor that argument, the package_message() macro will not print messages when
# said flag has been given.
#
# Arguments to tx8fw_package_message() are identical to regular CMake message()
# function.
macro(tx8fw_package_message)
  if(NOT TX8FW_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()
macro(include_boilerplate location)

  set(TX8FW_DIR ${TX8FW_BASE}/share/tx8fw-package/cmake CACHE PATH
      "The directory containing a CMake configuration file for TX8FW." FORCE
  )
  include( ${CMAKE_CURRENT_LIST_DIR}/pre_verify.cmake)
  if(NOT TX8FW_PRODUCT_SERIES_FOUND)
    message(FATAL_ERROR "TX8FW pacakge:${TX8FW_BASE} is for ${TX8FW_PRODUCT_SERIES}" )
  endif ()
  unset(TX8FW_PRODUCT_SERIES_FOUND)
  unset(TX8FW_PRODUCT_SERIES)
  list(PREPEND CMAKE_MODULE_PATH ${TX8FW_BASE}/cmake/modules)
  set(TX8FW_FOUND True)# 这个意味着找到

  if(NOT DEFINED APPLICATION_SOURCE_DIR)
    set(APPLICATION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH
        "Application Source Directory"
    )
  endif()

  if(NOT DEFINED APPLICATION_BINARY_DIR)
    set(APPLICATION_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH
        "Application Binary Directory"
    )
  endif()

  set(__build_dir ${APPLICATION_BINARY_DIR}/tx8fw)
  set(PROJECT_BINARY_DIR ${__build_dir})

  list(LENGTH TX8FW_FIND_COMPONENTS components_length)
  # The module messages are intentionally higher than STATUS to avoid the -- prefix
  # and make them more visible to users. This does result in them being output
  # to stderr, but that is an implementation detail of cmake.
  if(components_length EQUAL 0)
    tx8fw_package_message(NOTICE "Loading TX8FW default modules (${location}).")
    include(tx8fw_default NO_POLICY_SCOPE)
  else()
    string(JOIN " " msg_components ${TX8FW_FIND_COMPONENTS})
    tx8fw_package_message(NOTICE "Loading TX8FW module(s) (${location}): ${msg_components}")
    foreach(component ${TX8FW_FIND_COMPONENTS})
      if(${component} MATCHES "^\([^:]*\):\(.*\)$")
        #字符串 (example: This is a test) 符合这个模式，而 (example This is a test) 或 example: This is a test) 则不符合。
        string(REPLACE "," ";" SUB_COMPONENTS ${CMAKE_MATCH_2})
        set(component ${CMAKE_MATCH_1})
      endif()
      include(${component})
    endforeach()
  endif()
endmacro()

# cmake  在查找到符合的版本过程中会许多加载 <Package>ConfigVersion.cmake, 而 <Package>Config.cmake 只有在找到合适的version.cmake,将
# 设置 <Package>_CONFIG 之后 才会加载解析 <Package>Config.cmake
#
# 如果没有在 xxx.make或者 CMakeList.txt 中定义了ZEPHYR_BASE,而在环境变量通过中定义的话直接用环境变量通过中定义的ZEPHYR_BASE
# 如果在xxx.make或者 CMakeList.txt 中定义了ZEPHYR_BASE,直接用定义的ZEPHYR_BASE
# TX8FW_BASE 的来源一个是 用户在 cmake的模块文件中设定, 一个是通过命令行设定, 另外一个就是本文件中通set 语句设定

set(ENV_TX8FW_BASE $ENV{TX8FW_BASE})
if((NOT DEFINED TX8FW_BASE) AND (DEFINED ENV_TX8FW_BASE))
  # Get rid of any double folder string before comparison, as example, user provides
  # TX8FW_BASE=//path/to//tx8fw_base/
  # must also work.
  get_filename_component(TX8FW_BASE ${ENV_TX8FW_BASE} ABSOLUTE)
  set(TX8FW_BASE ${TX8FW_BASE} CACHE PATH "TX8FW base")
  include_boilerplate("TX8FW base")
  return()
endif()

if (DEFINED TX8FW_BASE)
  include_boilerplate("TX8FW base (cached)")
  return()
endif()

# If TX8FW_CANDIDATE is set, it means this file was include instead of called via find_package directly.
# tx8fw_package_search.cmake 中 85行 遍历  foreach(TX8FW_CANDIDATE ${TX8FW_CONSIDERED_CONFIGS}) 时候会有这个
# 变量的设置而且会在tx8fw_package_search.cmake 的 99行执行include(${TX8FW_CANDIDATE} NO_POLICY_SCOPE)所以set(IS_INCLUDED TRUE)
if(TX8FW_CANDIDATE)
  set(IS_INCLUDED TRUE)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/tx8fw_package_search.cmake)
endif()

# Find out the current TX8FW base.
get_filename_component(CURRENT_TX8FW_DIR ${CMAKE_CURRENT_LIST_FILE}/${ZEPHYR_RELATIVE_DIR} ABSOLUTE)
get_filename_component(CURRENT_WORKSPACE_DIR ${CMAKE_CURRENT_LIST_FILE}/${WORKSPACE_RELATIVE_DIR} ABSOLUTE)

# We are in TX8FW repository.
string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${CURRENT_TX8FW_DIR}/" COMMON_INDEX)
if (COMMON_INDEX EQUAL 0)
  # Project is in TX8FW repository.
  # We are in TX8FW repository.
  set(TX8FW_BASE ${CURRENT_TX8FW_DIR} CACHE PATH "TX8FW base")
  include_boilerplate("TX8FW repository")
  return()
endif()

if(IS_INCLUDED)
  # A higher level did the checking and included us and as we are not in TX8FW repository
  # (checked above) then we must be in TX8FW workspace.
  set(TX8FW_BASE ${CURRENT_TX8FW_DIR} CACHE PATH "TX8FW base")
  include_boilerplate("TX8FW workspace")
  return()
endif()

if(NOT IS_INCLUDED)
  string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${CURRENT_WORKSPACE_DIR}/" COMMON_INDEX)
  if (COMMON_INDEX EQUAL 0)
    # Project is in TX8FW workspace.
    # This means this TX8FW is likely the correct one, but there could be an alternative installed along-side
    # Thus, check if there is an even better candidate.
    # This check works the following way.
    # CMake finds packages will look all packages registered in the user package registry.
    # As this code is processed inside registered packages, we simply test if another package has a
    # common path with the current sample.
    # and if so, we will return here, and let CMake call into the other registered package for real
    # version checking.
    check_tx8fw_package(WORKSPACE_DIR ${CURRENT_WORKSPACE_DIR})

    if(TX8FW_PREFER)
      check_tx8fw_package(SEARCH_PARENTS CANDIDATES_PREFERENCE_LIST ${TX8FW_PREFER})
    endif()

    # We are the best candidate, so let's include boiler plate.
    set(TX8FW_BASE ${CURRENT_TX8FW_DIR} CACHE PATH "TX8FW base")
    include_boilerplate("TX8FW workspace")
    return()
  endif()

  check_tx8fw_package(SEARCH_PARENTS CANDIDATES_PREFERENCE_LIST ${TX8FW_PREFER})

  # Ending here means there were no candidates in workspace of the app.
  # Thus, the app is built as a TX8FW Freestanding application.
  # CMake find_package has already done the version checking, so let's just include boiler plate.
  # Previous find_package would have cleared TX8FW_FOUND variable, thus set it again.
  set(TX8FW_BASE ${CURRENT_TX8FW_DIR} CACHE PATH "TX8FW base")
  include_boilerplate("Freestanding")

endif()
