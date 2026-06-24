# SPDX-License-Identifier: Apache-2.0

# This file provides TX8FW Config Package version information.
#
# The purpose of the version file is to ensure that CMake find_package can correctly locate a
# usable TX8FW installation for building of applications.

# Checking for version 0.0.0 is a way to allow other TX8FW installation to determine if there is a better match.
# A better match would be an installed TX8FW that has a common index with current source dir.
# Version 0.0.0 indicates that we should just return, in order to obtain our path.
if(0.0.0 STREQUAL PACKAGE_FIND_VERSION)
  return()
endif()

include( ${CMAKE_CURRENT_LIST_DIR}/pre_verify.cmake)
if(NOT TX8FW_PRODUCT_SERIES_FOUND)
  return()
endif ()
unset(TX8FW_PRODUCT_SERIES_FOUND)
unset(TX8FW_PRODUCT_SERIES)

macro(check_tx8fw_version)
  if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    if(IS_INCLUDED)
      # We are just a candidate, meaning we have been included from other installed module.
      message("\n  The following TX8FW repository configuration file were considered but not accepted:")
      message("\n    ${CMAKE_CURRENT_LIST_FILE}, version: ${PACKAGE_VERSION}\n")
    endif()

    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  else()
    # For now, TX8FW is capable to find the right base on all older versions as long as they define
    # a TX8FW config package (This code)
    # In future, this is the place to update in case TX8FW 3.x is not backward compatible with version 2.x
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
      set(PACKAGE_VERSION_EXACT TRUE)
    endif()
  endif()
endmacro()

# First check to see if user has provided a TX8FW base manually and it is first run (cache not set).
set(ENV_TX8FW_BASE $ENV{TX8FW_BASE})
if((NOT DEFINED TX8FW_BASE) AND (DEFINED ENV_TX8FW_BASE))
  # Get rid of any double folder string before comparison, as example, user provides
  # TX8FW_BASE=//path/to//tx8fw_base/
  # must also work.
  get_filename_component(TX8FW_BASE $ENV{TX8FW_BASE} ABSOLUTE)
endif()

# If TX8FW_CANDIDATE is set, it means this file was include instead of called via find_package directly.
if(TX8FW_CANDIDATE)
  set(IS_INCLUDED TRUE)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/tx8fw_package_search.cmake)
endif()

if((DEFINED TX8FW_BASE) OR (DEFINED ENV_TX8FW_BASE))
  # TX8FW_BASE was set in cache from earlier run or in environment (first run),
  # meaning the package version must be ignored and the TX8FW pointed to by
  # TX8FW_BASE is to be used regardless of version.
  if (${TX8FW_BASE}/share/tx8fw-package/cmake STREQUAL ${CMAKE_CURRENT_LIST_DIR})
    # We are the TX8FW to be used

    set(NO_PRINT_VERSION True)
    include(${TX8FW_BASE}/cmake/modules/version.cmake)
    # TX8FW uses project version, but CMake package uses PACKAGE_VERSION
    set(PACKAGE_VERSION ${PROJECT_VERSION})
    check_tx8fw_version()

    if(IS_INCLUDED)
      # We are included, so we need to ensure that the version of the top-level
      # package file is returned. This TX8FW version has already been printed
      # as part of `check_tx8fw_version()`
      if(NOT ${PACKAGE_VERSION_COMPATIBLE}
        OR (TX8FW_FIND_VERSION_EXACT AND NOT PACKAGE_VERSION_EXACT)
      )
        # When TX8FW base is set and we are checked as an included file
        # (IS_INCLUDED=True), then we are unable to retrieve the version of the
        # parent TX8FW, therefore just mark it as ignored.
        set(PACKAGE_VERSION "ignored (TX8FW_BASE is set)")
      endif()
    endif()
  elseif ((NOT IS_INCLUDED) AND (DEFINED TX8FW_BASE))
    check_tx8fw_package(TX8FW_BASE ${TX8FW_BASE} VERSION_CHECK)
  else()
    # User has pointed to a different TX8FW installation, so don't use this version
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  endif()
  return()
endif()

# Find out the current TX8FW base.
get_filename_component(CURRENT_TX8FW_DIR ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
get_filename_component(CURRENT_WORKSPACE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../.. ABSOLUTE)

# Temporary set local TX8FW base to allow using version.cmake to find this TX8FW repository current version
set(TX8FW_BASE ${CURRENT_TX8FW_DIR})

# Tell version.cmake to not print as printing version for all TX8FW installations being tested
# will lead to confusion on which is being used.
set(NO_PRINT_VERSION True)
include(${TX8FW_BASE}/cmake/modules/version.cmake)
# TX8FW uses project version, but CMake package uses PACKAGE_VERSION
set(PACKAGE_VERSION ${PROJECT_VERSION})
set(TX8FW_BASE)

# Do we share common index, if so, this is the correct version to check.
string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${CURRENT_TX8FW_DIR}/" COMMON_INDEX)
if (COMMON_INDEX EQUAL 0)
  # Project is a TX8FW repository app.

  check_tx8fw_version()
  return()
endif()

if(NOT IS_INCLUDED)
  # Only do this if we are an installed CMake Config package and checking for workspace candidates.

  string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${CURRENT_WORKSPACE_DIR}/" COMMON_INDEX)
  if (COMMON_INDEX EQUAL 0)
    # Project is a TX8FW workspace app.
    # This means this TX8FW is likely the correct one, but there could be an alternative installed along-side
    # Thus, check if there is an even better candidate.
    check_tx8fw_package(WORKSPACE_DIR ${CURRENT_WORKSPACE_DIR} VERSION_CHECK)

    # We are the best candidate, so let's check our own version.
    check_tx8fw_version()
    return()
  endif()

  # Checking for installed candidates which could also be an workspace candidates.
  # This check works the following way.
  # CMake finds packages will look all packages registered in the user package registry.
  # As this code is processed inside registered packages, we simply test if
  # another package has a common path with the current sample, and if so, we
  # will return here, and let CMake call into the other registered package for
  # real version checking.
  check_tx8fw_package(CHECK_ONLY VERSION_CHECK)  #CHECK_ONLY 这个就是检查

  # Check for workspace candidates.
  check_tx8fw_package(SEARCH_PARENTS VERSION_CHECK)
endif()

# Ending here means there were no candidates in workspace of the app.
# Thus, the app is built as a TX8FW Freestanding application.
# Let's do basic CMake version checking.
check_tx8fw_version()
