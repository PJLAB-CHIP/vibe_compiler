# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2021, Nordic Semiconductor ASA

# Validate board and setup boards target.
#
# This CMake module will validate the BOARD argument as well as splitting the
# BOARD argument into <BOARD> and <BOARD_REVISION>. When BOARD_EXTENSIONS option
# is enabled (default) this module will also take care of finding board
# extension directories.
#
# If a board implementation is not found for the specified board an error will
# be raised and list of valid boards will be printed.
#
# If user provided board is a board alias, the board will be adjusted to real
# board name.
#
# If board name is deprecated, then board will be adjusted to new board name and
# a deprecation warning will be printed to the user.
#
# Outcome:
# The following variables will be defined when this CMake module completes:
#
# - BOARD:                       Board, without revision field.
# - BOARD_REVISION:              Board revision
# - BOARD_QUALIFIERS:            Board qualifiers
# - NORMALIZED_BOARD_QUALIFIERS: Board qualifiers in lower-case format where slashes have been
#                                replaced with underscores
# - NORMALIZED_BOARD_TARGET:     Board target in lower-case format where slashes have been
#                                replaced with underscores
# - BOARD_DIR:                   Board directory with the implementation for selected board
# - BOARD_ROOT:                  BOARD_ROOT with ZEPHYR_BASE appended
# - BOARD_EXTENSION_DIRS:        List of board extension directories (If
#                                BOARD_EXTENSIONS is not explicitly disabled)
#
# The following targets will be defined when this CMake module completes:
# - board: when invoked, a list of valid boards will be printed
#
# Required variables:
# - BOARD: Board name, including any optional revision field, for example: `foo` or `foo@1.0.0`
#
# Optional variables:
# - BOARD_ROOT: CMake list of board roots containing board implementations

# Optional environment variables:
# - TX8FW_BOARD_ALIASES: Environment setting pointing to a CMake file
#                         containing board aliases.
#
# Variables set by this module and not mentioned above are for internal
# use only, and may be removed, renamed, or re-purposed without prior notice.

include_guard(GLOBAL)

include(python)
include(extensions)

# Check that BOARD has been provided, and that it has not changed.
# If user tries to change the BOARD, the BOARD value is reset to the BOARD_CACHED value.
zephyr_check_cache(BOARD REQUIRED)

if(NOT DEFINED BOARD)
  SET(BOARD "xuantie_riscv_tx81/kernel")
  message(NOTICE "BOARD is not set. then set  default value : ${BOARD} to BOARD")
endif ()
if(NOT DEFINED BOARD_ROOT)
  SET(BOARD_ROOT ${TX8FW_BASE}/bsp)
  message(NOTICE "BOARD_ROOT is not set. then set  default value : ${BOARD_ROOT}")
endif ()

if(NOT DEFINED BOARD_DIR)
  SET(BOARD_DIR ${BOARD_ROOT}/${BOARD})
  message(NOTICE "BOARD_DIR is not set. then set  default value : ${BOARD_DIR}")
endif ()


# 'BOARD_ROOT' is a prioritized list of directories where boards may
# be found. It always includes ${ZEPHYR_BASE} at the lowest priority (except for unittesting).
if(NOT unittest IN_LIST TX8FW_FIND_COMPONENTS)
  list(APPEND BOARD_ROOT ${TX8FW_BASE})
endif()

# Helper function for parsing a board's name, revision, and qualifiers,
# from one input variable to three separate output variables.
function(parse_board_components board_in name_out revision_out qualifiers_out)
  if(NOT "${${board_in}}" MATCHES "^([^@/]+)(@[^@/]+)?(/[^@]+)?$")
    message(FATAL_ERROR
      "Invalid revision / qualifiers format for ${board_in} (${${board_in}}). "
      "Valid format is: <board>@<revision>/<qualifiers>"
    )
  endif()
  string(REPLACE "@" "" board_revision "${CMAKE_MATCH_2}")

  set(${name_out}       ${CMAKE_MATCH_1}  PARENT_SCOPE)
  set(${revision_out}   ${board_revision} PARENT_SCOPE)
  set(${qualifiers_out} ${CMAKE_MATCH_3}  PARENT_SCOPE)
endfunction()

parse_board_components(
  BOARD
  BOARD BOARD_REVISION BOARD_QUALIFIERS
)

zephyr_get(TX8FW_BOARD_ALIASES)
if(DEFINED TX8FW_BOARD_ALIASES)
  include(${TX8FW_BOARD_ALIASES})
  if(${BOARD}_BOARD_ALIAS)
    set(BOARD_ALIAS ${BOARD} CACHE STRING "Board alias, provided by user")
    parse_board_components(
      ${BOARD}_BOARD_ALIAS
      BOARD BOARD_ALIAS_REVISION BOARD_ALIAS_QUALIFIERS
    )
    message(STATUS "Aliased BOARD=${BOARD_ALIAS} changed to ${BOARD}")
    if(NOT DEFINED BOARD_REVISION)
      set(BOARD_REVISION ${BOARD_ALIAS_REVISION})
    endif()
    set(BOARD_QUALIFIERS ${BOARD_ALIAS_QUALIFIERS}${BOARD_QUALIFIERS})
  endif()
endif()

include(${TX8FW_BASE}/bsp/deprecated.cmake OPTIONAL)
if(${BOARD}${BOARD_QUALIFIERS}_DEPRECATED)
  set(BOARD_DEPRECATED ${BOARD}${BOARD_QUALIFIERS} CACHE STRING "Deprecated BOARD, provided by user")
  message(WARNING
    "Deprecated BOARD=${BOARD_DEPRECATED} specified, "
    "board automatically changed to: ${${BOARD}${BOARD_QUALIFIERS}_DEPRECATED}."
  )
  parse_board_components(
    ${BOARD}${BOARD_QUALIFIERS}_DEPRECATED
    BOARD BOARD_DEPRECATED_REVISION BOARD_QUALIFIERS
  )
  if(DEFINED BOARD_DEPRECATED_REVISION)
    if(DEFINED BOARD_REVISION)
      message(FATAL_ERROR
        "Invalid board revision: ${BOARD_REVISION}\n"
        "Deprecated board '${BOARD_DEPRECATED}' is now implemented as a revision of another board "
        "(${BOARD}@${BOARD_DEPRECATED_REVISION}), so the specified revision does not apply. "
        "Please consult the documentation for '${BOARD}' to see how to build for the new board."
      )
    endif()
    set(BOARD_REVISION ${BOARD_DEPRECATED_REVISION})
  endif()
endif()

zephyr_boilerplate_watch(BOARD)


cmake_path(IS_PREFIX TX8FW_BASE "${BOARD_DIR}" NORMALIZE in_zephyr_tree)
if(NOT in_zephyr_tree)
  set(USING_OUT_OF_TREE_BOARD 1)
endif()

set(board_message "Bsp: ${BOARD}")

if(DEFINED BOARD_REVISION)
  set(board_message "${board_message}, Revision: ${BOARD_REVISION}")
  if(DEFINED ACTIVE_BOARD_REVISION)
    set(board_message "${board_message} (Active: ${ACTIVE_BOARD_REVISION})")
    set(BOARD_REVISION ${ACTIVE_BOARD_REVISION})
  endif()

  string(REPLACE "." "_" BOARD_REVISION_STRING ${BOARD_REVISION})
endif()

if(DEFINED BOARD_QUALIFIERS)
  string(REGEX REPLACE "^/" "qualifiers: " board_message_qualifiers "${BOARD_QUALIFIERS}")
  set(board_message "${board_message}, ${board_message_qualifiers}")

  string(REPLACE "/" "_" NORMALIZED_BOARD_QUALIFIERS "${BOARD_QUALIFIERS}")
endif()

set(NORMALIZED_BOARD_TARGET "${BOARD}${BOARD_QUALIFIERS}")
string(REPLACE "/" "_" NORMALIZED_BOARD_TARGET "${NORMALIZED_BOARD_TARGET}")

message(STATUS "${board_message}")


# Board extensions are enabled by default
set(BOARD_EXTENSIONS ON CACHE BOOL "Support board extensions")
zephyr_get(BOARD_EXTENSIONS)

# Process board extensions
if(BOARD_EXTENSIONS)
  get_filename_component(board_dir_name ${BOARD_DIR} NAME)

  foreach(root ${BOARD_ROOT})
    set(board_extension_dir ${root}/boards/extensions/${board_dir_name})
    if(NOT EXISTS ${board_extension_dir})
      continue()
    endif()

    list(APPEND BOARD_EXTENSION_DIRS ${board_extension_dir})
  endforeach()
endif()

if(EXISTS ${BOARD_DIR}/bsp_base_env.cmake)
  include(${BOARD_DIR}/bsp_base_env.cmake )
endif ()
if(NOT DEFINED TARGET_CPU)
  if(NOT EXISTS ${BOARD_DIR}/bsp_base_env.cmake)
    message(NOTICE "${BOARD_DIR}/bsp_base_env.cmake did not exists.")
  endif ()
   message(FATAL_ERROR "TARGET_CPU is not defined")
endif ()
if(NOT DEFINED ARCH)
  if(NOT EXISTS ${BOARD_DIR}/bsp_base_env.cmake)
    message(NOTICE "${BOARD_DIR}/bsp_base_env.cmake did not exists.")
  endif ()
  message(FATAL_ERROR "ARCH is not defined")
endif ()