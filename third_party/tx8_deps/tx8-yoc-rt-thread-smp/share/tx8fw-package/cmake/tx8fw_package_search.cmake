# The purpose of this file is to provide search mechanism for locating TX8FW in-work-tree package
# even when they are not installed into CMake package system
# Linux/MacOS: ~/.cmake/packages
# Windows:     Registry database

# Relative directory of workspace project dir as seen from TX8FW package file
set(WORKSPACE_RELATIVE_DIR "../../../../..")

# Relative directory of TX8FW dir as seen from TX8FW package file
set(ZEPHYR_RELATIVE_DIR "../../../..")

# This function updates TX8FW_DIR to the point to the candidate dir.
# For TX8FW 3.0 and earlier, the TX8FW_DIR might in some cases be
# `TX8FW_DIR-NOTFOUND` or pointing to the TX8FW package including the
# boilerplate code instead of the TX8FW package of the included boilerplate.
# This code ensures that when TX8FW releases <=3.0 is loaded, then TX8FW_DIR
# will point correctly, see also #43094 which relates to this.
function(set_tx8fw_dir tx8fw_candidate)
  get_filename_component(tx8fw_candidate_dir "${tx8fw_candidate}" DIRECTORY)
  if(NOT "${tx8fw_candidate_dir}" STREQUAL "${TX8FW_DIR}")
    set(TX8FW_DIR ${tx8fw_candidate_dir} CACHE PATH
        "The directory containing a CMake configuration file for TX8FW." FORCE
    )
  endif()
endfunction()

# This macro returns a list of parent folders to use for later searches.
macro(get_search_paths START_PATH SEARCH_PATHS PREFERENCE_LIST)
  get_filename_component(SEARCH_PATH ${START_PATH} DIRECTORY)
  while(NOT (SEARCH_PATH STREQUAL SEARCH_PATH_PREV))
    foreach(preference ${PREFERENCE_LIST})
      list(APPEND SEARCH_PATHS ${SEARCH_PATH}/${preference})
    endforeach()
    list(APPEND SEARCH_PATHS ${SEARCH_PATH}/tx8fw)
    list(APPEND SEARCH_PATHS ${SEARCH_PATH})
    set(SEARCH_PATH_PREV ${SEARCH_PATH})
    get_filename_component(SEARCH_PATH ${SEARCH_PATH} DIRECTORY)
  endwhile()
endmacro()

# This macro can check for additional TX8FW package that has a better match
# Options:
# - TX8FW_BASE                : Use the specified TX8FW_BASE directly.
# - WORKSPACE_DIR              : Search for projects in specified  workspace.
# - SEARCH_PARENTS             : Search parent folder of current source file (application)
#                                to locate in-project-tree TX8FW candidates.
# - CHECK_ONLY                 : Only set PACKAGE_VERSION_COMPATIBLE to false if a better candidate
#                                is found, default is to also include the found candidate.
# - VERSION_CHECK              : This is the version check stage by CMake find package
# - CANDIDATES_PREFERENCE_LIST : List of candidate to be preferred, if installed
macro(check_tx8fw_package)
  set(options CHECK_ONLY SEARCH_PARENTS VERSION_CHECK)
  set(single_args WORKSPACE_DIR TX8FW_BASE)
  set(list_args CANDIDATES_PREFERENCE_LIST)
  cmake_parse_arguments(CHECK_PKG_PREFIX "${options}" "${single_args}" "${list_args}" ${ARGN})

  if(CHECK_PKG_PREFIX_ZEPHYR_BASE)
    set(SEARCH_SETTINGS PATHS ${CHECK_PKG_PREFIX_ZEPHYR_BASE} NO_DEFAULT_PATH)
  endif()

  if(CHECK_PKG_PREFIX_WORKSPACE_DIR)
    set(SEARCH_SETTINGS PATHS ${CHECK_PKG_PREFIX_WORKSPACE_DIR}/tx8fw ${CHECK_PKG_PREFIX_WORKSPACE_DIR} NO_DEFAULT_PATH)
  endif()

  if(CHECK_PKG_PREFIX_SEARCH_PARENTS)
    get_search_paths(${CMAKE_CURRENT_SOURCE_DIR} SEARCH_PATHS "${CHECK_PKG_PREFIX_CANDIDATES_PREFERENCE_LIST}")
    set(SEARCH_SETTINGS PATHS ${SEARCH_PATHS} NO_DEFAULT_PATH)
  endif()

  # Searching for version zero means there will be no match, but we obtain
  # a list of all potential TX8FW candidates in the tree to consider.

  find_package(TX8FW 0.0.0 EXACT QUIET ${SEARCH_SETTINGS}) # 在${SEARCH_SETTINGS} 中查找

  # The find package will also find ourself when searching using installed candidates.
  # So avoid re-including unless NO_DEFAULT_PATH is set.
  # NO_DEFAULT_PATH means explicit search and we could be part of a preference list.
  if(NOT (NO_DEFAULT_PATH IN_LIST SEARCH_SETTINGS))
    list(REMOVE_ITEM TX8FW_CONSIDERED_CONFIGS ${CMAKE_CURRENT_LIST_DIR}/ZephyrConfig.cmake)
  endif()
  list(REMOVE_DUPLICATES TX8FW_CONSIDERED_CONFIGS)

  #TX8FW_CONSIDERED_CONFIGS 是 cmake 查找过程找到的的候选

  foreach(TX8FW_CANDIDATE ${TX8FW_CONSIDERED_CONFIGS})
    if(CHECK_PKG_PREFIX_WORKSPACE_DIR)
      # Check is done in TX8FW workspace already, thus check only for pure TX8FW candidates.
      get_filename_component(CANDIDATE_DIR ${TX8FW_CANDIDATE}/${ZEPHYR_RELATIVE_DIR} ABSOLUTE)
    else()
      get_filename_component(CANDIDATE_DIR ${TX8FW_CANDIDATE}/${WORKSPACE_RELATIVE_DIR} ABSOLUTE)
    endif()

    if(CHECK_PKG_PREFIX_ZEPHYR_BASE)
        if(CHECK_PKG_PREFIX_VERSION_CHECK)
          string(REGEX REPLACE "\.cmake$" "Version.cmake" TX8FW_VERSION_CANDIDATE ${TX8FW_CANDIDATE})
          #不用 TX8FW_CONSIDERED_VERSIONS,估计是为了代码简练
          include(${TX8FW_VERSION_CANDIDATE} NO_POLICY_SCOPE)
          return()
        else()
          include(${TX8FW_CANDIDATE} NO_POLICY_SCOPE)
          set_tx8fw_dir(${TX8FW_CANDIDATE})
          return()
        endif()
    endif()

    string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${CANDIDATE_DIR}/" COMMON_INDEX)
    if (COMMON_INDEX EQUAL 0)
      #这行代码的目的是检查${CANDIDATE_DIR}/是否是${CMAKE_CURRENT_SOURCE_DIR}的一个子目录
      if(CHECK_PKG_PREFIX_CHECK_ONLY)
        # A better candidate exists, thus return
        set(PACKAGE_VERSION_COMPATIBLE FALSE)
        return()
      elseif(TX8FW_CANDIDATE STREQUAL ${CMAKE_CURRENT_LIST_DIR}/ZephyrConfig.cmake)
        # Current TX8FW is preferred one, let's just break the loop and continue processing.
        break()
      else()
        if(CHECK_PKG_PREFIX_VERSION_CHECK)
          string(REGEX REPLACE "\.cmake$" "Version.cmake" TX8FW_VERSION_CANDIDATE ${TX8FW_CANDIDATE})
          include(${TX8FW_VERSION_CANDIDATE} NO_POLICY_SCOPE)
          return()
        else()
          #如果执行的是check_tx8fw_package(WORKSPACE_DIR ${CURRENT_WORKSPACE_DIR}) 那么这里不会被执行到
          #因为 假设Project的名称是my_first_app, 那它在tx8fw-workspace中
          #<projects>/tx8fw-workspace
          #  ├── tx8fw
          #  ├── ...
          #  └── my_applications
          #         └── my_first_app
          #             └── CMakeLists.txt
          #
          include(${TX8FW_CANDIDATE} NO_POLICY_SCOPE)
          set_tx8fw_dir(${TX8FW_CANDIDATE})
	      return()
        endif()
      endif()
    endif()
  endforeach()
endmacro()
