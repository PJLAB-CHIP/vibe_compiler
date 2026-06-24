
macro(find_sub_dir_list result cur_dir)
    FILE(GLOB children RELATIVE ${cur_dir} ${cur_dir}/*)
    SET(dirlist "")
    foreach(child ${children})
        if (IS_DIRECTORY ${cur_dir}/${child})
            LIST(APPEND dirlist ${child})
        endif()
    endforeach()
    SET(${result} ${dirlist})
endmacro()

# 定义一个宏来将列表合并成字符串
macro(list_to_string LIST_VAR OUTPUT_VAR)
    set(${OUTPUT_VAR})  # 初始化输出变量为空
    foreach(item IN LISTS ${LIST_VAR})
        set(${OUTPUT_VAR} "${${OUTPUT_VAR}} ${item}")
    endforeach()
    # 去除字符串开头的多余空格
    string(STRIP "${${OUTPUT_VAR}}" ${OUTPUT_VAR})
endmacro()

# This module provides function for joining paths
# known from from most languages
#
# Original license:
# SPDX-License-Identifier: (MIT OR CC0-1.0)
# Explicit permission given to distribute this module under
# the terms of the project as described in /LICENSE.rst.
# Copyright 2020 Jan Tojnar
# https://github.com/jtojnar/cmake-snips
#
# Modelled after Python’s os.path.join
# https://docs.python.org/3.7/library/os.path.html#os.path.join
# Windows not supported
function(join_paths joined_path first_path_segment)
    set(temp_path "${first_path_segment}")
    foreach(current_segment IN LISTS ARGN)
        if(NOT ("${current_segment}" STREQUAL ""))
            if(IS_ABSOLUTE "${current_segment}")
                set(temp_path "${current_segment}")
            else()
                set(temp_path "${temp_path}/${current_segment}")
            endif()
        endif()
    endforeach()
    set(${joined_path} "${temp_path}" PARENT_SCOPE)
endfunction()
# Joins arguments and places the results in ${result_var}.
function(join result_var)
    set(result "")
    foreach (arg ${ARGN})
        set(result "${result}${arg}")
    endforeach ()
    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()
# Sets a cache variable with a docstring joined from multiple arguments:
#   set(<variable> <value>... CACHE <type> <docstring>...)
# This allows splitting a long docstring for readability.
function(set_verbose)
    # cmake_parse_arguments is broken in CMake 3.4 (cannot parse CACHE) so use
    # list instead.
    list(GET ARGN 0 var)
    list(REMOVE_AT ARGN 0)
    list(GET ARGN 0 val)
    list(REMOVE_AT ARGN 0)
    list(REMOVE_AT ARGN 0)
    list(GET ARGN 0 type)
    list(REMOVE_AT ARGN 0)
    join(doc ${ARGN})
    set(${var} ${val} CACHE ${type} ${doc})
endfunction()
function(delete_dir_then_create_dir dir_path)
    file(REMOVE_RECURSE         ${dir_path})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${dir_path})
endfunction()