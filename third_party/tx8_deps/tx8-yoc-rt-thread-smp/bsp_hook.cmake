  include(${CMAKE_CURRENT_LIST_DIR}/bsp_option.cmake OPTIONAL)
    set(VALUE_FOR_CHECK_BSP xuantie_riscv_tx81)
    if(NOT ${BOARD} STREQUAL ${VALUE_FOR_CHECK_BSP})
     message(FATAL_ERROR "This Package is for bsp: ${VALUE_FOR_CHECK_BSP}")
    endif ()
  get_property(INTERFACE_LINK_OPTIONS_PROPERTY TARGET zephyr_interface PROPERTY INTERFACE_LINK_OPTIONS)
  set(key_world gcc_tx8_smarth.ld)
  foreach (item ${INTERFACE_LINK_OPTIONS_PROPERTY})
   string(FIND item "${key_world}" script_found)
   if (NOT ${script_found} EQUAL -1)
      list(REMOVE_ITEM item INTERFACE_LINK_OPTIONS_PROPERTY)
      break()
   endif ()
  endforeach ()
if(DEFINED ENV{CONFIG_TX81_MEMORY_LAYOUT})
    zephyr_compile_definitions("CONFIG_TX81_MEMORY_LAYOUT=$ENV{CONFIG_TX81_MEMORY_LAYOUT}")
endif()

if(DEFINED ENV{RT_USING_GDB} AND "$ENV{RT_USING_GDB}" STREQUAL "1")
    zephyr_compile_definitions(RT_USING_GDB)
    zephyr_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:${KCORE_GDB_CFLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${KCORE_GDB_CFLAGS}>"
        "$<$<COMPILE_LANGUAGE:ASM>:${KCORE_GDB_CFLAGS}>"
    )
endif()
if (DEFINED ENV{RT_GDB_INTTER} AND "$ENV{RT_GDB_INTTER}" STREQUAL "1")
  zephyr_compile_definitions(RT_GDB_INTTER)
endif()

 set_property(TARGET linker PROPERTY TX8_LINK_FILE_PATH "${CMAKE_CURRENT_LIST_DIR}/${key_world}")
 set_target_properties(zephyr_interface PROPERTIES INTERFACE_LINK_OPTIONS "")
 target_link_options(zephyr_interface INTERFACE ${INTERFACE_LINK_OPTIONS_PROPERTY})
