
set(relate_lib_cmake_file_path "lib/cmake/zephyr_interface/zephyr_interface-targets;lib/cmake/tx8fw_rtt/tx8fw-targets;lib/cmake/libc_stub/libc_stub-targets")
foreach (lib_cmake_file_path ${relate_lib_cmake_file_path})
 include(${CMAKE_CURRENT_LIST_DIR}/${lib_cmake_file_path}.cmake)
endforeach ()

set(relate_lib_cmake_file_path "")
while(relate_lib_cmake_file_path)
    list(POP_FRONT relate_lib_cmake_file_path CONFIG_CONDITION)
    list(POP_FRONT relate_lib_cmake_file_path lib_cmake_file_path)
    if(${${CONFIG_CONDITION}})
      include(${CMAKE_CURRENT_LIST_DIR}/${lib_cmake_file_path}.cmake)
    endif ()
endwhile()


 # "zephyr" is a catch-all CMake library for source files that can be
 # built purely with the include paths, defines, and other compiler
 # flags that come with zephyr_interface.
 zephyr_library_named(zephyr)

 zephyr_sources(${CMAKE_CURRENT_LIST_DIR}/misc/empty_file.c)

 set(relate_includes "include/build;include/bsp/xuantie_riscv_tx81/board_riscv_tx81/include;include/bsp/xuantie_riscv_tx81/chip_riscv_c908_series/include;include/bsp/xuantie_riscv_tx81/chip_riscv_c908_series/include/asm;include/rtthread/include;include/rtthread/libcpu/riscv/c908;include/components/gdbstub/.;include/components/gdbstub/libcpu/riscv/c908;include/components/libc_stub/include;include/components/libc_stub/compilers/gcc;include/components/libc/posix/libdl/include;include/components/csi/csi2/include;include/components/drivers/csi2/include;include/components/aos/include;include/components/cli/include;include/components/ulog/include;include/components/ulog/internal;include/components/oplib_tx81/riscv/riscv/include;include/components/common_tx81/riscv/dw_uart/include;include/components/devices/internal_inc;include/components/devices/include;include/components/posix/include;include/components/vfs/include;include/components/tx81_mhu2/include;include/components/tx81_profiling/include")

 set(zephyr_interface_includes )
 foreach (dir ${relate_includes})
  list(APPEND zephyr_interface_includes "${CMAKE_CURRENT_LIST_DIR}/${dir}")
 endforeach ()

 set_target_properties(zephyr_interface PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
 target_include_directories(zephyr_interface INTERFACE ${zephyr_interface_includes})
 target_link_libraries(zephyr   PUBLIC ${TX8FW_FIRMWARE_LIB})
 include(${CMAKE_CURRENT_LIST_DIR}/bsp_hook.cmake OPTIONAL)
