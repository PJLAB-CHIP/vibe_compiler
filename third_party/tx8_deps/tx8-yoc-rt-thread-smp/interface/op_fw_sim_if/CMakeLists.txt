cmake_minimum_required(VERSION 3.20.0)
project(op_fw_sim_if C CXX ASM)
if(USING_RISCV)

else ()
    add_library(op_fw_interface INTERFACE)
    target_compile_options(op_fw_interface INTERFACE  -fPIC)
endif ()

add_subdirectory(peripheral)
