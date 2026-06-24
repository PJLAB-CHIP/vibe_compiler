
 set(relate_includes "include/build;include/bsp/xuantie_riscv_tx81/board_riscv_tx81/include;include/bsp/xuantie_riscv_tx81/chip_riscv_c908_series/include;include/bsp/xuantie_riscv_tx81/chip_riscv_c908_series/include/asm;include/rtthread/include;include/rtthread/libcpu/riscv/c908;include/components/gdbstub/.;include/components/gdbstub/libcpu/riscv/c908;include/components/libc_stub/include;include/components/libc_stub/compilers/gcc;include/components/libc/posix/libdl/include;include/components/csi/csi2/include;include/components/drivers/csi2/include;include/components/aos/include;include/components/cli/include;include/components/ulog/include;include/components/ulog/internal;include/components/oplib_tx81/riscv/riscv/include;include/components/common_tx81/riscv/dw_uart/include;include/components/devices/internal_inc;include/components/devices/include;include/components/posix/include;include/components/vfs/include;include/components/tx81_mhu2/include;include/components/tx81_profiling/include")

 set(temp_interface_includes )
 foreach (dir ${relate_includes})
  list(APPEND temp_interface_includes "${CMAKE_CURRENT_LIST_DIR}/${dir}")
 endforeach ()

 # Create imported target tx8fw_include_interface
 add_library(tx8fw_include_interface INTERFACE IMPORTED)

 set(TX8FW_COMPILE_DEFINITIONS_PROPERTY BUILD_VERSION_DESCRIBE_STRING="";CONFIG_ARCH_MAINSTACK=4096;CONFIG_ARCH_INTERRUPTSTACK=4096;CONFIG_SYSTICK_HZ=100;CONFIG_BOARD_SMARTH_EVB=1;CONFIG_BOARD_SMARTH_EVB_FOR_TX81=1;CONFIG_PLIC_BASE=0x10000000;CONFIG_SMP=1;CONFIG_NR_CPUS=2;CONFIG_KERNEL_RTTHREAD=1;CONFIG_CSI_V2=1;CONFIG_CSI=csi2;CONFIG_SUPPORT_TSPEND=1;CONFIG_CPU_C908=1;CONFIG_CPU_XUANTIE_C908=1;CONFIG_TX8_KERNEL_PRINTF_SUPPORT=1;USING_RISCV=1;CONFIG_ECC_L1_ENABLE=1;CONFIG_ECC_L2_ENABLE=1;__PLATFORM_RTT__=1;AOS_COMP_POSIX=1;CONFIG_INIT_TASK_STACK_SIZE=65536)
 target_include_directories(tx8fw_include_interface INTERFACE ${temp_interface_includes})
 target_compile_definitions(tx8fw_include_interface INTERFACE ${TX8FW_COMPILE_DEFINITIONS_PROPERTY})
