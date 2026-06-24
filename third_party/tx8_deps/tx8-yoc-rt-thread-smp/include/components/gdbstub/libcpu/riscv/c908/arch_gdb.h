/*
 * risc-v GDB support
 * arch-specific portion of GDB stub
 * 
 * File      : arch_gdb.h(risc-v)
 * This file is part of RT-Thread RTOS
 * COPYRIGHT (C) 2006, RT-Thread Develop Team
 *
 * The license and distribution terms for this file may be
 * found in the file LICENSE in this distribution or at
 * http://www.rt-thread.org/license/LICENSE
 *
 * Change Logs:
 * Date           Author       Notes
 * 2024-12-11     tangjb      first version
 */
#ifndef __RISC_V_GDB_H__
#define __RISC_V_GDB_H__ 

#include <rtthread.h>

#ifndef RT_GDB_HAVE_HWBP
    #define RT_GDB_HAVE_HWBP 0
#endif
#ifndef RT_GDB_HAVE_SWBP
    #define RT_GDB_HAVE_SWBP 1
#endif


#if RT_GDB_HAVE_HWBP
    #error GDB:No hardware_breakpoint support
#endif

#define CONFIG_RISCV_ISA_C
/*
 * By doing this as an undefined instruction trap, we force a mode
 * switch from SVC to UND mode, allowing us to save full kernel state.
 * We also define a GDB_COMPILED_BREAK which can be used to compile
 * in breakpoints. 
 */
#ifdef CONFIG_RISCV_ISA_C
#define BREAK_INSTR_SIZE	2
#else
#define BREAK_INSTR_SIZE	4
#endif

#define CACHE_FLUSH_IS_SAFE	1

#define RT_GDB_ICACHE

#define DBG_MAX_REG_NUM  33

#define BUFMAX			2048

#define GDB_SIZEOF_REG sizeof(unsigned long)

#define NUMREGBYTES ((DBG_MAX_REG_NUM) * GDB_SIZEOF_REG)

//#ifndef __ASSEMBLY__

//#endif

struct dbg_reg_def_t {
	char *name;
	int size;
	int offset;
};

#undef offsetof
#define offsetof(TYPE, MEMBER)	__builtin_offsetof(TYPE, MEMBER)

#define DBG_REG_ZERO "zero"
#define DBG_REG_RA "ra"
#define DBG_REG_SP "sp"
#define DBG_REG_GP "gp"
#define DBG_REG_TP "tp"
#define DBG_REG_T0 "t0"
#define DBG_REG_T1 "t1"
#define DBG_REG_T2 "t2"
#define DBG_REG_FP "fp"
#define DBG_REG_S1 "s1"
#define DBG_REG_A0 "a0"
#define DBG_REG_A1 "a1"
#define DBG_REG_A2 "a2"
#define DBG_REG_A3 "a3"
#define DBG_REG_A4 "a4"
#define DBG_REG_A5 "a5"
#define DBG_REG_A6 "a6"
#define DBG_REG_A7 "a7"
#define DBG_REG_S2 "s2"
#define DBG_REG_S3 "s3"
#define DBG_REG_S4 "s4"
#define DBG_REG_S5 "s5"
#define DBG_REG_S6 "s6"
#define DBG_REG_S7 "s7"
#define DBG_REG_S8 "s8"
#define DBG_REG_S9 "s9"
#define DBG_REG_S10 "s10"
#define DBG_REG_S11 "s11"
#define DBG_REG_T3 "t3"
#define DBG_REG_T4 "t4"
#define DBG_REG_T5 "t5"
#define DBG_REG_T6 "t6"
#define DBG_REG_EPC "pc"
#define DBG_REG_STATUS "sstatus"
#define DBG_REG_BADADDR "stval"
#define DBG_REG_CAUSE "scause"

#define DBG_REG_ZERO_OFF 0
#define DBG_REG_RA_OFF 1
#define DBG_REG_SP_OFF 2
#define DBG_REG_GP_OFF 3
#define DBG_REG_TP_OFF 4
#define DBG_REG_T0_OFF 5
#define DBG_REG_T1_OFF 6
#define DBG_REG_T2_OFF 7
#define DBG_REG_FP_OFF 8
#define DBG_REG_S1_OFF 9
#define DBG_REG_A0_OFF 10
#define DBG_REG_A1_OFF 11
#define DBG_REG_A2_OFF 12
#define DBG_REG_A3_OFF 13
#define DBG_REG_A4_OFF 14
#define DBG_REG_A5_OFF 15
#define DBG_REG_A6_OFF 16
#define DBG_REG_A7_OFF 17
#define DBG_REG_S2_OFF 18
#define DBG_REG_S3_OFF 19
#define DBG_REG_S4_OFF 20
#define DBG_REG_S5_OFF 21
#define DBG_REG_S6_OFF 22
#define DBG_REG_S7_OFF 23
#define DBG_REG_S8_OFF 24
#define DBG_REG_S9_OFF 25
#define DBG_REG_S10_OFF 26
#define DBG_REG_S11_OFF 27
#define DBG_REG_T3_OFF 28
#define DBG_REG_T4_OFF 29
#define DBG_REG_T5_OFF 30
#define DBG_REG_T6_OFF 31
#define DBG_REG_EPC_OFF 32
#define DBG_REG_STATUS_OFF 33
#define DBG_REG_BADADDR_OFF 34
#define DBG_REG_CAUSE_OFF 35

/* arch */
extern struct gdb_arch		arch_gdb_ops;
void gdb_breakpoint();
void gdb_get_register(unsigned long *gdb_regs);
void gdb_get_regbythrd(unsigned long *gdb_regs, void* sp);
void gdb_put_register(unsigned long *gdb_regs);
void gdb_set_register(void *hw_regs);
int gdb_arch_handle_exception(char *remcom_in_buffer,
                              char *remcom_out_buffer);
void gdb_flush_icache_range(unsigned long start, unsigned long end);
int gdb_undef_hook(void *regs);

int gdb_handle_exception(int signo, void *regs);

#endif /* __RISC_V_GDB_H__ */
