/*
 * Copyright (c) 2006-2024, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2022-12-08     RT-Thread    first version
 */

#include <rthw.h>
#include <rtthread.h>
#include <stdint.h>

#include "riscv_common.h"
#include "rt_interrupt.h"
#include "stack.h"
#include <csi_core.h>

#ifdef RT_USING_GDB
#include <gdb_stub.h>
#endif

#include <lib_log.h>
#ifdef rt_kprintf
#undef rt_kprintf
#endif
#define rt_kprintf tx8_kernel_printf

void dump_regs(struct rt_hw_stack_frame *regs)
{
    __EP_LOG__(KCORE_LOG_FATAL,"--------------Dump Registers-----------------\n");

    __EP_LOG__(KCORE_LOG_FATAL,"Function Registers:\n");
    __EP_LOG__(KCORE_LOG_FATAL,"\tra(x1) = %p\tuser_sp = %p\n", regs->ra, regs->x2);
    __EP_LOG__(KCORE_LOG_FATAL,"\tgp(x3) = %p\ttp(x4) = %p\n", regs->gp, regs->tp);
    __EP_LOG__(KCORE_LOG_FATAL,"Temporary Registers:\n");
    __EP_LOG__(KCORE_LOG_FATAL,"\tt0(x5) = %p\tt1(x6) = %p\n", regs->t0, regs->t1);
    __EP_LOG__(KCORE_LOG_FATAL,"\tt2(x7) = %p\n", regs->t2);
    __EP_LOG__(KCORE_LOG_FATAL,"\tt3(x28) = %p\tt4(x29) = %p\n", regs->t3, regs->t4);
    __EP_LOG__(KCORE_LOG_FATAL,"\tt5(x30) = %p\tt6(x31) = %p\n", regs->t5, regs->t6);
    __EP_LOG__(KCORE_LOG_FATAL,"Saved Registers:\n");
    __EP_LOG__(KCORE_LOG_FATAL,"\ts0/fp(x8) = %p\ts1(x9) = %p\n", regs->s0_fp, regs->s1);
    __EP_LOG__(KCORE_LOG_FATAL,"\ts2(x18) = %p\ts3(x19) = %p\n", regs->s2, regs->s3);
    __EP_LOG__(KCORE_LOG_FATAL,"\ts4(x20) = %p\ts5(x21) = %p\n", regs->s4, regs->s5);
    __EP_LOG__(KCORE_LOG_FATAL,"\ts6(x22) = %p\ts7(x23) = %p\n", regs->s6, regs->s7);
    __EP_LOG__(KCORE_LOG_FATAL,"\ts8(x24) = %p\ts9(x25) = %p\n", regs->s8, regs->s9);
    __EP_LOG__(KCORE_LOG_FATAL,"\ts10(x26) = %p\ts11(x27) = %p\n", regs->s10, regs->s11);
    __EP_LOG__(KCORE_LOG_FATAL,"Function Arguments Registers:\n");
    __EP_LOG__(KCORE_LOG_FATAL,"\ta0(x10) = %p\ta1(x11) = %p\n", regs->a0, regs->a1);
    __EP_LOG__(KCORE_LOG_FATAL,"\ta2(x12) = %p\ta3(x13) = %p\n", regs->a2, regs->a3);
    __EP_LOG__(KCORE_LOG_FATAL,"\ta4(x14) = %p\ta5(x15) = %p\n", regs->a4, regs->a5);
    __EP_LOG__(KCORE_LOG_FATAL,"\ta6(x16) = %p\ta7(x17) = %p\n", regs->a6, regs->a7);
    __EP_LOG__(KCORE_LOG_FATAL,"mstatus = %p\n", regs->mstatus);
    __EP_LOG__(KCORE_LOG_FATAL,"mepc = %p\n", regs->epc);
    __EP_LOG__(KCORE_LOG_FATAL,"-----------------Dump OK---------------------\n");
}

static const char *Exception_Name[] = {"Instruction Address Misaligned",
                                       "Instruction Access Fault",
                                       "Illegal Instruction",
                                       "Breakpoint",
                                       "Load Address Misaligned",
                                       "Load Access Fault",
                                       "Store/AMO Address Misaligned",
                                       "Store/AMO Access Fault",
                                       "Environment call from U-mode",
                                       "Environment call from S-mode",
                                       "Reserved-10",
                                       "Reserved-11",
                                       "Instruction Page Fault",
                                       "Load Page Fault",
                                       "Reserved-14",
                                       "Store/AMO Page Fault"};


static const char *get_exception_msg(int id)
{
    const char *msg;
    if (id < sizeof(Exception_Name) / sizeof(const char *)) {
        msg = Exception_Name[id];
    } else {
        msg = "Unknown Exception";
    }
    return msg;
}

extern void power_off_hook_for_trap();

static void handle_nested_trap_panic(rt_ubase_t cause, rt_ubase_t tval,
                                     rt_ubase_t epc,
                                     struct rt_hw_stack_frame *eframe)
{
    rt_ubase_t id = __MASKVALUE(cause, __MASK(63UL));
    const char *msg = get_exception_msg(id);
    __EP_LOG__(KCORE_LOG_FATAL,"\n-------- [SEVER ERROR] --------");
    __EP_LOG__(KCORE_LOG_FATAL,"Exception_Name: %s\n", msg);
    __EP_LOG__(KCORE_LOG_FATAL,"mcause:%p,mtval:%p,mepc:%p\n", cause, tval, epc);
    dump_regs(eframe);

    power_off_hook_for_trap();

    rt_hw_cpu_shutdown();
}


/* Trap entry */
void do_exception_trap(rt_ubase_t mcause, rt_ubase_t mtval,
                       struct rt_hw_stack_frame *regs)
{
#ifdef RT_USING_GDB
    if (mcause == EP_BREAKPOINT) {
        if (gdb_undef_hook(regs)){
            if (csi_get_cpu_id() == 0)
                csi_coret_reset_value2_by_id(0);
            return;
        }
    }
#endif

    handle_nested_trap_panic(mcause, mtval, regs->epc, regs);
    while (1);
}
