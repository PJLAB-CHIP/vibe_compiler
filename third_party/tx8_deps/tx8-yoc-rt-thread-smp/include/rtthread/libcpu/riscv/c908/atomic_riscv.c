/*
 * Copyright (c) 2006-2023, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2023-03-14     WangShun     first version
 */

#include <rtthread.h>

rt_atomic_t rt_hw_atomic_exchange(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile(
        "1:     lr.w    %0, %2\n"
        "       sc.w    %1, %3, %2\n"
        "       bnez    %1, 1b\n"
        : "=&r"(result), "=&r"(temp), "+A"(*ptr)
        : "r"(val)
        : "memory");
#elif __riscv_xlen == 64
    asm volatile(
        "1:     lr.d    %0, %2\n"
        "       sc.d    %1, %3, %2\n"
        "       bnez    %1, 1b\n"
        : "=&r"(result), "=&r"(temp), "+A"(*ptr)
        : "r"(val)
        : "memory");
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoswap.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoswap.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_add(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
      "1:     lr.w    %0, %2\n"
      "       add     %0, %0, %3\n"
      "       sc.w    %1, %0, %2\n"
      "       bnez    %1, 1b\n"
      : "=&r" (result), "=&r" (temp), "+A" (*ptr)
      : "r" (val)
      : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
      "1:     lr.d    %0, %2\n"
      "       add     %0, %0, %3\n"
      "       sc.d    %1, %0, %2\n"
      "       bnez    %1, 1b\n"
      : "=&r" (result), "=&r" (temp), "+A" (*ptr)
      : "r" (val)
      : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoadd.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoadd.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_sub(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w %0, %2\n"
        "       sub %0, %0, %3\n"
        "       sc.w %1, %0, %2\n"
        "       bnez %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d %0, %2\n"
        "       sub %0, %0, %3\n"
        "       sc.d %1, %0, %2\n"
        "       bnez %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
    val = -val;
#if __riscv_xlen == 32
    asm volatile ("amoadd.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoadd.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_xor(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w     %0, %2\n"
        "       xor      %0, %0, %3\n"
        "       sc.w     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d     %0, %2\n"
        "       xor      %0, %0, %3\n"
        "       sc.d     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoxor.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoxor.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_and(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w     %0, %2\n"
        "       and      %0, %0, %3\n"
        "       sc.w     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d     %0, %2\n"
        "       and      %0, %0, %3\n"
        "       sc.d     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoand.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoand.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_or(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w     %0, %2\n"
        "       or       %0, %0, %3\n"
        "       sc.w     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d     %0, %2\n"
        "       or       %0, %0, %3\n"
        "       sc.d     %1, %0, %2\n"
        "       bnez     %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        : "r" (val)
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoor.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoor.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

rt_atomic_t rt_hw_atomic_load(volatile rt_atomic_t *ptr)
{
    rt_atomic_t result = 0;
#ifdef ATOMIC_WITH_LR_SC
#if __riscv_xlen == 32
    asm volatile ("lr.w %0, (%1)" : "=&r"(result) : "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("lr.d %0, (%1)" : "=&r"(result) : "r"(ptr) : "memory");
#endif
#else
#if __riscv_xlen == 32
    asm volatile ("amoxor.w %0, x0, (%1)" : "=r"(result) : "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoxor.d %0, x0, (%1)" : "=r"(result) : "r"(ptr) : "memory");
#endif
#endif
    return result;
}

void rt_hw_atomic_store(volatile rt_atomic_t *ptr, rt_atomic_t val)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w     %1, %0\n"
        "       sc.w     %1, %2, %0\n"
        "       bnez     %1, 1b\n"
        : "+A" (*ptr), "=r" (temp)
        : "r" (val)
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d     %1, %0\n"
        "       sc.d     %1, %2, %0\n"
        "       bnez     %1, 1b\n"
        : "+A" (*ptr), "=r" (temp)
        : "r" (val)
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoswap.w %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoswap.d %0, %1, (%2)" : "=r"(result) : "r"(val), "r"(ptr) : "memory");
#endif
#endif
}

rt_atomic_t rt_hw_atomic_flag_test_and_set(volatile rt_atomic_t *ptr)
{
#ifdef ATOMIC_WITH_LR_SC
    rt_atomic_t result = 0, temp;
#if __riscv_xlen == 32
    asm volatile (
        "1:     lr.w    %0, %2\n"
        "       li      %1, 1\n"
        "       sc.w    %1, %1, %2\n"
        "       bnez    %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        :
        : "memory"
    );
#elif __riscv_xlen == 64
    asm volatile (
        "1:     lr.d    %0, %2\n"
        "       li      %1, 1\n"
        "       sc.d    %1, %1, %2\n"
        "       bnez    %1, 1b\n"
        : "=&r" (result), "=&r" (temp), "+A" (*ptr)
        :
        : "memory"
    );
#endif
#else
    rt_atomic_t result = 0;
    rt_atomic_t temp = 1;
#if __riscv_xlen == 32
    asm volatile ("amoor.w %0, %1, (%2)" : "=r"(result) : "r"(temp), "r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoor.d %0, %1, (%2)" : "=r"(result) : "r"(temp), "r"(ptr) : "memory");
#endif
#endif
    return result;
}

void rt_hw_atomic_flag_clear(volatile rt_atomic_t *ptr)
{
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile ("amoand.w %0, x0, (%1)" : "=r"(result) :"r"(ptr) : "memory");
#elif __riscv_xlen == 64
    asm volatile ("amoand.d %0, x0, (%1)" : "=r"(result) :"r"(ptr) : "memory");
#endif
}

rt_atomic_t rt_hw_atomic_compare_exchange_strong(volatile rt_atomic_t *ptr, rt_atomic_t *old, rt_atomic_t desired)
{
    rt_atomic_t tmp = *old;
    rt_atomic_t result = 0;
#if __riscv_xlen == 32
    asm volatile(
            " fence iorw, ow\n"
            "1: lr.w.aq  %[result], (%[ptr])\n"
            "   bne      %[result], %[tmp], 2f\n"
            "   sc.w.rl  %[tmp], %[desired], (%[ptr])\n"
            "   bnez     %[tmp], 1b\n"
            "   li  %[result], 1\n"
            "   j 3f\n"
            " 2:sw  %[result], (%[old])\n"
            "   li  %[result], 0\n"
            " 3:\n"
            : [result]"+r" (result), [tmp]"+r" (tmp), [ptr]"+r" (ptr)
            : [desired]"r" (desired), [old]"r"(old)
            : "memory");
#elif __riscv_xlen == 64
    asm volatile(
            " fence iorw, ow\n"
            "1: lr.d.aq  %[result], (%[ptr])\n"
            "   bne      %[result], %[tmp], 2f\n"
            "   sc.d.rl  %[tmp], %[desired], (%[ptr])\n"
            "   bnez     %[tmp], 1b\n"
            "   li  %[result], 1\n"
            "   j 3f\n"
            " 2:sd  %[result], (%[old])\n"
            "   li  %[result], 0\n"
            " 3:\n"
            : [result]"+r" (result), [tmp]"+r" (tmp), [ptr]"+r" (ptr)
            : [desired]"r" (desired), [old]"r"(old)
            : "memory");
#endif
    return result;
}

int rt_hw_atomic_compare_exchange(volatile int *p, int expected, int new)
{
    int prev, rc;
    asm volatile (
            "0: lr.w %[prev], (%[p])\n"
            "   bne %[prev], %[e], 1f\n"
            "   sc.w %[rc], %[new], (%[p])\n"
            "   bnez %[rc], 0b\n"
            "1:"
            : [prev]"=&r"(prev), [rc]"=&r"(rc)
            : [p]"r"(p), [e]"r"(expected), [new]"r"(new)
            : "memory");
    return prev;
}

