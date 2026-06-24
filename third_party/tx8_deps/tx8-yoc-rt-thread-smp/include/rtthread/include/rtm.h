/*
 * Copyright (c) 2006-2021, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 */

#ifndef __RTM_H__
#define __RTM_H__

#include <rtdef.h>
#include <rtthread.h>

#ifdef RT_USING_MODULE
struct rt_module_symtab
{
    void       *addr;
    const char *name;
};

#if defined(_MSC_VER)
#pragma section("RTMSymTab$f",read)
#define RTM_EXPORT(symbol)                                            \
__declspec(allocate("RTMSymTab$f"))const char __rtmsym_##symbol##_name[] = "__vs_rtm_"#symbol;
#pragma comment(linker, "/merge:RTMSymTab=mytext")

#pragma section("ExportedDYNSYMTab$f",read)
#define DYNSYM_EXPORT(symbol)                                            \
__declspec(allocate("ExportedDYNSYMTab$f"))const char __dynsym_##symbol##_name[] = "__vs_dyn_"#symbol;
#pragma comment(linker, "/merge:ExportedDYNSYMTab=mytext")

#elif defined(__MINGW32__)
#define RTM_EXPORT(symbol)
#define DYNSYM_EXPORT(symbol)
#else

#define RTM_EXPORT(symbol)                                                    \
const char __rtmsym_##symbol##_name[] rt_section(".rodata.name") = #symbol;   \
const struct rt_module_symtab __rtmsym_##symbol rt_section("RTMSymTab")=      \
{                                                                             \
    .addr = (void *)&symbol,                                                  \
    .name = __rtmsym_##symbol##_name                                          \
};

#define DYNSYM_EXPORT(symbol)                                                     \
const char __dynsym_##symbol##_name[] rt_section(".rodata.name") = #symbol;       \
const struct rt_module_symtab __dynsym_##symbol rt_section("ExportedDYNSYMTab")=  \
{                                                                                 \
    .addr = (void *)&symbol,                                                      \
    .name = __dynsym_##symbol##_name                                              \
};

#define SET_DYN_CONTROL_STRING_SYM(control_type,control_value)                                 \
const char __dynsym_##control_type##_name[] rt_section(".rodata.name") = control_value;        \
const  Elf64_Dyn __dynsym_##control_type rt_section(ELF_CONTROL_SYMTAB)=                       \
{                                                                                              \
        .d_tag = control_type,                                                                 \
        .d_un = { .d_ptr = __dynsym_##control_type##_name }                                   \
};

#define SET_DYN_CONTROL_INT_SYM(control_type,control_value)                   \
const  Elf64_Dyn __dynsym_##control_type rt_section(ELF_CONTROL_SYMTAB)=      \
{                                                                             \
    .d_tag = control_type,                                                    \
    .d_un = { .d_val = control_value }                                   \
};
#define EXPORT_ALL_SYM SET_DYN_CONTROL_INT_SYM(CTR_EXPORT_SYM_POLICY,EXPORT_ALL_SYM_POLICY)
#define MARK_AS_SYSTEM_COMPONENT      SET_DYN_CONTROL_INT_SYM(CTR_SO_TYPE,CTR_SO_TYPE_SYSTEM_COMPONENT)
#define SET_SO_FILE_PATH(SO_FILE_PATH)  \
SET_DYN_CONTROL_STRING_SYM(CTR_SO_FILE_PATH,SO_FILE_PATH)

#define SET_SO_DEPS_PATH(SO_DEPS_PATH)  \
SET_DYN_CONTROL_STRING_SYM(CTR_SO_DEPS_FILE_PATH,SO_DEPS_PATH)

#endif

#else
#define RTM_EXPORT(symbol)
#define DYNSYM_EXPORT(symbol)
#endif

#endif
