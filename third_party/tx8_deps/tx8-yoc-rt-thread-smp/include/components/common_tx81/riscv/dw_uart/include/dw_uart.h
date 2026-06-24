/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef MOD_DW_UART_H
#define MOD_DW_UART_H

#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#if CONFIG_TX8_KERNEL_PRINTF_SUPPORT
extern int tx8_kernel_printf(const char* format, ...);
#endif


#endif