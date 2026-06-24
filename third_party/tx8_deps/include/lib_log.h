/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef LIB_LOG_H
#define LIB_LOG_H

#include <stdarg.h>
#include <stddef.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define vdk_printf tx8_kernel_printf

//#define vdk_printf(fmt, args...) printf(fmt,##args)

/* Log in freely format without prefix */
#define LOG_F(fmt, args...) printf(fmt,##args)
extern int tx8_log_init(void);

#if CONFIG_TX8_KERNEL_PRINTF_SUPPORT
extern int tx8_kernel_printf(const char* format, ...);
extern int tx8_kernel_vprintf(const char* format, va_list va);
int tx8_kernel_vsnprintf(char* buffer, size_t count, const char* format, va_list va);
#endif

typedef enum KcoreLogLevel {
    KCORE_LOG_DEBUG = 0,
    KCORE_LOG_INFO,
    KCORE_LOG_WARNING,
    KCORE_LOG_ERROR,
    KCORE_LOG_FATAL,
    KCORE_LOG_ALWAYS,
} KcoreLogLevel;

#define __PUSH_BT__(func_name) 
#define __POP_BT__(func_name) 
void monitor_write_log(const char *file_name, const char *func_name, uint32_t line_number, char *format, ...);
void kcore_write_log(char *format, ...);
uint32_t get_log_level();

#define __LOG_NONE__(log_level, format, ...) \
do { \
} while (0)
#define __LOG__(log_level, format, ...) \
do { \
    if (log_level >= KCORE_LOG_ERROR) { \
        tsm_ep_log(__FILE__, __func__, __LINE__, log_level, format, ##__VA_ARGS__); \
    }   \
    if (get_log_level() <= log_level) { \
        monitor_write_log(__FILE__, __func__, __LINE__, format, ##__VA_ARGS__); \
    } \
} while (0)

#define ARM_RISCV_SH_BASE 0x180000000
#define MODULE_RINGBUF_SIZE 0x200000
#define MODULE_INDEX_OFFSET 0x7F00000
#define MODULE_INDEXBUF_SIZE 0x1000
#define TILE_LOG_OFFSET 3
/*	format as follows,three modules per row,each module occupies 2M*/
/*  -----------------------------------------------------------------*/
/*	tile0_instream	tile0_outstream		tile0_baremental    */
/*	tile1_instream	tile1_outstream		tile1_baremental    */
/*	......   */
/*	tile15_instream	tile15_outstream	tile15_baremental    */
/*	kernel			media				reserve_for_more    */
/*  -----------------------------------------------------------------*/
typedef enum {
    TILE0_INSTREAM, TILE0_OUTSTREAM, TILE0_BAREMENTAL,
    TILE1_INSTREAM, TILE1_OUTSTREAM, TILE1_BAREMENTAL,
    TILE2_INSTREAM, TILE2_OUTSTREAM, TILE2_BAREMENTAL,
    TILE3_INSTREAM, TILE3_OUTSTREAM, TILE3_BAREMENTAL,
    TILE4_INSTREAM, TILE4_OUTSTREAM, TILE4_BAREMENTAL,
    TILE5_INSTREAM, TILE5_OUTSTREAM, TILE5_BAREMENTAL,
    TILE6_INSTREAM, TILE6_OUTSTREAM, TILE6_BAREMENTAL,
    TILE7_INSTREAM, TILE7_OUTSTREAM, TILE7_BAREMENTAL,
    TILE8_INSTREAM, TILE8_OUTSTREAM, TILE8_BAREMENTAL,
    TILE9_INSTREAM, TILE9_OUTSTREAM, TILE9_BAREMENTAL,
    TILE10_INSTREAM, TILE10_OUTSTREAM, TILE10_BAREMENTAL,
    TILE11_INSTREAM, TILE11_OUTSTREAM, TILE11_BAREMENTAL,
    TILE12_INSTREAM, TILE12_OUTSTREAM, TILE12_BAREMENTAL,
    TILE13_INSTREAM, TILE13_OUTSTREAM, TILE13_BAREMENTAL,
    TILE14_INSTREAM, TILE14_OUTSTREAM, TILE14_BAREMENTAL,
    TILE15_INSTREAM, TILE15_OUTSTREAM, TILE15_BAREMENTAL,
    KERNEL,                OOB,            TSMVS,
    TXMM,                DISCOVERY,
    MODULE_TYPE_NUM,
    MODULE_TYPE_MAX = 62,
} module_type_e;

struct module_index {
    uint64_t wptr_ep;
    uint64_t rptr_rc;
    // used to change safety_log_level,console_log_level dynamically
    uint8_t safety_log_level;
    uint8_t console_log_level;
    uint16_t crit_cnt_ep;
    uint16_t crit_cnt_rc;
    uint16_t crit_cnt_rd;
};

#define __EP_LOG__(log_level, format, ...) tsm_ep_log(__FILE__, __func__, __LINE__, log_level, format, ##__VA_ARGS__);
#define GET_KCORE_LOG_OFFSET(tile_id) ((tile_id) * TILE_LOG_OFFSET + (TILE_LOG_OFFSET - 1))
void tsm_ep_log(const char *file_name, const char *func_name, uint32_t line_number, uint32_t level, const char *format, ...);
void _tsm_ep_log(const char *file_name, const char *func_name, uint32_t line_number, uint32_t level, const char *format, va_list v_list);
int tsm_ep_log_init(module_type_e module_type, int safety_log_level, int console_log_level);
#endif
