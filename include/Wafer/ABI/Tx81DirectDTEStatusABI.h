//===- Tx81DirectDTEStatusABI.h - TX81 Direct DTE status ABI -*- C -*-===//

#ifndef WAFER_ABI_TX81DIRECTDTESTATUSABI_H
#define WAFER_ABI_TX81DIRECTDTESTATUSABI_H

#include <stdint.h>

/* The value remains at offset zero and owns one complete TX81 cache line so
 * Kcore cache writeback and invalidation cannot touch another resource. */
#define WAFER_TX81_DIRECT_DTE_STATUS_ABI "wafer-direct-dte-status-v2"

enum {
  WAFER_TX81_DIRECT_DTE_STATUS_PENDING = 0,
  WAFER_TX81_DIRECT_DTE_STATUS_SUCCESS = 1,
  WAFER_TX81_DIRECT_DTE_STATUS_TRANSPORT_ERROR = 2,
};

#define WAFER_TX81_DIRECT_DTE_STATUS_POISON UINT32_MAX

#define WAFER_TX81_DIRECT_DTE_STATUS_VALUE_OFFSET 0U
#define WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES 4U
#define WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES 64U
#define WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT 64U
#define WAFER_TX81_DIRECT_DTE_STATUS_CACHE_LINE_BYTES 64U

#if WAFER_TX81_DIRECT_DTE_STATUS_VALUE_OFFSET != 0
#error "TX81 Direct DTE status value must remain at offset zero"
#endif

#if WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES !=                              \
    WAFER_TX81_DIRECT_DTE_STATUS_CACHE_LINE_BYTES
#error "TX81 Direct DTE status storage must own exactly one cache line"
#endif

#if WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT !=                          \
    WAFER_TX81_DIRECT_DTE_STATUS_CACHE_LINE_BYTES
#error "TX81 Direct DTE status storage must be cache-line aligned"
#endif

#if WAFER_TX81_DIRECT_DTE_STATUS_VALUE_OFFSET +                                \
        WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES >                             \
    WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES
#error "TX81 Direct DTE status value must fit in its owned storage"
#endif

#if (WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT &                          \
     (WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT - 1)) != 0
#error "TX81 Direct DTE status alignment must be a power of two"
#endif

#endif // WAFER_ABI_TX81DIRECTDTESTATUSABI_H
