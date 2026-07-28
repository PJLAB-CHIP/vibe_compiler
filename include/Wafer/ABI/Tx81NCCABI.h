//===- Tx81NCCABI.h - TX81 NCC worker-domain ABI --------------*- C -*-===//

#ifndef WAFER_ABI_TX81NCCABI_H
#define WAFER_ABI_TX81NCCABI_H

#include <stdint.h>

#define WAFER_TX81_NCC_WORKER_COUNT 3U
#define WAFER_TX81_NCC_ALL_WORKERS_MASK                                  \
  ((UINT32_C(1) << WAFER_TX81_NCC_WORKER_COUNT) - UINT32_C(1))

#endif // WAFER_ABI_TX81NCCABI_H
