#ifndef WAFER_TX81_PROFILER_H
#define WAFER_TX81_PROFILER_H

#include "Wafer/ABI/Tx81ProfilerABI.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These symbols exist only in a profiling CRT module.  A production target
 * module must neither declare nor call them.
 */
void wafer_tx81_profile_entry_begin(uint64_t buffer_address,
                                    uint64_t buffer_bytes, uint32_t tile_id,
                                    uint32_t flags);
void wafer_tx81_profile_entry_begin_from_config(uint64_t buffer_address);
void wafer_tx81_profile_site_begin(uint32_t site_id);
void wafer_tx81_profile_site_end(uint32_t site_id);
void wafer_tx81_profile_entry_end(void);

#ifdef __cplusplus
}
#endif

#endif // WAFER_TX81_PROFILER_H
