/* Entry-only capture. Included with the production command/wait wrappers.
 * The two hardware seams are also supplied by the host protocol test. */
static volatile WaferTx81KernelTimingRecord *wafer_timing_record;
static uint32_t wafer_timing_tile;
static uint64_t wafer_timing_begin_us;

void wafer_tx81_profile_entry_begin_from_config(uint64_t buffer_address) {
  wafer_timing_record = NULL;
  if ((buffer_address & (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT - 1U)) != 0U ||
      buffer_address > UINT64_MAX - WAFER_TX81_KERNEL_TIMING_RECORD_BYTES)
    return;
  wafer_profile_cache_range(buffer_address,
                            WAFER_TX81_KERNEL_TIMING_RECORD_BYTES, 1U);
  const volatile WaferTx81ProfilerLaunchConfig *config =
      (const volatile WaferTx81ProfilerLaunchConfig *)(uintptr_t)buffer_address;
  if (config->magic != WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC ||
      config->config_bytes != WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES ||
      config->record_bytes != WAFER_TX81_KERNEL_TIMING_RECORD_BYTES ||
      config->tile_id >= WAFER_TX81_PROFILER_TILE_COUNT ||
      config->flags != WAFER_TX81_PROFILER_ENTRY_TIMING_ONLY ||
      config->guard != WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD ||
      config->reserved0 != 0U || config->trace_event_limit != 0U ||
      config->reserved[0] != 0U || config->reserved[1] != 0U)
    return;
  wafer_timing_tile = config->tile_id;
  wafer_timing_record =
      (volatile WaferTx81KernelTimingRecord *)(uintptr_t)buffer_address;
  wafer_timing_begin_us = wafer_profile_microseconds();
}

void wafer_tx81_profile_entry_end(void) {
  uint64_t end_us = wafer_profile_microseconds();
  volatile WaferTx81KernelTimingRecord *record = wafer_timing_record;
  wafer_timing_record = NULL;
  if (record == NULL)
    return;
  record->magic = WAFER_TX81_KERNEL_TIMING_RECORD_MAGIC;
  record->record_bytes = WAFER_TX81_KERNEL_TIMING_RECORD_BYTES;
  record->tile_id = wafer_timing_tile;
  record->entry_begin_us = wafer_timing_begin_us;
  record->entry_end_us = end_us;
  record->reserved[0] = 0;
  record->reserved[1] = 0;
  record->reserved[2] = 0;
  record->completion_guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
  wafer_profile_cache_range((uint64_t)(uintptr_t)record,
                            WAFER_TX81_KERNEL_TIMING_RECORD_BYTES, 0U);
}
