/*
 * This file is included only when wafer_tx81_crt.c is compiled with
 * WAFER_TX81_PROFILE_CRT.  Keeping the implementation behind a translation
 * time guard makes the production CRT contain no profiler branch or state.
 */

#define WAFER_PROFILE_PMU_BASE UINT64_C(0x590000)
#define WAFER_PROFILE_PMU_WORKER_STRIDE UINT32_C(0x30)
#define WAFER_PROFILE_PMU_STABLE_RETRIES 8U

static const uint32_t
    wafer_profile_pmu64_offsets[WAFER_TX81_PROFILER_PMU64_COUNTERS] = {
        GR_PMU_STATISTICS_WINDOW, GR_PMU_FU_EXE_TIME,     GR_PMU_CT_EXE_TIME,
        GR_PMU_NE_EXE_TIME,       GR_PMU_RDMA_EXE_TIME,   GR_PMU_WDMA_EXE_TIME,
        GR_PMU_TDMA_EXE_TIME,     GR_PMU_SCALAR_EXE_TIME,
};

static const uint32_t
    wafer_profile_instruction_offsets[WAFER_TX81_PROFILER_QUEUES] = {
        GR_PMU_CT_INST_NUMS,   GR_PMU_NE_INST_NUMS,   GR_PMU_RDMA_INST_NUMS,
        GR_PMU_WDMA_INST_NUMS, GR_PMU_TDMA_INST_NUMS,
};

static const uint32_t
    wafer_profile_blocking_offsets[WAFER_TX81_PROFILER_QUEUES] = {
        GR_PMU_CT_BLOCKING_TIME,   GR_PMU_NE_BLOCKING_TIME,
        GR_PMU_RDMA_BLOCKING_TIME, GR_PMU_WDMA_BLOCKING_TIME,
        GR_PMU_TDMA_BLOCKING_TIME,
};

static volatile WaferTx81ProfilerRecordHeader *wafer_profile_header;
static volatile WaferTx81ProfilerTSMCallEvent *wafer_profile_events;
static volatile uint64_t *wafer_profile_buffer_guard;

static void wafer_profile_reset_binding(void) {
  wafer_profile_header = NULL;
  wafer_profile_events = NULL;
  wafer_profile_buffer_guard = NULL;
}

static uint64_t wafer_profile_cycle(void) {
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static uint32_t wafer_profile_read_pmu32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_PMU_BASE +
                                                 offset);
}

static uint64_t wafer_profile_read_pmu64(uint32_t offset, uint32_t index,
                                         uint32_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_PROFILE_PMU_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_profile_read_pmu32(offset + 4U);
    low = wafer_profile_read_pmu32(offset);
    high_after = wafer_profile_read_pmu32(offset + 4U);
    if (high_before == high_after) {
      *stable_mask |= UINT32_C(1) << index;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static void
wafer_profile_snapshot(volatile WaferTx81ProfilerPMUSnapshot *snapshot) {
  uint32_t stable_mask = 0;
  for (uint32_t index = 0; index < WAFER_TX81_PROFILER_PMU64_COUNTERS; ++index)
    snapshot->counters[index] = wafer_profile_read_pmu64(
        wafer_profile_pmu64_offsets[index], index, &stable_mask);
  for (uint32_t worker = 0; worker < WAFER_TX81_PROFILER_WORKERS; ++worker) {
    uint32_t worker_offset = worker * WAFER_PROFILE_PMU_WORKER_STRIDE;
    for (uint32_t queue = 0; queue < WAFER_TX81_PROFILER_QUEUES; ++queue) {
      snapshot->instructions[worker][queue] = wafer_profile_read_pmu32(
          wafer_profile_instruction_offsets[queue] + worker_offset);
      snapshot->blocking[worker][queue] = wafer_profile_read_pmu32(
          wafer_profile_blocking_offsets[queue] + worker_offset);
    }
  }
  snapshot->enable = wafer_profile_read_pmu32(GR_PMU_EN);
  snapshot->stable_mask = stable_mask;
}

static void wafer_profile_cache_range(uint64_t begin, uint64_t bytes,
                                      uint32_t invalidate_only) {
  enum {
    WAFER_PROFILE_SUPERVISOR_MODE = 1,
    WAFER_PROFILE_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uint64_t address = begin; address < begin + bytes;
       address += WAFER_TX81_PROFILER_CACHE_LINE_BYTES) {
    if (mode == WAFER_PROFILE_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0"
                         :
                         : "r"((uintptr_t)address)
                         : "memory");
      else
        __asm__ volatile("dcache.cipa %0"
                         :
                         : "r"((uintptr_t)address)
                         : "memory");
    } else if (mode == WAFER_PROFILE_SUPERVISOR_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.iva %0"
                         :
                         : "r"((uintptr_t)address)
                         : "memory");
      else
        __asm__ volatile("dcache.civa %0"
                         :
                         : "r"((uintptr_t)address)
                         : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_profile_zero(volatile void *storage, uint64_t bytes) {
  volatile uint8_t *cursor = (volatile uint8_t *)storage;
  for (uint64_t index = 0; index < bytes; ++index)
    cursor[index] = 0;
}

static void wafer_profile_mark_protocol_error(void) {
  if (wafer_profile_header == NULL)
    return;
  wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR;
  wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_INVALID;
}

void wafer_tx81_profile_entry_begin(uint64_t buffer_address,
                                    uint64_t buffer_bytes, uint32_t tile_id,
                                    uint32_t flags) {
  wafer_profile_reset_binding();
  if ((buffer_address & (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT - 1U)) != 0U ||
      buffer_bytes < WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
      (buffer_bytes & (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT - 1U)) != 0U ||
      buffer_address > UINT64_MAX - buffer_bytes ||
      tile_id >= WAFER_TX81_PROFILER_TILE_COUNT)
    return;
  uint64_t event_storage = buffer_bytes - WAFER_TX81_PROFILER_HEADER_BYTES -
                           WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  uint64_t capacity = event_storage / WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  if (capacity > UINT32_MAX)
    return;

  wafer_profile_cache_range(buffer_address, buffer_bytes, 1U);
  wafer_profile_header =
      (volatile WaferTx81ProfilerRecordHeader *)(uintptr_t)buffer_address;
  wafer_profile_events =
      (volatile WaferTx81ProfilerTSMCallEvent
           *)(uintptr_t)(buffer_address + WAFER_TX81_PROFILER_EVENTS_OFFSET);
  uint64_t guard_offset = buffer_bytes - WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  wafer_profile_buffer_guard =
      (volatile uint64_t *)(uintptr_t)(buffer_address + guard_offset);

  wafer_profile_zero(wafer_profile_header, WAFER_TX81_PROFILER_HEADER_BYTES);
  wafer_profile_header->magic = WAFER_TX81_PROFILER_RECORD_MAGIC;
  wafer_profile_header->schema_version = WAFER_TX81_PROFILER_SCHEMA_VERSION;
  wafer_profile_header->header_bytes = WAFER_TX81_PROFILER_HEADER_BYTES;
  wafer_profile_header->event_bytes = WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  wafer_profile_header->events_offset = WAFER_TX81_PROFILER_EVENTS_OFFSET;
  wafer_profile_header->buffer_bytes = buffer_bytes;
  wafer_profile_header->buffer_address = buffer_address;
  wafer_profile_header->tile_id = tile_id;
  wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_RECORDING;
  wafer_profile_header->event_capacity = (uint32_t)capacity;
  wafer_profile_header->active_site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  wafer_profile_header->header_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  wafer_profile_header->buffer_guard_offset = guard_offset;
  wafer_profile_header->header_footer_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  *wafer_profile_buffer_guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
  wafer_profile_header->flags = WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN;
  if ((flags & WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED) != 0U)
    wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  if ((flags & WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY) != 0U)
    wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_COUNT_ONLY;
  if ((flags & ~(WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED |
                 WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY)) != 0U ||
      (flags & (WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED |
                WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY)) ==
          (WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED |
           WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY))
    wafer_profile_mark_protocol_error();

  wafer_profile_snapshot(&wafer_profile_header->pmu_before);
  wafer_profile_header->summary_validity |=
      WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED;
  if (wafer_profile_header->pmu_before.stable_mask ==
      (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE;
  wafer_profile_header->entry_begin_cycle = wafer_profile_cycle();
}

void wafer_tx81_profile_entry_begin_from_config(uint64_t buffer_address) {
  /*
   * Rejecting a malformed next-launch configuration must not leave the prior
   * invocation's DDR record bound.  Otherwise later TsmExecute wrappers could
   * write through a stale address before entry_end observes the rejection.
   */
  wafer_profile_reset_binding();
  if ((buffer_address & (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT - 1U)) != 0U ||
      buffer_address > UINT64_MAX - WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES)
    return;
  wafer_profile_cache_range(buffer_address,
                            WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES, 1U);
  const volatile WaferTx81ProfilerLaunchConfig *source =
      (const volatile WaferTx81ProfilerLaunchConfig *)(uintptr_t)buffer_address;
  WaferTx81ProfilerLaunchConfig config;
  config.magic = source->magic;
  config.schema_version = source->schema_version;
  config.config_bytes = source->config_bytes;
  config.record_bytes = source->record_bytes;
  config.tile_id = source->tile_id;
  config.flags = source->flags;
  config.guard = source->guard;
  for (uint32_t index = 0; index < 3U; ++index)
    config.reserved[index] = source->reserved[index];
  if (config.magic != WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC ||
      config.schema_version != WAFER_TX81_PROFILER_SCHEMA_VERSION ||
      config.config_bytes != WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES ||
      config.guard != WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD ||
      config.reserved[0] != 0U || config.reserved[1] != 0U ||
      config.reserved[2] != 0U)
    return;
  wafer_tx81_profile_entry_begin(buffer_address, config.record_bytes,
                                 config.tile_id, config.flags);
}

void wafer_tx81_profile_site_begin(uint32_t site_id) {
  if (wafer_profile_header == NULL)
    return;
  if (wafer_profile_header->active_site_depth != 0U ||
      site_id == WAFER_TX81_PROFILER_INVALID_SITE_ID) {
    wafer_profile_mark_protocol_error();
    return;
  }
  wafer_profile_header->active_site_id = site_id;
  wafer_profile_header->next_sub_index = 0;
  wafer_profile_header->active_site_depth = 1;
}

void wafer_tx81_profile_site_end(uint32_t site_id) {
  if (wafer_profile_header == NULL)
    return;
  if (wafer_profile_header->active_site_depth != 1U ||
      wafer_profile_header->active_site_id != site_id) {
    wafer_profile_mark_protocol_error();
    return;
  }
  wafer_profile_header->active_site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  wafer_profile_header->next_sub_index = 0;
  wafer_profile_header->active_site_depth = 0;
}

static uint8_t wafer_profile_worker_metadata(uint32_t inter_type) {
  uint32_t worker = (inter_type >> 8) & UINT32_C(0x3);
  if (worker > 2U)
    return 0;
  return (uint8_t)worker | WAFER_TX81_PROFILER_EVENT_WORKER_VALID;
}

static uint64_t wafer_tx81_profile_execute_tsm(void *instruction,
                                               uint32_t inter_type,
                                               uint8_t engine) {
  if (wafer_profile_header == NULL)
    return TsmExecute(instruction);
  if ((wafer_profile_header->flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) !=
      0U) {
    ++wafer_profile_header->next_sequence;
    return TsmExecute(instruction);
  }
  if ((wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) == 0U)
    return TsmExecute(instruction);

  uint64_t sequence = wafer_profile_header->next_sequence++;
  uint32_t site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  uint16_t sub_index = WAFER_TX81_PROFILER_INVALID_SUB_INDEX;
  uint8_t metadata = wafer_profile_worker_metadata(inter_type);
  if (wafer_profile_header->active_site_depth == 1U) {
    site_id = wafer_profile_header->active_site_id;
    sub_index = wafer_profile_header->next_sub_index;
    metadata |= WAFER_TX81_PROFILER_EVENT_SITE_VALID;
    if (sub_index == WAFER_TX81_PROFILER_INVALID_SUB_INDEX) {
      wafer_profile_header->flags |=
          WAFER_TX81_PROFILER_RECORD_SUB_INDEX_OVERFLOW;
      metadata &= (uint8_t)~WAFER_TX81_PROFILER_EVENT_SITE_VALID;
      site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
    } else {
      ++wafer_profile_header->next_sub_index;
    }
  }

  uint64_t begin_cycle = wafer_profile_cycle();
  uint64_t raw_return = TsmExecute(instruction);
  uint64_t end_cycle = wafer_profile_cycle();

  uint32_t count = wafer_profile_header->event_count;
  if (count < wafer_profile_header->event_capacity) {
    volatile WaferTx81ProfilerTSMCallEvent *event =
        &wafer_profile_events[count];
    event->sequence = sequence;
    event->begin_cycle = begin_cycle;
    event->end_cycle = end_cycle;
    event->raw_return = raw_return;
    event->site_id = site_id;
    event->sub_index = sub_index;
    event->engine = engine;
    event->metadata = metadata;
    wafer_profile_header->event_count = count + 1U;
  } else {
    ++wafer_profile_header->dropped_event_count;
    wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_OVERFLOW;
    if (wafer_profile_header->trace_state != WAFER_TX81_PROFILER_TRACE_INVALID)
      wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_OVERFLOW;
  }
  return raw_return;
}

void wafer_tx81_profile_entry_end(void) {
  if (wafer_profile_header == NULL)
    return;
  if (wafer_profile_header->active_site_depth != 0U)
    wafer_profile_mark_protocol_error();

  wafer_profile_header->entry_end_cycle = wafer_profile_cycle();
  wafer_profile_snapshot(&wafer_profile_header->pmu_after);
  wafer_profile_header->summary_validity |=
      WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES |
      WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED;
  if (wafer_profile_header->pmu_after.stable_mask ==
      (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE;

  wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                                 WAFER_TX81_PROFILER_RECORD_PUBLISHED;
  if ((wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR) != 0U)
    wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_INVALID;
  else if ((wafer_profile_header->flags &
            WAFER_TX81_PROFILER_RECORD_OVERFLOW) != 0U)
    wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_OVERFLOW;
  else
    wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_COMPLETE;
  *wafer_profile_buffer_guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
  wafer_profile_cache_range(wafer_profile_header->buffer_address,
                            wafer_profile_header->buffer_bytes, 0U);

  /*
   * The recovery read happens after the first publication pass and outside
   * the measured entry-cycle interval.  It does not create a fence event or
   * claim engine completion.  Equal end/recovery values merely qualify that
   * the aggregate counters stopped changing across this bounded observation.
   */
  wafer_profile_snapshot(&wafer_profile_header->pmu_recovery);
  wafer_profile_header->summary_validity |=
      WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED;
  if (wafer_profile_header->pmu_recovery.stable_mask ==
      (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_STABLE;

  uint32_t recovered = 1U;
  for (uint32_t index = 0; index < WAFER_TX81_PROFILER_PMU64_COUNTERS; ++index)
    if (wafer_profile_header->pmu_after.counters[index] !=
        wafer_profile_header->pmu_recovery.counters[index])
      recovered = 0U;
  for (uint32_t worker = 0; worker < WAFER_TX81_PROFILER_WORKERS; ++worker)
    for (uint32_t queue = 0; queue < WAFER_TX81_PROFILER_QUEUES; ++queue)
      if (wafer_profile_header->pmu_after.instructions[worker][queue] !=
              wafer_profile_header->pmu_recovery.instructions[worker][queue] ||
          wafer_profile_header->pmu_after.blocking[worker][queue] !=
              wafer_profile_header->pmu_recovery.blocking[worker][queue])
        recovered = 0U;
  if (recovered != 0U)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED;
  if (wafer_profile_header->pmu_before.enable ==
          wafer_profile_header->pmu_after.enable &&
      wafer_profile_header->pmu_after.enable ==
          wafer_profile_header->pmu_recovery.enable)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED;
  wafer_profile_cache_range(wafer_profile_header->buffer_address,
                            wafer_profile_header->buffer_bytes, 0U);
  wafer_profile_reset_binding();
}

#undef WAFER_PROFILE_PMU_STABLE_RETRIES
#undef WAFER_PROFILE_PMU_WORKER_STRIDE
#undef WAFER_PROFILE_PMU_BASE
