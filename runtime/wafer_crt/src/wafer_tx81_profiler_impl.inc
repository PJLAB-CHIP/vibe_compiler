/*
 * This file is included only when wafer_tx81_crt.c is compiled with
 * WAFER_TX81_PROFILE_TRACE_CRT. Keeping the implementation behind a translation
 * time guard makes the production CRT contain no profiler branch or state.
 */

#define WAFER_PROFILE_PMU_BASE UINT64_C(0x590000)
#define WAFER_PROFILE_DTE_PMU_BASE UINT64_C(0x400000)
#define WAFER_PROFILE_PMU_WORKER_STRIDE UINT32_C(0x30)
#define WAFER_PROFILE_PMU_STABLE_RETRIES 8U
#define WAFER_PROFILE_NCC_ENGINE_COUNT 5U

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
static uint64_t
    wafer_profile_last_engine_counters[WAFER_PROFILE_NCC_ENGINE_COUNT];
static uint32_t
    wafer_profile_latest_engine_events[WAFER_PROFILE_NCC_ENGINE_COUNT];
static uint64_t
    wafer_profile_last_engine_read_begin_cycles[WAFER_PROFILE_NCC_ENGINE_COUNT];
static uint32_t wafer_profile_last_engine_counter_valid_mask;
static uint64_t wafer_profile_active_site_begin_cycle;
static uint32_t wafer_profile_active_site_first_event;
static uint32_t wafer_profile_direct_event_index;
static uint32_t wafer_profile_direct_role_metadata;
static uint8_t wafer_profile_direct_aggregate_kind;
static uint64_t wafer_profile_direct_operation_begin_cycle;
static uint64_t wafer_profile_direct_observation_begin_cycle;
static uint64_t wafer_profile_direct_start_counters[2];
static uint32_t wafer_profile_direct_start_valid_mask;
static uint32_t wafer_profile_ncc_enable_before;
static uint32_t wafer_profile_ncc_enable_owned;
static uint32_t wafer_profile_dte_enable_before;
static uint32_t wafer_profile_dte_enable_owned;
static uint64_t wafer_profile_pending_setup_begin_cycle;

static void wafer_profile_reset_binding(void) {
  wafer_profile_header = NULL;
  wafer_profile_events = NULL;
  wafer_profile_buffer_guard = NULL;
  wafer_profile_last_engine_counter_valid_mask = 0;
  wafer_profile_active_site_begin_cycle = 0;
  wafer_profile_active_site_first_event = 0;
  wafer_profile_direct_event_index = UINT32_MAX;
  wafer_profile_direct_role_metadata = 0;
  wafer_profile_direct_aggregate_kind = 0;
  wafer_profile_direct_operation_begin_cycle = 0;
  wafer_profile_direct_observation_begin_cycle = 0;
  wafer_profile_direct_start_counters[0] = 0;
  wafer_profile_direct_start_counters[1] = 0;
  wafer_profile_direct_start_valid_mask = 0;
  wafer_profile_ncc_enable_before = 0;
  wafer_profile_ncc_enable_owned = 0;
  wafer_profile_dte_enable_before = 0;
  wafer_profile_dte_enable_owned = 0;
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine) {
    wafer_profile_last_engine_counters[engine] = 0;
    wafer_profile_last_engine_read_begin_cycles[engine] = 0;
    wafer_profile_latest_engine_events[engine] = UINT32_MAX;
  }
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

static uint32_t wafer_profile_read_dte_pmu32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_DTE_PMU_BASE +
                                                 offset);
}

static uint64_t wafer_profile_read_dte_pmu64(uint32_t low_offset,
                                             uint32_t validity_bit,
                                             uint32_t *valid_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_PROFILE_PMU_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_profile_read_dte_pmu32(low_offset + 4U);
    low = wafer_profile_read_dte_pmu32(low_offset);
    high_after = wafer_profile_read_dte_pmu32(low_offset + 4U);
    if (high_before == high_after) {
      *valid_mask |= validity_bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
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

static void wafer_profile_add_cost(volatile uint64_t *category,
                                   uint64_t begin_cycle,
                                   uint64_t end_cycle) {
  if (wafer_profile_header == NULL || category == NULL)
    return;
  if (end_cycle < begin_cycle) {
    *category = UINT64_MAX;
    wafer_profile_mark_protocol_error();
    return;
  }
  uint64_t delta = end_cycle - begin_cycle;
  if (*category > UINT64_MAX - delta) {
    *category = UINT64_MAX;
    wafer_profile_mark_protocol_error();
    return;
  }
  *category += delta;
}

static uint32_t wafer_profile_is_trace_capture(void) {
  return wafer_profile_header != NULL &&
         (wafer_profile_header->flags &
          WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0U &&
         (wafer_profile_header->flags &
          WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) == 0U &&
         wafer_profile_header->trace_state ==
             WAFER_TX81_PROFILER_TRACE_RECORDING;
}

static uint16_t wafer_profile_take_sub_index(uint32_t *site_id,
                                             uint32_t *metadata) {
  uint16_t sub_index = WAFER_TX81_PROFILER_INVALID_SUB_INDEX;
  *site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  if (wafer_profile_header->active_site_depth != 1U)
    return sub_index;
  *site_id = wafer_profile_header->active_site_id;
  sub_index = wafer_profile_header->next_sub_index;
  *metadata |= WAFER_TX81_PROFILER_EVENT_SITE_VALID;
  if (sub_index == WAFER_TX81_PROFILER_INVALID_SUB_INDEX) {
    wafer_profile_header->flags |=
        WAFER_TX81_PROFILER_RECORD_SUB_INDEX_OVERFLOW;
    *metadata &= ~WAFER_TX81_PROFILER_EVENT_SITE_VALID;
    *site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  } else {
    ++wafer_profile_header->next_sub_index;
  }
  return sub_index;
}

static uint32_t wafer_profile_allocate_event(uint8_t engine, uint8_t kind,
                                             uint32_t metadata) {
  uint64_t bookkeeping_begin = wafer_profile_cycle();
  uint64_t sequence = wafer_profile_header->next_sequence++;
  uint32_t site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  uint16_t sub_index = wafer_profile_take_sub_index(&site_id, &metadata);
  uint32_t count = wafer_profile_header->event_count;
  if (count >= wafer_profile_header->event_capacity) {
    ++wafer_profile_header->dropped_event_count;
    wafer_profile_header->flags |= WAFER_TX81_PROFILER_RECORD_OVERFLOW;
    if (wafer_profile_header->trace_state != WAFER_TX81_PROFILER_TRACE_INVALID)
      wafer_profile_header->trace_state = WAFER_TX81_PROFILER_TRACE_OVERFLOW;
    uint64_t bookkeeping_end = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
        bookkeeping_begin, bookkeeping_end);
    return UINT32_MAX;
  }

  volatile WaferTx81ProfilerTSMCallEvent *event = &wafer_profile_events[count];
  event->sequence = sequence;
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->counter_delta = 0;
  event->site_begin_cycle = 0;
  event->site_end_cycle = 0;
  event->operation_begin_cycle = 0;
  event->operation_end_cycle = 0;
  event->site_id = site_id;
  event->observation_count = 0;
  event->sub_index = sub_index;
  event->engine = engine;
  event->kind = kind;
  event->metadata = metadata;
  wafer_profile_header->event_count = count + 1U;
  uint64_t bookkeeping_end = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
      bookkeeping_begin, bookkeeping_end);
  return count;
}

static void wafer_profile_invalidate_ncc_event(uint32_t event_index) {
  if (event_index == UINT32_MAX ||
      event_index >= wafer_profile_header->event_count)
    return;
  volatile WaferTx81ProfilerTSMCallEvent *event =
      &wafer_profile_events[event_index];
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->counter_delta = 0;
  event->observation_count = 0;
  event->metadata &=
      ~(WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
        WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE |
        WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID);
}

/*
 * Each observation brackets five hardware execution-counter reads.  Stable
 * engines retain their independent bounds and deltas.  An unstable split read
 * invalidates only the affected engine event and establishes no baseline; it
 * must never turn otherwise usable tile evidence into a protocol failure.
 */
static void wafer_profile_observe_ncc_activity(void) {
  if (!wafer_profile_is_trace_capture())
    return;

  uint64_t read_begin_cycle = wafer_profile_cycle();
  uint64_t counters[WAFER_PROFILE_NCC_ENGINE_COUNT];
  uint32_t stable_mask = 0;
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine)
    counters[engine] = wafer_profile_read_pmu64(
        wafer_profile_pmu64_offsets[engine + 2U], engine, &stable_mask);
  uint64_t read_end_cycle = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.ncc_pmu_sample_cycles,
      read_begin_cycle, read_end_cycle);
  uint64_t bookkeeping_begin = wafer_profile_cycle();
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine) {
    uint32_t bit = UINT32_C(1) << engine;
    uint32_t event_index = wafer_profile_latest_engine_events[engine];
    if ((stable_mask & bit) == 0U) {
      wafer_profile_last_engine_counter_valid_mask &= ~bit;
      wafer_profile_invalidate_ncc_event(event_index);
      continue;
    }

    uint64_t previous = wafer_profile_last_engine_counters[engine];
    uint64_t current = counters[engine];
    if ((wafer_profile_last_engine_counter_valid_mask & bit) == 0U) {
      wafer_profile_last_engine_counters[engine] = current;
      wafer_profile_last_engine_read_begin_cycles[engine] = read_begin_cycle;
      wafer_profile_last_engine_counter_valid_mask |= bit;
      continue;
    }
    if (current < previous) {
      wafer_profile_invalidate_ncc_event(event_index);
      wafer_profile_last_engine_counters[engine] = current;
      wafer_profile_last_engine_read_begin_cycles[engine] = read_begin_cycle;
      continue;
    }
    uint64_t delta = current - previous;
    if (event_index != UINT32_MAX &&
        event_index < wafer_profile_header->event_count) {
      volatile WaferTx81ProfilerTSMCallEvent *event =
          &wafer_profile_events[event_index];
      if ((event->metadata &
           WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID) == 0U) {
        /* This event crossed an unusable sample and cannot regain full bounds. */
      } else if (event->observation_count == UINT32_MAX) {
        wafer_profile_mark_protocol_error();
      } else {
        if (event->observation_count == 0U)
          event->observed_begin_cycle =
              wafer_profile_last_engine_read_begin_cycles[engine];
        ++event->observation_count;
        event->observed_end_cycle = read_end_cycle;
        event->metadata |=
            WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID;
        if (event->counter_delta > UINT64_MAX - delta) {
          wafer_profile_mark_protocol_error();
        } else {
          event->counter_delta += delta;
          if (event->counter_delta != 0U)
            event->metadata |=
                WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
        }
      }
    }
    wafer_profile_last_engine_counters[engine] = current;
    wafer_profile_last_engine_read_begin_cycles[engine] = read_begin_cycle;
  }
  uint64_t bookkeeping_end = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
      bookkeeping_begin, bookkeeping_end);
}

void wafer_tx81_profile_entry_begin(uint64_t buffer_address,
                                    uint64_t buffer_bytes, uint32_t tile_id,
                                    uint32_t flags) {
  uint64_t setup_begin_cycle = wafer_profile_pending_setup_begin_cycle;
  wafer_profile_pending_setup_begin_cycle = 0;
  if (setup_begin_cycle == 0U)
    setup_begin_cycle = wafer_profile_cycle();
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
  uint32_t capture_mode =
      flags & (WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED |
               WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY);
  if ((flags & ~(WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED |
                 WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY)) != 0U ||
      (capture_mode != WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED &&
       capture_mode != WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY))
    wafer_profile_mark_protocol_error();

  /*
   * Count is observational and must not change DTE PMU or NCC PMU state.
   * Trace alone owns both temporary enable mutations and
   * restores the exact prior values at entry_end. Counters are never cleared:
   * before/after subtraction remains valid whether the original enable was
   * zero or nonzero.
   */
  if (wafer_profile_is_trace_capture()) {
    wafer_profile_ncc_enable_before = wafer_profile_read_pmu32(GR_PMU_EN);
    *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_PMU_BASE + GR_PMU_EN) =
        wafer_profile_ncc_enable_before | UINT32_C(0x1);
    wafer_profile_ncc_enable_owned = 1U;
    wafer_profile_dte_enable_before = wafer_profile_read_dte_pmu32(DTE_PMU_EN);
    *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_DTE_PMU_BASE + DTE_PMU_EN) =
        wafer_profile_dte_enable_before | UINT32_C(0x3);
    wafer_profile_dte_enable_owned = 1U;
  }

  wafer_profile_snapshot(&wafer_profile_header->pmu_before);
  wafer_profile_header->summary_validity |=
      WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED;
  if (wafer_profile_header->pmu_before.stable_mask ==
      (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U)
    wafer_profile_header->summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE;
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine) {
    wafer_profile_last_engine_counters[engine] =
        wafer_profile_header->pmu_before.counters[engine + 2U];
    if ((wafer_profile_header->pmu_before.stable_mask &
         (UINT32_C(1) << (engine + 2U))) != 0U)
      wafer_profile_last_engine_counter_valid_mask |= UINT32_C(1) << engine;
  }

  if (wafer_profile_is_trace_capture()) {
    uint64_t setup_end_cycle = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.entry_setup_cycles,
        setup_begin_cycle, setup_end_cycle);
  }
  wafer_profile_header->entry_begin_cycle = wafer_profile_cycle();
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine)
    wafer_profile_last_engine_read_begin_cycles[engine] =
        wafer_profile_header->entry_begin_cycle;
}

void wafer_tx81_profile_entry_begin_from_config(uint64_t buffer_address) {
  /*
   * Rejecting a malformed next-launch configuration must not leave the prior
   * invocation's DDR record bound.  Otherwise later TsmExecute wrappers could
   * write through a stale address before entry_end observes the rejection.
   */
  wafer_profile_pending_setup_begin_cycle = 0;
  uint64_t setup_begin_cycle = wafer_profile_cycle();
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
  wafer_profile_pending_setup_begin_cycle = setup_begin_cycle;
  wafer_tx81_profile_entry_begin(buffer_address, config.record_bytes,
                                 config.tile_id, config.flags);
}

void wafer_tx81_profile_site_begin(uint32_t site_id) {
  if (wafer_profile_header == NULL)
    return;
  /*
   * Count carries the same site hooks as Trace so its zero-storage
   * next_sequence is an exact capacity preflight for TARGET_SITE containers
   * and their children.  It must not populate the Trace-only cost summary.
   */
  if ((wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0U) {
    if (wafer_profile_header->active_site_depth != 0U ||
        site_id == WAFER_TX81_PROFILER_INVALID_SITE_ID) {
      wafer_profile_mark_protocol_error();
      return;
    }
    wafer_profile_header->active_site_id = site_id;
    wafer_profile_header->next_sub_index = 0;
    wafer_profile_header->active_site_depth = 1;
    ++wafer_profile_header->next_sequence;
    return;
  }
  uint64_t hook_begin_cycle = wafer_profile_cycle();
  if (wafer_profile_header->active_site_depth != 0U ||
      site_id == WAFER_TX81_PROFILER_INVALID_SITE_ID) {
    wafer_profile_mark_protocol_error();
    uint64_t hook_end_cycle = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.site_hook_cycles,
        hook_begin_cycle, hook_end_cycle);
    return;
  }
  wafer_profile_header->active_site_id = site_id;
  wafer_profile_header->next_sub_index = 0;
  wafer_profile_header->active_site_depth = 1;
  wafer_profile_active_site_first_event = wafer_profile_header->event_count;
  uint64_t hook_end_cycle = wafer_profile_cycle();
  wafer_profile_add_cost(&wafer_profile_header->cost_summary.site_hook_cycles,
                         hook_begin_cycle, hook_end_cycle);
  if (wafer_profile_is_trace_capture()) {
    (void)wafer_profile_allocate_event(
        WAFER_TX81_PROFILER_ENGINE_NONE,
        WAFER_TX81_PROFILER_EVENT_TARGET_SITE, 0U);
  }
  wafer_profile_active_site_begin_cycle = wafer_profile_cycle();
}

void wafer_tx81_profile_site_end(uint32_t site_id) {
  if (wafer_profile_header == NULL)
    return;
  if ((wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0U) {
    if (wafer_profile_header->active_site_depth != 1U ||
        wafer_profile_header->active_site_id != site_id) {
      wafer_profile_mark_protocol_error();
      return;
    }
    wafer_profile_header->active_site_id =
        WAFER_TX81_PROFILER_INVALID_SITE_ID;
    wafer_profile_header->next_sub_index = 0;
    wafer_profile_header->active_site_depth = 0;
    return;
  }
  uint64_t site_end_cycle = wafer_profile_cycle();
  if (wafer_profile_header->active_site_depth != 1U ||
      wafer_profile_header->active_site_id != site_id) {
    wafer_profile_mark_protocol_error();
    uint64_t hook_end_cycle = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.site_hook_cycles,
        site_end_cycle, hook_end_cycle);
    return;
  }
  uint32_t first = wafer_profile_active_site_first_event;
  uint32_t end = wafer_profile_header->event_count;
  if (first > end)
    wafer_profile_mark_protocol_error();
  else
    for (uint32_t index = first; index < end; ++index) {
      volatile WaferTx81ProfilerTSMCallEvent *event =
          &wafer_profile_events[index];
      if ((event->metadata & WAFER_TX81_PROFILER_EVENT_SITE_VALID) == 0U ||
          event->site_id != site_id) {
        wafer_profile_mark_protocol_error();
        continue;
      }
      event->site_begin_cycle = wafer_profile_active_site_begin_cycle;
      event->site_end_cycle = site_end_cycle;
      event->metadata |= WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID;
    }
  wafer_profile_header->active_site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  wafer_profile_header->next_sub_index = 0;
  wafer_profile_header->active_site_depth = 0;
  wafer_profile_active_site_begin_cycle = 0;
  wafer_profile_active_site_first_event = end;
  uint64_t hook_end_cycle = wafer_profile_cycle();
  wafer_profile_add_cost(&wafer_profile_header->cost_summary.site_hook_cycles,
                         site_end_cycle, hook_end_cycle);
}

static uint32_t wafer_profile_worker_metadata(uint32_t inter_type) {
  uint32_t worker =
      (inter_type & WAFER_TX81_NCC_WORKER_INTER_TYPE_MASK) >>
      WAFER_TX81_NCC_WORKER_INTER_TYPE_SHIFT;
  if (worker >= WAFER_TX81_PROFILER_WORKERS)
    return 0;
  return worker | WAFER_TX81_PROFILER_EVENT_WORKER_VALID;
}

static uint64_t wafer_profile_execute_ncc(void *instruction,
                                          uint32_t inter_type,
                                          uint8_t engine) {
  if (wafer_profile_header == NULL)
    return TsmExecute(instruction);
  if ((wafer_profile_header->flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) !=
      0U) {
    ++wafer_profile_header->next_sequence;
    return TsmExecute(instruction);
  }
  if (!wafer_profile_is_trace_capture())
    return TsmExecute(instruction);

  wafer_profile_observe_ncc_activity();
  uint32_t metadata = wafer_profile_worker_metadata(inter_type);
  if (engine < WAFER_PROFILE_NCC_ENGINE_COUNT &&
      (wafer_profile_last_engine_counter_valid_mask &
       (UINT32_C(1) << engine)) != 0U)
    metadata |= WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;
  uint32_t prior_event = UINT32_MAX;
  if (engine < WAFER_PROFILE_NCC_ENGINE_COUNT)
    prior_event = wafer_profile_latest_engine_events[engine];
  if (prior_event != UINT32_MAX &&
      prior_event < wafer_profile_header->event_count)
    metadata |= WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS;
  uint32_t event_index = wafer_profile_allocate_event(
      engine, WAFER_TX81_PROFILER_EVENT_NCC_COMMAND, metadata);
  if (prior_event != UINT32_MAX &&
      prior_event < wafer_profile_header->event_count) {
    uint64_t bookkeeping_begin = wafer_profile_cycle();
    wafer_profile_events[prior_event].metadata |=
        WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS;
    uint64_t bookkeeping_end = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
        bookkeeping_begin, bookkeeping_end);
  }
  if (engine < WAFER_PROFILE_NCC_ENGINE_COUNT)
    wafer_profile_latest_engine_events[engine] = event_index;
  uint64_t operation_begin_cycle = wafer_profile_cycle();
  uint64_t raw_return = TsmExecute(instruction);
  uint64_t operation_end_cycle = wafer_profile_cycle();
  if (event_index != UINT32_MAX &&
      event_index < wafer_profile_header->event_count) {
    uint64_t bookkeeping_begin = wafer_profile_cycle();
    wafer_profile_events[event_index].operation_begin_cycle =
        operation_begin_cycle;
    wafer_profile_events[event_index].operation_end_cycle = operation_end_cycle;
    wafer_profile_events[event_index].metadata |=
        WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
    uint64_t bookkeeping_end = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
        bookkeeping_begin, bookkeeping_end);
  }
  wafer_profile_observe_ncc_activity();
  return raw_return;
}

/*
 * Profile-only local completion polling samples execution counters while the
 * real engines drain.  Production CRT keeps using the untouched
 * TsmWaitfinish path.
 */
static uint64_t wafer_profile_cost_delta(uint64_t before, uint64_t after) {
  if (after < before) {
    wafer_profile_mark_protocol_error();
    return UINT64_MAX;
  }
  return after - before;
}

static void wafer_profile_complete_wait_event(
    uint32_t event_index, uint64_t operation_begin_cycle,
    uint64_t operation_end_cycle, uint32_t poll_count,
    uint64_t ncc_sample_before, uint64_t event_bookkeeping_before,
    uint64_t status_poll_before) {
  uint64_t total =
      wafer_profile_cost_delta(operation_begin_cycle, operation_end_cycle);
  uint64_t ncc = wafer_profile_cost_delta(
      ncc_sample_before,
      wafer_profile_header->cost_summary.ncc_pmu_sample_cycles);
  uint64_t event_bookkeeping = wafer_profile_cost_delta(
      event_bookkeeping_before,
      wafer_profile_header->cost_summary.event_bookkeeping_cycles);
  uint64_t status =
      wafer_profile_cost_delta(status_poll_before,
                               wafer_profile_header->cost_summary
                                   .status_poll_cycles);
  uint64_t nested = 0;
  if (ncc == UINT64_MAX || event_bookkeeping == UINT64_MAX ||
      status == UINT64_MAX || ncc > UINT64_MAX - event_bookkeeping ||
      ncc + event_bookkeeping > UINT64_MAX - status) {
    wafer_profile_mark_protocol_error();
  } else {
    nested = ncc + event_bookkeeping + status;
    if (nested > total)
      wafer_profile_mark_protocol_error();
    else
      wafer_profile_add_cost(
          &wafer_profile_header->cost_summary
               .completion_loop_bookkeeping_cycles,
          0, total - nested);
  }

  if (event_index == UINT32_MAX ||
      event_index >= wafer_profile_header->event_count)
    return;
  uint64_t bookkeeping_begin = wafer_profile_cycle();
  volatile WaferTx81ProfilerTSMCallEvent *event =
      &wafer_profile_events[event_index];
  event->operation_begin_cycle = operation_begin_cycle;
  event->operation_end_cycle = operation_end_cycle;
  event->observation_count = poll_count;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
  uint64_t bookkeeping_end = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
      bookkeeping_begin, bookkeeping_end);
}

static void wafer_profile_clear_completed_worker(uint32_t worker) {
  uint64_t bookkeeping_begin = wafer_profile_cycle();
  for (uint32_t engine = 0; engine < WAFER_PROFILE_NCC_ENGINE_COUNT; ++engine) {
    uint32_t event_index = wafer_profile_latest_engine_events[engine];
    if (event_index == UINT32_MAX ||
        event_index >= wafer_profile_header->event_count)
      continue;
    volatile WaferTx81ProfilerTSMCallEvent *event =
        &wafer_profile_events[event_index];
    if ((event->metadata & WAFER_TX81_PROFILER_EVENT_WORKER_VALID) != 0U &&
        (event->metadata & WAFER_TX81_PROFILER_EVENT_WORKER_MASK) == worker &&
        (event->metadata &
         WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS) == 0U) {
      wafer_profile_latest_engine_events[engine] = UINT32_MAX;
    }
  }
  uint64_t bookkeeping_end = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
      bookkeeping_begin, bookkeeping_end);
}

static uint32_t
wafer_profile_wait_ncc_worker_completion(uint32_t worker) {
  if (!wafer_profile_is_trace_capture()) {
    if (wafer_profile_header != NULL &&
        (wafer_profile_header->flags &
         WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0U)
      ++wafer_profile_header->next_sequence;
    return TsmWaitfinish_bywork(worker);
  }

  uint32_t metadata = WAFER_TX81_PROFILER_EVENT_WAIT_WORKER;
  if (worker < WAFER_TX81_PROFILER_WORKERS)
    metadata |= worker | WAFER_TX81_PROFILER_EVENT_WORKER_VALID;
  else
    wafer_profile_mark_protocol_error();
  uint32_t event_index = wafer_profile_allocate_event(
      WAFER_TX81_PROFILER_ENGINE_NONE,
      WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT, metadata);
  uint64_t ncc_sample_before =
      wafer_profile_header->cost_summary.ncc_pmu_sample_cycles;
  uint64_t event_bookkeeping_before =
      wafer_profile_header->cost_summary.event_bookkeeping_cycles;
  uint64_t status_poll_before =
      wafer_profile_header->cost_summary.status_poll_cycles;
  uint64_t operation_begin_cycle = wafer_profile_cycle();
  uint32_t poll_count = 0;
  for (;;) {
    wafer_profile_observe_ncc_activity();
    uint64_t poll_begin_cycle = wafer_profile_cycle();
    uint32_t done = TsmGetCsrTaskstatus_bywork(worker);
    uint64_t poll_end_cycle = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.status_poll_cycles,
        poll_begin_cycle, poll_end_cycle);
    if (poll_count == UINT32_MAX)
      wafer_profile_mark_protocol_error();
    else
      ++poll_count;
    if (done == 1U)
      break;
  }
  wafer_profile_observe_ncc_activity();
  uint64_t operation_end_cycle = wafer_profile_cycle();
  wafer_profile_complete_wait_event(
      event_index, operation_begin_cycle, operation_end_cycle, poll_count,
      ncc_sample_before, event_bookkeeping_before, status_poll_before);
  wafer_profile_clear_completed_worker(worker);
  return 1U;
}

static void wafer_profile_direct_dte_begin(uint64_t direct_event,
                                           uint8_t aggregate_kind) {
  if (wafer_profile_header == NULL)
    return;
  if ((aggregate_kind != WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE &&
       aggregate_kind != WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT) ||
      (aggregate_kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE &&
       direct_event != WAFER_DIRECT_DTE_SEND_EVENT)) {
    wafer_profile_mark_protocol_error();
    return;
  }
  uint32_t role = 0;
  if (direct_event == WAFER_DIRECT_DTE_SEND_EVENT)
    role = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND;
  else if (direct_event >= WAFER_DIRECT_DTE_RECV_EVENT_BASE &&
           direct_event < WAFER_DIRECT_DTE_RECV_EVENT_BASE +
                              WAFER_DIRECT_DTE_MAX_RECEIVERS)
    role = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV;
  else
    wafer_profile_mark_protocol_error();
  wafer_profile_direct_role_metadata = role;
  wafer_profile_direct_aggregate_kind = aggregate_kind;
  if ((wafer_profile_header->flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) !=
      0U) {
    ++wafer_profile_header->next_sequence;
    return;
  }
  if (!wafer_profile_is_trace_capture())
    return;
  if (wafer_profile_direct_event_index != UINT32_MAX) {
    wafer_profile_mark_protocol_error();
    return;
  }
  wafer_profile_observe_ncc_activity();
  wafer_profile_direct_event_index = wafer_profile_allocate_event(
      WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE, aggregate_kind, role);
  uint64_t sample_begin_cycle = wafer_profile_cycle();
  wafer_profile_direct_start_valid_mask = 0;
  wafer_profile_direct_start_counters[0] =
      wafer_profile_read_dte_pmu64(DTE_PMU_CH0_L_EXE_TIME, UINT32_C(1) << 0,
                                   &wafer_profile_direct_start_valid_mask);
  wafer_profile_direct_start_counters[1] =
      wafer_profile_read_dte_pmu64(DTE_PMU_CH1_L_EXE_TIME, UINT32_C(1) << 1,
                                   &wafer_profile_direct_start_valid_mask);
  uint64_t sample_end_cycle = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.dte_pmu_sample_cycles,
      sample_begin_cycle, sample_end_cycle);
  wafer_profile_direct_observation_begin_cycle = sample_begin_cycle;
  wafer_profile_direct_operation_begin_cycle = wafer_profile_cycle();
}

static void wafer_profile_direct_dte_end(void) {
  if (wafer_profile_header == NULL ||
      wafer_profile_direct_event_index == UINT32_MAX)
    return;
  uint64_t operation_end_cycle = wafer_profile_cycle();
  uint64_t sample_begin_cycle = wafer_profile_cycle();
  uint32_t end_valid_mask = 0;
  uint64_t end_counters[2] = {
      wafer_profile_read_dte_pmu64(DTE_PMU_CH0_L_EXE_TIME, UINT32_C(1) << 0,
                                   &end_valid_mask),
      wafer_profile_read_dte_pmu64(DTE_PMU_CH1_L_EXE_TIME, UINT32_C(1) << 1,
                                   &end_valid_mask)};
  uint64_t sample_end_cycle = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.dte_pmu_sample_cycles,
      sample_begin_cycle, sample_end_cycle);
  if (wafer_profile_direct_event_index < wafer_profile_header->event_count) {
    uint64_t bookkeeping_begin = wafer_profile_cycle();
    volatile WaferTx81ProfilerTSMCallEvent *event =
        &wafer_profile_events[wafer_profile_direct_event_index];
    event->observed_begin_cycle =
        wafer_profile_direct_observation_begin_cycle;
    event->observed_end_cycle = sample_end_cycle;
    event->observation_count = 2U;
    event->operation_begin_cycle =
        wafer_profile_direct_operation_begin_cycle;
    event->operation_end_cycle = operation_end_cycle;
    event->counter_delta = 0;
    event->metadata |=
        WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
        WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
    if (wafer_profile_direct_start_valid_mask == UINT32_C(0x3) &&
        end_valid_mask == UINT32_C(0x3) &&
        end_counters[0] >= wafer_profile_direct_start_counters[0] &&
        end_counters[1] >= wafer_profile_direct_start_counters[1]) {
      uint64_t channel0_delta =
          end_counters[0] - wafer_profile_direct_start_counters[0];
      uint64_t channel1_delta =
          end_counters[1] - wafer_profile_direct_start_counters[1];
      if (channel0_delta <= UINT64_MAX - channel1_delta) {
        event->counter_delta = channel0_delta + channel1_delta;
        event->metadata |= WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID;
        if (event->counter_delta != 0U)
          event->metadata |=
              WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
      }
    }
    uint64_t bookkeeping_end = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
        bookkeeping_begin, bookkeeping_end);
  }
  wafer_profile_direct_event_index = UINT32_MAX;
  wafer_profile_direct_role_metadata = 0;
  wafer_profile_direct_aggregate_kind = 0;
  wafer_profile_direct_operation_begin_cycle = 0;
  wafer_profile_direct_observation_begin_cycle = 0;
  wafer_profile_observe_ncc_activity();
}

static uint32_t wafer_profile_direct_dte_phase_begin(uint8_t kind) {
  if (wafer_profile_header == NULL)
    return UINT32_MAX;
  if ((wafer_profile_header->flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) !=
      0U) {
    ++wafer_profile_header->next_sequence;
    return UINT32_MAX;
  }
  if (!wafer_profile_is_trace_capture())
    return UINT32_MAX;
  uint32_t is_issue_phase =
      kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT ||
      kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE;
  uint32_t is_wait_phase =
      kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT ||
      kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP;
  if (wafer_profile_direct_event_index == UINT32_MAX ||
      wafer_profile_direct_role_metadata == 0U ||
      ((!is_issue_phase || wafer_profile_direct_aggregate_kind !=
                               WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE) &&
       (!is_wait_phase || wafer_profile_direct_aggregate_kind !=
                              WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT))) {
    wafer_profile_mark_protocol_error();
    return UINT32_MAX;
  }
  uint32_t event_index = wafer_profile_allocate_event(
      WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE, kind,
      wafer_profile_direct_role_metadata);
  if (event_index != UINT32_MAX &&
      event_index < wafer_profile_header->event_count)
    wafer_profile_events[event_index].operation_begin_cycle =
        wafer_profile_cycle();
  return event_index;
}

static void wafer_profile_direct_dte_phase_end(uint32_t event_index) {
  uint64_t operation_end_cycle = wafer_profile_cycle();
  if (!wafer_profile_is_trace_capture() || event_index == UINT32_MAX)
    return;
  if (event_index >= wafer_profile_header->event_count) {
    wafer_profile_mark_protocol_error();
    return;
  }
  uint64_t bookkeeping_begin = wafer_profile_cycle();
  wafer_profile_events[event_index].operation_end_cycle = operation_end_cycle;
  wafer_profile_events[event_index].metadata |=
      WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
  uint64_t bookkeeping_end = wafer_profile_cycle();
  wafer_profile_add_cost(
      &wafer_profile_header->cost_summary.event_bookkeeping_cycles,
      bookkeeping_begin, bookkeeping_end);
}

void wafer_tx81_profile_entry_end(void) {
  if (wafer_profile_header == NULL)
    return;
  if (wafer_profile_header->active_site_depth != 0U)
    wafer_profile_mark_protocol_error();
  if (wafer_profile_direct_event_index != UINT32_MAX)
    wafer_profile_mark_protocol_error();

  wafer_profile_observe_ncc_activity();
  wafer_profile_header->entry_end_cycle = wafer_profile_cycle();
  uint64_t teardown_begin_cycle = wafer_profile_header->entry_end_cycle;
  uint32_t trace_capture =
      (wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0U &&
      (wafer_profile_header->flags &
       WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) == 0U;
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
  if (wafer_profile_dte_enable_owned != 0U)
    *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_DTE_PMU_BASE + DTE_PMU_EN) =
        wafer_profile_dte_enable_before;
  if (wafer_profile_ncc_enable_owned != 0U)
    *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_PMU_BASE + GR_PMU_EN) =
        wafer_profile_ncc_enable_before;
  if (trace_capture != 0U) {
    uint64_t teardown_end_cycle = wafer_profile_cycle();
    wafer_profile_add_cost(
        &wafer_profile_header->cost_summary.entry_teardown_cycles,
        teardown_begin_cycle, teardown_end_cycle);
  }
  wafer_profile_cache_range(wafer_profile_header->buffer_address,
                            wafer_profile_header->buffer_bytes, 0U);
  wafer_profile_reset_binding();
}

#undef WAFER_PROFILE_PMU_STABLE_RETRIES
#undef WAFER_PROFILE_PMU_WORKER_STRIDE
#undef WAFER_PROFILE_NCC_ENGINE_COUNT
#undef WAFER_PROFILE_DTE_PMU_BASE
#undef WAFER_PROFILE_PMU_BASE
