/* Target-only clock, register and cache access for the profiler. */
#define WAFER_PROFILE_PMU_BASE UINT64_C(0x590000)
#define WAFER_PROFILE_DTE_PMU_BASE UINT64_C(0x400000)

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

static void wafer_profile_write_pmu32(uint32_t offset, uint32_t value) {
  *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_PMU_BASE + offset) = value;
}

static void wafer_profile_write_dte_pmu32(uint32_t offset, uint32_t value) {
  *(volatile uint32_t *)(uintptr_t)(WAFER_PROFILE_DTE_PMU_BASE + offset) =
      value;
}

static void wafer_profile_fence_io(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

#undef WAFER_PROFILE_DTE_PMU_BASE
#undef WAFER_PROFILE_PMU_BASE
