#include <rthw.h>
#include "csi_core.h"
void rt_hw_cpu_icache_enable(void) {
    csi_icache_enable();
}

void rt_hw_cpu_icache_disable(void) {
    csi_icache_disable();
}

// rt_base_t rt_hw_cpu_icache_status(void);

void rt_hw_cpu_icache_ops(int ops, void* addr, int size) {
    if (ops == RT_HW_CACHE_INVALIDATE) {
        csi_icache_invalid();
    }
}

void rt_hw_cpu_dcache_enable(void) {
    csi_dcache_enable();
}

void rt_hw_cpu_dcache_disable(void) {
    csi_dcache_disable();
}

// rt_base_t rt_hw_cpu_dcache_status(void);

void rt_hw_cpu_dcache_ops(int ops, void* addr, int size) {
    if (ops == RT_HW_CACHE_FLUSH) {     // fluash = clean + invalid    
        csi_dcache_clean_invalid_range((unsigned long *)addr, (unsigned int)size);
    } else if (ops == RT_HW_CACHE_INVALIDATE) {
        csi_dcache_invalid_range((unsigned long *)addr, (unsigned int)size);
    }
}