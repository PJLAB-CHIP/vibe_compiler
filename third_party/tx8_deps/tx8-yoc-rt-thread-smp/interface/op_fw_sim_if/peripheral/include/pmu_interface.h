#ifndef __PMU_INTERFACE_H__
#define __PMU_INTERFACE_H__
    #include "stddef.h"
    #include "stdint.h"
    #include "stdbool.h"
    void set_pmu_reg(size_t index, bool value);
    uint64_t get_pmu_reg(size_t index);
    uint64_t get_dte_pmu_reg(size_t index);
    
    void set_cycle_mode(bool mode);
    bool get_cycle_mode();
#endif