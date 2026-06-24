#ifndef _MOD_PROTOCOL_WITH_SCORE_H_
#define _MOD_PROTOCOL_WITH_SCORE_H_
#include <stdint.h>
uint32_t StartProfilingOnScore0(uint16_t profiling_data_type,uint64_t addr_base, uint32_t length);
uint32_t  StartProfilingOnScore1(uint16_t profiling_data_type,uint64_t addr_base, uint32_t length);
uint32_t StopProfilingOnScore0();
uint32_t StopProfilingOnScore1();
#endif