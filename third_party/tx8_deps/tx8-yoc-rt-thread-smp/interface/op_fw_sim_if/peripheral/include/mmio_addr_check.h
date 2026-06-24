#ifndef __MMIO_ADDR_CHECK_H__
#define __MMIO_ADDR_CHECK_H__

#ifdef USING_RISCV
#ifdef __DEBUG__
uint64_t set_gddr_info(uint64_t gaddr, uint64_t size);
#else
#define set_gddr_info(gaddr, size)    gaddr
#endif
#else
uint64_t set_gddr_info(uint64_t gaddr, uint64_t size);
#endif

uint64_t get_gddr_area_min();
uint64_t get_gddr_area_max();
int8_t *get_ddr_memory_mapping(uint64_t addr);
int8_t *get_ddr_memory_mapping_with_size(uint64_t addr, int32_t range_size);
#endif