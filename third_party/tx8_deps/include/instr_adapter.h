#ifndef INSTR_ADAPTER_H
#define INSTR_ADAPTER_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef USING_RISCV
#include <stdbool.h>
#endif

#include "instr_def.h"
#include "instr_adapter_plat.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifndef USING_RISCV
#define __CHECK_INSTR__
#endif
//#define __PLAT_FREERTOS__
// #define RECORD_INSTR_INVALID
#define SPM_LOWER_BOUND 0
#define SPM_UPPER_BOUND 0x2EFFFF
#define DDR_LOWER_BOUND 0x280000000
#define IS_WITHIN_SPM_BOUND(value) (((value) >= SPM_LOWER_BOUND) && ((value) <= SPM_UPPER_BOUND))
#define IS_WITHIN_DDR_BOUND(value) ((value) >= DDR_LOWER_BOUND)
// 设置 times (0-7 位)
#define TIMES_INVALID_OFFET 0
// 设置 last_invalid_barrier_id (8-35 位)
#define LAST_INVALID_BARRIER 8
// 设置 first_invalid_barrier_id (36-63 位)
#define FIRST_INVALID_BARRIER 36

static volatile uint64_t g_instr_param_buf_wptr = 0;

typedef struct InstrParamHead {
    uint32_t type;
    uint64_t length;
} InstrParamHead;

typedef struct InstrCTParam {
    InstrParamHead head;
    TsmArithInstr ct_param;
    uint32_t group_id;
} InstrCTParam;

typedef struct InstrNEParam {
    InstrParamHead head;
    TsmNeInstr ne_param;
    uint32_t group_id;
} InstrNEParam;

typedef struct InstrRDMAParam {
    InstrParamHead head;
    TsmRdmaInstr rdma_param;
    uint32_t group_id;
} InstrRDMAParam;

typedef struct InstrWDMAParam {
    InstrParamHead head;
    TsmWdmaInstr wdma_param;
    uint32_t group_id;
} InstrWDMAParam;

typedef struct InstrTDMAParam {
    InstrParamHead head;
    TD_Param tdma_param;
    uint32_t group_id;
} InstrTDMAParam;

typedef struct InstrInvalidInfo {
    volatile uint64_t ne_error_info;
    volatile uint64_t ct_error_info;
    volatile uint64_t td_error_info;
    volatile uint64_t rdma_error_info;
    volatile uint64_t wdma_error_info;
} InstrInvalidInfo;

/*
  # 0-shape(nhwc)     # 1-wshape(Kx,Ky,f,c)  # 2-bias  # 3-stride(Kx,Ky,Sx,Sy)
  # 4-pad(top,bottom,left,right) # 5- dilation(0,0,dilation[0],dilation[1])
*/
/*=================================TDMA=================================*/

/*=================================RDMA WDMA=================================*/

/*=================================Scale=================================*/

/*=================================run=================================*/
uint32_t __execute_ne(TsmNeInstr *instr);
uint32_t __execute_ct(TsmArithInstr *instr);
uint32_t __execute_td(TsmDataMoveInstr *instr);
uint32_t __execute_rdma(TsmRdmaInstr *instr);
uint32_t __execute_wdma(TsmWdmaInstr *instr);
void __execute_sc(SC_Param *instr);
uint64_t TsmExecute(void *instr);

#ifndef USING_RISCV
uint64_t instr_tick_cc(void *instr);
bool get_cycle_mode();
void set_cycle_mode(bool valid);
#endif

/*=================================debug=================================*/
void set_device_ddr_base(uint64_t base);
uint64_t get_device_ddr_base();

#ifdef __cplusplus
}
#endif

#endif /*INSTR_ADAPTER_H*/
