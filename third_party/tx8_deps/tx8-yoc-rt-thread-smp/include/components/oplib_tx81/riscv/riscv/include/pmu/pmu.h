#ifndef TX81_PMU_H
#define TX81_PMU_H

#include <stdint.h>
#include "pmu/pmu_reg.h"
#include "lib_log.h"

/*
 * 1. NE PMU
 * pmu_ncc_en();
 * pmu_ncc_ne_st record={0}；//使用局部/全局变量
 * pmu_ncc_ne_start(&record);
 * Your NE code;
 * pmu_ncc_ne_end(&record);
 * pmu_ncc_disable()
 *
 *
 * 2. # NCC TIMER: w0/w1/w2 x time0/time1/time2/time01/time012
 * 2-1 使用数据结构的计时器API，可以向ddr存储
 * # Notice: ncc timer is uint32_t, which can wrap around, so it onle supports a limited amount of time(ns).
 * pmu_ncc_w0_user_time0_en();
 * pmu_ncc_w0_user_time0_clr();
 * PmuNccUserTime ncc_time;
 * pmu_ncc_w0_user_time0_start(&ncc_time);
 * Your code;
 * pmu_ncc_w0_ser_time0_end(&ncc_time);
 * pmu_ncc_w0_user_time0_disable();
 * __LOG__(KCORE_LOG_DEBUG, "(%u)\n", ncc_time.time);
 * // todo : report_to_host(&ncc_time)
 *
 * 2-2 使用返回值的计时器API
 * pmu_ncc_w0_user_time0_en();
 * pmu_ncc_w0_user_time0_clr();
 * uint32_t t0 = pmu_ncc_w0_user_time0();
 * uint32_t t1 = pmu_ncc_w0_user_time0();
 * __LOG__(KCORE_LOG_DEBUG, "(%u)\n", t1-t0);
 * pmu_ncc_w0_user_time0_disable();
 *
 * 2 指令计时器
 * pmu_ncc_w0_user_time_en();  // en  timer 0/1/2/01/12
 * pmu_ncc_w0_user_time_clr(); // clr timer 0/1/2/01/12
 * 假设代码逻辑如下: 算子中存在rdma/ne/ct/dte/wdma四种指令，ne/ct/dte可能处于并行
 * do_rdma()
 * {
 *      ....
 *      pmu_ncc_record(&ncctime); //
 * 记录一次ncc_window_x_x时间;最大调用次数为PMU_NCC_MAX_RECORD;与初始ncc_window的差表示下一个trigger指令之前的kcore运行时间。
 *      config_and_tigger_reg();
 *      ....
 * }
 * do_ne()
 * {
 *      ....
 *      pmu_ncc_record(&ncctime);
 *      config_and_tigger_reg();
 *      ....
 * }
 * do_wdma()
 * {
 *      ....
 *      pmu_ncc_record(&ncctime);
 *      config_and_tigger_reg();
 *      ....
 * }
 * PmuNcc ncctime; //uint64_t ncctime = 0x400;  //存入SPM时，速度更快.也可以用局部变量
 *
 * func0() {
 *      pmu_ncc_en();
 *      pmu_ncc_clr();
 *
 *      pmu_ncc_start(); //记录本算子运行数据
 *      ....
 *      do_rdma();
 *      ....
 *      do_ne();
 *      do_ct();
 *      ....
 *      do_wdma();
 *      ....
 *      pmu_ncc_end();//记录本算子运行数据结束
 *      pmu_ncc_disable();
 * }
 * # ncctime.ncc_window_x_x             : 统计窗口 statistics_window
 * # ncctime.ncc_window_record_x_x      : PMU_NCC_MAX_RECORD次的统计窗口值
 * # ncctime.ne_exe_time_x_x            : NE单元执行时间
 * # ncctime.rdma_exe_time_x_x          : rdma单元执行时间
 * # ncctime.wdma_exe_time_x_x          : wdma单元执行时间
 * # ncctime.tdma_exe_time_x_x          : tdma单元执行时间
 * # ncctime.ct_exe_time_x_x            : ct各硬件单元执行时间
 * # ncctime.fu_exe_time_x_x            : 各个单元工作时间的并集
 * # ncctime.worker0_ne_inst_nums       : ne单元执行指令个数
 * # ncctime.worker0_rdma_inst_nums     : rdma单元执行指令个数
 * # ncctime.worker0_wdma_inst_nums     : wdma单元执行指令个数
 * # ncctime.worker0_tdma_inst_nums     : tdma单元执行指令个数
 * # ncctime.worker0_ct_inst_nums       : ct单元执行指令个数
 * # ncctime.ne_blocking_time_x_x       : NE单元queue反压时间
 * # ncctime.rdma_blocking_time_x_x     : rdma单元queue反压时间
 * # ncctime.wdma_blocking_time_x_x     : wdma单元queue反压时间
 * # ncctime.tdma_blocking_time_x_x     : tdma单元queue反压时间
 * # ncctime.ct_blocking_time_x_x       : ct各硬件单元queue反压时间
 *
 * ncc_window
 * |----------------------------------------------------------------------------------------|ncc_window
 * |--------|                                                                               |record[0],
 * Rdma配置之前的kcore运行时间 |        |--------------------------|Rdma |rdma_exe_time + rdma_blocking_time |
 * |---------------|                                                               |record[1], NE配置之前的kcore运行时间
 * |                        |Ncc_queue|-----------------|NE                                 |ne_exe_time +
 * ne_blocking_time(如果Rdma和NE不能并行，fu=0) |                        |---------------| |record[2],
 * CT配置之前的kcore运行时间 |                                        |--------------------------|CT |ct_exe_time +
 * ct_blocking_time(如果NE和CT能够并行) | | |                                        |---------------| |record[3],
 * Wdma配置之前的kcore运行时间 |                                                        |Ncc_queue |--------------|Wdma
 * |wdma_exe_time + wdma_blocking_time(如果Wdma和CT不能并行，fu=0) | |-----|record[4]，Ncc窗口结束之前的kcore运行时间
 * |----------------------------------------------------------------------------------------|ncc_window
 *
 * 3. SPM-DTE PMU
 * pmu_spm_en_all()
 * pmu_spm_en_clr()
 * pmu_spm_dte_all_start(PmuSpmDte *record)
 * pmu_spm_dte_all_end(PmuSpmDte *record)
 * pmu_spm_disable()
 */
#define PMU_SPM_GET_U32(OFFSET) (*((volatile uint32_t *)((intptr_t)(OFFSET + SCT_REG_BASE_SPM))))
#define PMU_NCC_GET_U32(OFFSET) (*((volatile uint32_t *)((intptr_t)(OFFSET + SCT_REG_BASE_NCC))))
#define PMU_DTE_GET_U32(OFFSET) (*((volatile uint32_t *)((intptr_t)(OFFSET + SCT_REG_BASE_DTE))))
#define PMU_FUNC_NAME_LEN_MAX 48
#define PMU_NCC_MAX_RECORD 32
#define PMU_NCC_WORKER_OFFSET 0x30
#define PROF_TYPE_NCC 1
#define PROF_TYPE_LSU 2
#define PROF_TYPE_DTE 4
#define PROF_TYPE_SPM 8

enum PMU_TYPE_INFO {
    PMU_TYPE_SPM_DTE,
    PMU_TYPE_SPM_LSU,
    PMU_TYPE_DTE,
    PMU_TYPE_NCC,
    PMU_TYPE_NCC_CT,
    PMU_TYPE_NCC_NE,
    PMU_TYPE_NCC_RDMA,
    PMU_TYPE_NCC_WDMA,
    PMU_TYPE_NCC_TDMA,
    PMU_TYPE_NCC_SCALAR,
    PMU_TYPE_NCC_USER_TIME,
    PMU_TYPE_END
};

typedef struct PmuTLVHead {
    uint32_t pmu_type;
    uint32_t length;
} PmuTLVHead;

typedef struct PmuSpmLsu {
    PmuTLVHead head;
    uint32_t func_id;
    uint32_t t2_rdma_31_0;
    uint32_t t2_rdma_63_32;
    uint32_t t2_wdma_31_0;
    uint32_t t2_wdma_63_32;
    uint32_t t2_rdma_31_0_end;
    uint32_t t2_rdma_63_32_end;
    uint32_t t2_wdma_31_0_end;
    uint32_t t2_wdma_63_32_end;
    uint32_t t3_rdma_31_0;
    uint32_t t3_rdma_63_32;
    uint32_t t3_wdma_31_0;
    uint32_t t3_wdma_63_32;
    uint32_t t3_rdma_31_0_end;
    uint32_t t3_rdma_63_32_end;
    uint32_t t3_wdma_31_0_end;
    uint32_t t3_wdma_63_32_end;
    uint64_t t2_rdma;
    uint64_t t2_wdma;
    uint64_t t3_rdma;
    uint64_t t3_wdma;
} PmuSpmLsu;

typedef struct PmuSpmDte {
    PmuTLVHead head;
    uint32_t func_id;
    uint32_t t0_31_0;
    uint32_t t0_63_32;
    uint32_t t1_31_0;
    uint32_t t1_63_32;
    uint32_t t2_31_0;
    uint32_t t2_63_32;
    uint32_t t3_31_0;
    uint32_t t3_63_32;
    uint32_t t4_31_0;
    uint32_t t4_63_32;
    uint32_t t5_31_0;
    uint32_t t5_63_32;
    uint32_t t0_31_0_end;
    uint32_t t0_63_32_end;
    uint32_t t1_31_0_end;
    uint32_t t1_63_32_end;
    uint32_t t2_31_0_end;
    uint32_t t2_63_32_end;
    uint32_t t3_31_0_end;
    uint32_t t3_63_32_end;
    uint32_t t4_31_0_end;
    uint32_t t4_63_32_end;
    uint32_t t5_31_0_end;
    uint32_t t5_63_32_end;
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;
    uint64_t t3;
    uint64_t t4;
    uint64_t t5;
} PmuSpmDte;

typedef struct PmuNccUnit {
    uint32_t workeridx;
    uint32_t window_31_0;
    uint32_t window_63_32;
    uint32_t window_31_0_end;
    uint32_t window_63_32_end;
    uint32_t inst_nums;
    uint32_t inst_nums_end;
    uint32_t blocking_time;
    uint32_t blocking_time_end;
    uint32_t exe_time_31_0;
    uint32_t exe_time_63_32;
    uint32_t exe_time_31_0_end;
    uint32_t exe_time_63_32_end;
    uint32_t last_cmd_info_31_0;  // 0x264
    uint32_t last_cmd_info_37_32; // 0x268
} PmuNccUnit;

typedef struct PmuNccRecord {
    uint32_t num;
    uint32_t func_id; // instr op + id
    uint32_t time_31_0[PMU_NCC_MAX_RECORD];
    uint32_t time_63_32[PMU_NCC_MAX_RECORD];
} PmuNccRecord;

typedef struct PmuNccParallel {
    uint32_t workeridx;   //worker0   1       2
    uint32_t reserved;
    uint64_t ct_last_stamp; // ct上次时间戳
    uint64_t ne_last_stamp;
    uint64_t rd_last_stamp;
    uint64_t wd_last_stamp;
    uint64_t td_last_stamp;
    uint64_t fu_last_stamp;
    uint64_t scalar_last_stamp;
    uint64_t ct_exe_time; // ct上次执行时间
    uint64_t ne_exe_time;
    uint64_t rd_exe_time;
    uint64_t wd_exe_time;
    uint64_t td_exe_time;
    uint64_t scalar_exe_time;
    uint64_t fu_exe_time;
    uint64_t rd_wd_pa_total; // rd wd并行累计值
    uint64_t ct_td_pa_total; // ct td并行累计值
} PmuNccParallel;

typedef struct PmuNcc {
    PmuTLVHead head;
    uint32_t func_id;
    uint32_t workeridx;         // worker0   1       2
    uint32_t window_31_0;       // 0x8
    uint32_t window_63_32;      // 0xc
    uint32_t window_31_0_end;   // 0x8
    uint32_t window_63_32_end;  // 0xc
    uint32_t fu_exe_time_31_0;  // 0x13c
    uint32_t fu_exe_time_63_32; // 0x140
    PmuNccRecord record;
    PmuNccUnit ct_info;
    PmuNccUnit ne_info;
    PmuNccUnit rdma_info;
    PmuNccUnit wdma_info;
    PmuNccUnit tdma_info;
    PmuNccUnit scalar_info;
    PmuNccParallel pa_info;
} PmuNcc;

typedef struct PmuNccUserTime {
    PmuTLVHead head;
    uint32_t func_id;
    uint32_t workeridx; // worker0   1       2
    uint32_t time_start;
    uint32_t time_end;
    uint32_t time;
} PmuNccUserTime;

typedef struct PmuDteChannel {
    uint32_t work_status;
    uint32_t cmd_fail_num;
    uint32_t cmd_fail_num_end;
    uint32_t cmd_success_num;
    uint32_t cmd_success_num_end;
    uint32_t trans_data_num_l;
    uint32_t trans_data_num_l_end;
    uint32_t trans_data_num_h;
    uint32_t trans_data_num_h_end;
    uint32_t idle_count_l;
    uint32_t idle_count_l_end;
    uint32_t idle_count_h;
    uint32_t idle_count_h_end;
    uint32_t trans_count_l;
    uint32_t trans_count_l_end;
    uint32_t trans_count_h;
    uint32_t trans_count_h_end;
} PmuDteChannel;

typedef struct PmuDte {
    PmuTLVHead head;
    PmuDteChannel channel_infos[2];
} PmuDte;

//------------------------------------------------------------------------------------------------------------
static volatile uint64_t g_pmu_buf_addr = 0;     // profiling基地址
static volatile uint64_t g_pmu_buf_wptr = 0;     // profiling写指针
static volatile uint32_t g_pmu_buf_size = 0;     // profiling数据大小
static volatile uint16_t g_pmu_prof_type = 0;    // profiling类型

static inline bool pmu_ncc_en() {
    __LOG__(KCORE_LOG_DEBUG, "enable pmu\n");
    bool en = *((volatile uint64_t *)(SCT_NCC_PMU_EN | SCT_REG_BASE_NCC));
    *((volatile uint32_t *)(SCT_NCC_PMU_EN | SCT_REG_BASE_NCC)) = 0x1;
    return en;
}

static inline void pmu_ncc_disable() {
    __LOG__(KCORE_LOG_DEBUG, "disable pmu\n");
    *((volatile uint32_t *)(SCT_NCC_PMU_EN | SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_clr() {
    __LOG__(KCORE_LOG_DEBUG, "clr pmu\n");
    *((volatile uint32_t *)(SCT_NCC_PMU_CLR | SCT_REG_BASE_NCC)) = 0x1;
    *((volatile uint32_t *)(SCT_NCC_PMU_CLR | SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_ct_start(PmuNccUnit *record) {
    record->inst_nums = PMU_NCC_GET_U32(SCT_NCC_WORKER0_CT_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_CT_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_CT_EXE_TIME_31_0);
    record->exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_CT_EXE_TIME_63_32);
}

// all 首次500+ns, 其余200+ns，平均20+ns一个寄存器. 使用SPM地址本函數只需要50ns
static inline void pmu_ncc_ne_start(PmuNccUnit *record) { 
    record->inst_nums = PMU_NCC_GET_U32(SCT_NCC_WORKER0_NE_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_NE_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_NE_EXE_TIME_31_0);
    record->exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_NE_EXE_TIME_63_32);
}

static inline void pmu_ncc_rdma_start(PmuNccUnit *record) {
    record->inst_nums = PMU_NCC_GET_U32(SCT_NCC_WORKER0_RDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_RDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_31_0);
    record->exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_63_32);
}

static inline void pmu_ncc_wdma_start(PmuNccUnit *record) {
    record->inst_nums = PMU_NCC_GET_U32(SCT_NCC_WORKER0_WDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_WDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_31_0);
    record->exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_63_32);
}

static inline void pmu_ncc_tdma_start(PmuNccUnit *record) {
    record->inst_nums = PMU_NCC_GET_U32(SCT_NCC_WORKER0_TDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_TDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_TDMA_EXE_TIME_31_0);
    record->exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_TDMA_EXE_TIME_63_32);
}

static inline void pmu_ncc_ct_end(PmuNccUnit *record) {
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->inst_nums_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_CT_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_CT_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->exe_time_31_0_end = PMU_NCC_GET_U32(SCT_NCC_CT_EXE_TIME_31_0);
    record->exe_time_63_32_end = PMU_NCC_GET_U32(SCT_NCC_CT_EXE_TIME_63_32);
    record->last_cmd_info_31_0 = PMU_NCC_GET_U32(SCT_NCC_CT_LAST_CMD_INFO_31_0);
    record->last_cmd_info_37_32 = PMU_NCC_GET_U32(SCT_NCC_CT_LAST_CMD_INFO_37_32);
}

static inline void pmu_ncc_ne_end(PmuNccUnit *record) {
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->inst_nums_end = PMU_NCC_GET_U32(SCT_NCC_WORKER0_NE_INST_NUMS);
    record->blocking_time_end = PMU_NCC_GET_U32(SCT_NCC_WORKER0_NE_BLOCKING_TIME);
    record->exe_time_31_0_end = PMU_NCC_GET_U32(SCT_NCC_NE_EXE_TIME_31_0);
    record->exe_time_63_32_end = PMU_NCC_GET_U32(SCT_NCC_NE_EXE_TIME_63_32);
    record->last_cmd_info_31_0 = PMU_NCC_GET_U32(SCT_NCC_NE_LAST_CMD_INFO_31_0);
    record->last_cmd_info_37_32 = PMU_NCC_GET_U32(SCT_NCC_NE_LAST_CMD_INFO_37_32);
}

static inline void pmu_ncc_rdma_end(PmuNccUnit *record) {
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->inst_nums_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_RDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_RDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->exe_time_31_0_end = PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_31_0);
    record->exe_time_63_32_end = PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_63_32);
    record->last_cmd_info_31_0 = PMU_NCC_GET_U32(SCT_NCC_RDMA_LAST_CMD_INFO_31_0);
    record->last_cmd_info_37_32 = PMU_NCC_GET_U32(SCT_NCC_RDMA_LAST_CMD_INFO_45_32);
}

static inline void pmu_ncc_wdma_end(PmuNccUnit *record) {
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->inst_nums_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_WDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_WDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->exe_time_31_0_end = PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_31_0);
    record->exe_time_63_32_end = PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_63_32);
    record->last_cmd_info_31_0 = PMU_NCC_GET_U32(SCT_NCC_WDMA_LAST_CMD_INFO_31_0);
    record->last_cmd_info_37_32 = PMU_NCC_GET_U32(SCT_NCC_WDMA_LAST_CMD_INFO_45_32);
}

static inline void pmu_ncc_tdma_end(PmuNccUnit *record) {
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->inst_nums_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_TDMA_INST_NUMS); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->blocking_time_end =
        PMU_NCC_GET_U32(SCT_NCC_WORKER0_TDMA_BLOCKING_TIME); // + record->workeridx * PMU_NCC_WORKER_OFFSET
    record->exe_time_31_0_end = PMU_NCC_GET_U32(SCT_NCC_TDMA_EXE_TIME_31_0);
    record->exe_time_63_32_end = PMU_NCC_GET_U32(SCT_NCC_TDMA_EXE_TIME_63_32);
    record->last_cmd_info_31_0 = PMU_NCC_GET_U32(SCT_NCC_TDMA_LAST_CMD_INFO_31_0);
    record->last_cmd_info_37_32 = PMU_NCC_GET_U32(SCT_NCC_TDMA_LAST_CMD_INFO_37_32);
}

static inline void pmu_ncc_start(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "start pmu ncc\n");
    PmuNcc *record = (PmuNcc *)prof_addr;
    record->head.pmu_type = PMU_TYPE_NCC;
    record->head.length = sizeof(PmuNcc) - sizeof(PmuTLVHead);
    record->func_id = 0;
    record->window_31_0 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32 = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
    record->record.num = 0;
    pmu_ncc_ct_start(&record->ct_info);
    pmu_ncc_ne_start(&record->ne_info);
    pmu_ncc_rdma_start(&record->rdma_info);
    pmu_ncc_wdma_start(&record->wdma_info);
    pmu_ncc_tdma_start(&record->tdma_info);
}

static inline void pmu_ncc_end(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "end pmu ncc\n");
    PmuNcc *record = (PmuNcc *)prof_addr;
    pmu_ncc_ct_end(&record->ct_info);
    pmu_ncc_ne_end(&record->ne_info);
    pmu_ncc_rdma_end(&record->rdma_info);
    pmu_ncc_wdma_end(&record->wdma_info);
    pmu_ncc_tdma_end(&record->tdma_info);
    record->fu_exe_time_31_0 = PMU_NCC_GET_U32(SCT_NCC_FU_EXE_TIME_31_0);
    record->fu_exe_time_63_32 = PMU_NCC_GET_U32(SCT_NCC_FU_EXE_TIME_63_32);
    record->window_31_0_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
    record->window_63_32_end = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
}

/*
 * 函数功能：记录上一次RDMA/WDMA的运行时间，并返回二者的最小值，视为并行的时间
 * 函数使用:
 * 1 第一次调用前必须清零Pmu Ncc，必须初始化输入结构体空间。
 * 2 必须在RDMA WDMA单元停止运行时使用。
 * 入参：ncc pmu地址
 * 返回值: wd_exe_time与rd_exe_time的最小值
 */
static inline uint64_t pmu_rd_wd_parallel_record(PmuNcc *record) {
    uint64_t rdexe = (uint64_t)PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_63_32) << 32
            | (uint64_t)PMU_NCC_GET_U32(SCT_NCC_RDMA_EXE_TIME_31_0);
    uint64_t wdexe = (uint64_t)PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_63_32) << 32
            | (uint64_t)PMU_NCC_GET_U32(SCT_NCC_WDMA_EXE_TIME_31_0);
    uint64_t fuexe = (uint64_t)PMU_NCC_GET_U32(SCT_NCC_FU_EXE_TIME_63_32) << 32
            | (uint64_t)PMU_NCC_GET_U32(SCT_NCC_FU_EXE_TIME_31_0);
    record->pa_info.rd_exe_time = rdexe - record->pa_info.rd_last_stamp;
    record->pa_info.wd_exe_time = wdexe - record->pa_info.wd_last_stamp;
    record->pa_info.fu_exe_time = fuexe - record->pa_info.fu_last_stamp;
    record->pa_info.rd_last_stamp = rdexe;
    record->pa_info.wd_last_stamp = wdexe;
    record->pa_info.fu_last_stamp = fuexe;
    uint64_t pa_time = record->pa_info.wd_exe_time + record->pa_info.rd_exe_time - record->pa_info.fu_exe_time;
    __LOG__(KCORE_LOG_DEBUG, "exe_time(rd %lld wd %lld fu %lld) rdexe %lld wdexe %lld fdexe %lld pa_time %lld rd_wd_pa_total %lld.\n",
        record->pa_info.rd_exe_time, record->pa_info.wd_exe_time, record->pa_info.fu_exe_time, rdexe, wdexe, fuexe, pa_time, record->pa_info.rd_wd_pa_total);
    record->pa_info.rd_wd_pa_total += pa_time;
    return pa_time;
}

static inline void pmu_ncc_record(PmuNccRecord *record) {
    if (record == NULL) {
        __LOG__(KCORE_LOG_DEBUG, "Pmu inparam record is null.\n");
        return;
    }
    if (record->num < PMU_NCC_MAX_RECORD) {
        record->time_31_0[record->num] = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_31_0);
        record->time_63_32[record->num] = PMU_NCC_GET_U32(SCT_NCC_STATISTICS_WINDOW_63_32);
        // record->func_id = flag;
        record->num++;
    } else {
        __LOG__(KCORE_LOG_DEBUG, "statisc window time record(%d) full.\n", record->num);
    }
}

//------------------------------------------------------------------------------------------------------------
static inline void pmu_ncc_w0_user_time0_en() {
    *((volatile uint32_t *)(SCT_NCC_WORKER0_USER_TIMER0_EN | SCT_REG_BASE_NCC)) = 0x1;
}

static inline void pmu_ncc_w0_user_time0_disable() {
    *((volatile uint32_t *)(SCT_NCC_WORKER0_USER_TIMER0_EN | SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_w0_user_time0_clr() {
    *((volatile uint32_t *)(SCT_NCC_WORKER0_USER_TIMER0_CLR + SCT_REG_BASE_NCC)) = 0x1;
    *((volatile uint32_t *)(SCT_NCC_WORKER0_USER_TIMER0_CLR + SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_w1_user_time0_en() {
    *((volatile uint32_t *)(SCT_NCC_WORKER1_USER_TIMER0_EN + SCT_REG_BASE_NCC)) = 0x1;
}

static inline void pmu_ncc_w1_user_time0_disable() {
    *((volatile uint32_t *)(SCT_NCC_WORKER1_USER_TIMER0_EN + SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_w1_user_time0_clr() {
    *((volatile uint32_t *)(SCT_NCC_WORKER1_USER_TIMER0_CLR + SCT_REG_BASE_NCC)) = 0x1;
    *((volatile uint32_t *)(SCT_NCC_WORKER1_USER_TIMER0_CLR + SCT_REG_BASE_NCC)) = 0x0;
}

static inline void pmu_ncc_w0_user_time0_start(PmuNccUserTime *record) {
    record->time_start = PMU_NCC_GET_U32(SCT_NCC_WORKER0_USER_TIMER0);
    return;
}

static inline void pmu_ncc_w0_user_time0_end(PmuNccUserTime *record) {
    record->time_end = PMU_NCC_GET_U32(SCT_NCC_WORKER0_USER_TIMER0);
    record->time = record->time_end - record->time_start;
    record->head.pmu_type = PMU_TYPE_NCC_USER_TIME;
    record->head.length = sizeof(PmuNccUserTime) - sizeof(PmuTLVHead);
    record->func_id = 0;
    record->workeridx = 0 << 16 | 0; // workidx + timeidx
    return;
}

static inline uint32_t pmu_ncc_w1_user_time0() { return PMU_NCC_GET_U32(SCT_NCC_WORKER1_USER_TIMER0); }

static inline void pmu_ncc_w1_user_time0_start(PmuNccUserTime *record) {
    record->time_start = PMU_NCC_GET_U32(SCT_NCC_WORKER1_USER_TIMER0);
    return;
}

static inline void pmu_ncc_w1_user_time0_end(PmuNccUserTime *record) {
    record->time_end = PMU_NCC_GET_U32(SCT_NCC_WORKER1_USER_TIMER0);
    record->time = record->time_end - record->time_start;
    record->head.pmu_type = PMU_TYPE_NCC_USER_TIME;
    record->head.length = sizeof(PmuNccUserTime) - sizeof(PmuTLVHead);
    record->func_id = 0;
    record->workeridx = 1 << 16 | 0; // workidx + timeidx
    return;
}
//------------------------------------------------------------------------------------------------------------
static inline void pmu_spm_en_all() { *((volatile uint32_t *)(SPM1_PMU_EN + SCT_REG_BASE_SPM)) = 0b11111; }

static inline void pmu_spm_disable_all() { *((volatile uint32_t *)(SPM1_PMU_EN + SCT_REG_BASE_SPM)) = 0x0; }

static inline void pmu_spm_clr() {
    *((volatile uint32_t *)(SPM1_PMU_CLR + SCT_REG_BASE_SPM)) = 0x1;
    *((volatile uint32_t *)(SPM1_PMU_CLR + SCT_REG_BASE_SPM)) = 0x0;
}

/**
 * @brief 统计pmu_spm寄存器中lsu的起始时刻的性能数据
 * @param prof_addr lsu性能数据保存地址
 * return void
*/
static inline void pmu_spm_lsu_all_start(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "start pmu spm lsu\n");
    PmuSpmLsu *record = (PmuSpmLsu *)prof_addr;
    record->head.pmu_type = PMU_TYPE_SPM_LSU;
    record->head.length = sizeof(PmuSpmLsu) - sizeof(PmuTLVHead);
    record->func_id = 0;
    record->t2_rdma_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T2_0_31_0);   // port 0 : RDMA
    record->t2_rdma_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T2_0_63_32);
    record->t2_wdma_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T2_6_31_0);   // port 6 : WDMA
    record->t2_wdma_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T2_6_63_32);
    record->t3_rdma_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T3_0_31_0); // port 0 : RDMA
    record->t3_rdma_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T3_0_63_32);
    record->t3_wdma_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T3_6_31_0); // port 6 : WDMA
    record->t3_wdma_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T3_6_63_32);
}

/**
 * @brief 统计pmu_spm寄存器中lsu的结束时刻的性能数据
 * @param prof_addr lsu性能数据保存地址
 * return void
*/
static inline void pmu_spm_lsu_all_end(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "end pmu spm lsu\n");
    PmuSpmLsu *record = (PmuSpmLsu *)prof_addr;
    record->t2_rdma_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T2_0_31_0);   // port 0 : RDMA
    record->t2_rdma_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T2_0_63_32);
    record->t2_wdma_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T2_6_31_0);   // port 6 : WDMA
    record->t2_wdma_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T2_6_63_32);
    record->t3_rdma_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T3_0_31_0); // port 0 : RDMA
    record->t3_rdma_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T3_0_63_32);
    record->t3_wdma_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T3_6_31_0); // port 6 : WDMA
    record->t3_wdma_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T3_6_63_32);
}

static inline void pmu_spm_dte_all_start(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "start pmu spm dte\n");
    PmuSpmDte *record = (PmuSpmDte *)prof_addr;
    record->head.pmu_type = PMU_TYPE_SPM_DTE;
    record->head.length = sizeof(PmuSpmDte) - sizeof(PmuTLVHead);
    record->func_id = 0;
    record->t0_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T0_31_0); // globle
    record->t0_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T0_63_32);
    record->t1_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T1_0_31_0);   // xbar 1 RP1
    record->t1_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T1_0_63_32); // xbar 1 RP1
    record->t2_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T2_8_31_0);   // port 8 : R_DTE
    record->t2_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T2_8_63_32);
    record->t3_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T3_8_31_0); // port 8 : R_DTE
    record->t3_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T3_8_63_32);
    record->t4_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T4_31_0); // globle
    record->t4_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T4_63_32);
    record->t5_31_0 = PMU_SPM_GET_U32(SPM1_PMU_T5_1_31_0); // xbar 1 RP1
    record->t5_63_32 = PMU_SPM_GET_U32(SPM1_PMU_T5_1_63_32);
}

static inline void pmu_spm_dte_all_end(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "end pmu spm dte\n");
    PmuSpmDte *record = (PmuSpmDte *)prof_addr;
    record->t0_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T0_31_0); // globle
    record->t0_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T0_63_32);
    record->t1_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T1_0_31_0);   // xbar 1 RP1
    record->t1_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T1_0_63_32); // xbar 1 RP1
    record->t2_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T2_8_31_0);   // port 8 : R_DTE
    record->t2_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T2_8_63_32);
    record->t3_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T3_8_31_0); // port 8 : R_DTE
    record->t3_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T3_8_63_32);
    record->t4_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T4_31_0); // globle
    record->t4_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T4_63_32);
    record->t5_31_0_end = PMU_SPM_GET_U32(SPM1_PMU_T5_1_31_0); // xbar 1 RP1
    record->t5_63_32_end = PMU_SPM_GET_U32(SPM1_PMU_T5_1_63_32);
}

//------------------------------------------------------------------------------------------------------------
static inline void pmu_dte_en() { *((volatile uint32_t *)(SCT_DTE_PMU_EN | SCT_REG_BASE_DTE)) = 0x3; }

static inline void pmu_dte_disable() { *((volatile uint32_t *)(SCT_DTE_PMU_EN | SCT_REG_BASE_DTE)) = 0x0; }

static inline void pmu_dte_clr() {
    *((volatile uint32_t *)(SCT_DTE_PMU_CLR | SCT_REG_BASE_DTE)) = 0x3;
    *((volatile uint32_t *)(SCT_DTE_PMU_CLR | SCT_REG_BASE_DTE)) = 0x0;
}

// kcore默认使用的channel_1的block 0和block 1
static inline void pmu_dte_start_chn1(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "start pmu dte\n");
    PmuDte *record = (PmuDte *)prof_addr;
    record->head.pmu_type = PMU_TYPE_DTE;
    record->head.length = sizeof(PmuDte) - sizeof(PmuTLVHead);
    record->channel_infos[1].work_status      = PMU_DTE_GET_U32(SCT_DTE_DTE_WORK_STATUS);
    record->channel_infos[1].cmd_success_num  = PMU_DTE_GET_U32(SCT_DTE_CH1_CMD_SUCCESS_COUNTER);
    record->channel_infos[1].cmd_fail_num     = PMU_DTE_GET_U32(SCT_DTE_CH_CMD_FAIL_COUNTER);
    record->channel_infos[1].trans_data_num_l = PMU_DTE_GET_U32(SCT_DTE_CH1_TRANS_DATA_L);
    record->channel_infos[1].trans_data_num_h = PMU_DTE_GET_U32(SCT_DTE_CH1_TRANS_DATA_H);
    record->channel_infos[1].idle_count_l     = PMU_DTE_GET_U32(SCT_DTE_CH1_IDLE_CLK_COUNTER_L);
    record->channel_infos[1].idle_count_h = PMU_DTE_GET_U32(SCT_DTE_CH1_IDLE_CLK_COUNTER_H);
    record->channel_infos[1].trans_count_l    = PMU_DTE_GET_U32(SCT_DTE_CH1_CLK_COUNTER_L);
    record->channel_infos[1].trans_count_h    = PMU_DTE_GET_U32(SCT_DTE_CH1_CLK_COUNTER_H);
}

static inline void pmu_dte_end_chn1(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "end pmu dte\n");
    PmuDte *record = (PmuDte *)prof_addr;
    record->channel_infos[1].work_status       = PMU_DTE_GET_U32(SCT_DTE_DTE_WORK_STATUS);
    record->channel_infos[1].trans_data_num_l_end = PMU_DTE_GET_U32(SCT_DTE_CH1_TRANS_DATA_L);
    record->channel_infos[1].trans_data_num_h_end = PMU_DTE_GET_U32(SCT_DTE_CH1_TRANS_DATA_H);
    record->channel_infos[1].idle_count_l_end     = PMU_DTE_GET_U32(SCT_DTE_CH1_IDLE_CLK_COUNTER_L);
    record->channel_infos[1].idle_count_h_end     = PMU_DTE_GET_U32(SCT_DTE_CH1_IDLE_CLK_COUNTER_H);
    record->channel_infos[1].trans_count_l_end    = PMU_DTE_GET_U32(SCT_DTE_CH1_CLK_COUNTER_L);
    record->channel_infos[1].trans_count_h_end    = PMU_DTE_GET_U32(SCT_DTE_CH1_CLK_COUNTER_H);
    record->channel_infos[1].cmd_success_num_end  = PMU_DTE_GET_U32(SCT_DTE_CH1_CMD_SUCCESS_COUNTER);
    record->channel_infos[1].cmd_fail_num_end     = PMU_DTE_GET_U32(SCT_DTE_CH_CMD_FAIL_COUNTER);
}

static inline void pmu_dte_start_chn0(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "start pmu dte\n");
    PmuDte *record = (PmuDte *)prof_addr;
    record->head.pmu_type = PMU_TYPE_DTE;
    record->head.length = sizeof(PmuDte) - sizeof(PmuTLVHead);
    record->channel_infos[0].work_status      = PMU_DTE_GET_U32(SCT_DTE_DTE_WORK_STATUS);
    record->channel_infos[0].cmd_success_num  = PMU_DTE_GET_U32(SCT_DTE_CH0_CMD_SUCCESS_COUNTER);
    record->channel_infos[0].cmd_fail_num     = (PMU_DTE_GET_U32(SCT_DTE_CH_CMD_FAIL_COUNTER)) & 0x0000FF00;
    record->channel_infos[0].trans_data_num_l = PMU_DTE_GET_U32(SCT_DTE_CH0_TRANS_DATA_L);
    record->channel_infos[0].trans_data_num_h = PMU_DTE_GET_U32(SCT_DTE_CH0_TRANS_DATA_H);
    record->channel_infos[0].idle_count_l     = PMU_DTE_GET_U32(SCT_DTE_CH0_IDLE_CLK_COUNTER_L);
    record->channel_infos[0].idle_count_h     = PMU_DTE_GET_U32(SCT_DTE_CH0_IDLE_CLK_COUNTER_H);
    record->channel_infos[0].trans_count_l    = PMU_DTE_GET_U32(SCT_DTE_CH0_CLK_COUNTER_L);
    record->channel_infos[0].trans_count_h    = PMU_DTE_GET_U32(SCT_DTE_CH0_CLK_COUNTER_H);
}

static inline void pmu_dte_end_chn0(uint64_t prof_addr) {
    __LOG__(KCORE_LOG_DEBUG, "end pmu dte\n");
    PmuDte *record = (PmuDte *)prof_addr;
    record->channel_infos[0].work_status       = PMU_DTE_GET_U32(SCT_DTE_DTE_WORK_STATUS);
    record->channel_infos[0].trans_data_num_l_end = PMU_DTE_GET_U32(SCT_DTE_CH0_TRANS_DATA_L);
    record->channel_infos[0].trans_data_num_h_end = PMU_DTE_GET_U32(SCT_DTE_CH0_TRANS_DATA_H);
    record->channel_infos[0].idle_count_l_end     = PMU_DTE_GET_U32(SCT_DTE_CH0_IDLE_CLK_COUNTER_L);
    record->channel_infos[0].idle_count_h_end     = PMU_DTE_GET_U32(SCT_DTE_CH0_IDLE_CLK_COUNTER_H);
    record->channel_infos[0].trans_count_l_end    = PMU_DTE_GET_U32(SCT_DTE_CH0_CLK_COUNTER_L);
    record->channel_infos[0].trans_count_h_end    = PMU_DTE_GET_U32(SCT_DTE_CH0_CLK_COUNTER_H);
    record->channel_infos[0].cmd_success_num_end  = PMU_DTE_GET_U32(SCT_DTE_CH0_CMD_SUCCESS_COUNTER);
    record->channel_infos[0].cmd_fail_num_end     = (PMU_DTE_GET_U32(SCT_DTE_CH_CMD_FAIL_COUNTER)) & 0x0000000F;
}

// 对外接口（main.c / stream_riscv.inc）
#include <rtthread.h>
typedef struct ProfAddrNode {
    uint64_t prof_addr;
    rt_list_t list;
} ProfAddrNode;

void profiling_start(uint64_t addr, uint32_t size, uint16_t type);
void profiling_stop(uint64_t addr, uint32_t size);
uint64_t pmu_get_exetime(uint32_t type);
uint32_t pmu_wait_finish(uint32_t type, uint64_t start_exe_time);
uint32_t pmu_check_engine_status(uint32_t type, uint64_t start_exe_time, uint8_t check);
void PMU_START();
void PMU_END();

void pmu_ncc_print(uint64_t prof_addr);
/**
 * @brief 打印spm寄存器中lsu的性能数据
 * @param prof_addr 性能数据输出的地址
*/
void pmu_spm_lsu_print(uint64_t prof_addr);
void pmu_spm_dte_print(uint64_t prof_addr);
void pmu_dte_print(uint64_t prof_addr);
void pmu_dte_print_chn0(uint64_t prof_addr);
uint32_t get_pmu_dte_status(uint64_t prof_addr);
uint32_t get_pmu_dte_trans_data(uint64_t prof_addr);
uint32_t get_pmu_dte_trans_time(uint64_t prof_addr);
#endif
