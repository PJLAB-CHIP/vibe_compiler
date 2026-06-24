#ifndef __SPM_OFFSET_MAP_H__
#define __SPM_OFFSET_MAP_H__

#define PG_MAG_TABLE_ADDR           0X17F000000

/* spm地址规划：同步默认使用的是3M SPM中的最后64k中的前32k，具体分配如下:
 * 0x000~0x040 reserved for barrier, full good tile10 , partial good tile0
 * 0x200~0x210 direct dte, max(Fsmid) * uint32_t
 * 0x280~0x290 direct dte, dte counter for sender and receiver.
 * 0x300~0x320 spm_sync_addr, 8 * uint32_t
 * 0x400~0x404 uint32_t, barrier count
 * @ Spm-Scheme Version 2024.11.4 by liujiangtao
 *
*/
#define KCORE_SPM_ADDR_BASE           0X2F0000
#define INNER_CHIP_BARRIER_OFFSET     0x0    // barrier
#define BOOT_PARAM_OFFSET             0x40   // boot param
#define DIRECT_DTE_SYNC_SPM_OFFSET    0X200  // dte sync
#define DIRECT_DTE_COUNTER_OFFSET     0X280  // dte counter
#define DUAL_SPM_SYNC_OFFSET          0X300  // 双环all-reduce同步
#define SINGLE_SPM_SYNC_OFFSET        0X320  // 单环all-reduce同步, 0x320-0x370预留给all-reduce同步使用！
#define INNER_CHIP_ERROR_CODE         0X3A0  // 记录kcore错误码信息，4字节. 1:assert,2:WDMA问题 3:RDMA问题 4:get_ddrmapping问题
#define INNER_CHIP_G_BAR_COUNTER_OFFSET     0X400  // barrier count
#define INNER_CHIP_BOOT_HEAD_OFFSET   0X404  // 启动参数地址, 8字节
#define INNER_CHIP_BOOTPARAM_OFFSET   0X40C  // 收到启动参数的次数, 4字节
#define COMPUTE_START_OFFSET          0X410  // 开始计算(main_subgraph0)的次数,4字节
#define COMPUTE_END_OFFSET            0X414  // 结束计算(main_subgraph0)的次数,4字节
#define MODEL_ID_OFFSET               0X418  // 模型在链表中记录的ID,4字节
#define KCORE_STATE_OFFSET            0X41c  // 0:init; 1:exit while1; 2:kcore all done.
#define INSTR_INVALID_RECORD_OFFSET   0X420  // 记录指令地址异常信息,size=40字节
#define LOGIC_ID_OFFSET               0X450  // 记录TILE的LOGIC_ID,size=4字节
#define TILE_ID_OFFSET                0X454  // 记录TILE的TILE ID, size=4字节
#define ROW_LENGTH_OFFSET             0X458  // 记录row length, size=4字节
#define MHU_POWER_MAIN_STATE_OFFSET   0X45C  // 记录下电流程主task的状态，size=1字节，每个task 1字节
#define MHU_POWER_CORE0_STATE_OFFSET  0X45D  // 记录下电流程核0 task的状态，size=1字节，for kernel core0 power off task
#define MHU_POWER_CORE1_STATE_OFFSET  0X45E  // 记录下电流程核1 task的状态，size=1字节，for kernel core1 power off task
#define POWER_OFF_STRATEGY_MAGIC_NUM_OFFSET    0X460  // 记录 power off strategy magic number，size=4字节
#define MULTI_GRAPH_MEM_SPACE         0X464  // 记录多图的内存起始地址，size=8字节
#define KCORE_ERR_INFO_OFFSET         0x48C  // 记录kcore报错信息，32个字节，存放函数名+行号+barrier
#define MULTI_GRAPH_DEBUG_OFFSET      0X4AC  // 记录多图debug信息,size=(8+132*50)=6608字节,下一个可用地址：0x1E7C
#define STREAM_POP_COUNT              0X1F44  // 记录stream_pop的次数, 4字节
#define STREAM_WAIT_COUNT             0X1F48  // 记录stream_wait的次数，4字节
#define NONZERO_DEBUG_OFFSET          0X1F4C  // 记录NONZERO 信息, 4字节
#define SINGLE_SPM_SYNC_DEBUG_OFFSET  0X1F50  // 记录单环同步信息,176字节=44个int, (0x1F50 - 0x1FFF)
#define ALL2ALL_SPM_SYNC_OFFSET       0X2000  // all2all 同步地址 4KB （0x2000 - 0x2FFF)


#define TRITON_PID_OFFSET             0x23c0 // Triton pid,16字节
#define TRITON_BLOCK_DIM_OFFSET       0x23d0 // Triton Block dim 16字节

#define PRODUCT_TYPE_PG               0x23e0 // 产品类型：FG:0;PG:1 4字节

#define HRT_BARRIER_OFFSET            0X3000  //  hrt的barrier，4*16字节
#define RING_BUFFER_OFFSET            0X3040 // Triton ML ringbuffer,size=1024字节
#define PROC_MSG_PERF_OFFSET          0x3440  // 记录kcore处理ringbuffer的性能，10*8字节
#define SCORE_PROF_EXIT_OFFSET        0x3490  // score profiling task退出标记位，4字节
#define PERF_TIME_SPACE               0X4000  // 记录性能地址，size=8*4=32字节

/*============================= GDB专用：大小:1K,32k的最后1K =====================================*/
#define GDB_MSG_STATUS               0x7C00  // gdb的rsp消息类型：0xab：请求；0xcd：应答
#define GDB_REQ_OFFSET               0x7C04  // gdb请求消息，大小256字节 (7c04-7cff)
#define GDB_RSP_OFFSET               0x7D00  // gdb回应消息，大小768字节 (7d00-7fff)

#define INNER_CHIP_BARRIER_ADDR      (KCORE_SPM_ADDR_BASE+INNER_CHIP_BARRIER_OFFSET)
#define BOOT_PARAM_OFFSET_ADDR       (KCORE_SPM_ADDR_BASE+BOOT_PARAM_OFFSET)
#define DIRECT_DTE_SYNC_SPM_ADDR     (KCORE_SPM_ADDR_BASE+DIRECT_DTE_SYNC_SPM_OFFSET)
#define DIRECT_DTE_COUNTER_ADDR      (KCORE_SPM_ADDR_BASE+DIRECT_DTE_COUNTER_OFFSET)
#define DUAL_SPM_SYNC_ADDR           (KCORE_SPM_ADDR_BASE+DUAL_SPM_SYNC_OFFSET)
#define SINGLE_SPM_SYNC_ADDR         (KCORE_SPM_ADDR_BASE+SINGLE_SPM_SYNC_OFFSET)
#define INNER_CHIP_BOOT_HEAD_ADDR    (KCORE_SPM_ADDR_BASE+INNER_CHIP_BOOT_HEAD_OFFSET)
#define INNER_CHIP_BOOTPARAM_COUNT   (KCORE_SPM_ADDR_BASE+INNER_CHIP_BOOTPARAM_OFFSET)
#define COMPUTE_START_COUNT          (KCORE_SPM_ADDR_BASE+COMPUTE_START_OFFSET)
#define COMPUTE_END_COUNT            (KCORE_SPM_ADDR_BASE+COMPUTE_END_OFFSET)
#define MODEL_ID_ADDR                (KCORE_SPM_ADDR_BASE + MODEL_ID_OFFSET)
#define INNER_CHIP_ERROR_CODE_ADDR   (KCORE_SPM_ADDR_BASE+INNER_CHIP_ERROR_CODE)
#define INNER_CHIP_G_BAR_COUNTER     (KCORE_SPM_ADDR_BASE+INNER_CHIP_G_BAR_COUNTER_OFFSET)
#define KCORE_STATE_ADDR             (KCORE_SPM_ADDR_BASE+KCORE_STATE_OFFSET)
#define INSTR_INVALID_RECORD_ADDR    (KCORE_SPM_ADDR_BASE+INSTR_INVALID_RECORD_OFFSET)
#define LOGIC_ID_ADDR                (KCORE_SPM_ADDR_BASE+LOGIC_ID_OFFSET)
#define TILE_ID_ADDR                 (KCORE_SPM_ADDR_BASE+TILE_ID_OFFSET)
#define ROW_LENGTH_ADDR              (KCORE_SPM_ADDR_BASE+ROW_LENGTH_OFFSET)
#define MHU_POWER_MAIN_STATE_ADDR    (KCORE_SPM_ADDR_BASE+MHU_POWER_MAIN_STATE_OFFSET)
#define MHU_POWER_CORE0_STATE_ADDR   (KCORE_SPM_ADDR_BASE+MHU_POWER_CORE0_STATE_OFFSET)
#define MHU_POWER_CORE1_STATE_ADDR   (KCORE_SPM_ADDR_BASE+MHU_POWER_CORE1_STATE_OFFSET)
#define POWER_OFF_STRATEGY_MAGIC_NUM_ADDR   (KCORE_SPM_ADDR_BASE+POWER_OFF_STRATEGY_MAGIC_NUM_OFFSET)
#define MULTI_GRAPH_MEM_SPACE_ADDR   (KCORE_SPM_ADDR_BASE+MULTI_GRAPH_MEM_SPACE)
#define PERF_TIME_SPACE_ADDR         (KCORE_SPM_ADDR_BASE+PERF_TIME_SPACE)
#define KCORE_ERR_INFO_ADDR          (KCORE_SPM_ADDR_BASE+KCORE_ERR_INFO_OFFSET)
#define PERF_BARRIER_TIME_ADDR       (0) // 前64K，16bit存放barrier性能数据，用于性能分析
#define MULTI_GRAPH_DEBUG_ADDR       (KCORE_SPM_ADDR_BASE+MULTI_GRAPH_DEBUG_OFFSET)
#define STREAM_POP_COUNT_ADDR        (KCORE_SPM_ADDR_BASE + STREAM_POP_COUNT)
#define STREAM_WAIT_COUNT_ADDR       (KCORE_SPM_ADDR_BASE + STREAM_WAIT_COUNT)
#define NONZERO_DEBUG_ADDR           (KCORE_SPM_ADDR_BASE+NONZERO_DEBUG_OFFSET)
#define SINGLE_SPM_SYNC_DEBUG_ADDR   (KCORE_SPM_ADDR_BASE+SINGLE_SPM_SYNC_DEBUG_OFFSET)
#define ALL2ALL_SPM_SYNC_ADDR        (KCORE_SPM_ADDR_BASE+ALL2ALL_SPM_SYNC_OFFSET)
#define HRT_BARRIER_OFFSET_ADDR      (KCORE_SPM_ADDR_BASE + HRT_BARRIER_OFFSET)

#define RING_BUFFER_ADDR             (KCORE_SPM_ADDR_BASE + RING_BUFFER_OFFSET)
#define TRITON_PID_ADDR              (KCORE_SPM_ADDR_BASE + TRITON_PID_OFFSET)
#define TRITON_BLOCK_DIM_ADDR        (KCORE_SPM_ADDR_BASE + TRITON_BLOCK_DIM_OFFSET)
#define PRODUCT_TYPE_PG_ADDR         (KCORE_SPM_ADDR_BASE + PRODUCT_TYPE_PG)
#define HRT_BARRIER_OFFSET_ADDR      (KCORE_SPM_ADDR_BASE + HRT_BARRIER_OFFSET)
#define PROC_MSG_PERF_ADDR           ((KCORE_SPM_ADDR_BASE + PROC_MSG_PERF_OFFSET))
#define SCORE_PROF_EXIT_ADDR         (KCORE_SPM_ADDR_BASE + SCORE_PROF_EXIT_OFFSET)

#define GDB_MSG_STATUS_ADDR          (KCORE_SPM_ADDR_BASE + GDB_MSG_STATUS)
#define GDB_REQ_ADDR                 (KCORE_SPM_ADDR_BASE + GDB_REQ_OFFSET)
#define GDB_RSP_ADDR                 (KCORE_SPM_ADDR_BASE + GDB_RSP_OFFSET)

int8_t *get_spm_memory_mapping(uint64_t offset);
static inline void set_spm_memory_value(uint64_t offset, uint32_t value) {
    *(volatile uint32_t *)(get_spm_memory_mapping(offset)) = value;
}
uint64_t get_tile_spm_addr_base(uint32_t tile_id_1d, int32_t tile_x, int32_t tile_y) ;

void tile_sync_by_spm_single_direction(int32_t tile_this, int32_t tile_a, int32_t tile_x, int32_t tile_y,
        uint32_t this_sync_spm_index, uint32_t other_sync_spm_index) ;

void tile_ready_read_other_tile_spm(int32_t tile_this, int32_t tile_a, int32_t tile_x, int32_t tile_y,
        uint32_t this_sync_spm_index);

void tile_ready_write_other_tile_spm(int32_t tile_this, int32_t tile_a, int32_t tile_x, int32_t tile_y,
        uint32_t other_sync_spm_index);

void tile_sync_by_spm(int32_t tile_this, int32_t tile_a, int32_t tile_b, int32_t tile_x, int32_t tile_y);

#endif