#ifndef __DIRECT_DTE_AND_FSM_H__
#define __DIRECT_DTE_AND_FSM_H__
#include <stdint.h>
#include <stddef.h>


/* DTE相关id和FSM-id定义
DTE-ID:
    DTE-RDMA:    2
    DTE-WDMA:    3
    DTE-DDR2DDR  2
FSM-ID:
    DTE-RDMA:    2
    DTE-WDMA:    3
    DTE-DDR2DDR  2
*/
#define DIRECT_DTE_INDEX_0         0
#define DIRECT_DTE_INDEX_1         1
#define DIRECT_DTE_INDEX_2         2
#define DIRECT_DTE_INDEX_3         3
#define DIRECT_DTE_FSM_ID_0        0
#define DIRECT_DTE_FSM_ID_1        1
#define DIRECT_DTE_FSM_ID_2        2
#define DIRECT_DTE_FSM_ID_3        3

typedef struct {
    uint32_t stride;
    uint32_t iteration;
} StrideIteration;
// direct-dte
typedef struct DirectDTESendInfo{
    size_t src_addr;
    size_t dst_addr;
    uint32_t length;
    uint8_t remote_fsm_id;
    uint8_t mode;
    uint16_t dst_tile;
    uint16_t tile_this;
    uint16_t rsv;
    StrideIteration stride_iterations[3];
    void *dte_node; // dte transfer node, need malloc first.
} DirectDTESendInfo;
void direct_sync_init(int once_sync_num);
void direct_sync_post(uint32_t tile_this, uint32_t tile_other);
void direct_sync_wait(uint32_t tile_this, uint32_t tile_other);

void *direct_fsm_monitor_init(int fsm_id, size_t addr, int packet_size, int packet_cnt);
void *direct_fsm_monitor_init_ddr(int fsm_id, size_t addr, int packet_size, int packet_cnt);
int direct_fsm_monitor_deinit(void *fsm_hd);
void *set_direct_fsm_monitor_dst_addr(int fsm_id, size_t addr);
void *set_direct_fsm_monitor_length(int fsm_id, uint32_t packet_size);

void direct_fsm_monitor_receive(uint16_t tile_this, uint16_t tile_other, void *fsm_hd);
/*
    dte使用的id: 1-3,
    dte-id = 0，预留给score，防止异步run的时候kcore和score使用同一个dte id导致卡死.
    dte-id = 2, 方便配置dte outstanding寄存器，为了提升性能。如果指定is_high_performance为1，只能分配dte id为2，分配不到则报错。
    DDR2DDR时，需要满足：
        channel 1: read_outstanding * axi_read_burst_length <= 12
        读写地址均为256Byte对齐，可获得更高效率
*/
void *direct_dte_attach(uint32_t is_high_performance);
int direct_dte_release(void *dte_node);
int direct_dte_send_async(DirectDTESendInfo *dte_info);
int direct_dte_wait_done(DirectDTESendInfo *dte_info);
int direct_dte_send_sync(DirectDTESendInfo *dte_info);
#endif