/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef MOD_DTE_H
#define MOD_DTE_H

#include <stdint.h>
#include <dte/dte_cfg.h>

#include <kuiper_common.h>
#define KUIPER_OFFSET_IN_REMOTE_TILE(tile_id) (((uint64_t)(0x1) << 39) + (((uint64_t)(tile_id)) << 40))
typedef enum {
	MOD_DTE_STATE_FREE,
	MOD_DTE_STATE_ALLOC,
	MOD_DTE_STATE_CONFIG,
	MOD_DTE_STATE_TRANSFER,
	MOD_DTE_STATE_DONE,
	MOD_DTE_STATE_ERR,
} mod_kuiper_dte_state_t;

typedef enum {
    KUIPER_DTE_MODE_UNICAST     = 0,
    KUIPER_DTE_MODE_SCATTER     = 1,
    KUIPER_DTE_MODE_BROADCAST   = 2,
    KUIPER_DTE_MODE_SHUFFLE     = 3,
    KUIPER_DTE_MODE_RDMA        = 4,
    KUIPER_DTE_MODE_WDMA        = 5,
    KUIPER_DTE_MODE_DDR2DDR_U2U = 6,
    KUIPER_DTE_MODE_DDR2DDR_SHUFFLE = 7
} kuiper_dte_mode_t;

typedef enum {
	KUIPER_DTE_CH_0    = 0,
	KUIPER_DTE_CH_1    = 1,
	KUIPER_DTE_CH_MAX
} kuiper_dte_channels_t;

/*keep same as stream config other end struct*/
typedef struct {
	sct_dte_block_stride_iteration_s dim_cfg[3];
} kuiper_dte_shuffle_cfg_t;


typedef struct {
	uint64_t dst_addr;
	sct_dte_block_user_id_s dst_id;
	uint16_t dst_tile;
} mode_kuiper_dte_dst_config_t;

typedef struct {
	uint32_t alloc_stream_id;
	mod_kuiper_dte_state_t state;
	kuiper_dte_mode_t mode;
	uint8_t dte_index;
	uint64_t src_addr;
	uint32_t data_len;
	uint8_t dst_cnt;
	mode_kuiper_dte_dst_config_t dst_cfg[32]; /*max 32 dest*/
	kuiper_dte_shuffle_cfg_t *src_dim;
	kuiper_dte_shuffle_cfg_t *dest_dim;
} mod_kuiper_dte_node_t;

typedef union {
	struct {
		volatile uint32_t                send: 1;
		volatile uint32_t                : 3;
		volatile uint32_t                tgt_packet_id: 5;
		volatile uint32_t                : 3;
		volatile uint32_t                tgt_stream_id: 6;
		volatile uint32_t                : 14;

	} field;

	volatile uint32_t value;
} packet_cnts_update_info_s;

/*!
 * \}
 */

int dte_init();
uint8_t mod_kuiper_dte_check_free_cnt(void);
mod_kuiper_dte_node_t *mod_kuiper_dte_alloc(uint32_t is_high_performance);
int mod_kuiper_dte_release(mod_kuiper_dte_node_t *node);
int mod_kuiper_dte_config_src_and_dst(mod_kuiper_dte_node_t *node, uint16_t tile_logic_id, uint64_t src_addr,
	uint64_t dst_addr, uint32_t data_len, kuiper_dte_shuffle_cfg_t *shuffle_cfg);
int mod_kuiper_dte_trig_send(mod_kuiper_dte_node_t *node);
int mod_kuiper_dte_check_send_status(mod_kuiper_dte_node_t *node);
mod_kuiper_dte_node_t *mod_kuiper_dte_find_finished_block(void);
void mod_kuiper_dte_auto_update_packet_cnt(mod_kuiper_dte_node_t *node);

int mod_kuiper_dte_clear_dma_status(mod_kuiper_dte_node_t *node);

/* DTE PMU funcitons */
int mod_kuiper_dte_set_pmu_en(uint8_t channel, uint8_t en);
uint8_t mod_kuiper_dte_get_pmu_en(uint8_t channel);
int mod_kuiper_dte_clear_pmu_reg(uint8_t channel);
uint8_t mod_kuiper_dte_pmu_get_work_status(uint8_t channel);
uint64_t mod_kuiper_dte_pmu_get_clk_cnts(void);
uint64_t mod_kuiper_dte_pmu_get_clk_cnts_one_channel(uint8_t channel);
uint32_t mod_kuiper_dte_pmu_get_cmd_success_cnts(uint8_t channel);
uint8_t mod_kuiper_dte_pmu_get_cmd_failed_cnts(uint8_t channel);
uint64_t mod_kuiper_dte_pmu_get_transfer_data_cnts(uint8_t channel);
uint64_t mod_kuiper_dte_pmu_get_unalign_burst_cnts(uint8_t channel);
uint64_t mod_kuiper_dte_pmu_get_idle_clk_cnts(uint8_t channel);
uint64_t mod_kuiper_dte_pmu_get_all_idle_clk_cnts(void);
/*!
 * \}
 */

#endif /* MOD_DTE_H */
