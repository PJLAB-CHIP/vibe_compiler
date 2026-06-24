/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef KUIPER_DTE_H
#define KUIPER_DTE_H

#include <stdbool.h>
#include <stddef.h>

#include <stdint.h>

#include <dte/dte_cfg.h>
#include <dte/mod_dte.h>
#include <dte/kuiper_soc_mmap.h>

#define KUIPER_DTE_BLOCK_REG_ADDR(index)    (KUIPER_DTE_BASE + (index * 0x200))
#define KUIPER_DTE_PMU_REG_ADDR             (KUIPER_DTE_BASE + 0x800)

#define KUIPER_DTE_CH_CNT                   (2)
#define KUIPER_DTE_BLOCK_PER_CH             (2)
#define KUIPER_DTE_BLOCK_TOTAL              (4)
#define KUIPER_DTE_BLOCK_TOTAL_KCORE        (4)
#define KUIPER_DTE_KCORE_IDX_START          (2)
#define KUIPER_DTE_KCORE_IDX_MAX            (3)

#define KUIPER_DTE_KCORE_RDMA_CH            (0)
#define KUIPER_DTE_KCORE_RWMA_CH            (1)

#define KUIPER_DTE_UNICAST_FLAG_SCATTER     (0)
#define KUIPER_DTE_UNICAST_FLAG_GATHER      (1)

#define KUIPER_DTE_CMD_VALID_TRIG_SEND      (1)

#define KUIPER_DTE_STATUS_DMA_DONE          (1)


#define DTE_BASE_ADDR      0X400000
uint32_t get_dte_reg(uint32_t dte_idx,size_t index);
#define set_dte_reg(dte_idx, index, value)  (*((volatile uint32_t *)((uint64_t)(index + DTE_BASE_ADDR + 0x200*dte_idx))) = (uint32_t)value)

/*!
 * \}
 */
void kuiper_dte_init_reg(uint32_t dte_index, int32_t fsm_id, uint32_t mode);
int kuiper_dte_set_dst_info(uint8_t dte_idx, uint64_t dst_addr, kuiper_dte_shuffle_cfg_t *shuffle_cfg);
int kuiper_dte_set_src_mode(uint8_t dte_idx, uint64_t src_addr, uint32_t data_len, kuiper_dte_shuffle_cfg_t *shuffle_cfg);
int kuiper_dte_trig_send(uint8_t dte_idx);
int kuiper_dte_check_dma_done(uint8_t dte_idx);
int kuiper_dte_clear_dma_status(uint8_t dte_idx);
int kuiper_dte_init(void);

/* DTE PMU functions */
int kuiper_dte_set_pmu_en(uint8_t channel, bool en);
uint8_t kuiper_dte_get_pmu_en(uint8_t channel);
int kuiper_dte_clear_pmu_reg(uint8_t channel);
uint8_t kuiper_dte_pmu_get_work_status(uint8_t channel);
uint64_t kuiper_dte_pmu_get_clk_cnts(void);
uint64_t kuiper_dte_pmu_get_clk_cnts_one_channel(uint8_t channel);
uint32_t kuiper_dte_pmu_get_cmd_success_cnts(uint8_t channel);
uint8_t kuiper_dte_pmu_get_cmd_failed_cnts(uint8_t channel);
uint64_t kuiper_dte_pmu_get_transfer_data_cnts(uint8_t channel);
uint64_t kuiper_dte_pmu_get_unalign_burst_cnts(uint8_t channel);
uint64_t kuiper_dte_pmu_get_idle_clk_cnts(uint8_t channel);
uint64_t kuiper_dte_pmu_get_all_idle_clk_cnts(void);

/*!
 * \}
 */

#endif /* KUIPER_DTE_H */
