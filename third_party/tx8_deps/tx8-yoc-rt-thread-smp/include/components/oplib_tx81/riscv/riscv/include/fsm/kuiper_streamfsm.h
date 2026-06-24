/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef KUIPER_STREAMFSM_H
#define KUIPER_STREAMFSM_H

// #include <fwk_id.h>
// #include <fwk_list.h>
#include <stdint.h>

#include <fsm/streamfsm_cfg.h>
#include <kuiper_common.h>

#define KUIPER_DDR_STREAMFSM_CNT          32 //All
#define KUIPER_SRAM_STREAMFSM_CNT         32 //All
#define KUIPER_STREAMFSM_MAX_CNT (KUIPER_DDR_STREAMFSM_CNT + KUIPER_SRAM_STREAMFSM_CNT)

#define KUIPER_DDR_STREAMFSM_KCORE_CNT          4 //kcore , ALL 0~31.  Score: 0~27;  Kcore: 28-31(current not used.)
#define KUIPER_SRAM_STREAMFSM_KCORE_CNT         4 //kcore , ALL 32~63. Score: 32~59; Kcore: 60-63

#define KUIPER_DDR_STREAMFSM_KCORE_START        (KUIPER_DDR_STREAMFSM_CNT - KUIPER_DDR_STREAMFSM_KCORE_CNT) // 28
#define KUIPER_SRAM_STREAMFSM_KCORE_START       (KUIPER_STREAMFSM_MAX_CNT - KUIPER_SRAM_STREAMFSM_KCORE_CNT) //60

#define KUIPER_DDR_STREAMFSM_KCORE_RDMA	  28
#define KUIPER_DDR_STREAMFSM_KCORE_WDMA	  29

#define KUIPER_STREAMFSM_PACKET_CNT_MAX   32
#define KUIPER_STREAMFSM_PACKET_SIZE_MAX  0x800000

typedef struct {
	// struct fwk_slist_node fsm_node;
	uint8_t enable;
	kuiper_streamfsm_locate_t type;
	uint8_t fsm_id;
	uint32_t stream_id; /*record stream id*/
} kuiper_streamfsm_node_t;
/*!
 * \}
 */
int fsm_init(void);
kuiper_streamfsm_node_t *kuiper_streamfsm_alloc(kuiper_streamfsm_locate_t type, uint32_t fsm_id, uint32_t stream_id, uint64_t buff_addr, uint32_t packet_size, uint8_t packet_cnt);
kuiper_streamfsm_node_t *set_kuiper_streamfsm_addr(int fsm_id, int stream_id, uint64_t addr);
kuiper_streamfsm_node_t *set_kuiper_streamfsm_length(uint32_t fsm_id, uint32_t packet_size);
int kuiper_streamfsm_release(kuiper_streamfsm_node_t *release_node);
uint64_t kuiper_streamfsm_check_stream_ready();
uint32_t kuiper_streamfsm_check_packet_ready(uint8_t fsm_id);
int kuiper_streamfsm_clear_packet_status(uint8_t fsm_id, uint8_t packet_id);

uint32_t kuiper_streamfsm_get_packet_receive_len(uint8_t fsm_id, uint8_t packet_id);
int kuiper_streamfsm_clear_packet_receive_len(uint8_t fsm_id, uint8_t packet_id);

uint8_t kuiper_streamfsm_check_free_cnts(kuiper_streamfsm_locate_t locate);

int kuiper_streamfsm_set_stream_base_addr_check_en(uint8_t fsm_id, uint8_t enable);
uint32_t kuiper_streamfsm_get_stream_base_addr_check_en(uint8_t fsm_id);
/*!
 * \}
 */

#endif /* KUIPER_STREAMFSM_H */
