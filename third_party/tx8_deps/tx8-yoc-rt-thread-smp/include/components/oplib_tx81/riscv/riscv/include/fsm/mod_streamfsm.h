/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef MOD_STREAMFSM_H
#define MOD_STREAMFSM_H


#include <stdint.h>
#include <fsm/kuiper_streamfsm.h>
#include <kuiper_common.h>

#define KUPER_STREAMFSM_CNTS	(64)
#define KUPER_STREAMFSM_PACKETS (32)

typedef struct {
	uint8_t online;
	kuiper_streamfsm_locate_t locate;
	kuiper_streamfsm_node_t *fsm_node;
} mod_kuiper_streamfsm_config;
/*!
 * \}
 */
int mod_kuiper_streamfsm_online(mod_kuiper_streamfsm_config *fsm_config, uint64_t buff_addr,
    uint32_t packet_size,uint8_t packet_cnt, uint32_t fsm_id, uint32_t stream_id);
int mod_kuiper_streamfsm_offline(mod_kuiper_streamfsm_config *fsm_config);
uint64_t mod_kuiper_streamfsm_ready_bitmap(void);
uint32_t mod_kuiper_streamfsm_packet_ready_bitmap(uint8_t fsm_id);
int mod_kuiper_streamfsm_clear_packet_status(uint8_t fsm_id, uint8_t packet_id);

uint32_t mod_kuiper_streamfsm_get_packet_receive_len(uint8_t fsm_id, uint8_t packet_id);
int mod_kuiper_streamfsm_clear_packet_receive_len(uint8_t fsm_id, uint8_t packet_id);

//RAS functions
uint8_t mod_kuiper_streamfsm_check_free_cnts(kuiper_streamfsm_locate_t locate);

int mod_kuiper_streamfsm_set_stream_base_addr_check_en(uint8_t fsm_id, uint8_t enable);
uint32_t mod_kuiper_streamfsm_get_stream_base_addr_check_en(uint8_t fsm_id);
/*!
 * \}
 */

#endif /* MOD_STREAMFSM_H */
