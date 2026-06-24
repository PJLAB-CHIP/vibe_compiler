/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef _MOD_KUIPER_MAILBOX_H_
#define _MOD_KUIPER_MAILBOX_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "mailbox_tx_cfg.h"
#include "mailbox_rx_cfg.h"

/*Mailbox registre base, each block take 128KB*/
#define KUIPER_MAILBOX_TX_REG_BASE   0x640000
#define KUIPER_MAILBOX_RX_REG_BASE   0x660000

#define KUIPER_MAILBOX_CORE_MAX            6

/* kuiper mailbox supported channel count*/
#define KUIPER_MAILBOX_TX_CH_MAX           8
#define KUIPER_MAILBOX_RX_CH_MAX           8

/* each channel can support window(queue) size */
#define KUIPER_MAILBOX_TX_WIN_MAX           8
#define KUIPER_MAILBOX_RX_WIN_MAX           16

/* mailbox can support max payload register count */
#define KUIPER_MAILBOX_PAYLOAD_REG_MAX           8


/*TX channel lock register bits definitions*/
#define KUIPER_MAILBOX_TX_LOCKCH_FULL_FLAG         0x1
#define KUIPER_MAILBOX_TX_LOCKCH_WIN_MASK         0x70
#define KUIPER_MAILBOX_TX_LOCKCH_WIN_OFFSET       0x4

/*Tx send command: 0B11 for the corresponding 2bits field*/
#define KUIPER_MAILBOX_TX_SEND_CMD          3

/*Tx send state*/
#define KUIPER_MAILBOX_TX_STATE_SENDING     0
#define KUIPER_MAILBOX_TX_STATE_SUCCESS     5
#define KUIPER_MAILBOX_TX_STATE_REJECTED    6
#define KUIPER_MAILBOX_TX_STATE_MASK        7

typedef enum {
    KUIPER_MAILBOX_TYPE_TX = 0,
    KUIPER_MAILBOX_TYPE_RX = 1,
    KUIPER_MAILBOX_TYPE_COUNT,
} kuiper_mailbox_type;

typedef enum {
    KUIPER_MAILBOX_SCORE0 = 0,
    KUIPER_MAILBOX_SCORE1 = 1,
    KUIPER_MAILBOX_KCORE0 = 2,
    KUIPER_MAILBOX_KCORE1 = 3,
    KUIPER_MAILBOX_CORE_COUNT,
} kuiper_mailbox_core;


/*tx registers*/
#define TX_LOCK_REG_LOCK(reg, ch) \
    reg->apb_channel_lock[ch].field.channel_lock_lock
#define TX_LOCK_REG_VAL(reg, ch) \
    reg->apb_channel_lock[ch].value

#define TX_CONTROL_REG_CTRL(reg, ch, win) \
    reg->apb_channel_ctrl[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].field.channel_ctrl_send
#define TX_CONTROL_REG_VAL(reg, ch, win) \
    reg->apb_channel_ctrl[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].value

#define TX_STATUS_REG_ST(reg, ch, win) \
    reg->apb_channel_status[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].field.channel_status_status
#define TX_STATUS_REG_VAL(reg, ch, win) \
    reg->apb_channel_status[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].value

#define TX_FAIL_REG_CNT(reg, ch) \
    reg->apb_channel_tx_fail_cnt[ch].field.channel_tx_fail_cnt_fail_cnt
#define TX_FAIL_REG_VAL(reg, ch) \
    reg->apb_channel_tx_fail_cnt[ch].value

#define TX__LOCK_FAIL_REG_CNT(reg, ch) \
    reg->apb_channel_lock_fail_cnt[ch].field.channel_lock_fail_cnt_fail_cnt
#define TX__LOCK_FAIL_REG_VAL(reg, ch) \
    reg->apb_channel_lock_fail_cnt[ch].value

#define TX_TARGET_REG_ADDR(reg, ch, win) \
    reg->apb_msi_tx_info[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].addr.field.msi_tx_addr_tgt_base_addr
#define TX_TARGET_REG_CH(reg, ch, win) \
    reg->apb_msi_tx_info[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].addr.field.msi_tx_addr_tgt_channel
#define TX_TARGET_REG_CORE(reg, ch, win) \
    reg->apb_msi_tx_info[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].addr.field.msi_tx_addr_int_tgt_core
#define TX_TARGET_REG_VAL(reg, ch, win) \
    reg->apb_msi_tx_info[(ch * KUIPER_MAILBOX_TX_WIN_MAX) + win].addr.value

/*rx registers*/
#define RX_CORE_INT_REG_MAP(reg)  \
    reg->apb_core_doorbell_int.field.core_doorbell_int_core_doorbell_int
#define RX_CORE_INT_REG_VAL(reg)  \
    reg->apb_core_doorbell_int.value

#define RX_MB_INT_REG_MAP(reg)  \
    reg->apb_mb_doorbell_int.field.mb_doorbell_int_mb_doorbell_int
#define RX_MB_INT_REG_VAL(reg)  \
    reg->apb_mb_doorbell_int.value

#define RX_MB_DOORBELL_REG_STATUS(reg, core) \
    reg->apb_mb_doorbell_status[core].field.mb_doorbell_status_mb_doorbell_status
#define RX_MB_DOORBELL_REG_VAL(reg, core) \
    reg->apb_mb_doorbell_status[core].value

#define RX_INT_MAP_REG_BMP(reg, ch) \
    reg->apb_int_map[ch].field.int_map_int_map_bmp
#define RX_INT_MAP_REG_VAL(reg, ch) \
    reg->apb_int_map[ch].value

#define RX_CNT_REG_PENDING(reg, ch) \
    reg->rx_info[ch].rx_cnt.field.msi_rx_cnt_pending_req_cnt
#define RX_CNT_REG_READ_DONE(reg, ch) \
    reg->rx_info[ch].rx_cnt.field.msi_rx_cnt_read_done
#define RX_CNT_REG_FAIL_CNT(reg, ch) \
    reg->rx_info[ch].rx_cnt.field.msi_rx_cnt_rx_fail_cnt
#define RX_CNT_REG_VAL(reg, ch) \
    reg->rx_info[ch].rx_cnt.value

/*local functions get mailbox register base*/
static inline mailbox_tx_cfg_s *get_kuiper_mailbox_tx_reg()
{
    return (mailbox_tx_cfg_s *)(KUIPER_MAILBOX_TX_REG_BASE);
}
static inline mailbox_rx_cfg_s *get_kuiper_mailbox_rx_reg()
{
    return (mailbox_rx_cfg_s *)(KUIPER_MAILBOX_RX_REG_BASE);
}

/*!
 * \brief Device driver module for the KUIPER_MAILBOX.
 *
 * \{
 */
int kuiper_mailbox_init(kuiper_mailbox_type type, uint8_t ch);

static inline uint8_t kuiper_mailbox_tx_aquire_win(uint8_t ch)
{
    uint8_t win;
    mailbox_tx_cfg_s *tx_reg;

    tx_reg = get_kuiper_mailbox_tx_reg();
    win = TX_LOCK_REG_LOCK(tx_reg, ch);

    if((win & KUIPER_MAILBOX_TX_LOCKCH_FULL_FLAG) != 0) {
        /*no valid tx window*/
        return KUIPER_MAILBOX_TX_CH_MAX;
    }
    /**/
    win &= KUIPER_MAILBOX_TX_LOCKCH_WIN_MASK;
    return (win >> KUIPER_MAILBOX_TX_LOCKCH_WIN_OFFSET);
}

static inline int kuiper_mailbox_tx_release_win(uint8_t ch, uint8_t win)
{

    uint32_t reg_val;
    mailbox_tx_cfg_s *tx_reg;

    /* when did not trigger send, status is 0, the caller should avoid to release
     * a not used window
     */
    tx_reg = get_kuiper_mailbox_tx_reg();
    /*valid state*/
    reg_val = 1 << win;
    TX_LOCK_REG_VAL(tx_reg, ch) = reg_val;
    return 0;
}

static inline int kuiper_mailbox_tx_prepare(uint8_t ch, uint8_t win, apb_msi_tx_addr_s target, uint32_t *payload, uint8_t reg_cnt)
{
    mailbox_tx_cfg_s *tx_reg;
    apb_msi_tx_info_s *tx_info_cfg;
    volatile uint32_t *tx_info_payload;
    //if((payload == NULL) || (reg_cnt == 0) || (reg_cnt > KUIPER_MAILBOX_PAYLOAD_REG_MAX)) {
    //    return FWK_E_DATA;
    //}

    tx_reg = get_kuiper_mailbox_tx_reg();
    tx_info_cfg = &tx_reg->apb_msi_tx_info[(ch * 8) + win];
    tx_info_cfg->addr.value = target.value;

    tx_info_payload = &tx_info_cfg->payload[0];
    /*payload register*/
    for(int i = 0; i < reg_cnt; i++) {
        tx_info_payload[i] = payload[i];
    }

    return 0;
}
static inline int kuiper_mailbox_tx_trig_send(uint8_t ch, uint8_t win)
{
    uint32_t reg_val;
    mailbox_tx_cfg_s *tx_reg;

    tx_reg = get_kuiper_mailbox_tx_reg();
    reg_val = KUIPER_MAILBOX_TX_SEND_CMD;
    TX_CONTROL_REG_VAL(tx_reg, ch, win) = reg_val;

    return 0;
}

static inline int kuiper_mailbox_tx_check_send_state(uint8_t ch, uint8_t win)
{
	int status;
	uint32_t send_status;
	mailbox_tx_cfg_s *tx_reg;

	tx_reg = get_kuiper_mailbox_tx_reg();
	send_status = TX_STATUS_REG_ST(tx_reg, ch, win);
	//send_status &= KUIPER_MAILBOX_TX_STATE_MASK;

	if (send_status == 0) {
		return 1;
	}

	/*mailbox get response*/
	if (send_status == KUIPER_MAILBOX_TX_STATE_SUCCESS) {
		status = 0;
	} else if (send_status == KUIPER_MAILBOX_TX_STATE_REJECTED) {
		status = -8;
	} else {
		status = -5;
	}
	/*clear status*/
	TX_STATUS_REG_VAL(tx_reg, ch, win) = KUIPER_MAILBOX_TX_STATE_MASK;
	/*if send status keep 0b00 for long time, need confirm if trig send success*/
	return status;
}

/*
 * tx function
 */


/*
 * rx function
 */

int kuiper_mailbox_rx_set_intmap(uint8_t core_id, uint8_t bitmap);
uint8_t  kuiper_mailbox_rx_get_core_int_status(uint8_t core_id);
bool  kuiper_mailbox_rx_check_channel_int_status(uint8_t ch);

static inline volatile uint8_t  kuiper_mailbox_rx_get_pending_count(uint8_t ch)
{
    volatile uint8_t msg_cnt = 0;
    mailbox_rx_cfg_s *rx_reg;

    rx_reg = get_kuiper_mailbox_rx_reg();
    msg_cnt = RX_CNT_REG_PENDING(rx_reg, ch);
    return msg_cnt;
}

static inline int  kuiper_mailbox_rx_read_message(uint8_t ch, uint32_t *payload, uint8_t reg_cnt)
{
    mailbox_rx_cfg_s *rx_reg;
    volatile uint32_t *tx_info_payload;
    rx_reg = get_kuiper_mailbox_rx_reg();

    tx_info_payload = &rx_reg->rx_info[ch].payload[0];

    /*read payload*/
    for(int i = 0; i < reg_cnt; i++) {
        payload[i] = tx_info_payload[i];
    }

    return 0;
}

static inline int  kuiper_mailbox_rx_read_clear(uint8_t ch)
{
    mailbox_rx_cfg_s *rx_reg;
    apb_msi_rx_cnt_s rx_cnt_reg;

    rx_reg = get_kuiper_mailbox_rx_reg();
    /*clear current msg by write 1 to msi_rx_cnt_c[X].read_done*/
    rx_cnt_reg.value = RX_CNT_REG_VAL(rx_reg, ch);
    rx_cnt_reg.field.msi_rx_cnt_read_done = 1;

    RX_CNT_REG_VAL(rx_reg, ch) = rx_cnt_reg.value;

    while(RX_CNT_REG_READ_DONE(rx_reg, ch) == 1) {
        /*wait till clear done*/
    }
    return 0;
}

static inline int kuiper_mailbox_rx_intr_enable(uint8_t ch, kuiper_mailbox_core core)
{
    mailbox_rx_cfg_s *rx_reg = get_kuiper_mailbox_rx_reg();
    RX_INT_MAP_REG_VAL(rx_reg, ch) |= (1 << core);
    return 0;
}

static inline int kuiper_mailbox_rx_intr_disable(uint8_t ch, kuiper_mailbox_core core)
{
    mailbox_rx_cfg_s *rx_reg = get_kuiper_mailbox_rx_reg();
    RX_INT_MAP_REG_VAL(rx_reg, ch) &= ~(1 << core);
    return 0;
}

/*kuiper mailbox RAS function*/
uint32_t kuiper_mailbox_ras_get_tx_lockfail_count(uint8_t ch);
uint32_t kuiper_mailbox_ras_get_rx_miss_count(uint8_t ch);


/*!
 * \}
 */

/*!
 * \}
 */

#endif /* MOD_KUIPER_MAILBOX_H */
