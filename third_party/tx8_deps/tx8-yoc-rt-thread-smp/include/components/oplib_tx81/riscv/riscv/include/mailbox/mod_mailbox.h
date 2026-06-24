/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef _MOD_MAILBOX_H_
#define _MOD_MAILBOX_H_

#include <stdint.h>
#include <stdbool.h>

/* same as KUIPER_MAILBOX_PAYLOAD_REG_MAX but not include internal
 * header file use another flag name
 */
#define MAILBOX_PAYLOAD_REG_CNT   8

/*channel name*/
typedef enum {
    MAILBOX_TX_CH_OUTSTREAM_LOCAL = 0,
    MAILBOX_TX_CH_OUTSTREAM_REMOTE = 1,
    MAILBOX_TX_CH_INSTREAM_LOCAL = 2,
    MAILBOX_TX_CH_INSTREAM_REMOTE = 3,
    MAILBOX_TX_CH_KERNEL0_LOCAL = 4,
    MAILBOX_TX_CH_KERNEL1_LOCAL = 5,
    MAILBOX_TX_CH_KERNEL2_LOCAL = 6,
    MAILBOX_TX_CH_KERNEL3_LOCAL = 7,
    MAILBOX_TX_CH_RESERVED,
} mod_mailbox_tx_channel_t;

typedef enum {
    MAILBOX_RX_CH_OUTSTREAM_LOCAL = 0,
    MAILBOX_RX_CH_OUTSTREAM_REMOTE = 1,
    MAILBOX_RX_CH_INSTREAM_LOCAL = 2,
    MAILBOX_RX_CH_INSTREAM_REMOTE = 3,
    MAILBOX_RX_CH_KERNEL0_LOCAL = 4,
    MAILBOX_RX_CH_KERNEL1_LOCAL = 5,
    MAILBOX_RX_CH_KERNEL2_LOCAL = 6,
    MAILBOX_RX_CH_KERNEL3_LOCAL = 7,
    MAILBOX_RX_CH_RESERVED,
} mod_mailbox_rx_channel_t;

/*target information*/
typedef struct {
    uint16_t tile_id;
    uint8_t target_ch;
    uint8_t payload_reg_cnt;
    uint32_t *payload;
} mod_mailbox_tx_config_t;

/*for remote mailbox*/
typedef struct {
    uint8_t tx_ch;
    uint8_t win;
    uint8_t operation;
    uint8_t stream_type;
    uint32_t stream_id;
} mod_mailbox_tx_remote_param_t;
/*!
 * \addtogroup GroupModules Modules
 * \{
 */

/*!
 * \defgroup GroupModuleMAILBOX  Driver
 *
 * \brief Device driver module for the MOD_MAILBOX.
 *
 * \{
 */
volatile uint8_t mod_mailbox_rx_check_msg_cnt(mod_mailbox_rx_channel_t ch);
int mod_mailbox_rx_pop_msg(mod_mailbox_rx_channel_t ch, uint32_t *msg, uint8_t size);
int mod_mailbox_aquire_tx_win(mod_mailbox_tx_channel_t ch, uint8_t *win);
int mod_mailbox_release_tx_win(mod_mailbox_tx_channel_t ch, uint8_t win);
int mod_mailbox_tx_msg(mod_mailbox_tx_channel_t ch, uint8_t win, mod_mailbox_tx_config_t *config, 
            bool remote, mod_mailbox_tx_remote_param_t *remote_param);
/*!
 * \brief mailbox configuration data.
 */


/*!
 * \}
 */

/*!
 * \}
 */

#endif /* MOD_MAILBOX_H */
