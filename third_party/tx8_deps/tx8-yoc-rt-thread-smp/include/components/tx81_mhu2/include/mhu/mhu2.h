/*
 * Arm SCP/MCP Software
 * Copyright (c) 2017-2020, Arm Limited and Contributors. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef MHU2_H
#define MHU2_H


/*!
 * \brief Read-only register.
 *
 * \details This qualifier can be used to describe a memory mapped read-only
 *      register.
 */
#define FWK_R const volatile

/*!
 * \brief Write-only register.
 *
 * \details This qualifier can be used to describe a memory mapped write-only
 *      register.
 */
#define FWK_W volatile

/*!
 * \brief Read/write register.
 *
 * \details This qualifier can be used to describe a memory mapped read/write
 *      register.
 */
#define FWK_RW volatile


#include <stdint.h>

#define CHANNEL_MAX 124
#define MHU_INT_ACCESS_STAT_CHCOMB  0x4 /*channel combine*/
#define MHU_INT_ACCESS_STAT_R2NR    0x2 /*ready to not ready*/
#define MHU_INT_ACCESS_STAT_NR2R    0x1 /*not ready to ready*/

#define MHU_INT_ACCESS_CLR_R2NR     0x2
#define MHU_INT_ACCESS_CLR_NR2R     0x1

#define MHU_INT_ACCESS_EN_CHCOMB    0x4
#define MHU_INT_ACCESS_EN_R2NR      0x2
#define MHU_INT_ACCESS_EN_NR2R      0x1

typedef struct {
    FWK_R  uint32_t  IIDR;
    FWK_R  uint32_t  AIDR;
    FWK_R  uint32_t  PID4;
    FWK_R  uint32_t  PID0;
    FWK_R  uint32_t  PID1;
    FWK_R  uint32_t  PID2;
    FWK_R  uint32_t  PID3;
    FWK_R  uint32_t  COMPID0;
    FWK_R  uint32_t  COMPID1;
    FWK_R  uint32_t  COMPID2;
    FWK_R  uint32_t  COMPID3;
} mhu2_id_reg_t;

typedef struct {
    FWK_R  uint32_t  STAT;
    uint8_t   RESERVED0[0xC - 0x4];
    FWK_W  uint32_t  STAT_SET;
    FWK_W  uint32_t  INT_STAT;
    FWK_W  uint32_t  INT_CLR;
    FWK_W  uint32_t  INT_EN;
    uint8_t   RESERVED1[0x20 - 0x1C];
} mhu2_send_channel_reg_t;

typedef struct {
    mhu2_send_channel_reg_t channel[CHANNEL_MAX];
    FWK_R  uint32_t  MHU_CFG; /*bit0-6*/
    FWK_RW uint32_t  RESP_CFG;
    FWK_RW uint32_t  ACCESS_REQUEST;
    FWK_R  uint32_t  ACCESS_READY;
    FWK_R  uint32_t  INT_ACCESS_STAT;
    FWK_W  uint32_t  INT_ACCESS_CLR;
    FWK_W  uint32_t  INT_ACCESS_EN;
    uint32_t  RESERVED0;
    FWK_R  uint32_t  CHCOMB_INT[4]; /*combined state for channel0-123*/
    uint8_t   RESERVED1[0xFC8 - 0xFB0];
    mhu2_id_reg_t id;
} mhu2_send_reg_t;

typedef struct {
    FWK_R  uint32_t  STAT;
    FWK_R  uint32_t  STAT_MASK;
    FWK_W  uint32_t  STAT_CLEAR;
    uint8_t   RESERVED0[0x10 - 0x0C];
    FWK_R  uint32_t  MASK;
    FWK_W  uint32_t  MASK_SET;
    FWK_W  uint32_t  MASK_CLEAR;
    uint8_t   RESERVED1[0x20 - 0x1C];
} mhu2_recv_channel_reg_t;

typedef struct {
    mhu2_recv_channel_reg_t channel[CHANNEL_MAX];
    FWK_R  uint32_t  MHU_CFG; /*bit0-6*/
    uint8_t   RESERVED0[0xF8C - 0xF84];
    FWK_RW uint32_t  RECV_READY;
    FWK_R  uint32_t  INT_STAT;
    FWK_W  uint32_t  INT_CLR;
    FWK_W  uint32_t  INT_EN;
    uint32_t  RESERVED1;
    FWK_R  uint32_t  CHCOMB_INT[4]; /*combined state for channel0-123*/
    uint8_t   RESERVED2[0xFC8 - 0xFB0];
    mhu2_id_reg_t id;
} mhu2_recv_reg_t;

#endif /* MHU2_H */
