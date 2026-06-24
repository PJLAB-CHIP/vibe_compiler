/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef _KUIPER_COMMON_H_
#define _KUIPER_COMMON_H_
#include  "tx8_config.h"

#define KUIPER_TILE_GLOBAL_BASE(tile_id) (((uint64_t)(0x10000 + (tile_id))) << 23)

/*! Success, request is pending */
#define FWK_PENDING         1

/*! Success */
#define FWK_SUCCESS         0

/*! Invalid parameter(s) */
#define FWK_E_PARAM         -1

/*! Invalid alignment */
#define FWK_E_ALIGN         -2

/*! Invalid size */
#define FWK_E_SIZE          -3

/*! Invalid handler or callback */
#define FWK_E_HANDLER       -4

/*! Invalid access or permission denied */
#define FWK_E_ACCESS        -5

/*! Value out of range */
#define FWK_E_RANGE         -6

/*! Operation timed out */
#define FWK_E_TIMEOUT       -7

/*! Memory allocation failed */
#define FWK_E_NOMEM         -8

/*! Invalid power state */
#define FWK_E_PWRSTATE      -9

/*! Not supported or disabled */
#define FWK_E_SUPPORT       -10

/*! Device error */
#define FWK_E_DEVICE        -11

/*! Handler or resource busy */
#define FWK_E_BUSY          -12

/*! OS error response */
#define FWK_E_OS            -13

/*! Unexpected or invalid data */
#define FWK_E_DATA          -14

/*! Invalid state for the device or component */
#define FWK_E_STATE         -15

/*! Accessing an uninitialized resource */
#define FWK_E_INIT          -16

/*! Configuration overwritten */
#define FWK_E_OVERWRITTEN   -17

/*! Unrecoverable error */
#define FWK_E_PANIC         -18


typedef enum {
    KUIPER_STREAMFSM_DDR = 0,
    KUIPER_STREAMFSM_SRAM = 1,
    KUIPER_STREAMFSM_CNT = 2,
} kuiper_streamfsm_locate_t;


typedef enum {
    KUIPER_OUTSTREAM_STATE_OFFLINE,
    KUIPER_OUTSTREAM_STATE_WAITFSM,
    KUIPER_OUTSTREAM_STATE_BROADCAST_ONLINE,
    KUIPER_OUTSTREAM_STATE_WAIT_ONLINE,
    KUIPER_OUTSTREAM_STATE_ONLINE,
    KUIPER_OUTSTREAM_STATE_BROADCAST_OFFLINE,
    KUIPER_OUTSTREAM_STATE_WAIT_OFFLINE,
} kuiper_outstream_state_t;

typedef enum {
    KUIPER_INSTREAM_STATE_OFFLINE,
    KUIPER_INSTREAM_STATE_WAIT_ONLINE,
    KUIPER_INSTREAM_STATE_ONLINE,
    KUIPER_INSTREAM_STATE_WAIT_OFFLINE,
} kuiper_instream_state_t;
#if 0
typedef enum {
    KUIPER_OUTSTREAM_ATTR_UNICAST,
    KUIPER_OUTSTREAM_ATTR_GATHER,
} kuiper_outstream_attribute_t;
typedef enum {
    KUIPER_INSTREAM_ATTR_UNICAST,
    KUIPER_INSTREAM_ATTR_BROADCAST,
    KUIPER_INSTREAM_ATTR_SCATTER,
    KUIPER_INSTREAM_ATTR_STRIDE_SLICE,
} kuiper_instream_attribute_t;

//Bidirection Kernel->Spatial and Spatial->Kernel use this simplify payload format

#define ONLINESTREAM 0
#define OFFLINESTREAM 1
#define WAITSTREAM 2
#define REQSTREAM 3
#define POPSTREAM 4
#define PUSHSTREAM 5
#define ONLINESTREAMREP 6
#define OFFLINESTREAMREP 7
#define WAITSTREAMREP 8
#define REQSTREAMREP 9

typedef struct __attribute((packed)) {
        uint16_t src_tile; //logic tile id src_tile = tile_id_coordx + tile_x_len * tile_id_coordy
        uint8_t src_core; //for kernel core(2,3,4,5) for spatial core(0,1)
        uint8_t op; // As above define
        uint32_t stream_id; // logic stream id in stream config
        uint64_t stream_addr;
}kernel_spatial_payload

#endif
/*NOTE:keep with CFG definition*/
typedef enum {
    KUIPER_STREAM_ATTR_UNICAST = 0,
    KUIPER_INSTREAM_ATTR_BROADCAST = 1,
    KUIPER_INSTREAM_ATTR_SCATTER = 2,
    KUIPER_INSTREAM_ATTR_SHUFFLE = 3,
    KUIPER_OUTSTREAM_ATTR_GATHER = 4,/*gather for out stream*/
} kuiper_stream_attribute_t;


typedef enum {
    /*local command from kernel core*/
    KUIPER_MAILBOX_CMD_STREAM_ONLINE_REQUEST = 0x0,
    KUIPER_MAILBOX_CMD_STREAM_OFFLINE_REQUEST = 0x1,
    KUIPER_MAILBOX_CMD_OUTSTREAM_WAITSTREAM = 0x2,
    KUIPER_MAILBOX_CMD_INSTREAM_REQSTREAM = 0x3,
    KUIPER_MAILBOX_CMD_OUTSTREAM_POP = 0x4,
    KUIPER_MAILBOX_CMD_INSTREAM_PUSH = 0x5,
    KUIPER_MAILBOX_CMD_STREAM_ONLINE_ACK = 0x6,
    KUIPER_MAILBOX_CMD_STREAM_OFFLINE_ACK = 0x7,
    KUIPER_MAILBOX_CMD_OUTSTREAM_WAITSTREAM_ACK = 0x8,
    KUIPER_MAILBOX_CMD_INSTREAM_REQSTREAM_ACK = 0x9,

    /*remote command from spatial core*/
    KUIPER_MAILBOX_CMD_OUTSTREAM_ONLINE_BROADCAST = 0x20,
    KUIPER_MAILBOX_CMD_OUTSTREAM_ONLINE_BROADCAST_RESP,
    KUIPER_MAILBOX_CMD_OUTSTREAM_OFFLINE_BROADCAST,
    KUIPER_MAILBOX_CMD_OUTSTREAM_OFFLINE_BROADCAST_RESP,

} kuiper_mailbox_cmd_t;

/*local mailbox command definition*/
typedef struct __attribute((packed)) {
        uint16_t src_tile; //logic tile id src_tile = tile_id_coordx + tile_x_len * tile_id_coordy
        uint8_t src_core; //for kernel core(2,3,4,5) for spatial core(0,1)
        uint8_t op; // As above define
        uint32_t stream_id; // logic stream id in stream config
        uint64_t stream_addr;
	uint8_t preload_pkt_cnt;
}kuiper_mailbox_local_cmd_t;


/*remote mailbox command definition*/
typedef struct __attribute((packed)){
    uint16_t src_tile;
    uint8_t cmd; /*kuiper_mailbox_cmd_t*/
    uint8_t flag;
    uint32_t payload[];
} kuiper_mailbox_remote_head_t;

typedef struct __attribute((packed)){
    uint32_t tgt_instream_id;
    uint32_t src_outstream_id;
    uint8_t src_fsm_id;
    uint8_t buff_type;
    uint64_t rb_ctrl_addr;//
} kuiper_outstream_broadcast_payload_t;

typedef struct __attribute((packed)){
    uint32_t resp_instream_id;
    uint32_t tgt_outstream_id;
    uint64_t bundrb_ctrl_addr;//
} kuiper_outstream_broadcast_resp_t;

#define SCFG_TILE_ID_ADDR  0x6A0058 //KUIPER_ADDR_MAP_REG_BASE 0x6A0000
static inline uint16_t kuiper_get_tileid(void)
{
#if 0
    uint16_t result;
    /*TEMP*/
    __asm volatile("csrr %0, marchid" : "=r"(result));
    return result;
#else
    uint16_t id;
    id = *(volatile uint16_t *)(SCFG_TILE_ID_ADDR);
    return id;
#endif
}

static inline uint16_t kuiper_get_phyid(void)
{
    #define SCFG_TILE_PHY_ID_ADDR  0x6A005c //KUIPER_ADDR_MAP_REG_BASE 0x6A0000

    uint16_t id;

    id = *(volatile uint16_t *)(SCFG_TILE_PHY_ID_ADDR);
    return id;
}

static inline unsigned long long get_heap_addr(int index)
{
    unsigned long long addr = 0;

    switch(index) {
    case 0:
        addr = LOCAL_TILE_HEAP_ADDRESS00;
        break;
    case 1:
        addr = LOCAL_TILE_HEAP_ADDRESS01;
        break;
    case 2:
        addr = LOCAL_TILE_HEAP_ADDRESS02;
        break;
    case 3:
        addr = LOCAL_TILE_HEAP_ADDRESS03;
        break;
    case 4:
        addr = LOCAL_TILE_HEAP_ADDRESS04;
        break;
    case 5:
        addr = LOCAL_TILE_HEAP_ADDRESS05;
        break;
    case 6:
        addr = LOCAL_TILE_HEAP_ADDRESS06;
        break;
    case 7:
        addr = LOCAL_TILE_HEAP_ADDRESS07;
        break;
    case 8:
        addr = LOCAL_TILE_HEAP_ADDRESS08;
        break;
    case 9:
        addr = LOCAL_TILE_HEAP_ADDRESS09;
        break;
    case 10:
        addr = LOCAL_TILE_HEAP_ADDRESS10;
        break;
    case 11:
        addr = LOCAL_TILE_HEAP_ADDRESS11;
        break;
    case 12:
        addr = LOCAL_TILE_HEAP_ADDRESS12;
        break;
    case 13:
        addr = LOCAL_TILE_HEAP_ADDRESS13;
        break;
    case 14:
        addr = LOCAL_TILE_HEAP_ADDRESS14;
        break;
    case 15:
        addr = LOCAL_TILE_HEAP_ADDRESS15;
        break;
    default:
        addr = 0;
    }

    return addr;
}

/* Common MHU command definition */
typedef enum {
    KUIPER_MHU_CMD_DISCOVERY_START = 1,
    KUIPER_MHU_CMD_DISCOVERY_DONE = 2,
    KUIPER_MHU_CMD_FIRST_LAYER_ONLINE = 3,
    KUIPER_MHU_CMD_LAST_LAYER_ONLINE = 4,
    KUIPER_MHU_CMD_LAST_LAYER_KICK = 5,
    KUIPER_MHU_CMD_STREAM_CONFIG = 6,
    KUIPER_MHU_CMD_STREAM_CONFIG_FINISHED = 7,
    KUIPER_MHU_CMD_FLOW_CTL = 8,
    KUIPER_MHU_CMD_SCORE_READY = 9,
    KUIPER_MHU_CMD_KCORE_POWEROFF = 10,
    KUIPER_MHU_CMD_AP_REQ_POWEROFF = 11,
    KUIPER_MHU_CMD_NPU_RESP_POWEROFF = 12,
    KUIPER_MHU_CMD_MSG_COUNT,

    KUIPER_MHU_CMD_KCORE_RINGBUFFER = 16,
    KUIPER_MHU_CMD_KCORE_CALCULATE_DONE = 17,
    KUIPER_MHU_CMD_RESET_DONE = 18,
    KUIPER_MHU_CMD_GLOBALID_NOTIFY = 19,

    KUIPER_MHU_CMD_UNKNOWN = 0xFF,
} kuiper_mhu_cmd_t;

typedef struct __attribute((packed)){
    uint64_t cfg_addr;
} kuiper_mhu_stream_cfg_info_t;

typedef struct __attribute((packed)){
    uint32_t stream_id;
    uint64_t rb_ctrl_addr;
} kuiper_mhu_last_layer_info_t;

typedef struct __attribute((packed)){
    uint16_t tile_id;
    uint8_t  fsm_id;
    uint8_t  locate; /*buffer in DDR:0, SRAM:1*/
    uint32_t stream_id;
    uint64_t rb_ctrl_addr;
} kuiper_mhu_first_layer_info_t;

#endif
