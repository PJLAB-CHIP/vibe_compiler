#ifndef KCORE_MHU_H
#define KCORE_MHU_H

#include <stdint.h>
#include "kuiper_common.h"

/*TODO:collect device address into map head file*/
#define KUIPER_MHU_INSTREAM_SEND_BASE      0x680000
#define KUIPER_MHU_INSTREAM_RECV_BASE      0x681000

#define KUIPER_MHU_OUTSTREAM_SEND_BASE     0x682000
#define KUIPER_MHU_OUTSTREAM_RECV_BASE     0x683000

#define KUIPER_MHU_KCORE_SEND_BASE         0x684000
#define KUIPER_MHU_KCORE_RECV_BASE         0x685000

#define KUIPER_MHU_PAYLOAD_BASE            0x1F5000000 /*AP defined at DDR*/
#define KUIPER_MHU_PAYLOAD_CH_BLOCK_SIZE   256	/* MHU message head size + payload size */

typedef enum {
	KUIPER_MHU_SLOT_0        = 0,

	/*the counts of slot used. TBD: 0*/
	KUIPER_MHU_SLOT_CNTS_USED,

	/*the max counts of slot, (0, 31)*/
	KUIPER_MHU_SLOT_MAX      = 32
} mod_kuiper_mhu2_slot_t;

typedef enum {
	KUIPER_MHU2_CH_CMD       = 0,
	KUIPER_MHU2_CH_DATA      = 1,
	KUIPER_MHU2_CH_CNT       = 2,
} mod_kuiper_mhu2_channel_t;

typedef enum {
	KUIPER_MHU2_DEV_INSTREAM       = 0,
	KUIPER_MHU2_DEV_OUTSTREAM      = 1,
	KUIPER_MHU2_DEV_KCORE          = 2,
	KUIPER_MHU2_DEV_CNT            = 3,
} mod_kuiper_mhu2_dev_t;

#define KUIPER_MHU_BUFFER_MSG_CNT_MAX 16

/*!
 *	MHU message head struct.
 *	All of messages send from MHU or receive by MHU must contain this head at the beginning,
 *	and other additional information is placed in payload[].
*/
typedef struct __attribute((packed)) {
	/*! MHU message magic flag */
	uint8_t magic;

	/*! payload length in bytes, not include this head length */
	uint8_t payload_len;

	/*! message command id */
	uint8_t cmd;

	/*! not used now */
	uint8_t reserve;

	/*! the additional information carried by this MHU message */
	uint8_t payload[];
} kuiper_mhu_cmd_head_t;

#define KUIPER_MHU_MSG_HEAD_LEN			(sizeof(kuiper_mhu_cmd_head_t))
#define KUIPER_MHU_MSG_PAYLOAD_LEN_MAX	(KUIPER_MHU_PAYLOAD_CH_BLOCK_SIZE - sizeof(kuiper_mhu_cmd_head_t))
#define KUIPER_MHU_MSG_LEN_MAX			(KUIPER_MHU_PAYLOAD_CH_BLOCK_SIZE)

#define KUIPER_MHU_CMD_HEAD_CONSTRUCT(_msg, _cmd, _payload_size, _flag) \
do { \
	kuiper_mhu_cmd_head_t *mhu_head = (kuiper_mhu_cmd_head_t *)_msg; \
	mhu_head->magic = 0x6e; \
	mhu_head->cmd = _cmd; \
	mhu_head->payload_len = ((_payload_size) > KUIPER_MHU_MSG_PAYLOAD_LEN_MAX) ? KUIPER_MHU_MSG_PAYLOAD_LEN_MAX : (_payload_size); \
	mhu_head->reserve = _flag; \
} while (0)

#define KUIPER_MHU_MSG_GET_PAYLOAD_LEN(msg_addr) (((kuiper_mhu_cmd_head_t *)(msg_addr))->payload_len)

typedef struct {
	/*! Base address of the registers of the outgoing MHU */
	uint64_t send;

	/*! Base address of the registers of the incoming MHU */
	uint64_t recv;

	/*! Base address of payload */
	uint64_t payload_base;

	/*! device number */
	uint8_t dev;
} mod_mhu2_config_t;

/*!
 * \}
 */

int mhu_send_cmd_without_payload(uint8_t slot, kuiper_mhu_cmd_t cmd);
/* Init functions*/
int kcore_mhu_init(const void *data);

/* Receive functions */
uint32_t mhu_recv_check_state(uint8_t ch);
int mhu_recv_read_payload(uint8_t ch, uint8_t slot, uint8_t *target, uint16_t length);

/* Send functions */
int mod_mhu_send_msg(uint8_t ch, uint8_t slot, uint8_t *msg, uint16_t msg_len);

/* mhu monitor functions*/
int init_mhu_monitor();

void init_mhu_poweroff_task();

/* tile powers it off herself*/
int mhu_power_off_self(void);
/*!
 * \}
 */

enum {
	POWER_MHU_INIT=1,
	POWER_MHU_WAIT_MSG=2,
	POWER_MHU_GET_MSG=3,
	POWER_MHU_HANDLE_MSG=4,
	POWER_MHU_SEMI_SEND=5,
	POWER_MHU_MAIN_END =6,
};
enum {
	KCORE0_POWER_MHU_INIT=1,
	KCORE0_POWER_MHU_SEMI_RECV=2,
	KCORE0_POWER_MHU_RESP_AP=3,
	KCORE0_POWER_MHU_DISABLE_DCACHE=4,
	KCORE0_POWER_MHU_WFI=5,
};

enum {
	KCORE1_POWER_MHU_INIT=1,
	KCORE1_POWER_MHU_SEMI_RECV=2,
	KCORE1_POWER_MHU_RESP_AP=3,
	KCORE1_POWER_MHU_DISABLE_DCACHE=4,
	KCORE1_POWER_MHU_WFI=5,
};

#endif /* KCORE_MHU_H */
