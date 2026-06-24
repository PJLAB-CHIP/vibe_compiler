#ifndef _STREAM_RT_H_
#define _STREAM_RT_H_

#include "mailbox/mod_mailbox.h"

#define INSTREAM (uint64_t)0
#define OUTSTREAM (uint64_t)1

#define ONLINESTREAM (uint64_t)0
#define OFFLINESTREAM (uint64_t)1
#define WAITSTREAM (uint64_t)2
#define REQSTREAM (uint64_t)3
#define	POPSTREAM (uint64_t)4
#define PUSHSTREAM (uint64_t)5
#define ONLINESTREAMREP (uint64_t)6
#define OFFLINESTREAMREP (uint64_t)7
#define WAITSTREAMREP (uint64_t)8
#define REQSTREAMREP (uint64_t)9

#define PAYLOAD_STREAM_ID_OFFSET 32
#define PAYLOAD_OP_TYPE_OFFSET 24
#define PAYLOAD_CORE_ID_OFFSET 16


uint32_t GenPayload(uint64_t* payload, uint64_t tile_id, uint64_t core_id, uint64_t channel_id,
                    uint64_t op_type, uint64_t stream_type, uint64_t stream_id, uint64_t stream_addr);

//uint32_t RecieveMailbox(uint32_t tile_id, uint32_t core_id, uint32_t channel_id); 

uint32_t SendMailbox(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                     uint32_t remote, uint64_t* payload);

uint32_t OnlineStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                      uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);

uint32_t OnlineStreamPreload(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                             uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr , uint8_t preload_pkt_cnt);

uint32_t OfflineStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                       uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
uint32_t WaitStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                    uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);

uint32_t ReqStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                   uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);

uint32_t PushStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                    uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);

uint32_t PopStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
                   uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);

#endif
