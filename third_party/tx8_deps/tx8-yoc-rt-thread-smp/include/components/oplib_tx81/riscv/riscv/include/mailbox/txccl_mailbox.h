#include <riscv.h>
#include "mailbox/mod_mailbox.h"

/*  为TXCCL功能传递基础配置的MailxBox
    1. TsmTxcclRecv作为P2P的接收端，需要通过MailBox告知TsmTxcclSend侧接收端的物理地址及自己的tild_id
    2. 通知完后，需要阻塞等待TsmTxcclSend侧回复p2p传输完成的消息，退出
    3. TsmTxcclSend作为P2P的发送端，需要知道对端的物理地址以及tile_id，所以需要阻塞住轮询MailBox，查到对端发来的配置
    4. 发起p2p传输，完成后向对端发送MailBox通知传输完成消息，退出
*/

KrtRetCode txccl_send(TileDteCfg tile_dte_cfg);
KrtRetCode txccl_recv(TileDteCfg tile_dte_cfg);