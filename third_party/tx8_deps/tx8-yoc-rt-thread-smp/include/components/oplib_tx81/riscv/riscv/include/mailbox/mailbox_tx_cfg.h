#include <stdint.h>

typedef union {
    volatile struct {
       volatile uint32_t channel_lock_lock:8;
       volatile uint32_t :24;

    } field;

    volatile  uint32_t value;
} apb_channel_lock_s;

typedef union {
    volatile struct {
       volatile uint32_t channel_ctrl_send:2;
       volatile uint32_t :30;

    } field;

    volatile  uint32_t value;
} apb_channel_ctrl_s;

typedef union {
    volatile struct {
     volatile uint32_t channel_status_status:24;
     volatile uint32_t :8;

    } field;

    volatile  uint32_t value;
} apb_channel_status_s;

typedef union {
    volatile  struct {
      volatile uint32_t channel_tx_fail_cnt_fail_cnt:16;
      volatile uint32_t :16;

    } field;

    volatile  uint32_t value;
} apb_channel_tx_fail_cnt_s;

typedef union {
    volatile struct {
      volatile uint32_t channel_lock_fail_cnt_fail_cnt:16;
      volatile uint32_t :16;

    } field;

    volatile  uint32_t value;
} apb_channel_lock_fail_cnt_s;

typedef union {
   volatile struct {
        volatile uint32_t msi_tx_addr_tgt_base_addr:17;
        volatile uint32_t msi_tx_addr_tgt_to_self:1;
        volatile uint32_t :2;
        volatile uint32_t msi_tx_addr_tgt_channel:3;
	    volatile uint32_t :1;
        volatile uint32_t msi_tx_addr_int_tgt_core:6;
        volatile uint32_t :2;

    } field;

    volatile uint32_t value;
} apb_msi_tx_addr_s;

typedef struct {
    apb_msi_tx_addr_s addr;
    volatile  uint32_t payload[8];
    volatile  uint32_t unused[7];
} apb_msi_tx_info_s;

typedef struct {
    apb_channel_lock_s                   apb_channel_lock[8];
    volatile  uint32_t                   reserved_0[16];
    apb_channel_ctrl_s                   apb_channel_ctrl[64];
    apb_channel_status_s                 apb_channel_status[64];
    volatile uint32_t                    reserved_1[7];
    apb_msi_tx_info_s                    apb_msi_tx_info[64];

    apb_channel_tx_fail_cnt_s            apb_channel_tx_fail_cnt[8];
    apb_channel_lock_fail_cnt_s          apb_channel_lock_fail_cnt[8];

} mailbox_tx_cfg_s;
