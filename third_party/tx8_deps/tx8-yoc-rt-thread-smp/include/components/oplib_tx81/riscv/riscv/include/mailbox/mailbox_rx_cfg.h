#include <stdint.h>

typedef union {
    volatile struct {
        volatile uint32_t core_doorbell_int_core_doorbell_int:6;
        volatile uint32_t :26;

    } field;

   volatile uint32_t value;
} apb_core_doorbell_int_s;

typedef union {
   volatile struct {
        volatile uint32_t mb_doorbell_int_mb_doorbell_int:8;
        volatile uint32_t :24;

    } field;

   volatile  uint32_t value;
} apb_mb_doorbell_int_s;

typedef union {
    volatile struct {
        volatile uint32_t mb_doorbell_status_mb_doorbell_status:8;
        volatile uint32_t :24;

    } field;

  volatile  uint32_t value;
} apb_mb_doorbell_status_s;

typedef union {
   volatile struct {
        volatile uint32_t int_map_int_map_bmp:6;
        volatile uint32_t :26;

    } field;

   volatile  uint32_t value;
} apb_int_map_s;

typedef union {
  volatile  struct {
        volatile uint32_t msi_rx_cnt_pending_req_cnt:6;
        volatile uint32_t :6;
        volatile uint32_t msi_rx_cnt_read_done:1;
        volatile uint32_t :3;
        volatile uint32_t msi_rx_cnt_rx_fail_cnt:16;

    } field;

  volatile  uint32_t value;
} apb_msi_rx_cnt_s;

typedef struct {
    apb_msi_rx_cnt_s rx_cnt;
    volatile uint32_t payload[8];
    volatile  uint32_t unused[7];
} aapb_msi_rx_info_s;

typedef struct {
    apb_core_doorbell_int_s              apb_core_doorbell_int;
    apb_mb_doorbell_int_s                apb_mb_doorbell_int;
    apb_mb_doorbell_status_s             apb_mb_doorbell_status[6];
    volatile uint32_t                    reserved_0[8];
    apb_int_map_s                        apb_int_map[8];
    volatile uint32_t                    reserved_1[8];
    aapb_msi_rx_info_s                   rx_info[8];
} mailbox_rx_cfg_s;
