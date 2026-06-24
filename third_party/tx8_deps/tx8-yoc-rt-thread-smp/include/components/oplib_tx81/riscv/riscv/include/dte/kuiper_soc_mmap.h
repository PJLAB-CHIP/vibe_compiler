/*
 * Kuiper Spatial Software
 * Copyright (c) 2022, tsingmicro. All rights reserved.
 *
 */

#ifndef KUIPER_SOC_MMAP_H
#define KUIPER_SOC_MMAP_H

//memory layout in soc
#define KUIPER_L1SPM_BASE                       0x000000
#define KUIPER_DTE_BASE                         0x400000
#define KUIPER_SCONFIG_BASE                     0x500000
#define KUIPER_TILE_WRAP_CRG_BASE               0x600000
#define KUIPER_STREAMINT_REG_BASE               0x610000
#define KUIPER_STREAMFSM_REG_BASE               0x620000
#define KUIPER_MAILBOX_TX_REG_BASE              0x640000
#define KUIPER_MAILBOX_RX_REG_BASE              0x660000
#define KUIPER_MHU_TILE_AP_TX                   0x680000
#define KUIPER_MHU_TILE_AP_RX                   0x681000
#define KUIPER_MHU_AP_TILE_TX                   0x690000
#define KUIPER_MHU_AP_TILE_RX                   0x691000
#define KUIPER_ADDR_MAP_REG_BASE                0x6A0000
#define KUIPER_BROAD_CTRL_BASE                  0x6C0000
#define KUIPER_TILE_MISC_BASE                   0x6E0000

#define KUIPER_DDR_BASE                         0x80000000
#define KUIPER_DDR_UNCACHE_WEAKORDER_BASE       0x180000000
#define KUIPER_DDR_UNCACHE_WEAKORDER_END        0x27FFFFFFF
#define KUIPER_EXTERN_DDR_BASE                  0x8000000000

#define KUIPER_L1SPM_UNCACHE_WEAKORDER_BASE     0x30400000  /*uncacheable weak order L1SPM memory*/
#define KUIPER_L1SPM_UNCACHE_STRONGORDER_BASE   0x30800000  /*uncacheable strong order L1SPM memory*/
#define KUIPER_L1SPM_SIZE                       0x300000    /* 3M bytes */

//The register which need to bu update after DTE transfer done.
#define KUIPER_PACKET_CNT_UPDATE_REG            0x670000

#endif /* KUIPER_SOC_MMAP_H */
