/*
 * Copyright (C) 2018-2024 Alibaba Group Holding Limited
 */

/*
 * attention: don't modify this file as a suggest
 * you should copy from chip_riscv_dummy/include/asm/riscv_csr.h and keep it newer
 * please contact xuantie-rtos os team if have question 
 */

#ifndef __RISCV_CSR_H__
#define __RISCV_CSR_H__

#if __riscv_xlen == 64
	#define portWORD_SIZE 8
	#define store_x       sd
	#define load_x        ld
#elif __riscv_xlen == 32
	#define store_x       sw
	#define load_x        lw
	#define portWORD_SIZE 4
#else
	#error Assembler did not define __riscv_xlen
#endif

#if __riscv_flen == 64
	#define fstore_x      fsd
	#define fload_x       fld
#elif __riscv_flen == 32
	#define fstore_x      fsw
	#define fload_x       flw
#endif

#if defined(CONFIG_RISCV_SMODE) && CONFIG_RISCV_SMODE
#define MODE_PREFIX(suffix) s##suffix
#else
#define MODE_PREFIX(suffix) m##suffix
#endif

/* Status register flags */
#define SR_SIE      0x00000002UL        /*  Supervisor Interrupt Enable */
#define SR_MIE      0x00000008UL        /*  Machine Interrupt Enable */
#define SR_SPIE     0x00000020UL        /*  Previous Supervisor IE */
#define SR_MPIE     0x00000080UL        /*  Previous Machine IE */
#define SR_SPP_U    0x00000000UL        /*  Previously User mode */
#define SR_SPP_S    0x00000100UL        /*  Previously Supervisor mode */
#define SR_MPP_U    0x00000000UL        /*  Previously User mode */
#define SR_MPP_S    0x00000800UL        /*  Previously Supervisor mode */
#define SR_MPP_M    0x00001800UL        /*  Previously Machine mode */
#define SR_SUM      0x00040000UL        /*  Supervisor User Memory Access */

#define SR_FS           0x00006000UL    /*  Floating-point Status */
#define SR_FS_OFF       0x00000000UL
#define SR_FS_INITIAL   0x00002000UL
#define SR_FS_CLEAN     0x00004000UL
#define SR_FS_DIRTY     0x00006000UL

#if CONFIG_CPU_C906 || CONFIG_CPU_C906FD || CONFIG_CPU_C906FDV || CONFIG_CPU_C910 || CONFIG_CPU_C920
#define SR_VS          0x01800000
#define SR_VS_OFF      0x00000000
#define SR_VS_INITIAL  0x00800000
#define SR_VS_CLEAN    0x01000000
#define SR_VS_DIRTY    0x01800000
#else
#define SR_VS          0x00000600
#define SR_VS_OFF      0x00000000
#define SR_VS_INITIAL  0x00000200
#define SR_VS_CLEAN    0x00000400
#define SR_VS_DIRTY    0x00000600
#endif

#if __riscv_matrix
#define SR_MS          0x06000000
#define SR_MS_OFF      0x00000000
#define SR_MS_INITIAL  0x02000000
#define SR_MS_CLEAN    0x04000000
#define SR_MS_DIRTY    0x06000000
#endif

/* Interrupt-enable Registers */
#define IE_MTIE         0x00000080UL
#define IE_MEIE         0x00000800UL

/*  ===== Trap/Exception Causes ===== */
#define CAUSE_MISALIGNED_FETCH          0x0
#define CAUSE_FETCH_ACCESS              0x1
#define CAUSE_ILLEGAL_INSTRUCTION       0x2
#define CAUSE_BREAKPOINT                0x3
#define CAUSE_MISALIGNED_LOAD           0x4
#define CAUSE_LOAD_ACCESS               0x5
#define CAUSE_MISALIGNED_STORE          0x6
#define CAUSE_STORE_ACCESS              0x7
#define CAUSE_USER_ECALL                0x8
#define CAUSE_SUPERVISOR_ECALL          0x9
#define CAUSE_VIRTUAL_SUPERVISOR_ECALL  0xa
#define CAUSE_MACHINE_ECALL             0xb
#define CAUSE_FETCH_PAGE_FAULT          0xc
#define CAUSE_LOAD_PAGE_FAULT           0xd
#define CAUSE_STORE_PAGE_FAULT          0xf

#define PRV_U                           0
#define PRV_S                           1
#define PRV_M                           3


#define MSTATUS_SIE                     0x00000002
#define MSTATUS_MIE                     0x00000008
#define MSTATUS_SPIE_SHIFT              5
#define MSTATUS_SPIE                    (1 << MSTATUS_SPIE_SHIFT)
#define MSTATUS_UBE                     0x00000040
#define MSTATUS_MPIE                    0x00000080
#define MSTATUS_SPP_SHIFT               8
#define MSTATUS_SPP                     (1 << MSTATUS_SPP_SHIFT)
#define MSTATUS_MPP_SHIFT               11
#define MSTATUS_MPP                     (3 << MSTATUS_MPP_SHIFT)

#if CONFIG_CPU_C906 || CONFIG_CPU_C906FD || CONFIG_CPU_C906FDV || CONFIG_CPU_C910 || CONFIG_CPU_C920
#define MSTATUS_VS_SHIFT                23
#else
#define MSTATUS_VS_SHIFT                9
#endif
#define MSTATUS_FS_SHIFT                13
#define MSTATUS_MS_SHIFT                25

#define INSERT_FIELD(val, which, fieldval)	(((val) & ~(which)) | ((fieldval) * ((which) & ~((which)-1))))

#if CONFIG_CPU_C906 || CONFIG_CPU_C906FD || CONFIG_CPU_C906FDV || CONFIG_CPU_C910 || CONFIG_CPU_C920 || CONFIG_CPU_C908 || CONFIG_CPU_C908V ||CONFIG_CPU_C908I
#define ATTR_SO                 (1ull << 4)
#define ATTR_CA                 (1ull << 3)
#define ATTR_BU                 (1ull << 2)
#define ATTR_SH                 (1ull << 1)
#define ATTR_SE                 (1ull << 0)

#define UPPER_ATTRS_SHIFT       (59)
#define UPPER_ATTRS(x)          (((x) & 0x1f) << UPPER_ATTRS_SHIFT)
#else
#if __riscv_xlen == 32
#define PTE_PBMT_SHIFT          (30)
#else
#define PTE_PBMT_SHIFT          (61)
#endif /* end __riscv_xlen */
#define SVPBMT_PMA              ((unsigned long)0x0 << PTE_PBMT_SHIFT)
#define SVPBMT_NC               ((unsigned long)0x1 << PTE_PBMT_SHIFT)
#define SVPBMT_IO               ((unsigned long)0x2 << PTE_PBMT_SHIFT)
#define SVPBMT_MASK             ((unsigned long)0x3 << PTE_PBMT_SHIFT)

#endif

#define DIRTY_FLAG              (1 << 6)
#define ACCESS_FLAG             (1 << 5)
#define GLOBAL_FLAG             (1 << 4)
#define AP_UNPRIV               (1 << 3)
#define AP_X                    (1 << 2)
#define AP_W                    (1 << 1)
#define AP_R                    (1 << 0)

#define LOWER_ATTRS_SHIFT               1
#define LOWER_ATTRS(x)                  (((x) & 0x1ff) << LOWER_ATTRS_SHIFT)


#define CSR_MCOR         	0x7c2
#define CSR_MHCR         	0x7c1
#define CSR_MCCR2        	0x7c3
#define CSR_MHINT        	0x7c5
#define CSR_MHINT2       	0x7cc
#define CSR_MHINT3       	0x7cd
#define CSR_MHINT4       	0x7ce
#define CSR_MXSTATUS     	0x7c0
#define CSR_PLIC_BASE    	0xfc1
#define CSR_MRMR         	0x7c6
#define CSR_MRVBR        	0x7c7
#define CSR_MCOUNTERWEN  	0x7c9
#define CSR_MSMPR	 		0x7f3

#define CSR_MSTATUS			0x300
#define CSR_MISA			0x301
#define CSR_MEDELEG			0x302
#define CSR_MIDELEG			0x303
#define CSR_MIE				0x304
#define CSR_MTVEC			0x305
#define CSR_MCOUNTEREN		0x306
#define CSR_MENVCFG			0x30a
#define CSR_MSTATUSH		0x310
#define CSR_MSCRATCH		0x340
#define CSR_MEPC			0x341
#define CSR_MCAUSE			0x342
#define CSR_MTVAL			0x343
#define CSR_MIP				0x344
#define CSR_MTINST			0x34a
#define CSR_MTVAL2			0x34b

/* Machine Memory Protection */
#define CSR_PMPCFG0   		0x3a0

#define CSR_PMPCFG2   		0x3a2

#define CSR_PMPCFG4   		0x3a4

#define CSR_PMPCFG6   		0x3a6

#define CSR_PMPCFG8   		0x3a8

#define CSR_PMPCFG10  		0x3aa

#define CSR_PMPCFG12  		0x3ac

#define CSR_PMPCFG14  		0x3ae


#define CSR_PMPADDR0  0x3B0
#define CSR_PMPADDR1  0x3B1
#define CSR_PMPADDR2  0x3B2
#define CSR_PMPADDR3  0x3B3
#define CSR_PMPADDR4  0x3B4
#define CSR_PMPADDR5  0x3B5
#define CSR_PMPADDR6  0x3B6
#define CSR_PMPADDR7  0x3B7
#define CSR_PMPADDR8  0x3B8
#define CSR_PMPADDR9  0x3B9
#define CSR_PMPADDR10 0x3BA
#define CSR_PMPADDR11 0x3BB
#define CSR_PMPADDR12 0x3BC
#define CSR_PMPADDR13 0x3BD
#define CSR_PMPADDR14 0x3BE
#define CSR_PMPADDR15 0x3BF
#define CSR_PMPADDR16 0x3C0
#define CSR_PMPADDR17 0x3C1
#define CSR_PMPADDR18 0x3C2
#define CSR_PMPADDR19 0x3C3
#define CSR_PMPADDR20 0x3C4
#define CSR_PMPADDR21 0x3C5
#define CSR_PMPADDR22 0x3C6
#define CSR_PMPADDR23 0x3C7
#define CSR_PMPADDR24 0x3C8
#define CSR_PMPADDR25 0x3C9
#define CSR_PMPADDR26 0x3CA
#define CSR_PMPADDR27 0x3CB
#define CSR_PMPADDR28 0x3CC
#define CSR_PMPADDR29 0x3CD
#define CSR_PMPADDR30 0x3CE
#define CSR_PMPADDR31 0x3CF
#define CSR_PMPADDR32 0x3D0
#define CSR_PMPADDR33 0x3D1
#define CSR_PMPADDR34 0x3D2
#define CSR_PMPADDR35 0x3D3
#define CSR_PMPADDR36 0x3D4
#define CSR_PMPADDR37 0x3D5
#define CSR_PMPADDR38 0x3D6
#define CSR_PMPADDR39 0x3D7
#define CSR_PMPADDR40 0x3D8
#define CSR_PMPADDR41 0x3D9
#define CSR_PMPADDR42 0x3DA
#define CSR_PMPADDR43 0x3DB
#define CSR_PMPADDR44 0x3DC
#define CSR_PMPADDR45 0x3DD
#define CSR_PMPADDR46 0x3DE
#define CSR_PMPADDR47 0x3DF
#define CSR_PMPADDR48 0x3E0
#define CSR_PMPADDR49 0x3E1
#define CSR_PMPADDR50 0x3E2
#define CSR_PMPADDR51 0x3E3
#define CSR_PMPADDR52 0x3E4
#define CSR_PMPADDR53 0x3E5
#define CSR_PMPADDR54 0x3E6
#define CSR_PMPADDR55 0x3E7
#define CSR_PMPADDR56 0x3E8
#define CSR_PMPADDR57 0x3E9
#define CSR_PMPADDR58 0x3EA
#define CSR_PMPADDR59 0x3EB
#define CSR_PMPADDR60 0x3EC
#define CSR_PMPADDR61 0x3ED
#define CSR_PMPADDR62 0x3EE
#define CSR_PMPADDR63 0x3EF


#define CSR_MARCHID			0xf12
#define CSR_MIMPID			0xf13
#define CSR_MHARTID			0xf14
#define CSR_MCPUID			0xfc0
#define CSR_CYCLEH			0xc80
#define CSR_TIMEH			0xc81

#endif /* __RISCV_CSR_H__ */

