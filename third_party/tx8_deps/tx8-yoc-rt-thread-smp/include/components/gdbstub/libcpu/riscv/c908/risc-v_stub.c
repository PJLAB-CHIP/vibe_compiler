/*
 * risc-v GDB support
 * arch-specific portion of GDB stub
 * 
 * File      : arch_gdb.h(risc-v)
 * This file is part of RT-Thread RTOS
 * COPYRIGHT (C) 2006, RT-Thread Develop Team
 *
 * The license and distribution terms for this file may be
 * found in the file LICENSE in this distribution or at
 * http://www.rt-thread.org/license/LICENSE
 *
 * Change Logs:
 * Date           Author       Notes
 * 2024-12-11     tangjb      first version
 */
#include <rtthread.h>
#include <rthw.h>
#include <gdb_stub.h>
#include <insn.h>
#include <string.h>
#include <lib_log.h>

static int compiled_break = 0;
static unsigned long step_addr = 0;
static unsigned long target_ins(unsigned long pc, unsigned long ins);

#ifdef CONFIG_RISCV_ISA_C
struct gdb_arch arch_gdb_ops = {
	.gdb_bpt_instr   =  {0x02, 0x90},	/* c.ebreak */
};
#else
/*struct gdb_arch - Describe architecture specific values.*/
struct gdb_arch arch_gdb_ops = {
	.gdb_bpt_instr  =  {0x73, 0x00, 0x10, 0x00}  //Little-Endian
};
#endif

struct rt_gdb_register
{
    unsigned long ra;         /* x1  - ra     - return address for jumps            1*/
    unsigned long sp;         /* x2  - sp     - stack pointer                       2*/
    unsigned long gp;         /* x3  - gp     - global pointer                      3*/
    unsigned long tp;         /* x4  - tp     - thread pointer                      4*/
    unsigned long t0;         /* x5  - t0     - temporary register 0                5*/
    unsigned long t1;         /* x6  - t1     - temporary register 1                6*/
    unsigned long t2;         /* x7  - t2     - temporary register 2                7*/
    unsigned long s0;         /* x8  - s0/fp  - saved register 0 or frame pointer   8*/
    unsigned long s1;         /* x9  - s1     - saved register 1                    9*/
    unsigned long a0;         /* x10 - a0     - return value or function argument 0 10*/
    unsigned long a1;         /* x11 - a1     - return value or function argument 1 11*/
    unsigned long a2;         /* x12 - a2     - function argument 2                12*/
    unsigned long a3;         /* x13 - a3     - function argument 3                13*/
    unsigned long a4;         /* x14 - a4     - function argument 4                14*/
    unsigned long a5;         /* x15 - a5     - function argument 5                15*/
    unsigned long a6;         /* x16 - a6     - function argument 6                16*/
    unsigned long a7;         /* x17 - s7     - function argument 7                17*/
    unsigned long s2;         /* x18 - s2     - saved register 2                   18*/
    unsigned long s3;         /* x19 - s3     - saved register 3                   19*/
    unsigned long s4;         /* x20 - s4     - saved register 4                   20*/
    unsigned long s5;         /* x21 - s5     - saved register 5                   21*/
    unsigned long s6;         /* x22 - s6     - saved register 6                   22*/
    unsigned long s7;         /* x23 - s7     - saved register 7                   23*/
    unsigned long s8;         /* x24 - s8     - saved register 8                   24*/
    unsigned long s9;         /* x25 - s9     - saved register 9                   25*/
    unsigned long s10;        /* x26 - s10    - saved register 10                  26*/
    unsigned long s11;        /* x27 - s11    - saved register 11                  27*/
    unsigned long t3;         /* x28 - t3     - temporary register 3               28*/
    unsigned long t4;         /* x29 - t4     - temporary register 4               29*/
    unsigned long t5;         /* x30 - t5     - temporary register 5               30*/
    unsigned long t6;         /* x31 - t6     - temporary register 6               31*/
    unsigned long epc;        /* epc - epc    - program counter                    32*/
    unsigned long status;     /*              - supervisor status register         33*/
};

static struct rt_gdb_register *regs;

//linux struct
struct pt_regs {
	unsigned long epc;
	unsigned long ra;
	unsigned long sp;
	unsigned long gp;
	unsigned long tp;
	unsigned long t0;
	unsigned long t1;
	unsigned long t2;
	unsigned long s0;
	unsigned long s1;
	unsigned long a0;
	unsigned long a1;
	unsigned long a2;
	unsigned long a3;
	unsigned long a4;
	unsigned long a5;
	unsigned long a6;
	unsigned long a7;
	unsigned long s2;
	unsigned long s3;
	unsigned long s4;
	unsigned long s5;
	unsigned long s6;
	unsigned long s7;
	unsigned long s8;
	unsigned long s9;
	unsigned long s10;
	unsigned long s11;
	unsigned long t3;
	unsigned long t4;
	unsigned long t5;
	unsigned long t6;
	/* Supervisor/Machine CSRs */
	unsigned long status;
	unsigned long badaddr;
	unsigned long cause;
	/* a0 value before the syscall */
	unsigned long orig_a0;
};

extern unsigned long kgdb_compiled_break;

/**
 * gdb_breakpoint - generate a compiled_breadk
 * It is used to sync up with a debugger and stop progarm
 */
void gdb_breakpoint()
{
//     rt_kprintf("gdb_breakpoint\n");
    asm(".global kgdb_compiled_break\n"
	    ".option norvc\n"
	    "kgdb_compiled_break: ebreak\n"
	    ".option rvc\n");
}

struct dbg_reg_def_t dbg_reg_def[DBG_MAX_REG_NUM] = {
    {DBG_REG_ZERO, GDB_SIZEOF_REG, -1},
	{DBG_REG_RA, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, ra)},
	{DBG_REG_SP, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, sp)},
	{DBG_REG_GP, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, gp)},
	{DBG_REG_TP, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, tp)},
	{DBG_REG_T0, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t0)},
	{DBG_REG_T1, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t1)},
	{DBG_REG_T2, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t2)},
	{DBG_REG_FP, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s0)},
	{DBG_REG_S1, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a1)},
	{DBG_REG_A0, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a0)},
	{DBG_REG_A1, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a1)},
	{DBG_REG_A2, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a2)},
	{DBG_REG_A3, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a3)},
	{DBG_REG_A4, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a4)},
	{DBG_REG_A5, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a5)},
	{DBG_REG_A6, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a6)},
	{DBG_REG_A7, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, a7)},
	{DBG_REG_S2, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s2)},
	{DBG_REG_S3, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s3)},
	{DBG_REG_S4, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s4)},
	{DBG_REG_S5, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s5)},
	{DBG_REG_S6, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s6)},
	{DBG_REG_S7, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s7)},
	{DBG_REG_S8, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s8)},
	{DBG_REG_S9, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s9)},
	{DBG_REG_S10, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s10)},
	{DBG_REG_S11, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, s11)},
	{DBG_REG_T3, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t3)},
	{DBG_REG_T4, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t4)},
	{DBG_REG_T5, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t5)},
	{DBG_REG_T6, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, t6)},
	{DBG_REG_EPC, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, epc)},
	//{DBG_REG_STATUS, GDB_SIZEOF_REG, offsetof(struct rt_gdb_register, status)},
};


void gdb_set_register(void *hw_regs)
{
    regs = (struct rt_gdb_register *)hw_regs;
}


void gdb_get_register(unsigned long *gdb_regs)
{
    int regno;
    for (regno = 0; regno < DBG_MAX_REG_NUM; regno++)
    {
        if(dbg_reg_def[regno].offset != -1)
            memcpy(gdb_regs + regno, (void *)regs + dbg_reg_def[regno].offset,
                    dbg_reg_def[regno].size);
        else
            memset(gdb_regs + regno, 0,
                    dbg_reg_def[regno].size);
    }
}


void gdb_get_regbythrd(unsigned long *gdb_regs, void* sp)
{
	memset(gdb_regs, 0, CTX_GENERAL_REG_NR * REGBYTES);

#if defined(__riscv_vector) && defined(ENABLE_VECTOR)
	sp += CTX_VECTOR_CSR_REG_NR * REGBYTES;
#endif

#if defined(__riscv_flen) && defined(ENABLE_FPU)
	sp += (CTX_FPU_CSR_REG_NR + CTX_FPU_REG_NR) * REGBYTES;
#endif

	struct rt_gdb_register* gdbregs = (struct rt_gdb_register *)sp;

	gdb_regs[DBG_REG_SP_OFF]  = gdbregs->sp;
	gdb_regs[DBG_REG_FP_OFF]  = gdbregs->s0;
	gdb_regs[DBG_REG_S1_OFF]  = gdbregs->s1;
	gdb_regs[DBG_REG_S2_OFF]  = gdbregs->s2;
	gdb_regs[DBG_REG_S3_OFF]  = gdbregs->s3;
	gdb_regs[DBG_REG_S4_OFF]  = gdbregs->s4;
	gdb_regs[DBG_REG_S5_OFF]  = gdbregs->s5;
	gdb_regs[DBG_REG_S6_OFF]  = gdbregs->s6;
	gdb_regs[DBG_REG_S7_OFF]  = gdbregs->s7;
	gdb_regs[DBG_REG_S8_OFF]  = gdbregs->s8;
	gdb_regs[DBG_REG_S9_OFF]  = gdbregs->s10;
	gdb_regs[DBG_REG_S10_OFF] = gdbregs->s11;
	gdb_regs[DBG_REG_EPC_OFF] = gdbregs->ra;

}


void gdb_put_register(unsigned long *gdb_regs)
{
    int regno;
    for (regno = 0; regno < DBG_MAX_REG_NUM; regno++)
    {
        if(dbg_reg_def[regno].offset != -1)
            memcpy((void *)regs + dbg_reg_def[regno].offset, gdb_regs + regno, 
                    dbg_reg_def[regno].size);
    }
}


/* It will be called during process_packet */
int gdb_arch_handle_exception(char *remcom_in_buffer,
                              char *remcom_out_buffer)
{
    unsigned long addr,curins;
    char *ptr;

    /*clear single step*/
    if (step_addr) {
        gdb_remove_sw_break(step_addr);
        step_addr = 0;
    }

    switch (remcom_in_buffer[0]) {
        case 'D':
        case 'k':
        case 'c':
            /*
             * If this was a compiled breakpoint, we need to move
             * to the next instruction or we will breakpoint
             * over and over again
             */
            ptr = &remcom_in_buffer[1];
            if (gdb_hex2long(&ptr, &addr))
                regs->epc = addr;
            else if (compiled_break == 1)
                regs->epc += 4;
            compiled_break = 0;
            return 0;

        case 's':
            ptr = &remcom_in_buffer[1];
            if (gdb_hex2long(&ptr, &addr))
                regs->epc = addr; 
            
            curins = *(unsigned long*)(regs->epc);
            //Decode instruction to decide what the next pc will be 
            step_addr = target_ins(regs->epc, curins);

#ifdef RT_GDB_DEBUG
            rt_kprintf("\n curret addr %x, next will be %x \n", regs->epc, step_addr);
#endif
            gdb_set_sw_break(step_addr); 

            if (compiled_break == 1)
                regs->epc += 4;         
            compiled_break = 0;
            return 0;
    }

	return -1;

}

/* flush icache to let the sw breakpoint working */
void gdb_flush_icache_range(unsigned long start, unsigned long end)
{
    rt_hw_cpu_icache_ops(RT_HW_CACHE_INVALIDATE, (void*)start, end - start);
}

/* register a hook in undef*/
int gdb_undef_hook(void *regs)
{
    struct rt_gdb_register *gdb_reg = (struct rt_gdb_register *)regs;

    /* it is a compiled break */ 
#ifdef BREAK_INSTR_SIZE

	if (*(uint16_t*)gdb_reg->epc != *(uint16_t*)arch_gdb_ops.gdb_bpt_instr) {
#else
    if (gdb_reg->epc == (unsigned long)&kgdb_compiled_break) {
#endif
        compiled_break = 1;
        gdb_handle_exception(SIGTRAP, regs);
        return 1;
    }
    /* it is a sw break */ 
    else if (gdb_isswbreak(gdb_reg->epc)) {
        gdb_handle_exception(SIGTRAP, regs);
        return 1;
    }
    /*or we just go */
    return 0;

}

static int decode_register_index(unsigned long opcode, int offset)
{
	return (opcode >> offset) & 0x1F;
}

static int decode_register_index_short(unsigned long opcode, int offset)
{
	return ((opcode >> offset) & 0x7) + 8;
}

static void ajust_gdb_registers(struct pt_regs* pst_regs)
{
    pst_regs->epc      =        regs->epc;
    pst_regs->ra       =        regs->ra;
    pst_regs->sp       =        regs->sp;
    pst_regs->gp       =        regs->gp;
    pst_regs->tp       =        regs->tp;
    pst_regs->t0       =        regs->t0;
    pst_regs->t1       =        regs->t1;
    pst_regs->t2       =        regs->t2;
    pst_regs->s0       =        regs->s0;
    pst_regs->s1       =        regs->s1;
    pst_regs->a0       =        regs->a0;
    pst_regs->a1       =        regs->a1;
    pst_regs->a2       =        regs->a2;
    pst_regs->a3       =        regs->a3;
    pst_regs->a4       =        regs->a4;
    pst_regs->a5       =        regs->a5;
    pst_regs->a6       =        regs->a6;
    pst_regs->a7       =        regs->a7;
    pst_regs->s2       =        regs->s2;
    pst_regs->s3       =        regs->s3;
    pst_regs->s4       =        regs->s4;
    pst_regs->s5       =        regs->s5;
    pst_regs->s6       =        regs->s6;
    pst_regs->s7       =        regs->s7;
    pst_regs->s8       =        regs->s8;
    pst_regs->s9       =        regs->s9;
    pst_regs->s10      =        regs->s10;
    pst_regs->s11      =        regs->s11;
    pst_regs->t3       =        regs->t3;
    pst_regs->t4       =        regs->t4;
    pst_regs->t5       =        regs->t5;
    pst_regs->t6       =        regs->t6;
    pst_regs->status   =        regs->status;
    pst_regs->badaddr  =        0;
    pst_regs->cause    =        3;
    pst_regs->orig_a0  =        0;
}


// Decide the next instruction to be executed for a given instruction
static unsigned long target_ins(unsigned long pc, unsigned long ins)
{
    const int op_code = ins;
	unsigned int rs1_num, rs2_num;

    unsigned long new_pc = 0;
    struct pt_regs st_regs;

    ajust_gdb_registers(&st_regs);
    

    unsigned long *regs_ptr = (unsigned long *)&st_regs;

	if ((op_code & __INSN_LENGTH_MASK) != __INSN_LENGTH_GE_32) {
		if (riscv_insn_is_c_jalr(op_code) ||
		    riscv_insn_is_c_jr(op_code)) {
			rs1_num = decode_register_index(op_code, RVC_C2_RS1_OPOFF);
			new_pc = regs_ptr[rs1_num];
		} else if (riscv_insn_is_c_j(op_code) ||
			   riscv_insn_is_c_jal(op_code)) {
			new_pc = RVC_EXTRACT_JTYPE_IMM(op_code) + pc;
		} else if (riscv_insn_is_c_beqz(op_code)) {
			rs1_num = decode_register_index_short(op_code,
							      RVC_C1_RS1_OPOFF);
			if (!rs1_num || regs_ptr[rs1_num] == 0)
				new_pc = RVC_EXTRACT_BTYPE_IMM(op_code) + pc;
			else
				new_pc = pc + 2;
		} else if (riscv_insn_is_c_bnez(op_code)) {
			rs1_num =
			    decode_register_index_short(op_code, RVC_C1_RS1_OPOFF);
			if (rs1_num && regs_ptr[rs1_num] != 0)
				new_pc = RVC_EXTRACT_BTYPE_IMM(op_code) + pc;
			else
				new_pc = pc + 2;
		} else {
			new_pc = pc + 2;
		}
	} else {
		if ((op_code & __INSN_OPCODE_MASK) == __INSN_BRANCH_OPCODE) {
			bool result = false;
			long imm = RV_EXTRACT_BTYPE_IMM(op_code);
			unsigned long rs1_val = 0, rs2_val = 0;

			rs1_num = decode_register_index(op_code, RVG_RS1_OPOFF);
			rs2_num = decode_register_index(op_code, RVG_RS2_OPOFF);
			if (rs1_num)
				rs1_val = regs_ptr[rs1_num];
			if (rs2_num)
				rs2_val = regs_ptr[rs2_num];

			if (riscv_insn_is_beq(op_code))
				result = (rs1_val == rs2_val) ? true : false;
			else if (riscv_insn_is_bne(op_code))
				result = (rs1_val != rs2_val) ? true : false;
			else if (riscv_insn_is_blt(op_code))
				result =
				    ((long)rs1_val <
				     (long)rs2_val) ? true : false;
			else if (riscv_insn_is_bge(op_code))
				result =
				    ((long)rs1_val >=
				     (long)rs2_val) ? true : false;
			else if (riscv_insn_is_bltu(op_code))
				result = (rs1_val < rs2_val) ? true : false;
			else if (riscv_insn_is_bgeu(op_code))
				result = (rs1_val >= rs2_val) ? true : false;
			if (result)
				new_pc = imm + pc;
			else
				new_pc = pc + 4;
		} else if (riscv_insn_is_jal(op_code)) {
			new_pc = RV_EXTRACT_JTYPE_IMM(op_code) + pc;
		} else if (riscv_insn_is_jalr(op_code)) {
			rs1_num = decode_register_index(op_code, RVG_RS1_OPOFF);
			if (rs1_num)
				new_pc = regs_ptr[rs1_num];
			new_pc += RV_EXTRACT_ITYPE_IMM(op_code);
		} else if (riscv_insn_is_sret(op_code)) {
			new_pc = pc;
		} else {
			new_pc = pc + 4;
		}
	}
	return new_pc;
}
