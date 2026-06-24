#ifndef INSTR_ADAPTER_OPT_H
#define INSTR_ADAPTER_OPT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "instr_def.h"
#include "instr_adapter_plat.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifndef USING_RISCV
#define __CHECK_INSTR__
#endif
uint32_t __execute_ct_argmaxmin(TsmArithInstr *instr);
uint32_t __ct_init_argmaxmin(TsmArithInstr *instr);
uint64_t __ct__ctrl_reg_value(TsmArithInstr *instr);
uint32_t __ct_execute_argmaxmin(TsmArithInstr *instr, uint64_t ctrl);
uint32_t __ct_get_argmaxmin_result(TsmArithInstr *instr);

#ifdef __cplusplus
}
#endif

#endif /*INSTR_ADAPTER_H*/
