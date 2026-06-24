#ifndef INSTR_OPERATOR_H
#define INSTR_OPERATOR_H

#include "instr_def.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct TsmOperatorPointer{
    void *conv_pointer;
    void *depthwiseConv_pointer;
    void *gemm_pointer;
    void *rdma_pointer;
    void *wdma_pointer;
    void *arith_pointer;
    void *relation_pointer;
    void *logic_pointer;
    void *transcendental_pointer;
    void *activation_pointer;
    void *reduce_pointer;
    void *pool_pointer;
    void *unpool_pointer;
    void *maskdatamove_pointer;
    void *convert_pointer;
    void *peripheral_pointer;
    void *datamove_pointer;
    void *stream_pointer;
} TsmOperatorPointer;
//have in common
TsmOperatorPointer* g_intrinsic();

//cmodel version
TsmOperatorPointer* initTsmOpPointer_cmodel();
void freeTsmOpPointer_cmodel(TsmOperatorPointer *p_tsmOphandle);

//riscv version
void initTsmOpPointer();
void freeTsmOpPointer();

//have in common
TsmOperatorPointer* getTsmOpPointer();

void debug_td_info(TD_Param *param);
void debug_ct_info(TsmArithInstr *instr);
#ifdef __cplusplus
}
#endif

#endif
