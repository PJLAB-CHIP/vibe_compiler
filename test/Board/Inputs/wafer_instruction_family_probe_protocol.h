#ifndef WAFER_INSTRUCTION_FAMILY_PROBE_PROTOCOL_H
#define WAFER_INSTRUCTION_FAMILY_PROBE_PROTOCOL_H

#include <stdint.h>

/*
 * Bounded wire contract for compiler-emitted instruction qualification.
 *
 * The host selects one typed catalog row.  Geometry and observable write
 * ranges are part of that row and are mirrored by the device before any
 * target call is issued.  Rows marked DEFERRED are discoverable but can never
 * reach device dispatch.
 */

#define WAFER_IFP_REQUEST_MAGIC UINT64_C(0x3151455246494657)
#define WAFER_IFP_RECORD_MAGIC UINT64_C(0x3143455246494657)
#define WAFER_IFP_SCHEMA 1U
#define WAFER_IFP_REQUEST_WORDS 16U
#define WAFER_IFP_RECORD_WORDS 32U
#define WAFER_IFP_RESOURCE_BYTES 16384U
#define WAFER_IFP_SLOT_BYTES 4096U
#define WAFER_IFP_BODY_OFFSET 256U
#define WAFER_IFP_OUTPUT_DDR_OFFSET 4096U
#define WAFER_IFP_SPM_A UINT64_C(0x10000)
#define WAFER_IFP_SPM_B UINT64_C(0x20000)
#define WAFER_IFP_SPM_OUTPUT UINT64_C(0x30000)
#define WAFER_IFP_SPM_AUX UINT64_C(0x40000)
#define WAFER_IFP_SLOT_CANARY UINT8_C(0xa7)
#define WAFER_IFP_BIT2FP_TRUE_F16 UINT16_C(0x3c00)
#define WAFER_IFP_BIT2FP_TRUE_BF16 UINT16_C(0x3f80)
#define WAFER_IFP_BIT2FP_FALSE UINT16_C(0x0000)
#define WAFER_IFP_REQUEST_GUARD UINT64_C(0x8e7c6a5948372615)
#define WAFER_IFP_RECORD_GUARD UINT64_C(0x192a3b4c5d6e7f80)

enum WaferIFPDisposition {
  WAFER_IFP_SAFE = 0,
  WAFER_IFP_DEFERRED = 1,
};

enum WaferIFPFamily {
  WAFER_IFP_CT_ELEMENTWISE = 0,
  WAFER_IFP_CT_CONVERT = 1,
  WAFER_IFP_CT_REDUCE = 2,
  WAFER_IFP_CT_SELECT_COMPOSITE = 3,
  WAFER_IFP_NE_GEMM = 4,
  WAFER_IFP_TDMA_PAD = 5,
  WAFER_IFP_TDMA_IMG2COL = 6,
  WAFER_IFP_CONV = 7,
  WAFER_IFP_POOL = 8,
  WAFER_IFP_UNPOOL = 9,
  WAFER_IFP_PERIPHERAL = 10,
};

enum WaferIFPDType {
  WAFER_IFP_F16 = 0,
  WAFER_IFP_BF16 = 1,
  WAFER_IFP_I8_TO_F16 = 2,
  WAFER_IFP_I8_TO_BF16 = 3,
  WAFER_IFP_BF16_TO_F16 = 4,
  WAFER_IFP_F16_TO_BF16 = 5,
  WAFER_IFP_F16_TO_I16 = 6,
};

enum WaferIFPOracle {
  WAFER_IFP_EXACT_BITS = 0,
  WAFER_IFP_EXACT_COMPOSITE = 1,
  WAFER_IFP_NO_ORACLE = 2,
};

enum WaferIFPDeferredReason {
  WAFER_IFP_REASON_NONE = 0,
  WAFER_IFP_REASON_ISOLATED_COMPLETION_WRITEBACK_UNQUALIFIED = 1,
  WAFER_IFP_REASON_GEOMETRY_UNQUALIFIED = 2,
  WAFER_IFP_REASON_NUMERIC_UNQUALIFIED = 3,
};

/*
 * SYMBOL, id, spelling, disposition, family, dtype, oracle, reason,
 * logical result bytes, allowed output write span, allowed auxiliary span.
 */
#define WAFER_IFP_CASES(X)                                                     \
  X(CT_NEG_F16, 1, "ct-neg-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,       \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_NEG_BF16, 2, "ct-neg-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_ADD_F16, 3, "ct-add-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,       \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_ADD_BF16, 4, "ct-add-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_SUB_F16, 5, "ct-sub-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,       \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_SUB_BF16, 6, "ct-sub-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MUL_F16, 7, "ct-mul-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,       \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MUL_BF16, 8, "ct-mul-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MAX_F16, 9, "ct-max-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,       \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MAX_BF16, 10, "ct-max-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,   \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MIN_F16, 11, "ct-min-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,      \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_MIN_BF16, 12, "ct-min-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS,   \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_POW2_F16, 13, "ct-pow2-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_POW2_BF16, 14, "ct-pow2-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS, \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_RELU_F16, 15, "ct-relu-f16", SAFE, CT_ELEMENTWISE, F16, EXACT_BITS,    \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CT_RELU_BF16, 16, "ct-relu-bf16", SAFE, CT_ELEMENTWISE, BF16, EXACT_BITS, \
    REASON_NONE, 256, 256, 0)                                                  \
  X(CONVERT_I8_F16_ZP0, 17, "convert-i8-f16-zp0", SAFE, CT_CONVERT,           \
    I8_TO_F16, EXACT_BITS, REASON_NONE, 256, 256, 0)                           \
  X(CONVERT_I8_BF16_ZP0, 18, "convert-i8-bf16-zp0", SAFE, CT_CONVERT,         \
    I8_TO_BF16, EXACT_BITS, REASON_NONE, 256, 256, 0)                          \
  X(CONVERT_BF16_F16_PLAIN, 19, "convert-bf16-f16-plain", SAFE, CT_CONVERT,   \
    BF16_TO_F16, EXACT_BITS, REASON_NONE, 256, 256, 0)                         \
  X(CONVERT_F16_BF16_ROUND, 20, "convert-f16-bf16-round", SAFE, CT_CONVERT,   \
    F16_TO_BF16, EXACT_BITS, REASON_NONE, 256, 256, 0)                         \
  X(CONVERT_F16_I16_ROUND, 21, "convert-f16-i16-round", SAFE, CT_CONVERT,     \
    F16_TO_I16, EXACT_BITS, REASON_NONE, 256, 256, 0)                          \
  X(REDUCE_SUM_F16, 22, "reduce-sum-f16", SAFE, CT_REDUCE, F16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                  \
  X(REDUCE_MAX_F16, 23, "reduce-max-f16", SAFE, CT_REDUCE, F16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                  \
  X(REDUCE_MIN_BF16, 24, "reduce-min-bf16", SAFE, CT_REDUCE, BF16,            \
    EXACT_BITS, REASON_NONE, 128, 256, 0)                                      \
  X(REDUCE_AVG_BF16, 25, "reduce-avg-bf16", SAFE, CT_REDUCE, BF16,            \
    EXACT_BITS, REASON_NONE, 128, 256, 0)                                      \
  X(SELECT_F16, 26, "select-bit2fp-maskmove-f16", SAFE,                        \
    CT_SELECT_COMPOSITE, F16, EXACT_COMPOSITE, REASON_NONE, 256, 256, 256)    \
  X(SELECT_BF16, 27, "select-bit2fp-maskmove-bf16", SAFE,                      \
    CT_SELECT_COMPOSITE, BF16, EXACT_COMPOSITE, REASON_NONE, 256, 256, 256)   \
  X(GEMM_F16, 28, "ne-gemm-f16", SAFE, NE_GEMM, F16, EXACT_BITS,             \
    REASON_NONE, 32, 256, 0)                                                   \
  X(TDMA_PAD_F16, 29, "tdma-pad-f16", SAFE, TDMA_PAD, F16, EXACT_BITS,        \
    REASON_NONE, 2048, 2048, 0)                                               \
  X(PERIPHERAL_ARGMAX_F16, 100, "peripheral-argmax-f16", SAFE, PERIPHERAL,    \
    F16, EXACT_COMPOSITE, REASON_NONE, 8, 8, 0)                                \
  X(PERIPHERAL_ARGMIN_F16, 101, "peripheral-argmin-f16", SAFE, PERIPHERAL,    \
    F16, EXACT_COMPOSITE, REASON_NONE, 8, 8, 0)                                \
  X(CONV_F16, 102, "conv-f16", SAFE, CONV, F16, EXACT_BITS, REASON_NONE,      \
    16, 256, 0)                                                                \
  X(POOL_F16, 103, "pool-f16", SAFE, POOL, F16, EXACT_BITS, REASON_NONE,      \
    256, 256, 0)                                                               \
  X(UNPOOL_F16, 104, "unpool-f16", DEFERRED, UNPOOL, F16, NO_ORACLE,          \
    REASON_GEOMETRY_UNQUALIFIED, 0, 0, 0)                                     \
  X(TDMA_IMG2COL_F16, 105, "tdma-img2col-f16", SAFE, TDMA_IMG2COL, F16,       \
    EXACT_BITS, REASON_NONE, 2048, 2048, 0)                                   \
  X(PERIPHERAL_LUT16_F16, 106, "peripheral-lut16-f16", SAFE, PERIPHERAL,      \
    F16, EXACT_BITS, REASON_NONE, 256, 256, 0)                                 \
  X(GEMM_BF16, 107, "ne-gemm-bf16", SAFE, NE_GEMM, BF16, EXACT_BITS,         \
    REASON_NONE, 32, 256, 0)                                                    \
  X(POOL_BF16, 108, "pool-bf16", SAFE, POOL, BF16, EXACT_BITS, REASON_NONE,  \
    256, 256, 0)                                                               \
  X(TDMA_IMG2COL_BF16, 109, "tdma-img2col-bf16", SAFE, TDMA_IMG2COL, BF16,   \
    EXACT_BITS, REASON_NONE, 2048, 2048, 0)

enum WaferIFPCase {
#define WAFER_IFP_ENUM_CASE(SYMBOL, ID, SPELLING, DISPOSITION, FAMILY, DTYPE,  \
                            ORACLE, REASON, RESULT_BYTES, OUTPUT_SPAN,         \
                            AUX_SPAN)                                          \
  WAFER_IFP_CASE_##SYMBOL = ID,
  WAFER_IFP_CASES(WAFER_IFP_ENUM_CASE)
#undef WAFER_IFP_ENUM_CASE
};

enum WaferIFPStatus {
  WAFER_IFP_STATUS_OK = 0,
  WAFER_IFP_STATUS_BAD_REQUEST = 1,
  WAFER_IFP_STATUS_UNKNOWN_CASE = 2,
  WAFER_IFP_STATUS_DEFERRED_CASE = 3,
  WAFER_IFP_STATUS_DISPATCH_FAILED = 4,
};

enum WaferIFPRequestWord {
  WAFER_IFP_REQ_MAGIC = 0,
  WAFER_IFP_REQ_SCHEMA_AND_WORDS = 1,
  WAFER_IFP_REQ_CASE = 2,
  WAFER_IFP_REQ_DISPOSITION = 3,
  WAFER_IFP_REQ_FAMILY = 4,
  WAFER_IFP_REQ_DTYPE = 5,
  WAFER_IFP_REQ_ORACLE = 6,
  WAFER_IFP_REQ_RESULT_BYTES = 7,
  WAFER_IFP_REQ_OUTPUT_SPAN = 8,
  WAFER_IFP_REQ_AUX_SPAN = 9,
  WAFER_IFP_REQ_SAMPLE = 10,
  WAFER_IFP_REQ_RESOURCE_BYTES = 11,
  WAFER_IFP_REQ_SLOT_BYTES = 12,
  WAFER_IFP_REQ_BODY_OFFSET = 13,
  WAFER_IFP_REQ_GUARD = 15,
};

enum WaferIFPRecordWord {
  WAFER_IFP_REC_MAGIC = 0,
  WAFER_IFP_REC_SCHEMA_AND_WORDS = 1,
  WAFER_IFP_REC_STATUS = 2,
  WAFER_IFP_REC_CASE = 3,
  WAFER_IFP_REC_DISPOSITION = 4,
  WAFER_IFP_REC_FAMILY = 5,
  WAFER_IFP_REC_DTYPE = 6,
  WAFER_IFP_REC_ORACLE = 7,
  WAFER_IFP_REC_RESULT_BYTES = 8,
  WAFER_IFP_REC_OUTPUT_SPAN = 9,
  WAFER_IFP_REC_AUX_SPAN = 10,
  WAFER_IFP_REC_SAMPLE = 11,
  WAFER_IFP_REC_OUTPUT_GUARD_MISMATCHES = 12,
  WAFER_IFP_REC_AUX_GUARD_MISMATCHES = 13,
  WAFER_IFP_REC_REQUEST_GUARD = 14,
  WAFER_IFP_REC_OUTPUT_DDR_OFFSET = 15,
  WAFER_IFP_REC_SLOT_BYTES = 16,
  WAFER_IFP_REC_BODY_OFFSET = 17,
  WAFER_IFP_REC_STEP_FLAGS = 18,
  WAFER_IFP_REC_BIT2FP_MISMATCHES = 19,
  WAFER_IFP_REC_RECORD_GUARD = 31,
};

enum WaferIFPStepFlag {
  WAFER_IFP_STEP_TARGET_ISSUED = UINT32_C(1) << 0,
  WAFER_IFP_STEP_BIT2FP_COMPLETED = UINT32_C(1) << 1,
  WAFER_IFP_STEP_MASK_MOVE_ISSUED = UINT32_C(1) << 2,
  WAFER_IFP_STEP_FINAL_FENCE_COMPLETED = UINT32_C(1) << 3,
};

#endif
