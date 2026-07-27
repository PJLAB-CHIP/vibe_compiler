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
#define WAFER_IFP_SCHEMA 2U
#define WAFER_IFP_REQUEST_WORDS 16U
#define WAFER_IFP_RECORD_WORDS 32U
#define WAFER_IFP_RESOURCE_BYTES 16384U
#define WAFER_IFP_SLOT_BYTES 4096U
#define WAFER_IFP_BODY_OFFSET 256U
#define WAFER_IFP_OUTPUT_DDR_OFFSET 4096U
#define WAFER_IFP_AUX_DDR_OFFSET 8192U
#define WAFER_IFP_SPM_A UINT64_C(0x10000)
#define WAFER_IFP_SPM_B UINT64_C(0x20000)
#define WAFER_IFP_SPM_OUTPUT UINT64_C(0x30000)
#define WAFER_IFP_SPM_AUX UINT64_C(0x40000)
#define WAFER_IFP_REPEATED_SENTINEL_OFFSET 2048U
#define WAFER_IFP_REPEATED_SENTINEL_BYTES 512U
#define WAFER_IFP_REPEATED_AUX_BYTES 512U
#define WAFER_IFP_REPEATED_AUX_SNAPSHOT_OFFSET 512U
#define WAFER_IFP_SLOT_CANARY UINT8_C(0xa7)
#define WAFER_IFP_BIT2FP_TRUE_F16 UINT16_C(0x3c00)
#define WAFER_IFP_BIT2FP_TRUE_BF16 UINT16_C(0x3f80)
#define WAFER_IFP_BIT2FP_FALSE UINT16_C(0x0000)
#define WAFER_IFP_GEMM_ORIENTATION_NORMAL UINT32_C(0)
#define WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE UINT32_C(1)
#define WAFER_IFP_ARGMIN_WRITEBACK_POLL_BUDGET UINT32_C(1000000)
#define WAFER_IFP_ARGMIN_STATE_POLL_BUDGET UINT32_C(100000)
#define WAFER_IFP_ARGMIN_VALUE_CSR UINT32_C(0x130)
#define WAFER_IFP_ARGMIN_INDEX_CSR UINT32_C(0x140)
#define WAFER_IFP_ARGMIN_DATA_VALID (UINT64_C(1) << 32)
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
  WAFER_IFP_F32 = 7,
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
  WAFER_IFP_REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT = 4,
};

/*
 * SYMBOL, id, spelling, disposition, family, dtype, oracle, reason,
 * logical result bytes, allowed output write span, allowed auxiliary span.
 */
#define WAFER_IFP_BASE_CASES(X)                                                \
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
  X(UNPOOL_F16, 104, "unpool-f16", SAFE, UNPOOL, F16, EXACT_COMPOSITE,       \
    REASON_NONE, 512, 512, 256)                                               \
  X(TDMA_IMG2COL_F16, 105, "tdma-img2col-f16", SAFE, TDMA_IMG2COL, F16,       \
    EXACT_BITS, REASON_NONE, 2048, 2048, 0)                                   \
  X(PERIPHERAL_LUT16_F16, 106, "peripheral-lut16-f16", SAFE, PERIPHERAL,      \
    F16, EXACT_BITS, REASON_NONE, 256, 256, 0)                                 \
  X(GEMM_BF16, 107, "ne-gemm-bf16", SAFE, NE_GEMM, BF16, EXACT_BITS,         \
    REASON_NONE, 32, 256, 0)                                                    \
  X(POOL_BF16, 108, "pool-bf16", SAFE, POOL, BF16, EXACT_BITS, REASON_NONE,  \
    256, 256, 0)                                                               \
  X(TDMA_IMG2COL_BF16, 109, "tdma-img2col-bf16", SAFE, TDMA_IMG2COL, BF16,   \
    EXACT_BITS, REASON_NONE, 2048, 2048, 0)                                    \
  X(CONV_BF16, 110, "conv-bf16", SAFE, CONV, BF16, EXACT_BITS, REASON_NONE,  \
    16, 256, 0)                                                                \
  X(GEMM_BF16_ACCUM_ROUND, 111, "ne-gemm-bf16-accum-round", SAFE, NE_GEMM,   \
    BF16, EXACT_BITS, REASON_NONE, 32, 256, 0)                                  \
  X(GEMM_F16_ACCUM_ROUND, 112, "ne-gemm-f16-accum-round", SAFE, NE_GEMM,     \
    F16, EXACT_BITS, REASON_NONE, 32, 256, 0)                                   \
  X(GEMM_F16_M4, 113, "ne-gemm-f16-m4", SAFE, NE_GEMM, F16, EXACT_BITS,      \
    REASON_NONE, 128, 256, 0)                                                   \
  X(GEMM_F16_BATCH2_M8, 114, "ne-gemm-f16-batch2-m8", SAFE, NE_GEMM, F16,   \
    EXACT_BITS, REASON_NONE, 512, 512, 0)                                      \
  X(GEMM_F16_N17, 115, "ne-gemm-f16-n17", SAFE, NE_GEMM, F16, EXACT_BITS,   \
    REASON_NONE, 34, 256, 0)                                                    \
  X(GEMM_F16_K17, 116, "ne-gemm-f16-k17", SAFE, NE_GEMM, F16, EXACT_BITS,   \
    REASON_NONE, 32, 256, 0)                                                    \
  X(GEMM_F16_ORIENTED_NT, 117, "ne-gemm-f16-oriented-nt", SAFE, NE_GEMM,    \
    F16, EXACT_BITS, REASON_NONE, 64, 256, 0)                                  \
  X(GEMM_F16_PSUM, 118, "ne-gemm-f16-psum", SAFE, NE_GEMM, F16,             \
    EXACT_COMPOSITE, REASON_NONE, 32, 256, 256)                               \
  X(GEMM_F16_ORIENTED_TN, 119, "ne-gemm-f16-oriented-tn", SAFE, NE_GEMM,    \
    F16, EXACT_BITS, REASON_NONE, 64, 256, 0)                                  \
  X(GEMM_F16_ORIENTED_TT, 120, "ne-gemm-f16-oriented-tt", SAFE, NE_GEMM,    \
    F16, EXACT_BITS, REASON_NONE, 64, 256, 0)                                  \
  X(GEMM_F16_N65, 121, "ne-gemm-f16-n65", SAFE, NE_GEMM, F16, EXACT_BITS,   \
    REASON_NONE, 130, 256, 0)                                                  \
  X(GEMM_BF16_BATCH2_M8, 122, "ne-gemm-bf16-batch2-m8", SAFE, NE_GEMM,     \
    BF16, EXACT_BITS, REASON_NONE, 512, 512, 0)                                \
  X(GEMM_BF16_K17, 123, "ne-gemm-bf16-k17", SAFE, NE_GEMM, BF16,           \
    EXACT_BITS, REASON_NONE, 32, 256, 0)                                      \
  X(GEMM_BF16_N65, 124, "ne-gemm-bf16-n65", SAFE, NE_GEMM, BF16,           \
    EXACT_BITS, REASON_NONE, 130, 256, 0)                                     \
  X(GEMM_BF16_ORIENTED_NT, 125, "ne-gemm-bf16-oriented-nt", SAFE, NE_GEMM, \
    BF16, EXACT_BITS, REASON_NONE, 64, 256, 0)                                 \
  X(CT_ADD_F16_TAIL130, 126, "ct-add-f16-tail130", SAFE, CT_ELEMENTWISE,    \
    F16, EXACT_BITS, REASON_NONE, 260, 512, 0)                                 \
  X(CT_ADD_BF16_TAIL130, 127, "ct-add-bf16-tail130", SAFE, CT_ELEMENTWISE,  \
    BF16, EXACT_BITS, REASON_NONE, 260, 512, 0)                                \
  X(CT_ADD_F32, 128, "ct-add-f32", SAFE, CT_ELEMENTWISE, F32, EXACT_BITS,   \
    REASON_NONE, 512, 512, 0)                                                  \
  X(CT_ADD_SPECIAL_F16, 129, "ct-add-special-f16", SAFE, CT_ELEMENTWISE,    \
    F16, EXACT_BITS, REASON_NONE, 256, 256, 0)                                 \
  X(CT_ADD_SPECIAL_BF16, 130, "ct-add-special-bf16", SAFE, CT_ELEMENTWISE,  \
    BF16, EXACT_BITS, REASON_NONE, 256, 256, 0)                                \
  /* Opcode 121 completed with intact guards but did not establish the       \
   * indexed-scatter numeric contract; keep the real path as bounded raw. */ \
  X(UNPOOL_INDEX_F16, 131, "unpool-index-f16", SAFE, UNPOOL, F16,           \
    NO_ORACLE, REASON_NONE, 512, 512, 256)                                    \
  X(POOL_AVG_F16, 132, "pool-avg-f16", SAFE, POOL, F16, EXACT_BITS,          \
    REASON_NONE, 256, 256, 0)                                                  \
  X(POOL_SUM_F16, 133, "pool-sum-f16", SAFE, POOL, F16, EXACT_BITS,          \
    REASON_NONE, 256, 256, 0)                                                  \
  X(POOL_MIN_F16, 134, "pool-min-f16", SAFE, POOL, F16, EXACT_BITS,          \
    REASON_NONE, 256, 256, 0)                                                  \
  X(POOL_INDEXED_MIN_F16, 135, "pool-indexed-min-f16", SAFE, POOL, F16,      \
    EXACT_COMPOSITE, REASON_NONE, 512, 512, 0)                                 \
  X(UNPOOL_AVG_F16, 136, "unpool-avg-f16", SAFE, UNPOOL, F16, EXACT_BITS,    \
    REASON_NONE, 512, 512, 0)                                                  \
  /* Identity geometry produced a bounded non-identity writeback; the public \
   * scale ABI does not establish a portable numeric reference yet. */       \
  X(PERIPHERAL_BILINEAR_F16, 137, "peripheral-bilinear-f16", SAFE,           \
    PERIPHERAL, F16, NO_ORACLE, REASON_NONE, 256, 256, 0)                     \
  X(PERIPHERAL_FACTORIZE_F32_OBSERVED, 138,                                  \
    "peripheral-factorize-f32-observed", SAFE, PERIPHERAL, F32, NO_ORACLE,   \
    REASON_NONE, 1536, 1536, 0)                                               \
  X(PERIPHERAL_LUT32_OBSERVED, 139, "peripheral-lut32-observed", SAFE,       \
    PERIPHERAL, F32, NO_ORACLE, REASON_NONE, 512, 512, 0)                    \
  X(PERIPHERAL_RANDGEN_F16_OBSERVED, 140,                                    \
    "peripheral-randgen-f16-observed", SAFE, PERIPHERAL, F16, NO_ORACLE,     \
    REASON_NONE, 1536, 1536, 0)                                               \
  X(PERIPHERAL_ELEMMASK_F16_OBSERVED, 141,                                   \
    "peripheral-elemmask-f16-observed", SAFE, PERIPHERAL, F16, NO_ORACLE,    \
    REASON_NONE, 256, 256, 0)                                                \
  /* Negative finite ArgMin has completed before but disagreed with the      \
   * semantic minimum.  Preserve it as bounded raw evidence rather than      \
   * extending the positive-finite exact qualification to the whole opcode. */ \
  X(PERIPHERAL_ARGMIN_NEGATIVE_F16_OBSERVED, 142,                            \
    "peripheral-argmin-negative-f16-observed", SAFE, PERIPHERAL, F16,       \
    NO_ORACLE, REASON_NONE, 8, 8, 0)                                         \
  /* Independent opcode-118 value/index qualification with ABI-ordered      \
   * Kx=3, Ky=2, Sx=2, Sy=1 geometry.  The asymmetry exposes X/Y swaps. */   \
  X(POOL_INDEXED_MAX_F16_ASYMMETRIC, 143,                                   \
    "pool-indexed-max-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_COMPOSITE,     \
    REASON_NONE, 1024, 1024, 0)

#define WAFER_IFP_CT_UNPOOL_CAPABILITY_CASES(X)                              \
  X(UNPOOL_INDEX_BF16_OBSERVED, 236, "unpool-index-bf16-observed", SAFE,    \
    UNPOOL, BF16, NO_ORACLE, REASON_NONE, 512, 512, 256)                    \
  X(UNPOOL_INDEX_F32_OBSERVED, 237, "unpool-index-f32-observed", SAFE,      \
    UNPOOL, F32, NO_ORACLE, REASON_NONE, 1024, 1024, 512)                   \
  X(UNPOOL_AVG_BF16, 238, "unpool-avg-bf16", SAFE, UNPOOL, BF16,           \
    EXACT_BITS, REASON_NONE, 512, 512, 0)                                   \
  X(UNPOOL_AVG_F32, 239, "unpool-avg-f32", SAFE, UNPOOL, F32, EXACT_BITS,  \
    REASON_NONE, 1024, 1024, 0)                                             \
  X(UNPOOL_MASK_BF16, 240, "unpool-mask-bf16", SAFE, UNPOOL, BF16,         \
    EXACT_COMPOSITE, REASON_NONE, 512, 512, 256)                            \
  X(UNPOOL_MASK_F32, 241, "unpool-mask-f32", SAFE, UNPOOL, F32,            \
    NO_ORACLE, REASON_NONE, 1024, 1024, 512)                                \
  X(UNPOOL_INDEX_F16_ASYMMETRIC_OBSERVED, 242,                              \
    "unpool-index-f16-k3x2-s2x1-observed", SAFE, UNPOOL, F16, NO_ORACLE,   \
    REASON_NONE, 1920, 2048, 512)                                           \
  X(UNPOOL_AVG_F16_ASYMMETRIC_OBSERVED, 243,                                \
    "unpool-avg-f16-k3x2-s2x1-observed", SAFE, UNPOOL, F16, NO_ORACLE,     \
    REASON_NONE, 1920, 2048, 0)                                             \
  X(UNPOOL_MASK_F16_ASYMMETRIC, 244,                                        \
    "unpool-mask-f16-k3x2-s2x1", SAFE, UNPOOL, F16, NO_ORACLE,             \
    REASON_NONE, 1920, 2048, 512)                                           \
  X(UNPOOL_INDEX_F16_REPEATED_OVERLAP_OBSERVED, 245,                        \
    "unpool-index-f16-repeated-overlap-observed", SAFE, UNPOOL, F16,       \
    NO_ORACLE, REASON_NONE, 1920, 2048, 1024)                               \
  X(UNPOOL_MASK_F16_REPEATED_OVERLAP_OBSERVED, 246,                         \
    "unpool-mask-f16-repeated-overlap-observed", SAFE, UNPOOL, F16,        \
    NO_ORACLE, REASON_NONE, 1920, 2048, 1024)

#define WAFER_IFP_PENDING_NUMERIC_DOMAIN_CASES(X)                            \
  /* Three positive-finite vectors place equal minima before, across and     \
   * after the 64-element boundary.  The host records which tied index is    \
   * selected without assuming first- or last-wins semantics. */             \
  X(PERIPHERAL_ARGMIN_TIE_F16_OBSERVED, 247,                                 \
    "peripheral-argmin-tie-f16-observed", SAFE, PERIPHERAL, F16,           \
    NO_ORACLE, REASON_NONE, 8, 8, 256)                                      \
  /* Three vectors independently place positive quiet, positive signaling   \
   * and negative quiet NaNs around a unique finite minimum.  Raw value and  \
   * index are retained so NaN selection/ignoring and quieting remain        \
   * distinguishable instead of being folded into finite-domain support. */  \
  X(PERIPHERAL_ARGMIN_NAN_F16_OBSERVED, 248,                                 \
    "peripheral-argmin-nan-f16-observed", SAFE, PERIPHERAL, F16,           \
    NO_ORACLE, REASON_NONE, 8, 8, 256)

#define WAFER_IFP_CT_REDUCE_CAPABILITY_CASES(X)                               \
  X(REDUCE_SUM_F16_C_NCX, 144,                                                \
    "reduce-sum-f16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_SUM_F16_W_NCX, 145,                                                \
    "reduce-sum-f16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_SUM_F16_H_NCX, 146,                                                \
    "reduce-sum-f16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_SUM_F16_HW_NCX, 147,                                               \
    "reduce-sum-f16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,    \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_SUM_BF16_C_NCX, 148,                                               \
    "reduce-sum-bf16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_SUM_BF16_W_NCX, 149,                                               \
    "reduce-sum-bf16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_SUM_BF16_H_NCX, 150,                                               \
    "reduce-sum-bf16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_SUM_BF16_HW_NCX, 151,                                              \
    "reduce-sum-bf16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,  \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_SUM_F32_C_NCX, 152,                                                \
    "reduce-sum-f32-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 24, 256, 0)                                                  \
  X(REDUCE_SUM_F32_W_NCX, 153,                                                \
    "reduce-sum-f32-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 256, 256, 0)                                                 \
  X(REDUCE_SUM_F32_H_NCX, 154,                                                \
    "reduce-sum-f32-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 520, 768, 0)                                                 \
  X(REDUCE_SUM_F32_HW_NCX, 155,                                               \
    "reduce-sum-f32-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,    \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_AVG_F16_C_NCX, 156,                                                \
    "reduce-avg-f16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_AVG_F16_W_NCX, 157,                                                \
    "reduce-avg-f16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_AVG_F16_H_NCX, 158,                                                \
    "reduce-avg-f16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_AVG_F16_HW_NCX, 159,                                               \
    "reduce-avg-f16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,    \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_AVG_BF16_C_NCX, 160,                                               \
    "reduce-avg-bf16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_AVG_BF16_W_NCX, 161,                                               \
    "reduce-avg-bf16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_AVG_BF16_H_NCX, 162,                                               \
    "reduce-avg-bf16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_AVG_BF16_HW_NCX, 163,                                              \
    "reduce-avg-bf16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,  \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_AVG_F32_C_NCX, 164,                                                \
    "reduce-avg-f32-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 24, 256, 0)                                                  \
  X(REDUCE_AVG_F32_W_NCX, 165,                                                \
    "reduce-avg-f32-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 256, 256, 0)                                                 \
  X(REDUCE_AVG_F32_H_NCX, 166,                                                \
    "reduce-avg-f32-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 520, 768, 0)                                                 \
  X(REDUCE_AVG_F32_HW_NCX, 167,                                               \
    "reduce-avg-f32-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,    \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MAX_F16_C_NCX, 168,                                                \
    "reduce-max-f16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_MAX_F16_W_NCX, 169,                                                \
    "reduce-max-f16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_MAX_F16_H_NCX, 170,                                                \
    "reduce-max-f16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MAX_F16_HW_NCX, 171,                                               \
    "reduce-max-f16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,    \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_MAX_BF16_C_NCX, 172,                                               \
    "reduce-max-bf16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_MAX_BF16_W_NCX, 173,                                               \
    "reduce-max-bf16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_MAX_BF16_H_NCX, 174,                                               \
    "reduce-max-bf16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MAX_BF16_HW_NCX, 175,                                              \
    "reduce-max-bf16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,  \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_MAX_F32_C_NCX, 176,                                                \
    "reduce-max-f32-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 24, 256, 0)                                                  \
  X(REDUCE_MAX_F32_W_NCX, 177,                                                \
    "reduce-max-f32-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 256, 256, 0)                                                 \
  X(REDUCE_MAX_F32_H_NCX, 178,                                                \
    "reduce-max-f32-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 520, 768, 0)                                                 \
  X(REDUCE_MAX_F32_HW_NCX, 179,                                               \
    "reduce-max-f32-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,    \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MIN_F16_C_NCX, 180,                                                \
    "reduce-min-f16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_MIN_F16_W_NCX, 181,                                                \
    "reduce-min-f16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_MIN_F16_H_NCX, 182,                                                \
    "reduce-min-f16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,     \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MIN_F16_HW_NCX, 183,                                               \
    "reduce-min-f16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F16, EXACT_BITS,    \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_MIN_BF16_C_NCX, 184,                                               \
    "reduce-min-bf16-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 12, 256, 0)                                                  \
  X(REDUCE_MIN_BF16_W_NCX, 185,                                               \
    "reduce-min-bf16-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 128, 256, 0)                                                 \
  X(REDUCE_MIN_BF16_H_NCX, 186,                                               \
    "reduce-min-bf16-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,   \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_MIN_BF16_HW_NCX, 187,                                              \
    "reduce-min-bf16-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, BF16, EXACT_BITS,  \
    REASON_NONE, 130, 256, 0)                                                 \
  X(REDUCE_MIN_F32_C_NCX, 188,                                                \
    "reduce-min-f32-c-ncx-n1h2w3c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 24, 256, 0)                                                  \
  X(REDUCE_MIN_F32_W_NCX, 189,                                                \
    "reduce-min-f32-w-ncx-n1h1w4c64", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 256, 256, 0)                                                 \
  X(REDUCE_MIN_F32_H_NCX, 190,                                                \
    "reduce-min-f32-h-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,     \
    REASON_NONE, 520, 768, 0)                                                 \
  X(REDUCE_MIN_F32_HW_NCX, 191,                                               \
    "reduce-min-f32-hw-ncx-n1h3w2c65", SAFE, CT_REDUCE, F32, EXACT_BITS,    \
    REASON_NONE, 260, 512, 0)                                                 \
  X(REDUCE_SUM_F16_C_CX, 192, "reduce-sum-f16-c-cx-w4c8", SAFE,             \
    CT_REDUCE, F16, EXACT_BITS, REASON_NONE, 8, 256, 0)                       \
  X(REDUCE_AVG_F16_C_CX, 193, "reduce-avg-f16-c-cx-w4c8", SAFE,             \
    CT_REDUCE, F16, EXACT_BITS, REASON_NONE, 8, 256, 0)                       \
  X(REDUCE_MAX_F16_C_CX, 194, "reduce-max-f16-c-cx-w4c8", SAFE,             \
    CT_REDUCE, F16, EXACT_BITS, REASON_NONE, 8, 256, 0)                       \
  X(REDUCE_MIN_F16_C_CX, 195, "reduce-min-f16-c-cx-w4c8", SAFE,             \
    CT_REDUCE, F16, EXACT_BITS, REASON_NONE, 8, 256, 0)                       \
  /* Historical raw dimensions 3/5 are absent from the version-matched       \
   * public enum.  An isolated dimension-3 launch exceeded its outer         \
   * deadline, so the entire N/HWC raw class remains discoverable but cannot \
   * reach dispatch without a future, separately authorized protocol. */     \
  X(REDUCE_SUM_F16_N_RAW, 196, "reduce-sum-f16-n-raw-observed", DEFERRED,   \
    CT_REDUCE, F16, NO_ORACLE, REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT,      \
    0, 0, 0)                                                                 \
  X(REDUCE_SUM_F16_HWC_RAW, 197,                                             \
    "reduce-sum-f16-hwc-raw-observed", DEFERRED, CT_REDUCE, F16, NO_ORACLE, \
    REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT, 0, 0, 0)                        \
  X(REDUCE_AVG_F16_N_RAW, 198, "reduce-avg-f16-n-raw-observed", DEFERRED,   \
    CT_REDUCE, F16, NO_ORACLE, REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT,      \
    0, 0, 0)                                                                 \
  X(REDUCE_AVG_F16_HWC_RAW, 199,                                             \
    "reduce-avg-f16-hwc-raw-observed", DEFERRED, CT_REDUCE, F16, NO_ORACLE, \
    REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT, 0, 0, 0)                        \
  X(REDUCE_MAX_F16_N_RAW, 200, "reduce-max-f16-n-raw-observed", DEFERRED,   \
    CT_REDUCE, F16, NO_ORACLE, REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT,      \
    0, 0, 0)                                                                 \
  X(REDUCE_MAX_F16_HWC_RAW, 201,                                             \
    "reduce-max-f16-hwc-raw-observed", DEFERRED, CT_REDUCE, F16, NO_ORACLE, \
    REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT, 0, 0, 0)                        \
  X(REDUCE_MIN_F16_N_RAW, 202, "reduce-min-f16-n-raw-observed", DEFERRED,   \
    CT_REDUCE, F16, NO_ORACLE, REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT,      \
    0, 0, 0)                                                                 \
  X(REDUCE_MIN_F16_HWC_RAW, 203,                                             \
    "reduce-min-f16-hwc-raw-observed", DEFERRED, CT_REDUCE, F16, NO_ORACLE, \
    REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT, 0, 0, 0)

#define WAFER_IFP_CT_POOL_CAPABILITY_CASES(X)                                \
  X(CT_POOL_AVG_BF16_SYMMETRIC, 205,                                        \
    "pool-avg-bf16-k2x2-s2x2", SAFE, POOL, BF16, EXACT_BITS,               \
    REASON_NONE, 256, 256, 0)                                               \
  X(CT_POOL_AVG_F32_SYMMETRIC, 206,                                         \
    "pool-avg-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_SUM_BF16_SYMMETRIC, 208,                                        \
    "pool-sum-bf16-k2x2-s2x2", SAFE, POOL, BF16, EXACT_BITS,               \
    REASON_NONE, 256, 256, 0)                                               \
  X(CT_POOL_SUM_F32_SYMMETRIC, 209,                                         \
    "pool-sum-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_MAX_F32_SYMMETRIC, 212,                                         \
    "pool-max-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMAX_BF16_SYMMETRIC, 214,                                 \
    "pool-indexed-max-bf16-k2x2-s2x2", SAFE, POOL, BF16, EXACT_COMPOSITE,  \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMAX_F32_SYMMETRIC, 215,                                  \
    "pool-indexed-max-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_COMPOSITE,    \
    REASON_NONE, 1024, 1024, 0)                                             \
  X(CT_POOL_MIN_BF16_SYMMETRIC, 217,                                        \
    "pool-min-bf16-k2x2-s2x2", SAFE, POOL, BF16, EXACT_BITS,               \
    REASON_NONE, 256, 256, 0)                                               \
  X(CT_POOL_MIN_F32_SYMMETRIC, 218,                                         \
    "pool-min-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMIN_BF16_SYMMETRIC, 220,                                 \
    "pool-indexed-min-bf16-k2x2-s2x2", SAFE, POOL, BF16, EXACT_COMPOSITE,  \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMIN_F32_SYMMETRIC, 221,                                  \
    "pool-indexed-min-f32-k2x2-s2x2", SAFE, POOL, F32, EXACT_COMPOSITE,    \
    REASON_NONE, 1024, 1024, 0)                                             \
  X(CT_POOL_AVG_F16_ASYMMETRIC, 222,                                        \
    "pool-avg-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_SUM_F16_ASYMMETRIC, 223,                                        \
    "pool-sum-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_MAX_F16_ASYMMETRIC, 224,                                        \
    "pool-max-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_MIN_F16_ASYMMETRIC, 226,                                        \
    "pool-min-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_BITS,                 \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMIN_F16_ASYMMETRIC, 227,                                 \
    "pool-indexed-min-f16-k3x2-s2x1", SAFE, POOL, F16, EXACT_COMPOSITE,    \
    REASON_NONE, 1024, 1024, 0)                                             \
  X(CT_POOL_AVG_F16_PADDED_OBSERVED, 228,                                   \
    "pool-avg-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,             \
    NO_ORACLE, REASON_NONE, 512, 512, 0)                                    \
  X(CT_POOL_SUM_F16_PADDED_OBSERVED, 229,                                   \
    "pool-sum-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,             \
    NO_ORACLE, REASON_NONE, 512, 512, 0)                                    \
  X(CT_POOL_MAX_F16_PADDED_OBSERVED, 230,                                   \
    "pool-max-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,             \
    NO_ORACLE, REASON_NONE, 512, 512, 0)                                    \
  X(CT_POOL_INDEXEDMAX_F16_PADDED_OBSERVED, 231,                            \
    "pool-indexed-max-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,     \
    NO_ORACLE, REASON_NONE, 1024, 1024, 0)                                  \
  X(CT_POOL_MIN_F16_PADDED_OBSERVED, 232,                                   \
    "pool-min-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,             \
    NO_ORACLE, REASON_NONE, 512, 512, 0)                                    \
  X(CT_POOL_INDEXEDMIN_F16_PADDED_OBSERVED, 233,                            \
    "pool-indexed-min-f16-k3x2-s2x1-padded-observed", SAFE, POOL, F16,     \
    NO_ORACLE, REASON_NONE, 1024, 1024, 0)                                  \
  X(CT_POOL_INDEXEDMAX_F16_TIE_OBSERVED, 234,                               \
    "pool-indexed-max-f16-tie-observed", SAFE, POOL, F16, NO_ORACLE,       \
    REASON_NONE, 512, 512, 0)                                               \
  X(CT_POOL_INDEXEDMIN_F16_TIE_OBSERVED, 235,                               \
    "pool-indexed-min-f16-tie-observed", SAFE, POOL, F16, NO_ORACLE,       \
    REASON_NONE, 512, 512, 0)

#define WAFER_IFP_CASES(X)                                                    \
  WAFER_IFP_BASE_CASES(X)                                                     \
  WAFER_IFP_CT_REDUCE_CAPABILITY_CASES(X)                                    \
  WAFER_IFP_CT_POOL_CAPABILITY_CASES(X)                                      \
  WAFER_IFP_CT_UNPOOL_CAPABILITY_CASES(X)                                    \
  WAFER_IFP_PENDING_NUMERIC_DOMAIN_CASES(X)

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
  WAFER_IFP_STATUS_DIAGNOSTIC_BOUNDED = 5,
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
  WAFER_IFP_REC_ARGMIN_DIAGNOSTIC_FLAGS = 20,
  WAFER_IFP_REC_ARGMIN_WRITEBACK_POLLS = 21,
  WAFER_IFP_REC_ARGMIN_VALUE_RAW = 22,
  WAFER_IFP_REC_ARGMIN_INDEX_RAW = 23,
  WAFER_IFP_REC_ARGMIN_ARRIVAL_POLLS = 24,
  WAFER_IFP_REC_ARGMIN_TASKSTATUS_FIRST_LAST = 25,
  WAFER_IFP_REC_ARGMIN_IBCOUNTER_FIRST_LAST = 26,
  WAFER_IFP_REC_ARGMIN_STATE_POLLS = 27,
  WAFER_IFP_REC_ARGMIN_WRITEBACK_BUDGET = 28,
  WAFER_IFP_REC_ARGMIN_STATE_BUDGET = 29,
  WAFER_IFP_REC_RECORD_GUARD = 31,
};

enum WaferIFPStepFlag {
  WAFER_IFP_STEP_TARGET_ISSUED = UINT32_C(1) << 0,
  WAFER_IFP_STEP_BIT2FP_COMPLETED = UINT32_C(1) << 1,
  WAFER_IFP_STEP_MASK_MOVE_ISSUED = UINT32_C(1) << 2,
  WAFER_IFP_STEP_FINAL_FENCE_COMPLETED = UINT32_C(1) << 3,
  WAFER_IFP_STEP_REPEATED_OVERLAP_VALUES_STAGED = UINT32_C(1) << 4,
  WAFER_IFP_STEP_ARGMIN_INPUT_SNAPSHOTTED = UINT32_C(1) << 5,
  WAFER_IFP_STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED = UINT32_C(1) << 6,
  WAFER_IFP_STEP_ARGMIN_DIAGNOSTIC_RETURNED = UINT32_C(1) << 7,
};

enum WaferIFPArgMinDiagnosticFlag {
  WAFER_IFP_ARGMIN_VALUE_VALID = UINT32_C(1) << 0,
  WAFER_IFP_ARGMIN_INDEX_VALID = UINT32_C(1) << 1,
  WAFER_IFP_ARGMIN_WRITEBACK_BUDGET_EXHAUSTED = UINT32_C(1) << 2,
  WAFER_IFP_ARGMIN_TASK_DRAINED = UINT32_C(1) << 3,
  WAFER_IFP_ARGMIN_STATE_BUDGET_EXHAUSTED = UINT32_C(1) << 4,
};

#endif
