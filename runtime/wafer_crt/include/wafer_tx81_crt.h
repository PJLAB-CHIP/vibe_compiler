#ifndef WAFER_TX81_CRT_H
#define WAFER_TX81_CRT_H

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void wafer_tx81_rdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0, uint32_t iteration1,
                     uint32_t iteration2, uint32_t format);
void wafer_tx81_wdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0, uint32_t iteration1,
                     uint32_t iteration2, uint32_t format);
void wafer_tx81_gather_scatter(uint64_t src, uint64_t dst, uint32_t byte_count,
                               uint32_t inner_bytes, uint32_t src_stride0,
                               uint32_t src_stride1, uint32_t src_stride2,
                               uint32_t src_iteration0, uint32_t src_iteration1,
                               uint32_t src_iteration2, uint32_t dst_stride0,
                               uint32_t dst_stride1, uint32_t dst_stride2,
                               uint32_t dst_iteration0, uint32_t dst_iteration1,
                               uint32_t dst_iteration2);
void wafer_tx81_memset(uint64_t dst, uint32_t value, uint32_t elem_count,
                       uint32_t format);
void wafer_tx81_bit2fp(uint64_t src, uint64_t dst, uint32_t elem_count,
                       uint32_t format);
void wafer_tx81_mask_move(uint64_t src, uint32_t mask, uint64_t dst,
                          uint32_t elem_count, uint32_t format);
void wafer_tx81_gemm(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t m,
                     uint32_t k, uint32_t n, uint32_t batch_count,
                     uint32_t format);
void wafer_tx81_gemm_oriented_v2(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                 uint32_t m, uint32_t k, uint32_t n,
                                 uint32_t batch_count, uint32_t format,
                                 uint32_t lhs_orientation,
                                 uint32_t rhs_orientation);
void wafer_tx81_tdma_pad(uint64_t src, uint64_t dst, uint32_t src_n,
                         uint32_t src_h, uint32_t src_w, uint32_t src_c,
                         uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                         uint32_t dst_c, uint32_t pad_top, uint32_t pad_bottom,
                         uint32_t pad_left, uint32_t pad_right,
                         uint32_t format);
void wafer_tx81_tdma_img2col(uint64_t src, uint64_t dst, uint32_t src_n,
                             uint32_t src_h, uint32_t src_w, uint32_t src_c,
                             uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                             uint32_t dst_c, uint32_t pad_top,
                             uint32_t pad_bottom, uint32_t pad_left,
                             uint32_t pad_right, uint32_t kernel_x,
                             uint32_t kernel_y, uint32_t stride_x,
                             uint32_t stride_y, uint32_t format);
void wafer_tx81_local_fence(void);

void wafer_tx81_direct_dte_begin(uint64_t status_addr, uint32_t rank_count);
void wafer_tx81_direct_dte_begin_after_prepare(uint64_t status_addr,
                                               uint32_t rank_count);
uint64_t wafer_tx81_direct_dte_send_prepare(
    uint64_t src, uint64_t remote_dst, uint32_t byte_count, uint32_t local_tile,
    uint32_t remote_tile, uint32_t remote_fsm_id, uint32_t is_high_performance);
uint64_t wafer_tx81_direct_dte_recv_prepare(uint64_t dst, uint32_t byte_count,
                                            uint32_t local_tile,
                                            uint32_t remote_tile,
                                            uint32_t local_fsm_id);
void wafer_tx81_direct_dte_wait(uint64_t event);
void wafer_tx81_direct_dte_finish(void);

void wafer_tx81_elementwise_abs(uint64_t src, uint64_t dst, uint32_t elem_count,
                                uint32_t format);
void wafer_tx81_elementwise_recip(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_square(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_sqrt(uint64_t src, uint64_t dst,
                                 uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_rsqrt(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_neg(uint64_t src, uint64_t dst, uint32_t elem_count,
                                uint32_t format);
void wafer_tx81_elementwise_max(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_min(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_add(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_sub(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_mul(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_div(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_eq(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_ne(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_ge(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_gt(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_le(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_lt(uint64_t lhs, uint64_t rhs, uint64_t dst,
                               uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_logic_not(uint64_t src, uint64_t dst,
                                      uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_logic_and(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                      uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_logic_or(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                     uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_logic_xor(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                      uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_log2(uint64_t src, uint64_t dst,
                                 uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_ln(uint64_t src, uint64_t dst, uint32_t elem_count,
                               uint32_t format);
void wafer_tx81_elementwise_pow2(uint64_t src, uint64_t dst,
                                 uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_exp(uint64_t src, uint64_t dst, uint32_t elem_count,
                                uint32_t format);
void wafer_tx81_elementwise_exp_lp(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_sin(uint64_t src, uint64_t dst, uint32_t elem_count,
                                uint32_t format);
void wafer_tx81_elementwise_cos(uint64_t src, uint64_t dst, uint32_t elem_count,
                                uint32_t format);
void wafer_tx81_elementwise_tanh(uint64_t src, uint64_t dst,
                                 uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_sigmoid(uint64_t src, uint64_t dst,
                                    uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_relu(uint64_t src, uint64_t dst,
                                 uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_satrelu(uint64_t src, uint64_t dst,
                                    uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_leakyrelu(uint64_t src, uint64_t dst,
                                      uint32_t elem_count, uint32_t format);
void wafer_tx81_elementwise_softplus(uint64_t src, uint64_t dst,
                                     uint32_t elem_count, uint32_t format);

void wafer_tx81_reduce_sum(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,
                           uint32_t h, uint32_t w, uint32_t c, uint32_t format);
void wafer_tx81_reduce_max(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,
                           uint32_t h, uint32_t w, uint32_t c, uint32_t format);
void wafer_tx81_reduce_min(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,
                           uint32_t h, uint32_t w, uint32_t c, uint32_t format);
void wafer_tx81_reduce_avg(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,
                           uint32_t h, uint32_t w, uint32_t c, uint32_t format);

void wafer_tx81_convert_int8_fp16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_int8_bf16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_int8_fp32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_int8_tf32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_int16_fp16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int16_bf16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int16_fp32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int16_tf32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int32_fp16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int32_bf16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int32_fp32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_int32_tf32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_bf16_int8(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_bf16_int16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_bf16_int32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_bf16_fp16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_bf16_fp32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_bf16_tf32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp16_int8(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp16_int16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_fp16_int32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_fp16_bf16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp16_fp32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp16_tf32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp32_int8(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp32_int16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_fp32_int32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_fp32_fp16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp32_bf16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_fp32_tf32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_tf32_int8(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_tf32_int16(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_tf32_int32(uint64_t src, uint64_t dst,
                                   uint32_t elem_count, uint32_t zero_point,
                                   uint32_t rounding_mode);
void wafer_tx81_convert_tf32_fp16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_tf32_bf16(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);
void wafer_tx81_convert_tf32_fp32(uint64_t src, uint64_t dst,
                                  uint32_t elem_count, uint32_t zero_point,
                                  uint32_t rounding_mode);

void wafer_tx81_conv(uint64_t input, uint64_t weight, uint64_t dst,
                     uint32_t kind, uint32_t input_n, uint32_t input_h,
                     uint32_t input_w, uint32_t input_c, uint32_t weight_n,
                     uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
                     uint32_t output_n, uint32_t output_h, uint32_t output_w,
                     uint32_t output_c, uint32_t pad_top, uint32_t pad_bottom,
                     uint32_t pad_left, uint32_t pad_right, uint32_t unpad_top,
                     uint32_t unpad_bottom, uint32_t unpad_left,
                     uint32_t unpad_right, uint32_t kernel_x, uint32_t kernel_y,
                     uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
                     uint32_t dilation1, uint32_t format);
void wafer_tx81_depthwise_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
    uint32_t output_n, uint32_t output_h, uint32_t output_w, uint32_t output_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t unpad_top, uint32_t unpad_bottom,
    uint32_t unpad_left, uint32_t unpad_right, uint32_t kernel_x,
    uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format);
void wafer_tx81_backward_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
    uint32_t output_n, uint32_t output_h, uint32_t output_w, uint32_t output_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t unpad_top, uint32_t unpad_bottom,
    uint32_t unpad_left, uint32_t unpad_right, uint32_t kernel_x,
    uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format);

void wafer_tx81_pool_avg(uint64_t input, uint64_t dst, uint32_t kind,
                         uint32_t src_n, uint32_t src_h, uint32_t src_w,
                         uint32_t src_c, uint32_t dst_n, uint32_t dst_h,
                         uint32_t dst_w, uint32_t dst_c, uint32_t pad_top,
                         uint32_t pad_bottom, uint32_t pad_left,
                         uint32_t pad_right, uint32_t kernel_x,
                         uint32_t kernel_y, uint32_t stride_x,
                         uint32_t stride_y, uint32_t format);
void wafer_tx81_pool_sum(uint64_t input, uint64_t dst, uint32_t kind,
                         uint32_t src_n, uint32_t src_h, uint32_t src_w,
                         uint32_t src_c, uint32_t dst_n, uint32_t dst_h,
                         uint32_t dst_w, uint32_t dst_c, uint32_t pad_top,
                         uint32_t pad_bottom, uint32_t pad_left,
                         uint32_t pad_right, uint32_t kernel_x,
                         uint32_t kernel_y, uint32_t stride_x,
                         uint32_t stride_y, uint32_t format);
void wafer_tx81_pool_max(uint64_t input, uint64_t dst, uint32_t kind,
                         uint32_t src_n, uint32_t src_h, uint32_t src_w,
                         uint32_t src_c, uint32_t dst_n, uint32_t dst_h,
                         uint32_t dst_w, uint32_t dst_c, uint32_t pad_top,
                         uint32_t pad_bottom, uint32_t pad_left,
                         uint32_t pad_right, uint32_t kernel_x,
                         uint32_t kernel_y, uint32_t stride_x,
                         uint32_t stride_y, uint32_t format);
void wafer_tx81_pool_min(uint64_t input, uint64_t dst, uint32_t kind,
                         uint32_t src_n, uint32_t src_h, uint32_t src_w,
                         uint32_t src_c, uint32_t dst_n, uint32_t dst_h,
                         uint32_t dst_w, uint32_t dst_c, uint32_t pad_top,
                         uint32_t pad_bottom, uint32_t pad_left,
                         uint32_t pad_right, uint32_t kernel_x,
                         uint32_t kernel_y, uint32_t stride_x,
                         uint32_t stride_y, uint32_t format);
void wafer_tx81_pool_indexedmax(
    uint64_t input, uint64_t value_dst, uint64_t index_dst, uint32_t kind,
    uint32_t src_n, uint32_t src_h, uint32_t src_w, uint32_t src_c,
    uint32_t dst_n, uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t kernel_x, uint32_t kernel_y, uint32_t stride_x,
    uint32_t stride_y, uint32_t format);
void wafer_tx81_pool_indexedmin(
    uint64_t input, uint64_t value_dst, uint64_t index_dst, uint32_t kind,
    uint32_t src_n, uint32_t src_h, uint32_t src_w, uint32_t src_c,
    uint32_t dst_n, uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t kernel_x, uint32_t kernel_y, uint32_t stride_x,
    uint32_t stride_y, uint32_t format);

void wafer_tx81_unpool_unpool(uint64_t input, uint64_t dst, uint32_t kind,
                              uint32_t index, uint32_t src_n, uint32_t src_h,
                              uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                              uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                              uint32_t kernel_x, uint32_t kernel_y,
                              uint32_t stride_x, uint32_t stride_y,
                              uint32_t format);
void wafer_tx81_unpool_avg(uint64_t input, uint64_t dst, uint32_t kind,
                           uint32_t index, uint32_t src_n, uint32_t src_h,
                           uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                           uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                           uint32_t kernel_x, uint32_t kernel_y,
                           uint32_t stride_x, uint32_t stride_y,
                           uint32_t format);
void wafer_tx81_unpool_mask(uint64_t input, uint64_t dst, uint32_t kind,
                            uint32_t index, uint32_t src_n, uint32_t src_h,
                            uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                            uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                            uint32_t kernel_x, uint32_t kernel_y,
                            uint32_t stride_x, uint32_t stride_y,
                            uint32_t format);

void wafer_tx81_peripheral_argmax(uint64_t src, uint64_t value_dst,
                                  uint64_t index_dst, uint32_t kind,
                                  uint32_t elem_count, uint32_t format,
                                  uint32_t lut_elem_count, uint32_t scale,
                                  uint32_t probability, uint32_t rounding_mode);
void wafer_tx81_peripheral_argmin(uint64_t src, uint64_t value_dst,
                                  uint64_t index_dst, uint32_t kind,
                                  uint32_t elem_count, uint32_t format,
                                  uint32_t lut_elem_count, uint32_t scale,
                                  uint32_t probability, uint32_t rounding_mode);
void wafer_tx81_peripheral_bilinear(
    uint64_t src, uint64_t dst, uint32_t kind, uint32_t elem_count,
    uint32_t format, uint32_t src_n, uint32_t src_h, uint32_t src_w,
    uint32_t src_c, uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
    uint32_t dst_c, uint32_t lut_elem_count, uint32_t scale,
    uint32_t probability, uint32_t rounding_mode);
void wafer_tx81_peripheral_lut16(uint64_t src, uint64_t lut, uint64_t dst,
                                 uint32_t kind, uint32_t elem_count,
                                 uint32_t format, uint32_t lut_elem_count,
                                 uint32_t scale, uint32_t probability,
                                 uint32_t rounding_mode);
void wafer_tx81_peripheral_lut32(uint64_t src, uint64_t lut, uint64_t dst,
                                 uint32_t kind, uint32_t elem_count,
                                 uint32_t format, uint32_t lut_elem_count,
                                 uint32_t scale, uint32_t probability,
                                 uint32_t rounding_mode);
void wafer_tx81_peripheral_rand_gen(uint64_t src0, uint64_t src1, uint64_t dst0,
                                    uint64_t dst1, uint64_t dst2, uint32_t kind,
                                    uint32_t elem_count, uint32_t format,
                                    uint32_t lut_elem_count, uint32_t scale,
                                    uint32_t probability,
                                    uint32_t rounding_mode);
void wafer_tx81_peripheral_elem_mask(uint64_t src, uint64_t dst, uint32_t kind,
                                     uint32_t elem_count, uint32_t format,
                                     uint32_t lut_elem_count, uint32_t scale,
                                     uint32_t probability,
                                     uint32_t rounding_mode);

#ifdef __cplusplus
}
#endif

#endif
