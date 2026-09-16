// SDK call interception: verify exact byte geometry at the hardware boundary.
#include "instr_adapter.h"
#include <assert.h>

static uint64_t expected_src, expected_dst;
static uint32_t expected_inner, expected_stride[3], expected_iteration[3];
static uint32_t packet_bits, calls, live_objects;

static void capture_addresses(DMA_Param *instr, uint64_t src, uint64_t dst,
                              Data_Format format) {
  (void)instr;
  assert(src == expected_src && dst == expected_dst);
  // The pinned SDK get_dma_reg_dtype maps values >7 to INT8. Decode the
  // register format independently of the CRT's logical element-size helper.
  static const uint32_t register_bits[] = {8, 16, 16, 16, 32, 32, 32, 1};
  packet_bits = register_bits[format <= Fmt_BOOL ? format : Fmt_INT8];
}

static void capture_geometry(DMA_Param *instr, uint32_t count,
                             uint32_t stride0, uint32_t iteration0,
                             uint32_t stride1, uint32_t iteration1,
                             uint32_t stride2, uint32_t iteration2) {
  (void)instr;
  assert((uint64_t)count * packet_bits == (uint64_t)expected_inner * 8);
  const uint32_t strides[] = {stride0, stride1, stride2};
  const uint32_t iterations[] = {iteration0, iteration1, iteration2};
  for (unsigned i = 0; i < 3; ++i) {
    assert((uint64_t)strides[i] * packet_bits ==
           (uint64_t)expected_stride[i] * 8);
    assert(iterations[i] == expected_iteration[i]);
  }
}

static TsmRdma rdma = {.AddSrcDst = capture_addresses,
                      .ConfigStrideIteration = capture_geometry};
static TsmWdma wdma = {.AddSrcDst = capture_addresses,
                      .ConfigStrideIteration = capture_geometry};
TsmRdma *TsmNewRdma(void) { ++live_objects; return &rdma; }
TsmWdma *TsmNewWdma(void) { ++live_objects; return &wdma; }
void TsmDeleteRdma(TsmRdma *p) { assert(p == &rdma); --live_objects; }
void TsmDeleteWdma(TsmWdma *p) { assert(p == &wdma); --live_objects; }
static void wafer_execute_rdma(TsmRdmaInstr *instr, uint32_t worker) {
  (void)instr;
  assert(worker == 2);
  ++calls;
}
static void wafer_execute_wdma(TsmWdmaInstr *instr, uint32_t worker) {
  (void)instr;
  assert(worker == 2);
  ++calls;
}

/* PRODUCTION_FUNCTIONS */

int main(void) {
  static const uint32_t logical_bits[] = {
      8, 16, 16, 16, 32, 32, 32, 1, 8, 16, 32, 64, 64};
  const uint32_t extents[] = {1024, 1025, 1031};
  for (unsigned format = 0; format < 13; ++format)
    for (unsigned shape = 0; shape < 3; ++shape)
      for (unsigned strided = 0; strided < 2; ++strided)
        for (unsigned direction = 0; direction < 2; ++direction) {
          // Dense rank-3 [2,3,N] or a rank-4 [2,3,4,N] strided view.
          // Packed BOOL includes its final owned byte.
          uint32_t row_bytes = (extents[shape] * logical_bits[format] + 7) / 8;
          expected_inner = strided ? row_bytes : row_bytes * 6;
          expected_stride[0] = strided ? row_bytes + 256 : 0;
          expected_stride[1] = strided ? expected_stride[0] * 4 + 512 : 0;
          expected_stride[2] = strided ? expected_stride[1] * 3 + 1024 : 0;
          expected_iteration[0] = strided ? 4 : 1;
          expected_iteration[1] = strided ? 3 : 1;
          expected_iteration[2] = strided ? 2 : 1;
          expected_src = direction ? 0x12300 : UINT64_C(0x280004000);
          expected_dst = direction ? UINT64_C(0x280008000) : 0x45600;
          uint32_t before = calls;
          (direction ? wafer_tx81_wdma : wafer_tx81_rdma)(
              expected_src, expected_dst, strided ? row_bytes * 24 : expected_inner,
              expected_inner, expected_stride[0], expected_stride[1],
              expected_stride[2], expected_iteration[0], expected_iteration[1],
              expected_iteration[2], format, 2);
          assert(calls == before + 1 && live_objects == 0);
        }
  assert(calls == 156);
  puts("DMA packet byte geometry: 156 configurations passed");
  return 0;
}
