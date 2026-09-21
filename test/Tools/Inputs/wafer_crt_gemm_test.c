// Setter mocks preserve the public SDK schema; the production CRT computes
// and emits every register. Golden Cx extents are independent fixed values.
#include "Wafer/ABI/Tx81NCCABI.h"
#include "instr_adapter.h"
#include "instr_operator.h"
#include <assert.h>

static uint64_t expected[31];
static uint32_t seen, worker, writes, cases;

static void wafer_write_ne_register(uint64_t base, uint32_t offset,
                                    uint64_t value) {
  assert(base == UINT64_C(0x1000000) + ((uint64_t)worker << 20));
  assert(offset >= 0x200 && offset <= 0x3e0 && offset % 16 == 0);
  unsigned slot = (offset - 0x200) / 16;
  assert(!(seen & (1U << slot)) && value == expected[slot]);
  if (offset == 0x200)
    assert(seen == 0x7ffffffeU); // The command becomes valid only at the end.
  seen |= 1U << slot;
  ++writes;
}

static void set_input(TsmNeInstr *i, uint64_t a, uint64_t b, Data_Format f) {
  i->inter_type = I_NEUR;
  i->ctrl.type = 3;
  i->ctrl.input_format = f;
  i->param.src_a = a;
  i->param.src_w = b;
}
static void set_shape(TsmNeInstr *i, uint32_t m, uint32_t k, uint32_t n) {
  i->param.gemm_m = m;
  i->param.gemm_k = k;
  i->param.gemm_n = n;
}
static void set_batch(TsmNeInstr *i, uint32_t a, uint32_t b) {
  i->param.gemm_lb = a;
  i->param.gemm_rb = b;
}
static void set_output(TsmNeInstr *i, uint64_t a, Data_Format f) {
  i->param.out = a;
  i->ctrl.output_format = f;
}
static void set_psum(TsmNeInstr *i, uint8_t en, uint64_t a, Data_Format f) {
  i->ctrl.inpsum_en = en;
  i->param.psum = a;
  i->ctrl.inpsum_format = f;
}
static void set_transpose(TsmNeInstr *i, uint8_t a, uint8_t b) {
  i->param.gemm_l_trs = a;
  i->param.gemm_r_trs = b;
}
static void set_quant(TsmNeInstr *i, uint8_t a, uint8_t b, uint8_t c,
                      uint8_t d) {
  (void)i;
  assert(a == 0 && b == 0 && c == 0 && d == 0);
}
static void disable_optional(TsmNeInstr *i, uint8_t en, uint64_t a) {
  (void)i;
  assert(en == 0 && a == 0);
}
static void disable_activation(TsmNeInstr *i) { (void)i; }
static TsmGemm gemm = {
    .AddInput = set_input,
    .ConfigMKN = set_shape,
    .ConfigBatch = set_batch,
    .AddOutput = set_output,
    .SetPsum = set_psum,
    .SetTransflag = set_transpose,
    .SetQuant = set_quant,
    .AddBias = disable_optional,
    .SetNegativeAxisScale = disable_optional,
    .SetPositiveAxisScale = disable_optional,
    .DisableRelu = disable_activation,
    .DisableLeakyRelu = disable_activation,
};

static TsmOperatorPointer operators = {
    .gemm_pointer = &gemm,
};
TsmOperatorPointer *g_intrinsic(void) { return &operators; }

/* PRODUCTION_FUNCTIONS */

// Inputs pair each logical extent with its independently specified padded Cx
// extent. Matrix storage includes all internal padding and each batch suffix.
typedef struct {
  uint32_t logical, physical;
} Extent;
static uint64_t span(uint32_t rows, uint32_t padded_columns, uint32_t bytes,
                     uint32_t batch) {
  uint64_t units = ((uint64_t)rows * padded_columns * bytes + 255) / 256;
  return units * 256 * batch;
}
static void check(Extent m, Extent k, Extent n, uint32_t batch, uint32_t input,
                  uint32_t output, uint32_t psum, uint32_t ta, uint32_t tb,
                  uint32_t ordinary) {
  memset(expected, 0, sizeof(expected));
  seen = writes = 0;
  // Disjoint real SPM slots include internal padding and each batch suffix.
  expected[0] = 0x100003U | (psum << 16) | (output << 12) | (input << 8) |
                (psum != Fmt_UNUSED ? 128U : 0U);
  expected[1] = 0x10000;
  expected[2] = 0x90000;
  expected[3] = psum == Fmt_UNUSED ? 0 : 0x190000;
  expected[7] = 0x210000;
  expected[14] = expected[15] = batch;
  expected[16] = n.logical;
  expected[17] = m.logical;
  expected[18] = k.logical;
  expected[19] = ta;
  expected[20] = !tb;
  expected[23] =
      expected[1] - 1 +
      span(ta ? k.logical : m.logical, ta ? m.physical : k.physical, 2, batch);
  expected[24] =
      expected[2] - 1 +
      span(tb ? n.logical : k.logical, tb ? k.physical : n.physical, 2, batch);
  expected[25] = psum == Fmt_UNUSED
                     ? 0
                     : expected[3] - 1 + span(m.logical, n.physical, 4, batch);
  expected[29] = expected[7] - 1 +
                 span(m.logical, n.physical, output == Fmt_FP32 ? 4 : 2, batch);
  if (ordinary)
    wafer_tx81_gemm(expected[1], expected[2], expected[7], expected[3],
                    m.logical, k.logical, n.logical, batch, input, output, psum,
                    worker);
  else
    wafer_tx81_gemm_oriented(expected[1], expected[2], expected[7], expected[3],
                             m.logical, k.logical, n.logical, batch, input,
                             output, psum, ta, tb, worker);
  assert(seen == 0x7fffffffU && writes == 31);
  ++cases;
}

int main(void) {
  const Extent ks[] = {{1024, 1024}, {1025, 1028}, {1031, 1032}};
  const Extent ns[] = {{17, 32}, {43, 64}, {65, 68}};
  const Extent ms[] = {{16, 16}, {17, 32}};
  for (worker = 0; worker < 3; ++worker)
    for (uint32_t input = Fmt_FP16; input <= Fmt_BF16; ++input)
      for (unsigned ki = 0; ki < 3; ++ki)
        for (unsigned ni = 0; ni < 3; ++ni)
          for (unsigned mi = 0; mi < 2; ++mi)
            for (unsigned wide = 0; wide < 2; ++wide)
              for (unsigned partial = 0; partial < 2; ++partial)
                for (unsigned orientation = 0; orientation < 5; ++orientation)
                  check(
                      ms[mi], ks[ki], ns[ni], 2, input, wide ? Fmt_FP32 : input,
                      partial ? Fmt_FP32 : Fmt_UNUSED,
                      orientation < 4 ? orientation / 2 : 0,
                      orientation < 4 ? orientation % 2 : 0, orientation == 4);
  // Bounded reproducer of the actual SDK field error, alongside rank-3
  // production-scale cases above. Four-byte output/psum must span 4096B.
  worker = 0;
  check((Extent){16, 16}, (Extent){384, 384}, (Extent){43, 64}, 1, Fmt_FP16,
        Fmt_FP32, Fmt_FP32, 0, 1, 0);
  assert(expected[29] - expected[7] + 1 == 4096);
  assert(expected[25] - expected[3] + 1 == 4096);
  assert(cases == 2161);
  printf("GEMM final registers: %u configurations passed\n", cases);
  return 0;
}
