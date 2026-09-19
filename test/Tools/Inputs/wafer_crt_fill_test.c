// SDK interception verifies the production ABI, not unqualified CT hardware
// arithmetic/tail behavior. Those still require a fresh device qualification.
#include "instr_adapter.h"
#include <assert.h>

static uint64_t destination;
static uint32_t scalar, count, worker, calls, constructed, live_objects;
static Data_Format format;

static void capture_xor(TsmLogicInstr *p, uint64_t a, uint64_t b, uint64_t dst,
                        uint32_t n, Data_Format f) {
  assert(constructed == 0 && calls == 0);
  assert(a == destination && b == destination && dst == destination);
  assert(n == count && f == format);
  p->ctrl.opcode = 81;
  ++constructed;
}

static void capture_add(TsmArithInstr *p, uint64_t a, uint32_t value,
                        uint64_t dst, uint32_t n, RND_MODE rounding,
                        Data_Format f) {
  assert(constructed == 1 && calls == 1);
  assert(a == destination && dst == destination && value == scalar);
  assert(n == count && f == format && rounding == RND_NEAREST_EVEN);
  p->ctrl.opcode = 15;
  ++constructed;
}

static TsmLogic logic = {.XorVV = capture_xor};
static TsmArith arith = {.AddVS = capture_add};
TsmLogic *TsmNewLogic(void) { ++live_objects; return &logic; }
TsmArith *TsmNewArith(void) { ++live_objects; return &arith; }
void TsmDeleteLogic(TsmLogic *p) { assert(p == &logic); --live_objects; }
void TsmDeleteArith(TsmArith *p) { assert(p == &arith); --live_objects; }
static void wafer_execute_ct(CT_Param *p, uint32_t w) {
  assert(w == worker && calls < 2);
  assert(p->ctrl.opcode == (calls == 0 ? 81 : 15));
  ++calls;
}

/* PRODUCTION_FUNCTIONS */

int main(void) {
  const Data_Format storage[] = {
      Fmt_INT8, Fmt_INT16, Fmt_INT16, Fmt_INT16, Fmt_INT32, Fmt_INT32,
      Fmt_INT32, Fmt_INT8, Fmt_INT8, Fmt_INT16, Fmt_INT32,
      Fmt_INT64, Fmt_INT64};
  // Rank-3 [2,3,N], the observed 65,536-element fill, and scalar/strided-row
  // boundaries. Tiny counts are intentional ABI boundary witnesses.
  const uint32_t elements[] = {6*1024, 6*1025, 6*1031, 65536, 1, 64, 130};
  // Raw +/-zero, infinities, subnormal and NaN payloads plus integer edges.
  const uint32_t values[] = {0, 1, 0x8000, 0x7c00, 0xfc00, 0x7e35,
                             0x80000000, 0x7f800000, 0xff800000,
                             0x7fc01234, 0x7f801234, 0xffffffff};
  unsigned configurations = 0;
  for (unsigned f = 0; f < 13; ++f)
    for (unsigned n = 0; n < sizeof(elements)/sizeof(elements[0]); ++n)
      for (unsigned v = 0; v < sizeof(values)/sizeof(values[0]); ++v)
        for (worker = 0; worker < 3; ++worker) {
          destination = 0x30006; // Also preserve a non-256B-aligned subview.
          count = f == Fmt_BOOL ? (elements[n] + 7)/8 : elements[n];
          format = storage[f];
          uint32_t value = f == Fmt_BOOL ? values[v] != 0
                           : format == Fmt_INT8 ? values[v] & 0xff
                           : format == Fmt_INT16 ? values[v] & 0xffff
                                                : values[v];
          scalar = f == Fmt_BOOL ? (value ? 255 : 0) : value;
          constructed = calls = 0;
          // The compiler's BOOL domain owns complete bytes, including tail
          // bits; its ABI count is consequently always a multiple of eight.
          uint32_t input_count = f == Fmt_BOOL ? count * 8 : elements[n];
          wafer_tx81_memset(destination, value, input_count, f, worker);
          assert(calls == 2 && constructed == 2 && live_objects == 0);
          ++configurations;
        }
  calls = constructed = 0;
  wafer_tx81_memset(destination, 0, 0, Fmt_FP32, 0);
  wafer_tx81_memset(destination, 0, UINT32_MAX, Fmt_FP32, 0);
  wafer_tx81_memset(destination, 0, 1024, UINT32_MAX, 0);
  assert(calls == 0 && constructed == 0 && live_objects == 0);
  printf("CT fill: %u configurations, two whole-range issues each\n", configurations);
}
