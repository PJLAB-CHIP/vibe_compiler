// Public SDK dispatch oracle for the real CRT macro. It checks the selected
// entry, every argument, one CT issue, worker and factory lifetime. Hardware
// output/tail behavior is qualified separately on a recovered device.
#include "instr_adapter.h"
#include "wafer_tx81_crt.h"
#include <assert.h>
#include <stdio.h>

static unsigned expected_method, expected_form, calls, live_objects;
static uint32_t count, unit, scalar, worker;
static Data_Format format;
static const uint64_t lhs = 0x20000, rhs = 0x90000, destination = 0xa0000;

static void capture(TsmRelationInstr *p, unsigned method, unsigned form,
                    uint64_t a, uint64_t b, uint64_t dst, uint32_t n,
                    uint32_t u, Data_Format f) {
  assert(method == expected_method && form == expected_form && calls == 0);
  assert(a == lhs && b == (form == 1 ? scalar : rhs) && dst == destination);
  assert(n == count && u == unit && f == format);
  p->ctrl.opcode = method;
}
#define CAPTURE(NAME, ID)                                                      \
  static void NAME##VV(TsmRelationInstr *p, uint64_t a, uint64_t b,            \
                       uint64_t d, uint32_t n, Data_Format f) {                \
    capture(p, ID, 0, a, b, d, n, 0, f);                                       \
  }                                                                            \
  static void NAME##VS(TsmRelationInstr *p, uint64_t a, uint64_t b,            \
                       uint64_t d, uint32_t n, Data_Format f) {                \
    capture(p, ID, 1, a, b, d, n, 0, f);                                       \
  }                                                                            \
  static void NAME##VuV(TsmRelationInstr *p, uint64_t a, uint64_t b,           \
                        uint64_t d, uint32_t n, uint32_t u, Data_Format f) {   \
    capture(p, ID, 2, a, b, d, n, u, f);                                       \
  }
CAPTURE(BoolEqual, 0)
CAPTURE(Equal, 1) CAPTURE(BoolUnEqual, 2) CAPTURE(UnEqual, 3)
    CAPTURE(BoolGreaterEqual, 4) CAPTURE(GreaterEqual, 5)
        CAPTURE(BoolGreater, 6) CAPTURE(Greater, 7) CAPTURE(BoolLessEqual, 8)
            CAPTURE(LessEqual, 9) CAPTURE(BoolLessThen, 10)
                CAPTURE(LessThen, 11)
#define METHODS(NAME)                                                          \
  .NAME##VV = NAME##VV, .NAME##VS = NAME##VS, .NAME##VuV = NAME##VuV
                    static TsmRelation relation = {
                        METHODS(Equal),        METHODS(BoolEqual),
                        METHODS(UnEqual),      METHODS(BoolUnEqual),
                        METHODS(GreaterEqual), METHODS(BoolGreaterEqual),
                        METHODS(Greater),      METHODS(BoolGreater),
                        METHODS(LessEqual),    METHODS(BoolLessEqual),
                        METHODS(LessThen),     METHODS(BoolLessThen)};
TsmRelation *TsmNewRelation(void) {
  ++live_objects;
  return &relation;
}
void TsmDeleteRelation(TsmRelation *p) {
  assert(p == &relation);
  --live_objects;
}
static void wafer_execute_ct(CT_Param *p, uint32_t w) {
  assert(live_objects == 1 && calls == 0 && p->ctrl.opcode == expected_method);
  assert(w == worker);
  ++calls;
}

/* PRODUCTION_FUNCTIONS */

int main(void) {
  typedef void (*Relation)(uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, uint32_t);
  Relation functions[] = {wafer_tx81_elementwise_eq, wafer_tx81_elementwise_ne,
                          wafer_tx81_elementwise_ge, wafer_tx81_elementwise_gt,
                          wafer_tx81_elementwise_le, wafer_tx81_elementwise_lt};
  const Data_Format formats[] = {Fmt_FP16, Fmt_BF16, Fmt_FP32};
  const uint32_t extents[] = {1024, 1025, 1031};
  unsigned configurations = 0;
  for (unsigned op = 0; op < 6; ++op)
    for (unsigned f = 0; f < 3; ++f)
      for (unsigned n = 0; n < 3; ++n)
        for (unsigned form = 0; form < 5; ++form)
          for (unsigned numeric = 0; numeric < 2; ++numeric)
            for (worker = 0; worker < 3; ++worker) {
              format = formats[f];
              count = 2 * 32 * extents[n];
              unit = form < 2 ? 0 : form == 2 ? 1 : form == 3 ? 32 : 64;
              scalar = format == Fmt_FP16   ? 0xfc00
                       : format == Fmt_BF16 ? 0xff80
                                            : 0xff800000;
              expected_method = op * 2 + numeric;
              expected_form = form < 2 ? form : 2;
              calls = live_objects = 0;
              functions[op](lhs, form == 1 ? scalar : rhs, destination, count,
                            format, unit, form == 1, numeric, worker);
              assert(calls == 1 && live_objects == 0);
              ++configurations;
            }
  printf("relation SDK dispatch: %u configurations passed\n", configurations);
}
