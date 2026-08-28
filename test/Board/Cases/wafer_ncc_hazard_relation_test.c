#include "wafer_ncc_hazard_relation.h"

#include <stdint.h>
#include <stdlib.h>

/* Keep test checks active in Release builds where CMake defines NDEBUG. */
#undef assert
#define assert(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      abort();                                                                 \
  } while (0)

static const WaferNccHazardRange slot = {UINT64_C(0x1000), UINT64_C(0x2200)};
static const WaferNccHazardRange first_baseline = {UINT64_C(0x1200),
                                                   UINT64_C(0x1300)};
static const WaferNccHazardRange second_baseline = {UINT64_C(0x1800),
                                                    UINT64_C(0x1900)};

static WaferNccHazardLaneMemory lane_with_operand(uint32_t operand,
                                                  WaferNccHazardRange range) {
  WaferNccHazardLaneMemory lane = {0};
  uint32_t effects[] = {WAFER_NCC_MEMORY_READ0, WAFER_NCC_MEMORY_READ1,
                        WAFER_NCC_MEMORY_WRITE};
  lane.effects = effects[operand];
  lane.operands[operand].baseline = range;
  return lane;
}

static void relation_lanes(uint32_t relation, WaferNccHazardLaneMemory *first,
                           WaferNccHazardLaneMemory *second,
                           uint32_t *first_operand, uint32_t *second_operand) {
  switch (relation) {
  case WAFER_NCC_EFFECT_RAW:
    *first_operand = WAFER_NCC_HAZARD_OPERAND_WRITE;
    *second_operand = WAFER_NCC_HAZARD_OPERAND_READ0;
    break;
  case WAFER_NCC_EFFECT_WAR:
    *first_operand = WAFER_NCC_HAZARD_OPERAND_READ0;
    *second_operand = WAFER_NCC_HAZARD_OPERAND_WRITE;
    break;
  case WAFER_NCC_EFFECT_WAW:
    *first_operand = WAFER_NCC_HAZARD_OPERAND_WRITE;
    *second_operand = WAFER_NCC_HAZARD_OPERAND_WRITE;
    break;
  case WAFER_NCC_EFFECT_RAR:
    *first_operand = WAFER_NCC_HAZARD_OPERAND_READ0;
    *second_operand = WAFER_NCC_HAZARD_OPERAND_READ0;
    break;
  default:
    abort();
  }
  *first = lane_with_operand(*first_operand, first_baseline);
  *second = lane_with_operand(*second_operand, second_baseline);
}

static void assert_range(WaferNccHazardRange actual, uint64_t begin,
                         uint64_t end) {
  assert(actual.begin == begin);
  assert(actual.end == end);
}

static WaferNccHazardComposition build(uint32_t effect_relation,
                                       uint32_t range_relation) {
  WaferNccHazardLaneMemory first;
  WaferNccHazardLaneMemory second;
  uint32_t first_operand;
  uint32_t second_operand;
  relation_lanes(effect_relation, &first, &second, &first_operand,
                 &second_operand);
  WaferNccHazardSelection selection = {
      WAFER_NCC_HAZARD_OPERAND_AUTO,
      WAFER_NCC_HAZARD_OPERAND_AUTO,
  };
  WaferNccHazardComposition composition;
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, effect_relation, range_relation, selection, slot,
             UINT64_C(0x40), &composition) == WAFER_NCC_HAZARD_OK);
  assert(composition.first_operand == first_operand);
  assert(composition.second_operand == second_operand);
  return composition;
}

static void test_effect_relation_matrix(void) {
  const uint32_t effect_relations[] = {
      WAFER_NCC_EFFECT_RAW,
      WAFER_NCC_EFFECT_WAR,
      WAFER_NCC_EFFECT_WAW,
      WAFER_NCC_EFFECT_RAR,
  };
  const uint32_t range_relations[] = {
      WAFER_NCC_RANGE_EXACT,
      WAFER_NCC_RANGE_PARTIAL,
      WAFER_NCC_RANGE_ADJACENT,
  };
  for (uint32_t effect = 0; effect < 4; ++effect) {
    for (uint32_t relation = 0; relation < 3; ++relation) {
      WaferNccHazardComposition composition =
          build(effect_relations[effect], range_relations[relation]);
      assert_range(composition.first_range, UINT64_C(0x1200), UINT64_C(0x1300));
      if (range_relations[relation] == WAFER_NCC_RANGE_EXACT) {
        assert_range(composition.second_range, UINT64_C(0x1200),
                     UINT64_C(0x1300));
        assert_range(composition.overlap, UINT64_C(0x1200), UINT64_C(0x1300));
      } else if (range_relations[relation] == WAFER_NCC_RANGE_PARTIAL) {
        assert_range(composition.second_range, UINT64_C(0x1280),
                     UINT64_C(0x1380));
        assert_range(composition.overlap, UINT64_C(0x1280), UINT64_C(0x1300));
      } else {
        assert_range(composition.second_range, UINT64_C(0x1300),
                     UINT64_C(0x1400));
        assert(composition.overlap.begin == composition.overlap.end);
      }
      assert(composition.canary_before.end == composition.footprint.begin);
      assert(composition.canary_after.begin == composition.footprint.end);
      assert(composition.canary_before.end - composition.canary_before.begin ==
             UINT64_C(0x40));
      assert(composition.canary_after.end - composition.canary_after.begin ==
             UINT64_C(0x40));
    }
  }
}

static void test_expected_composition(void) {
  WaferNccHazardComposition raw =
      build(WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_PARTIAL);
  assert(raw.second_read_span_count == 2);
  assert_range(raw.second_read_spans[0].range, UINT64_C(0x1280),
               UINT64_C(0x1300));
  assert(raw.second_read_spans[0].source ==
         WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE);
  assert_range(raw.second_read_spans[1].range, UINT64_C(0x1300),
               UINT64_C(0x1380));
  assert(raw.second_read_spans[1].source == WAFER_NCC_HAZARD_EXPECTED_INITIAL);
  assert(raw.final_span_count == 2);
  assert(raw.final_spans[0].source == WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE);
  assert(raw.final_spans[1].source == WAFER_NCC_HAZARD_EXPECTED_INITIAL);

  WaferNccHazardComposition war =
      build(WAFER_NCC_EFFECT_WAR, WAFER_NCC_RANGE_PARTIAL);
  assert(war.second_read_span_count == 0);
  assert(war.final_span_count == 2);
  assert(war.final_spans[0].source == WAFER_NCC_HAZARD_EXPECTED_INITIAL);
  assert(war.final_spans[1].source == WAFER_NCC_HAZARD_EXPECTED_SECOND_WRITE);

  WaferNccHazardComposition waw =
      build(WAFER_NCC_EFFECT_WAW, WAFER_NCC_RANGE_PARTIAL);
  assert(waw.final_span_count == 2);
  assert_range(waw.final_spans[0].range, UINT64_C(0x1200), UINT64_C(0x1280));
  assert(waw.final_spans[0].source == WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE);
  assert_range(waw.final_spans[1].range, UINT64_C(0x1280), UINT64_C(0x1380));
  assert(waw.final_spans[1].source == WAFER_NCC_HAZARD_EXPECTED_SECOND_WRITE);

  WaferNccHazardComposition rar =
      build(WAFER_NCC_EFFECT_RAR, WAFER_NCC_RANGE_EXACT);
  assert(rar.second_read_span_count == 1);
  assert(rar.second_read_spans[0].source == WAFER_NCC_HAZARD_EXPECTED_INITIAL);
  assert(rar.final_span_count == 1);
  assert(rar.final_spans[0].source == WAFER_NCC_HAZARD_EXPECTED_INITIAL);
}

static void test_disjoint_is_identity(void) {
  WaferNccHazardComposition composition =
      build(WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_DISJOINT);
  assert_range(composition.second_range, second_baseline.begin,
               second_baseline.end);
  assert(composition.overlap.begin == composition.overlap.end);
  assert(composition.second_read_span_count == 1);
  assert(composition.second_read_spans[0].source ==
         WAFER_NCC_HAZARD_EXPECTED_INITIAL);
}

static void test_adjacent_is_a_non_alias_control(void) {
  const uint32_t effect_relations[] = {
      WAFER_NCC_EFFECT_RAW,
      WAFER_NCC_EFFECT_WAR,
      WAFER_NCC_EFFECT_WAW,
      WAFER_NCC_EFFECT_RAR,
  };
  for (uint32_t index = 0; index < 4; ++index) {
    WaferNccHazardComposition composition =
        build(effect_relations[index], WAFER_NCC_RANGE_ADJACENT);
    assert(composition.first_range.end == composition.second_range.begin);
    assert(composition.overlap.begin == composition.overlap.end);
    for (uint32_t span = 0; span < composition.second_read_span_count; ++span)
      assert(composition.second_read_spans[span].source ==
             WAFER_NCC_HAZARD_EXPECTED_INITIAL);
  }
}

static void test_strided_envelope_is_not_segmented(void) {
  WaferNccHazardLaneMemory first = lane_with_operand(
      WAFER_NCC_HAZARD_OPERAND_WRITE,
      (WaferNccHazardRange){UINT64_C(0x1200), UINT64_C(0x1380)});
  WaferNccHazardLaneMemory second = lane_with_operand(
      WAFER_NCC_HAZARD_OPERAND_READ0,
      (WaferNccHazardRange){UINT64_C(0x1300), UINT64_C(0x1480)});
  first.operands[WAFER_NCC_HAZARD_OPERAND_WRITE].flags =
      WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE;
  second.operands[WAFER_NCC_HAZARD_OPERAND_READ0].flags =
      WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE;
  WaferNccHazardSelection selection = {
      WAFER_NCC_HAZARD_OPERAND_AUTO,
      WAFER_NCC_HAZARD_OPERAND_AUTO,
  };
  WaferNccHazardComposition composition;
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW,
             WAFER_NCC_RANGE_STRIDED_ENVELOPE, selection, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_OK);
  assert_range(composition.second_range, UINT64_C(0x1300), UINT64_C(0x1480));
  assert_range(composition.overlap, UINT64_C(0x1300), UINT64_C(0x1380));
  assert(composition.flags == WAFER_NCC_HAZARD_COMPOSITION_ENVELOPE_ONLY);
  assert(composition.second_read_span_count == 0);
  assert(composition.final_span_count == 0);
  assert_range(composition.canary_before, UINT64_C(0x11c0), UINT64_C(0x1200));
  assert_range(composition.canary_after, UINT64_C(0x1480), UINT64_C(0x14c0));

  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW,
             WAFER_NCC_RANGE_STRIDED_ENVELOPE, selection,
             (WaferNccHazardRange){UINT64_C(0x1000), UINT64_C(0x14a0)},
             UINT64_C(0x40), &composition) == WAFER_NCC_HAZARD_OUT_OF_SLOT);

  first.operands[WAFER_NCC_HAZARD_OPERAND_WRITE].flags = 0;
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW,
             WAFER_NCC_RANGE_STRIDED_ENVELOPE, selection, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION);
}

static void test_explicit_read_operand_and_fail_closed_effects(void) {
  WaferNccHazardLaneMemory first =
      lane_with_operand(WAFER_NCC_HAZARD_OPERAND_WRITE, first_baseline);
  WaferNccHazardLaneMemory second = {0};
  second.effects = WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1;
  second.operands[WAFER_NCC_HAZARD_OPERAND_READ0].baseline =
      (WaferNccHazardRange){UINT64_C(0x1600), UINT64_C(0x1700)};
  second.operands[WAFER_NCC_HAZARD_OPERAND_READ1].baseline = second_baseline;
  WaferNccHazardComposition composition;
  WaferNccHazardSelection ambiguous = {
      WAFER_NCC_HAZARD_OPERAND_AUTO,
      WAFER_NCC_HAZARD_OPERAND_AUTO,
  };
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_EXACT,
             ambiguous, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT);

  WaferNccHazardSelection read1 = {
      WAFER_NCC_HAZARD_OPERAND_WRITE,
      WAFER_NCC_HAZARD_OPERAND_READ1,
  };
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_EXACT,
             read1, slot, UINT64_C(0x40), &composition) == WAFER_NCC_HAZARD_OK);
  assert(composition.second_operand == WAFER_NCC_HAZARD_OPERAND_READ1);

  first = lane_with_operand(WAFER_NCC_HAZARD_OPERAND_READ0, first_baseline);
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_EXACT,
             ambiguous, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT);
  first.effects |= UINT32_C(1) << 31;
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAR, WAFER_NCC_RANGE_EXACT,
             ambiguous, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT);
}

static void test_slot_alias_and_canary_validation(void) {
  WaferNccHazardLaneMemory first =
      lane_with_operand(WAFER_NCC_HAZARD_OPERAND_WRITE, first_baseline);
  WaferNccHazardLaneMemory second =
      lane_with_operand(WAFER_NCC_HAZARD_OPERAND_READ0, second_baseline);
  WaferNccHazardSelection selection = {
      WAFER_NCC_HAZARD_OPERAND_AUTO,
      WAFER_NCC_HAZARD_OPERAND_AUTO,
  };
  WaferNccHazardComposition composition;

  second.operands[WAFER_NCC_HAZARD_OPERAND_READ0].baseline.end =
      UINT64_C(0x2300);
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_DISJOINT,
             selection, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_OUT_OF_SLOT);

  second = lane_with_operand(
      WAFER_NCC_HAZARD_OPERAND_READ0,
      (WaferNccHazardRange){UINT64_C(0x1000), UINT64_C(0x1100)});
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_ADJACENT,
             selection,
             (WaferNccHazardRange){UINT64_C(0x1000), UINT64_C(0x1340)},
             UINT64_C(0x40), &composition) == WAFER_NCC_HAZARD_OUT_OF_SLOT);

  second = lane_with_operand(WAFER_NCC_HAZARD_OPERAND_READ0, second_baseline);
  first.effects |= WAFER_NCC_MEMORY_READ0;
  first.operands[WAFER_NCC_HAZARD_OPERAND_READ0].baseline =
      (WaferNccHazardRange){UINT64_C(0x1180), UINT64_C(0x11e0)};
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_EXACT,
             selection, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_CANARY_CONFLICT);

  first.operands[WAFER_NCC_HAZARD_OPERAND_READ0].baseline =
      (WaferNccHazardRange){UINT64_C(0x1280), UINT64_C(0x12c0)};
  assert(wafer_ncc_hazard_build_composition(
             &first, &second, WAFER_NCC_EFFECT_RAW, WAFER_NCC_RANGE_EXACT,
             selection, slot, UINT64_C(0x40),
             &composition) == WAFER_NCC_HAZARD_UNINTENDED_ALIAS);
}

static void test_observe_is_the_inclusive_boundary(void) {
  uint64_t begin = 0;
  uint64_t inclusive_end = 0;
  assert(wafer_ncc_hazard_observe_inclusive_range(
             (WaferNccHazardRange){UINT64_C(0x1200), UINT64_C(0x1300)}, &begin,
             &inclusive_end) == WAFER_NCC_HAZARD_OK);
  assert(begin == UINT64_C(0x1200));
  assert(inclusive_end == UINT64_C(0x12ff));
  assert(wafer_ncc_hazard_observe_inclusive_range(
             (WaferNccHazardRange){UINT64_C(0x1200), UINT64_C(0x1200)}, &begin,
             &inclusive_end) == WAFER_NCC_HAZARD_INVALID_ARGUMENT);
}

int main(void) {
  test_effect_relation_matrix();
  test_expected_composition();
  test_disjoint_is_identity();
  test_adjacent_is_a_non_alias_control();
  test_strided_envelope_is_not_segmented();
  test_explicit_read_operand_and_fail_closed_effects();
  test_slot_alias_and_canary_validation();
  test_observe_is_the_inclusive_boundary();
  return 0;
}
