#ifndef WAFER_NCC_HAZARD_RELATION_H
#define WAFER_NCC_HAZARD_RELATION_H

#include "wafer_ncc_probe_protocol.h"

#include <stdint.h>

/*
 * Generic address-relation support for NCC calibration probes.
 *
 * All ranges in this contract are half-open.  Engine adapters may convert a
 * non-empty range to the packet's inclusive end only through
 * wafer_ncc_hazard_observe_inclusive_range().
 */

#define WAFER_NCC_HAZARD_MAX_EXPECTED_SPANS 3U

enum WaferNccHazardOperand {
  WAFER_NCC_HAZARD_OPERAND_READ0 = WAFER_NCC_OPERAND_READ0,
  WAFER_NCC_HAZARD_OPERAND_READ1 = WAFER_NCC_OPERAND_READ1,
  WAFER_NCC_HAZARD_OPERAND_WRITE = WAFER_NCC_OPERAND_WRITE,
  WAFER_NCC_HAZARD_OPERAND_AUTO = WAFER_NCC_OPERAND_AUTO,
};

enum WaferNccHazardOperandFlag {
  /*
   * The range is a caller-provided conservative envelope for a strided
   * operand.  It does not assert that every byte in the envelope is touched.
   */
  WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE = UINT32_C(1) << 0,
};

enum WaferNccHazardCompositionFlag {
  WAFER_NCC_HAZARD_COMPOSITION_ENVELOPE_ONLY = UINT32_C(1) << 0,
};

enum WaferNccHazardExpectedSource {
  WAFER_NCC_HAZARD_EXPECTED_INITIAL = 0,
  WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE = 1,
  WAFER_NCC_HAZARD_EXPECTED_SECOND_WRITE = 2,
};

enum WaferNccHazardStatus {
  WAFER_NCC_HAZARD_OK = 0,
  WAFER_NCC_HAZARD_INVALID_ARGUMENT = 1,
  WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT = 2,
  WAFER_NCC_HAZARD_INVALID_RANGE = 3,
  WAFER_NCC_HAZARD_OUT_OF_SLOT = 4,
  WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION = 5,
  WAFER_NCC_HAZARD_UNINTENDED_ALIAS = 6,
  WAFER_NCC_HAZARD_CANARY_CONFLICT = 7,
};

typedef struct WaferNccHazardRange {
  uint64_t begin;
  uint64_t end;
} WaferNccHazardRange;

typedef struct WaferNccHazardOperandRange {
  WaferNccHazardRange baseline;
  uint32_t flags;
} WaferNccHazardOperandRange;

typedef struct WaferNccHazardLaneMemory {
  uint32_t effects;
  WaferNccHazardOperandRange operands[3];
} WaferNccHazardLaneMemory;

typedef struct WaferNccHazardSelection {
  uint32_t first_operand;
  uint32_t second_operand;
} WaferNccHazardSelection;

typedef struct WaferNccHazardExpectedSpan {
  WaferNccHazardRange range;
  uint32_t source;
} WaferNccHazardExpectedSpan;

typedef struct WaferNccHazardComposition {
  uint32_t first_operand;
  uint32_t second_operand;
  uint32_t flags;
  WaferNccHazardRange first_range;
  WaferNccHazardRange second_range;
  WaferNccHazardRange overlap;
  WaferNccHazardRange footprint;
  WaferNccHazardRange canary_before;
  WaferNccHazardRange canary_after;

  /*
   * Expected source of bytes observed by a second-lane read after lane zero
   * has issued.  Empty for a second-lane write.
   */
  uint32_t second_read_span_count;
  WaferNccHazardExpectedSpan
      second_read_spans[WAFER_NCC_HAZARD_MAX_EXPECTED_SPANS];

  /*
   * Expected final byte ownership over footprint after lane zero followed by
   * lane one.  Sources are symbolic because instruction-specific numeric
   * golden functions remain adapter-owned.
   */
  uint32_t final_span_count;
  WaferNccHazardExpectedSpan final_spans[WAFER_NCC_HAZARD_MAX_EXPECTED_SPANS];
} WaferNccHazardComposition;

/*
 * Resolve one requested two-lane effect relation and materialize the concrete
 * range used by the second lane.  slot is the complete allocation available
 * to this calibration pair; every declared operand, relocated operand and
 * canary must remain inside it.
 *
 * AUTO is accepted only when the required operand is unique.  In particular,
 * a lane exposing both read0 and read1 must select one explicitly.
 */
uint32_t wafer_ncc_hazard_build_composition(
    const WaferNccHazardLaneMemory *first,
    const WaferNccHazardLaneMemory *second, uint32_t effect_relation,
    uint32_t range_relation, WaferNccHazardSelection selection,
    WaferNccHazardRange slot, uint64_t canary_bytes,
    WaferNccHazardComposition *composition);

/*
 * The only conversion supplied for packet-observation fields.  Planning,
 * relocation, alias validation and expected composition remain half-open.
 */
uint32_t wafer_ncc_hazard_observe_inclusive_range(WaferNccHazardRange range,
                                                  uint64_t *begin,
                                                  uint64_t *inclusive_end);

#endif
