#include "wafer_ncc_hazard_relation.h"

#include <stddef.h>

static void wafer_ncc_hazard_zero(void *pointer, uint32_t bytes) {
  uint8_t *output = (uint8_t *)pointer;
  for (uint32_t index = 0; index < bytes; ++index)
    output[index] = 0;
}

static int wafer_ncc_hazard_range_empty(WaferNccHazardRange range) {
  return range.begin >= range.end;
}

static int wafer_ncc_hazard_range_contains(WaferNccHazardRange outer,
                                           WaferNccHazardRange inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

static int wafer_ncc_hazard_range_overlaps(WaferNccHazardRange first,
                                           WaferNccHazardRange second) {
  return first.begin < second.end && second.begin < first.end;
}

static WaferNccHazardRange
wafer_ncc_hazard_intersection(WaferNccHazardRange first,
                              WaferNccHazardRange second) {
  WaferNccHazardRange result;
  result.begin = first.begin > second.begin ? first.begin : second.begin;
  result.end = first.end < second.end ? first.end : second.end;
  if (result.end < result.begin)
    result.end = result.begin;
  return result;
}

static WaferNccHazardRange wafer_ncc_hazard_hull(WaferNccHazardRange first,
                                                 WaferNccHazardRange second) {
  WaferNccHazardRange result;
  result.begin = first.begin < second.begin ? first.begin : second.begin;
  result.end = first.end > second.end ? first.end : second.end;
  return result;
}

static uint32_t wafer_ncc_hazard_operand_effect(uint32_t operand) {
  switch (operand) {
  case WAFER_NCC_HAZARD_OPERAND_READ0:
    return WAFER_NCC_MEMORY_READ0;
  case WAFER_NCC_HAZARD_OPERAND_READ1:
    return WAFER_NCC_MEMORY_READ1;
  case WAFER_NCC_HAZARD_OPERAND_WRITE:
    return WAFER_NCC_MEMORY_WRITE;
  default:
    return 0;
  }
}

static uint32_t
wafer_ncc_hazard_validate_lane(const WaferNccHazardLaneMemory *lane,
                               WaferNccHazardRange slot) {
  const uint32_t all_effects =
      WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1 | WAFER_NCC_MEMORY_WRITE;
  if ((lane->effects & ~all_effects) != 0 || lane->effects == 0)
    return WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT;

  for (uint32_t operand = 0; operand < 3; ++operand) {
    const WaferNccHazardOperandRange *operand_range = &lane->operands[operand];
    uint32_t effect = wafer_ncc_hazard_operand_effect(operand);
    if ((operand_range->flags & ~WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE) !=
        0)
      return WAFER_NCC_HAZARD_INVALID_ARGUMENT;
    if ((lane->effects & effect) == 0) {
      if (operand_range->baseline.begin != 0 ||
          operand_range->baseline.end != 0 || operand_range->flags != 0)
        return WAFER_NCC_HAZARD_INVALID_ARGUMENT;
      continue;
    }
    if (wafer_ncc_hazard_range_empty(operand_range->baseline))
      return WAFER_NCC_HAZARD_INVALID_RANGE;
    if (!wafer_ncc_hazard_range_contains(slot, operand_range->baseline))
      return WAFER_NCC_HAZARD_OUT_OF_SLOT;
  }
  return WAFER_NCC_HAZARD_OK;
}

static uint32_t wafer_ncc_hazard_resolve_operand(
    const WaferNccHazardLaneMemory *lane, uint32_t required_effects,
    uint32_t requested_operand, uint32_t *resolved_operand) {
  if (requested_operand != WAFER_NCC_HAZARD_OPERAND_AUTO) {
    uint32_t effect = wafer_ncc_hazard_operand_effect(requested_operand);
    if (effect == 0 || (effect & required_effects) == 0 ||
        (lane->effects & effect) == 0)
      return WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT;
    *resolved_operand = requested_operand;
    return WAFER_NCC_HAZARD_OK;
  }

  uint32_t count = 0;
  uint32_t result = 0;
  for (uint32_t operand = 0; operand < 3; ++operand) {
    uint32_t effect = wafer_ncc_hazard_operand_effect(operand);
    if ((effect & required_effects) != 0 && (lane->effects & effect) != 0) {
      result = operand;
      ++count;
    }
  }
  if (count != 1)
    return WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT;
  *resolved_operand = result;
  return WAFER_NCC_HAZARD_OK;
}

static uint32_t wafer_ncc_hazard_resolve_effect_pair(
    const WaferNccHazardLaneMemory *first,
    const WaferNccHazardLaneMemory *second, uint32_t relation,
    WaferNccHazardSelection selection, uint32_t *first_operand,
    uint32_t *second_operand) {
  const uint32_t reads = WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1;
  uint32_t first_required = 0;
  uint32_t second_required = 0;
  switch (relation) {
  case WAFER_NCC_EFFECT_RAW:
    first_required = WAFER_NCC_MEMORY_WRITE;
    second_required = reads;
    break;
  case WAFER_NCC_EFFECT_WAR:
    first_required = reads;
    second_required = WAFER_NCC_MEMORY_WRITE;
    break;
  case WAFER_NCC_EFFECT_WAW:
    first_required = WAFER_NCC_MEMORY_WRITE;
    second_required = WAFER_NCC_MEMORY_WRITE;
    break;
  case WAFER_NCC_EFFECT_RAR:
    first_required = reads;
    second_required = reads;
    break;
  default:
    return WAFER_NCC_HAZARD_UNSUPPORTED_EFFECT;
  }

  uint32_t status = wafer_ncc_hazard_resolve_operand(
      first, first_required, selection.first_operand, first_operand);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;
  return wafer_ncc_hazard_resolve_operand(
      second, second_required, selection.second_operand, second_operand);
}

static uint32_t wafer_ncc_hazard_relocate_second(
    WaferNccHazardRange first, WaferNccHazardRange baseline, uint32_t relation,
    uint32_t first_flags, uint32_t second_flags, WaferNccHazardRange *second) {
  uint64_t first_bytes = first.end - first.begin;
  uint64_t second_bytes = baseline.end - baseline.begin;
  *second = baseline;

  if ((first_flags | second_flags) != 0 &&
      relation != WAFER_NCC_RANGE_STRIDED_ENVELOPE)
    return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;

  switch (relation) {
  case WAFER_NCC_RANGE_DISJOINT:
    if (wafer_ncc_hazard_range_overlaps(first, baseline))
      return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
    return WAFER_NCC_HAZARD_OK;
  case WAFER_NCC_RANGE_EXACT:
    if (first_bytes != second_bytes)
      return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
    *second = first;
    return WAFER_NCC_HAZARD_OK;
  case WAFER_NCC_RANGE_PARTIAL: {
    uint64_t smaller = first_bytes < second_bytes ? first_bytes : second_bytes;
    uint64_t overlap_bytes = smaller / 2U;
    if (overlap_bytes == 0)
      return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
    second->begin = first.end - overlap_bytes;
    if (second_bytes > UINT64_MAX - second->begin)
      return WAFER_NCC_HAZARD_INVALID_RANGE;
    second->end = second->begin + second_bytes;
    if (!(first.begin < second->begin && first.end < second->end))
      return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
    return WAFER_NCC_HAZARD_OK;
  }
  case WAFER_NCC_RANGE_ADJACENT:
    second->begin = first.end;
    if (second_bytes > UINT64_MAX - second->begin)
      return WAFER_NCC_HAZARD_INVALID_RANGE;
    second->end = second->begin + second_bytes;
    return WAFER_NCC_HAZARD_OK;
  case WAFER_NCC_RANGE_STRIDED_ENVELOPE:
    if ((first_flags & WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE) == 0 ||
        (second_flags & WAFER_NCC_HAZARD_OPERAND_EXPLICIT_ENVELOPE) == 0 ||
        !wafer_ncc_hazard_range_overlaps(first, baseline))
      return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
    /*
     * The caller already supplied both conservative envelopes.  Preserve the
     * second one exactly; synthesizing segments or relocating it would claim
     * descriptor knowledge that this helper does not own.
     */
    return WAFER_NCC_HAZARD_OK;
  default:
    return WAFER_NCC_HAZARD_UNSUPPORTED_RANGE_RELATION;
  }
}

static void wafer_ncc_hazard_sort_boundaries(uint64_t *values, uint32_t count) {
  for (uint32_t index = 1; index < count; ++index) {
    uint64_t value = values[index];
    uint32_t position = index;
    while (position != 0 && values[position - 1] > value) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
}

static void wafer_ncc_hazard_append_span(WaferNccHazardExpectedSpan *spans,
                                         uint32_t *count,
                                         WaferNccHazardRange range,
                                         uint32_t source) {
  if (wafer_ncc_hazard_range_empty(range))
    return;
  if (*count != 0 && spans[*count - 1].source == source &&
      spans[*count - 1].range.end == range.begin) {
    spans[*count - 1].range.end = range.end;
    return;
  }
  if (*count >= WAFER_NCC_HAZARD_MAX_EXPECTED_SPANS)
    return;
  spans[*count].range = range;
  spans[*count].source = source;
  ++*count;
}

static int wafer_ncc_hazard_segment_in_range(WaferNccHazardRange segment,
                                             WaferNccHazardRange range) {
  return range.begin <= segment.begin && segment.end <= range.end;
}

static void
wafer_ncc_hazard_build_expected(WaferNccHazardComposition *composition) {
  uint64_t boundaries[4] = {
      composition->first_range.begin,
      composition->first_range.end,
      composition->second_range.begin,
      composition->second_range.end,
  };
  wafer_ncc_hazard_sort_boundaries(boundaries, 4);

  if (composition->second_operand != WAFER_NCC_HAZARD_OPERAND_WRITE) {
    for (uint32_t index = 0; index + 1 < 4; ++index) {
      WaferNccHazardRange segment = {boundaries[index], boundaries[index + 1]};
      if (wafer_ncc_hazard_range_empty(segment) ||
          !wafer_ncc_hazard_segment_in_range(segment,
                                             composition->second_range))
        continue;
      uint32_t source = WAFER_NCC_HAZARD_EXPECTED_INITIAL;
      if (composition->first_operand == WAFER_NCC_HAZARD_OPERAND_WRITE &&
          wafer_ncc_hazard_segment_in_range(segment, composition->first_range))
        source = WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE;
      wafer_ncc_hazard_append_span(composition->second_read_spans,
                                   &composition->second_read_span_count,
                                   segment, source);
    }
  }

  for (uint32_t index = 0; index + 1 < 4; ++index) {
    WaferNccHazardRange segment = {boundaries[index], boundaries[index + 1]};
    if (wafer_ncc_hazard_range_empty(segment))
      continue;
    uint32_t source = WAFER_NCC_HAZARD_EXPECTED_INITIAL;
    if (composition->first_operand == WAFER_NCC_HAZARD_OPERAND_WRITE &&
        wafer_ncc_hazard_segment_in_range(segment, composition->first_range))
      source = WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE;
    if (composition->second_operand == WAFER_NCC_HAZARD_OPERAND_WRITE &&
        wafer_ncc_hazard_segment_in_range(segment, composition->second_range))
      source = WAFER_NCC_HAZARD_EXPECTED_SECOND_WRITE;
    wafer_ncc_hazard_append_span(composition->final_spans,
                                 &composition->final_span_count, segment,
                                 source);
  }
}

static uint32_t wafer_ncc_hazard_validate_effective_ranges(
    const WaferNccHazardLaneMemory *first,
    const WaferNccHazardLaneMemory *second, uint32_t first_operand,
    uint32_t second_operand, WaferNccHazardRange second_range,
    WaferNccHazardRange canary_before, WaferNccHazardRange canary_after) {
  WaferNccHazardRange effective[2][3];
  uint32_t effects[2] = {first->effects, second->effects};
  for (uint32_t operand = 0; operand < 3; ++operand) {
    effective[0][operand] = first->operands[operand].baseline;
    effective[1][operand] = second->operands[operand].baseline;
  }
  effective[1][second_operand] = second_range;

  for (uint32_t lane = 0; lane < 2; ++lane) {
    for (uint32_t first_index = 0; first_index < 3; ++first_index) {
      uint32_t first_effect = wafer_ncc_hazard_operand_effect(first_index);
      if ((effects[lane] & first_effect) == 0)
        continue;
      for (uint32_t second_index = first_index + 1; second_index < 3;
           ++second_index) {
        uint32_t second_effect = wafer_ncc_hazard_operand_effect(second_index);
        if ((effects[lane] & second_effect) != 0 &&
            wafer_ncc_hazard_range_overlaps(effective[lane][first_index],
                                            effective[lane][second_index]))
          return WAFER_NCC_HAZARD_UNINTENDED_ALIAS;
      }
    }
  }

  for (uint32_t first_index = 0; first_index < 3; ++first_index) {
    uint32_t first_effect = wafer_ncc_hazard_operand_effect(first_index);
    if ((first->effects & first_effect) == 0)
      continue;
    for (uint32_t second_index = 0; second_index < 3; ++second_index) {
      uint32_t second_effect = wafer_ncc_hazard_operand_effect(second_index);
      if ((second->effects & second_effect) == 0)
        continue;
      if (first_index == first_operand && second_index == second_operand)
        continue;
      if (wafer_ncc_hazard_range_overlaps(effective[0][first_index],
                                          effective[1][second_index]))
        return WAFER_NCC_HAZARD_UNINTENDED_ALIAS;
    }
  }

  for (uint32_t lane = 0; lane < 2; ++lane) {
    for (uint32_t operand = 0; operand < 3; ++operand) {
      uint32_t effect = wafer_ncc_hazard_operand_effect(operand);
      if ((effects[lane] & effect) == 0)
        continue;
      if (wafer_ncc_hazard_range_overlaps(effective[lane][operand],
                                          canary_before) ||
          wafer_ncc_hazard_range_overlaps(effective[lane][operand],
                                          canary_after))
        return WAFER_NCC_HAZARD_CANARY_CONFLICT;
    }
  }
  return WAFER_NCC_HAZARD_OK;
}

uint32_t wafer_ncc_hazard_build_composition(
    const WaferNccHazardLaneMemory *first,
    const WaferNccHazardLaneMemory *second, uint32_t effect_relation,
    uint32_t range_relation, WaferNccHazardSelection selection,
    WaferNccHazardRange slot, uint64_t canary_bytes,
    WaferNccHazardComposition *composition) {
  if (first == NULL || second == NULL || composition == NULL ||
      wafer_ncc_hazard_range_empty(slot) || canary_bytes == 0)
    return WAFER_NCC_HAZARD_INVALID_ARGUMENT;

  uint32_t status = wafer_ncc_hazard_validate_lane(first, slot);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;
  status = wafer_ncc_hazard_validate_lane(second, slot);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;

  uint32_t first_operand = 0;
  uint32_t second_operand = 0;
  status = wafer_ncc_hazard_resolve_effect_pair(first, second, effect_relation,
                                                selection, &first_operand,
                                                &second_operand);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;

  WaferNccHazardRange first_range = first->operands[first_operand].baseline;
  WaferNccHazardRange second_range;
  status = wafer_ncc_hazard_relocate_second(
      first_range, second->operands[second_operand].baseline, range_relation,
      first->operands[first_operand].flags,
      second->operands[second_operand].flags, &second_range);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;
  if (!wafer_ncc_hazard_range_contains(slot, second_range))
    return WAFER_NCC_HAZARD_OUT_OF_SLOT;

  WaferNccHazardRange footprint =
      wafer_ncc_hazard_hull(first_range, second_range);
  if (canary_bytes > footprint.begin - slot.begin ||
      canary_bytes > slot.end - footprint.end)
    return WAFER_NCC_HAZARD_OUT_OF_SLOT;
  WaferNccHazardRange canary_before = {footprint.begin - canary_bytes,
                                       footprint.begin};
  WaferNccHazardRange canary_after = {footprint.end,
                                      footprint.end + canary_bytes};

  status = wafer_ncc_hazard_validate_effective_ranges(
      first, second, first_operand, second_operand, second_range, canary_before,
      canary_after);
  if (status != WAFER_NCC_HAZARD_OK)
    return status;

  wafer_ncc_hazard_zero(composition, sizeof(*composition));
  composition->first_operand = first_operand;
  composition->second_operand = second_operand;
  composition->first_range = first_range;
  composition->second_range = second_range;
  composition->overlap =
      wafer_ncc_hazard_intersection(first_range, second_range);
  composition->footprint = footprint;
  composition->canary_before = canary_before;
  composition->canary_after = canary_after;
  if (range_relation == WAFER_NCC_RANGE_STRIDED_ENVELOPE) {
    composition->flags |= WAFER_NCC_HAZARD_COMPOSITION_ENVELOPE_ONLY;
    return WAFER_NCC_HAZARD_OK;
  }

  wafer_ncc_hazard_build_expected(composition);
  return WAFER_NCC_HAZARD_OK;
}

uint32_t wafer_ncc_hazard_observe_inclusive_range(WaferNccHazardRange range,
                                                  uint64_t *begin,
                                                  uint64_t *inclusive_end) {
  if (begin == NULL || inclusive_end == NULL ||
      wafer_ncc_hazard_range_empty(range))
    return WAFER_NCC_HAZARD_INVALID_ARGUMENT;
  *begin = range.begin;
  *inclusive_end = range.end - 1U;
  return WAFER_NCC_HAZARD_OK;
}
