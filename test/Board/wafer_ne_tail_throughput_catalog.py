#!/usr/bin/env python3
"""Typed, device-PMU-backed NE non-divisible tail characterization.

Pipeline position:
- Upstream artifact / IR:
  The already-qualified FP16 ``M=65, K=129, N=129`` GEMM tail request and
  physical Cx payload from the NE calibration catalog.
- Current stage responsibility:
  Reuse that exact numeric/layout case while requiring a three-sample NE PMU
  observation that can fill the held-out point of the single-engine slope.
- Output artifact / IR:
  Exact result/span/guard/lifecycle evidence plus raw NE instruction,
  blocking, and execution-counter deltas.  No latency constant is written.
- Downstream consumer:
  The ``single/ne/tail`` hardware-calibration cell and its later
  profile-scoped cost activation gate.
- User-level driver / named pipeline:
  ``wafer_board_ne_tail_throughput_test.py`` through the normal
  source-package-device-ELF-wafer-run path.
- Explicit non-goals:
  Host elapsed time as performance evidence, BF16-to-FP16 extrapolation, or
  treating one tail sample as a general NE rate.
- Completion gate:
  Three serial board launches, exact logical and physical output, unchanged
  prefix/suffix and out-of-record guards, exactly one NE instruction,
  positive NE execution delta, exact manifest-matched terminal completion, and
  complete runtime lifecycle each time.  The same CTest invocation must first
  execute fresh small/steady controls under the same runner session and
  board-qualification profile; only the merged direction may activate.
"""

from __future__ import annotations

import dataclasses

import wafer_ne_calibration_catalog as ne_catalog


CELL_KEY = "single/ne/tail"
GROUP_KEY = "single-ne-tail"
MINIMUM_REPEATS = 3


@dataclasses.dataclass(frozen=True)
class PipelineContract:
    upstream_artifact: str
    current_stage_responsibility: str
    output_artifact: str
    downstream_consumer: str
    user_level_driver: str
    explicit_non_goals: tuple[str, ...]
    completion_gate: tuple[str, ...]


PIPELINE_CONTRACT = PipelineContract(
    upstream_artifact=(
        "qualified FP16 non-divisible GEMM-tail request and Cx payload"
    ),
    current_stage_responsibility=(
        "exact tail execution with a compatible device-PMU measurement window"
    ),
    output_artifact=(
        "exact result/guard/lifecycle evidence and raw NE PMU deltas"
    ),
    downstream_consumer="single/ne/tail profile-scoped activation gate",
    user_level_driver="wafer_board_ne_tail_throughput_test.py",
    explicit_non_goals=(
        "host elapsed as a device metric",
        "fixed latency publication",
        "cross-dtype extrapolation",
    ),
    completion_gate=(
        "three distinct serial board samples",
        "exact logical and physical output with all guards",
        "one NE instruction and positive NE execution delta",
        "exact manifest-matched terminal completion and ordered lifecycle",
        "fresh small/steady controls in the same CTest invocation",
        "same runner session and exact board-qualification profile",
        "merged small/steady/tail activation direction",
    ),
)


@dataclasses.dataclass(frozen=True)
class NETailThroughputCase:
    key: str
    cell_key: str
    group_key: str
    source_case_name: str
    repeats: int
    phase: str
    work_macs: int
    expected_ne_instructions: int
    device_metric: str
    activation_prerequisite_cells: tuple[str, ...]

    @property
    def source_case(self) -> ne_catalog.NECase:
        case = ne_catalog.CASES_BY_NAME[self.source_case_name]
        if not isinstance(case, ne_catalog.NECase):
            raise RuntimeError(
                f"{self.source_case_name}: expected an executable NE case"
            )
        return case

    @property
    def name(self) -> str:
        return self.key.replace("/", "-")

    def as_dict(self) -> dict[str, object]:
        source = self.source_case
        return {
            "name": self.name,
            "characterization_key": self.key,
            "characterization_cell": self.cell_key,
            "activation_group": self.group_key,
            "source_case": self.source_case_name,
            "phase": self.phase,
            "dtype": source.dtype_name.lower(),
            "m": source.m,
            "k": source.k,
            "n": source.n,
            "work_macs": self.work_macs,
            "repeats": self.repeats,
            "expected_ne_instructions": self.expected_ne_instructions,
            "device_metric": self.device_metric,
            "activation_prerequisite_cells": list(
                self.activation_prerequisite_cells
            ),
            "oracle": {
                "exact_logical_bits": True,
                "exact_physical_padding": True,
                "prefix_suffix_guards": True,
                "out_of_record_guard": True,
                "matching_completion": True,
                "terminal_status_and_cleanup": True,
                "host_elapsed_is_evidence": False,
            },
        }


CASES = (
    NETailThroughputCase(
        key="single/ne/tail/f16-m65-k129-n129",
        cell_key=CELL_KEY,
        group_key=GROUP_KEY,
        source_case_name="ne-f16-tail-nn",
        repeats=MINIMUM_REPEATS,
        phase="held-out",
        work_macs=65 * 129 * 129,
        expected_ne_instructions=1,
        device_metric="gr-pmu-ne-execution-delta",
        activation_prerequisite_cells=(
            "single/ne/small",
            "single/ne/steady-16k",
        ),
    ),
)
CASES_BY_KEY = {case.key: case for case in CASES}


def validate_catalog() -> None:
    if len(CASES_BY_KEY) != len(CASES):
        raise RuntimeError("NE tail characterization keys are not unique")
    if {case.cell_key for case in CASES} != {CELL_KEY}:
        raise RuntimeError("NE tail cases do not bind the expected cell")
    if {case.group_key for case in CASES} != {GROUP_KEY}:
        raise RuntimeError("NE tail cases do not bind one activation group")
    for case in CASES:
        source = case.source_case
        if (
            case.repeats < MINIMUM_REPEATS
            or source.name != "ne-f16-tail-nn"
            or not source.exact
            or source.dtype_name != "F16"
            or source.kind_name != "GEMM"
            or source.geometry_name != "tail"
            or source.orientation_name != "NN"
            or (source.m, source.k, source.n) != (65, 129, 129)
            or case.work_macs != source.m * source.k * source.n
            or case.expected_ne_instructions != 1
            or case.activation_prerequisite_cells
            != ("single/ne/small", "single/ne/steady-16k")
        ):
            raise RuntimeError(
                f"{case.key}: incompatible NE tail characterization contract"
            )
        oracle = case.as_dict()["oracle"]
        if not isinstance(oracle, dict) or not all(
            value for key, value in oracle.items() if key != "host_elapsed_is_evidence"
        ):
            raise RuntimeError(f"{case.key}: incomplete correctness oracle")
        if oracle.get("host_elapsed_is_evidence") is not False:
            raise RuntimeError(f"{case.key}: host elapsed cannot activate cost")


validate_catalog()
