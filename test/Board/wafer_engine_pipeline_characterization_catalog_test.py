#!/usr/bin/env python3
"""Host tests for the engine/pipeline characterization matrix."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import tempfile
import unittest
from collections import Counter

import wafer_board_engine_pipeline_characterization_test as driver
import wafer_board_ncc_execution_probe_test as ncc_driver
import wafer_engine_pipeline_characterization_catalog as catalog
import wafer_memory_descriptor_calibration_catalog as memory_catalog
import wafer_ncc_probe_protocol as ncc_protocol


def observation(
    probe: catalog.ProbeCase,
    sample: int,
    execution: dict[str, int],
) -> dict[str, object]:
    counts = {
        f"worker{worker}.{engine.value}": 0
        for worker in range(3)
        for engine in catalog.Engine
    }
    for identity in probe.plan.issue_identities():
        lane = probe.plan.lanes[identity.lane]
        counts[
            f"worker{lane.worker}.{lane.engine.name.lower()}"
        ] += 1
    values = {engine.value: 1 for engine in catalog.Engine}
    values.update({"full": 1, **execution})
    return {
        "case": probe.as_dict(),
        "sample": sample,
        "instruction_delta": counts,
        "execution_delta": values,
        "runtime_lifecycle": catalog.RUNTIME_LIFECYCLE,
        "runtime_terminal_completion": 7,
    }


def tail_observation(sample: int, cycles: int) -> dict[str, object]:
    return {
        "case": {
            "characterization_key": catalog.NE_TAIL_CHARACTERIZATION_KEY,
            "characterization_cell": "single/ne/tail",
            "work_macs": catalog.NE_TAIL_WORK_MACS,
        },
        "sample": sample,
        "instruction_delta": {"worker0.ne": 1},
        "execution_delta": {"ne": cycles},
        "pmu_enable": 1,
        "execute_result": 1,
        "logical_sha256": "1" * 64,
        "physical_sha256": "2" * 64,
        "runtime_lifecycle": catalog.RUNTIME_LIFECYCLE,
        "runtime_terminal_completion": 9,
    }


class CatalogShapeTest(unittest.TestCase):
    def test_generic_engine_workloads_are_fp16(self) -> None:
        self.assertEqual(catalog.FMT_FP16, ncc_protocol.DMA_FORMAT_FP16)
        self.assertEqual(catalog.FP16_ELEMENT_BYTES, 2)
        for probe in catalog.BOARD_PROBES:
            record = probe.as_dict()
            self.assertEqual(
                record["formats"],
                [catalog.FMT_FP16] * len(probe.plan.lanes),
            )
            self.assertEqual(
                record["element_counts"],
                [
                    lane.transfer_bytes // catalog.FP16_ELEMENT_BYTES
                    for lane in probe.plan.lanes
                ],
            )
            for lane in probe.plan.lanes:
                self.assertEqual(lane.element_format, catalog.FMT_FP16)
                self.assertEqual(
                    lane.transfer_bytes % catalog.FP16_ELEMENT_BYTES, 0
                )

    def test_full_typed_inventory(self) -> None:
        self.assertEqual(len(catalog.SINGLE_ENGINE_CELLS), 21)
        self.assertEqual(len(catalog.ENGINE_PAIR_CELLS), 360)
        self.assertEqual(len(catalog.THREE_STAGE_CORE_CELLS), 90)
        self.assertEqual(len(catalog.THREE_STAGE_NEGATIVE_CELLS), 19)
        self.assertEqual(len(catalog.THREE_STAGE_CELLS), 109)
        self.assertEqual(len(catalog.BOARD_PROBES), 740)
        self.assertEqual(
            Counter(cell.disposition for cell in catalog.ALL_CELLS),
            {
                catalog.Disposition.BOARD_EXECUTABLE: 380,
                catalog.Disposition.BOARD_EXECUTABLE_EXTERNAL: 1,
                catalog.Disposition.PENDING_CONFIGURED_BOARD: 90,
                catalog.Disposition.HOST_EXACT_CLOSED: 19,
            },
        )

    def test_every_board_probe_serializes_and_fits_resources(self) -> None:
        for probe in catalog.BOARD_PROBES:
            words = probe.request_words(0)
            self.assertEqual(len(words), ncc_protocol.REQUEST_WORDS)
        ncc_driver.validate_catalog_resource_layout(catalog.BOARD_PROBES)

    def test_every_eight_round_pair_has_a_bounded_exact_oracle(self) -> None:
        probes = tuple(
            probe
            for cell in catalog.ENGINE_PAIR_CELLS
            if cell.dimension("iterations") == 8
            for probe in cell.probes
        )
        self.assertEqual(len(probes), 240)
        self.assertEqual(
            max(
                identity.slot
                for probe in probes
                for identity in probe.plan.issue_identities()
            ),
            15,
        )
        self.assertEqual(
            ncc_protocol.F16_POSITIVE_INTEGER_MAX,
            17,
        )
        self.assertEqual(
            ncc_protocol.F16_POSITIVE_INTEGERS[-4:],
            (0x4B00, 0x4B80, 0x4C00, 0x4C40),
        )
        for probe in probes:
            for identity in probe.plan.issue_identities():
                lane = probe.plan.lanes[identity.lane]
                expected = ncc_driver.v2_expected_result(
                    identity, lane, probe.plan
                )
                expected_bytes = (
                    ncc_driver.v2_ne_result_bytes(lane)
                    if lane.engine == ncc_protocol.Engine.NE
                    else lane.transfer_bytes
                )
                self.assertEqual(len(expected), expected_bytes)

    def test_correctness_and_measurement_oracles_are_strong(self) -> None:
        for probe in catalog.BOARD_PROBES:
            self.assertTrue(all(dataclasses.asdict(probe.correctness).values()))
            self.assertGreaterEqual(probe.repeat, 3)
            self.assertTrue(probe.measurement.requires_device_pmu)
            self.assertFalse(probe.measurement.host_elapsed_is_evidence)

    def test_single_engine_sweep_and_held_out_layouts(self) -> None:
        for engine in (catalog.Engine.RDMA, catalog.Engine.WDMA):
            keys = catalog.single_activation_cell_keys(engine)
            sizes = {
                int(catalog.CELLS_BY_KEY[key].dimension("bytes"))
                for key in keys
                if catalog.CELLS_BY_KEY[key].dimension("layout")
                == "contiguous"
            }
            self.assertEqual(sizes, {256, 4096, 16384, 65536})
            self.assertIn(
                "single/rdma-wdma/strided-roundtrip-16k", keys
            )
        tdma_keys = catalog.single_activation_cell_keys(catalog.Engine.TDMA)
        self.assertIn("single/tdma/inner-strided-16k", tdma_keys)
        self.assertEqual(
            catalog.CELLS_BY_KEY["single/ne/tail"].disposition,
            catalog.Disposition.BOARD_EXECUTABLE_EXTERNAL,
        )
        self.assertEqual(
            catalog.EXTERNAL_BOARD_GROUP_KEYS,
            ("single-ne-tail",),
        )
        self.assertEqual(
            catalog.CELLS_BY_KEY["single/ne/small"].dimension("work_macs"),
            1 * 16 * 16,
        )
        self.assertEqual(
            catalog.CELLS_BY_KEY["single/ne/steady-16k"].dimension(
                "work_macs"
            ),
            64 * 128 * 128,
        )
        self.assertEqual(
            catalog.CELLS_BY_KEY["single/ne/tail"].dimension("work_macs"),
            65 * 129 * 129,
        )

    def test_stage_balance_dimensions_are_complete(self) -> None:
        coordinates = {
            (
                cell.dimension("engine_a"),
                cell.dimension("engine_b"),
                cell.dimension("placement"),
                cell.dimension("direction"),
                cell.dimension("balance"),
                cell.dimension("iterations"),
            )
            for cell in catalog.ENGINE_PAIR_CELLS
        }
        self.assertEqual(len(coordinates), 360)
        for engine_a, engine_b in catalog.ENGINE_PAIRS:
            for placement in catalog.Placement:
                for iterations in (2, 4, 8):
                    self.assertEqual(
                        len(
                            catalog.pair_activation_cell_keys(
                                engine_a,
                                engine_b,
                                placement,
                                iterations,
                            )
                        ),
                        6,
                    )

    def test_board_groups_partition_cells_at_activation_grain(self) -> None:
        self.assertEqual(len(catalog.BOARD_ACTIVATION_GROUPS), 61)
        self.assertEqual(
            catalog.BOARD_ACTIVATION_GROUPS[0].key,
            "single-engines",
        )
        grouped = tuple(
            cell_key
            for group in catalog.BOARD_ACTIVATION_GROUPS
            for cell_key in group.cell_keys
        )
        executable = {
            cell.key
            for cell in catalog.ALL_CELLS
            if cell.disposition == catalog.Disposition.BOARD_EXECUTABLE
        }
        self.assertEqual(len(grouped), len(executable))
        self.assertEqual(set(grouped), executable)
        self.assertEqual(len(set(grouped)), len(grouped))
        for group in catalog.BOARD_ACTIVATION_GROUPS[1:]:
            self.assertEqual(len(group.cell_keys), 6)

    def test_pair_controls_are_one_factor_and_fixed_engine_work(self) -> None:
        for cell in catalog.ENGINE_PAIR_CELLS:
            if cell.disposition != catalog.Disposition.BOARD_EXECUTABLE:
                continue
            serial, window = cell.probes
            self.assertEqual(
                dataclasses.replace(
                    serial.plan,
                    schedule=window.plan.schedule,
                    flags=window.plan.flags,
                ),
                window.plan,
            )
            if cell.dimension("iterations") == 8:
                self.assertEqual(
                    window.plan.flags,
                    ncc_protocol.BOUNDED_PAIR_WINDOW,
                )
                self.assertTrue(window.plan.is_bounded_pair_window())
            else:
                self.assertEqual(window.plan.flags, 0)
            by_engine = {
                lane.engine.name.lower(): (
                    lane.worker,
                    lane.transfer_bytes,
                )
                for lane in serial.plan.lanes
            }
            engine_a = str(cell.dimension("engine_a"))
            engine_b = str(cell.dimension("engine_b"))
            expected_workers = (
                {engine_a: 0, engine_b: 0}
                if cell.dimension("placement") == "same-worker"
                else {engine_a: 0, engine_b: 1}
            )
            self.assertEqual(
                {engine: value[0] for engine, value in by_engine.items()},
                expected_workers,
            )
            if cell.dimension("direction") == "a-to-b":
                self.assertEqual(
                    serial.plan.lanes[0].engine.name.lower(), engine_a
                )
            else:
                self.assertEqual(
                    serial.plan.lanes[0].engine.name.lower(), engine_b
                )

    def test_long_and_tdma_windows_have_safe_explicit_execution(self) -> None:
        eight = [
            cell
            for cell in catalog.ENGINE_PAIR_CELLS
            if cell.dimension("iterations") == 8
        ]
        self.assertTrue(eight)
        self.assertTrue(
            all(
                cell.disposition
                == catalog.Disposition.BOARD_EXECUTABLE
                for cell in eight
            )
        )
        self.assertTrue(
            all(
                cell.probes[1].plan.is_bounded_pair_window()
                for cell in eight
                if cell.disposition
                == catalog.Disposition.BOARD_EXECUTABLE
            )
        )
        tdma_four = [
            cell
            for cell in catalog.ENGINE_PAIR_CELLS
            if (
                cell.dimension("iterations") == 4
                and "tdma"
                in (
                    cell.dimension("engine_a"),
                    cell.dimension("engine_b"),
                )
            )
        ]
        self.assertTrue(
            all(
                cell.disposition
                == catalog.Disposition.BOARD_EXECUTABLE
                for cell in tdma_four
            )
        )

    def test_delegated_smoke_resolves_to_real_case_names(self) -> None:
        names = set(memory_catalog.CASES_BY_NAME)
        delegated = [
            reference
            for cell in catalog.ENGINE_PAIR_CELLS
            for reference in cell.delegated
        ]
        self.assertEqual(len(delegated), 11)
        for reference in delegated:
            self.assertTrue(set(reference.case_names).issubset(names))
        raw_names = {
            case.name for case in ncc_driver.V2_DOUBLE_SLOT_OBSERVATION_CASES
        }
        self.assertEqual(
            set(catalog.DELEGATED_EVIDENCE[0].case_names), raw_names
        )


class ActivationTest(unittest.TestCase):
    def test_single_engine_activation_requires_complete_held_out_direction(
        self,
    ) -> None:
        rows: list[dict[str, object]] = []
        keys = catalog.single_activation_cell_keys(catalog.Engine.CT)
        for key in keys:
            probe = catalog.CELLS_BY_KEY[key].probes[0]
            size = probe.plan.lanes[0].transfer_bytes
            for sample in range(3):
                rows.append(
                    observation(
                        probe,
                        sample,
                        {"ct": 100 + size // 16, "full": 100 + size // 16},
                    )
                )
        decision = catalog.evaluate_single_engine_activation(
            catalog.Engine.CT, rows
        )
        self.assertTrue(decision.activated, decision.reasons)
        incomplete = catalog.evaluate_single_engine_activation(
            catalog.Engine.CT, rows[:-1]
        )
        self.assertFalse(incomplete.activated)
        self.assertTrue(
            any("distinct samples" in reason for reason in incomplete.reasons)
        )

    def test_ne_rate_stays_unknown_without_real_tail_adapter(self) -> None:
        rows: list[dict[str, object]] = []
        for key in catalog.single_activation_cell_keys(catalog.Engine.NE):
            probe = catalog.CELLS_BY_KEY[key].probes[0]
            for sample in range(3):
                rows.append(
                    observation(
                        probe,
                        sample,
                        {
                            "ne": 100
                            + probe.plan.lanes[0].transfer_bytes // 16,
                            "full": 100
                            + probe.plan.lanes[0].transfer_bytes // 16,
                        },
                    )
                )
        decision = catalog.evaluate_single_engine_activation(
            catalog.Engine.NE, rows
        )
        self.assertFalse(decision.activated)
        self.assertTrue(any("tail" in reason for reason in decision.reasons))

    def test_ne_rate_activates_only_after_typed_tail_merge(self) -> None:
        rows: list[dict[str, object]] = []
        raw_cycles = {
            "single/ne/small": 100,
            "single/ne/steady-16k": 1000,
        }
        for key in catalog.single_activation_cell_keys(catalog.Engine.NE):
            probe = catalog.CELLS_BY_KEY[key].probes[0]
            for sample in range(3):
                rows.append(
                    observation(
                        probe,
                        sample,
                        {
                            "ne": raw_cycles[key] + sample,
                            "full": raw_cycles[key] + sample,
                        },
                    )
                )
        rows.extend(tail_observation(sample, 1100 + sample) for sample in range(3))
        activated = catalog.evaluate_single_engine_activation(
            catalog.Engine.NE, rows
        )
        self.assertTrue(activated.activated, activated.reasons)
        non_monotonic = [
            (
                tail_observation(sample, 900 + sample)
                if row.get("case", {}).get("characterization_key")
                == catalog.NE_TAIL_CHARACTERIZATION_KEY
                else row
            )
            for sample, row in enumerate(rows)
        ]
        rejected = catalog.evaluate_single_engine_activation(
            catalog.Engine.NE, non_monotonic
        )
        self.assertFalse(rejected.activated)
        self.assertTrue(
            any("positive work direction" in reason for reason in rejected.reasons)
        )

    def test_pair_activation_uses_overlap_and_reciprocal_ratio_order(
        self,
    ) -> None:
        keys = catalog.pair_activation_cell_keys(
            catalog.Engine.CT,
            catalog.Engine.RDMA,
            catalog.Placement.SAME_WORKER,
            2,
        )
        rows: list[dict[str, object]] = []
        cycles = {
            catalog.Balance.A_HEAVY.value: (300, 100),
            catalog.Balance.EQUAL_BYTES.value: (200, 200),
            catalog.Balance.B_HEAVY.value: (100, 300),
        }
        for key in keys:
            cell = catalog.CELLS_BY_KEY[key]
            a_cycles, b_cycles = cycles[str(cell.dimension("balance"))]
            for probe in cell.probes:
                is_window = (
                    probe.plan.schedule == ncc_protocol.Schedule.WINDOW
                )
                full = a_cycles + b_cycles - (50 if is_window else 0)
                for sample in range(3):
                    rows.append(
                        observation(
                            probe,
                            sample,
                            {
                                "ct": a_cycles,
                                "rdma": b_cycles,
                                "full": full,
                            },
                        )
                    )
        decision = catalog.evaluate_pair_activation(
            catalog.Engine.CT,
            catalog.Engine.RDMA,
            catalog.Placement.SAME_WORKER,
            2,
            rows,
        )
        self.assertTrue(decision.activated, decision.reasons)
        incomplete = catalog.evaluate_pair_activation(
            catalog.Engine.CT,
            catalog.Engine.RDMA,
            catalog.Placement.SAME_WORKER,
            2,
            rows[:-1],
        )
        self.assertFalse(incomplete.activated)


class ProductionGateTest(unittest.TestCase):
    @staticmethod
    def _write_json(path: pathlib.Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _refresh_attestation_digest(
        cls, attestation: pathlib.Path, activation: pathlib.Path
    ) -> None:
        activation_value = json.loads(
            activation.read_text(encoding="utf-8")
        )
        activation_value["attestation_sha256"] = (
            "sha256:" + hashlib.sha256(attestation.read_bytes()).hexdigest()
        )
        cls._write_json(activation, activation_value)

    @classmethod
    def _make_valid_package(
        cls, root: pathlib.Path
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
        package = root / "package"
        package.mkdir()
        manifest_path = package / "manifest.json"
        cls._write_json(
            manifest_path,
            {
                "schema_version": 7,
                "rank_count": 1,
                "target": {
                    "identity": "wafer-tx81-single-card"
                },
            },
        )
        manifest_digest = (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )

        instrumentation = pathlib.Path(str(package) + ".qualification")
        instrumentation.mkdir()
        attestation_path = instrumentation / "attestation.json"
        cls._write_json(
            attestation_path,
            {
                "schema": "wafer-static-fixed-slot-qualification",
                "schema_version": 1,
                "selection_kind": "static-fixed-slot",
                "manifest_sha256": manifest_digest,
                "accepted_instr_digest_basis": (
                    "final-accepted-instr-module-text-v1"
                ),
                "target": {
                    "identity": "wafer-tx81-single-card",
                    "rank_count": 1,
                    "logical_ranks": [0],
                },
                "ranks": [
                    {
                        "logical_rank": 0,
                        "accepted_instr_sha256": "sha256:" + "1" * 64,
                        "spm_alloc_roots": [
                            {
                                "ordinal": 0,
                                "offset": 65536,
                                "bytes": 256,
                                "range_begin": 65536,
                                "range_end": 65792,
                            },
                            {
                                "ordinal": 1,
                                "offset": 65792,
                                "bytes": 256,
                                "range_begin": 65792,
                                "range_end": 66048,
                            },
                        ],
                        "static_loops": [
                            {
                                "ordinal": 0,
                                "lower": 0,
                                "upper": 3,
                                "step": 1,
                                "trip_count": 3,
                                "iter_arg_count": 2,
                                "spm_iter_arg_rotations": [
                                    {
                                        "iter_arg": 0,
                                        "initial_root": 0,
                                        "next_iter_arg": 1,
                                        "next_root": 1,
                                    },
                                    {
                                        "iter_arg": 1,
                                        "initial_root": 1,
                                        "next_iter_arg": 0,
                                        "next_root": 0,
                                    },
                                ],
                            }
                        ],
                        "engine_worker_issues": [
                            {"engine": "ct", "worker": 0, "count": 1},
                            {"engine": "rdma", "worker": 0, "count": 2},
                            {"engine": "wdma", "worker": 0, "count": 1},
                        ],
                        "dte": {
                            "token_count": 0,
                            "issues": [],
                            "waits": [],
                        },
                        "completion": {
                            "participant_joins": [
                                {
                                    "ordinal": 0,
                                    "participants": [0],
                                    "inside_static_loop": False,
                                }
                            ],
                            "behaviors": [
                                {"kind": "ordered-pending", "count": 4},
                                {"kind": "participant-join", "count": 1},
                                {
                                    "kind": "synchronous-writeback",
                                    "count": 0,
                                },
                            ],
                        },
                    }
                ],
            },
        )
        attestation_digest = (
            "sha256:"
            + hashlib.sha256(attestation_path.read_bytes()).hexdigest()
        )
        activation_path = instrumentation / "activation.json"
        cls._write_json(
            activation_path,
            {
                "schema": (
                    "wafer-static-fixed-slot-qualification-activation"
                ),
                "schema_version": 1,
                "manifest_sha256": manifest_digest,
                "attestation_sha256": attestation_digest,
            },
        )
        return package, manifest_path, attestation_path, activation_path

    def test_three_stage_matrix_separates_board_and_host_gates(self) -> None:
        dimensions = {
            (
                cell.dimension("compute"),
                cell.dimension("regime"),
                cell.dimension("schedule"),
                cell.dimension("iterations"),
            )
            for cell in catalog.THREE_STAGE_CORE_CELLS
        }
        self.assertEqual(len(dimensions), 90)
        self.assertTrue(
            all(
                cell.disposition
                == catalog.Disposition.PENDING_CONFIGURED_BOARD
                and not cell.probes
                for cell in catalog.THREE_STAGE_CORE_CELLS
            )
        )
        self.assertTrue(
            all(
                cell.disposition
                == catalog.Disposition.HOST_EXACT_CLOSED
                and not cell.probes
                for cell in catalog.THREE_STAGE_NEGATIVE_CELLS
            )
        )
        self.assertEqual(
            catalog.THREE_STAGE_PENDING_BOARD_CELLS,
            catalog.THREE_STAGE_CORE_CELLS,
        )
        self.assertEqual(
            catalog.THREE_STAGE_HOST_EXACT_CELLS,
            catalog.THREE_STAGE_NEGATIVE_CELLS,
        )
        negative_kinds = {
            cell.dimension("negative_kind")
            for cell in catalog.THREE_STAGE_NEGATIVE_CELLS
        }
        self.assertEqual(
            negative_kinds,
            {
                "capacity-fallback",
                "alias-legality",
                "handwritten-provenance",
            },
        )

    def test_gate_rejects_missing_package_and_qualification(self) -> None:
        missing = catalog.validate_production_pipeline_inputs(None)
        self.assertFalse(missing.ready)
        self.assertIn(
            "V2_DOUBLE_SLOT_OBSERVATION_CASES",
            missing.rejected_delegated_asset,
        )
        with tempfile.TemporaryDirectory() as temporary:
            package = pathlib.Path(temporary) / "package"
            package.mkdir()
            (package / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 7,
                        "rank_count": 1,
                        "target": {
                            "identity": "wafer-tx81-single-card"
                        },
                    }
                )
                + "\n"
            )
            no_qualification = (
                catalog.validate_production_pipeline_inputs(package)
            )
        self.assertFalse(no_qualification.ready)
        self.assertTrue(
            any("qualification sibling" in reason for reason in no_qualification.reasons)
        )

    def test_gate_rejects_package_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package, _, _, _ = self._make_valid_package(root)
            package_link = root / "package-link"
            package_link.symlink_to(package, target_is_directory=True)
            decision = catalog.validate_production_pipeline_inputs(package_link)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("not a regular directory" in reason for reason in decision.reasons)
        )

    def test_gate_rejects_stale_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, manifest, _, _ = self._make_valid_package(
                pathlib.Path(temporary)
            )
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["program"] = 0
            self._write_json(manifest, value)
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("stale manifest binding" in reason for reason in decision.reasons)
        )

    def test_gate_rejects_tampered_attestation_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, attestation, _ = self._make_valid_package(
                pathlib.Path(temporary)
            )
            attestation.write_text(
                attestation.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("tampered" in reason for reason in decision.reasons)
        )

    def test_gate_rejects_unknown_attestation_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, attestation, activation = self._make_valid_package(
                pathlib.Path(temporary)
            )
            value = json.loads(attestation.read_text(encoding="utf-8"))
            value["unknown"] = True
            self._write_json(attestation, value)
            self._refresh_attestation_digest(attestation, activation)
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("unknown=['unknown']" in reason for reason in decision.reasons)
        )

    def test_gate_rejects_spm_root_outside_target_arena(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, attestation, activation = self._make_valid_package(
                pathlib.Path(temporary)
            )
            value = json.loads(attestation.read_text(encoding="utf-8"))
            root = value["ranks"][0]["spm_alloc_roots"][0]
            root["offset"] = root["range_begin"] = 0
            root["range_end"] = root["bytes"]
            self._write_json(attestation, value)
            self._refresh_attestation_digest(attestation, activation)
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("outside the allocatable target arena" in reason
                for reason in decision.reasons)
        )

    def test_gate_rejects_misaligned_spm_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, attestation, activation = self._make_valid_package(
                pathlib.Path(temporary)
            )
            value = json.loads(attestation.read_text(encoding="utf-8"))
            root = value["ranks"][0]["spm_alloc_roots"][0]
            root["offset"] += 1
            root["range_begin"] += 1
            root["range_end"] += 1
            self._write_json(attestation, value)
            self._refresh_attestation_digest(attestation, activation)
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("not placement-aligned" in reason
                for reason in decision.reasons)
        )

    def test_gate_rejects_overlapping_spm_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, attestation, activation = self._make_valid_package(
                pathlib.Path(temporary)
            )
            value = json.loads(attestation.read_text(encoding="utf-8"))
            roots = value["ranks"][0]["spm_alloc_roots"]
            roots[1]["offset"] = roots[1]["range_begin"] = roots[0]["offset"]
            roots[1]["range_end"] = roots[1]["offset"] + roots[1]["bytes"]
            self._write_json(attestation, value)
            self._refresh_attestation_digest(attestation, activation)
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertFalse(decision.ready)
        self.assertTrue(
            any("intervals overlap" in reason for reason in decision.reasons)
        )

    def test_gate_accepts_valid_manifest_bound_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, _, _, _ = self._make_valid_package(
                pathlib.Path(temporary)
            )
            decision = catalog.validate_production_pipeline_inputs(package)
        self.assertTrue(decision.ready, decision.reasons)
        self.assertEqual(decision.reasons, ())

    def test_driver_group_parser_preserves_execution_boundaries(self) -> None:
        self.assertEqual(
            driver.activation_group_cell_keys("single:ct"),
            catalog.single_activation_cell_keys(catalog.Engine.CT),
        )
        pair = driver.activation_group_cell_keys(
            "pair:ct-rdma:same-worker:i2"
        )
        self.assertEqual(len(pair), 6)
        tdma = driver.activation_group_cell_keys(
            "pair:ct-tdma:same-worker:i4"
        )
        self.assertEqual(len(tdma), 6)
        tdma_board = tuple(
            key
            for key in tdma
            if (
                catalog.CELLS_BY_KEY[key].disposition
                == catalog.Disposition.BOARD_EXECUTABLE
            )
        )
        self.assertEqual(len(catalog.probes_for_cells(tdma_board)), 12)
        with self.assertRaises(RuntimeError):
            catalog.probes_for_cells(("single/ne/tail",))


class DriverEvidenceTest(unittest.TestCase):
    def test_execution_order_is_sample_major_and_counterbalanced(self) -> None:
        keys = catalog.pair_activation_cell_keys(
            catalog.Engine.CT,
            catalog.Engine.RDMA,
            catalog.Placement.SAME_WORKER,
            2,
        )
        probes = catalog.probes_for_cells(keys)
        scheduled = driver.counterbalanced_probe_schedule(probes, 3)
        self.assertEqual(len(scheduled), len(probes) * 3)
        sweeps = tuple(
            scheduled[index * len(probes) : (index + 1) * len(probes)]
            for index in range(3)
        )
        self.assertEqual(
            tuple(sample for _, sample in sweeps[0]),
            (0,) * len(probes),
        )
        self.assertEqual(
            tuple(probe for probe, _ in sweeps[0]),
            probes,
        )
        self.assertEqual(
            tuple(probe for probe, _ in sweeps[1]),
            tuple(reversed(probes)),
        )
        half = len(probes) // 2
        self.assertEqual(
            tuple(probe for probe, _ in sweeps[2]),
            probes[half:] + probes[:half],
        )
        for sweep in sweeps:
            self.assertEqual(
                Counter(probe.key for probe, _ in sweep),
                Counter(probe.key for probe in probes),
            )

    def test_manifest_and_board_terminal_completion_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = pathlib.Path(temporary)
            manifest = {
                "schema_version": 7,
                "rank_count": 1,
                "entries": [
                    {
                        "id": 0,
                        "rank": 0,
                        "terminal_completion": 17,
                    }
                ],
                "completions": [
                    {"id": 17, "rank": 0, "kind": "entry_return"}
                ],
            }
            (package / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(driver.rank_one_terminal_completion(package), 17)
            stdout = "\n".join(
                [
                    *(
                        f"board_stage: {stage}"
                        for stage in catalog.RUNTIME_LIFECYCLE
                    ),
                    "terminal_completion: 17 kind=entry_return",
                    "board_execution: true",
                ]
            )
            driver.require_exact_board_completion(stdout, 17, "probe")
            with self.assertRaisesRegex(
                RuntimeError, "differs from the package manifest"
            ):
                driver.require_exact_board_completion(
                    stdout.replace(
                        "terminal_completion: 17",
                        "terminal_completion: 18",
                    ),
                    17,
                    "probe",
                )
            manifest["entries"][0]["terminal_completion"] = 18
            (package / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                RuntimeError, "does not bind its exact terminal completion"
            ):
                driver.rank_one_terminal_completion(package)

    def test_session_and_board_qualification_are_typed(self) -> None:
        session_id = driver.require_calibration_session(
            {catalog.CALIBRATION_SESSION_ENVIRONMENT: "a" * 32}
        )
        self.assertEqual(session_id, "a" * 32)
        with self.assertRaisesRegex(RuntimeError, "32 lowercase hex"):
            driver.require_calibration_session(
                {catalog.CALIBRATION_SESSION_ENVIRONMENT: "stale"}
            )
        qualification = driver.board_qualification(
            argparse.Namespace(
                device_id=0,
                expected_runtime_version=17,
                expected_device_name="tx81",
                expected_pci_bus_id="0000:01:00.0",
                expected_tile_count=16,
                expected_runtime_library_sha256="A" * 64,
            )
        )
        self.assertEqual(
            qualification,
            {
                "target_identity": ncc_driver.TARGET_IDENTITY,
                "launch": {
                    "kind": "kernel",
                    "form": "per-rank",
                    "entry_abi": "rank-local-pointer-block",
                    "phases": ["main"],
                },
                "device_id": 0,
                "expected_runtime_version": 17,
                "expected_device_name": "tx81",
                "expected_pci_bus_id": "0000:01:00.0",
                "expected_tile_count": 16,
                "expected_runtime_library_sha256": "a" * 64,
            },
        )


if __name__ == "__main__":
    unittest.main()
