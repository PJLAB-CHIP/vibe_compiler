#!/usr/bin/env python3

import dataclasses
import contextlib
import io
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import wafer_ncc_probe_protocol as protocol
import wafer_board_ncc_execution_probe_test as execution_probe


def lane(engine: protocol.Engine, worker: int = 0) -> protocol.Lane:
    return protocol.Lane(
        engine=engine,
        worker=worker,
        issue_mode=protocol.IssueMode.RAW,
        transfer_bytes=4096,
        element_format=1,
    )


class ProtocolTest(unittest.TestCase):
    def test_header_is_the_numeric_source(self) -> None:
        self.assertEqual(protocol.SCHEMA, 5)
        self.assertEqual(protocol.MAX_LANES, 3)
        self.assertEqual(protocol.MAX_ROUNDS, 4)
        self.assertEqual(protocol.MAX_ISSUES, 12)
        self.assertEqual(protocol.REQUEST_WORDS, 58)
        self.assertEqual(protocol.RECORD_WORDS, 400)
        self.assertEqual(protocol.ISSUE_STRIDE, 22)
        self.assertEqual(protocol.WAIT_SAMPLE_BASE, 392)
        self.assertEqual(protocol.MAX_WAIT_SAMPLES, 8)
        self.assertEqual(protocol.MAX_DMA_ENVELOPE_BYTES, 65536)
        self.assertEqual(
            protocol.WAIT_SAMPLE_BASE,
            protocol.ISSUE_BASE
            + protocol.MAX_ISSUES * protocol.ISSUE_STRIDE,
        )
        self.assertEqual(
            execution_probe.PMU64_NAMES,
            (
                "window",
                "full",
                "ct",
                "ne",
                "rdma",
                "wdma",
                "tdma",
                "scalar",
            ),
        )
        self.assertEqual(execution_probe.PMU_STABLE_MASK, 0xFF)

    def test_prepare_failure_reports_issue_and_builder_stage(self) -> None:
        plan = protocol.Plan(
            lanes=(lane(protocol.Engine.CT),),
            rounds=1,
            effect_relation=protocol.EffectRelation.NONE,
            range_relation=protocol.RangeRelation.DISJOINT,
            schedule=protocol.Schedule.SERIAL,
            wait_kind=protocol.WaitKind.BY_WORKER,
            wait_worker_mask=1,
            seed=3,
        )
        words = [0] * protocol.RECORD_WORDS
        words[protocol.REC["MAGIC"]] = protocol.RECORD_MAGIC
        words[protocol.REC["SCHEMA_AND_WORDS"]] = (
            protocol.SCHEMA << 32
        ) | protocol.RECORD_WORDS
        words[protocol.REC["STATUS"]] = protocol.Status.PREPARE_FAILED
        words[
            protocol.ISSUE_BASE + protocol.ISSUE["FLAGS"]
        ] = protocol.PREPARE_STAGE_FLAGS["ENTERED"]
        with self.assertRaisesRegex(
            ValueError,
            (
                "PREPARE_FAILED: prepare issue 0 slot=0 engine=CT worker=0 "
                "stages=entered outcome=builder-not-acquired"
            ),
        ):
            protocol.validate_record(words, plan)

        words[protocol.ISSUE_BASE + protocol.ISSUE["FLAGS"]] |= (
            protocol.PREPARE_STAGE_FLAGS["BUILDER_ACQUIRED"]
        )
        with self.assertRaisesRegex(
            ValueError, "outcome=packet-not-materialized"
        ):
            protocol.validate_record(words, plan)

    def test_copyback_drains_every_wdma_issue(self) -> None:
        source = (
            pathlib.Path(__file__).resolve().parent
            / "Inputs"
            / "wafer_ncc_execution_probe.c"
        ).read_text()
        function = source.split(
            "static void wafer_ncc_v2_copy_results", maxsplit=1
        )[1].split(
            '__attribute__((visibility("hidden")))', maxsplit=1
        )[0]
        issue = function.index("wafer_tx81_wdma(")
        drain = function.index("wafer_tx81_local_fence();", issue)
        loop_end = function.index("\n    }", issue)
        self.assertLess(issue, drain)
        self.assertLess(drain, loop_end)

    def test_work_directory_rejects_repository_ancestors(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        for unsafe in (pathlib.Path("/"), repo, *repo.parents):
            with self.subTest(unsafe=unsafe):
                with self.assertRaisesRegex(RuntimeError, "too broad"):
                    execution_probe.validate_work_dir(repo, unsafe)
        safe = repo / "build" / "ncc-probe-test"
        self.assertEqual(
            execution_probe.validate_work_dir(repo, safe), safe.resolve()
        )

    def test_dual_lane_unique_issue_identity_and_order(self) -> None:
        plan = protocol.Plan(
            lanes=(
                lane(protocol.Engine.RDMA, 0),
                lane(protocol.Engine.CT, 1),
            ),
            rounds=4,
            effect_relation=protocol.EffectRelation.RAW,
            range_relation=protocol.RangeRelation.EXACT,
            schedule=protocol.Schedule.WINDOW,
            wait_kind=protocol.WaitKind.BY_WORKER,
            wait_worker_mask=0b011,
            seed=0x123456789ABCDEF0,
            first_operand=protocol.Operand.WRITE,
            second_operand=protocol.Operand.READ0,
        )
        words = plan.request_words()
        self.assertEqual(len(words), protocol.REQUEST_WORDS)
        self.assertEqual(plan.issue_order(), (0, 4, 1, 5, 2, 6, 3, 7))
        identities = plan.issue_identities()
        self.assertEqual(len({item.slot for item in identities}), 8)
        self.assertEqual(len({item.tag for item in identities}), 8)
        self.assertEqual(
            words[
                protocol.REQUEST_RESERVED_BASE :
                protocol.REQUEST_RESERVED_BASE
                + protocol.REQUEST_RESERVED_WORDS
            ],
            (0,) * protocol.REQUEST_RESERVED_WORDS,
        )
        inactive_begin = protocol.LANE_BASE + 2 * protocol.LANE_STRIDE
        self.assertEqual(
            words[inactive_begin : inactive_begin + protocol.LANE_STRIDE],
            (0,) * protocol.LANE_STRIDE,
        )

    def test_dma_stride_matrix_plan_and_exact_host_oracle(self) -> None:
        expected = (
            ((0, 10, 20, 30), 36, 12),
            ((0, 8, 16, 28, 36, 44), 48, 24),
            ((0, 8, 20, 28, 52, 60, 72, 80), 84, 52),
        )
        self.assertEqual(
            len(execution_probe.V2_DMA_STRIDE_MATRIX_CASES), len(expected)
        )
        for case, (offsets, envelope, hole_bytes) in zip(
            execution_probe.V2_DMA_STRIDE_MATRIX_CASES,
            expected,
            strict=True,
        ):
            plan = case.plan
            self.assertEqual(plan.issue_order(), (0, 4))
            self.assertTrue(plan.is_serial_dma_roundtrip())
            self.assertEqual(plan.schedule, protocol.Schedule.SERIAL)
            self.assertEqual(plan.wait_worker_mask, 1)
            self.assertEqual(
                tuple(lane.engine for lane in plan.lanes),
                (protocol.Engine.RDMA, protocol.Engine.WDMA),
            )
            self.assertTrue(
                all(
                    lane.issue_mode == protocol.IssueMode.WRAPPER
                    and lane.element_format == execution_probe.FMT_FP16
                    and lane.layout_kind == protocol.LayoutKind.DMA_STRIDED
                    and lane.dma_chunk_offsets() == offsets
                    and lane.dma_envelope_bytes() == envelope
                    for lane in plan.lanes
                )
            )
            lane = plan.lanes[0]
            selected = {
                offset + byte
                for offset in offsets
                for byte in range(lane.layout_inner_bytes)
            }
            self.assertEqual(envelope - len(selected), hole_bytes)

            with self.subTest(case=case.name):
                with contextlib.ExitStack() as stack:
                    directory = pathlib.Path(
                        stack.enter_context(
                            tempfile.TemporaryDirectory()
                        )
                    )
                    payload_path = directory / "payload.raw"
                    execution_probe.write_payload(payload_path, case)
                    payload = payload_path.read_bytes()
                    begin = execution_probe.V2_OUTPUT_GUARD_BYTES
                    compact = execution_probe.v2_expected_result(
                        plan.issue_identities()[0], lane, plan
                    )
                    rebuilt = b"".join(
                        payload[
                            begin + offset :
                            begin + offset + lane.layout_inner_bytes
                        ]
                        for offset in offsets
                    )
                    self.assertEqual(rebuilt, compact)
                    self.assertTrue(
                        all(
                            payload[begin + offset] == 0xCC
                            for offset in range(envelope)
                            if offset not in selected
                        )
                    )

                    output = bytearray(
                        [0xA5] * execution_probe.RESOURCE_BYTES
                    )
                    rdma_identity, wdma_identity = plan.issue_identities()
                    rdma_begin = (
                        execution_probe.V2_OUTPUT_SLOT_BASE
                        + rdma_identity.slot
                        * execution_probe.V2_OUTPUT_SLOT_STRIDE
                        + execution_probe.V2_OUTPUT_GUARD_BYTES
                    )
                    execution_probe.v2_scatter_compact(
                        output, rdma_begin, plan.lanes[0], compact
                    )
                    wdma_begin = (
                        execution_probe.V2_OUTPUT_SLOT_BASE
                        + wdma_identity.slot
                        * execution_probe.V2_OUTPUT_SLOT_STRIDE
                        + execution_probe.V2_OUTPUT_GUARD_BYTES
                    )
                    execution_probe.v2_scatter_compact(
                        output, wdma_begin, plan.lanes[1], compact
                    )
                    execution_probe.validate_output_payload_v2(
                        bytes(output), case, plan
                    )
            with self.assertRaisesRegex(ValueError, "bounded"):
                dataclasses.replace(
                    plan, schedule=protocol.Schedule.WINDOW
                ).request_words()
            with self.assertRaisesRegex(ValueError, "2-byte"):
                dataclasses.replace(
                    plan,
                    lanes=(
                        dataclasses.replace(
                            plan.lanes[0],
                            layout_stride0_bytes=(
                                plan.lanes[0].layout_stride0_bytes + 1
                            ),
                        ),
                        plan.lanes[1],
                    ),
                ).request_words()

    def test_tdma_crt_cases_are_exposed_as_exact_manual_cases(self) -> None:
        cases = execution_probe.SUITES["tdma-crt-manual"]
        self.assertIs(cases, execution_probe.NO_CARD_PROTOCOL_CASES)
        self.assertEqual(tuple(case.name for case in cases), (
            "tdma-crt-i8-physical16",
            "tdma-crt-bool-to-i8-physical17",
        ))
        expected_lanes = (
            (execution_probe.FMT_INT8, 16),
            (execution_probe.FMT_BOOL, 17),
        )
        for case, (expected_format, expected_bytes) in zip(
            cases, expected_lanes, strict=True
        ):
            plan = case.plan
            lane = plan.lanes[0]
            self.assertEqual(plan.schedule, protocol.Schedule.SERIAL)
            self.assertEqual(plan.issue_order(), (0,))
            self.assertEqual(lane.engine, protocol.Engine.TDMA)
            self.assertEqual(lane.issue_mode, protocol.IssueMode.WRAPPER)
            self.assertEqual(lane.element_format, expected_format)
            self.assertEqual(lane.transfer_bytes, expected_bytes)

            output = bytearray([0xA5] * execution_probe.RESOURCE_BYTES)
            identity = plan.issue_identities()[0]
            expected = execution_probe.v2_expected_result(identity, lane, plan)
            begin = (
                execution_probe.V2_OUTPUT_SLOT_BASE
                + execution_probe.V2_OUTPUT_GUARD_BYTES
            )
            output[begin : begin + len(expected)] = expected
            execution_probe.validate_output_payload_v2(
                bytes(output), case, plan
            )

    def test_three_lane_window_is_disjoint_only(self) -> None:
        plan = protocol.Plan(
            lanes=(
                lane(protocol.Engine.RDMA),
                lane(protocol.Engine.CT),
                lane(protocol.Engine.WDMA),
            ),
            rounds=4,
            effect_relation=protocol.EffectRelation.NONE,
            range_relation=protocol.RangeRelation.DISJOINT,
            schedule=protocol.Schedule.WINDOW,
            wait_kind=protocol.WaitKind.BY_WORKER,
            wait_worker_mask=1,
            seed=7,
        )
        self.assertEqual(
            plan.issue_order(), (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
        )
        invalid = dataclasses.replace(
            plan,
            effect_relation=protocol.EffectRelation.WAW,
            range_relation=protocol.RangeRelation.EXACT,
        )
        with self.assertRaisesRegex(ValueError, "three-lane"):
            invalid.request_words()

    def test_worker_catalog_routes_and_joins_exact_participants(self) -> None:
        expected = {
            "ct-worker0-raw-single": (
                (0,),
                protocol.Schedule.SERIAL,
                0b001,
                (0,),
            ),
            "ct-worker1-raw-single": (
                (1,),
                protocol.Schedule.SERIAL,
                0b010,
                (0,),
            ),
            "ct-worker2-raw-single": (
                (2,),
                protocol.Schedule.SERIAL,
                0b100,
                (0,),
            ),
            "ct-workers01-disjoint-r1-window": (
                (0, 1),
                protocol.Schedule.WINDOW,
                0b011,
                (0, 4),
            ),
            "ct-workers02-disjoint-r1-window": (
                (0, 2),
                protocol.Schedule.WINDOW,
                0b101,
                (0, 4),
            ),
            "ct-workers12-disjoint-r1-window": (
                (1, 2),
                protocol.Schedule.WINDOW,
                0b110,
                (0, 4),
            ),
            "ct-workers012-disjoint-r1-window": (
                (0, 1, 2),
                protocol.Schedule.WINDOW,
                0b111,
                (0, 4, 8),
            ),
        }
        self.assertEqual(
            {case.name for case in execution_probe.V2_WORKER_CASES},
            set(expected),
        )
        self.assertTrue(
            all(
                case in execution_probe.CALIBRATION_CASES
                for case in execution_probe.V2_WORKER_CASES
            )
        )
        for case in execution_probe.V2_WORKER_CASES:
            workers, schedule, wait_mask, issue_order = expected[case.name]
            plan = case.plan
            words = plan.request_words()
            self.assertEqual(
                tuple(item.worker for item in plan.lanes), workers
            )
            self.assertEqual(
                tuple(item.engine for item in plan.lanes),
                (protocol.Engine.CT,) * len(workers),
            )
            self.assertTrue(
                all(
                    item.issue_mode == protocol.IssueMode.RAW
                    and item.element_format == execution_probe.FMT_FP16
                    and item.transfer_bytes == 4096
                    for item in plan.lanes
                )
            )
            self.assertEqual(plan.rounds, 1)
            self.assertEqual(plan.schedule, schedule)
            self.assertEqual(plan.wait_kind, protocol.WaitKind.BY_WORKER)
            self.assertEqual(plan.wait_worker_mask, wait_mask)
            self.assertEqual(
                words[protocol.REQ["WAIT_WORKER_MASK"]], wait_mask
            )
            self.assertEqual(len(plan.issue_identities()), len(workers))
            self.assertEqual(plan.issue_order(), issue_order)

    def test_completion_scope_cases_observe_before_safety_drain(self) -> None:
        expected = {
            "ne-worker1-depth6-default-wait-window": (
                protocol.WaitKind.DEFAULT,
                0x6121,
                0,
            ),
            "ne-worker1-depth6-byworker-wait-window": (
                protocol.WaitKind.BY_WORKER,
                0x6122,
                0b010,
            ),
            "ne-worker1-depth6-local-fence-wait-window": (
                protocol.WaitKind.LOCAL_FENCE,
                0x6123,
                0,
            ),
        }
        cases = execution_probe.SUITES["completion-scope-manual"]
        self.assertIs(cases, execution_probe.V2_COMPLETION_SCOPE_CASES)
        self.assertEqual({case.name for case in cases}, set(expected))
        for case in cases:
            wait_kind, seed, wait_mask = expected[case.name]
            plan = case.plan
            words = plan.request_words()
            self.assertEqual(
                tuple(lane.engine for lane in plan.lanes),
                (
                    protocol.Engine.TDMA,
                    protocol.Engine.NE,
                    protocol.Engine.NE,
                ),
            )
            self.assertEqual(
                tuple(lane.worker for lane in plan.lanes), (0, 1, 1)
            )
            self.assertEqual(plan.lanes[0].transfer_bytes, 16)
            self.assertEqual(
                plan.lanes[0].element_format, execution_probe.FMT_INT8
            )
            self.assertTrue(
                all(
                    lane.transfer_bytes
                    == execution_probe.V2_NE_LARGE_RESULT_BYTES
                    and lane.element_format == execution_probe.FMT_FP16
                    for lane in plan.lanes[1:]
                )
            )
            self.assertEqual(plan.rounds, 3)
            self.assertEqual(plan.schedule, protocol.Schedule.WINDOW)
            self.assertEqual(
                plan.issue_order(), (0, 4, 8, 1, 5, 9, 2, 6, 10)
            )
            self.assertEqual(plan.wait_kind, wait_kind)
            self.assertEqual(plan.wait_worker_mask, wait_mask)
            self.assertEqual(plan.seed, seed)
            self.assertEqual(words[protocol.REQ["WAIT_KIND"]], wait_kind)
            self.assertEqual(
                words[protocol.REQ["WAIT_WORKER_MASK"]], wait_mask
            )
            self.assertEqual(
                execution_probe.v2_completion_marker(plan),
                (
                    0x49,
                    execution_probe.V2_SPM_SLOT_BASE
                    + 10 * execution_probe.V2_SPM_SLOT_STRIDE
                    + execution_probe.V2_NE_LARGE_WRITE_OFFSET
                    + execution_probe.V2_NE_LARGE_RESULT_BYTES
                    - 1,
                ),
            )

        source = (
            pathlib.Path(__file__).resolve().parent
            / "Inputs"
            / "wafer_ncc_probe_plan.c"
        ).read_text()
        execute = source.split(
            "uint32_t wafer_ncc_probe_execute_plan(", maxsplit=1
        )[1]
        requested_wait = execute.index("hooks->requested_wait(")
        boundary_oracle = execute.index(
            "WAFER_NCC_ORACLE_BOUNDARY", requested_wait
        )
        safety_drain = execute.index(
            "hooks->safety_drain(", boundary_oracle
        )
        self.assertLess(requested_wait, boundary_oracle)
        self.assertLess(boundary_oracle, safety_drain)

    def test_wait_overhead_cases_cover_every_engine(self) -> None:
        cases = execution_probe.SUITES["wait-overhead-manual"]
        self.assertIs(cases, execution_probe.V2_WAIT_OVERHEAD_CASES)
        self.assertEqual(
            {case.name for case in cases},
            {
                f"{engine.name.lower()}-worker0-r2-{spelling}"
                for engine in execution_probe.V2_ENGINES
                for spelling in ("wait-each", "wait-once")
            },
        )
        for engine in execution_probe.V2_ENGINES:
            engine_cases = tuple(
                case
                for case in cases
                if case.plan.lanes[0].engine == engine
            )
            self.assertEqual(
                {case.plan.schedule for case in engine_cases},
                {protocol.Schedule.SERIAL, protocol.Schedule.WINDOW},
            )
        for case in cases:
            plan = case.plan
            self.assertEqual(plan.rounds, 2)
            self.assertEqual(plan.issue_order(), (0, 1))
            self.assertEqual(plan.wait_kind, protocol.WaitKind.BY_WORKER)
            self.assertEqual(plan.wait_worker_mask, 1)
            self.assertEqual(len(plan.lanes), 1)
            self.assertEqual(plan.lanes[0].worker, 0)
            plan.request_words()
        self.assertLess(
            protocol.REC["PLAN_CYCLES"],
            protocol.REC["RECORD_GUARD"],
        )
        self.assertLess(
            protocol.REC["SERIAL_WAIT_CYCLES"],
            protocol.REC["RECORD_GUARD"],
        )
        self.assertLess(
            protocol.REC["SERIAL_WAIT_COUNT"],
            protocol.REC["RECORD_GUARD"],
        )

    def test_typed_hazard_catalog_and_composition_golden(self) -> None:
        self.assertEqual(len(execution_probe.V2_HAZARD_CASES), 24)
        execution_probe.validate_hazard_selection(
            execution_probe.V2_HAZARD_MANUAL_CASES
        )
        groups: dict[
            tuple[object, ...], set[protocol.Schedule]
        ] = {}
        for case in execution_probe.V2_HAZARD_CASES:
            plan = case.plan
            words = plan.request_words()
            self.assertEqual(
                words[protocol.REQ["FIRST_OPERAND"]],
                plan.first_operand,
            )
            self.assertEqual(
                words[protocol.REQ["SECOND_OPERAND"]],
                plan.second_operand,
            )
            key = (
                plan.effect_relation,
                plan.range_relation,
                tuple(lane.engine for lane in plan.lanes),
                plan.first_operand,
                plan.second_operand,
            )
            groups.setdefault(key, set()).add(plan.schedule)
            for identity in plan.issue_identities():
                expected = execution_probe.v2_expected_result(
                    identity, plan.lanes[identity.lane], plan
                )
                self.assertEqual(
                    len(expected),
                    plan.lanes[identity.lane].transfer_bytes,
                )
        self.assertEqual(len(groups), 12)
        self.assertTrue(
            all(
                schedules
                == {protocol.Schedule.SERIAL, protocol.Schedule.WINDOW}
                for schedules in groups.values()
            )
        )

        raw_partial = next(
            case
            for case in execution_probe.V2_HAZARD_CASES
            if case.name == "raw-rdma-ct-partial-r2-window"
        )
        ct_identity = next(
            identity
            for identity in raw_partial.plan.issue_identities()
            if identity.lane == 1 and identity.round == 0
        )
        actual = execution_probe.v2_expected_result(
            ct_identity,
            raw_partial.plan.lanes[ct_identity.lane],
            raw_partial.plan,
        )
        values = struct.unpack(f"<{len(actual) // 2}H", actual)
        self.assertEqual(
            set(values[: len(values) // 2]),
            {execution_probe.v2_half(5)},
        )
        self.assertEqual(
            set(values[len(values) // 2 :]),
            {execution_probe.v2_half(3)},
        )

    def test_hazard_requires_controls_but_not_positive_overlap(self) -> None:
        execution_probe.validate_hazard_selection(
            execution_probe.V2_DMA_STRIDE_MATRIX_CASES
        )
        hazards = tuple(
            case
            for case in execution_probe.V2_HAZARD_CASES
            if case.name
            in (
                "raw-rdma-ct-exact-r2-serial",
                "raw-rdma-ct-exact-r2-window",
            )
        )
        hazard = hazards[-1]
        with self.assertRaisesRegex(RuntimeError, "disjoint serial/window"):
            execution_probe.validate_hazard_selection((hazards[0],))
        with self.assertRaisesRegex(RuntimeError, "disjoint serial/window"):
            execution_probe.validate_hazard_selection((hazard,))
        controls = {
            (case.plan.rounds, case.plan.schedule): case
            for case in execution_probe.V2_HAZARD_DISJOINT_CONTROLS
            if execution_probe.hazard_pair_key(case.plan)
            == execution_probe.hazard_pair_key(hazard.plan)
        }
        serial = controls[(4, protocol.Schedule.SERIAL)]
        window = controls[(4, protocol.Schedule.WINDOW)]
        self.assertEqual(
            tuple(lane.engine for lane in serial.plan.lanes),
            (protocol.Engine.RDMA, protocol.Engine.CT),
        )
        self.assertEqual(
            tuple(lane.engine for lane in window.plan.lanes),
            (protocol.Engine.RDMA, protocol.Engine.CT),
        )
        execution_probe.validate_hazard_selection(
            (serial, window, *hazards)
        )
        with self.assertRaisesRegex(RuntimeError, "one common rounds"):
            execution_probe.validate_hazard_selection(
                (
                    controls[(2, protocol.Schedule.SERIAL)],
                    window,
                    *hazards,
                )
            )
        observations = [
            {
                "case": controls[
                    (2, protocol.Schedule.SERIAL)
                ].as_dict(),
                "execution_delta": {
                    "ct": 80,
                    "rdma": 70,
                    "full": 150,
                },
            },
            {
                "case": controls[
                    (2, protocol.Schedule.WINDOW)
                ].as_dict(),
                "execution_delta": {
                    "ct": 80,
                    "rdma": 70,
                    "full": 150,
                },
            },
            {
                "case": serial.as_dict(),
                "execution_delta": {
                    "ct": 80,
                    "rdma": 70,
                    "full": 150,
                },
            },
            {
                "case": window.as_dict(),
                "execution_delta": {
                    "ct": 80,
                    "rdma": 70,
                    "full": 120,
                },
            },
        ]
        self.assertIn(
            execution_probe.hazard_pair_key(hazard.plan),
            execution_probe.qualified_disjoint_pairs(observations),
        )
        mismatched_rounds = [
            observations[-2],
            {
                **observations[-1],
                "case": {
                    **window.as_dict(),
                    "rounds": 2,
                },
            },
        ]
        self.assertNotIn(
            execution_probe.hazard_pair_key(hazard.plan),
            execution_probe.qualified_disjoint_pairs(mismatched_rounds),
        )
        reversed_window = {
            **observations[-1],
            "case": {
                **window.as_dict(),
                "engines": ["ct", "rdma"],
            },
        }
        self.assertNotIn(
            execution_probe.hazard_pair_key(hazard.plan),
            execution_probe.qualified_disjoint_pairs(
                (observations[-2], reversed_window)
            ),
        )
        runner = pathlib.Path(execution_probe.__file__).read_text()
        self.assertNotIn(
            "has not passed the serial/window overlap qualification",
            runner,
        )
        self.assertIn(
            "zero overlap must\n        # not suppress",
            runner,
        )

    def test_hazard_control_catalog_adds_only_below_depth_r4(self) -> None:
        rounds_by_pair: dict[
            frozenset[protocol.Engine], set[int]
        ] = {}
        schedules_by_group: dict[
            tuple[frozenset[protocol.Engine], int],
            set[protocol.Schedule],
        ] = {}
        for case in execution_probe.V2_HAZARD_DISJOINT_CONTROLS:
            plan = case.plan
            pair = frozenset(lane.engine for lane in plan.lanes)
            rounds_by_pair.setdefault(pair, set()).add(plan.rounds)
            schedules_by_group.setdefault((pair, plan.rounds), set()).add(
                plan.schedule
            )
            self.assertTrue(
                all(
                    plan.rounds
                    < execution_probe.V2_DOCUMENTED_QUEUE_DEPTHS[
                        lane.engine
                    ]
                    for lane in plan.lanes
                )
            )

        self.assertEqual(
            rounds_by_pair[
                frozenset((protocol.Engine.CT, protocol.Engine.RDMA))
            ],
            {2, 4},
        )
        self.assertEqual(
            rounds_by_pair[
                frozenset((protocol.Engine.CT, protocol.Engine.WDMA))
            ],
            {2, 4},
        )
        for pair, rounds in rounds_by_pair.items():
            if protocol.Engine.TDMA in pair:
                self.assertEqual(rounds, {2})
        self.assertTrue(
            all(
                schedules
                == {protocol.Schedule.SERIAL, protocol.Schedule.WINDOW}
                for schedules in schedules_by_group.values()
            )
        )

    def test_range_requires_validity_bit(self) -> None:
        plan = protocol.Plan(
            lanes=(lane(protocol.Engine.CT),),
            rounds=1,
            effect_relation=protocol.EffectRelation.NONE,
            range_relation=protocol.RangeRelation.DISJOINT,
            schedule=protocol.Schedule.SERIAL,
            wait_kind=protocol.WaitKind.BY_WORKER,
            wait_worker_mask=1,
            seed=3,
        )
        words = [0] * protocol.RECORD_WORDS
        words[protocol.REC["MAGIC"]] = protocol.RECORD_MAGIC
        words[protocol.REC["SCHEMA_AND_WORDS"]] = (
            protocol.SCHEMA << 32
        ) | protocol.RECORD_WORDS
        words[protocol.REC["STATUS"]] = protocol.Status.OK
        words[protocol.REC["FLAGS"]] = protocol.ALL_PHASE_FLAGS
        words[protocol.REC["COMMAND"]] = plan.command
        words[protocol.REC["LANE_COUNT"]] = 1
        words[protocol.REC["ROUNDS"]] = 1
        words[protocol.REC["ISSUE_COUNT"]] = 1
        words[protocol.REC["EFFECT_RELATION"]] = plan.effect_relation
        words[protocol.REC["RANGE_RELATION"]] = plan.range_relation
        words[protocol.REC["SCHEDULE"]] = plan.schedule
        words[protocol.REC["WAIT_KIND"]] = plan.wait_kind
        words[protocol.REC["WAIT_WORKER_MASK"]] = plan.wait_worker_mask
        words[protocol.REC["FIRST_OPERAND"]] = plan.first_operand
        words[protocol.REC["SECOND_OPERAND"]] = plan.second_operand
        words[protocol.REC["SEED"]] = plan.seed
        words[protocol.REC["SAMPLE"]] = plan.sample
        identity = plan.issue_identities()[0]
        base = protocol.ISSUE_BASE
        words[base + protocol.ISSUE["ORDINAL"]] = identity.ordinal
        words[base + protocol.ISSUE["LANE"]] = identity.lane
        words[base + protocol.ISSUE["ROUND"]] = identity.round
        words[base + protocol.ISSUE["SLOT"]] = identity.slot
        words[base + protocol.ISSUE["TAG"]] = identity.tag
        words[base + protocol.ISSUE["ENGINE"]] = protocol.Engine.CT
        words[base + protocol.ISSUE["WORKER"]] = 0
        words[base + protocol.ISSUE["READ0_BEGIN"]] = 0x1000
        words[base + protocol.ISSUE["READ0_END"]] = 0x10FF
        with self.assertRaisesRegex(ValueError, "validity flag"):
            protocol.validate_record(words, plan)

    def test_raw_observation_requires_actual_ranges_and_packet_flag(self) -> None:
        plan = protocol.Plan(
            lanes=(lane(protocol.Engine.CT),),
            rounds=1,
            effect_relation=protocol.EffectRelation.NONE,
            range_relation=protocol.RangeRelation.DISJOINT,
            schedule=protocol.Schedule.SERIAL,
            wait_kind=protocol.WaitKind.BY_WORKER,
            wait_worker_mask=1,
            seed=3,
        )
        case = execution_probe.GenericProbeCase("ct-range", plan)
        identity = plan.issue_identities()[0]
        read0 = execution_probe.v2_spm_address(
            identity.slot, execution_probe.V2_SPM_READ0_OFFSET
        )
        read1 = execution_probe.v2_spm_address(
            identity.slot, execution_probe.V2_SPM_READ1_OFFSET
        )
        write = execution_probe.v2_spm_address(
            identity.slot, execution_probe.V2_SPM_WRITE_OFFSET
        )
        observation = protocol.IssueObservation(
            identity=identity,
            engine=protocol.Engine.CT,
            worker=0,
            execute_rc=1,
            inter_type=0,
            read0=protocol.InclusiveRange(read0, read0 + 4095),
            read1=protocol.InclusiveRange(read1, read1 + 4095),
            write=protocol.InclusiveRange(write, write + 4095),
            boundary_mismatches=0,
            boundary_guard_mismatches=0,
            final_mismatches=0,
            final_guard_mismatches=0,
            flags=(
                protocol.READ0_VALID
                | protocol.READ1_VALID
                | protocol.WRITE_VALID
                | protocol.PACKET_OBSERVED
            ),
            control_after_issue=0,
            execute_cycles=1,
        )
        execution_probe.validate_observed_ranges_v2(
            case, plan, (observation,)
        )
        with self.assertRaisesRegex(RuntimeError, "issue flags"):
            execution_probe.validate_observed_ranges_v2(
                case,
                plan,
                (
                    dataclasses.replace(
                        observation,
                        flags=observation.flags
                        & ~protocol.PACKET_OBSERVED,
                    ),
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "range is"):
            execution_probe.validate_observed_ranges_v2(
                case,
                plan,
                (
                    dataclasses.replace(
                        observation,
                        write=protocol.InclusiveRange(write, write + 4094),
                    ),
                ),
            )

    def test_two_lane_overlap_uses_pairwise_excess(self) -> None:
        metrics = execution_probe.overlap_metrics(
            ("rdma", "ct"),
            {"rdma": 80, "ct": 70, "full": 120},
        )
        self.assertEqual(metrics["pairwise_excess"], 30)
        self.assertIs(metrics["overlap_observed"], True)
        self.assertNotIn("triple_overlap_observed", metrics)

    def test_three_lane_overlap_requires_simultaneous_triple(self) -> None:
        pairwise_only = execution_probe.overlap_metrics(
            ("rdma", "ct", "wdma"),
            {"rdma": 80, "ct": 70, "wdma": 60, "full": 120},
        )
        self.assertEqual(pairwise_only["pairwise_excess"], 90)
        self.assertEqual(
            pairwise_only["simultaneous_triple_lower_bound"], 0
        )
        self.assertIs(pairwise_only["triple_overlap_observed"], False)

        triple = execution_probe.overlap_metrics(
            ("rdma", "ct", "wdma"),
            {"rdma": 100, "ct": 90, "wdma": 80, "full": 120},
        )
        self.assertEqual(triple["pairwise_excess"], 150)
        self.assertEqual(triple["simultaneous_triple_lower_bound"], 30)
        self.assertIs(triple["triple_overlap_observed"], True)

    def test_require_overlap_needs_paired_controls(self) -> None:
        window = next(
            case
            for case in execution_probe.V2_PAIR_CASES
            if case.name == "ct-rdma-disjoint-r2-window"
        )
        with self.assertRaisesRegex(RuntimeError, "paired serial/window"):
            execution_probe.validate_overlap_selection((window,))
        serial = next(
            case
            for case in execution_probe.V2_PAIR_CASES
            if case.name == "ct-rdma-disjoint-r2-serial"
        )
        execution_probe.validate_overlap_selection((serial, window))

    def test_overlap_control_must_be_exceeded(self) -> None:
        case = {
            "engines": ["ct", "rdma"],
            "rounds": 2,
        }
        observations = [
            {
                "case": {**case, "schedule": "serial"},
                "execution_delta": {"ct": 80, "rdma": 70, "full": 140},
            },
            {
                "case": {**case, "schedule": "window"},
                "execution_delta": {"ct": 80, "rdma": 70, "full": 140},
            },
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "overlap criterion"):
                execution_probe.report_overlap(
                    observations, require_overlap=True
                )
        observations[1]["execution_delta"] = {
            "ct": 80,
            "rdma": 70,
            "full": 120,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            execution_probe.report_overlap(observations, require_overlap=True)

    def test_depth_plus_one_is_typed_and_requires_a_wait(self) -> None:
        case = next(
            item
            for item in execution_probe.V2_DEPTH_PLUS_ONE_CASES
            if item.plan.lanes[0].engine == protocol.Engine.CT
        )
        plan = case.plan
        self.assertEqual(plan.flags, protocol.TIGHT_DEPTH_PLUS_ONE)
        self.assertEqual(plan.issue_limit, 7)
        self.assertEqual(len(plan.issue_identities()), 7)
        self.assertEqual(plan.issue_order(), (0, 4, 1, 5, 2, 6, 3))
        self.assertEqual(plan.wait_kind, protocol.WaitKind.BY_WORKER)
        self.assertEqual(
            plan.request_words()[protocol.REQ["ISSUE_LIMIT"]], 7
        )
        invalid = dataclasses.replace(
            plan,
            wait_kind=protocol.WaitKind.NONE,
            wait_worker_mask=0,
        )
        with self.assertRaisesRegex(ValueError, "matching wait"):
            invalid.request_words()

    def test_multi_issue_catalog_stays_below_documented_depth(self) -> None:
        for engine, depth in (
            execution_probe.V2_DOCUMENTED_QUEUE_DEPTHS.items()
        ):
            cases = [
                case
                for case in execution_probe.V2_MULTI_ISSUE_CASES
                if case.plan.lanes[0].engine == engine
            ]
            issue_counts = {
                len(case.plan.issue_identities()) for case in cases
            }
            self.assertEqual(
                issue_counts,
                {2}
                if engine == protocol.Engine.TDMA
                else {2, 4},
            )
            self.assertTrue(all(count < depth for count in issue_counts))

    def test_issue_paths_and_leaf_bindings_are_concrete(self) -> None:
        self.assertEqual(len(execution_probe.V2_ISSUE_PATH_CASES), 10)
        for engine in execution_probe.V2_ENGINES:
            cases = tuple(
                case
                for case in execution_probe.V2_ISSUE_PATH_CASES
                if case.plan.lanes[0].engine == engine
            )
            self.assertEqual(
                {case.plan.lanes[0].issue_mode for case in cases},
                {protocol.IssueMode.RAW, protocol.IssueMode.WRAPPER},
            )
            self.assertTrue(
                all(
                    case.plan.rounds == 2
                    and case.plan.schedule == protocol.Schedule.WINDOW
                    for case in cases
                )
            )
        execution_probe.validate_no_card_protocol_cases()
        self.assertTrue(execution_probe.CALIBRATION_LEAF_BINDINGS)
        self.assertTrue(
            all(execution_probe.CALIBRATION_LEAF_BINDINGS.values())
        )
        self.assertFalse(execution_probe.V2_DEFERRED_CASES)
        self.assertTrue(
            all(
                case.plan.is_strided_dependency_observation()
                for case in execution_probe.CALIBRATION_LEAF_BINDINGS[
                    "dependency-strided-envelope"
                ]
            )
        )
        self.assertTrue(
            any(
                lane.transfer_bytes == 65536
                for case in execution_probe.CALIBRATION_LEAF_BINDINGS[
                    "large-backlog-compute-movement"
                ]
                for lane in case.plan.lanes
            )
        )
        protocol_negative_groups = {
            "execute-engine-none-static-negative": "execute-engine-none",
            "execute-worker-out-of-range-static-negative": (
                "execute-worker-out-of-range"
            ),
            "wait-mask-nonparticipant-static-negative": (
                "wait-mask-nonparticipant"
            ),
        }
        for group, name in protocol_negative_groups.items():
            bound = execution_probe.CALIBRATION_LEAF_BINDINGS[group]
            self.assertEqual(tuple(case.name for case in bound), (name,))
            self.assertEqual(bound[0].disposition, "static-negative")
            self.assertIn("before serialization", bound[0].completion_oracle)
        with self.assertRaisesRegex(ValueError, "Engine.NONE"):
            dataclasses.replace(
                execution_probe.V2_SINGLE_CASES[0].plan,
                lanes=(
                    dataclasses.replace(
                        execution_probe.V2_SINGLE_CASES[0].plan.lanes[0],
                        engine=protocol.Engine.NONE,
                    ),
                ),
            ).request_words()
        with self.assertRaisesRegex(ValueError, "worker"):
            dataclasses.replace(
                execution_probe.V2_SINGLE_CASES[0].plan,
                lanes=(
                    dataclasses.replace(
                        execution_probe.V2_SINGLE_CASES[0].plan.lanes[0],
                        worker=3,
                    ),
                ),
            ).request_words()
        with self.assertRaisesRegex(ValueError, "non-participant"):
            dataclasses.replace(
                execution_probe.V2_SINGLE_CASES[0].plan,
                wait_worker_mask=0b010,
            ).request_words()

    def test_documented_depth_catalog_is_exact_and_one_shot(self) -> None:
        for engine, depth in (
            execution_probe.V2_DOCUMENTED_QUEUE_DEPTHS.items()
        ):
            cases = [
                case
                for case in execution_probe.V2_DOCUMENTED_DEPTH_CASES
                if case.plan.lanes[0].engine == engine
            ]
            self.assertEqual(len(cases), 1)
            self.assertEqual(
                len(cases[0].plan.issue_identities()), depth
            )
            self.assertEqual(cases[0].plan.flags, 0)
            self.assertEqual(cases[0].plan.issue_limit, 0)

    def test_depth_plus_one_catalog_is_exact(self) -> None:
        for engine, depth in (
            execution_probe.V2_DOCUMENTED_QUEUE_DEPTHS.items()
        ):
            cases = [
                case
                for case in execution_probe.V2_DEPTH_PLUS_ONE_CASES
                if case.plan.lanes[0].engine == engine
            ]
            self.assertEqual(len(cases), 1)
            self.assertEqual(
                len(cases[0].plan.issue_identities()), depth + 1
            )
            self.assertEqual(
                cases[0].plan.flags, protocol.TIGHT_DEPTH_PLUS_ONE
            )

    def test_large_tight_occupancy_stays_at_depth_plus_one(self) -> None:
        self.assertEqual(
            {
                case.plan.lanes[0].engine
                for case in execution_probe.V2_ACTIVE_OCCUPANCY_CASES
            },
            set(execution_probe.V2_ENGINES),
        )
        for case in execution_probe.V2_ACTIVE_OCCUPANCY_CASES:
            plan = case.plan
            engine = plan.lanes[0].engine
            depth = execution_probe.V2_DOCUMENTED_QUEUE_DEPTHS[engine]
            self.assertEqual(
                tuple(lane.engine for lane in plan.lanes),
                (engine, engine),
            )
            self.assertTrue(
                all(
                    lane.issue_mode == protocol.IssueMode.RAW
                    and lane.transfer_bytes
                    == execution_probe.V2_ACTIVE_OCCUPANCY_BYTES[engine]
                    for lane in plan.lanes
                )
            )
            self.assertEqual(plan.flags, protocol.TIGHT_DEPTH_PLUS_ONE)
            self.assertEqual(plan.issue_limit, depth + 1)
            self.assertEqual(len(plan.issue_identities()), depth + 1)
            self.assertIn(
                case,
                execution_probe.CALIBRATION_LEAF_BINDINGS[
                    "depth-plus-one-manual"
                ],
            )
            self.assertNotIn(
                case, execution_probe.BOARD_ALL_SAFE_CASES
            )
            self.assertIn(
                case, execution_probe.BOARD_ALL_PREFLIGHT_CASES
            )
            self.assertEqual(
                execution_probe.case_sample_count(case, 3), 3
            )
            plan.request_words()

    def test_depth_plus_one_report_uses_tight_issue_order(self) -> None:
        case = next(
            item
            for item in execution_probe.V2_DEPTH_PLUS_ONE_CASES
            if item.plan.lanes[0].engine == protocol.Engine.CT
        )
        issues = [
            {
                "identity": dataclasses.asdict(identity),
                "execute_cycles": identity.slot + 10,
                "control_after_issue": (
                    0x102
                    if identity.slot == case.plan.issue_order()[-1]
                    else 0
                ),
            }
            for identity in case.plan.issue_identities()
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            execution_probe.report_depth_plus_one(
                [
                    {
                        "case": case.as_dict(),
                        "issues": issues,
                        "blocking_delta": {"worker0.ct": 9},
                    }
                ]
            )
        self.assertIn("completed-with-pmu-backpressure", output.getvalue())
        self.assertIn('"control_after_tight_window": 258', output.getvalue())

    def test_active_occupancy_report_separates_activity_and_backpressure(
        self,
    ) -> None:
        for case in execution_probe.V2_ACTIVE_OCCUPANCY_CASES:
            engine_name = case.plan.lanes[0].engine.name.lower()
            issues = [
                {
                    "identity": dataclasses.asdict(identity),
                    "execute_cycles": identity.ordinal + 10,
                    "control_after_issue": (
                        0x2
                        if identity.slot
                        == case.plan.issue_order()[-1]
                        else 0
                    ),
                }
                for identity in case.plan.issue_identities()
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                execution_probe.report_active_occupancy(
                    [
                        {
                            "case": case.as_dict(),
                            "sample": sample,
                            "issues": issues,
                            "blocking_delta": {
                                f"worker0.{engine_name}": 17
                            },
                        }
                        for sample in range(3)
                    ],
                    3,
                )
            rendered = output.getvalue()
            self.assertIn(f'"engine": "{engine_name}"', rendered)
            self.assertIn('"sample_count": 3', rendered)
            self.assertIn('"active_samples": 3', rendered)
            self.assertIn('"backpressure_samples": 3', rendered)
            self.assertIn('"active_at_observation": true', rendered)
            self.assertIn('"pmu_backpressure_observed": true', rendered)
            self.assertIn(
                '"resident_count": "not-observable-from-control"',
                rendered,
            )
            self.assertEqual(
                execution_probe.case_sample_count(case, 3), 3
            )

    def test_constructor_observation_is_nonnull_and_releases_builder(self) -> None:
        (case,) = execution_probe.V2_CONSTRUCTOR_CASES
        self.assertTrue(case.plan.is_constructor_observation())
        self.assertEqual(
            case.plan.flags, protocol.CONSTRUCTOR_OBSERVATION
        )
        self.assertEqual(
            execution_probe.CALIBRATION_LEAF_BINDINGS[
                "constructor-return-address-nonnull"
            ],
            execution_probe.V2_CONSTRUCTOR_CASES,
        )
        self.assertNotIn(
            "constructor-zero-address",
            execution_probe.CALIBRATION_LEAF_BINDINGS,
        )
        source = (
            pathlib.Path(__file__).resolve().parent
            / "Inputs"
            / "wafer_ncc_execution_probe.c"
        ).read_text()
        self.assertIn(
            "instruction->has_owner = instruction->owner != NULL;", source
        )
        prepare = source.split(
            "static int wafer_ncc_v2_prepare(", maxsplit=1
        )[1].split(
            "static int wafer_ncc_v2_issue(", maxsplit=1
        )[0]
        failed_materialization = prepare.index("if (!built) {")
        failed_release = prepare.index(
            "wafer_ncc_probe_release(instruction)",
            failed_materialization,
        )
        failed_release_flag = prepare.index(
            "WAFER_NCC_ISSUE_PREPARE_BUILDER_RELEASED",
            failed_release,
        )
        failed_return = prepare.index("return 1;", failed_release_flag)
        self.assertLess(failed_materialization, failed_release)
        self.assertLess(failed_release, failed_release_flag)
        self.assertLess(failed_release_flag, failed_return)
        capture = prepare.index("context->constructor_address")
        release = prepare.index(
            "wafer_ncc_probe_release(instruction)", capture
        )
        self.assertLess(capture, release)
        self.assertIn(
            "WAFER_NCC_REC_CONSTRUCTOR_ADDRESS", source
        )
        self.assertIn(
            "WAFER_NCC_RECORD_CONSTRUCTOR_CAPTURED", source
        )

    def test_subset_join_and_ordered_boundaries_are_concrete(self) -> None:
        self.assertEqual(
            {
                case.plan.wait_worker_mask
                for case in execution_probe.V2_SUBSET_JOIN_CASES
            },
            {1, 2, 3, 4, 5, 6},
        )
        for case in execution_probe.V2_SUBSET_JOIN_CASES:
            self.assertEqual(case.plan.schedule, protocol.Schedule.WINDOW)
            self.assertEqual(
                tuple(lane.worker for lane in case.plan.lanes), (0, 1, 2)
            )
            self.assertEqual(
                case.plan.issue_order(), (0, 4, 8)
            )
            case.plan.request_words()
        self.assertEqual(
            {
                tuple(lane.engine for lane in case.plan.lanes)
                for case in execution_probe.V2_PRODUCER_CONSUMER_CASES
            },
            {
                (protocol.Engine.RDMA, protocol.Engine.CT),
                (protocol.Engine.CT, protocol.Engine.WDMA),
                (protocol.Engine.NE, protocol.Engine.WDMA),
                (protocol.Engine.TDMA, protocol.Engine.CT),
                (protocol.Engine.TDMA, protocol.Engine.NE),
            },
        )
        self.assertTrue(
            all(
                case.plan.is_ordered_producer_consumer()
                for case in execution_probe.V2_PRODUCER_CONSUMER_CASES
            )
        )
        self.assertEqual(
            {case.name for case in execution_probe.V2_KCORE_BOUNDARY_CASES},
            {
                "ct-to-kcore-read-boundary",
                "ne-to-kcore-read-boundary",
            },
        )

    def test_strided_dependency_has_serial_window_hole_oracles(self) -> None:
        groups: dict[
            tuple[str, protocol.EffectRelation, protocol.RangeRelation],
            set[protocol.Schedule],
        ] = {}
        for case in execution_probe.V2_STRIDED_DEPENDENCY_CASES:
            self.assertTrue(
                case.plan.is_strided_dependency_observation()
            )
            dimension = case.name.split("-")[2]
            key = (
                dimension,
                case.plan.effect_relation,
                case.plan.range_relation,
            )
            groups.setdefault(key, set()).add(case.plan.schedule)
            lane = case.plan.lanes[0]
            touched = {
                offset + byte
                for offset in lane.dma_chunk_offsets()
                for byte in range(lane.layout_inner_bytes)
            }
            self.assertLess(
                len(touched), lane.dma_envelope_bytes()
            )
            case.plan.request_words()
        self.assertTrue(
            all(
                schedules
                == {
                    protocol.Schedule.SERIAL,
                    protocol.Schedule.WINDOW,
                }
                for schedules in groups.values()
            )
        )
        self.assertEqual(
            {
                effect: sum(
                    case.plan.effect_relation == effect
                    for case in execution_probe.V2_STRIDED_DEPENDENCY_CASES
                )
                for effect in (
                    protocol.EffectRelation.RAW,
                    protocol.EffectRelation.WAR,
                    protocol.EffectRelation.WAW,
                    protocol.EffectRelation.RAR,
                )
            },
            {
                protocol.EffectRelation.RAW: 6,
                protocol.EffectRelation.WAR: 6,
                protocol.EffectRelation.WAW: 18,
                protocol.EffectRelation.RAR: 6,
            },
        )
        self.assertTrue(
            all(
                execution_probe.case_sample_count(case, 3) == 3
                for case in execution_probe.V2_STRIDED_DEPENDENCY_CASES
            )
        )

    def test_strided_war_rar_waw_sinks_have_strong_oracles(self) -> None:
        def one(
            effect: protocol.EffectRelation,
            relation: protocol.RangeRelation,
        ) -> execution_probe.GenericProbeCase:
            return next(
                case
                for case in execution_probe.V2_STRIDED_DEPENDENCY_CASES
                if "strided-1d-" in case.name
                and case.plan.schedule == protocol.Schedule.SERIAL
                and case.plan.effect_relation == effect
                and case.plan.range_relation == relation
            )

        war = one(
            protocol.EffectRelation.WAR,
            protocol.RangeRelation.STRIDED_ENVELOPE,
        )
        war_identities = war.plan.issue_identities()
        war_pre = execution_probe.v2_expected_result(
            war_identities[0], war.plan.lanes[0], war.plan
        )
        war_final = execution_probe.v2_expected_result(
            war_identities[1], war.plan.lanes[1], war.plan
        )
        self.assertEqual(
            war_pre,
            bytes(
                execution_probe.v2_pattern_byte(
                    execution_probe.V2_STRIDED_INITIAL_SOURCE_SLOT,
                    index,
                )
                for index in range(len(war_pre))
            ),
        )
        self.assertEqual(
            war_final,
            bytes(
                execution_probe.v2_pattern_byte(
                    protocol.MAX_ROUNDS, index
                )
                for index in range(len(war_final))
            ),
        )
        self.assertNotEqual(war_pre, war_final)

        rar = one(
            protocol.EffectRelation.RAR,
            protocol.RangeRelation.STRIDED_ENVELOPE,
        )
        self.assertEqual(
            *(
                execution_probe.v2_expected_result(
                    identity,
                    rar.plan.lanes[identity.lane],
                    rar.plan,
                )
                for identity in rar.plan.issue_identities()
            )
        )

        partial = one(
            protocol.EffectRelation.WAW,
            protocol.RangeRelation.PARTIAL,
        )
        first_identity, second_identity = partial.plan.issue_identities()
        first_result = execution_probe.v2_expected_result(
            first_identity, partial.plan.lanes[0], partial.plan
        )
        second_result = execution_probe.v2_expected_result(
            second_identity, partial.plan.lanes[1], partial.plan
        )
        first_base, second_base = (
            execution_probe.v2_strided_dependency_bases(partial.plan)
        )
        saw_first_only = False
        saw_second_wins = False
        cursor = 0
        for offset in partial.plan.lanes[0].dma_chunk_offsets():
            for byte in range(partial.plan.lanes[0].layout_inner_bytes):
                address = first_base + offset + byte
                second_index = execution_probe.v2_strided_compact_index(
                    partial.plan.lanes[0], second_base, address
                )
                if second_index is None:
                    saw_first_only = True
                    expected = execution_probe.v2_pattern_byte(0, cursor)
                else:
                    saw_second_wins = True
                    expected = execution_probe.v2_pattern_byte(
                        protocol.MAX_ROUNDS, second_index
                    )
                self.assertEqual(first_result[cursor], expected)
                cursor += 1
        self.assertTrue(saw_first_only)
        self.assertTrue(saw_second_wins)
        self.assertEqual(
            second_result,
            bytes(
                execution_probe.v2_pattern_byte(
                    protocol.MAX_ROUNDS, index
                )
                for index in range(len(second_result))
            ),
        )

        exact = one(
            protocol.EffectRelation.WAW,
            protocol.RangeRelation.EXACT,
        )
        exact_evidence = execution_probe.v2_strided_dependency_evidence(
            exact.plan
        )
        self.assertTrue(exact_evidence["command_semantics_sufficient"])
        self.assertFalse(
            exact_evidence["first_payload_independently_proven"]
        )
        self.assertIn(
            "first-payload-not-independently-observable",
            exact_evidence["evidence"],
        )
        generic_exact_waw = next(
            case.plan
            for case in execution_probe.V2_HAZARD_CASES
            if case.plan.effect_relation == protocol.EffectRelation.WAW
            and case.plan.range_relation == protocol.RangeRelation.EXACT
        )
        generic_evidence = execution_probe.v2_waw_evidence(
            generic_exact_waw
        )
        self.assertTrue(
            generic_evidence["two_write_instruction_counts_required"]
        )
        self.assertFalse(
            generic_evidence["first_payload_independently_proven"]
        )
        for case in execution_probe.V2_STRIDED_DEPENDENCY_CASES:
            if case.plan.effect_relation == protocol.EffectRelation.RAW:
                continue
            output = bytearray([0xA5] * execution_probe.RESOURCE_BYTES)
            for identity in case.plan.issue_identities():
                lane = case.plan.lanes[identity.lane]
                begin = (
                    execution_probe.V2_OUTPUT_SLOT_BASE
                    + identity.slot
                    * execution_probe.V2_OUTPUT_SLOT_STRIDE
                    + execution_probe.V2_OUTPUT_GUARD_BYTES
                )
                execution_probe.v2_scatter_compact(
                    output,
                    begin,
                    lane,
                    execution_probe.v2_expected_result(
                        identity, lane, case.plan
                    ),
                )
            execution_probe.validate_output_payload_v2(
                bytes(output), case, case.plan
            )
            lane = case.plan.lanes[0]
            touched = {
                offset + byte
                for offset in lane.dma_chunk_offsets()
                for byte in range(lane.layout_inner_bytes)
            }
            hole = next(
                byte
                for byte in range(lane.dma_envelope_bytes())
                if byte not in touched
            )
            first_slot_begin = (
                execution_probe.V2_OUTPUT_SLOT_BASE
                + execution_probe.V2_OUTPUT_GUARD_BYTES
            )
            output[first_slot_begin + hole] = 0xEE
            with self.assertRaisesRegex(
                RuntimeError, "outside record/result"
            ):
                execution_probe.validate_output_payload_v2(
                    bytes(output), case, case.plan
                )

    def test_large_backlog_resource_and_oracles_are_bounded(self) -> None:
        self.assertEqual(len(execution_probe.BOARD_ALL_SAFE_CASES), 196)
        self.assertEqual(
            sum(
                execution_probe.case_sample_count(case, 3)
                for case in execution_probe.BOARD_ALL_SAFE_CASES
            ),
            468,
        )
        for case in (
            execution_probe.V2_DOCUMENTED_DEPTH_CASES
            + execution_probe.V2_DEPTH_PLUS_ONE_CASES
        ):
            self.assertNotIn(case, execution_probe.BOARD_ALL_SAFE_CASES)
            self.assertIn(case, execution_probe.BOARD_ALL_PREFLIGHT_CASES)
        self.assertEqual(len(execution_probe.V2_LARGE_BACKLOG_CASES), 12)
        self.assertEqual(
            len(execution_probe.V2_LARGE_BACKLOG_SINGLE_CASES), 8
        )
        self.assertEqual(len(execution_probe.V2_LARGE_OVERLAP_CASES), 4)
        self.assertTrue(
            all(
                execution_probe.case_sample_count(case, 3) == 1
                for case in execution_probe.V2_LARGE_BACKLOG_SINGLE_CASES
            )
        )
        self.assertTrue(
            all(
                execution_probe.case_sample_count(case, 3) == 3
                for case in execution_probe.V2_LARGE_OVERLAP_CASES
            )
        )
        self.assertTrue(
            any(
                lane.transfer_bytes == 65536
                for case in execution_probe.V2_LARGE_BACKLOG_CASES
                for lane in case.plan.lanes
            )
        )
        self.assertTrue(
            any(
                execution_probe.v2_is_large_ne_lane(lane)
                for case in execution_probe.V2_LARGE_BACKLOG_CASES
                for lane in case.plan.lanes
            )
        )
        execution_probe.validate_catalog_resource_layout(
            execution_probe.BOARD_ALL_PREFLIGHT_CASES
        )
        maximum_ddr_end = max(
            execution_probe.V2_OUTPUT_SLOT_BASE
            + identity.slot * execution_probe.V2_OUTPUT_SLOT_STRIDE
            + 2 * execution_probe.V2_OUTPUT_GUARD_BYTES
            + case.plan.lanes[identity.lane].dma_envelope_bytes()
            for case in execution_probe.BOARD_ALL_PREFLIGHT_CASES
            for identity in case.plan.issue_identities()
        )
        self.assertEqual(
            maximum_ddr_end, execution_probe.RESOURCE_BYTES
        )
        self.assertEqual(
            execution_probe.RESOURCE_BYTES,
            execution_probe.RESOURCE_ELEMENTS * 4,
        )
        self.assertIn(
            f"tensor<{execution_probe.RESOURCE_ELEMENTS}xf32>",
            execution_probe.MODULE,
        )
        self.assertEqual(
            {
                tuple(signature["shape"])
                for signature in (
                    *execution_probe.METADATA["input_signature"],
                    *execution_probe.METADATA["output_signature"],
                )
            },
            {(execution_probe.RESOURCE_ELEMENTS,)},
        )
        self.assertLessEqual(
            execution_probe.V2_NE_LARGE_WRITE_OFFSET
            + execution_probe.V2_NE_LARGE_RESULT_BYTES
            + execution_probe.V2_OUTPUT_GUARD_BYTES,
            execution_probe.V2_SPM_SLOT_STRIDE,
        )

    def test_large_overlap_summary_requires_three_paired_samples(self) -> None:
        observations: list[dict[str, object]] = []
        for case in execution_probe.V2_LARGE_OVERLAP_CASES:
            engines = tuple(
                lane.engine.name.lower() for lane in case.plan.lanes
            )
            for sample in range(3):
                execution = {
                    "full": (
                        130
                        if case.plan.schedule == protocol.Schedule.SERIAL
                        else 110
                    ),
                    engines[0]: 80,
                    engines[1]: 70,
                }
                observations.append(
                    {
                        "case": case.as_dict(),
                        "sample": sample,
                        "execution_delta": execution,
                    }
                )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            execution_probe.report_stable_large_overlap(observations, 3)
        rendered = output.getvalue()
        self.assertIn('"repeats_per_case": 3', rendered)
        self.assertEqual(rendered.count("ncc_overlap_decision:"), 2)
        with self.assertRaisesRegex(RuntimeError, "requested repeats"):
            execution_probe.report_stable_large_overlap(
                observations[:-1], 3
            )

    def test_double_slot_is_hardware_only_and_has_pipeline_order(self) -> None:
        expected_orders = {
            1: (0, 4, 8),
            2: (0, 1, 4, 5, 8, 9),
            3: (0, 1, 4, 2, 5, 8, 6, 9, 10),
            4: (0, 1, 4, 2, 5, 8, 3, 6, 9, 7, 10, 11),
        }
        self.assertEqual(
            len(execution_probe.V2_DOUBLE_SLOT_OBSERVATION_CASES), 8
        )
        for case in execution_probe.V2_DOUBLE_SLOT_OBSERVATION_CASES:
            self.assertTrue(case.plan.is_double_slot_observation())
            self.assertEqual(
                case.plan.issue_order(), expected_orders[case.plan.rounds]
            )
            self.assertIn(
                case.plan.schedule,
                (protocol.Schedule.SERIAL, protocol.Schedule.WINDOW),
            )
        self.assertIn(
            "hardware_observation_only",
            pathlib.Path(
                execution_probe.__file__
            ).read_text(),
        )
        cmake = (
            pathlib.Path(__file__).resolve().parents[1] / "CMakeLists.txt"
        ).read_text()
        self.assertIn("wafer-board-ncc-all-safe-observations", cmake)
        self.assertIn("wafer-runtime-ncc-safe-observations-no-card", cmake)
        self.assertIn("join${_wafer_ncc_join_mask}-unjoined-boundary", cmake)


if __name__ == "__main__":
    unittest.main()
