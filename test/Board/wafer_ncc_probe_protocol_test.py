#!/usr/bin/env python3

import dataclasses
import contextlib
import io
import pathlib
import struct
import sys
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
        self.assertEqual(protocol.SCHEMA, 2)
        self.assertEqual(protocol.MAX_LANES, 3)
        self.assertEqual(protocol.MAX_ROUNDS, 4)
        self.assertEqual(protocol.MAX_ISSUES, 12)
        self.assertEqual(protocol.REQUEST_WORDS, 40)
        self.assertEqual(protocol.RECORD_WORDS, 400)
        self.assertEqual(protocol.ISSUE_STRIDE, 22)
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

    def test_hazard_requires_disjoint_overlap_qualification(self) -> None:
        hazard = next(
            case
            for case in execution_probe.V2_HAZARD_CASES
            if case.name == "raw-rdma-ct-exact-r2-window"
        )
        with self.assertRaisesRegex(RuntimeError, "disjoint serial/window"):
            execution_probe.validate_hazard_selection((hazard,))
        controls = tuple(
            case
            for case in execution_probe.V2_HAZARD_DISJOINT_CONTROLS
            if execution_probe.hazard_pair_key(case.plan)
            == execution_probe.hazard_pair_key(hazard.plan)
        )
        serial, window = (
            next(
                case
                for case in controls
                if case.plan.schedule == schedule
            )
            for schedule in (
                protocol.Schedule.SERIAL,
                protocol.Schedule.WINDOW,
            )
        )
        observations = [
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
            self.assertEqual(issue_counts, {min(4, depth - 2)})
            self.assertTrue(all(count < depth for count in issue_counts))

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


if __name__ == "__main__":
    unittest.main()
