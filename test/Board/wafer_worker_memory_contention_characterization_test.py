#!/usr/bin/env python3
"""Validate the full-card DDR adapter and genuine neighboring boundaries."""

from __future__ import annotations

import pathlib
import struct
import subprocess
import sys
import tempfile

import wafer_board_ddr_tile_offset_probe_test as ddr_tile
import wafer_board_ncc_execution_probe_test as ncc
import wafer_board_spm_cross_tile_conflict_probe_test as spm_tile
import wafer_cache_coherence_calibration_catalog as cache_catalog
import wafer_memory_descriptor_calibration_catalog as memory_catalog
import wafer_worker_memory_contention_characterization_catalog as catalog
import wafer_worker_memory_contention_characterization_driver as driver


def _validate_existing_bindings() -> None:
    actual = {
        "worker-routing-and-low-depth-disjoint": len(ncc.V2_WORKER_CASES),
        "cross-worker-ct-rdma-sustained-control": len(
            memory_catalog.CROSS_WORKER_PARALLEL_PAIR_CASES
        ),
        "worker-wait-scope-exclusion": len(
            ncc.V2_WORKER_WAIT_SCOPE_CASES
        ),
        "worker-subset-join-exclusion": len(
            ncc.V2_WORKER_SUBSET_SCOPE_CASES
        ),
        "ddr-single-active-rank-tile-offset": len(ddr_tile.matrix_cases()),
        "spm-sustained-ct-rdma-far-disjoint-control": len(
            memory_catalog.SUSTAINED_PARALLEL_PAIR_CASES
        ),
        "spm-conflict-equivalence-rank-one": len(
            memory_catalog.CONFLICT_EQUIVALENCE_PAIRS
        ),
        "spm-conflict-equivalence-physical-tile": len(
            spm_tile.conflict_pairs()
        ),
        "ddr-conflict-equivalence-rank-one": len(
            cache_catalog.PENDING_DDR_CONFLICT_CASES
        ),
        "ddr-conflict-equivalence-physical-tile": len(
            ddr_tile.conflict_equivalence_cases()
        ),
    }
    assert actual == {
        asset.key: asset.object_count for asset in catalog.DELEGATED_ASSETS
    }
    modules = {
        "test/Board/wafer_board_ncc_execution_probe_test.py": ncc,
        "test/Board/wafer_memory_descriptor_calibration_catalog.py": (
            memory_catalog
        ),
        "test/Board/wafer_board_spm_cross_tile_conflict_probe_test.py": (
            spm_tile
        ),
        "test/Board/wafer_cache_coherence_calibration_catalog.py": (
            cache_catalog
        ),
        "test/Board/wafer_board_ddr_tile_offset_probe_test.py": ddr_tile,
    }
    for asset in catalog.DELEGATED_ASSETS:
        assert hasattr(modules[asset.source_file], asset.selector)
        routed = driver.prepare_board_selection(asset.key)
        assert routed["route"]["source_file"] == asset.source_file
        assert routed["route"]["selector"] == asset.selector
        assert routed["route"]["object_count"] == asset.object_count


def _validate_catalog_and_requests() -> None:
    catalog.validate_catalog()
    driver.validate_static_contract()
    assert driver.SCHEMA == 2
    assert driver.STATUS_PMU_UNSTABLE == 3
    assert len(catalog.EXECUTABLE_CASES) == 45
    assert tuple(catalog.EXECUTABLE_CASES) == catalog.DDR_ACTIVE_RANK_CASES
    assert tuple(catalog.CASES) == catalog.DDR_ACTIVE_RANK_CASES
    assert all(
        case.disposition == catalog.Disposition.BOARD_EXECUTABLE
        and case.as_dict()["board_request_serializable"] is True
        for case in catalog.DDR_ACTIVE_RANK_CASES
    )
    assert {
        len(case.active_ranks) for case in catalog.DDR_ACTIVE_RANK_CASES
    } == {1, 2, 4, 8, 16}
    assert {
        case.ddr_direction for case in catalog.DDR_ACTIVE_RANK_CASES
    } == set(catalog.DDRDirection)
    assert {
        (
            case.streams[0].payload_bytes,
            case.streams[0].stride_bytes,
        )
        for case in catalog.DDR_ACTIVE_RANK_CASES
    } == {(4096, 0), (65536, 0), (65536, 8192)}
    assert len(catalog.DDR_ACTIVE_RANK_GROUPS) == 9
    assert all(
        [len(case.active_ranks) for case in group] == [1, 2, 4, 8, 16]
        for group in catalog.DDR_ACTIVE_RANK_GROUPS.values()
    )

    rank0_requests = {
        driver.request_words(case, 0, 0)
        for case in catalog.DDR_ACTIVE_RANK_CASES
    }
    assert len(rank0_requests) == len(catalog.DDR_ACTIVE_RANK_CASES)
    for case in catalog.DDR_ACTIVE_RANK_CASES:
        requests = [
            driver.request_words(case, rank, 0)
            for rank in range(catalog.RANK_COUNT)
        ]
        assert len(set(requests)) == catalog.RANK_COUNT
        assert {
            words[driver.REQ["RANK"]] for words in requests
        } == set(range(catalog.RANK_COUNT))
        assert all(
            words[driver.REQ["ACTIVE_MASK"]] == driver.active_mask(case)
            and words[driver.REQ["PAYLOAD_SEED"]] == case.payload_seed
            and words[driver.REQ["RESOURCE_BYTES"]]
            == driver.RESOURCE_BYTES
            for words in requests
        )
        routed = driver.prepare_board_selection(case.key)
        assert routed["rank_count"] == catalog.RANK_COUNT
        assert routed["active_mask"] == driver.active_mask(case)
        assert routed["sample_active_masks"] == [
            driver.active_mask(case, sample) for sample in range(4)
        ]
        assert routed["measurement"].endswith("no host time")

    phase_counts = {1: 4, 2: 4, 4: 4, 8: 2}
    for active_count, expected_phase_count in phase_counts.items():
        case = next(
            case
            for case in catalog.DDR_ACTIVE_RANK_CASES
            if len(case.active_ranks) == active_count
        )
        phased_ranks = [
            driver.active_ranks_for_sample(case, sample)
            for sample in range(4)
        ]
        assert all(len(ranks) == active_count for ranks in phased_ranks)
        assert len(set(phased_ranks)) == expected_phase_count
        assert all(
            driver.request_words(case, 0, sample)[
                driver.REQ["ACTIVE_MASK"]
            ]
            == driver.active_mask(case, sample)
            for sample in range(4)
        )

    group = next(iter(catalog.DDR_ACTIVE_RANK_GROUPS.values()))
    plan = driver.group_execution_plan(group, 4)
    assert len(plan) == 20
    base_counts = [len(case.active_ranks) for case in group]
    rotated_counts = (
        base_counts[len(base_counts) // 2 :]
        + base_counts[: len(base_counts) // 2]
    )
    expected_orders = [
        base_counts,
        list(reversed(base_counts)),
        rotated_counts,
        list(reversed(rotated_counts)),
    ]
    assert [sample for sample, _ in plan] == [
        sample for sample in range(4) for _ in group
    ]
    assert [
        [len(case.active_ranks) for plan_sample, case in plan
         if plan_sample == sample]
        for sample in range(4)
    ] == expected_orders
    assert all(
        sum(order.index(active_count) for order in expected_orders) == 8
        for active_count in base_counts
    )
    by_count = {
        len(case.active_ranks): case for case in group
    }
    for sample in range(4):
        baseline = set(
            driver.active_ranks_for_sample(by_count[1], sample)
        )
        assert all(
            baseline.issubset(
                driver.active_ranks_for_sample(
                    by_count[active_count], sample
                )
            )
            for active_count in (2, 4, 8, 16)
        )

    stride = next(
        case
        for case in catalog.DDR_ACTIVE_RANK_CASES
        if case.streams[0].stride_bytes
        and case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
    )
    input0, input1 = driver.rank_inputs(stride, 0, 0)
    assert len(input0) == len(input1) == driver.RESOURCE_BYTES
    payload, inner, stride0, iteration = driver._shape(stride)
    assert (payload, inner, stride0, iteration) == (
        65536,
        4096,
        8192,
        16,
    )
    source = input0[
        driver.SOURCE_OFFSET :
        driver.SOURCE_OFFSET + inner + stride0 * (iteration - 1)
        + 2 * driver.GUARD_BYTES
    ]
    assert source[: driver.GUARD_BYTES] == bytes(
        [driver.CANARY]
    ) * driver.GUARD_BYTES
    assert source[
        driver.GUARD_BYTES + inner :
        driver.GUARD_BYTES + stride0
    ] == bytes([driver.CANARY]) * (stride0 - inner)
    compact = driver._compact_payload_pattern(stride, 0, 0, "rdma")
    spm = driver._spm_span_pattern(stride, 0, 0, "rdma")
    assert source == driver._ddr_span_pattern(stride, 0, 0, "rdma")
    assert source[driver.GUARD_BYTES + inner] == driver.CANARY
    assert spm[driver.GUARD_BYTES + inner] == compact[inner]
    assert spm[
        driver.GUARD_BYTES : driver.GUARD_BYTES + payload
    ] == compact
    assert set(
        spm[
            driver.GUARD_BYTES + payload :
            driver.GUARD_BYTES + inner + stride0 * (iteration - 1)
        ]
    ) == {driver.CANARY}
    wdma_spm = driver._spm_span_pattern(stride, 0, 0, "wdma")
    assert input1[
        driver.SEED_WDMA_OFFSET :
        driver.SEED_WDMA_OFFSET + len(wdma_spm)
    ] == wdma_spm


def _record(
    case: catalog.CharacterizationCase,
    rank: int,
    sample: int,
) -> bytes:
    words = [0] * driver.RECORD_WORDS
    payload, inner, stride0, iteration = driver._shape(case)
    envelope = inner + stride0 * (iteration - 1)
    assert case.ddr_direction is not None
    active = int(rank in driver.active_ranks_for_sample(case, sample))
    rdma = int(
        active and case.ddr_direction != catalog.DDRDirection.WDMA
    )
    wdma = int(
        active and case.ddr_direction != catalog.DDRDirection.RDMA
    )
    values = {
        "MAGIC": driver.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            driver.SCHEMA << 32
        ) | driver.RECORD_WORDS,
        "STATUS": 0,
        "RANK": rank,
        "SAMPLE": sample,
        "ACTIVE_MASK": driver.active_mask(case, sample),
        "ACTIVE": active,
        "DIRECTION": driver.DIRECTION[case.ddr_direction],
        "PAYLOAD_BYTES": payload,
        "INNER_BYTES": inner,
        "STRIDE0": stride0,
        "ITERATION0": iteration,
        "ENVELOPE_BYTES": envelope,
        "RDMA_INST_DELTA": rdma,
        "WDMA_INST_DELTA": wdma,
        "RDMA_BLOCKING_DELTA": active * rdma,
        "WDMA_BLOCKING_DELTA": active * wdma,
        "RDMA_EXEC_DELTA": active * rdma * 101,
        "WDMA_EXEC_DELTA": active * wdma * 103,
        "FU_EXEC_DELTA": active * 107,
        "WINDOW_DELTA": active * 109,
        "COMPLETION_CYCLES": 1000 + rank if active else 10,
        "INPUT0_BASE": 0x10000000 + rank * 0x1000000,
        "INPUT1_BASE": 0x10200000 + rank * 0x1000000,
        "OUTPUT0_BASE": 0x10400000 + rank * 0x1000000,
        "OUTPUT1_BASE": 0x10600000 + rank * 0x1000000,
        "SPM_RDMA": driver.SPM_RDMA,
        "SPM_WDMA": driver.SPM_WDMA,
        "ISSUE_ORDER": (
            sample & 1
            if case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
            else 0
        ),
        "REQUEST_GUARD": driver.REQUEST_GUARD,
        "PAYLOAD_SEED": case.payload_seed,
        "TARGET_ONLY_WINDOW": 1,
        "PMU_BEFORE_STABLE": 1,
        "PMU_AFTER_STABLE": 1,
        "RECORD_GUARD": driver.RECORD_GUARD,
    }
    assert set(values) == set(driver.REC)
    for name, value in values.items():
        words[driver.REC[name]] = value
    return struct.pack(f"<{driver.RECORD_WORDS}Q", *words)


def _synthetic_outputs(
    root: pathlib.Path,
    case: catalog.CharacterizationCase,
    rank: int,
    sample: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    record = _record(case, rank, sample)
    output0 = bytearray([driver.CANARY]) * driver.RESOURCE_BYTES
    output0[: len(record)] = record
    if rank in driver.active_ranks_for_sample(case, sample):
        assert case.ddr_direction is not None
        if case.ddr_direction != catalog.DDRDirection.WDMA:
            span = driver._spm_span_pattern(
                case, rank, sample, "rdma"
            )
            output0[
                driver.RDMA_ARCHIVE_OFFSET :
                driver.RDMA_ARCHIVE_OFFSET + len(span)
            ] = span
        if case.ddr_direction != catalog.DDRDirection.RDMA:
            span = driver._ddr_span_pattern(
                case, rank, sample, "wdma"
            )
            output0[
                driver.WDMA_TARGET_OFFSET :
                driver.WDMA_TARGET_OFFSET + len(span)
            ] = span
    output1 = bytearray([driver.CANARY]) * driver.RESOURCE_BYTES
    output1[: driver.INACTIVE_CANARY_BYTES] = driver._inactive_canary(
        rank, sample
    )
    output0_path = root / f"{case.key}.{rank}.{sample}.output0"
    output1_path = root / f"{case.key}.{rank}.{sample}.output1"
    output0_path.write_bytes(bytes(output0))
    output1_path.write_bytes(bytes(output1))
    return output0_path, output1_path


def _expect_output_failure(
    case: catalog.CharacterizationCase,
    rank: int,
    output0: pathlib.Path,
    output1: pathlib.Path,
    needle: str,
) -> None:
    try:
        driver.validate_rank_outputs(
            case, rank, 0, output0, output1, (rank // 4, rank % 4)
        )
    except RuntimeError as error:
        assert needle in str(error), (needle, str(error))
    else:
        raise AssertionError(f"tampered output accepted; wanted {needle!r}")


def _validate_output_oracle() -> None:
    cases = (
        next(
            case
            for case in catalog.DDR_ACTIVE_RANK_CASES
            if case.ddr_direction == catalog.DDRDirection.RDMA
            and case.streams[0].payload_bytes == 4096
            and len(case.active_ranks) == 1
        ),
        next(
            case
            for case in catalog.DDR_ACTIVE_RANK_CASES
            if case.ddr_direction == catalog.DDRDirection.WDMA
            and case.streams[0].payload_bytes == 65536
            and not case.streams[0].stride_bytes
            and len(case.active_ranks) == 8
        ),
        next(
            case
            for case in catalog.DDR_ACTIVE_RANK_CASES
            if case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
            and case.streams[0].stride_bytes
            and len(case.active_ranks) == 16
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for case in cases:
            selected_ranks = [case.active_ranks[0]]
            if case.inactive_ranks:
                selected_ranks.append(case.inactive_ranks[0])
            for selected_rank in selected_ranks:
                output0, output1 = _synthetic_outputs(
                    root, case, selected_rank, 0
                )
                row = driver.validate_rank_outputs(
                    case,
                    selected_rank,
                    0,
                    output0,
                    output1,
                    (selected_rank // 4, selected_rank % 4),
                )
                assert row["exact_full_output"] is True
                assert row["inactive_canary"] is True
                assert row["host_elapsed_used"] is False

        phased_case = cases[1]
        shifted_rank = next(
            rank
            for rank in driver.active_ranks_for_sample(phased_case, 1)
            if rank not in phased_case.active_ranks
        )
        output0, output1 = _synthetic_outputs(
            root, phased_case, shifted_rank, 1
        )
        shifted_row = driver.validate_rank_outputs(
            phased_case,
            shifted_rank,
            1,
            output0,
            output1,
            (shifted_rank // 4, shifted_rank % 4),
        )
        assert shifted_row["active"] is True
        assert shifted_row["active_mask"] == driver.active_mask(
            phased_case, 1
        )

        case = cases[-1]
        rank = case.active_ranks[0]
        output0, output1 = _synthetic_outputs(root, case, rank, 0)
        raw = bytearray(output0.read_bytes())
        raw[driver.RDMA_ARCHIVE_OFFSET] ^= 1
        output0.write_bytes(raw)
        _expect_output_failure(case, rank, output0, output1, "output0 mismatch")

        output0, output1 = _synthetic_outputs(root, case, rank, 0)
        raw = bytearray(output1.read_bytes())
        raw[0] ^= 1
        output1.write_bytes(raw)
        _expect_output_failure(
            case, rank, output0, output1, "inactive canary mismatch"
        )

        output0, output1 = _synthetic_outputs(root, case, rank, 0)
        raw = bytearray(output0.read_bytes())
        words = list(
            struct.unpack(
                f"<{driver.RECORD_WORDS}Q",
                raw[: driver.RECORD_WORDS * 8],
            )
        )
        words[driver.REC["RDMA_INST_DELTA"]] = 0
        raw[: driver.RECORD_WORDS * 8] = struct.pack(
            f"<{driver.RECORD_WORDS}Q", *words
        )
        output0.write_bytes(raw)
        _expect_output_failure(case, rank, output0, output1, "record oracle")

        output0, output1 = _synthetic_outputs(root, case, rank, 0)
        raw = bytearray(output0.read_bytes())
        # wafer-run initializes every write-only byte to 0xa5.  A zero-filled
        # unwritten tail was the previous false oracle and must be rejected.
        raw[-1] = 0
        output0.write_bytes(raw)
        _expect_output_failure(case, rank, output0, output1, "output0 mismatch")

        for stable_field in ("PMU_BEFORE_STABLE", "PMU_AFTER_STABLE"):
            output0, output1 = _synthetic_outputs(root, case, rank, 0)
            raw = bytearray(output0.read_bytes())
            words = list(
                struct.unpack(
                    f"<{driver.RECORD_WORDS}Q",
                    raw[: driver.RECORD_WORDS * 8],
                )
            )
            words[driver.REC[stable_field]] = 0
            raw[: driver.RECORD_WORDS * 8] = struct.pack(
                f"<{driver.RECORD_WORDS}Q", *words
            )
            output0.write_bytes(raw)
            _expect_output_failure(
                case, rank, output0, output1, "record oracle"
            )


def _validate_raw_resource_cleanup() -> None:
    with tempfile.TemporaryDirectory() as directory:
        work_dir = pathlib.Path(directory)
        raw_dir = work_dir / "raw" / "case" / "sample-0"
        raw_dir.mkdir(parents=True)
        resources = []
        for rank in range(driver.RANK_COUNT):
            for role in ("input", "output"):
                for ordinal in range(2):
                    path = raw_dir / (
                        f"rank-{rank:02d}.{role}-{ordinal}.raw"
                    )
                    path.write_bytes(b"validated")
                    resources.append(path)
        unexpected = raw_dir / "unexpected.raw"
        unexpected.write_bytes(b"retain on mismatch")
        try:
            driver._discard_validated_resources(
                work_dir, raw_dir, resources
            )
        except RuntimeError as error:
            assert "unexpected file set" in str(error)
        else:
            raise AssertionError("unexpected raw evidence was deleted")
        assert unexpected.is_file()
        unexpected.unlink()
        driver._discard_validated_resources(work_dir, raw_dir, resources)
        assert not raw_dir.exists()
        archive = work_dir / "atomic.json"
        archive.write_text("previous-complete-archive\n")
        original_replace = driver.os.replace

        def fail_replace(_source: object, _destination: object) -> None:
            raise OSError("injected atomic replace failure")

        driver.os.replace = fail_replace
        try:
            driver._write_json_atomic(archive, {"complete": False})
        except OSError as error:
            assert "injected atomic replace failure" in str(error)
        else:
            raise AssertionError("injected archive replace failure was ignored")
        finally:
            driver.os.replace = original_replace
        assert archive.read_text() == "previous-complete-archive\n"
        staged = archive.with_name(".atomic.json.tmp")
        assert staged.is_file()
        staged.unlink()
        driver._write_json_atomic(archive, {"complete": True})
        assert archive.read_text() == '{\n  "complete": true\n}\n'
        assert not staged.exists()


def _validate_board_lifecycle_oracle() -> None:
    manifest = driver.direct_dte_evidence.DirectDTEManifestEvidence(
        status_resource_by_rank=tuple(
            (rank, 100 + rank) for rank in range(driver.RANK_COUNT)
        ),
        terminal_completion_by_rank=tuple(
            (rank, rank) for rank in range(driver.RANK_COUNT)
        ),
    )
    lines = [
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        *[
            f"terminal_completion: {rank} "
            f"kind=entry_return rank={rank}"
            for rank in range(driver.RANK_COUNT)
        ],
        f"invocation_ranks: {driver.RANK_COUNT}",
        "launch_pattern: cluster-prepare-main-x16",
        "logical_tile_execution_basis: "
        "cluster-pid-and-exact-rank-slices",
        "logical_tile_domain: 0..15",
        "board_execution: true",
    ]
    stdout = "\n".join(lines) + "\n"
    evidence = driver.validate_board_lifecycle(stdout, manifest, 180000)
    assert evidence["terminal_completion_by_rank"] == [
        {"rank": rank, "completion": rank}
        for rank in range(driver.RANK_COUNT)
    ]

    missing = stdout.replace(
        "terminal_completion: 15 kind=entry_return rank=15\n", ""
    )
    wrong = stdout.replace(
        "terminal_completion: 7 kind=entry_return rank=7",
        "terminal_completion: 9 kind=entry_return rank=7",
    )
    duplicate = stdout.replace(
        "terminal_completion: 15 kind=entry_return rank=15",
        "terminal_completion: 15 kind=entry_return rank=14",
    )
    for invalid in (missing, wrong, duplicate):
        try:
            driver.validate_board_lifecycle(invalid, manifest, 180000)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "invalid terminal-completion stdout was accepted"
            )


def _validate_blocked_neighbors() -> None:
    for boundary in catalog.TYPED_BOUNDARIES:
        try:
            driver.prepare_board_selection(boundary.key)
        except driver.PreparationBlocked:
            pass
        else:
            raise AssertionError(
                f"{boundary.key}: typed boundary became executable"
            )


def _validate_cli(repo: pathlib.Path) -> None:
    inventory = driver.inventory()
    assert inventory["executable_new_cases"] == 45
    assert inventory["executable_ddr_active_rank_matrix"] == 45
    assert "blocked_worker_matrix" not in inventory
    assert "blocked_spm_matrix" not in inventory
    script = (
        repo
        / "test/Board/wafer_worker_memory_contention_characterization_driver.py"
    )
    executable = catalog.DDR_ACTIVE_RANK_CASES[-1]
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "audit",
            "--case",
            executable.key,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"rank_count": 16' in result.stdout
    assert "wafer_ddr_active_rank_contention_probe.c" in result.stdout
    emitted = subprocess.run(
        [
            sys.executable,
            str(script),
            "--emit-board-group-keys",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert emitted == list(catalog.DDR_ACTIVE_RANK_GROUP_KEYS)
    grouped = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "audit",
            "--group",
            catalog.DDR_ACTIVE_RANK_GROUP_KEYS[0],
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert grouped.returncode == 0, grouped.stderr
    assert grouped.stdout.count('"rank_count": 16') == 5
    blocked = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "audit",
            "--case",
            "ddr-bank-owner-mapping-absent",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "mapping provider" in blocked.stderr


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    _validate_existing_bindings()
    _validate_catalog_and_requests()
    _validate_output_oracle()
    _validate_raw_resource_cleanup()
    _validate_board_lifecycle_oracle()
    _validate_blocked_neighbors()
    _validate_cli(repo)
    print(
        "worker-memory contention: 45 executable full-card DDR cases; "
        "no superseded synthetic worker/SPM blockers; "
        "11 delegated real assets and 7 genuine typed boundaries"
    )


if __name__ == "__main__":
    main()
