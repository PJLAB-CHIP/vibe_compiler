#!/usr/bin/env python3
"""Host contract tests for the real NE tail throughput board adapter."""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import tempfile

import wafer_board_ne_calibration_probe_test as ne_driver
import wafer_board_engine_pipeline_characterization_test as engine_driver
import wafer_board_ne_tail_throughput_test as tail_driver
import wafer_engine_pipeline_characterization_catalog as engine_catalog
import wafer_ne_calibration_catalog as ne_catalog
import wafer_ne_tail_throughput_catalog as catalog


def synthetic_output(
    case: catalog.NETailThroughputCase,
    sample: int,
    *,
    pmu_enable: int = 1,
    ne_instructions: int = 1,
    ne_blocking: int = 0,
    ne_execution: int = 123,
) -> tuple[bytes, ne_catalog.CasePayload]:
    source = case.source_case
    built = ne_catalog.build_case_payload(source, sample)
    assert built.expected_physical is not None
    raw = bytearray(
        [ne_driver.OUTPUT_INITIAL_CANARY] * ne_catalog.RESOURCE_BYTES
    )
    record = [0] * ne_catalog.RECORD_WORDS
    values = {
        "MAGIC": ne_catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            ne_catalog.SCHEMA << 32
        ) | ne_catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": source.case_id,
        "DTYPE": source.dtype,
        "LHS_ORIENTATION": source.lhs_orientation,
        "RHS_ORIENTATION": source.rhs_orientation,
        "BATCH": source.batch,
        "M": source.m,
        "K": source.k,
        "N": source.n,
        "LHS_SPAN": source.lhs_span,
        "RHS_SPAN": source.rhs_span,
        "OUTPUT_SPAN": source.output_span,
        "SAMPLE": sample,
        "EXECUTE_RESULT": 1,
        "REQUEST_GUARD": ne_catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": ne_catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": ne_catalog.SLOT_BYTES,
        "BODY_OFFSET": ne_catalog.BODY_OFFSET,
        "KIND": source.kind,
        "PROFILE": source.profile,
        "OPTION": source.option,
        "AUX_SPAN": source.aux_span,
        "DISPOSITION": source.disposition,
        "LHS_BATCH": source.lhs_batch,
        "RHS_BATCH": source.rhs_batch,
        "PMU_ENABLE": pmu_enable,
        "NE_INST_DELTA": ne_instructions,
        "NE_BLOCKING_DELTA": ne_blocking,
        "NE_EXEC_DELTA": ne_execution,
        "PMU_BASE": ne_catalog.PMU_BASE,
        "RECORD_GUARD": ne_catalog.RECORD_GUARD,
    }
    for name, value in values.items():
        record[ne_catalog.REC[name]] = value
    struct.pack_into(
        f"<{ne_catalog.RECORD_WORDS}Q", raw, 0, *record
    )
    output_slot = built.payload[
        2 * ne_catalog.SLOT_BYTES : 3 * ne_catalog.SLOT_BYTES
    ]
    slot_begin = ne_catalog.OUTPUT_DDR_OFFSET
    raw[
        slot_begin : slot_begin + ne_catalog.SLOT_BYTES
    ] = output_slot
    physical_begin = slot_begin + ne_catalog.BODY_OFFSET
    raw[
        physical_begin : physical_begin + source.output_span
    ] = built.expected_physical
    return bytes(raw), built


def validate_shape_and_pipeline_contract() -> None:
    catalog.validate_catalog()
    assert catalog.CELL_KEY == "single/ne/tail"
    assert catalog.GROUP_KEY == "single-ne-tail"
    assert len(catalog.CASES) == 1
    case = catalog.CASES[0]
    source = case.source_case
    assert case.key == "single/ne/tail/f16-m65-k129-n129"
    assert case.repeats == 3
    assert case.work_macs == 65 * 129 * 129
    assert case.activation_prerequisite_cells == (
        "single/ne/small",
        "single/ne/steady-16k",
    )
    assert (
        source.dtype_name,
        source.geometry_name,
        source.orientation_name,
        source.m,
        source.k,
        source.n,
    ) == ("F16", "tail", "NN", 65, 129, 129)
    assert source.exact
    assert not case.as_dict()["oracle"]["host_elapsed_is_evidence"]
    assert all(catalog.PIPELINE_CONTRACT.completion_gate)


def validate_protocol_and_real_execution_chain() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    header = (
        root
        / "test/Board/Inputs/wafer_ne_calibration_probe_protocol.h"
    ).read_text()
    probe = (
        root / "test/Board/Inputs/wafer_ne_calibration_probe.c"
    ).read_text()
    board_driver = pathlib.Path(tail_driver.__file__).read_text()
    cmake = (root / "test/CMakeLists.txt").read_text()
    assert "#define WAFER_NEC_SCHEMA 5U" in header
    assert "#define WAFER_NEC_RECORD_WORDS 36U" in header
    assert "WAFER_NEC_REC_NE_INST_DELTA = 29" in header
    assert "WAFER_NEC_REC_NE_EXEC_DELTA = 31" in header
    assert "WAFER_NEC_REC_RECORD_GUARD = 35" in header
    assert "GR_PMU_NE_INST_NUMS" in probe
    assert "GR_PMU_NE_BLOCKING_TIME" in probe
    assert "GR_PMU_NE_EXE_TIME" in probe
    assert "TsmExecute(&instruction)" in probe
    assert "wafer_tx81_wdma_v3(" in probe
    assert "wafer_tx81_ncc_join(1U);" in probe
    assert "compile_seed_package" in board_driver
    assert "build_probe" in board_driver
    assert "verify_no_card" in board_driver
    assert "board_command" in board_driver
    assert "rank_one_terminal_completion" in board_driver
    assert "require_exact_board_completion" in board_driver
    assert "runtime_terminal_completion" in board_driver
    assert "prepare_fresh_ne_prerequisites" in board_driver
    assert "fresh_ne_prerequisite_probes" in board_driver
    assert "reset_work_dir" in board_driver
    assert "--single-engine-observations" not in cmake
    assert "DEPENDS wafer-board-engine-pipeline-single-engines" not in cmake


def validate_strong_host_oracle() -> None:
    case = catalog.CASES[0]
    raw, built = synthetic_output(case, sample=2)
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "output.raw"
        output.write_bytes(raw)
        validated = ne_driver.validate_output(
            output, case.source_case, built, sample=2
        )
        observation = tail_driver.normalize_observation(
            case, 2, validated, terminal_completion=23
        )
        assert observation["instruction_delta"] == {"worker0.ne": 1}
        assert observation["execution_delta"] == {"ne": 123}
        assert observation["blocking_delta"] == {"ne": 0}
        assert (
            observation["runtime_lifecycle"]
            == engine_catalog.RUNTIME_LIFECYCLE
        )
        assert observation["runtime_terminal_completion"] == 23
        for mutation in (
            {"pmu_enable": 0},
            {"ne_instructions": 2},
            {"ne_execution": 0},
        ):
            mutated, mutated_built = synthetic_output(
                case, sample=2, **mutation
            )
            output.write_bytes(mutated)
            try:
                ne_driver.validate_output(
                    output,
                    case.source_case,
                    mutated_built,
                    sample=2,
                )
            except RuntimeError as error:
                assert "record/execute/guard oracle failed" in str(error)
            else:
                raise AssertionError(
                    f"NE PMU mutation was accepted: {mutation}"
                )


def validate_repeat_and_selection_gates() -> None:
    assert tail_driver.selected_cases(catalog.GROUP_KEY) == catalog.CASES
    try:
        tail_driver.selected_cases("other")
    except RuntimeError as error:
        assert "unknown NE tail activation group" in str(error)
    else:
        raise AssertionError("unknown NE tail group was accepted")
    tail_driver._require_repeat(3)
    try:
        tail_driver._require_repeat(2)
    except RuntimeError as error:
        assert "requires >= 3 repeats" in str(error)
    else:
        raise AssertionError("under-sampled NE tail run was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        work_dir = root / "work"
        work_dir.mkdir()
        stale = work_dir / "stale-board-output.raw"
        stale.write_bytes(b"historical")
        args = argparse.Namespace(repo_root=root, work_dir=work_dir)
        assert tail_driver.reset_work_dir(args) == work_dir
        assert work_dir.is_dir()
        assert not stale.exists()


def rank_one_stdout(terminal_completion: int) -> str:
    return "\n".join(
        [
            *(
                f"board_stage: {stage}"
                for stage in engine_catalog.RUNTIME_LIFECYCLE
            ),
            (
                f"terminal_completion: {terminal_completion} "
                "kind=entry_return"
            ),
            "board_execution: true",
        ]
    )


def validate_exact_terminal_completion_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        package = pathlib.Path(directory)
        manifest = {
            "schema_version": 7,
            "rank_count": 1,
            "entries": [
                {"id": 0, "rank": 0, "terminal_completion": 31}
            ],
            "completions": [
                {"id": 31, "rank": 0, "kind": "entry_return"}
            ],
        }
        (package / "manifest.json").write_text(json.dumps(manifest))
        assert tail_driver.rank_one_terminal_completion(package) == 31
        tail_driver.require_exact_board_completion(
            rank_one_stdout(31), 31, catalog.CASES[0].key
        )
        try:
            tail_driver.require_exact_board_completion(
                rank_one_stdout(32), 31, catalog.CASES[0].key
            )
        except RuntimeError as error:
            assert "differs from the package manifest" in str(error)
        else:
            raise AssertionError("wrong NE tail terminal completion was accepted")
        manifest["completions"][0]["rank"] = 1
        (package / "manifest.json").write_text(json.dumps(manifest))
        try:
            tail_driver.rank_one_terminal_completion(package)
        except RuntimeError as error:
            assert "does not bind its exact terminal completion" in str(error)
        else:
            raise AssertionError("wrong-rank NE tail completion was accepted")


def raw_ne_observation(
    probe: engine_catalog.ProbeCase, sample: int, cycles: int
) -> dict[str, object]:
    return {
        "case": probe.as_dict(),
        "sample": sample,
        "instruction_delta": {"worker0.ne": 1},
        "execution_delta": {"ne": cycles, "full": cycles},
        "runtime_lifecycle": engine_catalog.RUNTIME_LIFECYCLE,
        "runtime_terminal_completion": 11,
    }


def validate_small_steady_tail_merge_gate() -> None:
    prerequisite_rows: list[dict[str, object]] = []
    cycles_by_cell = {
        "single/ne/small": 100,
        "single/ne/steady-16k": 1000,
    }
    prerequisite_probes = tail_driver.fresh_ne_prerequisite_probes()
    assert tuple(probe.cell_key for probe in prerequisite_probes) == (
        "single/ne/small",
        "single/ne/steady-16k",
    )
    for probe in prerequisite_probes:
        for sample in range(3):
            prerequisite_rows.append(
                raw_ne_observation(
                    probe, sample, cycles_by_cell[probe.cell_key] + sample
                )
            )
    with tempfile.TemporaryDirectory() as directory:
        arguments = argparse.Namespace(
            device_id=0,
            expected_runtime_version=17,
            expected_device_name="tx81",
            expected_pci_bus_id="0000:01:00.0",
            expected_tile_count=16,
            expected_runtime_library_sha256="A" * 64,
        )
        qualification = tail_driver.board_qualification(arguments)
        assert qualification == engine_driver.board_qualification(arguments)
        session_id = "a" * 32
        assert (
            tail_driver.require_calibration_session(
                {
                    engine_catalog.CALIBRATION_SESSION_ENVIRONMENT: (
                        session_id
                    )
                }
            )
            == session_id
        )
        assert len(prerequisite_rows) == 6
        tail_driver.require_complete_ne_prerequisites(prerequisite_rows)
        try:
            tail_driver.require_complete_ne_prerequisites(
                prerequisite_rows[:-1]
            )
        except RuntimeError as error:
            assert "prerequisite observations are incomplete" in str(error)
        else:
            raise AssertionError("incomplete NE prerequisites were accepted")
        case = catalog.CASES[0]
        tail_rows: list[dict[str, object]] = []
        for sample in range(3):
            raw, built = synthetic_output(
                case, sample, ne_execution=1100 + sample
            )
            output = pathlib.Path(directory) / f"tail-{sample}.raw"
            output.write_bytes(raw)
            validated = ne_driver.validate_output(
                output, case.source_case, built, sample
            )
            tail_rows.append(
                tail_driver.normalize_observation(
                    case,
                    sample,
                    validated,
                    terminal_completion=31,
                )
            )
        activated = engine_catalog.evaluate_single_engine_activation(
            engine_catalog.Engine.NE, (*prerequisite_rows, *tail_rows)
        )
        assert activated.activated, activated.reasons
        incomplete = engine_catalog.evaluate_single_engine_activation(
            engine_catalog.Engine.NE,
            (*prerequisite_rows, *tail_rows[:-1]),
        )
        assert not incomplete.activated
        assert any(
            "distinct external samples" in reason
            for reason in incomplete.reasons
        )
def main() -> int:
    validate_shape_and_pipeline_contract()
    validate_protocol_and_real_execution_chain()
    validate_strong_host_oracle()
    validate_repeat_and_selection_gates()
    validate_exact_terminal_completion_gate()
    validate_small_steady_tail_merge_gate()
    print("wafer_ne_tail_throughput_catalog_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
