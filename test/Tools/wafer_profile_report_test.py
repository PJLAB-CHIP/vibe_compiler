#!/usr/bin/env python3
"""No-card tests for the 16-tile profile analyzer and offline report."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

from wafer_profile_report_fixture import make_evidence


def _load_report_module(repo: pathlib.Path) -> object:
    path = repo / "tools" / "wafer_profile_report.py"
    spec = importlib.util.spec_from_file_location(
        "wafer_profile_report_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _must_reject(module: object, evidence: object, marker: str) -> None:
    try:
        module.validate_evidence(evidence)
    except module.EvidenceError as error:
        assert marker in str(error), (marker, str(error))
    else:
        raise AssertionError(f"invalid evidence was accepted; expected {marker!r}")


def _test_verdicts(module: object) -> None:
    improved = make_evidence(winner_scale=0.8)
    first = module.analyze_evidence(improved)
    second = module.analyze_evidence(copy.deepcopy(improved))
    assert first == second, "analysis and paired bootstrap must be deterministic"
    assert first["verdict"] == "improved"
    assert first["validity"]["performance"]
    assert first["performance"]["speedup_ratio"] > 1.2
    assert first["performance"]["paired_bootstrap_ci95"][0] > 1.0
    assert first["performance"]["held_out_speedup_ratio"] > 1.0
    assert first["performance"]["materiality_threshold"] >= 0.01
    assert {row["strength"] for row in first["insights"]} == {
        "measured",
        "correlated",
        "unresolved",
    }
    assert len(first["tile_metrics"]) == 16
    assert len(first["changed_sites"]) == 3
    changed_by_key = {
        row["correlation_key"]: row for row in first["changed_sites"]
    }
    assert changed_by_key["input.read"]["baseline_site_ids"] == [2]
    assert changed_by_key["input.read"]["winner_site_ids"] == [1]
    assert len(changed_by_key["input.read"]["baseline_sites"]) == 16
    assert len(changed_by_key["input.read"]["winner_sites"]) == 16
    assert changed_by_key["input.read"]["change_kind"] == "count_changed"
    assert changed_by_key["baseline.prefill"]["change_kind"] == "removed"
    assert changed_by_key["winner.workspace"]["change_kind"] == "added"
    assert "compute.gemm" not in changed_by_key, (
        "shifted candidate-local ordinals must join by correlation key"
    )
    assert "output.write" not in changed_by_key
    assert first["cross_tile_order"]["baseline"]["total_pairs"] > 0
    assert (
        first["cross_tile_order"]["baseline"]["proven_pairs"]
        + first["cross_tile_order"]["baseline"]["overlapping_pairs"]
        == first["cross_tile_order"]["baseline"]["total_pairs"]
    )

    regressed = module.analyze_evidence(make_evidence(winner_scale=1.25))
    assert regressed["verdict"] == "regressed"
    assert regressed["performance"]["paired_bootstrap_ci95"][1] < 1.0
    assert regressed["performance"]["held_out_speedup_ratio"] < 1.0

    small = module.analyze_evidence(make_evidence(winner_scale=0.995))
    assert small["verdict"] == "inconclusive"

    contradicted = module.analyze_evidence(
        make_evidence(winner_scale=0.8, held_out_winner_scale=1.2)
    )
    assert contradicted["verdict"] == "inconclusive"
    assert contradicted["performance"]["speedup_ratio"] > 1.0
    assert contradicted["performance"]["held_out_speedup_ratio"] < 1.0


def _test_semantic_invalidity(module: object) -> None:
    equivalence = make_evidence()
    equivalence["output_validation"]["resources"][7][
        "candidate_equivalent_exact"
    ] = False
    analysis = module.analyze_evidence(equivalence)
    assert analysis["verdict"] == "invalid"
    assert not analysis["validity"]["output_equivalence"]
    assert any(
        row["code"] == "output_equivalence_failed"
        for row in analysis["diagnostics"]
    )
    equivalence_report = module.render_report(equivalence, analysis)
    assert (
        "production repeatability or candidate/instrumentation equivalence "
        "failed" in equivalence_report
    )

    no_external_expected = make_evidence()
    no_external_expected["output_validation"][
        "mode"
    ] = "same-session-production-winner"
    for row in no_external_expected["output_validation"]["resources"]:
        row["external_expected_exact"] = None
    analysis = module.analyze_evidence(no_external_expected)
    assert analysis["verdict"] == "improved"
    assert analysis["validity"]["output_equivalence"]
    assert analysis["validity"]["semantic_correctness"] is None
    assert any(
        row["code"] == "semantic_correctness_unknown"
        for row in analysis["diagnostics"]
    )
    report = module.render_report(no_external_expected, analysis)
    assert "absolute correctness unknown" in report

    failed_external = make_evidence()
    failed_external["output_validation"]["resources"][3][
        "external_expected_exact"
    ] = False
    analysis = module.analyze_evidence(failed_external)
    assert analysis["verdict"] == "invalid"
    assert analysis["validity"]["semantic_correctness"] is False
    assert any(
        row["code"] == "external_expected_failed"
        for row in analysis["diagnostics"]
    )
    failed_report = module.render_report(failed_external, analysis)
    assert '"Expected mismatch"' in failed_report
    assert (
        "one or more independently supplied expected outputs did not match"
        in failed_report
    )

    identity = make_evidence()
    identity["experiments"]["baseline"]["artifact"]["launch"] = {
        "kind": "kernel",
        "form": "per-rank",
        "entry_abi": "rank-local-pointer-block-v1",
        "phases": ["main"],
    }
    analysis = module.analyze_evidence(identity)
    assert analysis["verdict"] == "invalid"
    assert not analysis["validity"]["identity"]
    assert not analysis["validity"]["cross_candidate_explanation"]
    assert analysis["changed_sites"] == []
    assert all(row["delta"] is None for row in analysis["tile_metrics"])
    assert all(
        not row["valid"] and row["delta"] is None
        for row in analysis["pmu_engine_deltas"]
    )

    production_identity = make_evidence()
    production_identity["identity"][
        "production_manifest_sha256"
    ] = "sha256:stale-winner"
    analysis = module.analyze_evidence(production_identity)
    assert analysis["verdict"] == "invalid"
    assert not analysis["validity"]["identity"]

    aliased = make_evidence()
    aliased["identity"]["baseline_same_as_winner"] = True
    aliased["experiments"]["baseline"]["artifact"]["digest"] = aliased[
        "experiments"
    ]["winner"]["artifact"]["digest"]
    analysis = module.analyze_evidence(aliased)
    assert analysis["verdict"] == "inconclusive"
    assert analysis["validity"]["performance"]
    assert not analysis["validity"]["distinct_candidates"]
    assert not analysis["validity"]["cross_candidate_explanation"]
    assert analysis["changed_sites"] == []
    assert any(
        row["code"] == "identical_candidate_artifacts"
        for row in analysis["diagnostics"]
    )

    clock = make_evidence()
    clock["experiments"]["winner"]["clock"][5]["valid"] = False
    analysis = module.analyze_evidence(clock)
    assert analysis["verdict"] == "improved", (
        "host launch-to-completion timing remains valid when only "
        "cross-tile alignment is unavailable"
    )
    assert analysis["validity"]["performance"]
    assert not analysis["validity"]["clock"]
    assert not analysis["cross_tile_order"]["winner"]["valid"]
    local_report = module.render_report(clock, analysis)
    assert "entry-local cycle zero" in local_report
    assert "positions across different rows do not establish cross-tile order" in (
        local_report
    )

    clock_samples = make_evidence()
    clock_samples["experiments"]["baseline"]["clock"][2]["round_trips"] = 31
    analysis = module.analyze_evidence(clock_samples)
    assert analysis["verdict"] == "improved"
    assert analysis["validity"]["performance"]
    assert not analysis["validity"]["clock"]

    overflow = make_evidence()
    trace_tile = overflow["experiments"]["winner"]["trace"]["tiles"][4]
    trace_tile["capacity"] = trace_tile["count"]
    trace_tile["overflow"] = True
    trace_tile["dropped_event_count"] = 1
    trace_tile["next_sequence"] = trace_tile["count"] + 1
    trace_tile["preflight_count"] = trace_tile["next_sequence"]
    trace_tile["record_flags"] |= 1 << 3
    trace_tile["trace_state"] = 3
    analysis = module.analyze_evidence(overflow)
    assert analysis["verdict"] == "improved", (
        "trace is explanatory and must not replace low-overhead timing"
    )
    assert not analysis["validity"]["trace"]["winner"]
    assert not any(
        row["candidate"] == "winner" for row in analysis["timeline_events"]
    )
    overflow_report = module.render_report(overflow, analysis)
    assert "const defaultTimelineMode = traceComparable" in overflow_report
    assert (
        'node.dataset.mode === "difference" && !traceComparable'
        in overflow_report
    )
    assert (
        "!analysis.validity.trace[node.dataset.mode]" in overflow_report
    )
    assert (
        "blank rows do not mean zero TsmExecute calls" in overflow_report
    )
    assert "blank rows do not mean zero TsmExecute calls" in overflow_report
    assert (
        "!analysis.validity.trace[node.dataset.mode]" in overflow_report
    )
    assert analysis["changed_sites"] == []
    assert all(
        value is None
        for candidate in analysis["engine_counts"].values()
        for value in candidate.values()
    )
    assert any(row["code"] == "trace_invalid" for row in analysis["diagnostics"])

    empty = make_evidence()
    empty["sites"] = []
    for candidate in ("baseline", "winner"):
        for trace_tile in empty["experiments"][candidate]["trace"]["tiles"]:
            trace_tile["capacity"] = 0
            trace_tile["count"] = 0
            trace_tile["preflight_count"] = 0
            trace_tile["next_sequence"] = 0
            trace_tile["events"] = []
    empty_analysis = module.analyze_evidence(empty)
    assert empty_analysis["verdict"] == "improved"
    assert empty_analysis["validity"]["trace"] == {
        "baseline": True,
        "winner": True,
    }
    assert empty_analysis["cross_tile_order"]["baseline"]["total_pairs"] == 0
    empty_html = module.render_report(empty, empty_analysis)
    assert empty_html.count("data-static-tile=") == 16
    assert "const fullMin = timelineEvents.length ?" in empty_html
    assert empty_analysis["validity"]["trace"]["baseline"]
    assert empty_analysis["validity"]["trace"]["winner"]

    counter = make_evidence()
    snapshot = counter["experiments"]["winner"]["pmu"]["tiles"][3][
        "aggregates"
    ]["ne"]
    snapshot["end"] = snapshot["start"] - 1
    snapshot["recovery"] = snapshot["end"]
    analysis = module.analyze_evidence(counter)
    assert analysis["verdict"] == "improved"
    assert not analysis["validity"]["pmu"]["winner"]
    ne_execution = next(
        row
        for row in analysis["pmu_engine_deltas"]
        if row["engine"] == "NE" and row["metric"] == "execution"
    )
    assert not ne_execution["valid"] and ne_execution["baseline"] is None

    missing_recovery = make_evidence()
    del missing_recovery["experiments"]["baseline"]["pmu"]["tiles"][0][
        "workers"
    ][2]["engines"][4]["blocking"]["recovery"]
    analysis = module.analyze_evidence(missing_recovery)
    assert analysis["verdict"] == "improved"
    assert not analysis["validity"]["pmu"]["baseline"]
    worker_metric = next(
        row
        for row in analysis["pmu_worker_engine_deltas"]
        if row["worker"] == 2
        and row["engine"] == "TDMA"
        and row["metric"] == "blocking"
    )
    assert not worker_metric["valid"]
    assert any(
        row["code"] == "pmu_counter_invalid"
        and row["message"] == "terminal recovery snapshot is missing"
        for row in analysis["diagnostics"]
    )
    single_pmu_group = next(
        row
        for row in module._summarize_diagnostics_for_report(
            analysis["diagnostics"]
        )
        if row["code"] == "pmu_counter_invalid"
    )
    assert single_pmu_group["grouped"]
    assert single_pmu_group["grouped_count"] == 1
    assert single_pmu_group["tiles"] == [0]
    assert single_pmu_group["scopes"] == ["worker:2:TDMA:blocking"]

    all_pmu_disabled = make_evidence()
    for candidate in ("baseline", "winner"):
        for tile in all_pmu_disabled["experiments"][candidate]["pmu"]["tiles"]:
            for snapshot in tile["aggregates"].values():
                snapshot["enabled"] = False
            for worker in tile["workers"]:
                for engine in worker["engines"]:
                    engine["instructions"]["enabled"] = False
                    engine["blocking"]["enabled"] = False
    analysis = module.analyze_evidence(all_pmu_disabled)
    raw_pmu_diagnostics = [
        row
        for row in analysis["diagnostics"]
        if row["code"] == "pmu_counter_invalid"
    ]
    assert len(raw_pmu_diagnostics) == 2 * 16 * (8 + 3 * 5 * 2)
    display_diagnostics = module._summarize_diagnostics_for_report(
        analysis["diagnostics"]
    )
    pmu_groups = [
        row
        for row in display_diagnostics
        if row["code"] == "pmu_counter_invalid"
    ]
    assert len(pmu_groups) == 2
    assert {row["grouped_count"] for row in pmu_groups} == {608}
    assert all(row["tiles"] == list(range(16)) for row in pmu_groups)
    assert all(len(row["scopes"]) == 38 for row in pmu_groups)
    disabled_report = module.render_report(all_pmu_disabled, analysis)
    assert (
        disabled_report.count('"code":"pmu_counter_invalid"') == 2
    ), "the HTML payload must carry grouped PMU diagnostics, not raw detail"
    assert '"diagnostic_raw_count":1216' in disabled_report
    assert "Full per-counter details remain in" in disabled_report

    count_mismatch = make_evidence()
    count_mismatch["experiments"]["winner"]["trace"]["tiles"][6][
        "preflight_count"
    ] += 1
    analysis = module.analyze_evidence(count_mismatch)
    assert analysis["verdict"] == "improved"
    assert not analysis["validity"]["trace"]["winner"]
    assert any(
        row["code"] == "trace_invalid"
        and "preflight" in row["message"]
        for row in analysis["diagnostics"]
    )

    structurally_invalid_event = make_evidence()
    failed_event = structurally_invalid_event["experiments"]["winner"][
        "trace"
    ]["tiles"][2][
        "events"
    ][1]
    failed_event["raw_result"] = 0
    failed_event["valid"] = False
    analysis = module.analyze_evidence(structurally_invalid_event)
    assert not analysis["validity"]["trace"]["winner"]
    assert analysis["changed_sites"] == []

    raw_return_is_observation = make_evidence()
    raw_return_is_observation["experiments"]["winner"]["trace"]["tiles"][2][
        "events"
    ][1]["raw_result"] = 0
    analysis = module.analyze_evidence(raw_return_is_observation)
    assert analysis["validity"]["trace"]["winner"], (
        "TsmExecute raw return is an uninterpreted observation, not a "
        "success/completion oracle"
    )

    coarse_completion = make_evidence()
    coarse_sample = coarse_completion["measurement"]["samples"][0]
    coarse_sample["completion_observation_resolution_ns"] = coarse_sample[
        "host_elapsed_ns"
    ]
    analysis = module.analyze_evidence(coarse_completion)
    assert analysis["verdict"] == "invalid"
    assert not analysis["validity"]["completion_resolution"]
    assert not analysis["validity"]["measurement_basis"]
    assert any(
        row["code"] == "completion_resolution_too_coarse"
        for row in analysis["diagnostics"]
    )


def _test_structural_rejection(module: object) -> None:
    missing_tile = make_evidence()
    missing_tile["topology"].pop()
    _must_reject(module, missing_tile, "all-and-only tiles")

    duplicate_tile = make_evidence()
    duplicate_tile["experiments"]["winner"]["summary"]["tiles"][15]["tile"] = 14
    _must_reject(module, duplicate_tile, "all-and-only tiles")

    duplicate_coordinate = make_evidence()
    duplicate_coordinate["topology"][15]["x"] = duplicate_coordinate[
        "topology"
    ][14]["x"]
    duplicate_coordinate["topology"][15]["y"] = duplicate_coordinate[
        "topology"
    ][14]["y"]
    _must_reject(module, duplicate_coordinate, "duplicate physical coordinate")

    host_elapsed = make_evidence()
    next(
        row
        for row in host_elapsed["measurement"]["samples"]
        if row["candidate"] == "winner"
    )["host_elapsed_ns"] = 0
    _must_reject(module, host_elapsed, "must be at least 1")

    sequence = make_evidence()
    sequence["experiments"]["winner"]["trace"]["tiles"][0]["events"][2][
        "sequence"
    ] = 7
    _must_reject(module, sequence, "sequence must be contiguous")

    site = make_evidence()
    site["experiments"]["baseline"]["trace"]["tiles"][0]["events"][0][
        "site_id"
    ] = 999
    _must_reject(module, site, "unmapped baseline tile 0 site")

    engine = make_evidence()
    engine["experiments"]["baseline"]["trace"]["tiles"][0]["events"][0][
        "engine"
    ] = "NE"
    _must_reject(module, engine, "conflicts with site")

    inconsistent_overflow = make_evidence()
    inconsistent_overflow["experiments"]["baseline"]["trace"]["tiles"][0][
        "overflow"
    ] = True
    _must_reject(module, inconsistent_overflow, "raw terminal fields")

    clock = make_evidence()
    clock["experiments"]["winner"]["clock"][0]["slope"] = 0
    _must_reject(module, clock, "must be positive")

    raw_result = make_evidence()
    raw_result["experiments"]["winner"]["trace"]["tiles"][0]["events"][0][
        "raw_result"
    ] = -1
    _must_reject(module, raw_result, "must be at least 0")

    outside_entry = make_evidence()
    outside_entry["experiments"]["winner"]["trace"]["tiles"][0]["events"][0][
        "begin_cycle"
    ] = 1
    _must_reject(module, outside_entry, "outside the entry interval")

    unknown_key = make_evidence()
    unknown_key["opaque_payload"] = {}
    _must_reject(module, unknown_key, "unknown keys")

    legacy_launch = make_evidence()
    legacy_launch["identity"]["launch_abi"] = legacy_launch["identity"].pop(
        "launch"
    )
    _must_reject(module, legacy_launch, "launch")

    invalid_kernel_contract = make_evidence()
    invalid_kernel_contract["identity"]["launch"]["form"] = "cluster"
    _must_reject(
        module,
        invalid_kernel_contract,
        "kernel form, entry ABI and ordered phases are incompatible",
    )

    third_launch_kind = make_evidence()
    third_launch_kind["identity"]["launch"]["kind"] = "direct-dte"
    _must_reject(
        module,
        third_launch_kind,
        "must be either 'kernel' or 'model'",
    )

    output_mode = make_evidence()
    output_mode["output_validation"]["resources"][0][
        "external_expected_exact"
    ] = None
    _must_reject(module, output_mode, "expected coverage")


def _test_rank_local_site_deltas_and_u64_cycles(module: object) -> None:
    moved = make_evidence()
    source = moved["experiments"]["winner"]["trace"]["tiles"][0]
    destination = moved["experiments"]["winner"]["trace"]["tiles"][1]
    event = source["events"].pop(0)
    for sequence, row in enumerate(source["events"]):
        row["sequence"] = sequence
    source["count"] = len(source["events"])
    source["preflight_count"] = source["count"]
    source["next_sequence"] = source["count"]

    event["sequence"] = len(destination["events"])
    event["begin_cycle"] = destination["events"][-1]["return_cycle"] + 5
    event["return_cycle"] = event["begin_cycle"] + 7
    destination["events"].append(event)
    destination["count"] = len(destination["events"])
    destination["preflight_count"] = destination["count"]
    destination["next_sequence"] = destination["count"]
    destination["entry_end_cycle"] = event["return_cycle"] + 100

    analysis = module.analyze_evidence(moved)
    by_key = {row["correlation_key"]: row for row in analysis["changed_sites"]}
    relocation = by_key["compute.ct"]
    assert relocation["change_kind"] == "placement_changed"
    assert relocation["delta"] == 0
    assert relocation["placement_delta_magnitude"] == 2
    assert relocation["tile_count_changes"] == [
        {"tile": 0, "baseline": 1, "winner": 0, "delta": -1},
        {"tile": 1, "baseline": 1, "winner": 2, "delta": 1},
    ]

    original = make_evidence()
    translated = copy.deepcopy(original)
    translation = (1 << 56) + 3
    for candidate in ("baseline", "winner"):
        for row in translated["experiments"][candidate]["summary"]["tiles"]:
            row["entry_begin"] += translation
            row["entry_end"] += translation
        for row in translated["experiments"][candidate]["trace"]["tiles"]:
            row["entry_begin_cycle"] += translation
            row["entry_end_cycle"] += translation
            for trace_event in row["events"]:
                trace_event["begin_cycle"] += translation
                trace_event["return_cycle"] += translation
    before = module.analyze_evidence(original)
    after = module.analyze_evidence(translated)
    assert before["tile_metrics"] == after["tile_metrics"]
    assert before["cross_tile_order"] == after["cross_tile_order"]
    assert [
        (row["plot_begin"], row["plot_return"])
        for row in before["timeline_events"]
    ] == [
        (row["plot_begin"], row["plot_return"])
        for row in after["timeline_events"]
    ]
    html = module.render_report(translated, after)
    assert str(translation + 5_000) in html
    assert '"begin_cycle":' not in html


def _test_report_and_cli(module: object, repo: pathlib.Path) -> None:
    evidence = make_evidence()
    evidence["sites"][0]["position"] = "</script><script>alert(1)</script>"
    analysis = module.analyze_evidence(evidence)
    report = module.render_report(evidence, analysis)
    assert report.startswith("<!doctype html>")
    assert report.count("data-static-tile=") == 16
    for label in (
        "Baseline",
        "Winner",
        "Difference",
        "Physical tile map",
        "Balanced launch-to-completion latency",
        "Engine PMU deltas",
        "Heuristic TSM structure differences",
        "cross-tile pairs",
        "host steady-clock launch-to-completion",
    ):
        assert label in report
    lower = report.lower()
    assert "fence lane" not in lower
    assert "wait lane" not in lower
    assert "ready order" not in lower
    assert "ready-order" not in lower
    assert "engine idle" not in lower
    assert "http://" not in lower and "https://" not in lower
    assert "</script><script>alert(1)</script>" not in report
    assert "\\u003c/script\\u003e" in report
    node = shutil.which("node")
    if node:
        script = report.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        checked = subprocess.run(
            [node, "--check", "-"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        evidence_path = root / "evidence.json"
        output = root / "run"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(repo / "tools" / "wafer_profile_report.py"),
                str(evidence_path),
                "--output-directory",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            completed.stdout,
            completed.stderr,
        )
        assert (output / "index.html").is_file()
        assert (output / "evidence.json").is_file()
        result = json.loads((output / "analysis.json").read_text())
        assert result["verdict"] == "improved"
        assert result["schema"] == "wafer.profile.analysis"
        assert f"profile_report: {output / 'index.html'}" in completed.stdout
        generated_html = (output / "index.html").read_text()
        assert 'href="evidence.json"' in generated_html
        assert 'href="analysis.json"' in generated_html

        rejected = copy.deepcopy(evidence)
        rejected["schema_version"] = 1
        rejected_path = root / "rejected.json"
        rejected_path.write_text(json.dumps(rejected), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(repo / "tools" / "wafer_profile_report.py"),
                str(rejected_path),
                "--output-directory",
                str(root / "rejected-output"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "profile evidence rejected" in completed.stderr
        assert not (root / "rejected-output").exists()


def _test_machine_schema(repo: pathlib.Path) -> None:
    schema = json.loads(
        (repo / "tools" / "wafer_profile_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema"]["const"] == "wafer.profile.evidence"
    assert schema["properties"]["schema_version"]["const"] == 2
    assert "runtimeLaunch" in schema["$defs"]
    assert schema["$defs"]["sharedIdentity"]["properties"][
        "execution_ranks"
    ]["const"] == 16
    assert schema["$defs"]["event"]["properties"]["engine"]["$ref"].endswith(
        "/engine"
    )
    assert schema["$defs"]["event"]["properties"]["raw_result"]["$ref"].endswith(
        "/u64"
    )
    assert {"candidate", "tile", "correlation_key"}.issubset(
        schema["$defs"]["site"]["required"]
    )
    assert {
        "target_call_ordinal",
        "target_call_symbol",
    }.issubset(schema["$defs"]["site"]["required"])
    assert "host_elapsed_ns" in schema["$defs"]["measurementSample"]["required"]
    assert schema["$defs"]["measurementSample"]["properties"][
        "host_elapsed_ns"
    ] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 18446744073709551615,
    }
    assert (
        "completion_observation_resolution_ns"
        in schema["$defs"]["measurementSample"]["required"]
    )
    assert (
        "baseline_same_as_winner"
        in schema["$defs"]["sharedIdentity"]["required"]
    )
    assert {
        "preflight_count",
        "next_sequence",
        "dropped_event_count",
        "record_flags",
        "trace_state",
    }.issubset(schema["$defs"]["traceTile"]["required"])
    assert set(schema["$defs"]["aggregateCounters"]["required"]) == {
        "statistics_window",
        "fu",
        "ct",
        "ne",
        "rdma",
        "wdma",
        "tdma",
        "scalar",
    }
    assert "cycles" not in schema["$defs"]["workerCounters"]["properties"]
    assert (
        schema["$defs"]["workerEngineCounters"]["properties"]["instructions"][
            "$ref"
        ]
        == "#/$defs/counterSnapshot32"
    )
    assert schema["properties"]["measurement"]["properties"]["samples"][
        "minItems"
    ] == 20
    fixture = make_evidence()
    assert set(fixture["identity"]) == set(
        schema["$defs"]["sharedIdentity"]["required"]
    )
    assert set(fixture["output_validation"]) == set(
        schema["$defs"]["outputValidation"]["required"]
    )
    assert set(fixture["output_validation"]["resources"][0]) == set(
        schema["$defs"]["outputValidationResource"]["required"]
    )
    assert set(fixture["measurement"]["samples"][0]) == set(
        schema["$defs"]["measurementSample"]["required"]
    )
    assert set(
        fixture["experiments"]["baseline"]["trace"]["tiles"][0]
    ) == set(schema["$defs"]["traceTile"]["required"])
    assert set(
        fixture["experiments"]["baseline"]["trace"]["tiles"][0]["events"][0]
    ) == set(schema["$defs"]["event"]["required"])
    assert schema["properties"]["measurement"]["properties"]["samples"][
        "maxItems"
    ] == 20


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    module = _load_report_module(repo)
    _test_machine_schema(repo)
    _test_verdicts(module)
    _test_semantic_invalidity(module)
    _test_structural_rejection(module)
    _test_rank_local_site_deltas_and_u64_cycles(module)
    _test_report_and_cli(module, repo)
    print("wafer_profile_report_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
