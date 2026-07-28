#!/usr/bin/env python3
"""No-card tests for final-artifact profile analysis and offline HTML."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
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
        raise AssertionError(
            f"invalid evidence was accepted; expected {marker!r}"
        )


def _test_final_artifact(module: object) -> None:
    evidence = make_evidence()
    first = module.analyze_evidence(evidence)
    second = module.analyze_evidence(copy.deepcopy(evidence))
    assert first == second

    final = first["final_artifact"]
    latency = final["latency"]
    assert latency["sample_count"] == 10
    assert len(latency["samples_ns"]) == 10
    assert latency["minimum_ns"] <= latency["median_ns"]
    assert latency["median_ns"] <= latency["maximum_ns"]
    assert latency["qualified"]
    assert latency["high_resolution"]
    assert final["output"]["correctness_status"] == "expected-exact"
    assert len(final["tiles"]) == 16
    assert len(final["timeline_events"]) > 0
    for tile in final["tiles"]:
        assert tile["summary_entry_cycles"] >= 0
        assert tile["trace_entry_cycles"] >= 0
        assert tile["timeline_axis"] == "trace-capture-entry-span"
        assert {
            engine["engine"] for engine in tile["engines"]
        } == {"CT", "NE", "RDMA", "WDMA", "TDMA", "DIRECT_DTE"}
        for engine in tile["engines"]:
            if engine["engine"] == "DIRECT_DTE":
                assert engine["wait_window_cycles"] >= 0
                assert engine["raw_pmu_activity"] is None
                continue
            assert engine["busy_cycles"] is None or engine["busy_cycles"] >= 0
    assert all(
        0.0 <= event["plot_begin_fraction"]
        <= event["plot_end_fraction"]
        <= 1.0
        for event in final["timeline_events"]
    )
    assert all(
        event["trace_entry_offset_end"]
        <= final["tiles"][event["tile"]]["trace_entry_cycles"]
        for event in final["timeline_events"]
    )
    serialized = json.dumps(first, sort_keys=True)
    for legacy in ("baseline", "winner", "ABBA", "BAAB", "speedup"):
        assert legacy not in serialized

    report = module.render_report(evidence, first)
    for text in (
        "最终编译产物板卡 Profile",
        "最终产物整体耗时",
        "Tile 总览",
        "Engine 活动 Timeline",
        "DIRECT_DTE",
        "本目录只有三个产物",
    ):
        assert text in report
    assert "Measurement invalid" not in report


def _test_direct_dte(module: object) -> None:
    evidence = make_evidence()
    site = next(
        row
        for row in evidence["sites"]
        if row["tile"] == 0 and row["site_id"] == 0
    )
    site["engine"] = "DIRECT_DTE"
    event = evidence["experiment"]["trace"]["tiles"][0]["events"][0]
    event["engine"] = "DIRECT_DTE"
    event["dte_role"] = "send"
    event["counter_delta"] = 0
    event["dte_counter_valid"] = True
    analysis = module.analyze_evidence(evidence)
    direct = next(
        row
        for row in analysis["final_artifact"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert direct["wait_window_cycles"] == (
        event["observed_end_cycle"] - event["observed_begin_cycle"]
    )
    assert direct["measurement_kind"] == (
        "direct-dte-wait-completion-windows"
    )
    assert direct["raw_pmu_activity"] == 0
    assert direct["raw_pmu_activity_valid"]
    assert direct["activity_window_count"] == 1
    assert direct["wait_window_count"] == 1
    timeline = analysis["final_artifact"]["timeline_events"]
    assert any(
        row["engine"] == "DIRECT_DTE" and row["dte_role"] == "send"
        for row in timeline
    )

    raw_unavailable = make_evidence()
    raw_unavailable["sites"][0]["engine"] = "DIRECT_DTE"
    raw_event = raw_unavailable["experiment"]["trace"]["tiles"][0]["events"][0]
    raw_event["engine"] = "DIRECT_DTE"
    raw_event["dte_role"] = "receive"
    raw_event["counter_delta"] = 0
    raw_event["dte_counter_valid"] = False
    raw_analysis = module.analyze_evidence(raw_unavailable)
    raw_direct = next(
        row
        for row in raw_analysis["final_artifact"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert raw_direct["wait_windows_valid"]
    assert raw_direct["wait_window_cycles"] == (
        raw_event["observed_end_cycle"]
        - raw_event["observed_begin_cycle"]
    )
    assert raw_direct["raw_pmu_activity"] is None
    assert not raw_direct["raw_pmu_activity_valid"]
    assert any(
        row["code"] == "direct_dte_raw_pmu_unusable"
        for row in raw_analysis["diagnostics"]
    )


def _test_validity(module: object) -> None:
    coarse = make_evidence()
    for sample in coarse["measurement"]["samples"]:
        sample["completion_observation_resolution_ns"] = (
            sample["host_elapsed_ns"] // 2
        )
    analysis = module.analyze_evidence(coarse)
    latency = analysis["final_artifact"]["latency"]
    assert latency["qualified"]
    assert not latency["high_resolution"]
    warning = next(
        row
        for row in analysis["diagnostics"]
        if row["code"] == "completion_resolution_too_coarse"
    )
    assert warning["severity"] == "warning"
    report = module.render_report(coarse, analysis)
    assert "测量有效" in report

    mismatch = make_evidence()
    mismatch["output_validation"]["resources"][0][
        "diagnostic_captures_exact"
    ] = False
    analysis = module.analyze_evidence(mismatch)
    assert not analysis["final_artifact"]["latency"]["qualified"]
    assert not analysis["validity"]["output_equivalence"]

    backwards = make_evidence()
    counter = backwards["experiment"]["pmu"]["tiles"][3]["aggregates"]["ne"]
    counter["end"] = counter["start"] - 1
    counter["recovery"] = counter["end"]
    analysis = module.analyze_evidence(backwards)
    ne = next(
        row
        for row in analysis["final_artifact"]["tiles"][3]["engines"]
        if row["engine"] == "NE"
    )
    assert not ne["busy_cycles_valid"]
    assert ne["busy_cycles"] is None

    empty = make_evidence()
    empty["sites"] = []
    for tile in empty["experiment"]["trace"]["tiles"]:
        tile["capacity"] = 0
        tile["count"] = 0
        tile["preflight_count"] = 0
        tile["next_sequence"] = 0
        tile["events"] = []
    analysis = module.analyze_evidence(empty)
    assert analysis["validity"]["trace"]
    assert analysis["final_artifact"]["timeline_events"] == []


def _test_rejections(module: object) -> None:
    old = make_evidence()
    old["schema_version"] = 3
    _must_reject(module, old, "schema_version")

    old_companion = make_evidence()
    old_companion["identity"]["profile_companion_schema_version"] = 1
    _must_reject(module, old_companion, "profile_companion_schema_version")

    duplicate = make_evidence()
    duplicate["measurement"]["samples"][1]["sample_index"] = 0
    _must_reject(module, duplicate, "duplicate sample index")

    missing = make_evidence()
    del missing["measurement"]["samples"][0]["host_elapsed_ns"]
    _must_reject(module, missing, "missing keys")

    malformed_digest = make_evidence()
    malformed_digest["identity"]["production_manifest_sha256"] = "sha256:no"
    _must_reject(module, malformed_digest, "64 lowercase hex digits")

    legacy_site = make_evidence()
    legacy_site["sites"][0]["candidate"] = "winner"
    _must_reject(module, legacy_site, "unknown keys")

    missing_dte_window = make_evidence()
    missing_dte_window["sites"][0]["engine"] = "DIRECT_DTE"
    dte_event = missing_dte_window["experiment"]["trace"]["tiles"][0][
        "events"
    ][0]
    dte_event["engine"] = "DIRECT_DTE"
    dte_event["dte_role"] = "send"
    dte_event["dte_counter_valid"] = False
    dte_event["counter_delta"] = 0
    dte_event["activity_valid"] = False
    _must_reject(
        module,
        missing_dte_window,
        "completed Direct-DTE wait window",
    )

    negative_window = make_evidence()
    event = negative_window["experiment"]["trace"]["tiles"][0]["events"][0]
    event["observed_end_cycle"] = event["observed_begin_cycle"] - 1
    analysis = module.analyze_evidence(negative_window)
    assert not analysis["validity"]["trace"]
    assert any(
        row["code"] == "ncc_activity_window_unusable"
        for row in analysis["diagnostics"]
    )

    reversed_dte_window = make_evidence()
    reversed_dte_window["sites"][0]["engine"] = "DIRECT_DTE"
    dte_event = reversed_dte_window["experiment"]["trace"]["tiles"][0][
        "events"
    ][0]
    dte_event["engine"] = "DIRECT_DTE"
    dte_event["dte_role"] = "send"
    dte_event["dte_counter_valid"] = True
    dte_event["observed_end_cycle"] = dte_event["observed_begin_cycle"] - 1
    dte_analysis = module.analyze_evidence(reversed_dte_window)
    dte_row = next(
        row
        for row in dte_analysis["final_artifact"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert not dte_row["wait_windows_valid"]
    assert dte_row["wait_window_cycles"] is None
    assert dte_row["wait_window_count"] == 1
    assert dte_row["activity_window_count"] == 0


def _test_publication(repo: pathlib.Path, module: object) -> None:
    evidence = make_evidence()
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        input_path = root / "input.json"
        output = root / "run"
        input_path.write_text(json.dumps(evidence), encoding="utf-8")
        html_path, analysis_path = module.generate_report(input_path, output)
        assert html_path == output / "index.html"
        assert analysis_path == output / "analysis.json"
        assert {path.name for path in output.iterdir()} == {
            "evidence.json",
            "analysis.json",
            "index.html",
        }
        assert (output.stat().st_mode & 0o777) == 0o777
        assert all(
            (path.stat().st_mode & 0o777) == 0o777
            for path in output.iterdir()
        )
        json.loads((output / "evidence.json").read_text(encoding="utf-8"))
        analysis = json.loads(
            analysis_path.read_text(encoding="utf-8")
        )
        assert analysis["final_artifact"]["latency"]["qualified"]

        command = [
            sys.executable,
            str(repo / "tools" / "wafer_profile_report.py"),
            str(input_path),
            "--output-directory",
            str(root / "cli-run"),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "profile_report:" in result.stdout

    schema = json.loads(
        (repo / "tools" / "wafer_profile_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == 4
    assert (
        schema["$defs"]["sharedIdentity"]["properties"][
            "profile_companion_schema_version"
        ]["const"]
        == 2
    )
    assert "experiment" in schema["required"]
    assert "experiments" not in schema["required"]
    assert "candidate" not in schema["$defs"]["site"]["properties"]
    assert "recovery" in schema["$defs"]["counterSnapshot"]["required"]
    assert "recovery" in schema["$defs"]["counterSnapshot32"]["required"]
    assert schema["$defs"]["event"]["allOf"]
    assert (
        schema["$defs"]["sharedIdentity"]["properties"][
            "production_manifest_sha256"
        ]["$ref"]
        == "#/$defs/sha256"
    )


def main() -> int:
    repo = (
        pathlib.Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else pathlib.Path(__file__).resolve().parents[2]
    )
    module = _load_report_module(repo)
    _test_final_artifact(module)
    _test_direct_dte(module)
    _test_validity(module)
    _test_rejections(module)
    _test_publication(repo, module)
    print("wafer_profile_report_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
