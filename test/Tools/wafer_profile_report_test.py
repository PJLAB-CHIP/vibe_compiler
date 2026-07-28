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
from html.parser import HTMLParser

from wafer_profile_report_fixture import make_evidence


class ReportDOMContract(HTMLParser):
    """Small dependency-free DOM/static-interaction contract checker."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.view_targets: set[str] = set()
        self.view_panels: set[str] = set()
        self.raw_targets: set[str] = set()
        self.external_scripts: list[str] = []
        self.script_fragments: list[str] = []
        self._in_script = False

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        attrs = dict(attributes)
        node_id = attrs.get("id")
        if node_id:
            if node_id in self.ids:
                self.duplicate_ids.add(node_id)
            self.ids.add(node_id)
        if attrs.get("data-view"):
            self.view_targets.add(str(attrs["data-view"]))
        if attrs.get("data-view-panel"):
            self.view_panels.add(str(attrs["data-view-panel"]))
        if attrs.get("data-raw-target"):
            self.raw_targets.add(str(attrs["data-raw-target"]))
        if tag == "script":
            self._in_script = True
            if attrs.get("src"):
                self.external_scripts.append(str(attrs["src"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.script_fragments.append(data)


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
    assert first["schema_version"] == 3

    final = first["final_artifact"]
    duration = final["duration"]
    assert duration["sample_id"] == "primary"
    assert duration["sample_index"] == 0
    assert duration["host_elapsed_ns"] == 1_018_000
    assert duration["qualified"]
    assert duration["high_resolution"]
    assert duration["status"] == "Measured"
    assert "latency" not in final
    assert "samples_ns" not in json.dumps(first)
    assert final["output"]["production_execution_validated"]
    assert final["output"]["diagnostic_captures_match_primary"]
    assert final["output"]["correctness_status"] == "expected-exact"
    assert len(final["tiles"]) == 16
    assert len(final["timeline_events"]) > 0
    assert len(final["sites"]) == 16 * 6
    assert final["communication"]["direct_dte_event_count"] == 16
    assert not final["communication"]["cross_tile_order_available"]
    assert {
        (tile["x"], tile["y"]) for tile in final["tiles"]
    } == {(x, y) for y in range(4) for x in range(4)}
    for tile in final["tiles"]:
        assert tile["trace_entry_cycles"] >= 0
        assert tile["timeline_axis"] == "trace-capture-entry-span"
        assert tile["timeline_scope"] == "tile-local"
        assert tile["trace_status"] == "Bounded"
        assert {
            engine["engine"] for engine in tile["engines"]
        } == {"CT", "NE", "RDMA", "WDMA", "TDMA", "DIRECT_DTE"}
        for engine in tile["engines"]:
            if engine["engine"] == "DIRECT_DTE":
                assert engine["wait_window_cycles"] >= 0
                assert engine["raw_pmu_activity"] is None
                assert engine["wait_window_status"] == "Measured"
                assert engine["raw_pmu_activity_status"] == "Unavailable"
                continue
            assert engine["busy_cycles"] is None or engine["busy_cycles"] >= 0
            assert engine["busy_cycles_status"] == "Measured"
            assert engine["activity_window_status"] == "Bounded"
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
    assert all(event["timeline_scope"] == "tile-local" for event in final["timeline_events"])
    ncc_event = next(
        event for event in final["timeline_events"] if event["engine"] == "CT"
    )
    assert ncc_event["duration_status"] == "Bounded"
    assert ncc_event["counter_status"] == "Sampled"
    assert ncc_event["correlation_key"] == "compute.ct"
    assert ncc_event["target_call_symbol"] == "wafer_tx81_tsm_ct_execute"
    dte_event = next(
        event
        for event in final["timeline_events"]
        if event["engine"] == "DIRECT_DTE"
    )
    assert dte_event["duration_status"] == "Measured"
    assert dte_event["counter_status"] == "Unavailable"
    serialized = json.dumps(first, sort_keys=True)
    for legacy in (
        "baseline",
        "winner",
        "ABBA",
        "BAAB",
        "speedup",
        "median_ns",
        "minimum_ns",
        "maximum_ns",
        "summary_entry",
    ):
        assert legacy not in serialized

    report = module.render_report(evidence, first)
    for text in (
        "最终编译产物板卡 Profile",
        "Primary duration",
        "Trace Timeline",
        "Tile / Engine",
        "Program / Sites",
        "Communication / DTE",
        "Diagnostics / Raw",
        "Resource tree",
        "Engine summary",
        "derived activity volume",
        "DIRECT_DTE",
        "Measured",
        "Sampled",
        "Bounded",
        "Unavailable",
        "Incomplete",
        "Invalid",
        "tile-local",
        "index.html · analysis.json · evidence.json",
    ):
        assert text in report
    assert "Measurement invalid" not in report
    assert "中位数" not in report
    assert "范围 " not in report
    assert "gradient" not in report.lower()
    assert "grid-template-columns:repeat(4" in report
    _test_dom_contract(report)


def _test_dom_contract(report: str) -> None:
    parser = ReportDOMContract()
    parser.feed(report)
    expected_views = {
        "overview",
        "timeline",
        "engines",
        "sites",
        "communication",
        "diagnostics",
    }
    assert not parser.duplicate_ids
    assert parser.view_targets == expected_views
    assert parser.view_panels == expected_views
    assert parser.raw_targets == {"analysis", "evidence"}
    assert not parser.external_scripts
    assert {
        "resourceTree",
        "overviewEngineRows",
        "timelineTile",
        "engineFilters",
        "timelineZoom",
        "timelineFit",
        "timelineRuler",
        "timelineLanes",
        "eventDetail",
        "engineTile",
        "engineRows",
        "siteSearch",
        "siteEngine",
        "siteRows",
        "communicationRows",
        "diagnosticRows",
        "rawPayload",
    }.issubset(parser.ids)
    script = "\n".join(parser.script_fragments)
    for interaction in (
        'addEventListener("click"',
        'addEventListener("change"',
        'addEventListener("input"',
        "data-lane-engine",
        "data-event-sequence",
        "data-event-tile",
        'addEventListener("keydown"',
        "data-tree-tile",
        "window.__waferProfileUI",
        "focusEvent",
        "renderTimeline",
        "renderEventDetail",
        "renderSites",
        "renderCommunication",
    ):
        assert interaction in script
    assert all(f'"{engine}"' in script for engine in (
        "CT",
        "NE",
        "RDMA",
        "WDMA",
        "TDMA",
        "DIRECT_DTE",
    ))


def _test_direct_dte(module: object) -> None:
    evidence = make_evidence()
    event = evidence["experiment"]["trace"]["tiles"][0]["events"][-1]
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
    assert direct["raw_pmu_activity"] is None
    assert not direct["raw_pmu_activity_valid"]
    assert direct["wait_window_status"] == "Measured"
    assert direct["raw_pmu_activity_status"] == "Unavailable"
    assert direct["activity_window_count"] == 1
    assert direct["wait_window_count"] == 1
    timeline = analysis["final_artifact"]["timeline_events"]
    assert any(
        row["engine"] == "DIRECT_DTE" and row["dte_role"] == "send"
        for row in timeline
    )

    raw_available = make_evidence()
    raw_event = raw_available["experiment"]["trace"]["tiles"][0]["events"][-1]
    raw_event["counter_delta"] = 17
    raw_event["dte_counter_valid"] = True
    raw_analysis = module.analyze_evidence(raw_available)
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
    assert raw_direct["raw_pmu_activity"] == 17
    assert raw_direct["raw_pmu_activity_valid"]
    assert raw_direct["raw_pmu_activity_status"] == "Sampled"
    raw_timeline = raw_analysis["final_artifact"]["timeline_events"]
    raw_timeline_event = next(
        row
        for row in raw_timeline
        if row["tile"] == 0 and row["engine"] == "DIRECT_DTE"
    )
    assert raw_timeline_event["counter_status"] == "Sampled"


def _test_validity(module: object) -> None:
    coarse = make_evidence()
    sample = coarse["measurement"]["samples"][0]
    sample["completion_observation_resolution_ns"] = sample["host_elapsed_ns"] // 2
    analysis = module.analyze_evidence(coarse)
    duration = analysis["final_artifact"]["duration"]
    assert duration["qualified"]
    assert not duration["high_resolution"]
    assert duration["status"] == "Measured"
    warning = next(
        row
        for row in analysis["diagnostics"]
        if row["code"] == "completion_resolution_too_coarse"
    )
    assert warning["severity"] == "warning"
    report = module.render_report(coarse, analysis)
    assert "completion observation resolution is too coarse" in report

    mismatch = make_evidence()
    mismatch["output_validation"]["resources"][0][
        "diagnostic_captures_match_primary"
    ] = False
    analysis = module.analyze_evidence(mismatch)
    assert not analysis["final_artifact"]["duration"]["qualified"]
    assert analysis["final_artifact"]["duration"]["status"] == "Invalid"
    assert not analysis["validity"]["output_equivalence"]

    production_failure = make_evidence()
    production_failure["output_validation"]["resources"][0][
        "production_execution_validated"
    ] = False
    analysis = module.analyze_evidence(production_failure)
    assert analysis["validity"]["semantic_correctness"] is False
    assert (
        analysis["final_artifact"]["output"]["correctness_status"]
        == "production-validation-failed"
    )
    assert not analysis["final_artifact"]["duration"]["qualified"]

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
    assert ne["busy_cycles_status"] == "Unavailable"

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
    assert all(
        site["observation_status"] == "Unavailable"
        for site in analysis["final_artifact"]["sites"]
    )

    reversed_trace = make_evidence()
    tile = reversed_trace["experiment"]["trace"]["tiles"][4]
    tile["entry_end_cycle"] = tile["entry_begin_cycle"] - 1
    analysis = module.analyze_evidence(reversed_trace)
    assert not analysis["validity"]["trace"]
    assert analysis["final_artifact"]["tiles"][4]["trace_entry_cycles"] is None
    assert analysis["final_artifact"]["tiles"][4]["trace_status"] == "Invalid"
    assert not any(
        event["tile"] == 4
        for event in analysis["final_artifact"]["timeline_events"]
    )


def _test_rejections(module: object) -> None:
    old = make_evidence()
    old["schema_version"] = 4
    _must_reject(module, old, "schema_version")

    old_companion = make_evidence()
    old_companion["identity"]["profile_companion_schema_version"] = 2
    _must_reject(module, old_companion, "profile_companion_schema_version")

    duplicate = make_evidence()
    duplicate["measurement"]["samples"].append(
        copy.deepcopy(duplicate["measurement"]["samples"][0])
    )
    _must_reject(module, duplicate, "exactly one primary execution")

    wrong_primary = make_evidence()
    wrong_primary["measurement"]["samples"][0]["sample_id"] = "final-s0"
    _must_reject(module, wrong_primary, "must be 'primary'")

    missing = make_evidence()
    del missing["measurement"]["samples"][0]["host_elapsed_ns"]
    _must_reject(module, missing, "missing keys")

    malformed_digest = make_evidence()
    malformed_digest["identity"]["production_manifest_sha256"] = "sha256:no"
    _must_reject(module, malformed_digest, "64 lowercase hex digits")

    legacy_site = make_evidence()
    legacy_site["sites"][0]["candidate"] = "winner"
    _must_reject(module, legacy_site, "unknown keys")

    legacy_output = make_evidence()
    resource = legacy_output["output_validation"]["resources"][0]
    resource["production_repeats_exact"] = resource.pop(
        "production_execution_validated"
    )
    _must_reject(module, legacy_output, "unknown keys")

    stale_summary = make_evidence()
    stale_summary["experiment"]["summary"] = {"tiles": []}
    _must_reject(module, stale_summary, "unknown keys")

    oversized_output = make_evidence()
    oversized_output["output_validation"]["resources"][0]["bytes"] = 1 << 64
    _must_reject(module, oversized_output, "must be at most")

    for field in (
        "observed_begin_cycle",
        "observed_end_cycle",
        "counter_delta",
    ):
        oversized_event = make_evidence()
        oversized_event["experiment"]["trace"]["tiles"][0]["events"][0][
            field
        ] = 1 << 64
        _must_reject(module, oversized_event, "must be at most")

    for field in ("dropped_event_count", "record_flags"):
        oversized_trace_field = make_evidence()
        oversized_trace_field["experiment"]["trace"]["tiles"][0][field] = (
            1 << 32
        )
        _must_reject(module, oversized_trace_field, "must be at most")

    invalid_trace_state = make_evidence()
    invalid_trace_state["experiment"]["trace"]["tiles"][0]["trace_state"] = 5
    _must_reject(module, invalid_trace_state, "must be at most 4")

    oversized_worker_counter = make_evidence()
    oversized_worker_counter["experiment"]["pmu"]["tiles"][0]["workers"][0][
        "engines"
    ][0]["instructions"]["start"] = 1 << 32
    _must_reject(module, oversized_worker_counter, "must be at most")

    for invalid_worker in (False, 0.5, "0"):
        invalid_worker_id = make_evidence()
        invalid_worker_id["experiment"]["pmu"]["tiles"][0]["workers"][0][
            "worker"
        ] = invalid_worker
        _must_reject(module, invalid_worker_id, "expected an integer")

    duplicate_worker_engine = make_evidence()
    engines = duplicate_worker_engine["experiment"]["pmu"]["tiles"][0][
        "workers"
    ][0]["engines"]
    engines[-1] = copy.deepcopy(engines[0])
    _must_reject(module, duplicate_worker_engine, "exactly once")

    extra_worker_engine = make_evidence()
    engines = extra_worker_engine["experiment"]["pmu"]["tiles"][0]["workers"][
        0
    ]["engines"]
    engines.append(copy.deepcopy(engines[0]))
    _must_reject(module, extra_worker_engine, "exactly once")

    invalid_worker_engine = make_evidence()
    invalid_worker_engine["experiment"]["pmu"]["tiles"][0]["workers"][0][
        "engines"
    ][0]["engine"] = "DIRECT_DTE"
    _must_reject(module, invalid_worker_engine, "must be one of")

    invalid_comparison_type = make_evidence()
    invalid_comparison_type["output_validation"]["resources"][0][
        "external_expected_comparison"
    ] = ["exact"]
    _must_reject(module, invalid_comparison_type, "must be null")

    missing_dte_window = make_evidence()
    dte_event = missing_dte_window["experiment"]["trace"]["tiles"][0][
        "events"
    ][-1]
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
    dte_event = reversed_dte_window["experiment"]["trace"]["tiles"][0][
        "events"
    ][-1]
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
        assert analysis["schema_version"] == 3
        assert analysis["final_artifact"]["duration"]["qualified"]

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
    assert schema["properties"]["schema_version"]["const"] == 5
    assert (
        schema["$defs"]["sharedIdentity"]["properties"][
            "profile_companion_schema_version"
        ]["const"]
        == 3
    )
    assert "experiment" in schema["required"]
    assert "experiments" not in schema["required"]
    assert "candidate" not in schema["$defs"]["site"]["properties"]
    assert "entryTiming" not in schema["$defs"]
    assert set(schema["$defs"]["experiment"]["required"]) == {
        "artifact",
        "clock",
        "trace",
        "pmu",
    }
    assert "summary" not in schema["$defs"]["experiment"]["properties"]
    sample = schema["$defs"]["measurementSample"]["properties"]
    assert sample["sample_id"]["const"] == "primary"
    assert sample["sample_index"]["const"] == 0
    output_resource = schema["$defs"]["outputValidationResource"]
    assert "production_execution_validated" in output_resource["required"]
    assert "diagnostic_captures_match_primary" in output_resource["required"]
    assert "production_repeats_exact" not in output_resource["properties"]
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
