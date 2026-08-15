#!/usr/bin/env python3
"""No-card tests for primary-program profile analysis and offline HTML."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import re
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


def _test_program(module: object) -> None:
    evidence = make_evidence()
    first = module.analyze_evidence(evidence)
    second = module.analyze_evidence(copy.deepcopy(evidence))
    assert first == second
    assert first["record_abi"] == "wafer-tx81-profiler-record"

    permuted = make_evidence(permute_bindings=True)
    module.validate_evidence(permuted)

    final = first["program"]
    duration = final["duration"]
    assert duration["sample_id"] == "primary"
    assert duration["sample_index"] == 0
    assert duration["device_elapsed_ns"] == 8_400
    assert duration["device_timer_kind"] == "tx-stream-events"
    assert duration["host_submit_ns"] == 260_000
    assert duration["host_launch_to_completion_ns"] == 1_018_000
    assert (
        duration["measurement_kind"]
        == "tx-stream-kernel-launch-to-completion-envelope"
    )
    assert duration["scope"] == "primary-launch-to-stream-completion"
    assert not duration["is_engine_only"]
    assert duration["includes_device_control_wait_and_scheduling"]
    assert not duration["includes_host_submit"]
    assert not duration["includes_host_completion_polling"]
    assert not duration["includes_trace_instrumentation"]
    assert duration["qualified"]
    assert duration["host_completion_high_resolution"]
    assert duration["status"] == "Measured"
    assert "latency" not in final
    assert "samples_ns" not in json.dumps(first)
    engine_active = final["engine_active_time"]
    assert engine_active["source_capture"] == "trace"
    assert engine_active["measurement_run"] == "trace-diagnostic"
    assert engine_active["relation_to_primary"] == "separate-run-proxy"
    assert engine_active["unit"] == "ns"
    assert engine_active["scope"] == "per-tile-per-ncc-engine"
    assert not engine_active["additive_to_kernel_launch_envelope"]
    assert engine_active["cross_tile_sum_role"] == "work-volume-not-wall-time"
    assert engine_active["card_wide_engine_elapsed_ns"] is None
    assert engine_active["card_wide_engine_elapsed_status"] == "Unavailable"
    assert {
        row["engine"] for row in engine_active["by_engine"]
    } == {"CT", "NE", "RDMA", "WDMA", "TDMA"}
    assert all(
        row["measurement_kind"] == "trace-pmu-engine-active-time"
        and row["available_tile_count"] == 16
        and row["active_tile_count"] == 16
        and row["status"] == "Measured"
        for row in engine_active["by_engine"]
    )
    ct_summary = next(
        row for row in engine_active["by_engine"] if row["engine"] == "CT"
    )
    assert ct_summary["minimum_per_tile_ns"] == 84
    assert ct_summary["average_per_tile_ns"] == 91.5
    assert ct_summary["maximum_per_tile_ns"] == 99
    assert ct_summary["sum_across_tiles_work_ns"] == 1_464
    tile_work = engine_active["per_tile_work_volume"]
    assert tile_work == {
        "available_tile_count": 16,
        "minimum_per_tile_total_work_ns": 530,
        "average_per_tile_total_work_ns": 567.5,
        "maximum_per_tile_total_work_ns": 605,
        "sum_across_tiles_work_ns": 9_080,
        "status": "Measured",
    }
    hardware = final["hardware_cost_analysis"]
    assert hardware["model"] == "tx81-static-throughput-lower-bound"
    assert (
        hardware["scope"]
        == "complete-final-instruction-program-per-physical-tile"
    )
    assert hardware["source"] == "compiler-static-final-instruction-program"
    assert hardware["relation_to_primary"] == "non-additive-model-reference"
    assert not hardware["additive_to_primary"]
    assert [row["engine"] for row in hardware["by_engine"]] == [
        "CT",
        "NE",
        "RDMA",
        "WDMA",
        "TDMA",
        "DIRECT_DTE",
    ]
    hardware_by_engine = {
        row["engine"]: row for row in hardware["by_engine"]
    }
    assert hardware_by_engine["CT"]["model_status"] == (
        "theoretical-lower-bound"
    )
    assert hardware_by_engine["CT"]["work"]["aggregate"] == str(5_856 * 16)
    assert hardware_by_engine["CT"]["estimated_ns"] == 91.5
    assert hardware_by_engine["CT"]["measured_active_ns"][
        "average_per_tile_ns"
    ] == 91.5
    assert hardware_by_engine["CT"]["measured_to_model_ratio"] == 1.0
    assert hardware_by_engine["NE"]["estimated_ns"] == 102.5
    assert hardware_by_engine["NE"]["measured_to_model_ratio"] == 1.0
    assert hardware_by_engine["RDMA"]["model_status"] == "heuristic"
    assert hardware_by_engine["RDMA"]["floor_ns"] == 112.0
    assert hardware_by_engine["RDMA"]["estimated_ns"] == 112.0
    assert hardware_by_engine["WDMA"]["model_status"] == "heuristic"
    assert hardware_by_engine["WDMA"]["floor_ns"] == 124.48
    assert hardware_by_engine["TDMA"]["model_status"] == "unavailable"
    assert hardware_by_engine["TDMA"]["estimated_ns"] is None
    assert hardware_by_engine["TDMA"]["work"]["aggregate"] == str(1_024 * 16)
    assert hardware_by_engine["DIRECT_DTE"]["model_status"] == (
        "reference-only"
    )
    assert hardware_by_engine["DIRECT_DTE"]["work"]["aggregate"] == str(
        1_280 * 16
    )
    assert hardware_by_engine["DIRECT_DTE"]["estimated_scope"] == (
        "average-per-tile-single-link-payload-reference"
    )
    assert hardware_by_engine["DIRECT_DTE"]["estimated_ns"] == 10.0
    assert hardware_by_engine["DIRECT_DTE"]["floor_ns"] is None
    assert hardware_by_engine["DIRECT_DTE"]["measured_to_model_ratio"] is None
    assert final["output"]["primary_output_validated"]
    assert final["output"]["diagnostic_captures_match_primary"]
    assert final["output"]["correctness_status"] == "expected-exact"
    assert len(final["tiles"]) == 16
    assert len(final["timeline_events"]) > 0
    assert len(final["sites"]) == 16 * 10
    assert final["communication"]["direct_dte_event_count"] == 24
    assert final["communication"]["issue_event_count"] == 8
    assert final["communication"]["wait_event_count"] == 16
    assert not final["communication"]["cross_tile_order_available"]
    assert {
        (tile["x"], tile["y"]) for tile in final["tiles"]
    } == {(x, y) for y in range(4) for x in range(4)}
    for tile in final["tiles"]:
        assert tile["trace_entry_cpu_cycles"] >= 0
        assert tile["timeline_axis"] == "kcore-rdcycle-entry-span"
        assert tile["timeline_scope"] == "tile-local"
        assert tile["trace_status"] == "Measured"
        assert tile["statistics_window_raw_ticks"] == 820 + tile["tile"]
        assert tile["statistics_window_valid"]
        assert tile["statistics_window_status"] == "Measured"
        semantic = tile["semantic_partition"]
        assert semantic["exclusive_accounting_valid"]
        assert semantic["exclusive_cycles"] == tile["trace_entry_cpu_cycles"]
        assert sum(row["cycles"] for row in semantic["rows"]) == semantic[
            "entry_cycles"
        ]
        assert {
            "entry-prologue",
            "entry-epilogue",
            "between-site-gap",
            "site-control",
        }.issubset({row["category"] for row in semantic["rows"]})
        assert all(
            row["previous_site"] is not None
            and row["next_site"] is not None
            for row in semantic["rows"]
            if row["category"] == "between-site-gap"
        )
        overlay = tile["trace_overhead_overlay"]
        assert overlay["components_exclusive"]
        assert overlay["non_additive_to_semantic_partition"]
        assert overlay["coverage"] == "measured-categories-only"
        assert len(overlay["rows"]) == 8
        assert overlay["inside_entry_known_overhead_cycles"] == 57
        assert overlay["outside_entry_overhead_cycles"] == 14
        assert all(
            row["share_of_trace_entry"] is None
            for row in overlay["rows"]
            if row["location_granularity"] == "outside-entry"
        )
        overlay_by_reason = {
            row["reason"]: row for row in overlay["rows"]
        }
        assert all(
            row["counts_in_primary_device_elapsed"] == "no"
            and row["magnitude_relation"] == "trace-only"
            for row in overlay_by_reason.values()
        )
        assert {
            engine["engine"] for engine in tile["engines"]
        } == {"CT", "NE", "RDMA", "WDMA", "TDMA", "DIRECT_DTE"}
        for engine in tile["engines"]:
            if engine["engine"] == "DIRECT_DTE":
                assert engine["wait_window_cpu_cycles"] >= 0
                assert engine["raw_pmu_activity"] is None
                assert engine["wait_window_status"] == "Measured"
                assert engine["raw_pmu_activity_status"] == "Unavailable"
                continue
            assert (
                engine["engine_execution_time_ns"] is None
                or engine["engine_execution_time_ns"] >= 0
            )
            assert engine["engine_execution_time_status"] == "Measured"
            assert engine["activity_window_status"] == "Bounded"
    assert all(
        0.0 <= event["plot_begin_fraction"]
        <= event["plot_end_fraction"]
        <= 1.0
        for event in final["timeline_events"]
    )
    assert all(
        event["trace_entry_offset_end_cpu_cycles"]
        <= final["tiles"][event["tile"]]["trace_entry_cpu_cycles"]
        for event in final["timeline_events"]
    )
    assert all(event["timeline_scope"] == "tile-local" for event in final["timeline_events"])
    ncc_event = next(
        event for event in final["timeline_events"] if event["engine"] == "CT"
    )
    assert ncc_event["duration_status"] == "Bounded"
    assert ncc_event["display_interval_role"] == "command-submit"
    assert ncc_event["operation_window_status"] == "Measured"
    assert ncc_event["activity_window_status"] == "Bounded"
    assert (
        ncc_event["trace_entry_offset_end_cpu_cycles"]
        - ncc_event["trace_entry_offset_begin_cpu_cycles"]
        == ncc_event["operation_window_cpu_cycles"]
    )
    assert (
        ncc_event["activity_trace_entry_offset_end_cpu_cycles"]
        - ncc_event["activity_trace_entry_offset_begin_cpu_cycles"]
        == ncc_event["activity_window_cpu_cycles"]
    )
    assert ncc_event["counter_status"] == "Measured"
    assert ncc_event["ncc_engine_execution_time_ns"] > 0
    assert ncc_event["correlation_key"] == "compute.ct"
    assert ncc_event["target_call_symbol"] == "wafer_tx81_tsm_ct_execute"
    completion_event = next(
        event
        for event in final["timeline_events"]
        if event["kind"] == "ncc-completion-wait"
    )
    assert completion_event["duration_status"] == "Measured"
    assert completion_event["counter_status"] == "Unavailable"
    dte_event = next(
        event
        for event in final["timeline_events"]
        if event["kind"] == "direct-dte-wait"
    )
    assert dte_event["duration_status"] == "Measured"
    assert dte_event["display_interval_role"] == "operation-window"
    assert dte_event["counter_status"] == "Unavailable"
    dte_phase_events = [
        event
        for event in final["timeline_events"]
        if event["tile"] == 0 and event["kind"].startswith("direct-dte-")
    ]
    assert {event["kind"] for event in dte_phase_events} == {
        "direct-dte-issue",
        "direct-dte-wait",
        "direct-dte-peer-ready-wait",
        "direct-dte-setup-issue",
        "direct-dte-completion-wait",
        "direct-dte-cleanup",
    }
    assert all(
        event["operation_window_cpu_cycles"] > 0
        and event["duration_status"] == "Measured"
        for event in dte_phase_events
    )
    assert all(
        event["direct_dte_raw_pmu_activity"] is None
        and event["counter_status"] == "Unavailable"
        for event in dte_phase_events
        if event["kind"] not in ("direct-dte-issue", "direct-dte-wait")
    )
    assert all(
        site["observation_status"]
        in {
            "Measured",
            "Sampled",
            "Bounded",
            "Zero delta",
            "Attribution ambiguous",
            "Unavailable",
            "Incomplete",
            "Invalid",
        }
        for site in final["sites"]
    )
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

    overlapping = make_evidence()
    overlapping_events = overlapping["experiment"]["trace"]["tiles"][0][
        "events"
    ]
    overlap_ct = next(
        event
        for event in overlapping_events
        if event["kind"] == "ncc-command" and event["engine"] == "CT"
    )
    overlap_wdma = next(
        event
        for event in overlapping_events
        if event["kind"] == "ncc-command" and event["engine"] == "WDMA"
    )
    shared_sample_end = overlap_wdma["observed_end_cycle"]
    overlap_ct["observed_end_cycle"] = shared_sample_end
    overlap_ct_analysis = module.analyze_evidence(overlapping)
    overlap_rows = overlap_ct_analysis["program"]["timeline_events"]
    overlap_ct_row = next(
        event
        for event in overlap_rows
        if event["tile"] == 0 and event["engine"] == "CT"
    )
    overlap_wdma_row = next(
        event
        for event in overlap_rows
        if event["tile"] == 0 and event["engine"] == "WDMA"
    )
    assert (
        overlap_ct_row["activity_trace_entry_offset_begin_cpu_cycles"]
        < overlap_wdma_row["activity_trace_entry_offset_end_cpu_cycles"]
        and overlap_wdma_row["activity_trace_entry_offset_begin_cpu_cycles"]
        < overlap_ct_row["activity_trace_entry_offset_end_cpu_cycles"]
    )
    assert (
        overlap_ct_row["trace_entry_offset_end_cpu_cycles"]
        <= overlap_wdma_row["trace_entry_offset_begin_cpu_cycles"]
    )
    assert overlap_ct_row["activity_window_status"] == "Bounded"
    assert overlap_wdma_row["activity_window_status"] == "Bounded"
    assert overlap_ct_row["display_interval_role"] == "command-submit"
    assert overlap_wdma_row["display_interval_role"] == "command-submit"

    report = module.render_report(evidence, first)
    for text in (
        "最终编译产物板卡 Profile",
        "Kernel launch → completion",
        "NCC engine active work / Tile",
        "Host submit",
        "Host launch → trusted completion",
        "Timing domains · not additive",
        "Trace-run Kcore ledger",
        "Trace-run cost overlay",
        "capture-boundary residual",
        "inside-site-outside-operation",
        "between-site-gap",
        "Trace Timeline",
        "Tile / Engine",
        "Program / Sites",
        "Communication / DTE",
        "术语说明",
        "整次通信等待（总计）",
        "通信内部步骤",
        "等待 DTE 对端就绪",
        "配置并发起 DTE",
        "等待 DTE 传输完成",
        "收尾",
        "Operation CPU cycles",
        "仅 issue / wait 总计行提供",
        "Diagnostics / Raw",
        "Resource tree",
        "NCC engine active time · Trace PMU",
        "Σ Tile Work",
        "separate-run proxy",
        "not wall time",
        "Per-tile engine active time (Trace PMU ns)",
        "DIRECT_DTE",
        "Measured",
        "Sampled",
        "Bounded",
        "Unavailable",
        "Incomplete",
        "Invalid",
        "Zero delta",
        "Attribution ambiguous",
        "tile-local",
        "index.html · analysis.json · evidence.json",
        "逐次展示",
        "NCC 指令调用",
        "Direct-DTE 调用",
        "不折叠",
    ):
        assert text in report
    assert "Hardware cost reference · static model" in report
    assert "它不是实测值，不是Primary分项" in report
    assert "single-link payload serialization reference" in report
    assert "hardwareCostRows" in report
    assert "Device execution" not in report
    assert "Engine summary" not in report
    assert "Derived sum" not in report
    assert "Direct-DTE没有calibrated engine ns" in report
    assert "只知道活动发生在保守观测窗内" in report
    assert "实心块</b>：精确 command submit / DTE operation 区间" in report
    assert "浅色虚线框</b>：PMU 活动保守观测范围" in report
    assert "范围重叠不证明 engine 同时执行" in report
    assert 'data-interval-role="observation-bound"' in report
    assert 'data-interval-role="${escapeHtml(role)}"' in report
    for raw_key, display_name in (
        ("site-control", "调用点内控制与准备"),
        ("between-site-gap", "调用点之间的控制/等待"),
        ("ncc-submit", "NCC 指令提交"),
        ("completion-wait-proxy", "等待 NCC 完成（Trace 代理）"),
        ("entry-prologue", "入口准备区间"),
        ("entry-epilogue", "结束收尾区间"),
        ("ncc-pmu-sample", "NCC PMU 采样"),
        ("dte-pmu-sample", "DTE PMU 采样"),
        ("event-bookkeeping", "Trace 事件记录"),
        ("status-poll", "完成状态轮询"),
        ("site-hook", "调用点钩子"),
        ("completion-loop-bookkeeping", "完成循环记录"),
        ("entry-setup", "Trace 入口初始化"),
        ("entry-teardown", "Trace 退出收尾"),
        ("Bounded", "仅确定活动范围"),
        ("trace-only", "仅插桩值"),
    ):
        assert f'"{raw_key}":{{label:"{display_name}"' in report
    catalog_keys = {
        "semantic": {
            row["category"]
            for tile in final["tiles"]
            for row in tile["semantic_partition"]["rows"]
        }
        | {
            "capture-boundary-residual",
            "dte-wait-aggregate-fallback",
        },
        "trace": {
            row["reason"]
            for tile in final["tiles"]
            for row in tile["trace_overhead_overlay"]["rows"]
        },
        "status": {
            "Measured",
            "Sampled",
            "Bounded",
            "Derived",
            "Zero delta",
            "Zero-delta marker",
            "Ambiguous",
            "Attribution ambiguous",
            "Unavailable",
            "Incomplete",
            "Invalid",
            "Passed",
            "Gate unmet",
            "Not assessed",
        },
        "accounting": {
            "yes",
            "mixed",
            "unknown",
            "no",
        },
        "magnitude": {
            "proxy",
            "mixed/proxy",
            "unknown",
            "trace-only",
        },
        "interval": {
            "command-submit",
            "operation-window",
            "observation-bound",
            "site-envelope",
            "marker",
        },
        "location": {
            "inside-entry-unpositioned",
            "before-entry",
            "after-entry",
        },
        "event": set(module.EVENT_KINDS),
        "site": set(module.SITE_KINDS),
        "engine": set(module.ENGINES),
        "role": {"send", "receive"},
        "evidence": {"aggregate", "leaf"},
        "reason": {
            "overlapping-operation-spans",
            "overlapping-site-spans",
            "no-valid-site-boundaries",
            "operation@site/event",
            "inside-site-outside-operation",
            "before-first-site",
            "after-last-site",
            "between-sites",
        },
        "correctness": {
            "primary-validation-failed",
            "expected-exact",
            "expected-relaxed-f16",
            "partially-expected",
            "primary-validated",
        },
        "structure": {
            "trace-run-cost-overlay",
            "semantic-partition",
            "trace-overhead-overlay",
        },
        "validity": set(first["validity"]),
    }
    meta_body = report.split("const TERM_META={", 1)[1].split("\n};", 1)[0]
    guidance_body = report.split("const TERM_GUIDANCE={", 1)[1].split(
        "\n};", 1
    )[0]
    for group, keys in catalog_keys.items():
        group_body = report.split(f"  {group}:{{", 1)[1].split(
            "\n  },", 1
        )[0]
        missing = []
        incomplete = []
        for key in keys:
            prefix = f'"{key}":{{label:'
            if prefix not in group_body:
                missing.append(key)
                continue
            entry = re.search(
                rf'"{re.escape(key)}":\{{label:"[^"]+",'
                rf'definition:"[^"]+"(?:,meta:"[^"]+")?,'
                rf'not:"[^"]+"\}}',
                group_body,
            )
            if entry is None:
                incomplete.append(key)
        assert not missing, (group, sorted(missing))
        assert not incomplete, (group, sorted(incomplete))
        assert f"  {group}:" in meta_body
        assert f"  {group}:" in guidance_body
    assert "Measurement invalid" not in report
    assert "Trace-only overhead" not in report
    assert "Control span" not in report
    assert "Not applicable" not in report
    assert "中位数" not in report
    assert "范围 " not in report
    assert "gradient" not in report.lower()
    assert "grid-template-columns:repeat(4" in report
    assert ".ruler{display:grid;grid-template-columns:128px 1fr" in report
    assert ".cost-tables>*{min-width:0}" in report
    assert (
        ".diag{grid-template-columns:64px minmax(0,1fr)"
        in report
    )
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
        "glossary",
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
        "hardwareCostRows",
        "timelineTile",
        "engineFilters",
        "timelineEventNote",
        "timelineZoom",
        "timelineFit",
        "timelineRuler",
        "timelineLanes",
        "eventDetail",
        "semanticCostRows",
        "traceCostRows",
        "traceOverheadBar",
        "traceOutsideBar",
        "engineTile",
        "engineRows",
        "siteSearch",
        "siteEngine",
        "siteRows",
        "communicationRows",
        "timelineOverlapNotice",
        "glossaryRows",
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
        "data-interval-role",
        "observation-bound",
        "node.dataset.eventSequence",
        "data-event-tile",
        "DTE_PHASES",
        "operation_window_cpu_cycles",
        'addEventListener("keydown"',
        "data-tree-tile",
        "window.__waferProfileUI",
        "focusEvent",
        "renderTimeline",
        "renderEventDetail",
        "renderCostDetail",
        "renderSites",
        "renderCommunication",
        "renderGlossary",
        "crossEngineBoundOverlaps",
        "termCell",
    ):
        assert interaction in script
    assert (
        'event.display_interval_role==="command-submit"'
        in script
    )
    assert "site_instance_sequence" in script
    assert "operation_trace_entry_offset_begin_cpu_cycles" in script
    assert "operation_trace_entry_offset_end_cpu_cycles" in script
    for removed in (
        "commandSubmitGroups",
        "timelineDensity",
        "data-command-group-count",
        "TIMELINE_AUTO_SUBMIT_LIMIT",
        "timelineFullTrace",
        "state.fullTrace",
    ):
        assert removed not in script
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
    issue_event = next(
        row
        for row in evidence["experiment"]["trace"]["tiles"][0]["events"]
        if row["kind"] == "direct-dte-issue"
    )
    event = next(
        row
        for row in evidence["experiment"]["trace"]["tiles"][0]["events"]
        if row["kind"] == "direct-dte-wait"
    )
    analysis = module.analyze_evidence(evidence)
    direct = next(
        row
        for row in analysis["program"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert direct["wait_window_cpu_cycles"] == (
        event["operation_end_cycle"] - event["operation_begin_cycle"]
    )
    assert direct["issue_window_cpu_cycles"] == (
        issue_event["operation_end_cycle"] - issue_event["operation_begin_cycle"]
    )
    assert direct["measurement_kind"] == (
        "direct-dte-issue-and-wait-windows"
    )
    assert direct["raw_pmu_activity"] is None
    assert not direct["raw_pmu_activity_valid"]
    assert direct["wait_window_status"] == "Measured"
    assert direct["raw_pmu_activity_status"] == "Unavailable"
    assert direct["activity_window_count"] == 2
    assert direct["issue_window_count"] == 1
    assert direct["wait_window_count"] == 1
    timeline = analysis["program"]["timeline_events"]
    assert any(
        row["engine"] == "DIRECT_DTE" and row["dte_role"] == "send"
        for row in timeline
    )

    raw_available = make_evidence()
    raw_event = next(
        row
        for row in raw_available["experiment"]["trace"]["tiles"][0]["events"]
        if row["kind"] == "direct-dte-wait"
    )
    raw_event["counter_delta"] = 17
    raw_event["dte_counter_valid"] = True
    raw_event["positive_delta"] = True
    raw_issue_event = next(
        row
        for row in raw_available["experiment"]["trace"]["tiles"][0]["events"]
        if row["kind"] == "direct-dte-issue"
    )
    raw_issue_event["counter_delta"] = 5
    raw_issue_event["dte_counter_valid"] = True
    raw_issue_event["positive_delta"] = True
    raw_analysis = module.analyze_evidence(raw_available)
    raw_direct = next(
        row
        for row in raw_analysis["program"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert raw_direct["wait_windows_valid"]
    assert raw_direct["wait_window_cpu_cycles"] == (
        raw_event["operation_end_cycle"]
        - raw_event["operation_begin_cycle"]
    )
    assert raw_direct["raw_pmu_activity"] == 22
    assert raw_direct["raw_pmu_activity_valid"]
    assert raw_direct["raw_pmu_activity_status"] == "Sampled"
    raw_timeline = raw_analysis["program"]["timeline_events"]
    raw_timeline_event = next(
        row
        for row in raw_timeline
        if row["tile"] == 0
        and row["kind"] == "direct-dte-wait"
    )
    assert raw_timeline_event["counter_status"] == "Sampled"


def _test_cost_attribution(module: object) -> None:
    base = module.analyze_evidence(make_evidence())
    base_partition = base["program"]["tiles"][0]["semantic_partition"]
    categories = {row["category"] for row in base_partition["rows"]}
    assert {
        "dte-peer-ready-wait",
        "dte-setup-issue",
        "dte-completion-wait",
        "dte-cleanup",
    }.issubset(categories)
    assert "dte-wait-aggregate-fallback" not in categories
    assert not any(
        row["reason"] in (
            "overlapping-site-spans",
            "overlapping-operation-spans",
        )
        for row in base_partition["rows"]
    )
    explicit = [
        row
        for row in base_partition["rows"]
        if row["event_kind"] is not None
    ]
    assert explicit
    assert all(
        row["counts_in_primary_device_elapsed"] == "yes"
        and row["magnitude_relation"] == "proxy"
        for row in explicit
    )
    mixed_control = [
        row
        for row in base_partition["rows"]
        if row["category"] in (
            "site-control",
            "between-site-gap",
            "entry-prologue",
            "entry-epilogue",
        )
    ]
    assert mixed_control
    assert all(
        row["counts_in_primary_device_elapsed"] == "mixed"
        and row["magnitude_relation"] == "mixed/proxy"
        and row["reason"]
        and row["optimization_entry"]
        for row in mixed_control
    )
    assert all(
        row["reason"].startswith("inside-site-outside-operation@")
        and row["containing_site"] is not None
        for row in mixed_control
        if row["category"] == "site-control"
    )
    assert all(
        row["reason"].startswith("between-site-")
        and row["previous_site"] is not None
        and row["next_site"] is not None
        for row in mixed_control
        if row["category"] == "between-site-gap"
    )
    assert any(
        row["category"] == "entry-prologue"
        and row["next_site"] is not None
        for row in mixed_control
    )
    assert any(
        row["category"] == "entry-epilogue"
        and row["previous_site"] is not None
        for row in mixed_control
    )

    nested = make_evidence()
    events = nested["experiment"]["trace"]["tiles"][0]["events"]
    commands = [event for event in events if event["kind"] == "ncc-command"]
    first, second = commands[:2]
    for event in events:
        if event["site_id"] == second["site_id"]:
            event["site_begin_cycle"] = first["site_begin_cycle"]
            event["site_end_cycle"] = first["site_end_cycle"]
    second["operation_begin_cycle"] = first["operation_begin_cycle"] + 2
    second["operation_end_cycle"] = first["operation_end_cycle"] - 2
    analysis = module.analyze_evidence(nested)
    tile = analysis["program"]["tiles"][0]
    partition = tile["semantic_partition"]
    assert partition["exclusive_accounting_valid"]
    assert partition["exclusive_cycles"] == partition["entry_cycles"]
    assert sum(row["cycles"] for row in partition["rows"]) == partition[
        "entry_cycles"
    ]
    overlap = next(
        row
        for row in partition["rows"]
        if row["reason"] == "overlapping-operation-spans"
    )
    assert overlap["category"] == "capture-boundary-residual"
    assert overlap["counts_in_primary_device_elapsed"] == "unknown"
    assert overlap["magnitude_relation"] == "unknown"

    observations = make_evidence()
    observation_events = observations["experiment"]["trace"]["tiles"][0][
        "events"
    ]
    commands = [
        event for event in observation_events if event["kind"] == "ncc-command"
    ]
    zero, ambiguous = commands[:2]
    zero["counter_delta"] = 0
    zero["positive_delta"] = False
    zero["observation_status"] = "counter-no-change"
    ambiguous["attribution_ambiguous"] = True
    ambiguous["observation_status"] = "attribution-ambiguous"
    result = module.analyze_evidence(observations)
    zero_row = next(
        row
        for row in result["program"]["timeline_events"]
        if row["tile"] == 0 and row["sequence"] == zero["sequence"]
    )
    ambiguous_row = next(
        row
        for row in result["program"]["timeline_events"]
        if row["tile"] == 0 and row["sequence"] == ambiguous["sequence"]
    )
    assert zero_row["zero_delta_marker"]
    assert zero_row["marker"]
    assert zero_row["ncc_engine_execution_time_ns"] is None
    assert zero_row["counter_status"] == "Zero delta"
    assert ambiguous_row["attribution_ambiguous"]
    assert ambiguous_row["marker"]
    assert ambiguous_row["ncc_engine_execution_time_ns"] is None
    assert ambiguous_row["duration_status"] == "Attribution ambiguous"

    unavailable = make_evidence()
    unavailable_event = next(
        event
        for event in unavailable["experiment"]["trace"]["tiles"][0]["events"]
        if event["kind"] == "ncc-command"
    )
    unavailable_event["observed_begin_cycle"] = 0
    unavailable_event["observed_end_cycle"] = 0
    unavailable_event["counter_delta"] = 0
    unavailable_event["observation_count"] = 0
    unavailable_event["observed_span_valid"] = False
    unavailable_event["positive_delta"] = False
    unavailable_event["ncc_counter_valid"] = False
    unavailable_event["observation_status"] = "counter-unavailable"
    unavailable_analysis = module.analyze_evidence(unavailable)
    unavailable_row = next(
        row
        for row in unavailable_analysis["program"]["timeline_events"]
        if row["tile"] == 0
        and row["sequence"] == unavailable_event["sequence"]
    )
    assert unavailable_analysis["validity"]["trace"]
    assert unavailable_row["counter_status"] == "Unavailable"
    assert unavailable_row["duration_status"] == "Unavailable"
    assert unavailable_row["operation_window_status"] == "Measured"
    assert unavailable_row["activity_window_status"] == "Unavailable"
    assert unavailable_row["marker"]
    assert not unavailable_row["zero_delta_marker"]
    assert any(
        row["code"] == "ncc_event_counter_unavailable"
        for row in unavailable_analysis["diagnostics"]
    )
    unavailable_site = next(
        row
        for row in unavailable_analysis["program"]["sites"]
        if row["tile"] == 0 and row["site_id"] == unavailable_event["site_id"]
    )
    assert unavailable_site["site_capture_status"] == "Measured"
    assert unavailable_site["engine_observation_status"] == "Unavailable"

    invalid_site_capture = make_evidence()
    target_site = next(
        event
        for event in invalid_site_capture["experiment"]["trace"]["tiles"][0][
            "events"
        ]
        if event["kind"] == "target-site"
    )
    invalid_site_begin = (
        invalid_site_capture["experiment"]["trace"]["tiles"][0][
            "entry_begin_cycle"
        ]
        - 1
    )
    for event in invalid_site_capture["experiment"]["trace"]["tiles"][0]["events"]:
        if event["site_id"] == target_site["site_id"]:
            event["site_begin_cycle"] = invalid_site_begin
    invalid_site_analysis = module.analyze_evidence(invalid_site_capture)
    invalid_site = next(
        row
        for row in invalid_site_analysis["program"]["sites"]
        if row["tile"] == 0 and row["site_id"] == target_site["site_id"]
    )
    assert invalid_site["site_capture_status"] == "Invalid"

    unit_isolation = make_evidence()
    baseline = module.analyze_evidence(unit_isolation)
    ct_counter = unit_isolation["experiment"]["pmu"]["tiles"][0]["aggregates"][
        "ct"
    ]
    ct_counter["end"] += 1_000_000
    ct_counter["recovery"] += 1_000_000
    changed = module.analyze_evidence(unit_isolation)
    assert (
        changed["program"]["duration"]["device_elapsed_ns"]
        == baseline["program"]["duration"]["device_elapsed_ns"]
    )
    assert (
        changed["program"]["tiles"][0]["semantic_partition"]
        == baseline["program"]["tiles"][0]["semantic_partition"]
    )
    assert (
        changed["program"]["tiles"][0]["engines"][0][
            "engine_execution_time_ns"
        ]
        > baseline["program"]["tiles"][0]["engines"][0][
            "engine_execution_time_ns"
        ]
    )

def _test_validity(module: object) -> None:
    coarse = make_evidence()
    sample = coarse["measurement"]["samples"][0]
    sample["completion_observation_resolution_ns"] = (
        sample["host_launch_to_completion_ns"] // 2
    )
    analysis = module.analyze_evidence(coarse)
    duration = analysis["program"]["duration"]
    assert duration["qualified"]
    assert not duration["host_completion_high_resolution"]
    assert duration["status"] == "Measured"
    warning = next(
        row
        for row in analysis["diagnostics"]
        if row["code"] == "completion_resolution_too_coarse"
    )
    assert warning["severity"] == "warning"
    report = module.render_report(coarse, analysis)
    assert "host completion observation is coarse" in report

    quantized = make_evidence()
    quantized["measurement"]["samples"][0]["device_elapsed_ns"] = 0
    analysis = module.analyze_evidence(quantized)
    duration = analysis["program"]["duration"]
    assert duration["qualified"]
    assert duration["status"] == "Measured"
    assert duration["device_elapsed_quantized_zero"]
    warning = next(
        row
        for row in analysis["diagnostics"]
        if row["code"] == "device_event_elapsed_quantized_zero"
    )
    assert warning["severity"] == "warning"
    assert "at/below effective timer resolution" in module.render_report(
        quantized, analysis
    )

    zero_submit = make_evidence()
    zero_submit["measurement"]["samples"][0]["host_submit_ns"] = 0
    analysis = module.analyze_evidence(zero_submit)
    duration = analysis["program"]["duration"]
    assert duration["qualified"]
    assert duration["status"] == "Measured"
    assert duration["host_submit_quantized_zero"]
    assert any(
        row["code"] == "host_submit_quantized_zero"
        for row in analysis["diagnostics"]
    )
    assert "at/below host clock resolution" in module.render_report(
        zero_submit, analysis
    )

    zero_host_envelope = make_evidence()
    sample = zero_host_envelope["measurement"]["samples"][0]
    sample["host_submit_ns"] = 0
    sample["host_launch_to_completion_ns"] = 0
    sample["completion_observation_resolution_ns"] = 0
    analysis = module.analyze_evidence(zero_host_envelope)
    duration = analysis["program"]["duration"]
    assert duration["qualified"]
    assert duration["status"] == "Measured"
    assert duration["device_elapsed_ns"] == 8_400
    assert not duration["host_envelope_available"]
    assert duration["host_non_submit_envelope_ns"] == 0
    assert duration["completion_observation_fraction"] is None
    assert not analysis["validity"]["host_completion_resolution"]
    assert any(
        row["code"] == "host_completion_envelope_unavailable"
        for row in analysis["diagnostics"]
    )

    mismatch = make_evidence()
    mismatch["output_validation"]["resources"][0][
        "diagnostic_captures_match_primary"
    ] = False
    analysis = module.analyze_evidence(mismatch)
    assert not analysis["program"]["duration"]["qualified"]
    assert analysis["program"]["duration"]["status"] == "Invalid"
    assert not analysis["validity"]["output_equivalence"]

    primary_failure = make_evidence()
    primary_failure["output_validation"]["resources"][0][
        "primary_output_validated"
    ] = False
    analysis = module.analyze_evidence(primary_failure)
    assert analysis["validity"]["semantic_correctness"] is False
    assert (
        analysis["program"]["output"]["correctness_status"]
        == "primary-validation-failed"
    )
    assert not analysis["program"]["duration"]["qualified"]

    unassessed = make_evidence()
    unassessed["output_validation"]["mode"] = "same-session-primary"
    for resource in unassessed["output_validation"]["resources"]:
        resource["external_expected_comparison"] = None
    analysis = module.analyze_evidence(unassessed)
    assert analysis["validity"]["semantic_correctness"] is None
    assert (
        analysis["program"]["output"]["correctness_status"]
        == "primary-validated"
    )
    unassessed_report = module.render_report(unassessed, analysis)
    assert '"semantic_correctness":null' in unassessed_report
    assert (
        'value===true?"Passed":value===false?"Gate unmet":"Not assessed"'
        in unassessed_report
    )

    backwards = make_evidence()
    counter = backwards["experiment"]["pmu"]["tiles"][3]["aggregates"]["ne"]
    counter["end"] = counter["start"] - 1
    counter["recovery"] = counter["end"]
    analysis = module.analyze_evidence(backwards)
    ne = next(
        row
        for row in analysis["program"]["tiles"][3]["engines"]
        if row["engine"] == "NE"
    )
    assert not ne["engine_execution_time_valid"]
    assert ne["engine_execution_time_ns"] is None
    assert ne["engine_execution_time_status"] == "Unavailable"
    ne_summary = next(
        row
        for row in analysis["program"]["engine_active_time"]["by_engine"]
        if row["engine"] == "NE"
    )
    assert ne_summary["available_tile_count"] == 15
    assert ne_summary["active_tile_count"] == 15
    assert ne_summary["status"] == "Incomplete"
    tile_work = analysis["program"]["engine_active_time"][
        "per_tile_work_volume"
    ]
    assert tile_work["available_tile_count"] == 15
    assert tile_work["status"] == "Incomplete"

    zero_active = make_evidence()
    counter = zero_active["experiment"]["pmu"]["tiles"][0]["aggregates"]["ct"]
    counter["end"] = counter["start"]
    counter["recovery"] = counter["end"]
    analysis = module.analyze_evidence(zero_active)
    ct_summary = next(
        row
        for row in analysis["program"]["engine_active_time"]["by_engine"]
        if row["engine"] == "CT"
    )
    assert ct_summary["available_tile_count"] == 16
    assert ct_summary["active_tile_count"] == 15
    assert ct_summary["minimum_per_tile_ns"] == 0
    assert ct_summary["status"] == "Measured"

    unrecovered = make_evidence()
    counter = unrecovered["experiment"]["pmu"]["tiles"][2]["aggregates"]["ct"]
    counter["recovery"] = counter["end"] + 1
    analysis = module.analyze_evidence(unrecovered)
    ct = next(
        row
        for row in analysis["program"]["tiles"][2]["engines"]
        if row["engine"] == "CT"
    )
    assert not ct["engine_execution_time_valid"]
    assert any(
        row["code"] == "ncc_engine_execution_time_unusable"
        and row.get("tile") == 2
        and "terminal end" in row["message"]
        for row in analysis["diagnostics"]
    )

    bad_statistics_window = make_evidence()
    statistics = bad_statistics_window["experiment"]["pmu"]["tiles"][5][
        "aggregates"
    ]["statistics_window"]
    statistics["end"] = statistics["start"] - 1
    statistics["recovery"] = statistics["end"]
    analysis = module.analyze_evidence(bad_statistics_window)
    tile = analysis["program"]["tiles"][5]
    assert tile["statistics_window_raw_ticks"] is None
    assert not tile["statistics_window_valid"]
    assert tile["statistics_window_status"] == "Unavailable"
    assert analysis["validity"]["pmu"]
    assert any(
        row["code"] == "statistics_window_unusable" and row.get("tile") == 5
        for row in analysis["diagnostics"]
    )

    empty = make_evidence()
    empty["sites"] = []
    for tile in empty["experiment"]["trace"]["tiles"]:
        tile["capacity"] = 0
        tile["count"] = 0
        tile["counted_event_count"] = 0
        tile["next_sequence"] = 0
        tile["events"] = []
    analysis = module.analyze_evidence(empty)
    assert analysis["validity"]["trace"]
    assert analysis["program"]["timeline_events"] == []
    assert all(
        site["observation_status"] == "Unavailable"
        for site in analysis["program"]["sites"]
    )

    reversed_trace = make_evidence()
    tile = reversed_trace["experiment"]["trace"]["tiles"][4]
    tile["entry_end_cycle"] = tile["entry_begin_cycle"] - 1
    analysis = module.analyze_evidence(reversed_trace)
    assert not analysis["validity"]["trace"]
    assert (
        analysis["program"]["tiles"][4]["trace_entry_cpu_cycles"]
        is None
    )
    assert analysis["program"]["tiles"][4]["trace_status"] == "Invalid"
    assert not any(
        event["tile"] == 4
        for event in analysis["program"]["timeline_events"]
    )


def _test_rejections(module: object) -> None:
    inconsistent_binding = make_evidence(permute_bindings=True)
    inconsistent_binding["experiment"]["clock"][0]["launch_slot"] = 0
    inconsistent_binding["experiment"]["clock"][1]["launch_slot"] = 1
    _must_reject(
        module,
        inconsistent_binding,
        "explicit topology Tile binding",
    )

    wrong_static_scope = make_evidence()
    wrong_static_scope["static_cost_model"]["scope"] = "some-program"
    _must_reject(
        module,
        wrong_static_scope,
        "complete-final-instruction-program-per-physical-tile",
    )

    invalid_overflow_reason = make_evidence()
    overflow_metric = invalid_overflow_reason["static_cost_model"]["tiles"][0][
        "work"
    ]["ddr_read_bytes"]
    overflow_metric.update(
        {
            "knowledge": "overflow",
            "value": None,
            "reason": "unavailable-resource-bytes",
        }
    )
    _must_reject(module, invalid_overflow_reason, "overflow metrics")

    retired_unknown_cost = make_evidence()
    retired_metric = retired_unknown_cost["static_cost_model"]["tiles"][0][
        "work"
    ]["directional_noc_transmit_bytes"]["north"]
    retired_metric["knowledge"] = "unknown"
    _must_reject(module, retired_unknown_cost, "knowledge")

    duplicate = make_evidence()
    duplicate["measurement"]["samples"].append(
        copy.deepcopy(duplicate["measurement"]["samples"][0])
    )
    _must_reject(module, duplicate, "exactly one primary execution")

    wrong_primary = make_evidence()
    wrong_primary["measurement"]["samples"][0]["sample_id"] = "final-s0"
    _must_reject(module, wrong_primary, "must be 'primary'")

    missing = make_evidence()
    del missing["measurement"]["samples"][0]["device_elapsed_ns"]
    _must_reject(module, missing, "missing keys")

    wrong_timer = make_evidence()
    wrong_timer["measurement"]["samples"][0]["device_timer_kind"] = "host-clock"
    _must_reject(module, wrong_timer, "tx-stream-events")

    reversed_host_timing = make_evidence()
    reversed_host_timing["measurement"]["samples"][0]["host_submit_ns"] = (
        reversed_host_timing["measurement"]["samples"][0][
            "host_launch_to_completion_ns"
        ]
        + 1
    )
    _must_reject(module, reversed_host_timing, "must not exceed")

    malformed_digest = make_evidence()
    malformed_digest["program"]["program_manifest_sha256"] = "sha256:no"
    _must_reject(module, malformed_digest, "64 lowercase hex digits")

    legacy_site = make_evidence()
    legacy_site["sites"][0]["candidate"] = "winner"
    _must_reject(module, legacy_site, "unknown keys")

    legacy_output = make_evidence()
    resource = legacy_output["output_validation"]["resources"][0]
    resource["legacy_output_flag"] = resource.pop(
        "primary_output_validated"
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

    invalid_dte_observation_status = make_evidence()
    dte_event = next(
        row
        for row in invalid_dte_observation_status["experiment"]["trace"][
            "tiles"
        ][0]["events"]
        if row["kind"] == "direct-dte-wait"
    )
    dte_event["observation_status"] = "counter-no-change"
    _must_reject(
        module,
        invalid_dte_observation_status,
        "must be null for a Kcore/DTE phase",
    )

    missing_site_container = make_evidence()
    trace_tile = missing_site_container["experiment"]["trace"]["tiles"][0]
    trace_tile["events"].pop(0)
    for sequence, event in enumerate(trace_tile["events"]):
        event["sequence"] = sequence
    trace_tile["count"] = len(trace_tile["events"])
    trace_tile["counted_event_count"] = len(trace_tile["events"])
    trace_tile["next_sequence"] = len(trace_tile["events"])
    _must_reject(
        module,
        missing_site_container,
        "must follow its target-site container",
    )

    negative_window = make_evidence()
    event = next(
        row
        for row in negative_window["experiment"]["trace"]["tiles"][0][
            "events"
        ]
        if row["kind"] == "ncc-command"
    )
    event["observed_end_cycle"] = event["observed_begin_cycle"] - 1
    _must_reject(module, negative_window, "must not precede")

    reversed_dte_window = make_evidence()
    dte_event = next(
        row
        for row in reversed_dte_window["experiment"]["trace"]["tiles"][0][
            "events"
        ]
        if row["kind"] == "direct-dte-wait"
    )
    dte_event["operation_span_valid"] = False
    dte_event["operation_end_cycle"] = dte_event["operation_begin_cycle"] - 1
    dte_analysis = module.analyze_evidence(reversed_dte_window)
    dte_row = next(
        row
        for row in dte_analysis["program"]["tiles"][0]["engines"]
        if row["engine"] == "DIRECT_DTE"
    )
    assert not dte_row["wait_windows_valid"]
    assert dte_row["wait_window_cpu_cycles"] is None
    assert dte_row["wait_window_count"] == 1
    assert dte_row["issue_windows_valid"]
    assert dte_row["activity_window_count"] == 1


def _test_report_files(repo: pathlib.Path, module: object) -> None:
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
        assert analysis["program"]["duration"]["qualified"]

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
    assert "experiment" in schema["required"]
    assert "static_cost_model" in schema["required"]
    assert "experiments" not in schema["required"]
    assert "candidate" not in schema["$defs"]["site"]["properties"]
    assert "entryTiming" not in schema["$defs"]
    assert set(schema["$defs"]["experiment"]["required"]) == {
        "clock",
        "trace",
        "pmu",
    }
    assert "summary" not in schema["$defs"]["experiment"]["properties"]
    sample = schema["$defs"]["measurementSample"]["properties"]
    assert sample["sample_id"]["const"] == "primary"
    assert sample["sample_index"]["const"] == 0
    assert sample["device_elapsed_ns"]["minimum"] == 0
    assert sample["device_timer_kind"]["const"] == "tx-stream-events"
    assert sample["host_submit_ns"]["minimum"] == 0
    output_resource = schema["$defs"]["outputValidationResource"]
    assert "primary_output_validated" in output_resource["required"]
    assert "diagnostic_captures_match_primary" in output_resource["required"]
    assert "legacy_output_flag" not in output_resource["properties"]
    assert "recovery" in schema["$defs"]["counterSnapshot"]["required"]
    assert "recovery" in schema["$defs"]["counterSnapshot32"]["required"]
    assert schema["$defs"]["event"]["allOf"]
    assert "cost_summary" in schema["$defs"]["traceTile"]["required"]
    assert "site_kind" in schema["$defs"]["site"]["required"]
    static_model = schema["$defs"]["staticCostModel"]
    assert (
        static_model["properties"]["model"]["const"]
        == "tx81-static-throughput-lower-bound"
    )
    assert static_model["properties"]["scope"]["const"] == (
        "complete-final-instruction-program-per-physical-tile"
    )
    assert static_model["properties"]["tiles"]["minItems"] == 16
    static_metric = schema["$defs"]["staticCostMetric"]
    assert set(static_metric["required"]) == {"knowledge", "value", "reason"}
    assert set(static_metric["properties"]["knowledge"]["enum"]) == {
        "known",
        "unavailable",
        "unsupported",
        "overflow",
    }
    assert (
        schema["$defs"]["programMetadata"]["properties"]["record_abi"]["const"]
        == "wafer-tx81-profiler-record"
    )
    assert (
        schema["$defs"]["programMetadata"]["properties"][
            "program_manifest_sha256"
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
    _test_program(module)
    _test_direct_dte(module)
    _test_cost_attribution(module)
    _test_validity(module)
    _test_rejections(module)
    _test_report_files(repo, module)
    print("wafer_profile_report_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
