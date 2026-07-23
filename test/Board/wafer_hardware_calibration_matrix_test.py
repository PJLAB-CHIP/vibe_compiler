#!/usr/bin/env python3
"""Validate complete, non-duplicated calibration preparation accounting."""

from __future__ import annotations

import pathlib
import re

import wafer_hardware_calibration_matrix as matrix


EXPECTED_KEYS = {
    "profile-qualification",
    "constructor-ownership",
    "execute-result",
    "packet-routing-range",
    "ct-numeric-form",
    "instruction-physical-layout",
    "datamove-layout",
    "ne-numeric-layout",
    "rdma-wdma-descriptor",
    "tdma-memset",
    "tdma-bool-fill",
    "tdma-movement-variants",
    "spm-capacity-reservation",
    "spm-alignment-bank",
    "ddr-cache-coherence",
    "queue-shape-submission",
    "worker-scope",
    "cross-engine-overlap",
    "address-dependency",
    "issue-overhead",
    "local-completion",
    "cross-worker-join",
    "direct-dte",
    "multi-tile-arrival",
    "host-launch-runtime",
    "ncc-pmu-basis",
    "transport-pmu-basis",
    "scalar-csr-ordinary-issue",
}


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    domains = matrix.CALIBRATION_DOMAINS
    assert len(domains) == len(EXPECTED_KEYS)
    assert set(matrix.DOMAINS_BY_KEY) == EXPECTED_KEYS
    assert len(matrix.DOMAINS_BY_KEY) == len(domains)

    labels = {domain.document_label for domain in domains}
    assert len(labels) == len(domains)
    calibration = (
        repo / "docs" / "tx81-compiler-hardware-calibration.md"
    ).read_text()
    matrix_body = calibration[
        calibration.index("## 3. Compiler-sensitive calibration matrix") :
        calibration.index("## 4. 当前profile已经闭合的事实")
    ]
    document_rows = {
        match.group(1)
        for match in re.finditer(r"^\| ([^|]+?) \|", matrix_body, re.M)
        if match.group(1) not in {"域", "---"}
    }
    assert labels == document_rows

    cmake = (repo / "test" / "CMakeLists.txt").read_text()
    allowed_states = {"ready", "in-progress"}
    for domain in domains:
        assert domain.preparation in allowed_states
        assert domain.execution_scope
        assets = domain.positive_assets + domain.negative_assets
        if domain.preparation == "ready":
            assert assets
            assert not domain.remaining_preparation
        else:
            assert domain.remaining_preparation
        for relative in assets:
            path = repo / relative
            assert path.is_file(), f"{domain.key}: missing asset {relative}"
        for test_name in domain.no_card_tests:
            generated_launch_test = test_name in {
                "wafer-runtime-kernel-grid-add-no-card",
                "wafer-runtime-model-add-no-card",
            }
            registered = f"NAME {test_name}" in cmake
            if generated_launch_test:
                registered = (
                    "kernel-grid model" in cmake
                    and "wafer-runtime-${_wafer_no_card_launch_test}-add-no-card"
                    in cmake
                )
            assert registered, (
                f"{domain.key}: CTest {test_name} is not registered"
            )

    incomplete = tuple(
        domain.key for domain in domains if domain.preparation != "ready"
    )
    assert incomplete == (
        "ct-numeric-form",
        "instruction-physical-layout",
        "datamove-layout",
        "tdma-movement-variants",
        "spm-capacity-reservation",
        "spm-alignment-bank",
        "ddr-cache-coherence",
        "address-dependency",
        "multi-tile-arrival",
        "transport-pmu-basis",
    )
    print(
        "wafer_hardware_calibration_matrix_test: "
        f"rows={len(domains)} incomplete={len(incomplete)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
