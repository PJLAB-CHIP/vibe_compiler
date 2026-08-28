#!/usr/bin/env python3
"""Validate SPM boundary and relative-offset calibration preparation."""

from __future__ import annotations

import pathlib
import re

import wafer_spm_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 146
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert len(catalog.BOARD_CASES) == 24
    assert len(catalog.STATIC_NEGATIVE_CASES) == 10
    assert len(catalog.DEFERRED_CASES) == 0
    assert len(catalog.DELEGATED_CASES) == 112
    assert {case.domain for case in catalog.CATALOG} == {
        "capacity-reservation",
        "alignment-bank",
        "address-relation",
        "physical-layout",
        "lifetime-reuse",
        "engine-access",
        "bank-engine-pair",
    }
    assert {case.disposition for case in catalog.CATALOG}.issubset(
        catalog.DISPOSITIONS
    )
    for seed, case in enumerate(catalog.BOARD_CASES, start=1):
        assert catalog.ALLOCATABLE_BEGIN <= case.address
        assert case.address < case.end <= catalog.ALLOCATABLE_END
        payload = case.payload(seed)
        assert len(payload) == case.transfer_bytes * case.iterations
        assert payload != bytes(len(payload))
        assert len(set(payload[: min(len(payload), 256)])) > 16
        built = catalog.build_case_payload(case, seed)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert built.expected == case.payload(seed + 1)
        assert case.transfer_bytes * case.iterations <= catalog.SLOT_BYTES
        if case not in catalog.NON_PREFERRED_GEOMETRY_CASES:
            assert case.address % 256 == 0
            assert case.transfer_bytes % 256 == 0
    violations = {
        case.expected_violation for case in catalog.STATIC_NEGATIVE_CASES
    }
    assert {
        "below-allocatable-window",
        "crosses-reserved-window",
        "reserved-window",
        "outside-hardware-window",
        "zero-length",
        "address-overflow",
    }.issubset(violations)
    assert not any(
        case.domain == "alignment-bank"
        for case in catalog.STATIC_NEGATIVE_CASES
    )
    assert {
        case.disposition for case in catalog.NON_PREFERRED_GEOMETRY_CASES
    } == {"BOARD_OBSERVATION"}
    assert {
        (case.address - catalog.SWEEP_BASE, case.transfer_bytes)
        for case in catalog.NON_PREFERRED_GEOMETRY_CASES
    } == {
        (64, catalog.TRANSFER_BYTES),
        (128, catalog.TRANSFER_BYTES),
        (192, catalog.TRANSFER_BYTES),
        (0, 128),
        (0, 384),
    }
    assert {case.address % 256 for case in catalog.ALIGNMENT_CASES} == {0}
    assert any(
        case.address % 1024 == 0 for case in catalog.ALIGNMENT_CASES
    )
    assert any(
        case.address % 1024 != 0 for case in catalog.ALIGNMENT_CASES
    )
    assert set(catalog.ALIGNMENT_OFFSETS) == {
        0,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        65792,
    }
    assert all(
        case.repetitions == catalog.PMU_REPETITIONS
        for case in catalog.ALIGNMENT_CASES + catalog.LARGE_TRANSFER_CASES
    )
    assert all(
        case.repetitions == 1
        for case in catalog.BOARD_CASES
        if case not in catalog.ALIGNMENT_CASES + catalog.LARGE_TRANSFER_CASES
    )
    assert catalog.PMU_REPETITIONS >= 3
    assert {case.effect for case in catalog.ADDRESS_RELATION_CASES} == {
        "RAW",
        "WAR",
        "WAW",
        "RAR",
    }
    assert {case.relation for case in catalog.ADDRESS_RELATION_CASES} == {
        "exact",
        "half-partial",
        "adjacent",
        "far-disjoint",
        "strided",
    }
    assert all(
        case.disposition == "DELEGATED_BOARD_CASE"
        and case.expected_violation is None
        and case.evidence
        for case in catalog.ADDRESS_RELATION_CASES
    )
    assert {case.iterations for case in catalog.LIFETIME_CASES} == {1, 4, 5}
    assert all(case.evidence for case in catalog.PHYSICAL_LAYOUT_CASES)
    assert all(case.evidence for case in catalog.DELEGATED_CASES)
    assert all(
        row in catalog.datamove.CATALOG
        or row in catalog.BOARD_CASES
        or row in catalog.memory_descriptor.CATALOG
        for case in catalog.DELEGATED_CASES
        for row in case.evidence
    )
    assert {case.effect for case in catalog.FIVE_ENGINE_ACCESS_CASES} == {
        "CT",
        "NE",
        "RDMA",
        "WDMA",
        "TDMA",
    }
    assert len(catalog.BANK_ENGINE_PAIR_CASES) == 82
    assert {case.relation for case in catalog.BANK_ENGINE_PAIR_CASES} == {
        "serial",
        "window",
    }
    assert {
        case.address_b - case.address
        for case in catalog.BANK_ENGINE_PAIR_CASES
        if case.address_b is not None
    } == set(catalog.memory_descriptor.SPM_PAIR_RELATIVE_OFFSETS)
    assert all(
        len(case.evidence) == 1
        and case.evidence[0] in catalog.memory_descriptor.PAIR_CASES
        for case in catalog.BANK_ENGINE_PAIR_CASES
    )
    assert catalog.CALIBRATION_LEAF_BINDINGS
    allowed_leaf_objects = set(catalog.CATALOG)
    for key, rows in catalog.CALIBRATION_LEAF_BINDINGS.items():
        assert key
        assert rows
        assert all(row in allowed_leaf_objects for row in rows)

    probe = (
        pathlib.Path(__file__).parent.parent
        / "Inputs"
        / "wafer_spm_calibration_probe.c"
    ).read_text()
    encoded = tuple(
        (
            int(case_id),
            int(address, 16),
            int(transfer_bytes),
            int(kind),
            int(iterations),
            int(slot_stride),
        )
        for case_id, address, transfer_bytes, kind, iterations, slot_stride
        in re.findall(
            r"\{(\d+)U, UINT64_C\(0x([0-9a-f]+)\), (\d+)U, "
            r"(\d+)U, (\d+)U, (\d+)U\}",
            probe,
        )
    )
    assert encoded == tuple(
        (
            catalog.CASE_IDS[case.name],
            case.address,
            case.transfer_bytes,
            1 if case.kind == "lifetime" else 0,
            case.iterations,
            case.slot_stride,
        )
        for case in catalog.BOARD_CASES
    )
    entry = probe[
        probe.index("wafer_tx81_instruction_family_probe(uint64_t request_ddr")
        :
    ]
    assert "get_spm_memory_mapping" not in probe
    assert "wafer_spm_seed_guards" in probe
    assert "wafer_spm_readback_guards" in probe
    first_rdma = entry.index("wafer_spm_seed_guards(")
    first_wdma = entry.index("wafer_tx81_wdma(", first_rdma)
    readback = entry.index("wafer_spm_readback_guards(", first_wdma)
    terminal = entry.index("wafer_tx81_ncc_join(1U);", readback)
    assert first_rdma < first_wdma < readback < terminal
    assert entry.count("wafer_tx81_ncc_join(1U);") == 1
    print(
        "wafer_spm_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"board={len(catalog.BOARD_CASES)} "
        f"negative={len(catalog.STATIC_NEGATIVE_CASES)} "
        f"deferred={len(catalog.DEFERRED_CASES)} "
        f"delegated={len(catalog.DELEGATED_CASES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
