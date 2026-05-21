# TX8 Deps Reverse Engineering Docs

This directory contains the reverse-engineering outputs for the TX8/Wafer
hardware stack.  The original base is `/root/dlc_dev/tx8_deps`; later passes
also incorporate `/root/dlc_dev/firmware_kuiper` and the copied
`torch_txda` wheel where those artifacts clarify host runtime, driver, and
PyTorch eager integration behavior.

## Files

| file | purpose |
| --- | --- |
| [tx8-interface-contract.md](tx8-interface-contract.md) | Canonical implementer-facing contract: instruction APIs, packet fields, runtime/driver APIs, DTE/stream/mailbox/PMU semantics, verifier rules, bootparam/TLV layout, and golden-test targets. |
| [firmware-kuiper-runtime-hardware-analysis.md](firmware-kuiper-runtime-hardware-analysis.md) | Full SDK, HPGR runtime, KMD/UAPI, BO/BAR/ATU, DTE/C2C, PG, completion semantics, system tools, and host/driver evidence from `firmware_kuiper`. |
| [tx8-deps-reverse-engineering-reference.md](tx8-deps-reverse-engineering-reference.md) | Ground-truth reference derived from `tx8_deps` headers, build files, symbols, and disassembly.  It intentionally remains a tx8-deps-only evidence ledger. |
| [tx8-api-struct-contract-annex.md](tx8-api-struct-contract-annex.md) | Generated host/Kcore signatures, enums, wrapper structs, and packet structs. |
| [txda-pytorch-runtime-wheel-analysis.md](txda-pytorch-runtime-wheel-analysis.md) | Static analysis of the copied `torch_txda` PyTorch runtime wheel: `PrivateUse1` backend registration, CUDA compatibility patches, tx_runtime/txdnn dependencies, stream/event semantics, eager op coverage, and implications for Wafer runtime layering. |
| [tx8-symbol-coverage-matrix.md](tx8-symbol-coverage-matrix.md) | Human-readable symbol coverage summary. |
| [tx8-symbol-coverage-matrix.csv](tx8-symbol-coverage-matrix.csv) | Full generated symbol coverage matrix. |

The root-level Wafer documents are the canonical design-facing specs:

- [../wafer-hardware-instruction-set-and-programming-model.md](../wafer-hardware-instruction-set-and-programming-model.md) owns hardware topology, memory/SPM/layout, runtime boundary, DTE, stream/CSR, parallel execution, and remaining hardware-facing gaps.
- [../wafer-register-level-instruction-spec.md](../wafer-register-level-instruction-spec.md) owns backend ABI, register packet fields, wrapper lowering constraints, opcode/type tables, wait/CSR semantics, and PMU/profiling register notes.

Files in this directory should stay evidence- and contract-oriented.  When a
finding becomes compiler/runtime policy, keep the policy in the Wafer docs and
leave only the reverse-engineering proof or API contract here.

Regenerate generated outputs from repo root:

```bash
python3 tools/tx8_symbol_coverage.py
python3 tools/tx8_contract_annex.py
```

## Current Coverage Snapshot

The table below is generated from `tx8-symbol-coverage-matrix.csv` and covers
`tx8_deps` only.  It does not count the additional `firmware_kuiper` KMD/HPGR
or `torch_txda` wheel findings.

| metric | count | note |
| --- | ---: | --- |
| Total named symbols | 4034 | Toolchain archives are not expanded by default. |
| Compiler/runtime relevant symbols | 1146 | All non-`Dependency` rows. |
| Static ABI/implementation covered | 1025 | `ABI` + `Implemented` + `Reserved`; 89.44% of relevant rows. |
| Board/driver verification pending | 121 | `HardwareVerify`; 10.56% of relevant rows. |
| Intentionally excluded dependencies | 2888 | RTThread, board support, and libc shim rows; 71.59% of total. |

By category:

| category | count |
| --- | ---: |
| `ABI` | 257 |
| `Implemented` | 767 |
| `HardwareVerify` | 121 |
| `Reserved` | 1 |
| `Dependency` | 2888 |

By library:

| library | coverage state |
| --- | --- |
| `libinstr_tx81.a` | Fully classified: 42 ABI, 250 implemented, 1 reserved scalar stub. |
| `libcommon_util.a` | Fully classified as TX8 support implementation: 108 symbols. |
| `libtx8_runtime.so` | Fully classified: 42 host ABI exports, 409 host runtime implementation/support symbols. |
| `libtx8_profiling.so` / static | Fully classified as profiling ABI: 41 symbols total. |
| `libkcorert.a` | 132 Kcore ABI symbols covered, 121 hardware-measured symbols pending board validation, 2784 RTOS/board dependencies excluded. |
| `liblibc_stub.a` | Excluded dependency boundary: 104 libc shim symbols. |

Current shared state across the docs:

| area | current status |
| --- | --- |
| Instruction wrappers and register packets | Static ABI is sufficient for a correctness-first compiler/runtime design.  CT/NE/RDMA/WDMA/TDMA packet fields, wrapper units, CSR wait, and `serial_mode` semantics are documented. |
| Host runtime and driver | HPGR `tx_runtime.h`/`libhpgr.so` is the primary CUDA-like host runtime surface in `firmware_kuiper`; VS `Tsm*` remains compatibility evidence with several stubs. |
| Completion semantics | KMD compute fences are not proof of model/kernel completion in this driver snapshot; HPGR command-slot completion, async receive thread, module `completeSignal`, and stream waits carry the runtime completion meaning. |
| Address space and PG | BAR/ATU windows, BO pools, tile SPM/register layout, Kcore/Score firmware slots, global stream mapping table, and 8/16-tile PG behavior are source- or binary-confirmed. |
| DTE | KMD UAPI, Kcore raw registers, direct-DTE helper, and VS D2D/P2P TLV paths are separate layers.  V0 compiler communication should stay on fixed-size unicast Direct DTE unless a Wafer raw-DTE ABI is added and board-tested. |
| Parallel/SPM bank | `serial_mode=0`, per-packet range metadata, and SPM bank-resource hazards are confirmed.  Exact SPM address-to-bank mapping and the 64 KiB page-color rule remain conservative policy, not a proven hard ABI. |

## Remaining Gaps

1. `HardwareVerify` is the main real gap: PMU, MHU/power-off, DTE PMU counters, board services, and hardware state/timing paths need board validation even though the static ABI is known.
2. `__execute_sc` is reserved/stub in the current `libinstr_tx81.a`; production lowering should not target SCALAR.
3. Exact SPM bank mapping and aggressive parallel cost modeling require board sweeps or lower RTL/firmware evidence.
4. API/function semantics, instruction constraints, runtime/driver behavior, and DTE/stream/mailbox/PMU register meanings are consolidated in [tx8-interface-contract.md](tx8-interface-contract.md), with SDK/driver additions in [firmware-kuiper-runtime-hardware-analysis.md](firmware-kuiper-runtime-hardware-analysis.md).
5. Direct DTE V0 should stay on the documented unicast helper path. Raw multi-destination DTE modes need a Wafer-specific ABI and board tests before becoming compiler-facing.
