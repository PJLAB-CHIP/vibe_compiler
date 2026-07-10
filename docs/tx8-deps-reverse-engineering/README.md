# TX8 Deps Reverse Engineering Docs

This directory contains the reverse-engineering outputs for the TX8/Wafer
hardware stack. The original base is `third_party/tx8_deps`; later evidence
passes also used a firmware Kuiper SDK snapshot and a copied `torch_txda`
wheel where those artifacts clarify host runtime, driver, and PyTorch eager
integration behavior. Those external snapshots are provenance, not stable
repository paths.

## Files

| file | purpose |
| --- | --- |
| [tx8-interface-contract.md](tx8-interface-contract.md) | Canonical recovered-interface evidence for instruction APIs, packet fields, runtime/driver APIs, DTE/stream/mailbox/PMU semantics and bootparam/TLV layout; it does not own Wafer IR/ABI policy. |
| [firmware-kuiper-runtime-hardware-analysis.md](firmware-kuiper-runtime-hardware-analysis.md) | Full SDK, HPGR runtime, KMD/UAPI, BO/BAR/ATU, DTE/C2C, PG, completion semantics, system tools, and host/driver evidence from `firmware_kuiper`. |
| [tx8-deps-reverse-engineering-reference.md](tx8-deps-reverse-engineering-reference.md) | Snapshot evidence reference derived from `tx8_deps` headers, build files, symbols, and disassembly; it intentionally remains tx8-deps-only. |
| [tx8-api-struct-contract-annex.md](tx8-api-struct-contract-annex.md) | Generated evidence annex containing host/Kcore signatures, enums, wrapper structs, and packet structs. |
| [tx81-dlcompiler-crt-source-audit.md](tx81-dlcompiler-crt-source-audit.md) | Evidence audit of the old DLCompiler TX81 source that produced historical `libvr.a`: function inventory, wrapper-call findings, implementation conflicts, and missing proof. |
| [tx81-current-crt-conformance-matrix.md](tx81-current-crt-conformance-matrix.md) | Evidence comparison between the registered Wafer CRT surface, current implementation checks, and the old TX81 CRT source audit; it is not an ABI owner. |
| [tx81-extended-crt-surface-triage.md](tx81-extended-crt-surface-triage.md) | Evidence classification for old TX81 functions outside the registered surface: direct-wrapper, composite/layout/DTE, stub, and compatibility categories. |
| [txda-pytorch-runtime-wheel-analysis.md](txda-pytorch-runtime-wheel-analysis.md) | Static analysis of the copied `torch_txda` PyTorch runtime wheel: `PrivateUse1` backend registration, CUDA compatibility patches, tx_runtime/txdnn dependencies, stream/event semantics, eager op coverage, and implications for Wafer runtime layering. |
| [tx8-symbol-coverage-matrix.md](tx8-symbol-coverage-matrix.md) | Human-readable symbol coverage summary. |
| [tx8-symbol-coverage-matrix.csv](tx8-symbol-coverage-matrix.csv) | Full generated symbol coverage matrix. |

The root-level Wafer documents are hardware-facing evidence summaries:

- [../wafer-hardware-instruction-set-and-programming-model.md](../wafer-hardware-instruction-set-and-programming-model.md) summarizes hardware topology, memory/SPM/layout, runtime/provider evidence, DTE, stream/CSR, parallel execution, and remaining hardware-facing gaps.
- [../wafer-register-level-instruction-spec.md](../wafer-register-level-instruction-spec.md) is the wrapper/register evidence annex for packet fields, opcodes, units, wait/CSR behavior, and PMU/profiling observations.

Current compiler/runtime contracts are owned by the numbered `tasks/01-16`
design documents; use `tasks/README.md` for the complete pipeline-boundary to
owner map. Common consumers of this directory include `tasks/11` for instruction
legality, `tasks/13` for physical transport, `tasks/14` for target artifacts and
`tasks/15` for package/runtime behavior, but that list is not exhaustive. Files
in this directory and the two root evidence summaries retain source-backed
facts; when a fact becomes policy, update its numbered owner and leave the proof
here. The one legacy presence-only Markdown checker coupling is explicitly
documented in the CRT evidence matrix and queued for removal in `tasks/progress.md`.

Regenerate generated outputs from repo root:

```bash
python3 tools/tx8_symbol_coverage.py
python3 tools/tx8_contract_annex.py
```

## Generated Coverage Snapshot

The table below is generated from `tx8-symbol-coverage-matrix.csv` and covers
`tx8_deps` only.  It does not count the additional `firmware_kuiper` KMD/HPGR
or `torch_txda` wheel findings. Counts describe the audited dependency snapshot,
not Wafer task status or production acceptance.

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

Cross-document evidence summary:

| area | observed evidence and limitation |
| --- | --- |
| Instruction wrappers and register packets | CT/NE/RDMA/WDMA/TDMA packet fields, wrapper units, CSR wait, and observed `serial_mode` behavior have static evidence. Whether that evidence is sufficient for a production command is decided by the numbered instruction and target contracts. |
| Host runtime and driver | HPGR `tx_runtime.h`/`libhpgr.so` is the most complete CUDA-like provider surface observed in the `firmware_kuiper` snapshot; VS `Tsm*` remains compatibility evidence with several stubs. Provider selection and runtime policy belong to `tasks/15`. |
| Completion semantics | KMD compute fences are not proof of model/kernel completion in this driver snapshot. HPGR command-slot completion, async receive thread, module `completeSignal`, and stream waits are distinct observed mechanisms whose production mapping belongs to `tasks/15`. |
| Address space and PG | BAR/ATU windows, BO pools, tile SPM/register layout, Kcore/Score firmware slots, global stream mapping table, and 8/16-tile PG behavior are source- or binary-confirmed. |
| DTE | KMD UAPI, Kcore raw registers, direct-DTE helper, and VS D2D/P2P TLV paths are separate evidence layers. Production transport remains whatever `tasks/13`/`tasks/14` explicitly accept and board-test. |
| Parallel/SPM bank | `serial_mode=0`, per-packet range metadata, and SPM bank-resource hazards are confirmed. Exact SPM address-to-bank mapping is unproven; 64 KiB page coloring appears only as a historical heuristic. |

## Remaining Gaps

1. `HardwareVerify` is the main evidence gap: PMU, MHU/power-off, DTE PMU counters, board services, and hardware state/timing paths still need board validation.
2. `__execute_sc` is reserved/stub in the current `libinstr_tx81.a`; this snapshot does not prove a production SCALAR path. Acceptance remains with the numbered instruction and target contracts.
3. Exact SPM bank mapping and aggressive parallel cost modeling require board sweeps or lower RTL/firmware evidence.
4. API/function semantics, instruction constraints, runtime/driver behavior, and DTE/stream/mailbox/PMU register meanings are consolidated in [tx8-interface-contract.md](tx8-interface-contract.md), with SDK/driver additions in [firmware-kuiper-runtime-hardware-analysis.md](firmware-kuiper-runtime-hardware-analysis.md).
5. Raw multi-destination DTE modes have register evidence but incomplete public-helper evidence. `tasks/13`, `tasks/14`, and the board gates own any future compiler-facing acceptance.
