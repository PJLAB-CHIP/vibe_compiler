# TX8 Recovered Interface, Runtime, and Hardware Evidence

This is the canonical recovered-interface evidence ledger for the audited TX8
dependency package. It records what each public wrapper/API family writes into
instruction packets, which field relations appear in each instruction class, what the
TX8 hardware-facing register interfaces mean, and how the host runtime drives
device memory, bootparams, dyn data, topology, launch, profiling, and the
incomplete host CModel loading seams.

It is not a Wafer IR, ABI, transport, provider, or runtime-policy owner. The
numbered design documents linked from this directory's README decide which
recovered facts enter production and how they are represented and verified.

This ledger records host/runtime APIs as a distinct evidence layer. Backend
branch names recovered from the binary are implementation gates rather than an
architecture taxonomy. Raw tx8-deps evidence and broader dependency notes
remain in `tx8-deps-reverse-engineering-reference.md`; generated signatures and
struct shapes remain in `tx8-api-struct-contract-annex.md`. Full SDK/KMD/HPGR
evidence from `firmware_kuiper` is summarized in
`firmware-kuiper-runtime-hardware-analysis.md`; it contains the corresponding
host runtime, driver, BO/BAR/ATU, PG, C2C, and completion evidence.

Evidence used in this pass:

- Headers: `third_party/tx8_deps/include/instr_def.h`,
  `instr_adapter.h`, `instr_adapter_plat.h`, and Kcore headers under
  `tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include`.
- Static disassembly: `libinstr_tx81.a` objects
  `instr_adapter.c.o` and `instr_adapter_opt.c.o`.
- Kcore disassembly: `libkcorert.a` objects `kuiper_dte.c.o`,
  `mod_dte.c.o`, `kuiper_streamfsm.c.o`, `mod_streamfsm.c.o`,
  `stream_rt.c.o`, `pmu.c.o`, and mailbox objects.
- Host disassembly: `profiling_tool/examples/engtest_example/libtx8_runtime.so`.
- SDK/KMD cross-check: `firmware_kuiper` HPGR runtime and decrypted KMD source,
  especially BO/job/NPU/DTE/C2C UAPI and HPGR model/module completion paths.

## 1. Coverage Position After Re-analysis

The old gap was not mostly about missing C/C++ symbol names. The generated
annex already lists signatures and packet structs. The real gap was semantic:
argument-to-field behavior, observed field bounds, register meanings, host runtime
API semantics, and stub versus real driver behavior. This file records that
evidence without deciding Wafer compiler or runtime acceptance.

| area | static semantic status | remaining non-static work |
|---|---|---|
| `libinstr_tx81.a` wrapper APIs | Covered by function-family rules below and packet/register maps. | Hardware error behavior for invalid packets still needs board tests. |
| `TsmExecute` dispatch | Covered: only instruction types 0..4 dispatch. | None for dispatch itself. |
| CT/NE/RDMA/WDMA/TDMA packet layout | Covered from header offsets and execute disassembly. | Exact arithmetic corner cases, NaN/overflow flags, and timing require hardware. |
| SCALAR | `__execute_sc` is observed as a clearing stub with no register emission. | The snapshot does not establish an executable scalar path; acceptance belongs to `tasks/11` and `tasks/14`. |
| Kcore DTE/stream/mailbox | Register protocol and payload format covered. | Multi-destination DTE policies, mailbox failure recovery, and timing need board tests. |
| PMU/profiling | Register and TLV shape covered. | Counter units, wrap edge cases, and event accuracy need board tests. |
| Host runtime / driver layer | Exported `Tsm*` signatures, implemented/stub behavior, bootparam/dyn-data paths, launch/copy/topology/profiling calls covered; HPGR/KMD pass adds the largest observed `tx_runtime` surface, BO/BAR/ATU, KMD UAPI, PG, and completion mechanisms. | Provider selection and runtime semantics belong to `tasks/15`; target validation belongs to `tasks/16`. |
| Host CModel seams | Low-level instruction CModel declarations and high-level `libtx8_runtime.so` dynamic CModel loader are identified. | Required host implementations, libraries, headers, model resources, reachable launch path, numeric/timing contract, and internal framework are absent or unproven; target-model ownership belongs to `tasks/17`. |

Symbol coverage remains measured by `tx8-symbol-coverage-matrix.csv`. The
important practical result is that the observed instruction families have
documented packet fields and static constraints. Production verifier rules are
owned by the numbered instruction/target designs. The remaining
`HardwareVerify` rows are hardware-observable state, timing, PMU, MHU, power, or
closed device-driver paths.

### Interface Semantics Inventory

The reverse-engineered interface surface is split into several layers. Backend
names alone do not establish relationships between those layers.

| layer | interface family | semantics recovered here |
|---|---|---|
| Instruction wrapper API | `TsmElemWise`, CT relation/logic/reduce/convert/peripheral helpers | Function family, opcode range, address-versus-scalar operand role, dtype storage size, bool packing, end-field behavior, writeback behavior, and observed field bounds. |
| Instruction wrapper API | `TsmConv`, `TsmDepthWiseConv`, `TsmGemm` | Argument-to-field mapping for input/weight/output/psum/bias/scale/sparse/quant/pad/stride/dilation/GEMM MKN/batch/trans flags, plus shape and quant bounds. |
| Instruction wrapper API | `TsmRdma`, `TsmWdma`, `TsmDataMove`, `TsmPeripheral` | DDR/SPM direction, family-specific element/byte count and stride units, `iteration-1` encoding, TDMA opcodes, and register window offsets. |
| CSR/reserved API | CSR helpers and `I_SCALAR` | CSR status bit meanings and worker addressing are recovered; scalar packet execution is a stub with no observed register emission. Production acceptance belongs to `tasks/11` and `tasks/14`. |
| Kcore hardware API | DTE | Register fields, mode/user_id bits, high-level modes, source/destination setup, shuffle stride encoding, trigger, done/error return codes, and packet-counter update format. |
| Kcore hardware API | Stream FSM and mailbox | Stream config layout, packet counters, online/offline/request/push/pop payloads, mailbox TX/RX window protocol, payload register count, and observed status handling. |
| Profiling API | PMU helpers and host profiling dyn data | DTE/SPM/NCC PMU bases and record types, observed split 32-bit read orders, host `D_PROF_CFG` control path, and remaining hardware validation items. |
| Host runtime API | `Tsm*` runtime exports | Device selection, memory allocation/free, H2D/D2H/D2D/P2P copies, bootparam launch, kernel load/unload, tile topology, power hooks, profiling, return code semantics, and stub boundaries. |
| Host CModel API | instruction CModel declarations and `Runtime::SetCModelHandle` | Distinct low-level packet/operator and high-level `TsmDevice`/`TsmModel` dynamic ABI traces, plus the exact missing-package boundary. |
| HPGR/KMD API | `tx_runtime.h`, KMD UAPI | CUDA-like device/memory/stream/event/model/module API, command completion, BO pools, job/DTE/C2C ioctl surfaces, PG tile map, and BAR/ATU address-space handling. |

Interface units are explicit in the relevant sections: tensor element counts for
CT/NE logical work, packed bytes for bool storage, logical element strides for
RDMA/WDMA, byte strides for TDMA/DTE fields, byte lengths for DTE transfers, and device physical addresses for
runtime bootparams and dyn-data buffers.

## 2. TX8 Hardware Surface Map

The hardware surface recovered from headers and object code is broader than the
host runtime. In this snapshot, host runtime APIs perform device-memory,
bootparam, dyn-TLV, and launch operations; Wafer provider and resource ownership
is defined by `tasks/15`. The table below is the address-level map used by the
Kcore and instruction libraries.

| block | base | recovered semantic status |
|---|---:|---|
| L1 SPM | `0x000000` | Tensor/local SRAM for NCC and Kcore. Size is `0x300000`; Kcore/runtime code uses `0x2f0000..0x2fffff`. |
| NCC instruction MMIO | `0x01000000` | CT/NE/RDMA/WDMA/TDMA register windows, with three observed worker windows spaced by `0x100000`. |
| DTE | `0x400000` | Kcore DMA engine. Four blocks, block stride `0x200`; source/destination/mode/status semantics recovered. |
| SCONFIG | `0x500000` | System config base known from map; no compiler-facing bit protocol recovered beyond dependency boundary. |
| SPM PMU | `0x580000` | SPM performance counters used by Kcore PMU helpers. |
| NCC PMU | `0x590000` | NCC performance counters for CT/NE/RDMA/WDMA/TDMA/user timers. |
| Tile wrap CRG | `0x600000` | Clock/reset block base known; detailed sequencing stays hardware-validated. |
| Stream interrupt | `0x610000` | Base known; stream FSM protocol is at `0x620000`. |
| Stream FSM | `0x620000` | Stream packet config, ready bitmap, packet counters, and base-address check recovered. |
| Mailbox TX | `0x640000` | TX window acquire/release, MSI target, payload, and status polling recovered. |
| Mailbox RX | `0x660000` | RX pending count, payload pop, and read-done handshake recovered. |
| Packet counter update | `0x670000` | DTE completion updates stream packet counters. |
| MHU tile/AP | `0x680000..0x691000` | Tile/AP message-unit bases known; detailed protocol remains hardware-validated. |
| Address map | `0x6a0000` | Base known. |
| Broadcast control | `0x6c0000` | Base known. |
| Tile misc | `0x6e0000` | Base known. |
| TMNOC PMU | `0x30700000`, `0x30b00000` | Base macros known; no compiler-facing protocol decoded in this pass. |
| DDR cached | `0x80000000` | Kcore SoC DDR mapping. |
| DDR weak-order uncached | `0x180000000..0x27fffffff` | Uncached/weak-order DDR alias. |
| External DDR | `0x8000000000` | External DDR mapping. |
| L1SPM weak/strong-order aliases | `0x30400000`, `0x30800000` | Uncached weak/strong-order SPM aliases. |

The recovered evidence is grouped into these layers; production contracts remain in numbered designs:

- NCC instruction evidence: `TsmExecute` packet types 0..4 and the CT/NE/DMA/TDMA
  register windows.
- Kcore movement/stream evidence: DTE, stream FSM, mailbox, and packet counter
  update registers.
- Profiling evidence: NCC, DTE, and SPM PMU record formats and register bases.
- Board/runtime dependency boundary: MHU, power/CRG, TMNOC, and system config
  bases are identified, but their bit-level behavior is not compiler-facing yet.

Instruction, transport, target, provider, and verification acceptance is owned
by `tasks/11`, `tasks/13`, `tasks/14`, `tasks/15`, and `tasks/16`, respectively.

## 3. Common Instruction Model

All instruction packets start with `uint32_t inter_type`. Low bits select the
functional unit; bits 8..9 encode worker selection for NCC register windows.

| value | type | observed `TsmExecute` status |
|---:|---|---|
| 0 | `I_CGRA` / CT | dispatched |
| 1 | `I_NEUR` / NE | dispatched |
| 2 | `I_RDMA` | dispatched |
| 3 | `I_WDMA` | dispatched |
| 4 | `I_TDMA` | dispatched |
| 5 | `I_SCALAR` | reserved/stub |
| 6 | `I_DTE` | not dispatched by `TsmExecute` |
| 7 | `I_CSR` | not dispatched by `TsmExecute` |

Worker values are `I_WORKER0 = 0x0000`, `I_WORKER1 = 0x0100`, and
`I_WORKER2 = 0x0200`. Execute helpers derive the NCC worker window from
`(inter_type >> 8) & 3`, normalized to three workers, and add `worker * 0x100000`
to the NCC base `0x01000000`.

`TsmExecute(void *instr)` reads `*(uint8_t *)instr` and dispatches only values
0..4. Values greater than 4 return `1` without running scalar, DTE, or CSR
execution. The snapshot therefore provides no `TsmExecute` execution evidence
for `I_SCALAR`, `I_DTE`, or `I_CSR`; production acceptance and emission are
owned by `tasks/11` and `tasks/14`.

### Observed Wrapper Object Lifecycle

Wrapper objects in the snapshot are C structs of function pointers rather than
C++ classes. The recovered factory/table lifecycle has this call shape:

```c
TsmArith *arith = TsmNewArith();
TsmArithInstr instr = {0};
arith->AddVV(&instr, src0, src1, dst, elem_count, RND_NEAREST_EVEN, Fmt_FP16);
TsmExecute(&instr);
TsmWaitfinish();
TsmDeleteArith(arith);
```

The same lifecycle appears for `TsmConv`, `TsmDepthwiseConv`, `TsmGemm`,
`TsmRdma`, `TsmWdma`, `TsmDataMove`, `TsmPeripheral`, and the CT-family wrapper
tables listed in the generated annex. Production entry strategy and CRT usage
are owned by `tasks/14`.

### Data Formats

| enum | value | storage bytes |
|---|---:|---:|
| `Fmt_INT8` | 0 | 1 |
| `Fmt_INT16` | 1 | 2 |
| `Fmt_FP16` | 2 | 2 |
| `Fmt_BF16` | 3 | 2 |
| `Fmt_INT32` | 4 | 4 |
| `Fmt_FP32` | 5 | 4 |
| `Fmt_TF32` | 6 | 4 |
| `Fmt_BOOL` | 7 | bit-packed, `ceil(elem_count / 8)` bytes |
| `Fmt_UINT8` | 8 | 1 |
| `Fmt_UINT16` | 9 | 2 |
| `Fmt_UINT32` | 10 | 4 |
| `Fmt_INT64` | 11 | 8 |
| `Fmt_UINT64` | 12 | 8 |

Bool paths are special in the recovered wrappers: they either compute packed
byte counts directly or rewrite DMA format to byte format when moving packed
bool storage. Here `elem_count` denotes logical elements while storage uses
packed bytes. Representation and verifier acceptance are owned by `tasks/11`
and `tasks/14`.

### Address Domains

Instruction memory operands are NCC-visible addresses. The wrapper headers
define:

| consumer-visible domain | accepted range / view |
|---|---|
| NCC-visible SPM | `0x00000000..0x002effff` |
| Kcore reserved SPM | `0x002f0000..0x002fffff` |
| instruction-adapter accepted DDR operand | `>= 0x280000000` |

CT, NE, and TDMA wrapper operands are SPM addresses. RDMA source is DDR and destination
is SPM. WDMA source is SPM and destination is DDR. Kcore code marks the final
SPM window as reserved; allocator legality is owned by the numbered memory designs.
The DDR predicate has no static upper bound and is not a complete hardware or host
address-space map. Other aliases, BAR mappings, invocation resources, and consumer-specific
address views remain distinct until their owning contract binds them.

### Instruction Packet Struct Offsets

The generated annex gives the C struct shapes. These byte offsets are recovered
C-layout facts; production ABI acceptance and golden coverage are owned by
`tasks/14` and `tasks/16`.

`CT_Param`:

| byte offset | field |
|---:|---|
| `+0` | `inter_type` |
| `+4` | `ctrl.cmd_valid` |
| `+5` | `ctrl.rnd_mode` |
| `+6` | `ctrl.src0_format` |
| `+7` | `ctrl.opcode` |
| `+8` | `param.src0` |
| `+12` | `param.src1` |
| `+16` | `param.dst0` |
| `+20` | `param.dst1` |
| `+24` | `param.dst2` |
| `+32` | `param.src0_tfr` |
| `+40` | `param.dst_tfr` |
| `+48` | `param.pdr` |
| `+56` | `param.swr` |
| `+64` | `param.elem_count` |
| `+72` | `param.unit_elem_count` |
| `+80` | `param.int8_scale_val0` |
| `+88` | `param.int8_scale_val1` |
| `+96` | `param.int8_quant` |
| `+104` | `param.int8_bn_bias` |
| `+108` | `param.full_elem_count` |
| `+112` | `param.full_unit_elem_count` |
| `+120` | `param.wb_data0` |
| `+128` | `param.wb_data1` |
| `+136` | `param.src0_end` |
| `+140` | `param.src1_end` |
| `+144` | `param.dst0_end` |
| `+148` | `param.dst1_end` |
| `+152` | `param.dst2_end` |
| `+156` | `param.dims` |

`TsmNeInstr`:

| byte offset | field |
|---:|---|
| `+0` | `inter_type` |
| `+4..+15` | NE control bytes: sparse, valid, formats, psum/relu/scale/bias/dilation flags, type |
| `+16` | `src_a` |
| `+20` | `src_w` |
| `+24` | `psum` |
| `+28` | `bias` |
| `+32` | `scale_p` |
| `+36` | `scale_n` |
| `+40` | `out` |
| `+48` | `tfr_0` |
| `+56` | `tfr_1` |
| `+64` | `pdr` |
| `+72` | `unpdr` |
| `+80` | `swr` |
| `+88` | `dilation` |
| `+96` | `gemm_lb` |
| `+98` | `gemm_rb` |
| `+100` | `gemm_k` |
| `+102` | `gemm_m` |
| `+104` | `gemm_n` |
| `+106` | `gemm_l_trs` |
| `+107` | `gemm_r_trs` |
| `+108..+112` | quant fields |
| `+116` | `sparse_index` |
| `+120..+148` | end fields |

`DMA_Param`:

| byte offset | field |
|---:|---|
| `+0` | `inter_type` |
| `+4` | `cmd_valid` |
| `+8` | `dst` |
| `+16` | `src` |
| `+24/+28` | stride/iteration 0 |
| `+32/+36` | stride/iteration 1 |
| `+40/+44` | stride/iteration 2 |
| `+48` | `elem_count` |
| `+52` | `format` |
| `+56` | `src_end` |
| `+64` | `dst_end` |

`TD_Param` starts with `inter_type@+0`, `cmd_valid@+4`,
`src0_format@+5`, and `opcode@+6`; the payload fields are the TDMA fields
listed in the TDMA section below and in the generated annex.

## 4. CT / CGRA Tensor Semantics

Packet type: `CT_Param`.

Control encoding written by `__execute_ct`:

```c
control =
    (1 << 16) |                  // cmd_valid
    ((rnd_mode & 0xf) << 12) |
    ((src0_format & 0xf) << 8) |
    opcode;
```

Register base is `0x01000000 + worker_offset`.

| field | register offset |
|---|---:|
| control | `0x000` |
| `src0` | `0x010` |
| `src1` | `0x020` |
| `dst0` | `0x030` |
| `dst1` | `0x040` |
| `dst2` | `0x050` |
| `dims` | `0x060` |
| `src0_tfr` | `0x070` |
| `dst_tfr` | `0x080` |
| `pdr` | `0x090` |
| `swr` | `0x0a0` |
| `elem_count` | `0x0b0` |
| `unit_elem_count` | `0x0c0` |
| `int8_scale_val0` | `0x0d0` |
| `int8_scale_val1` | `0x0e0` |
| `int8_quant` | `0x0f0` |
| `int8_bn_bias` | `0x100` |
| `full_elem_count` | `0x110` |
| `full_unit_elem_count` | `0x120` |
| `wb_data0` | `0x130` |
| `wb_data1` | `0x140` |
| `src0_end` | `0x150` |
| `src1_end` | `0x160` |
| `dst0_end` | `0x170` |
| `dst1_end` | `0x180` |
| `dst2_end` | `0x190` |

### CT Function Families

Specific function names and opcodes are listed in
`tx8-api-struct-contract-annex.md`. Their packet semantics follow these common
families:

| family | examples | packet semantics |
|---|---|---|
| `V_V` | `AbsVV`, `SqrtVV`, `Relu` | `inter_type=I_CGRA`, one SPM source in `src0`, one SPM output in `dst0`, `elem_count`, `src0_format`, opcode, `src0_end`, `dst0_end`. |
| `V_VV` | `AddVV`, `SubVV`, relation `EqualVV` | `src0`, `src1`, `dst0`, `elem_count`, format, rounding if supported, all three end fields. |
| `V_VS` | `AddVS`, `MaxVS`, relation scalar variants | `src0` is SPM, `src1` stores the immediate scalar value, not an address; `src1_end` is not an operand range. |
| `V_VuV` | `AddVuV`, relation/logic unit-vector variants | `src0` uses `elem_count`; `src1` is a short/unit vector using `unit_elem_count`; the documented `unit_elem_count` range is `1..64`. |
| `V_VuVLoop` | loop unit-vector variants | Also writes `full_elem_count` and `full_unit_elem_count`; end fields are computed from full counts. |
| bool relation/logical | `BoolEqualVV`, `BoolAndV` | Output format is bit-packed bool. `elem_count` is logical bool count; storage is `ceil(elem_count/8)`. |
| tensor shape ops | pool, reduce, unpool, pad-like CT ops | Write `src0_tfr`, `dst_tfr`, `pdr`, `swr`, and `dims` as applicable. Shapes are packed as NHWC 16-bit lanes. |
| convert | `INT8_FP16`, `FP32_BF16`, etc. | Source format is implied by function group, destination format by opcode; `rnd_mode` is honored where present in signature. |
| writeback peripheral | `Count`, `ArgMax`, `ArgMin`, `BitCount` | Opcode 175..178 can poll `wb_data0` and optionally `wb_data1` after trigger; valid data is bit 31 of the readback register. |

CT opcode ranges:

| range | operation family |
|---:|---|
| 0..29 | arithmetic |
| 30..77 | relation |
| 78..97 | logic |
| 98..104 | transcendental |
| 105..110 | activation |
| 111..114 | reduce |
| 115..120 | pool |
| 121..138 | unpool, reshape, data movement, mask |
| 139..174 | convert |
| 175..186 | peripheral |

`__execute_ct` clears the 160-byte CT instruction object after normal
non-writeback execution. For opcodes 175..178, it preserves `wb_data0/1` by
polling register offsets `0x130` and `0x140` when the corresponding fields are
nonzero, then stores the low 32 bits back into the packet fields.

### Observed CT Field Relations and Bounds

- Wrapper paths set `cmd_valid` and select opcodes from the recovered CT table.
- `src0_format` uses `Data_Format`; bool helpers compute packed storage sizes.
- The documented unit-vector range is `1..64`, and the exposed rounding enum is
  `0..4`.
- The packet dimension encoding exposes `C=0`, `W=1`, `H=2`, `N=3`, `HW=4`,
  and `HWC=5`. This is the full observed packet enum; operation-specific Reduce
  acceptance belongs to `tasks/11` and `tasks/14`.
- Wrapper helpers derive SPM operand end fields and distinguish scalar immediates
  in `src1` from address operands.

These relations are evidence inputs, not a verifier contract. Production
legality and packet emission remain owned by `tasks/11` and `tasks/14`.

## 5. NE Semantics

Packet type: `TsmNeInstr`.

Control encoding written by `__execute_ne`:

```c
control =
    ((sparse_en & 1) << 21) |
    (1 << 20) |                   // cmd_valid
    ((inpsum_format & 0xf) << 16) |
    ((output_format & 0xf) << 12) |
    ((input_format & 0xf) << 8) |
    ((inpsum_en & 1) << 7) |
    ((lrelu_en & 1) << 6) |
    ((relu_en & 1) << 5) |
    ((scale_en & 1) << 4) |
    ((bias_en & 1) << 3) |
    ((dilation_conv & 1) << 2) |
    (type & 3);
```

NE register offsets:

| field | register offset |
|---|---:|
| control | `0x200` |
| `src_a` | `0x210` |
| `src_w` | `0x220` |
| `psum` | `0x230` |
| `bias` | `0x240` |
| `scale_p` | `0x250` |
| `scale_n` | `0x260` |
| `out` | `0x270` |
| `tfr_0` | `0x280` |
| `tfr_1` | `0x290` |
| `pdr` | `0x2a0` |
| `unpdr` | `0x2b0` |
| `swr` | `0x2c0` |
| `dilation` | `0x2d0` |
| `gemm_lb` | `0x2e0` |
| `gemm_rb` | `0x2f0` |
| `gemm_n` | `0x300` |
| `gemm_m` | `0x310` |
| `gemm_k` | `0x320` |
| `gemm_l_trs` | `0x330` |
| `gemm_r_trs` | `0x340` |
| quant packed word | `0x350` |
| `sparse_index` | `0x360` |
| end fields | `0x370..0x3e0` |

### NE API Definitions

| API | concrete behavior |
|---|---|
| `TsmConv::AddInput(instr, X_addr, shape, fmt)` | Writes `inter_type=I_NEUR`, `src_a=X_addr`, `input_format=fmt`, and packs `shape` into `tfr_0`. |
| `TsmConv::AddWeight(instr, W_addr, shape, fmt)` | Writes `src_w=W_addr`; for backward conv (`type=2`) packs full weight shape to `tfr_1`; regular conv/depthwise use shape fields for kernel/window calculations. |
| `TsmConv::AddOutput(instr, Out_addr, shape, fmt)` | Writes `out=Out_addr`, `output_format=fmt`; for non-backward conv packs output shape to `tfr_1`. |
| `AddBias(instr, bias_en, bias_addr)` | Writes `bias_en` and `bias`. |
| `SetNegativeAxisScale`, `SetPositiveAxisScale` | Write shared `scale_en` state and the corresponding scale address (`scale_n` or `scale_p`). |
| `SetSparse(instr, sparse_en, sparse_addr)` | Writes `sparse_en` and `sparse_index`. |
| `SetPsum(instr, psum_en, psum_addr, fmt)` | Writes `inpsum_en`, `psum`, and `inpsum_format`. |
| `SetPads` / `SetUnPads` | Pack top/bottom/left/right into 16-bit lanes of `pdr` or `unpdr`. |
| `SetKernelStrides` | Packs `Kx`, `Ky`, `Sx`, `Sy` into `swr`. |
| `SetDilations` | Packs dilation lanes and sets dilation behavior when used by conv type. |
| `EnableRelu` | Writes `relu_en=1`, `lrelu_en=0`. |
| `EnableLeakyRelu` | Writes `lrelu_en=1`, `relu_en=0`. |
| `DisableRelu`, `DisableLeakyRelu` | Clear the corresponding byte. |
| `SetQuant(q0,q1,zp_pre,zp_cur)` | Writes `quant_q0=q0`, `quant_q1=q1`, `quant_zp_pre=zp_pre`, `quant_zp_cur=zp_cur`. |
| `TsmGemm::AddInput(L,R,fmt)` | Writes `inter_type=I_NEUR`, `type=3`, `src_a=L`, `src_w=R`, `input_format=fmt`. |
| `TsmGemm::ConfigMKN(M,K,N)` | Writes `gemm_m=M`, `gemm_k=K`, `gemm_n=N`. |
| `TsmGemm::ConfigBatch(left,right)` | Writes `gemm_lb` and `gemm_rb`. |
| `TsmGemm::SetTransflag(L,R)` | Writes `gemm_l_trs` and `gemm_r_trs`. |
| `TsmGemm::AddOutput(out,fmt)` | Writes `out` and `output_format`. |

`__execute_ne` recomputes end fields before register emission. It uses tensor
shape helpers (`common_tensor_info_generate`, `get_aligned_ck`,
`get_chip_aligned_ck`, `bank_align_elem`, and `get_dtype_size`) and writes
`srca_end`, `srcw_end`, `psum_end`, `bias_end`, `scale_p_end`, `scale_n_end`,
`out_end`, and `sparse_end`. This is the observed wrapper execution path;
production end-field derivation and any direct packet emission are owned by
`tasks/11` and `tasks/14`.

### Observed NE Field Bounds

Headers and wrapper implementations expose the following field ranges and
relations. Their production legality is owned by `tasks/11` and `tasks/14`.

- `type` values: `0=conv`, `1=depthwise conv`, `2=backward conv`, `3=gemm`.
- `tfr_0` / `tfr_1` pack `n,h,w,c` as 16-bit lanes. `n/h/w` range is
  `1..4096`; `c` range is `1..16384`.
- Pad/unpad lanes are `0..1023`.
- `Kx/Ky` range is `1..255`; `Sx/Sy` range is `1..1023`.
- Dilation lanes are `1..1023`.
- GEMM `M/N` are 16-bit; `K` range is `1..16384`; batches are `1..4096`.
- Quant zero-points are `0..255`; quant shifts are `0..31`.
- Enabled optional operands (`psum`, `bias`, scales, sparse index) participate in
  the wrapper's SPM end-field calculations.

## 6. RDMA and WDMA Semantics

Packet type: `DMA_Param`.

`TsmRdma` is DDR to SPM. `TsmWdma` is SPM to DDR.

| API | behavior |
|---|---|
| `Rdma::AddSrcDst(src,dst,fmt)` | Sets `inter_type=I_RDMA`, `cmd_valid=1`, `src=src` DDR, `dst=dst` SPM, `format=fmt`. |
| `Wdma::AddSrcDst(src,dst,fmt)` | Sets `inter_type=I_WDMA`, `cmd_valid=1`, `src=src` SPM, `dst=dst` DDR, `format=fmt`. |
| `ConfigStrideIteration(elem_count, stride0, iteration0, stride1, iteration1, stride2, iteration2)` | Stores logical element strides and each logical iteration as `iteration - 1`; BOOL inputs are logical bit counts/strides and are packed by the setter; zero logical iteration is invalid. |
| contiguous helper API | Build a single contiguous movement, infer stride and end fields from format and element count. |

RDMA register offsets:

| field | offset |
|---|---:|
| control | `0x400` |
| `src` | `0x410` |
| `dst` | `0x420` |
| stride/iteration 0 | `0x430` |
| stride/iteration 1 | `0x440` |
| stride/iteration 2 | `0x450` |
| `elem_count` | `0x460` |
| `format` | `0x470` |
| `src_end` | `0x480` |
| `dst_end` | `0x490` |

WDMA register offsets are the same sequence starting at `0x4a0`:
control `0x4a0`, `src` `0x4b0`, `dst` `0x4c0`, stride pairs `0x4d0..0x4f0`,
`elem_count` `0x500`, format `0x510`, ends `0x520..0x530`.

`__execute_rdma` and `__execute_wdma` write all parameters first, write
`cmd_valid=1` last, then clear 72 bytes of the instruction object.

Observed DMA field conventions:

- RDMA wrapper paths encode DDR source and SPM destination; WDMA paths encode
  SPM source and DDR destination.
- Wrapper helpers use positive element counts and encode each nonzero logical
  iteration as `iteration-1`.
- Strides are bytes rather than elements.
- `Fmt_BOOL` is packed; wrappers compute byte counts with
  `ceil(elem_count/8)` and can store byte format for the transfer.

Production DMA legality and unit conversion are owned by `tasks/11` and
`tasks/14`.

## 7. TDMA and DataMove Semantics

Packet type: `TD_Param`.

TDMA control encoding:

```c
control = (1 << 12) | ((src0_format & 0xf) << 8) | opcode;
```

TDMA register offsets:

| field | offset |
|---|---:|
| control | `0x540` |
| `src0` | `0x550` |
| `src1` | `0x560` |
| `dst` | `0x570` |
| `dims` | `0x580` |
| `src0_tfr` | `0x590` |
| `dst_tfr` | `0x5a0` |
| `pdr` | `0x5b0` |
| `swr` | `0x5c0` |
| `elem_count` | `0x5d0` |
| source stride/iteration 0..2 | `0x5e0..0x600` |
| destination stride/iteration 0..2 | `0x610..0x630` |
| `src0_end` | `0x640` |
| `src1_end` | `0x650` |
| `dst_end` | `0x660` |

`__execute_td` writes the TDMA register window, writes control, calls
`debug_td_info`, clears 128 bytes of the object, and returns `1`.

### DataMove APIs

| API | opcode | semantic class |
|---|---:|---|
| `Mirror` | 124 | reshape/layout movement within SPM, source and destination shapes. |
| `Transpose` | 125 | swaps layout according to source/destination NHWC shape. |
| `Rotate90` | 126 | image/tensor rotate. |
| `Rotate180` | 127 | image/tensor rotate. |
| `Rotate270` | 128 | image/tensor rotate. |
| `Nchw2nhwc` | 129 | layout conversion. |
| `Nhwc2nchw` | 130 | layout conversion. |
| `Concat` | 131 | two-source concat; uses `src0`, `src1`, `dst`, `dims`, and shapes. |
| `Pad` | 132 | pad tensor; uses `pdr` and source/destination shapes. |
| `TensorNom` | 133 | tensor normalization style layout operation. |
| `GatherScatter` | 135 | byte-count gather/scatter using source and destination stride iteration triples. |
| `Img2col` | 138 | image-to-column movement using `swr`, `pdr`, and source/destination element counts. |

`TsmPeripheral::Memset` is TDMA, not CT. It writes `inter_type=I_TDMA`,
opcode `179`, destination address in the TD packet, the fill value, byte
strides, and TD end fields. The header says `si.stride` is byte size; `elem_count`
is nominal element count, while TD packet comments say memset/gatherscatter use
byte number. This is unit evidence rather than a lowering rule; conversion and
packet acceptance are owned by `tasks/11` and `tasks/14`.

`GatherScatter` writes `elem_count=size` in bytes, copies source and destination
stride/iteration triples, and computes only `src0_end` and `dst_end`;
`src1_end` is zero.

Observed TDMA field relations:

- Recovered TDMA wrapper data operands use SPM addresses.
- Shapes are packed as NHWC 16-bit lanes.
- Opcode-specific wrapper paths populate subsets of `src0_tfr`, `dst_tfr`,
  `pdr`, `swr`, `dims`, and end fields.
- Stride/iteration fields use byte strides and logical iteration counts.
- `GatherScatter` uses a byte count. `Memset` takes an element count while its
  stride fields are bytes; packet comments alone do not collapse those units.

Production opcode legality and byte-count interpretation are owned by
`tasks/11` and `tasks/14`.

## 8. Wrapper Call-Shape Evidence Boundary

Sections 3 through 7 and the generated annex record factory/table lifecycles,
argument order, and argument-to-field mappings observed in the snapshot. They do
not prescribe a production wrapper-first entry strategy, direct packet builder,
wait placement, or golden-test suite. Those choices and their verification are
owned by `tasks/11`, `tasks/14`, and `tasks/16`.

## 9. CSR and SCALAR

CSR helpers read NCC CSR register offset `0x740`.

| API | behavior |
|---|---|
| `TsmGetCsrTaskstatus()` | Reads `getreg(0x740)` and returns bit 8. |
| `TsmGetCsrIbcounter()` | Reads `getreg(0x740)` and returns bits 0..7. |
| `TsmWaitfinish()` | Loops until task status is `1`; no timeout. |
| `*_bywork(workerid)` | Uses `get_ncc_reg(workerid, 0x740)`. |

CSR exception register offsets from `instr_def.h`:

| offset | meaning |
|---:|---|
| `0x740` | IB counter bits 0..7 and task done bit 8 |
| `0x750` | exception summary |
| `0x760` | priority |
| `0x770` | exception mask/update/clear |
| `0x780` | serial mode |

Exception bit groups are `SCALAR [7:0]`, `CT [15:8]`, `NE [23:16]`,
`RDMA [31:24]`, `WDMA [39:32]`, `TDMA [47:40]`.

`__execute_sc(SC_Param *)` only clears the first 12 bytes of the packet and
returns. There is no scalar register emission in the current `libinstr_tx81.a`;
the snapshot therefore provides no executable SCALAR evidence. Production
acceptance is owned by `tasks/11` and `tasks/14`.

## 10. Kcore DTE Semantics

Base address: `KUIPER_DTE_BASE = 0x400000`.

Block register address:

```c
block_base = 0x400000 + dte_index * 0x200;
```

Relevant block fields:

| offset | field |
|---:|---|
| `0x000` | source low |
| `0x004` | source high |
| `0x008` | destination 0 low |
| `0x00c` | destination 0 high |
| `0x010` | `user_id[0]` |
| `0x014` | mode |
| `0x018` | length in bytes |
| `0x01c` | destination count |
| `0x020..0x034` | source stride/iteration triples |
| `0x038` | command valid trigger |
| `0x040` | DMA status |
| `0x1e0..0x1f4` | destination stride/iteration triples |

DTE mode bits:

| bit/range | meaning |
|---|---|
| `0..1` | low-level mode bits |
| `4` | memory bypass |
| `8` | scatter/gather flag |
| `16` | dimension flag |
| `24` | output slice flag |

`user_id` bits:

| bit/range | meaning |
|---|---|
| `0..5` | stream id |
| `6` | early complete |
| `7` | target NPU |
| `8` | switch DDR |
| `9` | RV-N |
| `10..14` | packet id |
| `15` | stream transaction |

`tx8_deps` Kcore/direct-DTE software modes:

| value | mode |
|---:|---|
| 0 | unicast |
| 1 | scatter |
| 2 | broadcast |
| 3 | shuffle |
| 4 | RDMA |
| 5 | WDMA |
| 6 | DDR-to-DDR U2U |
| 7 | DDR-to-DDR shuffle |

Evidence boundary: this software mode enum is broader than the KMD register helper
path.  KMD declares a driver enum with `gather=4`, but its register path writes
a 2-bit `mode` field and only dispatches unicast/scatter/broadcast/shuffle.
The snapshot therefore does not prove KMD `gather=4` as a raw register mode.
Raw non-unicast acceptance belongs to the numbered communication/target designs
and board gates.

### DTE APIs

| API | behavior |
|---|---|
| `kuiper_dte_set_src_mode(dte, src, len, shuffle_cfg)` | Writes source low/high, length bytes, `dest_num=0`; if shuffle config exists, writes three source stride/iteration pairs. Nonzero iterations are stored as `iteration - 1`. |
| `kuiper_dte_set_dst_info(dte, dst, shuffle_cfg)` | Writes destination 0 low/high; if shuffle config exists, writes destination stride/iteration pairs and sets mode bit 24; otherwise clears bit 24. |
| `kuiper_dte_trig_send(dte)` | Writes `1` to offset `0x38`. |
| `kuiper_dte_check_dma_done(dte)` | Reads status offset `0x40`; returns `1` while busy, returns `0` when done without error, returns `-11` when done with error bit 8 set, and clears done by writing `1` back. |
| `mod_kuiper_dte_alloc(high_perf)` | Allocates a DTE node from context; high-performance mode selects the dedicated node if free. Returns null if none available. |
| `mod_kuiper_dte_config_src_and_dst(node,tile,src,dst,len,shuffle)` | Null node returns `-11`; RDMA passes `shuffle` to source config, WDMA passes `shuffle` to destination config; destination is ORed with `tile_logic_id << 40`. |
| `mod_kuiper_dte_trig_send(node)` | Marks node transfer state and calls hardware trigger. |
| `mod_kuiper_dte_check_send_status(node)` | Returns `-16` if context is not initialized; otherwise returns busy/done/error from DTE status and updates packet counters on done. |
| `mod_kuiper_dte_release(node)` | Clears DMA status, marks node free, and updates free count. |

Kcore DTE call shape:

```c
mod_kuiper_dte_node_t *node = mod_kuiper_dte_alloc(is_high_performance);
mod_kuiper_dte_config_src_and_dst(
    node, tile_logic_id, src_addr, dst_addr, data_len, shuffle_cfg);
mod_kuiper_dte_trig_send(node);
while (mod_kuiper_dte_check_send_status(node) == 1) {
    /* busy */
}
mod_kuiper_dte_release(node);
```

Packet counter auto-update writes:

```c
update_word = 1 | (packet_id << 4) | (stream_id << 12);
```

Local base is `0x670000`. If the destination config has switch-DDR set, the
remote packet counter base is:

```c
remote_base = 0x670000 + ((dst_tile + 0x10000) << 23);
```

## 11. Stream FSM and Mailbox Semantics

Stream FSM MMIO base from disassembly: `0x620000`.

| register | offset | meaning |
|---|---:|---|
| `STREAM_CFG[64]` | `0x0e8` | packet length bits 0..23, max packet id bits 24..28, enable bit 31 |
| `STREAM_BASE_ADDR_CHK_EN[2]` | `0x1e8` | stream base address check enable bitmaps |
| `STREAM_BASE_ADDR[64][2]` | `0x1f0` | low/high stream base address |
| `STREAM_STA[2]` | `0x6a8` | stream ready status |
| `PACKET_STA[64]` | `0x7a8` | packet ready bitmap |
| `PACKET_CNT[2048]` | `0x8a8` | packet receive length/count indexed as `stream_id*32+packet_id` |

Kcore stream IDs:

| group | range |
|---|---|
| DDR streams | `0..31` |
| SRAM streams | `32..63` |
| Kcore DDR streams | `28..31` |
| Kcore SRAM streams | `60..63` |

`kuiper_streamfsm_alloc(type, fsm_id, stream_id, addr, packet_size, packet_cnt)`
rejects `type > 1` and `fsm_id > 3`. It writes:

```c
STREAM_CFG[stream] =
    (packet_size & 0x00ffffff) |
    (((packet_cnt - 1) & 31) << 24) |
    0x80000000;
```

`mod_kuiper_streamfsm_online` rejects null config, `packet_cnt > 32`, and
`packet_size > 0x800000`, then calls `kuiper_streamfsm_alloc` and marks the
config online. `offline` releases and clears online state. Packet APIs reject
`stream_id > 63` and packet IDs above 31.

Stream runtime payload:

| payload word | content |
|---|---|
| `payload[0]` | tile X/Y in low bits, `core_id << 16`, `op_type << 24`, `stream_id << 32` |
| `payload[1]` | stream address |
| `payload[2]` | preload packet count |
| `payload[3]` | zero |

Stream operations:

| op | value |
|---|---:|
| online | 0 |
| offline | 1 |
| wait stream | 2 |
| request stream | 3 |
| pop stream | 4 |
| push stream | 5 |
| online reply | 6 |
| offline reply | 7 |
| wait reply | 8 |
| request reply | 9 |

`SendMailbox` computes TX channel as `channel_id + 4`, acquires one TX window,
sets payload register count to 8, sends via `mod_mailbox_tx_msg`, then releases
the window. The remote flag selects the remote/local target-channel bit.
Wrapper stream functions return `0` after calling `SendMailbox`; they do not
surface the mailbox send result.

Mailbox TX/RX hardware details:

| path | recovered behavior |
|---|---|
| TX acquire | `mod_mailbox_aquire_tx_win(ch,&win)` reads TX channel lock at `0x640000`; if lock bit 0 is set it returns `-12`, otherwise returns window id bits 4..6. |
| TX release | `mod_mailbox_release_tx_win(ch,win)` writes `1 << win` to the TX channel lock register. |
| TX message | `mod_mailbox_tx_msg(ch,win,config,remote,param)` writes MSI target address, up to 8 payload registers, sets channel control/status to send, and polls channel status. Status `5` is the observed success terminal state; status `6` follows the retry/log path; other nonzero states are treated as failures by the driver loop. |
| RX pending | `mod_mailbox_rx_check_msg_cnt(ch)` reads RX info count and returns pending request count bits 0..5. |
| RX pop | `mod_mailbox_rx_pop_msg(ch,msg,size)` copies 8 payload words, sets the read-done bit, and waits for hardware to clear it. |

Mailbox config structures map directly to the register blocks:

- TX channel lock has an 8-bit lock bitmap.
- TX channel control has a 2-bit send command.
- TX channel status exposes a 24-bit status field.
- TX MSI address carries target base address, local/remote-to-self bit, target
  channel, and interrupt target core.
- RX info exposes pending request count bits 0..5, read-done bit 12, and RX fail
  counter bits 16..31.

## 12. PMU and Profiling Semantics

PMU base registers:

| PMU block | base | semantic use |
|---|---:|---|
| DTE PMU | `0x400000` + offsets below | DTE work status, success/fail, data, idle, and execution counters. |
| SPM PMU | `0x580000` | SPM LSU/DTE path counters. |
| NCC PMU | `0x590000` | NCC unit counters and user timers. |
| TMNOC PMU | `0x30700000`, `0x30b00000` | Base known; detailed decoded records not recovered as compiler-facing ABI. |

DTE PMU base is `0x400800`.

| register | offset |
|---|---:|
| enable | `0x800` |
| clear | `0x804` |
| work status | `0x808` |
| total clock low/high | `0x80c` / `0x810` |
| channel 0 success | `0x814` |
| channel 1 success | `0x818` |
| command fail counters | `0x81c` |
| channel 0 transfer data low/high | `0x820` / `0x824` |
| channel 1 transfer data low/high | `0x828` / `0x82c` |
| channel 0 unaligned burst low/high | `0x830` / `0x834` |
| channel 1 unaligned burst low/high | `0x838` / `0x83c` |
| channel 0 idle low/high | `0x840` / `0x844` |
| channel 1 idle low/high | `0x848` / `0x84c` |
| all idle low/high | `0x850` / `0x854` |
| channel 0 exec time low/high | `0x858` / `0x85c` |
| channel 1 exec time low/high | `0x860` / `0x864` |

Recovered helpers combine split 32-bit counters using inconsistent low/high or
high/low read orders; no universal high-low-high retry or latch protocol is
implemented in the vendored header. Stable 64-bit sampling therefore remains a
measurement gate rather than a recovered guarantee. Host profiling uses dyn TLV
`D_PROF_CFG` and `ProcessProfData`; Kcore records use
`PmuTLVHead { uint32_t pmu_type; uint32_t length; }`.

PMU record types:

| value | type |
|---:|---|
| 0 | `SPM_DTE` |
| 1 | `SPM_LSU` |
| 2 | `DTE` |
| 3 | `NCC` |
| 4 | `NCC_CT` |
| 5 | `NCC_NE` |
| 6 | `NCC_RDMA` |
| 7 | `NCC_WDMA` |
| 8 | `NCC_TDMA` |
| 9 | `NCC_SCALAR` |
| 10 | `NCC_USER_TIME` |

The static evidence covers register locations and record shape. Actual counter
units, saturation, wrap timing, and correlation with workloads are still
`HardwareVerify`.

NCC PMU records:

- `pmu_ncc_en`, `pmu_ncc_disable`, and `pmu_ncc_clr` drive NCC PMU enable/clear
  at the NCC PMU base.
- `PMU_NCC_WORKER_OFFSET = 0x30` and worker register macros exist, but the
  aggregate CT/NE/RDMA/WDMA/TDMA record helpers currently hard-code worker-0
  instruction/blocking addresses; their commented `workeridx` offset is not
  applied. Execution-time fields also must not be assumed per-worker merely
  from the record containing a worker index.
- Unit records capture instruction count, blocking time, statistics window
  low/high, execution time low/high, and last-command info.
- User-timer register macros include additional worker addresses, but recovered
  callable helpers only implement worker 0 and worker 1; observed values are
  32-bit timer values.

SPM PMU records:

- `pmu_spm_lsu_all_*` reads SPM paths tagged in comments as RDMA port 0 and WDMA
  port 6.
- `pmu_spm_dte_all_*` reads SPM global, xbar, and DTE-related ports.
- The helper comments distinguish start/end records; the static evidence covers
  register layout, not timing accuracy.

## 13. Observed Host Runtime and Device Driver Surfaces

The public `Tsm*` functions dispatch through
`Runtime::GetInstance()->_Api()`. The concrete hardware implementation is
`RuntimeApiImplHw`; it is wrapped by logging/error/profiling decorators in some
paths. The snapshot routes device memory, host-to-device copies, bootparam
construction, dyn TLV transport, module launch, tile topology, P2P/D2D setup,
and profiling control through this layer. Wafer provider, resource, launch, and
completion ownership remains in `tasks/15`, with acceptance gates in `tasks/16`.

After the `firmware_kuiper` pass, the host/runtime split is:

| layer | snapshot evidence |
|---|---|
| HPGR `tx_runtime` | Broad CUDA-like API surface observed in the SDK snapshot and digest-qualified external V5.6 installation: device, memory, stream/event, module/kernel/model/graph, rank/tile, and P2P. Current AP/Kcore traces close ordinary-grid block distribution, type-6 graph load, type-7 run, same-BPM broadcast, and `entry(head)`. Model-manager sync/async paths wait on command-slot completion; module launch polls a device-written `completeSignal`; stream finish waits on the queued command completion object. |
| KMD UAPI | Exposes `/dev/accel/dev-N` BO/job/NPU/DTE/C2C/log/info/topology ioctl families, BAR/ATU windows, BO pools, PG tile maps, and firmware loading. The observed compute-job fence is directly signaled after MHU doorbell kick and does not prove device-side compute completion. |
| VS/old `Tsm*` | Compatibility-layer and DTE-TLV evidence. Several launch/sync/discovery paths are stub/no-op in the recovered build. |

### 13.1 Incomplete Host CModel Seams

Two different CModel interface levels are visible and must not be merged merely
because they share the word "cmodel":

1. At the instruction/packet level, `instr_operator.h` declares
   `initTsmOpPointer_cmodel` and `freeTsmOpPointer_cmodel`, while the host branch
   of `instr_adapter.h` declares `instr_tick_cc` and cycle-mode functions. No
   matching host definitions are present. The host branch of
   `op_fw_sim_if/CMakeLists.txt` creates only an include-only INTERFACE target,
   and the supplied instruction/common/Kcore archives contain RISC-V objects.
   The repo CRT calls per-operation `TsmNew*`, fills a stack `Tsm*Instr`, calls
   `TsmExecute`, and then calls `TsmDelete*`; it does not use the CModel
   operator-table initializer. Obtaining only that entry would therefore not make
   the current CRT executable on the host.
2. At the host-runtime level, x86-64 `libtx8_runtime.so` implements
   `Runtime::SetCModelHandle` by calling
   `dlopen("libcmodel_runtime_api.so", RTLD_LAZY)` and resolving these 15
   entry-point families: `SetDevice`, `SetDeviceOld`, `DeviceMalloc`,
   `DeviceFree`, `InitDevice`, `Compile`, `Launch`, `Run`, `Terminate`,
   `MemcpyH2D`, `MemcpyD2H`, `ResetDevice`, `ReleaseDevice`, `GetTileInfo`, and
   `SetTileInfo`. Recovered debug signatures use a higher-level
   `TsmDevice`/`TsmModel`/`CompileOption` C++ ABI rather than the instruction
   packet ABI.

The relevant `libcmodel_runtime_api.so`, matching host-runtime/TsmML development
headers, `libtsmml.so`, and model resources are not available as a usable
package in this checkout. A digest-qualified external V5.6 `libhpgr.so` is
available and was audited independently, but it does not provide the missing
CModel seam. Within the recovered binary, `dlsym` results
are stored in CModel function-pointer fields and the library handle is later
passed to `dlclose`; the ordinary launch path is not proven to read or call
those fields. A missing external component could
still access public fields, so the evidence supports "unproven and currently
unusable", not an absolute claim that the interface is dead.

No visible dependency, symbol, header, or build reference proves that either
seam uses SystemC or TLM. Those frameworks could be internal to the missing
library, so the implementation technology remains unknown. A vendor delivery
must include the complete matching development package, transitive libraries,
model resources, target revision, artifact input contract, numeric profile,
thread/time behavior, license, and a reproducible positive path. Until then the
high-level seam cannot be a verified package provider, and a locally implemented
low-level packet builder cannot be assumed permissible or labeled vendor-exact.
Any low-level host seam requires project-owner/legal confirmation of the actual
license terms or an independently auditable specification; independent register
or board correlation is then still required for a vendor-exact claim.

The binary exposes a boolean backend gate in `Runtime::IsTriton()`. In this
document that branch is called the active `tx*` driver backend. The symbol name
is evidence for the host implementation only; it is not a TX8 hardware block,
compiler target, or instruction semantic category.

| API | static behavior |
|---|---|
| `TsmInitRuntime(bool)` / `TsmDeInitRuntime()` | Create/destroy runtime singleton. |
| `TsmGetDeviceNum`, `TsmGetDeviceList`, `TsmGetDeviceProperties` | `RuntimeApiImplHw` stubs return `0` and do not populate capability data. |
| `TsmSetDevice` | If the active `tx*` driver backend is enabled, calls `txSetDevice(device_id)`; returns `1` on tx error. On success writes device id at `TsmDevice+0x80` and clears `+0x88`. In inactive backend mode it returns `0`. |
| `TsmSetDeviceOld` | Stub returns `0`. |
| `TsmDeviceMalloc` | Active `tx*` backend calls `txMalloc(&ptr,size)` and stores output pointer; tx error returns `1`. In inactive backend mode it returns `1`. |
| `TsmDeviceFree` | Active `tx*` backend calls `txFree(ptr)`; tx error returns `1`. In inactive backend mode it returns `0`. |
| `TsmMemcpyH2D` | Active `tx*` backend calls `txMemcpy(dst,src,size,1)`; tx error returns `1`. In inactive backend mode it returns `0`. |
| `TsmMemcpyD2H` | Active `tx*` backend calls `txMemcpy(host_dst,dev_src,size,2)`; tx error returns `1`. In inactive backend mode it returns `0`. |
| `TsmMemcpyOffsetH2D`, `TsmMemcpyOffsetD2H` | Stub-like offset helpers in the recovered hardware implementation; no required-copy side effect was observed. |
| `TsmMemcpyD2D` | Builds a `D_MEMCPY_D2D` dyn TLV and launches a Kcore DTE copy program. Uses 16 tile configs and 4 KiB chunking in recovered implementation. |
| `TsmRun` | Converts the bootparam device pointer through `Runtime::GetPhyAddr`; the active `tx*` backend calls `txLaunchModelSync(phy_bootparam)` and returns `1` on tx error. In inactive backend mode it returns `0`. HPGR model/module completion is a separate observed mechanism, not a Wafer completion-policy owner. |
| current `txLaunchKernel` | AP partitions the total grid over fixed logical tile ids `0..15`; it does not renumber by the active-tile count. On the qualified full-good snapshot grid one runs on logical tile 0 and grid-x-16 assigns pid `t` to logical tile `t`. A missing tile loses its pid rather than remapping it. |
| current `txLoadGraph` | Reads 16 tile-specific shared objects and synchronously sends a type-6 dynamic-load TLV through the outer model packet. It loads and registers entries; it does not run one inference. |
| current `txLaunchModel` | Sends a device BPM address. Type 7 looks up the type-6-registered module on each tile and calls `entry(D_BootParamHead *)`; public `txMalloc` addresses are accepted directly, but no public BPM builder or layout-version contract exists. |
| `TsmAsyncRun` | Stub returns `0`. |
| `TsmLaunch`, `TsmLaunchPg` | `RuntimeApiImplHw` returns `0` without an observed launch side effect; the return value alone is not execution evidence. |
| `TsmDeviceSynchronize`, `TsmInitDevice`, `TsmReleaseDevice` | Stub/success-return paths with no device-completion proof in this snapshot. |
| `TsmGetTileInfo` | Calls `txGetDeviceAllTileInfo(device_id,temp)` and copies 16 records of 12 bytes into `TsmTileTotalInfo`; tx error returns `1`. |
| `TsmSetTileInfo` | Copies 8 selected tile records and calls `txSetDeviceSelectedTileInfo`; tx error returns `1`. |
| `TsmProcessProfData` | Builds profiling dyn TLV, runs bootparam, then dumps profiling data depending on action. |
| `TsmSend` / `TsmRecv` | Dispatch to the txccl send/recv implementation; target-specific transport success is not established by static evidence. |
| `TsmNpuPowerOn` / `TsmNpuPowerOff` | Implemented host calls but hardware/power sequencing remains board-validated. |

Observed host-runtime call shape:

```cpp
TsmInitRuntime(true);

TsmDevice *dev = nullptr;
if (TsmSetDevice(&dev, 0, 0) != 0) abort();

uint64_t d_in = 0;
uint64_t d_out = 0;
TsmDeviceMalloc(dev, d_in, input_size);
TsmDeviceMalloc(dev, d_out, output_size);
TsmMemcpyH2D(d_in, host_input, input_size);

// Recovered TsmRun takes a device bootparam address, not a host model object.

TsmMemcpyD2H(host_output, d_out, output_size);
TsmDeviceFree(d_in);
TsmDeviceFree(d_out);
TsmReleaseDevice(dev);
TsmDeInitRuntime();
```

Compile/load path:

```cpp
TsmModel model("graph_name");
CompileOption opt = {};

if (TsmCompile(dev, model, case_dir, opt) != 0) abort();

std::vector<TsmModel*> models;
models.push_back(&model);
if (TsmLoadKernel(dev, models, nullptr) != 0) abort();
```

Recovered compile/load behavior:

- `TsmCompile` optionally calls `buildRiscvTileBin`.
- It scans `/chip_out/chip{index}/tile*`.
- It loads `config_stream.bin` and `/tileN/kcore_fw.bin`.
- `TsmCompileMultiGraph` and `TsmGraphCompile` use `/tileN/kcore_fw.so`.
- `TsmGraphCompile` assumes 16 tiles.

Snapshot limitations and design ownership:

- `GetDeviceNum/List/Properties` return without populating discovery data in the
  recovered build.
- `DeviceSynchronize`, `Launch`, `LaunchPg`, and `AsyncRun` have stub or
  success-return paths without completion evidence.
- The observed KMD compute fence is signaled after submission. HPGR command
  completion, stream/event completion, Kcore CSR waits, DTE waits, and explicit
  sync are distinct snapshot mechanisms; this file assigns none of them Wafer
  completion priority.
- `DeviceMalloc`, `MemcpyH2D`, `MemcpyD2H`, and `Run` reach their observed tx
  operations only when the snapshot's active `tx*` backend gate is enabled.
- Public host headers in `tx8_deps` are incomplete; the `firmware_kuiper` HPGR
  headers provide additional API evidence without establishing Wafer ABI
  precedence.

Provider selection, adapter boundaries, completion semantics, and board/runtime
acceptance are owned by `tasks/15` and `tasks/16`.

## 14. Observed Device Bootparam and Dynamic TLV Layout

Device bootparam head:

```c
typedef struct D_BootParamHead {
    uint32_t MaxLen;
    uint32_t LdmemLen;
    uint32_t InputNum;
    uint32_t OutputNum;
    uint32_t ParamNum;
    uint32_t reserved;
    uint64_t CacheMemLen;
    uint64_t CacheMemAddr;
    uint32_t Datalen;
    uint32_t reserved1;
    uint64_t DataAddr;
} D_BootParamHead; // size 56
```

Dynamic input/output/parameter info:

```c
typedef struct D_BootParamDyninfo {
    uint64_t addr;
    uint64_t size;
    uint32_t dtype;
    uint32_t dim;
    uint64_t shape[6];
} D_BootParamDyninfo; // size 72
```

Observed layout:

- bootparam head starts at the bootparam buffer base.
- dyninfo begins at `head + 0x38`.
- dyninfo entries are ordered as inputs, then outputs, then params.
- the legacy `Tsm*` runtime converts its allocation wrapper through
  `Runtime::GetPhyAddr` before calling `txLaunchModelSync`; this is not a rule
  for the public HPGR pointer. In current V5.6, public `txMalloc` returns the
  address used by `txMemcpy` and accepted directly as the `uint64_t` BPM device
  address by `txLaunchModel`.

Current HPGR graph-load records are:

```c
typedef struct D_GraphInfo {
    char module_name[128];
    char module_symbol[128];
    uint32_t module_size[16];
    uint64_t module_addr[16];
} D_GraphInfo; // size 448

typedef struct D_DynMods {
    uint16_t module_num;
    // six bytes of ABI padding
    D_GraphInfo graph;
} D_DynMods; // size 456

typedef struct D_GraphTLV {
    uint32_t type;
    uint32_t len;
    uint64_t dyn_mods_addr;
} D_GraphTLV; // size 16
```

For type 6, `txLoadGraph(path, symbol)` reads exactly
`path/tile0/kcore_fw.so` through `path/tile15/kcore_fw.so`, fills the 16
size/address pairs, and uses one shared name and symbol. Kcore indexes those
arrays by its own tile id, loads the tile-specific object, resolves the shared
symbol, and registers the resulting entry under the module name. The outer
host command is packet type 5, but the inner operation remains
`DYNLIB_LOAD`; this synchronous call does not run an inference.

Across two legacy host builds, the type-7 run payload is the same 456-byte
one-module `D_DynMods` with only `module_name` material, referenced by the same
16-byte TLV form with `type=7` and `len=8`. Current Kcore receives the same
bootparam pointer on every active tile, finds the locally registered entry by
name, and calls exactly `entry(D_BootParamHead *)`. V5.6 graph modules confirm
that entry contract by reading input at `+56`, output at `+128`, parameter at
`+200`, and cache address at `+32` for the one-input/one-output/one-parameter
case. Per-tile compiled constants, rather than 16 host launches, select each
tile's work.

Dynamic TLV header:

```c
typedef struct D_DynTLV {
    uint32_t type;
    uint32_t len;
} D_DynTLV;
```

Dynamic TLV types:

| value | type |
|---:|---|
| 0 | final/terminate |
| 1 | cfg PMU |
| 2 | kcore cfg |
| 3 | export SPM |
| 4 | disable calc |
| 5 | profiling config |
| 6 | dynlib load |
| 7 | dynlib run |
| 8 | dynlib unload |
| 9 | memcpy D2D |
| 10 | P2P send |
| 11 | P2P recv |
| 12 | group data dump |
| 13 | max marker |

Known payloads are defined in the generated annex: terminate, kcore config,
profiling config, group data dump, `D_DteCfgList`, `TileMappingTable`,
`D_DynTLV_Cfgpmu`, and related runtime structs.

Evidence boundary: the structures and call chain above are statically closed
for the qualified V5.6 binary and cross-checked against legacy builders and
installed device modules. The public header still provides no BPM builder or
layout-version contract. Wafer current schema-v5 publication (the model
BootParam ABI was first introduced in schema-v4) and its board provider
now materialize typed graph I/O ordinals, checked sizes/offsets, nested
allocation lifetimes, module-name identity, and artifact export/readback as
the explicit `tx81-model-bootparam-v1` launch ABI. That ABI remains qualified
only for the pinned V5.6 runtime digest; it is neither an opaque sidecar nor a
silent substitute for either kernel launch ABI.

## 15. Evidence-to-Design Handoff

This evidence ledger does not define a minimum production verifier checklist,
packet acceptance set, compiler entry strategy, runtime provider policy, or
golden-test suite. The recovered facts above feed these numbered owners:

- `tasks/09` and `tasks/11`: memory reservations, operand ranges, instruction
  legality, shape/unit relations, and completion-relevant instruction facts.
- `tasks/13`: accepted physical transport and stream/DTE relations.
- `tasks/14`: target command ABI, CRT surface, packet emission, and target
  artifact checks.
- `tasks/15`: package, provider, resource, launch, and runtime completion
  semantics.
- `tasks/16`: negative, integration, board, and hardware-validation gates.
- `tasks/17`: target execution model, host CModel seam selection, packet/model
  provenance, and model/board correlation.

## 16. Remaining Hardware Validation

These are intentionally not claimed as statically complete:

- PMU counter units and workload correlation.
- MHU/mailbox retry, interrupt, and failure behavior.
- NPU power on/off sequencing and error propagation.
- Actual floating point exception bits and arithmetic edge cases.
- Host runtime behavior when the closed device driver returns target-specific
  errors.
- Complete vendor host CModel package, matching headers/resources, reachable
  execution path, internal framework, packet provenance, and numeric/time
  contract.
- Raw DTE multi-destination broadcast/scatter/shuffle policies outside the
  documented unicast/RDMA/WDMA helper path.
- Exact SPM bank mapping and 64 KiB parallel allocator coloring; the snapshot
  does not prove a 64 KiB hard ABI. Allocation policy belongs to `tasks/09`, and
  hardware validation belongs to `tasks/16`.
