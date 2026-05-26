# TX8 Interface, Runtime, and Hardware Contract

This is the canonical implementer-facing contract for the reversed TX8
dependency package. It records what each compiler-facing API family writes into
instruction packets, what each instruction class is allowed to contain, what the
TX8 hardware-facing register interfaces mean, and how the host runtime drives
device memory, bootparams, dyn data, topology, launch, and profiling.

Runtime is treated as a first-class host/driver layer. Backend branch names
recovered from the binary are implementation gates, not the organizing model for
the reverse engineering. Raw tx8-deps evidence and broader dependency notes
remain in `tx8-deps-reverse-engineering-reference.md`; generated signatures and
struct shapes remain in `tx8-api-struct-contract-annex.md`. Full SDK/KMD/HPGR
evidence from `firmware_kuiper` is summarized in
`firmware-kuiper-runtime-hardware-analysis.md` and should be used for host
runtime, driver, BO/BAR/ATU, PG, C2C, and completion semantics.

Evidence used in this pass:

- Headers: `/root/dlc_dev/tx8_deps/include/instr_def.h`,
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
argument-to-field behavior, verifier rules, register meanings, host runtime API
semantics, and stub versus real driver behavior. This file closes that layer for
compiler-facing and runtime-facing paths.

| area | static semantic status | remaining non-static work |
|---|---|---|
| `libinstr_tx81.a` wrapper APIs | Covered by function-family rules below and packet/register maps. | Hardware error behavior for invalid packets still needs board tests. |
| `TsmExecute` dispatch | Covered: only instruction types 0..4 dispatch. | None for dispatch itself. |
| CT/NE/RDMA/WDMA/TDMA packet layout | Covered from header offsets and execute disassembly. | Exact arithmetic corner cases, NaN/overflow flags, and timing require hardware. |
| SCALAR | Covered as reserved/stub. | Needs a real scalar sample before compiler lowering. |
| Kcore DTE/stream/mailbox | Register protocol and payload format covered. | Multi-destination DTE policies, mailbox failure recovery, and timing need board tests. |
| PMU/profiling | Register and TLV shape covered. | Counter units, wrap edge cases, and event accuracy need board tests. |
| Host runtime / driver layer | Exported `Tsm*` signatures, implemented/stub behavior, bootparam/dyn-data paths, launch/copy/topology/profiling calls covered; HPGR/KMD pass clarifies primary `tx_runtime` ABI, BO/BAR/ATU, KMD UAPI, PG, and completion semantics. | Closed driver availability, target-specific launch success, board services, and hardware error propagation need target validation. |

Symbol coverage remains measured by `tx8-symbol-coverage-matrix.csv`. The
important practical result is that all compiler-facing instruction families now
have a documented packet contract and verifier rule set. The remaining
`HardwareVerify` rows are hardware-observable state, timing, PMU, MHU, power, or
closed device-driver paths.

### Interface Semantics Inventory

The reverse-engineered interface surface is split into several layers. All are
in scope; none of them should be inferred from a backend name alone.

| layer | interface family | semantics recovered here |
|---|---|---|
| Instruction wrapper API | `TsmElemWise`, CT relation/logic/reduce/convert/peripheral helpers | Function family, opcode range, address-versus-scalar operand role, dtype storage size, bool packing, end-field policy, writeback behavior, and verifier constraints. |
| Instruction wrapper API | `TsmConv`, `TsmDepthWiseConv`, `TsmGemm` | Argument-to-field mapping for input/weight/output/psum/bias/scale/sparse/quant/pad/stride/dilation/GEMM MKN/batch/trans flags, plus shape and quant bounds. |
| Instruction wrapper API | `TsmRdma`, `TsmWdma`, `TsmDataMove`, `TsmPeripheral` | DDR/SPM direction, stride and iteration units, `iteration-1` encoding, TDMA opcodes, byte-oriented memset/gatherscatter semantics, and register window offsets. |
| CSR/reserved API | CSR helpers and `I_SCALAR` | CSR status bit meanings and worker addressing are recovered; scalar packet execution is a stub and must stay reserved. |
| Kcore hardware API | DTE | Register fields, mode/user_id bits, high-level modes, source/destination setup, shuffle stride encoding, trigger, done/error return codes, and packet-counter update format. |
| Kcore hardware API | Stream FSM and mailbox | Stream config layout, packet counters, online/offline/request/push/pop payloads, mailbox TX/RX window protocol, payload register count, and observed status handling. |
| Profiling API | PMU helpers and host profiling dyn data | DTE/SPM/NCC PMU bases and record types, stable 64-bit read policy where present, host `D_PROF_CFG` control path, and remaining hardware validation items. |
| Host runtime API | `Tsm*` runtime exports | Device selection, memory allocation/free, H2D/D2H/D2D/P2P copies, bootparam launch, kernel load/unload, tile topology, power hooks, profiling, return code semantics, and stub boundaries. |
| HPGR/KMD API | `tx_runtime.h`, KMD UAPI | CUDA-like device/memory/stream/event/model/module API, command completion, BO pools, job/DTE/C2C ioctl contracts, PG tile map, and BAR/ATU address-space ownership. |

Interface units are explicit in the relevant sections: tensor element counts for
CT/NE logical work, packed bytes for bool storage, byte strides for DMA/TDMA/DTE
stride fields, byte lengths for DTE transfers, and device physical addresses for
runtime bootparams and dyn-data buffers.

## 2. TX8 Hardware Surface Map

The hardware surface recovered from headers and object code is broader than the
host runtime, but the runtime remains the host-side owner of device memory,
bootparams, dyn TLVs, and launch. The table below is the address-level map used
by the Kcore and instruction libraries.

| block | base | recovered semantic status |
|---|---:|---|
| L1 SPM | `0x000000` | Tensor/local SRAM for NCC and Kcore. Size is `0x300000`; compiler must reserve `0x2f0000..0x2fffff` for Kcore/runtime. |
| NCC instruction MMIO | `0x01000000` | CT/NE/RDMA/WDMA/TDMA register windows, three worker windows spaced by `0x100000`; fully mapped for compiler-emitted packets. |
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

For production compiler purposes the hardware is split into these contracts:

- NCC instruction contract: `TsmExecute` packet types 0..4 and the CT/NE/DMA/TDMA
  register windows.
- Kcore movement/stream contract: DTE, stream FSM, mailbox, and packet counter
  update registers.
- Profiling contract: NCC, DTE, and SPM PMU record formats and register bases.
- Board/runtime dependency boundary: MHU, power/CRG, TMNOC, and system config
  bases are identified, but their bit-level behavior is not compiler-facing yet.

## 3. Common Instruction Model

All instruction packets start with `uint32_t inter_type`. Low bits select the
functional unit; bits 8..9 encode worker selection for NCC register windows.

| value | type | compiler status |
|---:|---|---|
| 0 | `I_CGRA` / CT | valid |
| 1 | `I_NEUR` / NE | valid |
| 2 | `I_RDMA` | valid |
| 3 | `I_WDMA` | valid |
| 4 | `I_TDMA` | valid |
| 5 | `I_SCALAR` | reserved/stub |
| 6 | `I_DTE` | not dispatched by `TsmExecute` |
| 7 | `I_CSR` | not dispatched by `TsmExecute` |

Worker values are `I_WORKER0 = 0x0000`, `I_WORKER1 = 0x0100`, and
`I_WORKER2 = 0x0200`. Execute helpers derive the NCC worker window from
`(inter_type >> 8) & 3`, normalized to three workers, and add `worker * 0x100000`
to the NCC base `0x01000000`.

`TsmExecute(void *instr)` reads `*(uint8_t *)instr` and dispatches only values
0..4. Values greater than 4 return `1` without running scalar, DTE, or CSR
execution. Production lowering must not emit `I_SCALAR`, `I_DTE`, or `I_CSR`
through `TsmExecute`.

### Compiler Entry Strategy and Wrapper Lifecycle

The safe first backend should generate Kcore C/C++ that calls the vendor wrapper
API. Raw register emission is possible, but wrapper-first gives packet
compatibility and lets golden tests compare raw builder packets against vendor
packets.

Wrapper objects are C structs of function pointers, not ABI-stable C++ classes.
Treat constructors/destructors as factory functions around those tables:

```c
TsmArith *arith = TsmNewArith();
TsmArithInstr instr = {0};
arith->AddVV(&instr, src0, src1, dst, elem_count, RND_NEAREST_EVEN, Fmt_FP16);
TsmExecute(&instr);
TsmWaitfinish();
TsmDeleteArith(arith);
```

The same lifecycle applies to `TsmConv`, `TsmDepthwiseConv`, `TsmGemm`,
`TsmRdma`, `TsmWdma`, `TsmDataMove`, `TsmPeripheral`, and the CT-family wrapper
tables listed in the generated annex.

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

Bool paths are special. The wrappers either compute packed byte counts directly
or rewrite DMA format to byte format when moving packed bool storage. A verifier
must treat `elem_count` as logical elements and storage as packed bytes.

### Address Domains

Instruction memory operands are NCC-visible addresses. The wrapper headers
define:

| domain | range |
|---|---|
| SPM | `0x00000000..0x002effff` |
| Kcore reserved SPM | `0x002f0000..0x002fffff` |
| DDR | `>= 0x280000000` |

CT, NE, and TDMA operands are SPM addresses. RDMA source is DDR and destination
is SPM. WDMA source is SPM and destination is DDR. The compiler must reserve the
Kcore SPM window and must not allocate user tensors there.

### Instruction Packet Struct Offsets

The generated annex gives the C struct shapes. These byte offsets are the ABI
facts that raw packet builders and wrapper-golden tests must match.

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
| `V_VuV` | `AddVuV`, relation/logic unit-vector variants | `src0` uses `elem_count`; `src1` is a short/unit vector using `unit_elem_count`; `unit_elem_count` must be `1..64`. |
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

### CT Verifier Rules

- `cmd_valid` must be set by wrappers or raw builder before execution.
- `opcode` must be a known opcode in the CT opcode table.
- `src0_format` must be a known `Data_Format`; bool requires packed sizing.
- `unit_elem_count` must be `1..64` when used.
- `rnd_mode` must be `0..4` for functions that expose rounding.
- `dims` must be one of the documented dimensions for the operation. Reduce
  supports `C=0`, `W=1`, `H=2`, and `HW=4`.
- All SPM operands and computed `*_end` fields must remain in usable SPM and
  outside the reserved Kcore window.
- Scalar variants must validate the scalar immediate width according to source
  format; do not treat `src1` as a pointer.

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
`out_end`, and `sparse_end`. Raw builders must either reproduce the same end
calculation or call the wrapper API and execute helper.

### NE Constraints

- `type` values: `0=conv`, `1=depthwise conv`, `2=backward conv`, `3=gemm`.
- `tfr_0` / `tfr_1` pack `n,h,w,c` as 16-bit lanes. `n/h/w` range is
  `1..4096`; `c` range is `1..16384`.
- Pad/unpad lanes are `0..1023`.
- `Kx/Ky` range is `1..255`; `Sx/Sy` range is `1..1023`.
- Dilation lanes are `1..1023`.
- GEMM `M/N` are 16-bit; `K` range is `1..16384`; batches are `1..4096`.
- Quant zero-points are `0..255`; quant shifts are `0..31`.
- All enabled optional operands (`psum`, `bias`, scales, sparse index) require
  valid SPM ranges and end fields.

## 6. RDMA and WDMA Semantics

Packet type: `DMA_Param`.

`TsmRdma` is DDR to SPM. `TsmWdma` is SPM to DDR.

| API | behavior |
|---|---|
| `Rdma::AddSrcDst(src,dst,fmt)` | Sets `inter_type=I_RDMA`, `cmd_valid=1`, `src=src` DDR, `dst=dst` SPM, `format=fmt`. |
| `Wdma::AddSrcDst(src,dst,fmt)` | Sets `inter_type=I_WDMA`, `cmd_valid=1`, `src=src` SPM, `dst=dst` DDR, `format=fmt`. |
| `ConfigStrideIteration(elem_count, stride0, iteration0, stride1, iteration1, stride2, iteration2)` | Stores byte strides and stores each logical iteration as `iteration - 1`; zero logical iteration is invalid. |
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

DMA verifier rules:

- RDMA `src` must be DDR and `dst` must be SPM.
- WDMA `src` must be SPM and `dst` must be DDR.
- `elem_count > 0`.
- Logical iterations must be nonzero before wrapper conversion to `iteration-1`.
- Strides are bytes, not elements.
- `Fmt_BOOL` is packed; wrappers convert byte counts with
  `ceil(elem_count/8)` and may store byte format for the actual transfer.

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
byte number. Production lowering should convert the user element count to byte
count consistently before raw packet construction, or call the wrapper.

`GatherScatter` writes `elem_count=size` in bytes, copies source and destination
stride/iteration triples, and computes only `src0_end` and `dst_end`;
`src1_end` is zero.

TDMA verifier rules:

- All TDMA data operands are SPM addresses.
- Shapes are packed NHWC 16-bit lanes.
- `src0_tfr`, `dst_tfr`, `pdr`, `swr`, `dims`, and end fields must match the
  selected opcode.
- Stride/iteration fields use byte strides and logical iteration counts.
- For byte-oriented TDMA operations (`Memset`, `GatherScatter`), treat
  `elem_count` as byte count in the final packet.

## 8. Production Wrapper Recipes

These examples preserve the concrete call shape from the removed standalone
production contract. They are examples of the wrapper-first path, not a
replacement for the per-field semantics above.

Conv/depthwise:

```c
TsmConv *conv = TsmNewConv();
TsmNeInstr instr = {0};
conv->AddInput(&instr, x_spm, (Data_Shape){n,h,w,c}, Fmt_FP16);
conv->AddWeight(&instr, w_spm, (Data_Shape){kx,ky,f,c}, Fmt_FP16);
conv->AddBias(&instr, bias_en, bias_spm);
conv->AddOutput(&instr, out_spm, (Data_Shape){n,oh,ow,f}, Fmt_FP16);
conv->SetOpType(&instr, 0);
conv->SetPads(&instr, top, bottom, left, right);
conv->SetKernelStrides(&instr, kx, ky, sx, sy);
conv->SetDilations(&instr, dx, dy);
conv->SetQuant(&instr, q0, q1, zp_pre, zp_cur);
TsmExecute(&instr);
TsmWaitfinish();
TsmDeleteConv(conv);
```

GEMM:

```c
TsmGemm *gemm = TsmNewGemm();
TsmNeInstr instr = {0};
gemm->AddInput(&instr, left_spm, right_spm, Fmt_FP16);
gemm->ConfigMKN(&instr, M, K, N);
gemm->ConfigBatch(&instr, left_batch, right_batch);
gemm->AddOutput(&instr, out_spm, Fmt_FP16);
gemm->SetTransflag(&instr, left_trans, right_trans);
gemm->SetPsum(&instr, psum_en, psum_spm, Fmt_FP16);
TsmExecute(&instr);
TsmWaitfinish();
TsmDeleteGemm(gemm);
```

RDMA/WDMA pseudo sequence:

```text
TsmRdma *rdma = TsmNewRdma();
TsmRdmaInstr rdma_instr = {0};
rdma_configure_contiguous(rdma, &rdma_instr, ddr_src, spm_dst, elem_count, Fmt_FP16);
TsmExecute(&rdma_instr);
TsmWaitfinish();
TsmDeleteRdma(rdma);

TsmWdma *wdma = TsmNewWdma();
TsmWdmaInstr wdma_instr = {0};
wdma_configure_contiguous(wdma, &wdma_instr, spm_src, ddr_dst, elem_count, Fmt_FP16);
TsmExecute(&wdma_instr);
TsmWaitfinish();
TsmDeleteWdma(wdma);
```

For 3D DMA, call `AddSrcDst` first and then `ConfigStrideIteration`; stride
arguments are bytes and iteration arguments are logical counts before the wrapper
stores `iteration - 1`.

CT wrapper families follow these call shapes:

| family | API shape | notes |
|---|---|---|
| unary vector | `OpVV(instr, src, dst, elem_count, fmt)` | uses `src0`, `dst0`, `elem_count` |
| binary vector | `OpVV(instr, src0, src1, dst, elem_count, rnd, fmt)` | uses `src0`, `src1`, `dst0` |
| vector-scalar | `OpVS(instr, src0, const_value, dst, elem_count, rnd, fmt)` | scalar immediate goes through the `src1` field path |
| unit vector | `OpVuV(instr, src0, src1, dst, elem_count, unit_elem_count, rnd, fmt)` | `unit_elem_count <= 64` |
| loop unit vector | `OpVuVLoop(instr, src0, src1, dst, elem_count, unit_elem_count, full_elem, full_unit_elem, rnd, fmt)` | fills full count fields |
| reduce | `Reduce*(instr, src, dst, dim, shape, fmt)` | `dim` must be `0/1/2/4` |
| pool/unpool | `Pool*(instr, src, src_shape, dst, pad, swr_shape, fmt)` | uses CT tensor transform registers |
| convert | `SRC_DST(instr, src, dst, elem_count, optional rnd/zp)` | function name encodes source/destination format |

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
SCALAR must remain reserved.

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

Layering rule: this software mode enum is broader than the KMD register helper
path.  KMD declares a driver enum with `gather=4`, but its register path writes
a 2-bit `mode` field and only dispatches unicast/scatter/broadcast/shuffle.
Compiler-facing code should not treat KMD `gather=4` as a confirmed raw
register mode.  Raw non-unicast communication still needs a Wafer ABI and board
tests.

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

64-bit PMU reads use a high-low-high stability pattern where implemented.
Host profiling uses dyn TLV `D_PROF_CFG` and `ProcessProfData`; Kcore records
use `PmuTLVHead { uint32_t pmu_type; uint32_t length; }`.

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

The static contract covers register locations and record shape. Actual counter
units, saturation, wrap timing, and correlation with workloads are still
`HardwareVerify`.

NCC PMU records:

- `pmu_ncc_en`, `pmu_ncc_disable`, and `pmu_ncc_clr` drive NCC PMU enable/clear
  at the NCC PMU base.
- `PMU_NCC_WORKER_OFFSET = 0x30`; per-worker CT/NE/RDMA/WDMA/TDMA counters are
  addressed by adding worker offset where helpers annotate it.
- Unit records capture instruction count, blocking time, statistics window
  low/high, execution time low/high, and last-command info.
- User timers exist for worker 0 and worker 1 in the recovered helpers and are
  32-bit timer values.

SPM PMU records:

- `pmu_spm_lsu_all_*` reads SPM paths tagged in comments as RDMA port 0 and WDMA
  port 6.
- `pmu_spm_dte_all_*` reads SPM global, xbar, and DTE-related ports.
- The helper comments distinguish start/end records; the static contract is
  register layout, not timing accuracy.

## 13. Host Runtime and Device Driver Semantics

The public `Tsm*` functions dispatch through
`Runtime::GetInstance()->_Api()`. The concrete hardware implementation is
`RuntimeApiImplHw`; it is wrapped by logging/error/profiling decorators in some
paths. This layer is important because it owns device memory, host-to-device
copies, bootparam construction, dyn TLV transport, module launch, tile topology,
P2P/D2D setup, and profiling control. It sits above the NCC/DTE/stream/mailbox
hardware contracts rather than replacing them.

After the `firmware_kuiper` pass, the host/runtime split is:

| layer | current contract |
|---|---|
| HPGR `tx_runtime` | Primary CUDA-like host ABI in the SDK: device, memory, stream/event, module/kernel/model/graph, rank/tile, and P2P.  Model-manager sync/async paths wait on command-slot completion; module launch polls a device-written `completeSignal`; stream finish waits on the queued command completion object. |
| KMD UAPI | Owns `/dev/accel/dev-N` BO/job/NPU/DTE/C2C/log/info/topology ioctl families, BAR/ATU windows, BO pools, PG tile maps, and firmware loading.  Current compute-job fence is directly signaled after MHU doorbell kick and is not proof of device-side compute completion. |
| VS/old `Tsm*` | Compatibility layer over HPGR plus DTE TLV evidence.  Several launch/sync/discovery paths are stub/no-op in the recovered build. |

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
| `TsmMemcpyOffsetH2D`, `TsmMemcpyOffsetD2H` | Stub-like offset helpers in hardware impl; do not rely on them for required copies. |
| `TsmMemcpyD2D` | Builds a `D_MEMCPY_D2D` dyn TLV and launches a Kcore DTE copy program. Uses 16 tile configs and 4 KiB chunking in recovered implementation. |
| `TsmRun` | Converts bootparam device pointer through `Runtime::GetPhyAddr`; active `tx*` backend calls `txLaunchModelSync(phy_bootparam)` and returns `1` on tx error. In inactive backend mode it returns `0`.  HPGR native model/module completion is the higher-priority completion evidence when available. |
| `TsmAsyncRun` | Stub returns `0`. |
| `TsmLaunch`, `TsmLaunchPg` | Stub returns `0` in `RuntimeApiImplHw`; do not use as proof of execution. |
| `TsmDeviceSynchronize`, `TsmInitDevice`, `TsmReleaseDevice` | Stub/success-return paths; not a true fence unless board validation proves target-specific behavior. |
| `TsmGetTileInfo` | Calls `txGetDeviceAllTileInfo(device_id,temp)` and copies 16 records of 12 bytes into `TsmTileTotalInfo`; tx error returns `1`. |
| `TsmSetTileInfo` | Copies 8 selected tile records and calls `txSetDeviceSelectedTileInfo`; tx error returns `1`. |
| `TsmProcessProfData` | Builds profiling dyn TLV, runs bootparam, then dumps profiling data depending on action. |
| `TsmSend` / `TsmRecv` | Dispatch to txccl send/recv implementation; exact transport success requires runtime target. |
| `TsmNpuPowerOn` / `TsmNpuPowerOff` | Implemented host calls but hardware/power sequencing remains board-validated. |

Minimal host runtime recipe:

```cpp
TsmInitRuntime(true);

TsmDevice *dev = nullptr;
if (TsmSetDevice(&dev, 0, 0) != 0) abort();

uint64_t d_in = 0;
uint64_t d_out = 0;
TsmDeviceMalloc(dev, d_in, input_size);
TsmDeviceMalloc(dev, d_out, output_size);
TsmMemcpyH2D(d_in, host_input, input_size);

// Use compiled/loaded model path or explicit bootparam path here.
// TsmRun expects a device bootparam address, not a host model object.

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

Runtime verifier rules:

- Do not use `GetDeviceNum/List/Properties` as hardware discovery on this build.
- Treat `DeviceSynchronize`, `Launch`, `LaunchPg`, and `AsyncRun` as unsafe
  stubs for correctness or fencing.
- Treat KMD compute fences as submit fences in this driver snapshot; use HPGR
  command completion, stream/event completion, Kcore CSR waits, DTE waits, or
  explicit runtime sync for device-side completion.
- Require an active `tx*` device backend for real `DeviceMalloc`, `MemcpyH2D`,
  `MemcpyD2H`, and `Run`.
- Keep a narrow adapter boundary around host runtime calls because public host
  headers are incomplete in `tx8_deps`; HPGR public headers from
  `firmware_kuiper` are the higher-priority host ABI when available.

## 14. Device Bootparam and Dynamic TLV Contract

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

Layout rules:

- bootparam head starts at the bootparam buffer base.
- dyninfo begins at `head + 0x38`.
- dyninfo entries are ordered as inputs, then outputs, then params.
- runtime launch converts the device bootparam pointer through
  `Runtime::GetPhyAddr` before calling `txLaunchModelSync`.

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

## 15. Production Verifier Checklist

Minimum checks before emitting or accepting a TX8 instruction packet:

- Reject `TsmExecute` packet types outside `0..4`.
- Reject SCALAR, DTE, and CSR as executable instruction packets.
- Enforce SPM/DDR domains per operand role.
- Exclude `0x2f0000..0x2fffff` from compiler SPM allocation.
- Validate every dtype and packed bool byte count.
- Recompute or verify every `*_end` field.
- Validate CT opcodes, rounding mode, dimensions, and unit-vector counts.
- Validate NE type, shape, pad, unpad, stride, dilation, quant, batch, and GEMM
  ranges.
- Validate DMA direction, nonzero element count, byte strides, and nonzero
  logical iterations.
- Validate TDMA opcode-specific fields, byte-oriented element counts, and shape
  packing.
- Reject stream packet count above 32 and packet size above `0x800000`.
- Treat host runtime stubs as no-ops unless a board-specific integration test
  proves otherwise.

Minimum golden tests:

- wrapper-generated CT packet versus raw builder for one unary, one binary, one
  unit-vector, and one loop-vector op.
- wrapper-generated NE conv and GEMM packet field offsets.
- RDMA/WDMA contiguous end-address calculation.
- TDMA transpose, pad, and gatherscatter packet fields.
- bootparam head and dyninfo layout.
- TLV serialization parse/roundtrip.
- DTE packet counter update word.
- stream FSM payload word and packet config word.

## 16. Remaining Hardware Validation

These are intentionally not claimed as statically complete:

- PMU counter units and workload correlation.
- MHU/mailbox retry, interrupt, and failure behavior.
- NPU power on/off sequencing and error propagation.
- Actual floating point exception bits and arithmetic edge cases.
- Host runtime behavior when the closed device driver returns target-specific
  errors.
- Raw DTE multi-destination broadcast/scatter/shuffle policies outside the
  documented unicast/RDMA/WDMA helper path.
- Exact SPM bank mapping and 64 KiB parallel allocator coloring; current docs
  treat 64 KiB as a conservative policy, not a proven hard ABI.
