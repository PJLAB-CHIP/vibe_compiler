# Wafer Instruction IR Design

日期：2026-06-05；更新：2026-06-08

状态：R3.2d 设计已按 memref-backed buffer contract 重新收口；R3.2c 前置已完成，
R3.2d.1 instruction op contract 已落地，当前 active task 是 R3.2d.2 DialectConversion

本文定义 R3.2d 的 instruction-level Wafer IR。核心结论：

- 只新增 `wafer.instr.*` 硬件指令级 op。
- instruction-level buffer value 统一使用 MLIR `memref`，不再把 `!wafer.storage` 作为长期 IR
  合同。
- Wafer 的 SPM / DDR address domain 和 physical layout marker 放在 memref memory-space attr 中，
  例如 `memref<2x65xf16, #wafer.memory<spm, cx>>`。
- `Cx/NCx` 只是 Wafer physical layout marker；`C0`、storage bytes、range-end、bool bitpack 和
  256B padding 必须由统一 Wafer layout calculator 从 memref type 推导，不写进 IR 字段。
- `Cx/NCx` 不使用 MLIR memref layout slot，也不实现为 `MemRefLayoutAttrInterface`。MLIR memref
  layout slot 仍只用于 MLIR 能按 affine / strided 语义解释的普通 layout。
- 不引入 `wafer.physical_view`、`!wafer.physical_memref`、side descriptor value、SPM offset、DDR
  allocation policy、raw packet 或 C ABI call。

`wafer.instr` 的作用是把 target-abstract tile-region op 变成可执行硬件动作，并让下游能从
memref SSA、Wafer memory attr、op operands、attrs、MemoryEffects 和显式 drain 直接推导 placement
输入。它不是另一层 buffer IR。

当前实现状态：

- 仓库代码当前已落地 `wafer.instr.local_drain`，以及
  `wafer.instr.rdma`、`wafer.instr.wdma`、`wafer.instr.gather_scatter`、`wafer.instr.fill`、
  `wafer.instr.elementwise`、`wafer.instr.reduce`、`wafer.instr.convert` 和 `wafer.instr.gemm`
  的 ODS、verifier、MemoryEffects、`WaferInstructionOpInterface` 和 lit/unit 覆盖。
- `wafer.instr.*` op 只读写 Wafer-tagged memref，不产生 buffer result，不携带 SPM offset、
  worker id、raw packet field 或 C ABI 字段。
- R3.2d.2 仍需实现 target-abstract tile-region op 到这些 instruction op 的 DialectConversion。
- R3.2c 已产出 memref-backed `wafer.tile.region`；R3.2d 必须基于该 unplaced Wafer-tagged memref
  graph 做 instruction lowering，不能再引入 storage/buffer IR 层。
- R3.2c 已支持 `scf.if` / `scf.for` 作为 tile-region 内 structured control-flow。R3.2d 必须递归
  legalize 这些 region body 内的 executable target-abstract op，并保留 `scf` container；是否选择
  硬件 branch/loop、predication 或 unroll 不是 R3.2d V0 的职责。

本文依赖：

- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2c `wafer.tile.region` IR，内部包含带 Wafer memory attr 的 memref values、
  `memref.alloc` / verifier-legal metadata view、`wafer.tile.load` / `wafer.tile.store`、
  `wafer.tile.materialize_layout`、`wafer.tile.fill/gemm/elementwise/reduce`、
  `wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast`、
  tile-region 内 `scf.if` / `scf.for` structured control-flow 和 `wafer.instr.local_drain`。
- Current stage responsibility:
  只做 Wafer instruction legalization / selection：把可执行的 target-abstract op 改写成
  `wafer.instr.*`，并保留 memref SSA graph。对 `scf.if` / `scf.for` 只递归转换其 region body，
  不改变 control-flow 结构。instruction op 通过 interface 显式暴露 issue family、memref read/write、
  descriptor attrs 和 issue effect。
- Output artifact / IR:
  同一个 `wafer.tile.region` execution scope 内的 instruction-level IR：
  memref values with `#wafer.memory<space, layout>` + `wafer.instr.*` +
  `wafer.instr.local_drain`，或结构化 legalization failure reason。
- Downstream consumer:
  R3.2e SPM placement、R3.2f DDR/resource legality、R3.2g closed-loop planner、
  R3.4 placed memref realization 和 R3.6 codegen emission。
- User-level driver / named pipeline:
  主线由 R3.2 closed-loop planner 调用；局部 bring-up pass 可命名为
  `--wafer-convert-tile-region-to-instr`，只作为 lit/debug 入口。
- Explicit non-goals:
  不新增第二套 storage/buffer IR，不决定 group boundary、tile shape、layout assignment、SPM offset、
  DDR allocation policy、raw register packet field、Tsm wrapper call、C ABI symbol 或 launch ABI。
  DTE、CSR 和 SCALAR 不进入普通 `wafer.instr` issue path。
- Completion gate:
  对 R3.2c 已支持的 load/store、layout materialize、fill、GEMM、elementwise/relation、
  reduce、copy/broadcast/transpose、static slice movement 和 metadata view 生成
  verifier-legal instruction-level IR，并覆盖 nested `scf.if` / `scf.for` body 递归转换。
  unsupported hardware instruction form 必须结构化失败，不能让 SPM placement 从 target-abstract op
  猜 demand。
```

## 2. Wafer MemRef Contract

Wafer instruction-level IR 的 buffer value 都是 `memref`：

```mlir
memref<128x64xf16, #wafer.memory<spm, tensor>>
memref<128x64xf16, #wafer.memory<spm, cx>>
memref<2x3x5x65xf16, #wafer.memory<spm, ncx>>
memref<128x64xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>` 是放在 memref memory-space slot 的 Wafer target attr：

| field | values | meaning |
| --- | --- | --- |
| `space` | `spm`, `ddr` | addressable storage domain |
| `layout` | `tensor`, `ntensor`, `cx`, `ncx` | Wafer physical layout family marker |

memref shape 仍是 logical shape，element type 仍是 logical element type。对
`#wafer.memory<spm, cx>` 和 `#wafer.memory<spm, ncx>` 来说，SPM footprint 不等于
`product(shape) * sizeof(element)`；必须调用统一解析入口：

```text
computeWaferPhysicalTensorInfo(memrefType)
```

该入口从 memref type 推导：

- address space。
- physical layout family。
- dtype storage size 和 bool bitpack policy。
- `Cx/NCx` 的 block、`C0` tail/fold、aligned C。
- 256B line/layout padding 和 layout footprint。
- begin/end range、wrapper layout enum、instruction operand legality。

所有 verifier、SPM placement、DDR/resource planning 和 instruction lowering 都必须使用这一个
入口。禁止每个 pass 自己写 `if layout == cx` 的局部解析。

### 2.1 Why Not MemRef Layout Slot

MLIR memref layout slot 表达的是 MLIR 可解释的 affine / strided address layout。Wafer
`Cx/NCx` 有 compact `C0` tail：

```text
full C blocks:
  hw-major blocks with full block width
tail block:
  hw-major compact C0 width
```

这不是普通 `strided<[aligned_C, 1]>`，也不是一个单一 affine map 能完整表达的 layout。因此
`Cx/NCx` 不放进 memref layout slot，不实现为 `MemRefLayoutAttrInterface`。普通 compact tensor、
metadata-only reshape 或标准 strided view 可以继续使用 MLIR memref layout / view 机制，但
Wafer `Cx/NCx` physical interpretation 只由 `#wafer.memory<..., cx/ncx>` marker 和
`computeWaferPhysicalTensorInfo` 解释。

### 2.2 Generic MemRef Op Boundary

带 Wafer memory attr 的 memref 是标准 SSA buffer value，但不是任意 generic memref op 都能在
instruction-level IR 中解释它：

- 允许 `memref.alloc` / ownership-preserving aliases 表达 allocation、lifetime 和 value identity。
- 允许 `memref.dim` 读取 logical shape。
- 对 `#wafer.memory<spm, tensor/ntensor>`，可在 verifier 证明 metadata view 与 logical layout
  一致时使用 `memref.cast` / `memref.reinterpret_cast` / `memref.subview`。
- 对 `#wafer.memory<spm, cx/ncx>`，不能用 generic `memref.load/store/copy/subview` 伪装硬件
  physical indexing；真实 layout conversion、slice movement 和 copy 必须通过
  `wafer.tile.materialize_layout` 或 `wafer.instr.gather_scatter` 等 Wafer op 表达。
- 在 Wafer placement/realization 前，不能让 generic memref-to-LLVM lowering 按 dense memref
  footprint 处理 `#wafer.memory<spm, cx/ncx>`。

## 3. IR Boundary

R3.2d 前：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.tile.load / wafer.tile.store
wafer.tile.materialize_layout
wafer.tile.fill/gemm/elementwise/reduce
wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast
scf.if / scf.for
wafer.instr.local_drain
```

R3.2d 后：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.instr.rdma / wafer.instr.wdma
wafer.instr.gather_scatter
wafer.instr.{fill, elementwise, reduce, convert}
wafer.instr.gemm
scf.if / scf.for
wafer.instr.local_drain
```

R3.2d 不做 memref type conversion。它只把 executable target-abstract op 改写成 instruction op，
并复用同一批 memref values。physical base address、SPM offset、end address、bank/color、
worker register window、runtime pointer 和 packet word 都不属于 R3.2d。

## 4. Instruction Ops And Issue Families

`wafer.instr.*` 的 op mnemonic 表达指令语义，不把 CT/NE/TDMA 这类硬件 issue family 做成
额外 namespace。Issue family 由 `WaferInstructionOpInterface` 派生；固定 family 的 op 不打印
冗余 attr。只有当同一个 instruction op 在相同 operand/result/attr contract 下确实能合法选择
多个 issue family 时，才允许引入显式 `issue_family` attr，并由 verifier 保证取值和 op contract
一致。

| issue family | V0 op | 来源 | 说明 |
| --- | --- | --- | --- |
| RDMA | `wafer.instr.rdma` | `wafer.tile.load` | DDR memref -> SPM memref |
| WDMA | `wafer.instr.wdma` | `wafer.tile.store` | SPM memref -> DDR memref |
| TDMA | `wafer.instr.gather_scatter` | `wafer.tile.materialize_layout`、tile movement ops | byte-counted SPM movement；contiguous copy 是 descriptor 特例 |
| CT | `wafer.instr.fill` | `wafer.tile.fill` | scalar/immediate fill |
| CT | `wafer.instr.elementwise` | `wafer.tile.elementwise` | arithmetic / relation / activation |
| CT | `wafer.instr.reduce` | `wafer.tile.reduce` | native local reduce |
| CT | `wafer.instr.convert` | future convert lowering | dtype conversion |
| NE | `wafer.instr.gemm` | `wafer.tile.gemm` | tile-local GEMM / batched GEMM |

V0 不定义 `wafer.instr.copy`。公开 SPM memcpy helper 本身也是
`TsmDataMove::GatherScatter` 样例；把 copy 单独做成 instruction op 会把 helper 名字提升为 IR
语义。

`ChannelNorm/DechannelNorm` 也不是 V0 单条 instruction op。它们是 layout materialization algorithm；
R3.2d 要么展开成一条或多条 `wafer.instr.gather_scatter`，要么结构化失败。对于
`C > block` 且存在 retained `C0` tail 的 `Cx/NCx`，full C blocks 和 compact tail block 的
inner width / stride 不同，lowering 通常需要至少两段 GatherScatter：一段搬 full blocks，一段搬
tail `C0`。如果 full-block 段和 tail 段都无法分别表示为 V0 三层 stride/iteration descriptor，
R3.2d 必须失败。

`TsmExecute` 普通 issue path 只覆盖 CT/NE/RDMA/WDMA/TDMA。DTE、CSR、SCALAR 走
tile communication、runtime/MMIO 或专门 sync 边界，不放进普通 `wafer.instr.*` issue path。

## 5. Operand And Result Model

instruction op 直接读写 memref，但 **instruction op 本身不产生 buffer result**。
凡是 target-abstract op 原来返回 buffer 的地方，R3.2d 先确保存在 destination `memref.alloc`
或 verifier-legal destination memref，再生成写入该 memref 的 instruction op，并用 destination
memref 替换原 op result 的 uses。

这个模型避免把指令 issue 和 buffer identity 混在一起：

- source memref 是 instruction operand。
- destination memref 也是 instruction operand。
- result/temp/psum/staging buffer 由 `memref.alloc` 或 accepted alias/view 创建。
- metadata-only reshape 由 verifier-legal memref view 表达；physical layout conversion 必须是
  explicit movement。
- SPM offset、range、bank span 由 R3.2e 写入，或在 R3.4 realization 降成 placed memref /
  address descriptor。
- RDMA/WDMA 的 DDR side 使用 `memref<..., #wafer.memory<ddr, layout>>`；DDR resource stage 负责
  external allocation contract、pool/domain、compiler-managed allocation 和 constant residency。

R3.2d 只 materialize **unplaced logical descriptor facts**：byte count、stride/iteration、op kind、
reduction dimensions、GEMM dimensions、elementwise kind 等。这些字段能从当前 IR type、attrs 和
source op verifier 重算。

## 6. Common Instruction Interface

所有 `wafer.instr.*` op 必须实现同一个窄 interface：

```text
WaferInstructionOpInterface {
  getInstructionQueueFamily() -> InstrQueue
  collectInstructionEffects(...) -> memref read/write + queue issue
  verifyInstructionContract()
}
```

`InstrQueue` 是 Wafer enum/interface fact，V0 只包含：

| enum | hardware issue family |
| --- | --- |
| `ct` | CT / CGRA queue |
| `ne` | NE queue |
| `rdma` | RDMA queue |
| `wdma` | WDMA queue |
| `tdma` | TDMA queue |

interface 返回的是 op-local facts，不返回 planner side table，也不复制全局 schedule。对
`rdma`、`wdma`、`gather_scatter`、`fill`、`elementwise`、`reduce`、`convert` 和 `gemm` 这类
固定 issue family 的 V0 op，`getInstructionQueueFamily()` 由 op class 静态派生，不要求 IR
打印 `issue_family` attr。后续若出现同一个 op contract 下可选择多个 issue family 的 instruction
op，才在该 op 上增加显式 attr，并把合法取值纳入 verifier。

resource effects 至少要表达：

| queue | buffer effects | issue effect |
| --- | --- | --- |
| RDMA | DDR memref read + SPM memref write | Movement/RDMA issue |
| WDMA | SPM memref read + DDR memref write | Movement/WDMA issue |
| TDMA | SPM memref read + SPM memref write | Movement/TDMA issue |
| CT | SPM memref read/write as operand contract requires | Compute/CT issue |
| NE | SPM memref read + SPM memref write | Compute/NE issue |

R3.2d 不建模 worker id。`TsmExecute` 的 worker bits、register window 和 packet field 属于 placed
instruction / codegen emission。

## 7. ODS-Level Op Contracts

本节是实现时的 ODS 合同。assembly format 可以按 MLIR 可读性微调，但 operand/result/attr
语义不能变。

### 7.1 DMA Descriptor Attributes

RDMA、WDMA 和 TDMA 共同使用 fixed-rank descriptor attrs。字段是 logical descriptor，不是
physical packet：

| attr | type | meaning |
| --- | --- | --- |
| `byte_count` | `I64Attr` | 该 instruction 的 total logical payload bytes，用于 resource effect |
| `inner_bytes` | `I64Attr` | 最内层 contiguous byte count |
| `src_strides` | `DenseI64ArrayAttr` | source byte strides，长度为 3 |
| `src_iterations` | `DenseI64ArrayAttr` | source logical iterations，长度为 3，值为正数 |
| `dst_strides` | `DenseI64ArrayAttr` | destination byte strides，长度为 3 |
| `dst_iterations` | `DenseI64ArrayAttr` | destination logical iterations，长度为 3，值为正数 |

contiguous movement 使用 `inner_bytes == byte_count`，stride 全 0，iteration 全 1。byte stride
必须已经从 element stride 转换完成。descriptor 表达不了的 dynamic stride、超过 3 层的静态 stride
或两端都需要复杂非连续访问的情况，R3.2d 必须结构化失败，不能生成名字上合法但下游无法 packetize
的 instruction op。

### 7.2 RDMA / WDMA

```text
wafer.instr.rdma source to dest attr-dict : type(source) to type(dest)
wafer.instr.wdma source to dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.rdma` | `source: MemRef<#wafer.memory<ddr, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | `byte_count`, `inner_bytes`, `src_strides`, `src_iterations` |
| `wafer.instr.wdma` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<ddr, *>>` | none | `byte_count`, `inner_bytes`, `dst_strides`, `dst_iterations` |

R3.2c 的 `wafer.tile.load/store` 边界默认是 compact tensor layout；如果 consumer 需要 `Cx/NCx`，
必须通过 `wafer.tile.materialize_layout` 再 lower 到 TDMA，而不是让 RDMA/WDMA 隐式承担 layout
conversion。

### 7.3 GatherScatter

```text
wafer.instr.gather_scatter source to dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gather_scatter` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | `byte_count`, `inner_bytes`, `src_strides`, `src_iterations`, `dst_strides`, `dst_iterations` |

V0 只定义这一条 TDMA-backed movement op。copy、layout materialization、static slice movement、broadcast
和 transpose 都要么映射成一条或多条 gather_scatter，要么失败。`wafer.instr.copy` 不作为
单独 IR op；contiguous copy 是 gather_scatter descriptor 特例。

### 7.4 Fill / Elementwise / Reduce / Convert

```text
wafer.instr.fill dest, value attr-dict : type(dest), type(value)
wafer.instr.elementwise kind inputs into dest attr-dict
    : type(inputs) into type(dest)
wafer.instr.reduce kind input into dest (, init)? attr-dict
    : type(input) into type(dest)
wafer.instr.convert source into dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.fill` | `dest: SPM memref`, `value: scalar` | none | none |
| `wafer.instr.elementwise` | `inputs: Variadic<SPM memref>`, `dest: SPM memref` | none | `kind`; optional `indexing_maps` copied from target-abstract op |
| `wafer.instr.reduce` | `input: SPM memref`, `dest: SPM memref`, optional scalar `init` | none | `kind`, `dimensions` |
| `wafer.instr.convert` | `source: SPM memref`, `dest: SPM memref` | none | `src_dtype`, `dst_dtype`; future source op only |

`wafer.instr.convert` 作为 instruction op 定义，因为 hardware convert 当前属于 CT issue family；
但当前 R3.2c 没有 `wafer.tile.convert` source op。因此 R3.2d V0 需要定义 ODS/verifier，但
completion 不要求 convert lowering pattern，直到 source op 存在。

### 7.5 GEMM

```text
wafer.instr.gemm lhs, rhs into dest attr-dict
    : type(lhs), type(rhs) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gemm` | `lhs: SPM memref`, `rhs: SPM memref`, `dest: SPM memref` | none | `m`, `k`, `n`; optional batched GEMM attrs copied from `wafer.tile.gemm` |

V0 要求三个 SPM memref operand 都使用 aligned layout marker：rank <= 2 使用
`#wafer.memory<spm, cx>`，rank > 2 使用 `#wafer.memory<spm, ncx>`。Fused bias、activation、
quant、psum accumulation policy 和 sparse / INT8 variants 不属于 R3.2d V0。

## 8. Lowering Rules

R3.2d 应实现为 MLIR DialectConversion：

- illegal：`wafer.tile.load`、`wafer.tile.store`、`wafer.tile.materialize_layout`、
  `wafer.tile.fill/gemm/elementwise/reduce` 和 tile movement ops。
- legal：`memref.alloc`、verifier-legal metadata view ops、`wafer.instr.*`、
  `wafer.instr.local_drain`、`wafer.tile.region` container、`scf.if` / `scf.for` container
  和必要 scalar/support op。
- no type conversion for Wafer tagged memref values。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 committed 主线 IR。

V0 mapping：

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.tile.load` | ensure / create destination `memref<..., #wafer.memory<spm, tensor>>`; emit `wafer.instr.rdma`; replace original result with dest memref |
| `wafer.tile.store` | emit `wafer.instr.wdma`; erase store |
| `wafer.tile.materialize_layout` | ensure / create destination memref with requested marker; emit one or more `wafer.instr.gather_scatter`; replace result with dest memref |
| `wafer.tile.fill` | emit `wafer.instr.fill` writing the existing dest memref |
| `wafer.tile.gemm` | ensure / create destination aligned SPM memref; emit `wafer.instr.gemm`; replace result with dest memref |
| `wafer.tile.elementwise` | ensure / create destination SPM memref; emit `wafer.instr.elementwise`; replace result with dest memref |
| `wafer.tile.reduce` | ensure / create destination SPM memref; emit `wafer.instr.reduce`; replace result with dest memref |
| `wafer.tile.copy` | ensure / create destination SPM memref; emit one gather_scatter; replace result with dest memref |
| `wafer.tile.extract_slice` | ensure / create destination compact SPM memref; emit gather_scatter from source slice to destination |
| `wafer.tile.insert_slice` | ensure / create result SPM memref; first gather_scatter copy dest to result, then gather_scatter source into result slice |
| `wafer.tile.broadcast` | ensure / create destination SPM memref; emit one or more gather_scatter if static broadcast descriptor is expressible |
| `wafer.tile.transpose` | ensure / create destination SPM memref; emit gather_scatter if permutation is statically expressible |
| metadata reshape/view | stays as verifier-legal memref alias; no instruction issue |
| `scf.if` / `scf.for` | preserve the structured control-flow op; recursively legalize executable target-abstract ops in each nested region; keep scalar and memref yields explicit |
| `wafer.tile.send/recv/wait/all_gather/reduce_scatter/all_reduce` | not handled by R3.2d V0 |

R3.2d may generate multiple instruction ops for a single target-abstract movement op, but it must not write a
global schedule attr. The instruction sequence is the region body itself.
Nested `scf` regions are part of that body: R3.2d rewrites their executable contents under MLIR region
scoping rules, but it does not lower them to hardware branch/loop instructions.

## 9. Failure Contract

R3.2d failure is a legalization result, not an IR artifact. A rejected legalization attempt may carry
diagnostics to the closed-loop planner or debug pass, but rejected instruction IR is discarded.

必须结构化失败的情况：

- non-ranked or dynamic-shaped memref where V0 needs static byte/stride computation.
- unsupported Wafer memory attr, address space or physical layout marker for an instruction family.
- unsupported dtype, including relation/elementwise/convert pairs not mapped to CT V0.
- movement descriptor cannot be represented with `inner_bytes` plus three stride/iteration levels.
- `Cx/NCx` materialization with retained `C0` tail cannot be split into separately representable
  full-block and tail GatherScatter descriptors.
- static slice/broadcast/transpose cannot be converted into one or more gather_scatter descriptors.
- unsupported control-flow op, multi-block region, or nested region whose executable body cannot be fully
  legalized under the same instruction conversion rules.
- NE GEMM dimension attrs cannot be derived from operand/result types and optional batch attrs.
- reduce `dimensions` cannot map to supported native reduce dimension encoding.
- any source op that would require DTE/CSR/SCALAR, raw packet fields, SPM offset, DDR allocation policy or
  runtime ABI call to be legal.

Diagnostics should mention the source op and the missing legality fact, for example:
`wafer.tile.transpose cannot lower to TDMA-backed wafer.instr.gather_scatter: unsupported permutation`.

## 10. Verifier Contract

R3.2d verifier checks only instruction legality:

- ODS type constraints enforce memref/tensor/scalar operand classes.
- all Wafer tagged memref types used by instruction ops are ranked and static for V0.
- memory attr matches instruction family:
  RDMA reads `#wafer.memory<ddr, *>` and writes `#wafer.memory<spm, *>`；
  WDMA reads `#wafer.memory<spm, *>` and writes `#wafer.memory<ddr, *>`；
  TDMA/CT/NE read/write tile-local SPM memrefs。
- RDMA/WDMA/TDMA descriptor attrs have fixed array length, positive iteration values, non-negative byte
  strides and positive byte counts.
- descriptor byte counts are consistent with compact tensor payload size or statically described
  slice/broadcast/transpose domain as applicable. Cx/NCx padding span is derived from
  `computeWaferPhysicalTensorInfo(memrefType)` and target policy, not copied into `byte_count`.
- NE GEMM and CT reduce require supported aligned layout marker, dtype and rank.
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- no SPM offset/end/bank attrs before R3.2e.
- no DTE/CSR/SCALAR ordinary instruction op.

R3.2d does **not** verify physical address range, SPM bank conflicts, DDR pool/domain capacity, runtime
symbol, packet bit layout or worker register window. Those checks belong to R3.2e/R3.2f/R3.4/R3.6.

## 11. Example

```mlir
wafer.tile.region ... {
  %a_ddr = ... : memref<128x64xf16, #wafer.memory<ddr, tensor>>
  %b_ddr = ... : memref<64x128xf16, #wafer.memory<ddr, tensor>>
  %c_ddr = ... : memref<128x128xf16, #wafer.memory<ddr, tensor>>

  %a_tensor = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
  %b_tensor = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
  %a_cx = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, cx>>
  %b_cx = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, cx>>
  %c_cx = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
  %c_tensor = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %a_ddr to %a_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<128x64xf16, #wafer.memory<ddr, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %b_ddr to %b_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<64x128xf16, #wafer.memory<ddr, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.gather_scatter %a_tensor to %a_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %b_tensor to %b_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gemm %a_cx, %b_cx into %c_cx
      {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : memref<128x64xf16, #wafer.memory<spm, cx>>,
        memref<64x128xf16, #wafer.memory<spm, cx>>
    into memref<128x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %c_cx to %c_tensor
      {byte_count = 32768 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 256, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
     to memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.wdma %c_tensor to %c_ddr
      {byte_count = 32768 : i64, inner_bytes = 32768 : i64,
       dst_strides = array<i64: 0, 0, 0>, dst_iterations = array<i64: 1, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>
     to memref<128x128xf16, #wafer.memory<ddr, tensor>>

  wafer.instr.local_drain
}
```

这是形态示例，不固定 parser/printer，也不固定 planner 对 layout materialization 的具体选择。
例子中 stride 数值只说明 descriptor 字段位置，不作为 Cx padding 或 hardware packet 的规范值；
真实 padded size、Cx/NCx 对齐、bool bitpack、descriptor stride 和 SPM offset 分别由
`computeWaferPhysicalTensorInfo`、SPM placement 和 later realization 处理。

## 12. Implementation Work

R3.2c 已完成的前置：

1. `#wafer.memory<space, layout>` target memory attr。
2. `computeWaferPhysicalTensorInfo(memrefType)`，统一计算 layout marker、Cx/C0、
   footprint、range-end、bool bitpack 和 wrapper layout enum。
3. 旧 `#wafer.memory_space` / `#wafer.mem_layout` / `!wafer.storage` / `wafer.tile.alloc`
   合同已从主线 IR 定义和测试中删除。

R3.2d.1 已完成：

1. 增加 `InstrQueue` enum、`WaferInstructionOpInterface` 和 instruction effect helper。
2. 增加 `wafer.instr.rdma`、`wafer.instr.wdma`、
   `wafer.instr.gather_scatter`、`wafer.instr.fill`、
   `wafer.instr.elementwise`、`wafer.instr.reduce`、
   `wafer.instr.convert` 和 `wafer.instr.gemm` ODS。
3. 为每个 op 实现 verifier、MemoryEffects、instruction interface 和 positive/negative lit tests。

R3.2d.2/R3.2d.3 仍需实现：

4. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion，并让 main R3.2 planner 调用同一
   conversion implementation。
5. conversion 递归处理 `scf.if` / `scf.for` region body，并保留 scalar / memref yield 关系。
6. 增加 conversion tests，覆盖 load/store、layout materialize、fill、GEMM、elementwise/relation、
   reduce、copy/broadcast/transpose、extract_slice/insert_slice、metadata view preserved 和 structured
   control-flow failure。
7. 为 `wafer.instr.convert` 增加 parser/verifier tests；convert lowering 等 `wafer.tile.convert`
   或等价 source op 出现后再接入 completion gate。
