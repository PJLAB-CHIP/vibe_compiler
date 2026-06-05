# Wafer Instruction IR Design

日期：2026-06-05

状态：R3.2d 实现级设计收口；2026-06-05 统一 storage 命名；实现未完成

本文定义 R3.2d 的 instruction-level Wafer IR。核心结论：

- 只新增 `wafer.instr.*` 硬件指令级 op。
- 复用现有 `!wafer.storage`、`wafer.tile.alloc`、`wafer.tile.reshape`。
- `!wafer.storage` 是唯一 storage-like IR 对象；不保留旧 buffer alias，
  不新增 storage role attr 或 size policy attr。
- 不生成 SPM offset、DDR BO binding、raw packet 或 C ABI call。

`wafer.instr` 的作用是把 target-abstract tile-region op 变成可执行硬件动作，并让下游能从
`!wafer.storage` SSA、op operands、attrs、MemoryEffects 和显式 drain 直接推导 placement
输入。它不是另一层 storage IR。

本文依赖：

- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `docs/wafer-register-level-instruction-spec.md`

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2c `wafer.tile.region` IR，内部包含 accepted layout 的 `!wafer.storage`、
  `wafer.tile.alloc`、`wafer.tile.load` / `wafer.tile.store`、
  `wafer.tile.materialize_layout`、`wafer.tile.fill/gemm/elementwise/reduce`、
  `wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast`、
  `wafer.tile.reshape` 和 `wafer.instr.local_drain`。
- Current stage responsibility:
  只做 Wafer instruction legalization / selection：把可执行的 target-abstract op 改写成
  `wafer.instr.*`，并保留现有 storage SSA graph。instruction op 显式表达 queue family、
  storage read/write、descriptor attrs 和 issue effect。
- Output artifact / IR:
  同一个 `wafer.tile.region` execution scope 内的 instruction-level IR：
  `!wafer.storage` + `wafer.tile.alloc` / `wafer.tile.reshape` + `wafer.instr.*` +
  `wafer.instr.local_drain`，或结构化 legalization failure reason。
- Downstream consumer:
  R3.2e SPM placement、R3.2f DDR/resource legality、R3.2g closed-loop planner、
  R3.4 placed storage realization 和 R3.6 codegen emission。
- User-level driver / named pipeline:
  主线由 R3.2 closed-loop planner 调用；局部 bring-up pass 可命名为
  `--wafer-convert-tile-region-to-instr`，只作为 lit/debug 入口。
- Explicit non-goals:
  不新增第二套 storage/buffer IR，不决定 group boundary、tile shape、layout assignment、SPM offset、
  DDR BO binding、raw register packet field、Tsm wrapper call、C ABI symbol 或 launch ABI。
  DTE、CSR 和 SCALAR 不进入普通 `wafer.instr` issue path。
- Completion gate:
  对 R3.2c 已支持的 load/store、layout materialize、fill、GEMM、elementwise/relation、
  reduce、copy/broadcast/transpose、static slice movement 和 static reshape view 生成
  verifier-legal instruction-level IR。unsupported hardware instruction form 必须结构化失败，
  不能让 SPM placement 从 target-abstract op 猜 demand。
```

## 2. IR Boundary

R3.2d 前：

```text
!wafer.storage
wafer.tile.alloc
wafer.tile.load / wafer.tile.store
wafer.tile.materialize_layout
wafer.tile.fill/gemm/elementwise/reduce
wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast
wafer.tile.reshape
wafer.instr.local_drain
```

R3.2d 后：

```text
!wafer.storage
wafer.tile.alloc
wafer.tile.reshape
wafer.instr.rdma / wafer.instr.wdma
wafer.instr.tdma.gather_scatter
wafer.instr.ct.*
wafer.instr.ne.gemm
wafer.instr.local_drain
```

没有 type conversion：`!wafer.storage<tensor, mem_layout, memory_space>` 从 target-abstract
tile-region 贯穿到 instruction-level tile-region。它仍然不带 physical offset、raw address、
packet 或 runtime handle。

## 3. Instruction Families

| family | V0 op | 来源 | 说明 |
| --- | --- | --- | --- |
| RDMA | `wafer.instr.rdma` | `wafer.tile.load` | DDR boundary value -> `#spm` storage |
| WDMA | `wafer.instr.wdma` | `wafer.tile.store` | `#spm` storage -> DDR boundary value |
| TDMA | `wafer.instr.tdma.gather_scatter` | `wafer.tile.materialize_layout`、tile movement ops | byte-counted movement；contiguous copy 是 descriptor 特例 |
| CT | `wafer.instr.ct.fill` | `wafer.tile.fill` | scalar/immediate fill |
| CT | `wafer.instr.ct.elementwise` | `wafer.tile.elementwise` | arithmetic / relation / activation |
| CT | `wafer.instr.ct.reduce` | `wafer.tile.reduce` | native local reduce |
| CT | `wafer.instr.ct.convert` | future convert lowering | dtype conversion |
| NE | `wafer.instr.ne.gemm` | `wafer.tile.gemm` | tile-local GEMM / batched GEMM |

V0 不定义 `wafer.instr.tdma.copy`。公开 SPM memcpy helper 本身也是
`TsmDataMove::GatherScatter` 样例；把 copy 单独做成 instruction op 会把 helper 名字提升为 IR
语义。

`ChannelNorm/DechannelNorm` 也不是 V0 单条 instruction op。它们是 layout materialization algorithm；
R3.2d 要么展开成一条或多条 `wafer.instr.tdma.gather_scatter`，要么结构化失败。

`TsmExecute` 普通 issue path 只覆盖 CT/NE/RDMA/WDMA/TDMA。DTE、CSR、SCALAR 走
tile communication、runtime/MMIO 或专门 sync 边界，不放进普通 `wafer.instr.*` issue path。

## 4. Operand And Result Model

instruction op 直接读写 `!wafer.storage`，但 **instruction op 本身不产生 storage result**。
凡是 target-abstract op 原来返回 storage 的地方，R3.2d 先创建新的 `wafer.tile.alloc`，
再生成写入该 storage 的 instruction op，并用 alloc result 替换原 op result 的 uses。

这个模型避免把指令 issue 和 storage identity 混在一起：

- source storage 是 instruction operand。
- destination storage 也是 instruction operand。
- result/temp/psum/staging storage 仍由 `wafer.tile.alloc` 创建。
- static reshape alias 仍由 `wafer.tile.reshape` 表达。
- SPM offset、range、bank span 由 R3.2e 写入，或在 R3.4 realization 降成 memref/descriptor。
- RDMA/WDMA 的 DDR side 使用 ranked tensor boundary；后续 DDR resource stage 可把它换成
  memref/DDR descriptor，但 R3.2d 不为 DDR side 新增 storage wrapper。

R3.2d 只 materialize **unplaced logical descriptor facts**：byte count、stride/iteration、op kind、
reduction dimensions、GEMM dimensions、elementwise kind 等。这些字段能从当前 IR type、attrs 和
source op verifier 重算。physical base address、end address、SPM bank/color、worker register window、
runtime pointer 和 packet word 都不属于 R3.2d。

## 5. Common Instruction Interface

所有 `wafer.instr.*` op 必须实现同一个窄 interface：

```text
WaferInstructionOpInterface {
  getInstructionQueueFamily() -> InstrQueue
  collectInstructionEffects(...) -> storage read/write + queue issue
  verifyInstructionContract()
}
```

`InstrQueue` 是 Wafer enum attr，V0 只包含：

| enum | hardware issue family |
| --- | --- |
| `ct` | CT / CGRA queue |
| `ne` | NE queue |
| `rdma` | RDMA queue |
| `wdma` | WDMA queue |
| `tdma` | TDMA queue |

interface 返回的是 op-local facts，不返回 planner side table，也不复制全局 schedule。resource
effects 至少要表达：

| queue | storage effects | issue effect |
| --- | --- | --- |
| RDMA | DDR boundary read + SPM storage write | Movement/RDMA issue |
| WDMA | SPM storage read + DDR boundary write | Movement/WDMA issue |
| TDMA | SPM storage read + SPM storage write | Movement/TDMA issue |
| CT | SPM storage read/write as operand contract requires | Compute/CT issue |
| NE | SPM storage read + SPM storage write | Compute/NE issue |

R3.2d 不建模 worker id。`TsmExecute` 的 worker bits、register window 和 packet field 属于 placed
instruction / codegen emission。

## 6. ODS-Level Op Contracts

本节是实现时的 ODS 合同。assembly format 可以按 MLIR 可读性微调，但 operand/result/attr
语义不能变。

### 6.1 DMA Descriptor Attributes

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

### 6.2 RDMA / WDMA

```text
wafer.instr.rdma source to dest attr-dict : type(source) to type(dest)
wafer.instr.wdma source to dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.rdma` | `source: AnyRankedTensor`, `dest: !wafer.storage` | none | `byte_count`, `inner_bytes`, `src_strides`, `src_iterations` |
| `wafer.instr.wdma` | `source: !wafer.storage`, `dest: AnyRankedTensor` | none | `byte_count`, `inner_bytes`, `dst_strides`, `dst_iterations` |

V0 要求 RDMA destination 和 WDMA source 使用 `#spm` memory space。R3.2c 的
`wafer.tile.load/store` 边界默认是 compact tensor layout；如果 consumer 需要 `Cx/NCx`，
必须通过 `wafer.tile.materialize_layout` 再 lower 到 TDMA，而不是让 RDMA/WDMA 隐式承担 layout
conversion。

### 6.3 TDMA GatherScatter

```text
wafer.instr.tdma.gather_scatter source to dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.tdma.gather_scatter` | `source: !wafer.storage`, `dest: !wafer.storage` | none | `byte_count`, `inner_bytes`, `src_strides`, `src_iterations`, `dst_strides`, `dst_iterations` |

V0 只定义这一条 TDMA movement op。copy、layout materialization、static slice movement、broadcast
和 transpose 都要么映射成一条或多条 gather_scatter，要么失败。`wafer.instr.tdma.copy` 不作为
单独 IR op；contiguous copy 是 gather_scatter descriptor 特例。

### 6.4 CT Fill / Elementwise / Reduce / Convert

```text
wafer.instr.ct.fill dest, value attr-dict : type(dest), type(value)
wafer.instr.ct.elementwise kind inputs into dest attr-dict
    : type(inputs) into type(dest)
wafer.instr.ct.reduce kind input into dest (, init)? attr-dict
    : type(input) into type(dest)
wafer.instr.ct.convert source into dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.ct.fill` | `dest: !wafer.storage`, `value: scalar` | none | none |
| `wafer.instr.ct.elementwise` | `inputs: Variadic<!wafer.storage>`, `dest: !wafer.storage` | none | `kind`; optional `indexing_maps` copied from target-abstract op |
| `wafer.instr.ct.reduce` | `input: !wafer.storage`, `dest: !wafer.storage`, optional scalar `init` | none | `kind`, `dimensions` |
| `wafer.instr.ct.convert` | `source: !wafer.storage`, `dest: !wafer.storage` | none | `src_dtype`, `dst_dtype`; future source op only |

`ct.convert` 作为 instruction op 定义，因为 hardware CT convert 属于同一 queue family；但当前
R3.2c 没有 `wafer.tile.convert` source op。因此 R3.2d V0 需要定义 ODS/verifier，但
completion 不要求 convert lowering pattern，直到 source op 存在。

### 6.5 NE GEMM

```text
wafer.instr.ne.gemm lhs, rhs into dest attr-dict
    : type(lhs), type(rhs) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.ne.gemm` | `lhs: !wafer.storage`, `rhs: !wafer.storage`, `dest: !wafer.storage` | none | `m`, `k`, `n`; optional batched GEMM attrs copied from `wafer.tile.gemm` |

V0 要求三个 storage operand 都使用 `#spm` 和 aligned layout：rank <= 2 使用 `Cx`，rank > 2
使用 `NCx`。Fused bias、activation、quant、psum accumulation policy 和 sparse / INT8 variants
不属于 R3.2d V0。

## 7. Lowering Rules

R3.2d 应实现为 MLIR DialectConversion：

- illegal：`wafer.tile.load`、`wafer.tile.store`、`wafer.tile.materialize_layout`、
  `wafer.tile.fill/gemm/elementwise/reduce` 和 tile movement ops。
- legal：`wafer.tile.alloc`、`wafer.tile.reshape`、`wafer.instr.*`、`wafer.instr.local_drain`、
  `wafer.tile.region` container 和必要 scalar/support op。
- no type conversion for `!wafer.storage`。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 committed 主线 IR。

V0 mapping：

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.tile.load` | create `wafer.tile.alloc` with original result type; emit `wafer.instr.rdma`; replace original result with alloc result |
| `wafer.tile.store` | emit `wafer.instr.wdma`; erase store |
| `wafer.tile.materialize_layout` | create destination `wafer.tile.alloc`; emit one or more `wafer.instr.tdma.gather_scatter`; replace result with alloc result |
| `wafer.tile.fill` | emit `wafer.instr.ct.fill` writing the existing dest storage |
| `wafer.tile.gemm` | create destination `wafer.tile.alloc`; emit `wafer.instr.ne.gemm`; replace result with alloc result |
| `wafer.tile.elementwise` | create destination `wafer.tile.alloc`; emit `wafer.instr.ct.elementwise`; replace result with alloc result |
| `wafer.tile.reduce` | create destination `wafer.tile.alloc`; emit `wafer.instr.ct.reduce`; replace result with alloc result |
| `wafer.tile.copy` | create destination `wafer.tile.alloc`; emit one gather_scatter; replace result with alloc result |
| `wafer.tile.extract_slice` | create destination `wafer.tile.alloc`; emit gather_scatter from source slice to compact destination |
| `wafer.tile.insert_slice` | create result `wafer.tile.alloc`; first gather_scatter copy dest to result, then gather_scatter source into result slice |
| `wafer.tile.broadcast` | create destination `wafer.tile.alloc`; emit one or more gather_scatter if static broadcast descriptor is expressible |
| `wafer.tile.transpose` | create destination `wafer.tile.alloc`; emit gather_scatter if permutation is statically expressible |
| `wafer.tile.reshape` | stays as aliasing view; no instruction issue |
| `wafer.tile.send/recv/wait/all_gather/reduce_scatter/all_reduce` | not handled by R3.2d V0 |

R3.2d may generate multiple instruction ops for a single target-abstract movement op, but it must not write a
global schedule attr. The instruction sequence is the region body itself.

## 8. Failure Contract

R3.2d failure is a legalization result, not an IR artifact. A rejected candidate may carry diagnostics to
the closed-loop planner or debug pass, but rejected instruction IR is discarded.

必须结构化失败的情况：

- non-ranked or dynamic-shaped storage tensor where V0 needs static byte/stride computation.
- unsupported memory space or layout for an instruction family.
- unsupported dtype, including relation/elementwise/convert pairs not mapped to CT V0.
- movement descriptor cannot be represented with `inner_bytes` plus three stride/iteration levels.
- static slice/broadcast/transpose cannot be converted into one or more gather_scatter descriptors.
- NE GEMM dimension attrs cannot be derived from operand/result types and optional batch attrs.
- reduce `dimensions` cannot map to supported native reduce dimension encoding.
- any source op that would require DTE/CSR/SCALAR, raw packet fields, SPM offset, DDR BO binding or
  runtime ABI call to be legal.

Diagnostics should mention the source op and the missing legality fact, for example:
`wafer.tile.transpose cannot lower to TDMA gather_scatter: unsupported permutation`.

## 9. Verifier Contract

R3.2d verifier checks only instruction legality:

- ODS type constraints enforce storage/tensor/scalar operand classes.
- all `!wafer.storage` tensor types used by instruction ops are ranked and static for V0.
- memory space matches instruction family:
  RDMA writes `#spm`; WDMA reads `#spm`; TDMA/CT/NE read/write tile-local storage values.
- RDMA/WDMA/TDMA descriptor attrs have fixed array length, positive iteration values, non-negative byte
  strides and positive byte counts.
- descriptor byte counts are consistent with compact tensor payload size or statically described
  slice/broadcast/transpose domain as applicable. Cx/NCx padding span is derived later from
  `!wafer.storage` layout and target policy, not copied into `byte_count`.
- NE GEMM and CT reduce require supported aligned layout, dtype and rank.
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- no SPM offset/end/bank attrs before R3.2e.
- no DTE/CSR/SCALAR ordinary instruction op.

R3.2d does **not** verify physical address range, SPM bank conflicts, DDR pool/domain capacity, runtime
symbol, packet bit layout or worker register window. Those checks belong to R3.2e/R3.2f/R3.4/R3.6.

## 10. Example

```mlir
wafer.tile.region ... {
  %a_tensor = wafer.tile.alloc
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %b_tensor = wafer.tile.alloc
      : !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %a_cx = wafer.tile.alloc
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %b_cx = wafer.tile.alloc
      : !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %c_cx = wafer.tile.alloc
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %c_tensor = wafer.tile.alloc
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>

  wafer.instr.rdma %arg0 to %a_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : tensor<128x64xf16>
     to !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  wafer.instr.rdma %arg1 to %b_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : tensor<64x128xf16>
     to !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>

  wafer.instr.tdma.gather_scatter %a_tensor to %a_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     to !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  wafer.instr.tdma.gather_scatter %b_tensor to %b_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     to !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.ne.gemm %a_cx, %b_cx into %c_cx
      {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
        !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    into !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.tdma.gather_scatter %c_cx to %c_tensor
      {byte_count = 32768 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 256, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
     to !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  wafer.instr.wdma %c_tensor to %arg2
      {byte_count = 32768 : i64, inner_bytes = 32768 : i64,
       dst_strides = array<i64: 0, 0, 0>, dst_iterations = array<i64: 1, 1, 1>}
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     to tensor<128x128xf16>

  wafer.instr.local_drain
}
```

这是形态示例，不固定 parser/printer，也不固定 planner 对 layout materialization 的具体选择。
例子中 stride 数值只说明 descriptor 字段位置，不作为 Cx padding 或 hardware packet 的规范值；
真实 padded size、Cx/NCx 对齐、bool bitpack、descriptor stride 和 SPM offset 分别由 verifier、
SPM placement 和 later realization 处理。

## 11. Implementation Work

R3.2d 实现需要：

1. 增加 `InstrQueue` enum、`WaferInstructionOpInterface` 和 instruction effect helper。
2. 增加 `wafer.instr.rdma`、`wafer.instr.wdma`、
   `wafer.instr.tdma.gather_scatter`、`wafer.instr.ct.fill`、
   `wafer.instr.ct.elementwise`、`wafer.instr.ct.reduce`、
   `wafer.instr.ct.convert` 和 `wafer.instr.ne.gemm` ODS。
3. 为每个 op 实现 verifier、MemoryEffects、instruction interface 和 positive/negative lit tests。
4. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion，并让 main R3.2 planner 调用同一
   conversion implementation。
5. 增加 conversion tests，覆盖 load/store、layout materialize、fill、GEMM、elementwise/relation、
   reduce、copy/broadcast/transpose、extract_slice/insert_slice、tile.reshape preserved 和 structured
   failure。
6. 为 `wafer.instr.ct.convert` 增加 parser/verifier tests；convert lowering 等 `wafer.tile.convert`
   或等价 source op 出现后再接入 completion gate。
