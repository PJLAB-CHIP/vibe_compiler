# Wafer Instruction IR Design

日期：2026-06-05

状态：R3.2d 设计收口；2026-06-05 简化为复用 `!wafer.tile_buffer`；实现未完成

本文定义 R3.2d 的 instruction-level Wafer IR。核心结论：

- 只新增 `wafer.instr.*` 硬件指令级 op。
- 复用现有 `!wafer.tile_buffer`、`wafer.alloc_tile`、`wafer.view.reshape`。
- 不新增 `!wafer.storage`、`wafer.storage.*`、tile-buffer role attr 或 size policy attr。
- 不生成 SPM offset、DDR BO binding、raw packet 或 C ABI call。

`wafer.instr` 的作用是把 target-abstract tile-region op 变成可执行硬件动作，并让下游能从
`!wafer.tile_buffer` SSA、op operands、attrs、MemoryEffects 和显式 drain 直接推导 placement
输入。它不是另一层 buffer IR。

本文依赖：

- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `docs/wafer-register-level-instruction-spec.md`

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2c `wafer.tile_region` IR，内部包含 accepted layout 的 `!wafer.tile_buffer`、
  `wafer.alloc_tile`、`wafer.load_tile` / `wafer.store_tile`、
  `wafer.layout.materialize`、`wafer.compute.*`、`wafer.move.*`、
  `wafer.view.reshape` 和 `wafer.sync.*`。
- Current stage responsibility:
  只做 Wafer instruction legalization / selection：把可执行的 target-abstract op 改写成
  `wafer.instr.*`，并保留现有 tile-buffer SSA graph。instruction op 显式表达 queue family、
  tile-buffer read/write、descriptor attrs 和 issue effect。
- Output artifact / IR:
  同一个 `wafer.tile_region` execution scope 内的 instruction-level IR：
  `!wafer.tile_buffer` + `wafer.alloc_tile` / `wafer.view.reshape` + `wafer.instr.*` +
  `wafer.sync.*`，或结构化 legalization failure reason。
- Downstream consumer:
  R3.2e SPM placement、R3.2f DDR/resource legality、R3.2g closed-loop planner、
  R3.4 placed tile-buffer realization 和 R3.6 codegen emission。
- User-level driver / named pipeline:
  主线由 R3.2 closed-loop planner 调用；局部 bring-up pass 可命名为
  `--wafer-convert-tile-region-to-instr`，只作为 lit/debug 入口。
- Explicit non-goals:
  不新增 storage IR，不决定 group boundary、tile shape、layout assignment、SPM offset、
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
!wafer.tile_buffer
wafer.alloc_tile
wafer.load_tile / wafer.store_tile
wafer.layout.materialize
wafer.compute.* / wafer.move.*
wafer.view.reshape
wafer.sync.*
```

R3.2d 后：

```text
!wafer.tile_buffer
wafer.alloc_tile
wafer.view.reshape
wafer.instr.rdma / wafer.instr.wdma
wafer.instr.tdma.gather_scatter
wafer.instr.ct.*
wafer.instr.ne.gemm
wafer.sync.*
```

没有 type conversion：`!wafer.tile_buffer<tensor, mem_layout, memory_space>` 从 target-abstract
tile-region 贯穿到 instruction-level tile-region。它仍然不带 physical offset、raw address、
packet 或 runtime handle。

## 3. Instruction Families

| family | V0 op | 来源 | 说明 |
| --- | --- | --- | --- |
| RDMA | `wafer.instr.rdma` | `wafer.load_tile` | DDR boundary value -> `#spm` tile buffer |
| WDMA | `wafer.instr.wdma` | `wafer.store_tile` | `#spm` tile buffer -> DDR boundary value |
| TDMA | `wafer.instr.tdma.gather_scatter` | `wafer.layout.materialize`、`wafer.move.*` | byte-counted movement；contiguous copy 是 descriptor 特例 |
| CT | `wafer.instr.ct.fill` | `wafer.compute.fill` | scalar/immediate fill |
| CT | `wafer.instr.ct.elementwise` | `wafer.compute.elementwise` | arithmetic / relation / activation |
| CT | `wafer.instr.ct.reduce` | `wafer.compute.reduce` | native local reduce |
| CT | `wafer.instr.ct.convert` | future convert lowering | dtype conversion |
| NE | `wafer.instr.ne.gemm` | `wafer.compute.gemm` | tile-local GEMM / batched GEMM |

V0 不定义 `wafer.instr.tdma.copy`。公开 SPM memcpy helper 本身也是
`TsmDataMove::GatherScatter` 样例；把 copy 单独做成 instruction op 会把 helper 名字提升为 IR
语义。

`ChannelNorm/DechannelNorm` 也不是 V0 单条 instruction op。它们是 layout materialization algorithm；
R3.2d 要么展开成一条或多条 `wafer.instr.tdma.gather_scatter`，要么结构化失败。

`TsmExecute` 普通 issue path 只覆盖 CT/NE/RDMA/WDMA/TDMA。DTE、CSR、SCALAR 走
`wafer.comm`、runtime/MMIO 或 sync 边界，不放进 `wafer.instr.*`。

## 4. Operand Model

instruction op 直接读写 `!wafer.tile_buffer`：

- source tile buffer 是 operand。
- destination tile buffer 也是 operand。
- 新 result/temp/psum/staging buffer 仍由 `wafer.alloc_tile` 创建。
- static reshape alias 仍由 `wafer.view.reshape` 表达。
- SPM offset、range、bank span 由 R3.2e 写入或在 R3.4 realization 降成 memref/descriptor。

RDMA/WDMA 的 DDR side 使用 tensor、memref 或后续 DDR descriptor boundary。R3.2d 不为 DDR side
新增 storage wrapper。

所有 `wafer.instr.*` op 应实现一个窄 interface：

```text
WaferInstructionOpInterface {
  getInstructionQueueFamily() -> InstrQueue
  collectInstructionEffects(...) -> tile-buffer read/write + queue issue
  verifyInstructionContract()
}
```

interface 只返回能从 op 本身和当前 IR 重算的事实，不返回 planner side table。

## 5. Lowering Rules

R3.2d 应实现为 MLIR DialectConversion：

- illegal：`wafer.load_tile`、`wafer.store_tile`、`wafer.layout.materialize`、
  `wafer.compute.*`、`wafer.move.*`。
- legal：`wafer.alloc_tile`、`wafer.view.reshape`、`wafer.instr.*`、`wafer.sync.*`、
  `wafer.tile_region` container 和必要 scalar/support op。
- no type conversion for `!wafer.tile_buffer`。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 committed 主线 IR。

V0 mapping：

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.load_tile` | `wafer.alloc_tile` + `wafer.instr.rdma` |
| `wafer.store_tile` | `wafer.instr.wdma` |
| `wafer.layout.materialize` | `wafer.alloc_tile` + one or more `wafer.instr.tdma.gather_scatter` |
| `wafer.compute.fill` | `wafer.instr.ct.fill` |
| `wafer.compute.gemm` | `wafer.alloc_tile` if needed + `wafer.instr.ne.gemm` |
| `wafer.compute.elementwise` | `wafer.alloc_tile` if needed + `wafer.instr.ct.elementwise` |
| `wafer.compute.reduce` | `wafer.alloc_tile` if needed + `wafer.instr.ct.reduce` |
| `wafer.move.copy/broadcast/transpose` | `wafer.alloc_tile` if needed + `wafer.instr.tdma.gather_scatter` |
| `wafer.move.extract_slice/insert_slice` | one or more `wafer.instr.tdma.gather_scatter` |
| `wafer.view.reshape` | stays as aliasing view; no instruction issue |
| `wafer.comm.*` | not handled by R3.2d V0 |

## 6. Verifier Contract

R3.2d verifier checks only instruction legality:

- `!wafer.tile_buffer` tensor type is ranked and static for V0.
- memory space matches instruction family:
  RDMA writes `#spm`; WDMA reads `#spm`; TDMA/CT/NE read/write tile-local buffers.
- NE GEMM and CT reduce require supported aligned layout, dtype and rank.
- GatherScatter/DMA descriptor attrs use byte count and byte stride, with non-negative static values.
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- no SPM offset/end/bank attrs before R3.2e.
- no DTE/CSR/SCALAR ordinary instruction op.

## 7. Example

```mlir
wafer.tile_region ... {
  %a = wafer.alloc_tile
      : !wafer.tile_buffer<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %b = wafer.alloc_tile
      : !wafer.tile_buffer<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %c = wafer.alloc_tile
      : !wafer.tile_buffer<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.rdma %arg0 to %a {byte_count = 16384 : i64}
      : tensor<128x64xf16>
     to !wafer.tile_buffer<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  wafer.instr.rdma %arg1 to %b {byte_count = 16384 : i64}
      : tensor<64x128xf16>
     to !wafer.tile_buffer<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.ne.gemm %a, %b into %c {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : !wafer.tile_buffer<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
        !wafer.tile_buffer<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    into !wafer.tile_buffer<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.wdma %c to %arg2 {byte_count = 32768 : i64}
      : !wafer.tile_buffer<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
     to tensor<128x128xf16>

  wafer.sync.local_drain
}
```

这是形态示例，不固定 parser/printer。真实 padded size、Cx/NCx 对齐、bool bitpack、
descriptor stride 和 SPM offset 分别由 verifier、SPM placement 和 later realization 处理。

## 8. Implementation Work

R3.2d 实现只需要：

1. 增加 `InstrQueue` 和必要 instruction kind enum。
2. 增加 `wafer.instr.*` ODS、verifier、MemoryEffects 和 interface。
3. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion。
4. 增加 supported op family 的 positive/negative tests。
