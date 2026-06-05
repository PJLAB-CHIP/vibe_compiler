# Wafer Instruction IR Design

日期：2026-06-05

状态：R3.2d 设计收口；实现未完成

本文定义 R3.2d 的 instruction-level Wafer IR：`wafer.instr.*` 和同阶段产生的
`wafer.storage.*`。它位于 target-abstract `wafer.tile_region` 之后、SPM placement 之前。

`wafer.instr` 不是 raw packet IR，也不是 C ABI IR。它表达“这个 tile-local 动作已经选择成某个
硬件 issue family，并且它读写哪些 concrete storage value”。后续 SPM allocator、DDR/resource
planner 和 codegen 只消费这层 IR 暴露的 storage、queue、effect 和 lifetime，不再从
`wafer.compute.*` / `wafer.move.*` op 名字或单个 case 反推内存需求。

本文依赖：

- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
- `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2c `wafer.tile_region` IR，内部包含 accepted layout 的 `!wafer.tile_buffer`、
  `wafer.load_tile` / `wafer.store_tile`、`wafer.layout.materialize`、
  `wafer.compute.*`、`wafer.move.*`、`wafer.view.*`、`wafer.sync.*`，以及可验证的
  layout/materialization/effect relation。
- Current stage responsibility:
  对 target-abstract load/store、layout materialize、compute、movement 和 view/alias op 做
  Wafer instruction legalization / selection；把可执行动作改写成 instruction-level
  `wafer.instr.*`，并显式生成 unplaced `wafer.storage.*` SSA value、storage role、
  memory space、physical layout、queue family、read/write/issue effects、alias/view relation、
  temp/psum/staging storage 和 storage size policy。
- Output artifact / IR:
  同一个 `wafer.tile_region` execution scope 内的 instruction-level Wafer IR：
  `wafer.instr.*` + unplaced `wafer.storage.*`，或结构化 legalization failure reason。
- Downstream consumer:
  R3.2e SPM placement、R3.2f DDR/resource legality、R3.2g closed-loop planner、
  R3.4 placed storage realization 和 R3.6 codegen emission。
- User-level driver / named pipeline:
  主线由 R3.2 closed-loop planner 从 `wafer-opt --program-pipeline=stablehlo-spmd-to-group`
  的输出继续调用；局部 bring-up pass 可命名为
  `--wafer-convert-tile-region-to-instr`，只作为 lit/debug 入口，不能替代用户级 compile flow。
- Explicit non-goals:
  不决定 group boundary、tile shape、traversal、layout assignment、SPM offset、DDR BO binding、
  raw register packet field、Tsm wrapper call、C ABI symbol、package manifest 或 launch ABI。
  DTE、CSR 和 SCALAR 不进入普通 `wafer.instr` issue path。
- Completion gate:
  对 R3.2c 已支持的 load/store、layout materialize、fill、GEMM、elementwise/relation、
  reduce、copy/broadcast/transpose、static slice movement 和 static reshape view 生成
  verifier-legal instruction-level IR；每个 supported op family 都能通过 interface 枚举 storage
  values、queue family、effects 和 placement input。unsupported hardware instruction form 必须
  返回结构化 failure，不能让 SPM placement 从 target-abstract op 猜 demand。
```

## 2. IR Layer Boundary

R3.2d 之后仍保留 `wafer.tile_region` 作为 execution scope；变化的是 region body 内的 value 和 op
层级：

```text
target-abstract tile-region IR
  !wafer.tile_buffer
  wafer.compute.* / wafer.move.* / wafer.layout.materialize
  wafer.load_tile / wafer.store_tile / wafer.view.*

instruction-level tile-region IR
  !wafer.storage
  wafer.storage.external / alloc / view
  wafer.instr.rdma / wdma
  wafer.instr.tdma.*
  wafer.instr.ct.*
  wafer.instr.ne.*
  wafer.sync.local_drain
```

这一层只表达已经被 instruction selection 选中的硬件动作族：

| queue family | 普通 issue path | 来源 |
| --- | --- | --- |
| CT / CGRA | supported | elementwise、relation、convert、fill、reduce、部分 peripheral/memset |
| NE | supported | GEMM，后续 conv/pool 类 op 在 verifier 完整后加入 |
| RDMA | supported | DDR -> SPM load |
| WDMA | supported | SPM -> DDR store |
| TDMA / DataMove | supported | GatherScatter、layout materialization 展开的 concrete movement；SPM memcpy 也用 GatherScatter 表达 |
| DTE | not `wafer.instr` V0 | tile 间 communication；走 `wafer.comm` / communication lowering 独立边界 |
| CSR / SCALAR | not `wafer.instr` V0 | wait、CSR/MMIO、Kcore helper；不经 `TsmExecute` 普通 path |

`docs/wafer-register-level-instruction-spec.md` 已确认 `TsmExecute` 只分派 `inter_type=0..4`，即
CT/NE/RDMA/WDMA/TDMA。DTE、CSR、SCALAR 即使在 enum 中存在，也不应被 `wafer.instr.*` 当成普通
issue op；它们需要独立 API、MMIO 或 communication/runtime lowering 边界。

## 3. Storage Values

### 3.1 Storage Type

新增 storage value type：

```text
!wafer.storage<tensor_type, mem_layout, memory_space>
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `tensor_type` | ranked tensor type，记录 logical shape 和 element type；R3.2d V0 要求 static shape |
| `mem_layout` | physical layout family：`tensor`、`ntensor`、`cx`、`ncx` |
| `memory_space` | `spm` 或 `ddr` |

`!wafer.storage` 不携带 physical offset、raw address、DDR BO handle、packet field、ABI symbol 或
runtime handle。placement 状态来自 defining storage op 的 attrs：R3.2d 产生 unplaced storage；
R3.2e 在同一 IR 上填入 SPM placement facts。

storage size 不能由每个 op 私下估算。R3.2d 需要提供统一 storage-size calculator，输入为
`tensor_type + mem_layout + memory_space + instruction descriptor kind`，覆盖：

- compact `tensor` / `ntensor`。
- aligned `cx` / `ncx`。
- dtype storage bytes 和 bool bitpack。
- C0 tail/fold、256B line/layout padding、NHWC bank alignment。
- GatherScatter / DMA / TDMA descriptor 的 byte count 和 byte stride 单位。

如果 size 可以由 type/layout/instruction descriptor 重算，就不写重复 byte-size attr。只有硬件
descriptor 本身需要长期保留的字段才进入 instruction op attr。

### 3.2 Storage Ops

R3.2d 至少需要三个 storage op：

| op | 作用 | verifier / effect |
| --- | --- | --- |
| `wafer.storage.external` | 把 tile-region 边界 tensor、constant、workspace 或 output boundary 显式表示成 `#ddr` storage | memory space 必须是 `ddr`；`binding` 使用受控 enum；不分配 SPM offset |
| `wafer.storage.alloc` | 创建 instruction-level internal storage；R3.2d 只创建 unplaced storage | result 必须是 `!wafer.storage`；`role` 使用受控 enum；SPM offset/range attrs 在 R3.2e 前不能出现 |
| `wafer.storage.view` | 表示 aliasing view，例如 static reshape；不移动数据、不创建新 allocation | source/result element count 和 physical layout relation 必须可验证；must-alias source |

`storage.role` 是下游 placement 和 diagnostic 的语义字段，不是 planner trace。V0 role：

- `input_tile`
- `output_tile`
- `intermediate`
- `temporary`
- `accumulator`
- `psum`
- `layout_staging`
- `communication_staging`
- `constant`
- `workspace`

role 只说明 storage 在 instruction-level program 中承担的资源职责；它不能替代 SSA use-def、effect
或 alias relation。比如 `psum` storage 仍必须由 `wafer.instr.ne.gemm` 的 operand/use-def 明确使用，
不能只靠 role 名字让后端猜。

## 4. Instruction Ops

instruction op 使用 mutable storage operand 表达读写关系。destination storage 是 op operand，
不是 op result；SSA value 表示 storage object，MemoryEffects 表示该 storage 在某条 instruction 中
被读写。这样 SPM placement 可以从 use-def、effect 和 drain 计算 lifetime，而不是维护一份 shadow
schedule。

所有 `wafer.instr.*` op 都应实现统一 interface：

```text
WaferInstructionOpInterface {
  getInstructionQueueFamily() -> InstrQueue
  collectInstructionStorageUses(...) -> read/write storage operands and roles
  collectInstructionEffects(...) -> storage read/write + queue issue
  verifyInstructionContract()
}
```

interface 返回的事实必须从 op operands、attrs、result types 和 enclosing region 重新推出。不能返回
pass side table、cost-model 选择过程或未接受候选。

### 4.1 DMA Ops

| op | 语义 | 主要 verifier |
| --- | --- | --- |
| `wafer.instr.rdma` | `#ddr` storage -> `#spm` storage | source/dest memory space、element type、logical slice、byte count、byte stride、rank/stride descriptor |
| `wafer.instr.wdma` | `#spm` storage -> `#ddr` storage | source/dest memory space、host-visible output boundary、byte count、byte stride、rank/stride descriptor |

RDMA/WDMA 可以覆盖 contiguous 和 strided descriptor。上游 tensor stride 是 element stride；进入
instruction op 后所有 DMA / GatherScatter stride 单位必须是 byte。R3.2d 如果不能把某个 strided
movement 映射到硬件 descriptor，必须失败；不能把 element stride 留给 codegen 猜。

### 4.2 TDMA / DataMove Ops

| op | 语义 | 来源 |
| --- | --- | --- |
| `wafer.instr.tdma.gather_scatter` | byte-counted SPM movement；contiguous copy 是 stride/iteration 可折叠的特例 | `wafer.move.copy`、static extract/insert slice、general layout materialization |

`ChannelNorm/DechannelNorm` 是 layout materialization algorithm，不是 V0 `wafer.instr` 单条 op。
当前硬件资料显示 V0 主路径用 `TsmDataMove::GatherScatter` 实现它们，因此 R3.2d 应把这类
materialization 展开成一条或多条 `wafer.instr.tdma.gather_scatter`，或者在无法表达时结构化失败。
只有硬件或 stable wrapper 后续暴露可验证的单条指令形态时，才新增对应 `wafer.instr.tdma.*` op。

V0 不定义 `wafer.instr.tdma.copy`。公开 SPM memcpy helper 本身也是 GatherScatter wrapper 样例；
把 copy 单独做成 instruction op 会把 helper 名字提升为 IR 语义，并让 storage-size / descriptor
verifier 出现第二套事实源。

### 4.3 CT Ops

| op | 语义 | 来源 |
| --- | --- | --- |
| `wafer.instr.ct.fill` | scalar/immediate fill 到 destination storage | `wafer.compute.fill` |
| `wafer.instr.ct.elementwise` | arithmetic / relation / activation / transcendental elementwise | `wafer.compute.elementwise` |
| `wafer.instr.ct.convert` | dtype conversion with rounding / zero-point policy | future `wafer.compute.convert` 或 elementwise convert kind |
| `wafer.instr.ct.reduce` | tile-local native reduce | `wafer.compute.reduce` |

CT op 的 kind 使用受控 enum，不用 wrapper 函数名字符串。CT non-convert 默认要求输入/输出 dtype 组合由
verifier 明确支持；relation 输出 bool/i1 时，logical element type 是 i1，storage bytes 由 bitpack
规则计算。

### 4.4 NE Ops

| op | 语义 | 来源 |
| --- | --- | --- |
| `wafer.instr.ne.gemm` | tile-local GEMM / batched GEMM issue | `wafer.compute.gemm` |

`wafer.instr.ne.gemm` 必须显式记录 M/K/N、transpose、batch、accumulator/psum relation 和 optional
bias/activation/scale operand。V0 只要求覆盖当前 `wafer.compute.gemm` 能验证的 simple GEMM；如果
没有 bias、scale、activation 或 psum operand，就不要把这些字段写成默认融合协议。

NE aligned-only layout 是 verifier 规则：2D 默认 `cx`，rank > 2 使用 `ncx`。如果 target-abstract
GEMM 还没有 materialize 到可发 NE 的 layout，R3.2d 不能把它交给 SPM placement；应返回结构化
failure，让 planner 回到 layout/materialization 阶段。

### 4.5 Sync

保留现有 `wafer.sync.local_drain` 作为本地 CT/NE/RDMA/WDMA/TDMA issue 的 completion boundary。
普通 `wafer.instr.*` V0 不制造虚假的 per-instruction async token；当前硬件 wrapper 只暴露全局
wait/drain 语义，若 IR 造一个无法 lower 的细粒度 token，会把 lifetime 和 codegen 都做错。

SPM placement 在没有更细粒度 completion 事实时采用保守 lifetime：被 local instruction 读取或写入的
storage 至少 live 到下一个 `wafer.sync.local_drain`、tile-region 结束，或 verifier 能证明的更强
ordering boundary。后续如果硬件/runtime 暴露可验证的 queue event，再引入明确 event type 和 wait op。

## 5. Lowering From Target-Abstract IR

R3.2d 应实现为 MLIR DialectConversion：

- conversion target 将 R3.2c target-abstract op 标记为 illegal：`wafer.load_tile`、
  `wafer.store_tile`、`wafer.layout.materialize`、`wafer.compute.*`、`wafer.move.*`、
  `wafer.view.*`。
- type conversion 将 `!wafer.tile_buffer<tensor, layout, space>` 映射为
  `!wafer.storage<tensor, layout, space>`。
- conversion patterns 只消费当前 op 的 SSA operands、attrs、types、interfaces 和 enclosing
  `wafer.tile_region`。不读取 buffer 名字、fixture 名字、pass side table 或历史 case。
- legal output dialect subset 是 `wafer.storage.*`、`wafer.instr.*`、`wafer.sync.*`、必要的
  scalar/support op 和 `wafer.tile_region` container。
- failure 用 structured diagnostic / pass failure 返回给 R3.2 planner；rejected instruction IR 不落入
  committed 主线 IR。

Mapping V0：

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.load_tile` | `wafer.storage.external` + `wafer.storage.alloc` + `wafer.instr.rdma` |
| `wafer.store_tile` | source storage + `wafer.storage.external` + `wafer.instr.wdma` |
| `wafer.layout.materialize` | `wafer.storage.alloc` + 一条或多条 `wafer.instr.tdma.gather_scatter`，无法展开则结构化失败 |
| `wafer.compute.fill` | destination storage + `wafer.instr.ct.fill` |
| `wafer.compute.gemm` | result/psum storage + `wafer.instr.ne.gemm` |
| `wafer.compute.elementwise` | result storage + `wafer.instr.ct.elementwise` |
| `wafer.compute.reduce` | result storage + `wafer.instr.ct.reduce` |
| `wafer.move.copy` | result storage + `wafer.instr.tdma.gather_scatter` |
| `wafer.move.broadcast` / `transpose` | result storage + `wafer.instr.tdma.gather_scatter` 或结构化失败 |
| `wafer.move.extract_slice` / `insert_slice` | result/update storage + 一条或多条 `wafer.instr.tdma.gather_scatter` |
| `wafer.view.reshape` | `wafer.storage.view`，无 instruction issue |
| `wafer.comm.*` | R3.2d V0 不映射到 `wafer.instr`；communication instruction lowering 独立推进 |

## 6. Verifier Contract

R3.2d 的 verifier 至少覆盖：

- `!wafer.storage` 的 tensor type 必须 ranked；V0 要求 static shape。
- `mem_layout` / `memory_space` 必须使用 Wafer 受控 attr；不能用字符串。
- unplaced `#spm` storage 在 R3.2e 前不能带 physical offset、end address 或 bank span。
- `wafer.storage.view` 必须能证明 alias source/result 的 element-count、layout relation 和 memory
  space；不能用 view 表示真实 data movement。
- 每个 `wafer.instr.*` op 必须返回唯一 queue family；queue family 必须和 op 语义一致。
- read/write storage operand 的 memory space 必须符合 queue family：RDMA `ddr->spm`，WDMA
  `spm->ddr`，TDMA/CT/NE 只读写 verifier 允许的 tile-local storage。
- NE GEMM、CT reduce、Pool/UnPool 等 aligned-only 指令必须检查 `cx` / `ncx` physical layout。
- GatherScatter、DMA 和 strided movement 的 byte count、byte stride、iteration 必须是非负静态值；
  动态 descriptor 需要先扩 IR，不用名字或 side table 补。
- bool/i1 storage 的 logical element count 和 physical byte size 必须由统一 calculator 支持。
- `wafer.sync.local_drain` 不能被 implicit 插入到每条 instruction 后；它是显式 completion boundary。

Verifier 失败应说明缺的 IR fact 或违反的硬件约束，例如：

- unsupported layout transition；
- rank/dtype/layout 不满足 CT/NE/RDMA/WDMA/TDMA instruction form；
- dynamic shape/stride 当前无 descriptor 表达；
- DTE/CSR/SCALAR 被错误送进 ordinary instruction path；
- SPM placement-only 字段过早出现。

## 7. Example

下面片段只是 IR 形态示例，不固定 parser/printer 细节：

```mlir
wafer.tile_region ... {
  %a_ddr = wafer.storage.external %arg0 {binding = #wafer.ddr_binding_kind<input>}
      : tensor<128x64xf16> -> !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
  %b_ddr = wafer.storage.external %arg1 {binding = #wafer.ddr_binding_kind<input>}
      : tensor<64x128xf16> -> !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
  %c_ddr = wafer.storage.external %arg2 {binding = #wafer.ddr_binding_kind<output>}
      : tensor<128x128xf16> -> !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>

  %a = wafer.storage.alloc {role = #wafer.storage_role<input_tile>}
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %b = wafer.storage.alloc {role = #wafer.storage_role<input_tile>}
      : !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %c = wafer.storage.alloc {role = #wafer.storage_role<output_tile>}
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.rdma %a_ddr to %a {byte_count = 16384 : i64}
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
     to !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  wafer.instr.rdma %b_ddr to %b {byte_count = 16384 : i64}
      : !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
     to !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.ne.gemm %a, %b into %c {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : !wafer.storage<tensor<128x64xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
        !wafer.storage<tensor<64x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    into !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>

  wafer.instr.wdma %c to %c_ddr {byte_count = 32768 : i64}
      : !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
     to !wafer.storage<tensor<128x128xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>

  wafer.sync.local_drain
}
```

示例中的 byte count 只说明 instruction-level IR 使用 byte 单位。真实 padded size、Cx/NCx 对齐、
bool bitpack、descriptor stride 和 SPM offset 由 verifier / storage-size calculator / R3.2e placement
共同决定。

## 8. Implementation Organization

后续实现按 IR 层组织，不按任务号或单个 case 组织：

- shared attrs/types：
  - `WaferAttrs.td`：`InstrQueue`、`StorageRole`、必要的 instruction kind enum。
  - `WaferTypes.td`：`StorageType`。
- storage IR：
  - `include/Wafer/IR/Resource/StorageOps.td`
  - `lib/Wafer/IR/Resource/StorageOps.cpp`
- instruction IR：
  - `include/Wafer/IR/Instr/DMAOps.td`
  - `include/Wafer/IR/Instr/TDMAOps.td`
  - `include/Wafer/IR/Instr/CTOps.td`
  - `include/Wafer/IR/Instr/NEOps.td`
  - existing `include/Wafer/IR/Instr/SyncOps.td`
- interfaces：
  - extend `WaferInterfaces.td` with `WaferInstructionOpInterface` and storage-use helper structs。
- conversion：
  - `include/Wafer/Conversion/TileRegionToInstr`
  - `lib/Wafer/Conversion/TileRegionToInstr`

每新增一个 op family，必须同时补 ODS、parser/printer、verifier、MemoryEffects、
`WaferInstructionOpInterface` 实现、positive/negative IR tests 和 R3.2d conversion pattern。只添加
dump 或 shape-only test 不算完成。

## 9. Open Work

本文只收口 `wafer.instr` IR 设计。R3.2d 实现仍需完成：

1. 增加 `!wafer.storage`、storage ops、instruction attrs 和 instruction ops。
2. 增加 storage-size calculator 和 verifier。
3. 增加 `WaferInstructionOpInterface`。
4. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion。
5. 增加 R3.2c supported op family 的 positive/negative tests。
6. 把 R3.2e SPM placement 输入改为从 `wafer.storage.*` / `wafer.instr.*` 收集。
