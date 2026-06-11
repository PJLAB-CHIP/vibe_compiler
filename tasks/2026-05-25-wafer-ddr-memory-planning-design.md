# Wafer DDR Memory Planning Design

2026-06-11 更新：R3.2g 的 compiler 合同收敛为 DDR memory planning gate。本文删除旧的
runtime 分配分类建模，不再把 driver/runtime 的分配种类提升成 Wafer IR attr 或 R3.2g
合法性分支。

本文定义 `#wafer.memory<ddr, layout>` 在 Wafer 编译器中的语义、资源检查和 lowering
边界。DDR 是 Wafer 可寻址的 global storage space；它和 SPM 使用同一套 Wafer memory attr
机制表达 address space 与 physical layout，但它不表达运行时分配对象的内部类别。

## 1. Goal and Non-Goals

目标：

- 让 R3.2g 从当前 instruction-level IR 重算 DDR access demand，而不是依赖名字、side table
  或历史 runtime 分类。
- 验证 RDMA/WDMA 的 descriptor payload、view byte range、root byte range、默认 DDR
  allocatable arena 的 capacity、largest contiguous range 和 bandwidth demand。
- 给 R3.2h 一个可重放的 candidate gate：成功表示当前候选的 DDR view/resource 已显式合法；
  失败返回结构化原因，供 candidate repair/split。
- 保持 DDR planned facts 由 SSA use-def、`#wafer.memory<ddr, layout>` memref type、
  `memref.subview` / static strided view、instruction descriptor 和 pass option resource bounds
  推出；成功路径不写重复 plan attr。

非目标：

- R3.2g V0 不做最终 runtime allocation，不生成 physical DDR address、runtime handle、ABI call
  或 package metadata。
- 不把 runtime/driver 的分配对象类别、host-visible window、executable/log storage 等低层事实建成
  Wafer compiler IR 类型或 attr。
- 不在 `wafer.group`、tile-region 或 instruction op 上保存 DDR allocation search trace。
- 不把 compiler-managed DDR temporary 的 owner/lifetime/range requirement 伪装成裸
  `memref.alloc`；当前 IR 表达不了时必须结构化失败。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2f 之后的 instruction-level IR。DDR operand 必须已经由
  `#wafer.memory<ddr, layout>` memref、tile-region block argument、
  `memref.subview` / static strided view 和 RDMA/WDMA descriptor 显式表达。
- Current stage responsibility:
  从当前 IR 重算 DDR access demand；验证 descriptor payload、static view expressibility、
  view/root byte range、default DDR allocatable arena 的 capacity、largest contiguous range、
  bandwidth limit，以及当前是否存在无法表达 owner/lifetime/range requirement 的
  compiler-managed DDR allocation。
- Output artifact / IR:
  同一份 instruction-level IR，或结构化 failure reason。成功不写 DDR plan attr、
  不写 runtime allocation object、不保存 search trace。
- Downstream consumer:
  R3.2h closed-loop candidate driver 用成功/失败作为候选 legality gate；
  R3.3/R3.4 只提交已通过 gate 的 DDR facts；
  R3.5 launch/runtime/package boundary 再把已通过的 DDR requirement 映射到运行时分配对象。
- User-level driver / named pipeline:
  局部 pass 是 `wafer-plan-ddr-memory`；
  主线验证入口是 `wafer-lower-groups-to-ddr-memory-planned-instr`，从 R3.2c/R3.2d/R3.2f
  重放到 R3.2g。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group
  boundary，不生成 ABI call、packet、physical address 或 runtime handle。
- Completion gate:
  R3.2f 输出的 tiled matmul/elementwise/storeback 能通过 DDR descriptor/range/capacity/
  largest-contiguous/bandwidth checks；非法 dynamic view、payload/range/resource failure 和
  当前无法表达的 compiler-managed DDR allocation 能被结构化拒绝。
```

## 3. Core Model

### 3.1 `#wafer.memory<ddr, layout>`

`#wafer.memory<ddr, layout>` 只说明这个 memref 是 Wafer 可寻址 DDR storage，并携带
physical layout marker。它不说明这个 storage 未来由哪个 runtime allocation path 创建，也不说明
host 是否可见。

允许的基本来源：

- function / tile-region boundary argument：由 launch/runtime 在更低层绑定或导入。
- `memref.subview` / view-like op：从已有 DDR memref 派生静态 view。
- future explicit DDR requirement op/interface：用于表达 compiler-managed DDR range、owner 和
  lifetime。R3.2g V0 尚未引入。

裸 `memref.alloc` 到 DDR 表达不出 owner、lifetime 和 range requirement，因此 R3.2g V0 拒绝它，
诊断为 `unsupported_compiler_managed_ddr`。

### 3.2 Default DDR Allocatable Arena

编译器侧只建模默认可分配 DDR arena 的资源边界：

```text
default_ddr_arena:
  base: runtime/placed stage 才能确定；R3.2g 不写入 IR
  capacity_bytes: 当前候选可用总量
  largest_contiguous_bytes: 单个 root/range 的最大连续可用空间
  alignment_bytes: target policy / descriptor preference，V0 默认由 view/descriptor 合法性覆盖
  bandwidth_limit_bytes: 当前 planning window 的 DDR movement byte budget
```

普通 tensor、workspace、temporary、resident constant 在 compiler 语义上都默认落到这个
allocatable arena。后续 runtime/package 层可以把它映射到真实运行时分配对象；这种映射不反向改变
R3.2g 的 IR 合同。

如果以后确实需要 host-visible/control/special arena，必须先引入明确的 compiler IR requirement
或 target policy 输入，并说明 verifier 如何检查。不能把 driver/runtime 名称直接塞成 generic
compiler attr。

## 4. Hardware and Runtime Facts Used by This Layer

R3.2g 只吸收会影响 compiler legality/cost 的稳定事实：

| fact | compiler meaning |
| --- | --- |
| 单卡 DDR 容量 64 GB / 128 GB | capacity resource bound 的来源之一 |
| 单卡 DDR 带宽约 200 GB/s | bandwidth cost / legality input |
| RDMA / WDMA 连接 DDR 与 SPM | descriptor byte count、stride、iteration 和 range 必须可验证 |
| public adapter DDR lower bound | placed descriptor / ABI lowering 的下游边界；R3.2g 不写 physical address |
| DDR2DDR / DTE preferred 256B alignment | cost / selected path constraint；不是所有 DDR view 的统一 hard rule |

runtime/driver 可能有多种内部 allocation path、visible aperture、executable/log/control buffer。
这些事实只在 launch/runtime/package 或 target policy 中使用；R3.2g 不把它们变成 Wafer dialect
allocation attrs。

## 5. IR Layer Boundary

| layer | carries | must not carry |
| --- | --- | --- |
| tensor / linalg / `wafer.group` | tensor shape、dtype、semantic layout、group boundary | DDR physical address、runtime handle、allocation object、DDR planning result |
| `wafer.tile.region` | DDR/SPM memref boundary、tile load/store、layout materialization、movement/effect | final DDR allocation, runtime-private allocation category |
| instruction-level IR | `wafer.instr.rdma/wdma` operands and descriptors over Wafer-tagged memrefs | duplicate DDR plan attr、runtime handle、packet field |
| R3.2g DDR memory planning gate | recomputed DDR demand, descriptor/range/resource verification, structured failure | tensor math semantics、SPM offset search、candidate search trace |
| launch / runtime / package | runtime allocation/import/query, physical address materialization, ABI/package metadata | group formation or layout search internals |

## 6. Demand Model

R3.2g does not consume a naked byte size. It reconstructs demand from current IR:

```text
DdrAccessDemand:
  root_value
  root_memref_type
  view_memref_type
  view_offset_bytes
  view_span_bytes
  root_physical_bytes
  role: read | write
  descriptor_byte_count
  descriptor_inner_bytes
  descriptor_strides
  descriptor_iterations
```

Demand recovery rules:

- RDMA source must be `#wafer.memory<ddr, *>`; destination must be `#wafer.memory<spm, *>`.
- WDMA source must be `#wafer.memory<spm, *>`; destination must be `#wafer.memory<ddr, *>`.
- tile-region block arguments are resolved back to the corresponding region operands.
- view-like chains are resolved with `ViewLikeOpInterface` until the root DDR memref.
- view/root memref shape, offset and strides must be static and non-negative.
- physical byte size uses `computeWaferPhysicalTensorInfo`, so Cx/NCx physical bytes, alignment padding
  and bitpacked limitations are handled consistently with other Wafer memory planning.

The scope summary keeps:

```text
DdrDemandSummary:
  unique_root_bytes
  movement_bandwidth_bytes
```

The summary is pass-local analysis and is recomputed from IR. It is not serialized into the IR.

## 7. Verification Rules

R3.2g verifies:

- descriptor payload: `byte_count == inner_bytes * iterations[0] * iterations[1] * iterations[2]`。
- descriptor local range: `inner_bytes + sum(stride_i * (iteration_i - 1))` must not overflow。
- local descriptor range must fit inside the DDR view span。
- `view_offset_bytes + descriptor_end` must fit inside the root DDR byte size。
- each unique DDR root demand must fit within `largest_contiguous_bytes`。
- total unique DDR root bytes must fit within `capacity_bytes`。
- total RDMA/WDMA DDR movement bytes must fit within `bandwidth_limit_bytes`。
- pass option resource limits must be non-negative。
- DDR roots created by bare `memref.alloc` fail until a proper owner/lifetime/range requirement is present。

R3.2g does not validate final physical address lower bound because physical address is not materialized yet.
That check belongs to placed descriptor / ABI lowering.

## 8. Failure Reasons

Current structured diagnostics:

- `unsupported_ddr_view`
- `descriptor_payload_mismatch`
- `range_end_overflow`
- `ddr_range_overflow`
- `memory_capacity_overflow`
- `largest_contiguous_range_too_small`
- `bandwidth_pressure_too_high`
- `invalid_ddr_resource_limit`
- `unsupported_compiler_managed_ddr`

These names describe compiler-visible failure classes. They intentionally do not mention runtime allocation
categories.

## 9. Interaction With Other Stages

### 9.1 SPM Memory Planning

SPM memory planning assigns offsets only for `#wafer.memory<spm, *>` memrefs. It may use DDR range and
bandwidth as cost/legality input, but it does not allocate DDR. DDR memory planning does not assign SPM
offsets.

### 9.2 Layout Materialization

Layout materialization may add DDR reads/writes or staging pressure. The inserted movement must still appear
as explicit DDR memref operands and descriptors so R3.2g can rederive demand. If a layout transform changes
physical bytes, the corresponding memref type/layout must make that visible.

### 9.3 R3.2h Candidate Driver

R3.2h enumerates tile/layout/resource candidates, then reruns R3.2e/R3.2d/R3.2f/R3.2g. A candidate rejected
by DDR memory planning is not written into the main IR; the driver may retry with a different tile shape,
layout cut, streaming/residency choice or split.

### 9.4 Launch / Runtime / Package

R3.5 materializes accepted DDR requirements into launch/runtime/package objects. It is responsible for actual
allocation/import/query, physical address assignment and ABI metadata. It must not redo tile/view/range
analysis by name; it consumes the explicit DDR facts already accepted by R3.2g/R3.4.

## 10. Example Shape

```mlir
%input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
  : memref<4x8xf16, #wafer.memory<ddr, tensor>>
 to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>

wafer.instr.rdma %input_tile to %spm
  {byte_count = 12 : i64, inner_bytes = 6 : i64,
   src_iterations = array<i64: 2, 1, 1>,
   src_strides = array<i64: 16, 0, 0>}
  : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
 to memref<2x3xf16, #wafer.memory<spm, tensor>>
```

R3.2g derives the root `%input`, the static view offset, the descriptor payload and the root byte demand
from this IR. It does not need a separate runtime-allocation attr to decide whether the access is legal.

## 11. Verification

Expected coverage:

- lit positive: explicit static DDR subview over external boundary passes without adding DDR plan attrs。
- lit negative: descriptor payload mismatch, descriptor/view/root range overflow, dynamic view, default arena
  capacity overflow, largest contiguous failure, bandwidth failure and unsupported compiler-managed DDR。
- text consistency: task/docs should not describe runtime allocation categories as R3.2g compiler IR attrs。
- build: TableGen and `wafer-opt` rebuild after removing obsolete attr definitions。

## 12. Deferred Work

后续恢复任务必须先扩 IR/interface，而不是复用旧 policy 名称：

- explicit compiler-managed DDR range requirement：owner、lifetime、required bytes、alignment、reuse
  relation 和 launch/runtime materialization contract。
- resident constant DDR requirement：read-only storage、packed backing data、streaming/residency choice 和
  package serialization boundary。
- optional non-default DDR arena：只有 ABI 或 runtime capability 需要时再引入，并作为 target policy /
  explicit requirement 表达。
- placed descriptor / ABI verification：physical address lower bound、alignment、range fence 和 runtime
  allocation mapping。
