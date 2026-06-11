# Wafer DDR Memory Planning Design

2026-06-11 更新：R3.2g 重新收敛为 **compiler-side DDR memory planning**。它不能只是
DDR access validation；凡是会影响 candidate 是否成立的 DDR byte footprint、lifetime、capacity、
largest-contiguous 和 bandwidth 约束，都必须在 DDR offset assignment / candidate-selection gate 内决定或拒绝。
R3.5 只 materialize 已接受的 DDR offset facts 和 IR-derived demand 到 runtime allocation/import/package，
不重新做 planning。
同日进一步收敛：compiler-managed DDR allocation 由 DDR `memref.alloc` 本身表达；R3.2g 只把
accepted offset 写入 IR，size、alignment、lifetime、read/write intent 和 external access-end 都从
当前 IR 重算，不作为长期 attr 字段保存。

本文定义 `#wafer.memory<ddr, layout>` 在 Wafer 编译器中的语义、资源规划、verifier 和
lowering 边界。DDR 是 Wafer 可寻址的 global storage space；它和 SPM 使用同一套 Wafer memory
attr 机制表达 address space 与 physical layout，但 runtime/driver 的分配对象类别不进入
compiler IR 合同。

## 1. Goal and Non-Goals

目标：

- 从 instruction-level candidate IR 重算 DDR access demand 和 compiler-managed DDR allocation demand。
- 对 external input/output DDR view 做 descriptor、view/root byte range、capacity 和 bandwidth validation。
- 对 compiler-managed workspace、resident constant、inter-group DDR temporary 等非 external allocation，
  在 default DDR arena 中规划 symbolic range/offset/size/alignment，并用 lifetime/reuse 证明互不冲突。
- 给 candidate-selection 一个真实 candidate gate：成功表示当前 candidate 的 DDR view、accepted offset fact 和
  IR-derived demand 都可被下游直接消费；失败返回结构化 reason，供
  traversal tile / 当前支持的 matmul `K` split candidate repair 或 split。layout 替代候选、
  multi-output coverage 和 general reduction split 需要先有显式 IR/interface 语义。
- 保持 DDR accepted allocation fact 显式：由 SSA use-def、memref type、view、descriptor 和
  offset fact 表达，不能靠名字、fixture 或 pass-local side table 复原。

非目标：

- R3.2g 不生成 physical DDR address、runtime handle、ABI call、packet 或 package metadata。
- R3.2g 不调用 runtime allocator，不 import user buffer，不 query physical address。
- 不把 runtime/driver 的分配对象类别、host-visible window、executable/log storage 等低层事实建成
  Wafer compiler IR 类型或 attr。
- 不把 planner search trace、lifetime timestamp、read/write intent merge 或 external access-end 写成
  主 IR attr；这些都是可从当前 IR 重算的 analysis。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  SPM offset assignment 之后的 instruction-level candidate IR。SPM side 已有 accepted SPM offset facts；
  DDR side 已由 `#wafer.memory<ddr, layout>` memref、tile-region block argument、
  `memref.alloc`、`memref.subview` / static strided view 和 RDMA/WDMA descriptor
  表达 external / compiler-managed / resident / inter-group demand。
- Current stage responsibility:
  从当前 IR 重算 DDR access demand、compiler-managed allocation demand 和 lifetime；验证 external DDR
  descriptor 与 view/root range；为 compiler-managed/resident/inter-group allocation 在 default DDR
  arena 内规划 symbolic offset；验证 range overlap、capacity、largest-contiguous、alignment、bandwidth
  和 descriptor 对 planned allocation 的覆盖。
- Output artifact / IR:
  同一 instruction-level candidate artifact，compiler-managed DDR `memref.alloc` 带 offset-only
  `wafer.ddr.offset` accepted fact；
  或结构化 failure reason。成功路径不能只写 diagnostic，也不能只把 plan 保存在 pass-local
  analysis 里。
- Downstream consumer:
  candidate-selection 用 DDR offset assignment 成功/失败选择 candidate；
  R3.3 只 commit 已通过 candidate gates 的 candidate；
  R3.4 把 accepted DDR offset facts realize 成 placed memref/access descriptor；
  R3.5 把 accepted DDR demand materialize 到 runtime allocation/import/query/package metadata。
- User-level driver / named pipeline:
  局部 pass 是 `wafer-plan-ddr-memory`；
  主线验证入口是从 tile-region materialization、instruction lowering、SPM offset assignment 跑到
  DDR offset assignment 的 named pipeline。completion proof 必须覆盖
  accepted DDR offset fact 和 descriptor/view/root validation，不接受只验证 external DDR view。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group boundary，
  不生成 ABI call、packet、physical DDR address 或 runtime handle。
- Completion gate:
  simple matmul/elementwise candidate 中 external input/output DDR views、compiler-managed DDR temporary
  或 resident constant demand 都能获得可验证 planned offset；非法 dynamic view、payload/range、
  overlap/capacity/largest-contiguous/bandwidth/alignment failure 能结构化拒绝。
```

## 3. Core Model

### 3.1 `#wafer.memory<ddr, layout>`

`#wafer.memory<ddr, layout>` 说明 memref 是 Wafer 可寻址 DDR storage，并携带 physical layout
marker。它不说明 future runtime allocation path，也不说明 host 是否可见。

允许来源：

- external function / tile-region boundary argument：由 launch/runtime 在更低层绑定或导入。
- DDR `memref.alloc`：compiler-managed DDR allocation。owner 由 SSA definition 表达，lifetime 由
  SSA use-def、region/control-flow 和 async token use 重算。
- `memref.subview` / view-like op：从已有 DDR memref 派生静态 view。
- resident constant lowering：表达 read-only backing data、storage transform 和 lifetime。

DDR `memref.alloc` 不需要额外 requirement attr 才能参与 planning。alignment 来自 target policy 和
`memref.alloc` alignment；read/write intent 来自 RDMA/WDMA uses；size 来自 memref type 和 Wafer layout。

### 3.2 Default DDR Arena

编译器侧默认只有一个普通 DDR allocatable arena：

```text
default_ddr_arena:
  symbolic_base: physical base 由 R3.4/R3.5 placed/runtime stage materialize
  capacity_bytes
  largest_contiguous_bytes
  alignment_bytes
  bandwidth_limit_bytes
```

R3.2g 在这个 arena 内规划 symbolic offset/range。R3.5 之后才把 symbolic base + offset 映射到
runtime allocation object 和 physical address。

如果以后确实需要 host-visible/control/special arena，必须先引入明确的 compiler requirement 或
target policy 输入，并说明 verifier 如何检查。不能把 driver/runtime 名称直接塞成 generic compiler attr。

### 3.3 Accepted DDR Offset

R3.2g 成功后，compiler-managed DDR allocation 必须有 accepted offset fact：

```text
DDROffset:
  offset_bytes       // symbolic offset within default DDR arena
```

当前实现的 accepted fact spelling 是：

```mlir
wafer.ddr.offset = #wafer.ddr_offset<offset>
```

以下事实不写入 attr，因为它们可由当前 IR 或 target policy 稳定重算：

- `size` 来自 `computeWaferPhysicalTensorInfo(memrefType).physicalBytes`。
- `alignment` 来自 target DDR alignment policy 和 `memref.alloc` alignment。
- lifetime 来自 SSA use-def、structured region/control-flow 和 async token use。
- read/write intent 来自 RDMA/WDMA uses。

下游如果需要 byte range，应从 `offset + physicalBytes(memref type/layout)` 重算，不能依赖 pass-local
map、名字或 fixture。

External input/output 不由 R3.2g 分配 offset，也不写 external access summary attr。R3.2g 只在当前
candidate 中验证 descriptor/view/root byte range 和 bandwidth；R3.5 若需要 runtime binding metadata，应从
committed placed IR / descriptors 重算或在 runtime/launch 层 materialize。

## 4. Demand Classes

| class | R3.2g responsibility | R3.5 responsibility |
| --- | --- | --- |
| external input | validate view/range/descriptor in current candidate | import/bind runtime object, query physical address, validate size/alignment |
| external output | validate view/range/descriptor and write use in current candidate | bind output object, query physical address, validate writeback visibility |
| compiler-managed workspace/temp | plan symbolic offset with lifetime/reuse | allocate runtime object backing accepted offsets/ranges |
| resident constant | plan read-only range or reject if residency/streaming choice is not explicit | serialize/load backing data and bind physical address |
| inter-group DDR value | plan range across producer-to-last-consumer lifetime when explicitly represented | materialize backing allocation and package metadata |
| executable/log/control metadata | not generic tensor DDR planning | runtime/package internal allocation |

## 5. Demand Recovery

R3.2g reconstructs demand from current IR:

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

Rules:

- RDMA source must be `#wafer.memory<ddr, *>`; destination must be `#wafer.memory<spm, *>`.
- WDMA source must be `#wafer.memory<spm, *>`; destination must be `#wafer.memory<ddr, *>`.
- tile-region block arguments are resolved back to the corresponding region operands.
- view-like chains are resolved with `ViewLikeOpInterface` until the root DDR memref.
- view/root memref shape, offset and strides must be static and non-negative unless a future descriptor form
  explicitly supports dynamic bounds and verifier can prove them.
- physical bytes use `computeWaferPhysicalTensorInfo`, so Cx/NCx physical bytes, alignment padding and
  bitpacked limitations stay consistent with SPM planning and instruction lowering.

## 6. Planning Algorithm

R3.2g planning is an analysis + transformation pair:

1. Collect compiler-managed workspace/temp, resident constant and inter-group DDR allocation demands from
   DDR `memref.alloc`。
2. Build structured lifetime dataflow for those allocations before descriptor validation, so accepted
   `wafer.ddr.offset` facts are available when RDMA/WDMA roots are checked。
3. Collect external DDR access demands from RDMA/WDMA descriptors for validation only；external demand
   summaries remain pass-local analysis。
4. Compute physical bytes and alignment from memref type, Wafer layout and target policy.
5. Build lifetime intervals from SSA use-def, region/control-flow and async token/wait/drain effects. Current
   scope is one function/candidate artifact; broader inter-group lifetime requires explicit producer/consumer relation.
6. Build conflict edges for intervals that may overlap in time and require distinct DDR bytes.
7. Pack allocation demands into default DDR arena using deterministic interval packing. Reuse offset only when
   lifetime analysis proves non-overlap.
8. Validate each descriptor/view range against either the external root byte size or the planned allocation range.
9. Validate capacity, largest contiguous range, alignment and bandwidth.
10. Materialize accepted offset facts into the candidate artifact or return structured failure.

The search order, lifetime bounds, intent merge and rejected candidates are analysis. The accepted offset is the
cross-stage fact and must be explicit.

## 7. Verification Rules

R3.2g verifies:

- descriptor payload: `byte_count == inner_bytes * iterations[0] * iterations[1] * iterations[2]`。
- descriptor local range: `inner_bytes + sum(stride_i * (iteration_i - 1))` must not overflow。
- local descriptor range fits inside the DDR view span or planned allocation range。
- `view_offset_bytes + descriptor_end` fits inside the root DDR byte size。
- each planned offset respects required alignment。
- planned allocation ranges with overlapping lifetimes do not overlap in bytes。
- each planned allocation range fits within `largest_contiguous_bytes`。
- total live/planned DDR bytes fit within `capacity_bytes` under the selected arena model。
- total RDMA/WDMA DDR movement bytes fit within `bandwidth_limit_bytes` for the candidate window。
- pass option resource limits are non-negative。
- unsupported dynamic DDR alloc/view or uncomputable physical size is rejected。

R3.2g does not validate final physical address lower bound because physical address is not materialized yet.
That check belongs to placed descriptor / ABI/runtime lowering.

## 8. Failure Reasons

Required failure classes:

- `unsupported_ddr_view`
- `descriptor_payload_mismatch`
- `range_end_overflow`
- `ddr_range_overflow`
- `memory_capacity_overflow`
- `largest_contiguous_range_too_small`
- `bandwidth_pressure_too_high`
- `invalid_ddr_resource_limit`
- `unsupported_compiler_managed_ddr`
- `ddr_range_overlap`
- `ddr_alignment_failure`
- `ddr_planned_range_missing`

Diagnostics should describe compiler-visible failure classes. They must not mention runtime allocation category
names as if those were compiler IR concepts.

## 9. Interaction With Other Stages

### 9.1 SPM Memory Planning

SPM memory planning assigns offsets for `#wafer.memory<spm, *>` memrefs. DDR memory planning assigns
symbolic offsets for compiler-managed `#wafer.memory<ddr, *>` allocations. They share lifetime/effect reasoning
but do not allocate each other's storage.

### 9.2 Layout Materialization

Layout materialization may add DDR reads/writes or staging pressure. The inserted movement must appear as
explicit DDR memref operands and descriptors so R3.2g can rederive demand. If a layout transform changes
physical bytes, the corresponding memref type/layout must make that visible.

### 9.3 Candidate Selection

Candidate selection enumerates bounded traversal tile candidates, same-domain output coverage and currently supported
reduction/internal split candidates, then reruns candidate tile-view materialization, instruction lowering,
SPM offset assignment and DDR offset assignment. A candidate rejected by DDR planning is
not written into main IR. The driver may retry with a different tile shape or supported internal split; future
layout cut, different output domain coverage, streaming/residency choice and group split require explicit
IR/interface support before they become candidate dimensions. SPM/DDR arena and bandwidth limits are inputs to
their planning gates, not candidate fields.

### 9.4 R3.5 Launch / Runtime / Package

R3.5 consumes accepted DDR offsets and rederives external binding requirements from committed descriptors. It:

- allocates/imports runtime objects,
- queries physical base/size,
- validates runtime object capacity/alignment/intent against the committed IR-derived demand,
- fills package/launch metadata,
- reports runtime allocation failure without changing compiler planning decisions.

R3.5 must not redo DDR lifetime/range planning by name or by inspecting high-level tensor semantics.

## 10. Example Shape

External view validation:

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

Compiler-managed DDR is a DDR `memref.alloc` plus its SSA uses. R3.2g computes physical bytes, alignment,
lifetime and read/write role from the current IR, then writes only the accepted offset:

```text
allocation demand:
  bytes = physical_bytes(memref type)
  alignment = target DDR alignment
  lifetime = producer to last consumer
  role = read/write uses recovered from descriptors

accepted fact:
  arena = default_ddr
  offset = symbolic offset
```

## 11. Verification

Expected coverage:

- lit positive: explicit static external DDR subview passes descriptor/view/root validation。
- lit positive: compiler-managed DDR `memref.alloc` receives accepted offset and descriptor uses it。
- lit positive: non-overlapping lifetimes reuse DDR range; overlapping lifetimes do not；`scf.if`
  mutually exclusive branches reuse；`scf.for` loop-carried value extends lifetime。
- lit negative: descriptor payload mismatch、descriptor/view/root range overflow、dynamic unsupported view。
- lit negative: capacity overflow、largest contiguous failure、bandwidth failure、alignment failure。
- text consistency: task/docs must not describe runtime allocation categories as R3.2g compiler IR attrs。
- build: TableGen and `wafer-opt` rebuild after IR/interface changes。

## 12. Deferred Work

- Multi-arena DDR planning if ABI/runtime capability requires a real non-default arena。
- Cross-group DDR lifetime beyond a single candidate artifact, once producer/consumer relation is explicit。
- Board-validated runtime allocation failure mapping and recovery policy。
- PMU-calibrated DDR bandwidth model。
