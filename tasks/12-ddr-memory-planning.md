# Wafer DDR Memory Planning Design

状态：本轮长期边界合同已收敛；实现状态以`tasks/progress.md`为准。范围：compiler-side DDR memory planning、accepted DDR offset facts 和
whole-variant candidate legality gate。它不能只是 DDR access validation；凡是会影响 candidate 是否成立的 DDR
byte footprint、lifetime、capacity、largest-contiguous 和 bandwidth 约束，都必须在 DDR offset
assignment / candidate-selection gate 内决定或拒绝。
pre-commit `ExecutableResourceView` analysis从accepted DDR facts、typed program resources、arena/placement、
transport/projection和IR demand重算launch-facing requirements，并在atomic commit时materialize为typed
executable resources/entry bindings。post-commit target只派生address/range，package/runtime不重新恢复role/
alias/lifetime；本stage不执行runtime allocation/import/query，也不重新做planning。
compiler-managed DDR allocation 由 DDR `memref.alloc` 本身表达；DDR memory planning 只把 accepted
offset 写入 IR，size、alignment、lifetime、read/write intent 和 external access-end 都从当前 IR 重算，
不作为长期 attr 字段保存。

本文定义 `#wafer.memory<ddr, layout>` 在 Wafer 编译器中的语义、资源规划、verifier 和
lowering 边界。DDR 是 Wafer 可寻址的 global storage space；它和 SPM 使用同一套 Wafer memory
attr 机制表达 address space 与 physical layout，但 runtime/driver 的分配对象类别不进入
compiler IR 合同。

## 1. Goal and Non-Goals

目标：

- 从 whole-variant clone 中所有 static rank entries 的完整 instruction-level programs 重算 DDR
  access demand 和 compiler-managed DDR allocation demand。
- 对 external input/output DDR view 做 descriptor、view/root byte range、capacity 和 bandwidth validation。
- 对compiler-managed workspace、resident constant、inter-group DDR temporary等non-external allocation，
  在其declared `DdrArenaId`/placement-domain instance中规划symbolic range/offset/size/alignment，并用跨group、完整rank entry/variant-set
  lifetime/reuse 证明互不冲突。
- 给 candidate-selection 一个真实 candidate gate：成功表示当前 candidate 的 DDR view、accepted offset fact 和
  IR-derived demand 都可被下游直接消费；失败返回结构化 reason，供
  traversal tile / 当前支持的 matmul `K` split candidate repair 或 split。layout 替代候选、
  multi-output coverage 和 general reduction split 需要先有显式 IR/interface 语义。
- 保持 DDR accepted allocation fact 显式：由 SSA use-def、memref type、view、descriptor 和
  offset fact 表达，不能靠名字、测试输入或 pass-local side table 复原。
- 只在整个 static rank variant set 通过时原子提交 offsets；任一 rank/group/transport/event/target
  gate 失败都丢弃 clone，不把已通过的 DDR ranges 部分写回主 IR。

非目标：

- DDR memory planning 不生成 physical DDR address、runtime handle、ABI call、packet 或 package metadata。
- DDR memory planning 不调用 runtime allocator，不 import user buffer，不 query physical address。
- 不把 runtime/driver 的分配对象类别、host-visible window、executable/log storage 等低层事实建成
  Wafer compiler IR 类型或 attr。
- 不把 planner search trace、lifetime timestamp、read/write intent merge 或 external access-end 写成
  主 IR attr；这些都是可从当前 IR 重算的 analysis。
- 不把单 group/candidate artifact、representative tile 或 `DirectFullShape` 特判当作完整 DDR
  lifetime/capacity proof。
- 不因 distributed rank equivalence 相同就复用未验证的 DDR plan、跳过 rank entry 或提前创建最终
  executable rank class；该 equivalence 只属于上游 distributed semantics。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  SPM offset assignment 之后的 whole-variant candidate clone，包含每个 static rank entry 的完整
  instruction-level structured program。SPM side 已有 whole-entry accepted SPM offset facts；
  DDR side 已由 `#wafer.memory<ddr, layout>` memref、tile-region block argument、
  `memref.alloc`、`memref.subview` / static strided view 和 RDMA/WDMA descriptor
  表达 external / compiler-managed / resident / inter-group demand。
- Current stage responsibility:
  从完整 variant clone 重算 DDR access demand、compiler-managed allocation demand 和跨 group/rank
  lifetime；验证 external DDR
  descriptor与view/root range，并验证immutable parameter/persistent state的capacity/access/alias；为
  compiler-managed/resident/inter-group allocation在各自declared arena instance内规划symbolic offset；
  验证range overlap、capacity、largest-contiguous、alignment、bandwidth
  和 descriptor 对 planned allocation 的覆盖；共享 arena/resource policy 在 variant-set 级合并验证。
- Output artifact / IR:
  只存在于 complete passing clone 中的同一 instruction-level variant artifact，compiler-managed
  DDR `memref.alloc` 带 offset-only
  `wafer.ddr.offset` accepted fact；
  或结构化 failure reason。成功路径不能只写 diagnostic，也不能只把 plan 保存在 pass-local
  analysis 里。
- Downstream consumer:
  candidate-selection 用完整 variant 的 DDR offset assignment 成功/失败选择 candidate；
  physical transport acceptance、launch projection和pre-commit `ExecutableResourceView`继续消费exact range；
  committed materialization只把已通过全部gates的complete variant及typed resources/entry bindings原子写回。
  post-commit target/package/runtime不得从raw instruction IR重新恢复resource语义。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 的 whole-variant
  candidate loop 调用本 stage；局部 `wafer-plan-ddr-memory` 和从 tile-region/instruction/SPM 跑到 DDR
  offset assignment 的 named pipeline 只用于 stage replay。completion proof 必须覆盖
  accepted DDR offset fact 和 descriptor/view/root validation，不接受只验证 external DDR view；该
  direct pipeline 仍只用于 stage replay，用户级 completion 必须由 whole-variant candidate-selection/
  commit pipeline 覆盖所有 static rank entries。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group boundary，
  不生成 ABI call、packet、physical DDR address 或 runtime handle；不按 group/rank 部分提交，
  不以 `busytable` 或 runtime 隐式同步替代 lifetime/completion 事实。
- Completion gate:
  每个static rank entry的完整traversal中external input/output、imported immutable parameter、persistent
  state views都通过range/capacity/access验证，compiler-managed temporary/resident backing/inter-group demand
  都在正确arena获得可验证planned offset；跨group lifetime、
  async completion、descriptor/root range 和 shared physical geometry/range/narrowing contract 全部通过。
  非法 dynamic view、payload/range、overlap/capacity/largest-contiguous/bandwidth/alignment failure 能结构化
  拒绝，任一失败时整个 clone 不提交。DDR facts 只为 whole-variant commit 决定最终 executable rank
  class 提供输入；每个 rank entry 都必须实际进入 variant-set resource gate。
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

### 3.2 DDR Arena And Placement Domain

DDR allocation不能假设全模型只有一个默认arena。`wafer.target.environment`声明并fingerprint可用typed
arena records，candidate /
executable resource为每个root显式引用`DdrArenaId`和typed `placement_domain`：

```text
DdrArena:
  arena_id
  memory_domain
  placement_domain: per_rank | per_endpoint | per_stage | replica_group | executable_shared
  symbolic_base: physical base 由 target LLVM call emission 或 runtime adapter binding 派生
  capacity_bytes
  largest_contiguous_bytes
  alignment_bytes
  bandwidth_limit_bytes
```

每个DDR root必须映射到恰好一个arena。不同rank/stage的workspace默认使用不同`per_rank`/
`per_stage` arena instance；只有typed resource显式声明共享placement domain时，weights/state/inter-group
resource才可进入同一arena并合并冲突。planner只在相同arena instance内检查range/lifetime overlap，同时
在variant-set级验证共享resource coverage。ABI/RuntimeSession之后才把每个`DdrArenaId`的symbolic base +
offset映射到runtime allocation object和physical address。

host-visible/control/special arena必须来自typed target/resource requirement并有独立`arena_id`/memory domain；
不能把driver/runtime名称直接塞成generic compiler attr。

### 3.3 Accepted DDR Offset

DDR memory planning 成功后，compiler-managed DDR allocation 必须有 accepted offset fact：

```text
DDROffset:
  arena_id
  offset_bytes       // symbolic offset within the referenced arena
```

当前实现的 offset spelling 是：

```mlir
wafer.ddr.offset = #wafer.ddr_offset<offset>
```

长期candidate/executable contract还必须由typed resource/entry relation提供唯一`DdrArenaId`；在只有一个
可推导arena的现有实现里attr暂只打印offset，不代表长期允许implicit global arena。arena relation未落地前，
multi-rank/shared-resource plan不能报完成。

以下事实不写入 attr，因为它们可由当前 IR 或 target policy 稳定重算：

- `size` 来自 `computeWaferPhysicalTensorInfo(memrefType).physicalBytes`。
- `alignment` 来自 target DDR alignment policy 和 `memref.alloc` alignment。
- lifetime 来自 SSA use-def、structured region/control-flow 和 async token use。
- read/write intent 来自 RDMA/WDMA uses。

下游如果需要 byte range，应从 `offset + physicalBytes(memref type/layout)` 重算，不能依赖 pass-local
map、名字或测试输入。

External input/output、runtime-imported immutable parameter和persistent-state root不由DDR memory planning
分配offset，也不写external access summary attr。DDR memory planning只在当前candidate中验证
descriptor/view/root byte range、capacity、access/alias/update和bandwidth；ABI/package/runtime若需要
launch-facing binding view，应从 committed instruction IR、accepted offset facts、
topology/execution-mesh、program parameter shard metadata、薄 launch/block binding 和 descriptors
在使用点重算。

## 4. Demand Classes

| class | DDR memory planning responsibility | committed executable resource responsibility |
| --- | --- | --- |
| external input | validate view/range/descriptor in current candidate | derive external binding requirement, shape/dtype/layout/size/alignment contract |
| external output | validate view/range/descriptor and write use in current candidate | derive output binding/writeback visibility requirement |
| imported immutable parameter/weight shard | validate read-only descriptor/view、content/shard/layout identity和range；不分配runtime-owned root offset | bind exact `ResourceId`/digest/shard requirement；禁止write alias |
| compiler-packaged resident immutable weight | plan read-only range in its declared arena，或在residency/streaming policy不明确时拒绝 | summarize content digest、packing/quant descriptor、residency scope和backing bytes |
| persistent state resource | validate declared capacity、subview range、alias/update relation、read/write effect和跨invocation lifetime；runtime-owned root不分配physical offset | preserve `ResourceId`、create/attach/reset/update policy、page geometry和exact consistency enum |
| compiler-managed workspace/temp | plan symbolic offset with lifetime/reuse | summarize workspace bytes/ranges and accepted offset contract |
| non-parameter resident constant | plan read-only range or reject if residency/streaming choice is not explicit | summarize resident constant bytes/ranges and backing-data requirement |
| inter-group DDR value | plan range across producer-to-last-consumer lifetime when explicitly represented | summarize producer/consumer-visible backing allocation requirement |
| executable/log/control metadata | not generic tensor DDR planning | launch/package internal resource requirement; runtime allocation is runtime adapter |

## 5. Demand Recovery

DDR memory planning reconstructs demand from current IR:

```text
DdrAccessDemand:
  resource_id
  arena_id
  placement_domain
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

Demand recovery必须同时读取candidate typed resource declarations，不能仅从`memref.alloc`推断resource
role。external IO、imported immutable parameter和runtime-owned persistent state的root由RuntimeSession绑定，
本stage只验证其全部views/descriptors不越过declared bounds/capacity；compiler-managed workspace/resident
backing才获得accepted offset。persistent state的alias/update和exact `atomic_version`/
`in_place_poison_on_failure` enum属于executable resource，
本stage验证access与之相容但不重写policy。

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

DDR memory planning is an analysis + transformation pair:

1. 从typed resource relation和DDR `memref.alloc`收集compiler-managed workspace/temp、resident
   immutable backing、constant和inter-group allocation demands，并解析唯一arena/placement domain。
2. Build structured lifetime dataflow for those allocations before descriptor validation, so accepted
   `wafer.ddr.offset` facts are available when RDMA/WDMA roots are checked。
3. 收集external IO、imported immutable parameter和persistent state的RDMA/WDMA access demands做
   range/capacity/access/alias验证；这些runtime-owned roots不获得compiler offset。
4. Compute physical bytes and alignment from memref type, Wafer layout and target policy.
5. Build lifetime intervals from SSA use-def, region/control-flow and explicit async token/fence/wait effects，
   covering every complete static rank entry and mandatory inter-group producer-to-last-consumer relations。
   Group/tile-region boundaries do not truncate lifetime；every exit path must have no pending event after
   terminal drain。
6. Build conflict edges for intervals that may overlap in time and require distinct DDR bytes.
7. 按`DdrArenaId + placement_domain instance`分组，用deterministic interval packing规划每个declared arena；
   只有同一arena内且lifetime analysis证明不重叠时才复用offset。
8. Validate each descriptor/view range against either the external root byte size or the planned allocation range.
9. Validate capacity, largest contiguous range, alignment and bandwidth.
10. Materialize accepted offset facts only in the complete passing clone or return structured failure；facts
    become main-IR state only with whole-variant atomic commit。

The search order, lifetime bounds, intent merge and rejected candidates are analysis. The accepted offset is the
cross-stage fact and must be explicit.

## 7. Verification Rules

DDR memory planning verifies:

- descriptor payload: `byte_count == inner_bytes * iterations[0] * iterations[1] * iterations[2]`。
- descriptor local range: `inner_bytes + sum(stride_i * (iteration_i - 1))` must not overflow。
- local descriptor range fits inside the DDR view span or planned allocation range。
- `view_offset_bytes + descriptor_end` fits inside the root DDR byte size。
- each planned offset respects required alignment。
- each root references one valid arena/placement domain；只有同一arena instance内的ranges参与overlap，
  shared resources必须在所有引用rank/stage上使用同一`ResourceId`和compatible scope。
- planned allocation ranges with overlapping lifetimes do not overlap in bytes。
- each planned allocation range fits within `largest_contiguous_bytes`。
- total live/planned DDR bytes fit within `capacity_bytes` under the selected arena model。
- total RDMA/WDMA DDR movement bytes fit within `bandwidth_limit_bytes` for the candidate window。
- pass option resource limits are non-negative。
- unsupported dynamic DDR alloc/view or uncomputable physical size is rejected。
- physical footprint、view/root/descriptor range、offset arithmetic 和 target ABI width narrowing 使用与
  layout、instruction、SPM planning 相同的 shared physical geometry/range/narrowing verifier；拒绝 silent
  truncation 或 consumer-local geometry divergence。
- inter-group producer-to-last-consumer lifetime 是 mandatory gate；每条 function exit path 在显式
  completion 后都必须有空的 pending token/effect set。

DDR memory planning在physical base尚未materialize时只验证`arena_id + symbolic offset + span`、offset/span
算术、arena capacity/alignment和ABI offset/size field width；这些是commit前gate。RuntimeSession取得actual
allocation base后、任何launch/copy前，必须用同一geometry library验证base alignment、`base + offset + span`
不溢出target address width且落在runtime allocation object内。若pinned artifact在compile time已有真实base，
可以提前执行同一actual-base gate；否则不能要求commit前验证尚不存在的physical begin/end，也不能让
target LLVM用unchecked narrowing静默通过。

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
explicit DDR memref operands and descriptors so DDR memory planning can rederive demand. If a layout transform changes
physical bytes, the corresponding memref type/layout must make that visible.

### 9.3 Candidate Selection

Candidate selection 把 `DirectFullShape` 当作没有 fallback/bypass 语义的普通第一个候选，然后搜索
shape-driven traversal/reduction refinement frontier，验证 same-domain output coverage 和当前支持的
reduction/internal split candidates，并重跑 candidate tile-view materialization、instruction lowering、
whole-entry SPM offset assignment 和 whole-variant DDR offset assignment。first/tail representative tiles
只允许便宜地拒绝 candidate，不能证明 traversal coverage、lifetime、capacity 或 completion。DDR planning
拒绝 candidate 时不写主 IR，并丢弃完整 clone。driver 可以重试其它 tile shape 或 supported internal split；future
layout cut, different output domain coverage, streaming/residency choice and group split require explicit
IR/interface support before they become candidate dimensions. SPM/DDR arena and bandwidth limits are inputs to
their planning gates, not candidate fields.

### 9.4 下游 Resource View

target LLVM call emission、package metadata emission 和 runtime adapter 需要 resource facts 时，统一从 accepted
DDR offsets、topology/execution-mesh、program parameter shard metadata、薄 launch/block binding 和 committed
descriptors 重算 view。该 view：

- 从 committed IR 重算 external binding requirements。
- 汇总 compiler-managed workspace 和 resident/inter-group DDR ranges。
- 验证 launch-visible resource metadata 与 accepted offsets、descriptor ranges、
  topology/execution-mesh、program parameter shard metadata 和 launch/block metadata 一致。
- 供 target LLVM call emission、package metadata 和 runtime adapter 使用，但不成为新的 IR
  artifact。

该 view 不能 allocate/import/query runtime object，不能 materialize physical address，不能持久化第二份
metadata fact source，也不能靠名字或高层 tensor 语义重做 DDR lifetime/range planning。Runtime binding
和 allocation failure reporting 属于 runtime adapter。

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

Compiler-managed DDR is a DDR `memref.alloc` plus its SSA uses. DDR memory planning computes physical bytes, alignment,
lifetime and read/write role from the current IR, then writes only the accepted offset:

```text
allocation demand:
  bytes = physical_bytes(memref type)
  alignment = target DDR alignment
  lifetime = producer to last consumer
  role = read/write uses recovered from descriptors

accepted fact:
  arena = declared DdrArenaId  # V0 example may resolve to default_ddr
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
- text consistency: task/docs must not describe runtime allocation categories as DDR memory planning compiler IR attrs。
- build: TableGen and `wafer-opt` rebuild after IR/interface changes。

## 12. Deferred Work

- Additional arena classes and placement policies beyond the required typed arena/placement identity。
- Board-validated runtime allocation failure mapping and recovery policy。
- PMU-calibrated DDR bandwidth model。

跨 group DDR lifetime 不是 deferred work：只要 producer/consumer relation 已由当前 SSA、view、region、
explicit allocation 或 transport facts 表达，它就是 complete static rank entry / variant-set gate 的
mandatory 输入。关系无法表达时 candidate 必须结构化失败或先扩 IR，不能退回 per-group planning。
