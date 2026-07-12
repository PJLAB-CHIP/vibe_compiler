# Wafer DDR Memory Planning Design

状态：2026-07-12重基线；当前合同覆盖default-arena DDR demand/range validation和accepted offsets。
multi-arena、state/streaming weight和launch-facing resource model延后。实现状态以`tasks/progress.md`为准。
它不能只是 DDR access validation；凡是会影响 candidate 是否成立的 DDR
byte footprint、lifetime、capacity、largest-contiguous 和 bandwidth 约束，都必须在 DDR offset
assignment / candidate-selection gate 内决定或拒绝。
Q16只能从accepted DDR facts和当前IR demand重算launch-facing requirements，并在bundle commit时形成typed
C++ resource/entry bindings；当前没有独立resource-view协议或executable dialect。
post-commit target只派生address/range，package/runtime不重新恢复role/
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
- 对当前rank clone内compiler-managed workspace、resident constant、inter-group DDR temporary等non-external
  allocation，在default arena中规划symbolic range/offset/size/alignment，并用跨group、完整rank-entry
  lifetime/reuse证明互不冲突。
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
- 不因presumed rank equivalence相同就复用未验证的DDR plan或跳过任何显式rank entry。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  SPM offset assignment 之后的 whole-variant candidate clone，包含每个 static rank entry 的完整
  instruction-level structured program。SPM side 已有 whole-entry accepted SPM offset facts；
  DDR side 已由 `#wafer.memory<ddr, layout>` memref、tile-region block argument、
  `memref.alloc`、`memref.subview` / static strided view 和 RDMA/WDMA descriptor
  表达external view与当前rank default-arena compiler-managed/resident/inter-group demand。
- Current stage responsibility:
  从完整variant clone重算DDR access demand、compiler-managed allocation demand和跨group/rank lifetime；
  验证external DDR descriptor与view/root range；为compiler-managed/resident/inter-group allocation在当前rank
  default arena内规划symbolic offset；验证range overlap、capacity、largest-contiguous、alignment、bandwidth
  和descriptor对planned allocation的覆盖。streaming/state/multi-arena demand当前fail closed。
- Output artifact / IR:
  只存在于 complete passing clone 中的同一 instruction-level variant artifact，compiler-managed
  DDR `memref.alloc` 带 offset-only
  `wafer.ddr.offset` accepted fact，或结构化failure reason。成功路径不能只写diagnostic，也不能只把plan保存在pass-local
  analysis 里。
- Downstream consumer:
  candidate-selection 用完整 variant 的 DDR offset assignment 成功/失败选择 candidate；
  physical transport acceptance、launch projection和Q16 rank-record validation继续消费exact range；
  committed materialization只把已通过全部gates的complete variant及typed C++ resources/entry bindings原子写回。
  post-commit target/package/runtime不得从raw instruction IR重新恢复resource语义。
- User-level driver / named pipeline:
  Q16以后由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16}`的whole-variant
  candidate loop调用本stage。当前Q15只产出verified grouped program directory，不执行DDR planning；
  `wafer-opt`、局部`wafer-plan-ddr-memory`和从tile-region/instruction/SPM跑到DDR offset assignment的
  named pipeline只处理显式IR，用于IR-local replay/test，不是用户stop-stage。completion proof必须覆盖
  accepted DDR offset fact 和 descriptor/view/root validation，不接受只验证 external DDR view；该
  direct pipeline 仍只用于IR-local replay，用户级completion必须由whole-variant candidate-selection/
  commit pipeline 覆盖所有 static rank entries。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group boundary，
  不生成 ABI call、packet、physical DDR address 或 runtime handle；不按 group/rank 部分提交，
  不以 `busytable` 或 runtime 隐式同步替代 lifetime/completion 事实；当前不实现streaming/state、multi-arena、
  artifact materialization或shared-resource placement。
- Completion gate:
  每个static rank entry的完整traversal中external input/output与imported immutable parameter views通过
  range/capacity/access验证，compiler-managed temporary/resident backing/inter-group demand都在当前rank
  default arena获得可验证planned offset；跨group lifetime、
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

### 3.2 Current Default Arena Boundary

当前IR只有单个可推导的compiler-managed DDR offset domain，没有已实现的target-environment op、arena ID或
placement-domain协议。因此planner只在当前rank clone的default arena内检查capacity、largest-contiguous、
alignment、bandwidth和range/lifetime overlap；`wafer.ddr.offset`只是该domain中的arena-relative offset，
不是physical address或runtime allocation handle。

multi-arena、跨rank共享weight/state和host-visible/control arena必须等真实target/package/runtime consumer出现后，
再由typed execution capability与resource binding共同定义arena identity、sharing和base binding。此前相关workload
应fail closed，不能用attr字符串、文件名、driver名称或默认global arena冒充共享关系。

### 3.3 Accepted DDR Offset

DDR memory planning 成功后，compiler-managed DDR allocation 必须有 accepted offset fact：

```text
DDROffset:
  offset_bytes       // arena-relative offset in the current rank default domain
```

当前实现的 offset spelling 是：

```mlir
wafer.ddr.offset = #wafer.ddr_offset<offset>
```

Q16 typed rank record必须说明offset属于当前rank的default arena，并保留Q17 target address lowering需要的
range/alignment facts；在没有显式共享arena/base binding前，multi-rank shared-resource plan不能报完成。

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
launch-facing binding，Q16必须把上述验证结果与frontend parameter boundary、accepted transport/projection及
当前IR use-def交叉验证，再写入typed C++ rank record。atomic commit后的typed record是唯一owner；ABI/package/runtime
不得扫描instruction IR、offset attr、parameter-shard sidecar或薄launch binding在使用点重算role/range/scope。

## 4. Demand Classes

当前active contract只覆盖external IO/imported parameter view validation与default-arena compiler-managed
workspace/temp/inter-group allocations。下表中的streamed immutable、persistent state和多scope resource行为只作
future extension约束，不是当前planner或Q16 typed fields。

| class | DDR memory planning responsibility | committed executable resource responsibility |
| --- | --- | --- |
| external input | validate view/range/descriptor in current candidate | derive external binding requirement, shape/dtype/layout/size/alignment contract |
| external output | validate view/range/descriptor and write use in current candidate | derive output binding/writeback visibility requirement |
| imported immutable parameter/weight shard | validate read-only descriptor/view、content/shard/layout identity和range；不分配runtime-owned root offset | bind exact `ResourceId`/digest/shard requirement；禁止write alias |
| compiler-packaged resident immutable weight | plan complete read-only range in its declared arena；resident capacity不足时拒绝该candidate，不暗转streaming | summarize content digest、packing/quant descriptor、resident scope和backing bytes |
| streamed immutable parameter | 从typed consumer slices形成canonical source windows，创建bounded launch-visible staging roots，规划其range/lifetime并materialize copy/consumer/reuse completion | 形成`streamed_planned`：clone-local provisional source refs、source ranges、staging ResourceId/range、consumer slots和completion refs；materializer把digest/chunk proof放入同candidate transaction，final ArtifactRef/ID只在commit完成 |
| persistent state resource | validate declared capacity、subview range、alias/update relation、read/write effect和跨invocation lifetime；runtime-owned root不分配physical offset | preserve `ResourceId`、create/attach/reset/update policy、page geometry和exact consistency enum |
| compiler-managed workspace/temp | plan symbolic offset with lifetime/reuse | summarize workspace bytes/ranges and accepted offset contract |
| non-parameter resident constant | plan read-only range or reject if residency/streaming choice is not explicit | summarize resident constant bytes/ranges and backing-data requirement |
| inter-group DDR value | plan range across producer-to-last-consumer lifetime when explicitly represented | summarize producer/consumer-visible backing allocation requirement |
| executable/log/control metadata | not generic tensor DDR planning | launch/package internal resource requirement; runtime allocation is runtime adapter |

### 4.1 Streamed Immutable Window Contract

本节只记录未来extension约束，当前planner、Q16 bundle和active vertical gate均不实现streaming/state resource；
它不能作为当前artifact、typed field或completion事实。

streaming是whole-variant resource/lifetime候选，不是package读取优化。planner只消费当前IR中已显式存在的immutable
logical slice reads；它先按source `ResourceId`、artifact byte/chunk coverage、logical slice、storage encoding和
consumer `EntryId/SlotId`形成window proposals，再在arena capacity、alignment、bandwidth和completion lifetime下选择
有限staging lanes。每个accepted window必须：

- 用nonzero scoped `StreamWindowId`记录exact source range/chunk coverage和logical slice，packed/noncontiguous
  representation必须由typed storage descriptor给出，不能用tensor name或文件offset猜语义；
- 通过未来typed candidate-resource builder创建或验证launch-visible staging resource，获得显式destination range，
  并把consumer ABI slot绑定该staging resource而非source resource；
- materialize `resource_copy_issue/resource_copy_complete` node，copy complete支配全部consumer issue；最后一个
  consumer terminal支配同一lane的下一window copy，从而用普通DAG表达single/double/multi-buffer；
- 对所有consumer reads形成exact coverage，无hole/overlap歧义；同时live destination ranges之和不超过declared
  staging capacity，所有copy/source/destination range满足target alignment、address width和bandwidth gate。

planner不能为形成window而隐式切分一个尚需完整weight的kernel。若source slice需要新的K/internal reduction、expert
partition或partial-sum combine，必须先由group/op tiling和instruction IR显式表达数学等价、scratch和completion，之后
本stage只计划其可见slices。无法证明时结构化拒绝streaming candidate。若未来实现，accepted window和staging
offset必须由accepted IR/completion事实直接解释并进入typed C++ rank record，不能维护pass-local streaming schedule。

## 5. Demand Recovery

DDR memory planning reconstructs demand from current IR:

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

Demand recovery从当前candidate的memref SSA use-def、view relation和instruction effects推导role。external IO与
imported immutable parameter root由后续manifest/runtime绑定，本stage只验证其views/descriptors不越过静态root
range；compiler-managed workspace/resident backing才获得accepted offset。persistent state或其它无法从当前IR
解释的resource policy当前结构化拒绝。

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

1. 从DDR `memref.alloc`及SSA uses收集当前rank clone内compiler-managed workspace/temp、resident
   immutable backing、constant和inter-group allocation demands；当前全部属于default arena。
2. Build structured lifetime dataflow for those allocations before descriptor validation, so accepted
   `wafer.ddr.offset` facts are available when RDMA/WDMA roots are checked。
3. 收集external IO、imported immutable parameter和persistent state的RDMA/WDMA access demands做
   range/capacity/access/alias验证；这些runtime-owned roots不获得compiler offset。
4. streaming/state demand在当前实现中fail closed；不得临时创建staging root或旁路resource record。
5. Compute physical bytes and alignment from memref type, Wafer layout and target policy.
6. Build lifetime intervals from SSA use-def, region/control-flow and explicit async token/fence/wait effects，
   covering every complete static rank entry and mandatory inter-group producer-to-last-consumer relations。
   Group/tile-region boundaries do not truncate lifetime；every exit path must have no pending event after
   terminal drain。
7. Build conflict edges for intervals that may overlap in time and require distinct DDR bytes.
8. 在当前rank的default arena内用deterministic interval packing规划offset；只有lifetime analysis证明不重叠时
   才复用range。
9. Validate each descriptor/view range against either the external root byte size or the planned allocation range.
10. Validate capacity, largest contiguous range, alignment and bandwidth.
11. Materialize accepted offset facts only in the complete passing clone or return structured failure；facts
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
- 每个compiler-managed root属于当前rank default arena；当前不接受shared/multi-arena relation。
- planned allocation ranges with overlapping lifetimes do not overlap in bytes。
- each planned allocation range fits within `largest_contiguous_bytes`。
- total live/planned DDR bytes fit within `capacity_bytes` under the selected arena model。
- total RDMA/WDMA DDR movement bytes fit within `bandwidth_limit_bytes` for the candidate window。
- streaming/state demand在当前实现中必须结构化拒绝。
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
拒绝 candidate 时不写主 IR，并丢弃完整 clone。driver 可以重试其它 tile shape、supported internal split或已由
typed resource policy声明的resident/streamed residency；streaming只能使用本文件定义的window/staging/completion
合同。future layout cut、different output domain coverage和group split仍需显式IR/interface支持后才能成为候选维度。
SPM/DDR arena and bandwidth limits are inputs to
their planning gates, not candidate fields.

### 9.4 Q16 Typed Rank-Record Handoff

当前没有executable dialect或独立resource-view analysis对象。Q16在candidate commit前直接从current
instruction IR、accepted DDR/SPM offsets、topology/execution mesh、transport/projection和memref
use-def/view relation重算并验证：

- external IO、immutable parameter和workspace的role、access、scope、alias、default-arena range；
- compiler-managed DDR range与完整entry slot requirements；
- launch-visible facts与accepted offsets、descriptor ranges、parameter shards和completion的一致性。

验证只产生typed C++ `RankExecutable`字段，不把第二份resource record写回IR，也不allocate/import/query
runtime object或materialize physical address。all-rank records通过后才能形成Q16 `ExecutableBundle`。
Q17 target lowering结合committed instruction IR与这些bindings派生address/range/descriptor；Q18 manifest
只从Q16 bundle和Q17 verified `TargetArtifactBundle`序列化runtime-observable fields。package/runtime不扫描
raw instruction IR、offset attrs、program shard sidecar或薄launch metadata恢复resource。

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
  arena = current-rank default DDR domain
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
