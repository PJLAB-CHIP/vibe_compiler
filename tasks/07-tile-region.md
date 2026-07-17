# Wafer Selected Tile-Dataflow IR 与 Materialization

状态：本文按 physical-dataflow synthesis 终态边界定义 selected candidate 的显式 IR 和原子物化合同。
当前代码覆盖和实施状态只看 `tasks/progress.md`；历史 task-dataflow scheduling 证据位于 `tasks/archive/`，
不在本文复制为长期架构。

`wafer.tile.region` 是完整 rank traversal 中的结构化 task/traversal fragment。它可以组织 Wafer-tagged
memref、view、movement、target-abstract compute、communication 和 completion，但不是 fusion group、独立
SPM arena、DDR 切边、candidate、executable 或提交单元。跨 region 的 resident value 和 event 必须通过
显式 SSA operand/result 传递；region 边界本身不产生 store/reload。

本文依赖：

- `tasks/06-physical-dataflow-synthesis.md`：physical-dataflow synthesis、candidate state、搜索、排序和 atomic commit 的唯一 owner。
- `tasks/08-physical-realization.md`：physical encoding、transfer route、descriptor cover、immutable storage 和
  layout materialization provider。
- `tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`：完整 rank 上的 lifetime、容量和 offset gate。
- `tasks/10-compute-movement.md`、`tasks/11-instruction-ir.md`：target implementation family、instruction legality
  和 target-abstract 到 instruction lowering。
- `tasks/13-communication.md`：collective、Direct DTE 和 completion。

## 1. 职责和非目标

本层只负责把同一次 transformation 中已经选中的 physical-dataflow candidate 物化成普通、typed、可验证的
MLIR：

- 按 selected tile domain 生成 all-and-only static traversal、tail 和合法 reduction step。
- 应用 candidate 已选定且 proof guard 已通过的等价 rewrite；不在物化时重新搜索等价图。
- 把 selected implementation family instance 物化成 target-abstract compute/movement/communication op。
- 为每个 selected physical version 建立 Wafer-tagged memref、view、valid logical domain 和 SSA use-def。
- 把 resident、view、compute-absorbed、boundary transfer、local movement、staged movement、spill/reload 和
  immutable encoded resource 选择物化成显式 IR。
- 把 async provider、consumer、buffer reuse、collective 和 terminal effect 的 dependency/completion 物化成
  event 或现有 structured control-flow relation。
- 在 candidate clone 中失败原子化：任何 task、edge、traversal 或 verifier 失败都丢弃整个 clone。

本层明确不负责：

- 不生成 candidate，不选择 implementation、tile、encoding、residency、spill、buffering 或 task order。
- 不运行第二套 layout planner，不根据 lowering 失败临时改选 Tensor/Cx、插入 fallback movement 或切 group。
- 不保存搜索 frontier、cost、candidate id、失败历史、模型角色或 shadow schedule。
- 不分配 SPM/DDR offset，不重新判断 whole-rank resource feasibility，不 lower 到 packet/CRT/LLVM。
- 不把单个 region、first/tail representative tile 或单 rank 通过当成完整 variant completion。
- 不通过 op/value/parameter 名、固定 shape、参数顺序或 workload topology 恢复语义。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal、rank-specialized structured tensor IR，以及同一次 transformation 内由
  physical-dataflow synthesis 选中的 candidate。candidate 只引用当前 IR 可重算的 source op、
  indexing relation、implementation-family parameters、tile domain、physical versions、edge realization、
  buffering 和 dependency/order；它不是可序列化 artifact 或跨 pass side table。
- Current stage responsibility:
  在隔离的 complete-rank clone 中应用 selected semantic rewrites和tile traversal，物化 typed
  target-abstract compute/movement/communication task、Wafer-tagged memref physical versions、metadata view、
  mapped boundary transfer、local/staged movement、explicit spill/reload、immutable encoded resource use 和
  completion；保证 selected candidate 的全部决定都能由输出 IR 的 op/type/SSA/effect 直接验证。
- Output artifact / IR:
  覆盖完整 rank traversal 的 verifier-legal selected tile-dataflow candidate IR，或者结构化 failure。
  输出只包含正式 typed IR，不包含 planner frontier、Transform handle、候选评分或 opaque implementation id。
  通过 whole-rank/whole-variant exact gates并 atomic commit 后，该 IR 才成为 accepted execution fact。
- Downstream consumer:
  target-abstract compute/movement legalization、instruction lowering、whole-rank SPM planning、whole-variant
  DDR planning、event/transport/ABI verification、target-call/CRT/SystemC lowering 和 executable bundle commit。
- User-level driver / named pipeline:
  production 只由 source-to-bundle named pipeline 调用共享 C++ planner/materializer utility；可选
  Transform Dialect extension 只提供同一 utility 的开发期编排、复现和诊断入口，不能形成第二条
  production pipeline 或用户 stop-stage。
- Explicit non-goals:
  不做 cycle-accurate scheduling，不在本层搜索 candidate，不维护 group/fusion op，不按模型或固定 shape
  特化，不让 package/runtime重选 physical realization，不用 Transform IR 保存 execution semantics。
- Completion gate:
  semantic-family tests覆盖 contraction、pointwise、ordered reduction、view/permutation、broadcast/slice、
  fanout/fanin、多root、structured control flow和collective；每个在selected target profile与Q32 scope已注册的edge
  realization都有正负例；immutable prepack在02/15 owner闭合并另行排期前provider不得返回；
  candidate clone失败不污染source；完整rank输出能被下游exact instruction/SPM/DDR/event/transport/ABI
  gates直接消费；若实现Transform控制面，它与production调用同一materializer并得到等价payload IR。
```

## 3. 表示生命周期

```text
rank-local structured tensor IR
  + transformation-local selected candidate
    -> CandidateMaterializer on an isolated clone
    -> explicit target-abstract tile-dataflow IR
    -> per-rank instruction/descriptor/SPM/local-completion gates
    -> compatibility-bucketed bounded rank frontier
    -> lazy all-rank variant join + DDR/transport/ABI/artifact-eligibility preflight
    -> atomic commit of one complete variant
```

selected candidate 是一次 transformation 内的控制对象，只用于驱动物化。它不能进入 IR attr、bundle、package、
Transform handle payload 或全局 cache。物化后，长期事实只有：

- structured loop、tile offsets/sizes 和 SSA use-def；
- selected target op form和typed parameters；
- `memref<..., #wafer.memory<space, encoding>>`、metadata view和allocation root；
- explicit movement、spill、immutable resource use和event；
- 下游接受后写入的offset、transport和ABI facts。

如果某项 selected decision 不能用这些对象稳定表达和验证，必须先扩 IR/interface，不能把它留在 candidate
side table 让下游猜测。

## 4. `wafer.tile.region` 合同

概念形式：

```text
wafer.tile.region (...) -> (...) {
  ^bb0(%traversal_indices..., %region_args...):
    ... typed tasks, views, movement and events ...
    wafer.tile.yield ...
}
```

region 必须表达：

- region argument/result 与 enclosing rank data/event edge 的 SSA relation；
- 当前 traversal coordinate，或可由 enclosing `scf` 结构重算的 coordinate；
- local allocation root、logical shape/dtype、memory space和selected physical encoding；
- explicit external/immutable/spill boundary以及 local/staged movement；
- compute/communication provider和必要 completion/reuse ordering。

region 不表达：

- fusion/group identity、candidate id、planner score或cost trace；
- raw packet field、runtime allocation handle或package locator；
- case-specific K tile、psum lifetime、epilogue位置或模型角色；
- 跨 region 隐式 capture、隐式 DDR round-trip或隐式 SPM arena。

同一 physical SPM allocation root可以跨多个 region 传递，也可以在不重叠 lifetime 上复用。是否属于同一
physical-dataflow region 是 resident edge 的派生结果，不由 container 边界定义。

## 5. Buffer、Physical Version 和 Memory Space

tile-dataflow 中的长期 buffer value 使用 MLIR `memref`：

```mlir
memref<64x256xf16, #wafer.memory<spm, tensor>>
memref<64x256xf16, #wafer.memory<spm, cx>>
memref<64x256xf16, #wafer.memory<ddr, tensor>>
```

memref shape和element type表达logical shape/dtype；Wafer memory attr表达address space和selected physical
encoding identity。block、tail、padding、physical bytes和logical-index-to-physical-offset由tasks/08的统一
calculator推导，不重复写入type。

一个logical SSA value可以按fanout需要拥有多个physical versions。每个version必须是显式memref allocation root、
view或provider result，并通过SSA连接自己的consumers；禁止用“value只有一个layout”的隐式假设，也禁止用map从
logical value旁路绑定多个buffers。

padding不属于logical domain。06中的`InvalidLaneState`只用于candidate求解；物化选中的
pointwise/reduce/convert/movement task时必须把证明落实为下列正式IR之一：

1. typed valid-lane mode/physical count明确的full-physical执行，且producer effects能重建所需known-splat/neutral状态；
2. 带local offset/count的显式valid block/tail分段，只处理logical domain；
3. 携带closed `#wafer.fill_domain<physical_footprint>`的explicit `wafer.tile.fill` + segmented movement、mask，或candidate
   已经选定的其它encoding/movement；该domain由materializer显式写入，11只无损lower。

只写valid segments的transfer若没有先行fill，padding仍是unknown；materializer不能把analysis state复制成opaque attr来绕过
这个事实。10/11 verifier必须从selected op fields、descriptor和SSA producer重建invalid-lane pre/post condition。

`memref.alloc`只表达allocation identity和lifetime，不表达physical offset。generic `memref.load/store/copy`
不得作为Wafer-tagged SPM compute/movement的语义逃逸；metadata view只有在physical-isomorphism verifier通过时合法。

## 6. Selected Edge Realization 到 IR 的映射

每条producer-consumer edge只能物化为下面一种已选实现。该分类由tasks/06选择，本文只落实：

| selected realization | 显式IR要求 |
| --- | --- |
| `ResidentPhysicalVersion` | producer和consumer直接共享同一memref SSA/version；没有中间movement |
| `View` | 使用标准memref view或typed Wafer reindex/view；必须证明logical relation与physical storage同构 |
| `ComputeAbsorbed` | relation由selected compute op的typed orientation/indexing contract吸收；IR中保留原operand SSA/view和相应op字段，不产生movement |
| `BoundaryTransfer` | external compact view与selected SPM encoding之间的mapped load/store显式存在；source slice、destination version、valid domain和effect可验证 |
| `LocalMovement` | 使用`wafer.tile.materialize_layout`或其它typed local movement生成新的physical version |
| `StagedMovement` | 显式temp allocation、每段transfer/local movement和completion；不能隐藏在一个opaque op里 |
| `SpillReload` | typed DDR allocation/view、store、completion、load全部显式，consumer不能从名字恢复spill binding |
| `ImmutablePrepackedResource` | load引用typed immutable source relation和selected storage encoding；package只实现已选事实，不重选packing |

`ComputeAbsorbed`不意味着删除数学语义。比如permutation被contraction orientation吸收时，selected GEMM op必须
以typed `transA/transB`或等价稳定字段表达；不能依赖前序transpose op名字、buffer名或planner记录。类似地，producer
直接生成consumer encoding时，producer result type和implementation contract必须明确支持该encoding。

## 7. Target-Abstract Task 合同

每个task至少具有：

- typed input/output memref或scalar SSA；
- tile coordinate、logical valid sizes、static tail class，以及需要时的typed lane policy/physical offset/count；
- selected implementation的必要typed参数；
- numeric contract或可从source/current op重算的numeric relation；
- read/write effect、temporary/accumulator/staging value；
- async provider、consumer和reuse所需event。

selected implementation不能只编码为字符串id。对下游legality有影响的orientation、batch、accumulator、psum、
native/composite form、valid-lane policy和completion必须进入op form、type、attr、region或SSA relation。只影响搜索的
lower-bound cost、family枚举历史和失败原因不进入IR。

target op family由tasks/10拥有。本文不为每种dtype、layout、shape或模型角色定义新的tile op；参数化family实例在
物化时落成少量稳定op加typed字段。

## 8. Candidate Materialization Algorithm

物化过程是确定性transformation，不是第二个optimizer：

1. **建立isolated clone**：校验selected candidate引用的source op/value仍属于当前IR且analysis未失效。
2. **应用bounded semantic choice**：只应用candidate已选中且proof guard已验证的rewrite；检查rewrite后的result/use
   relation，不重新枚举alternative。
3. **物化traversal**：按selected tile domain生成compact `scf.for`、static tail和合法reduction sequence；all-and-only
   coverage在clone中可验证。
4. **物化implementation**：调用family materializer生成typed target-abstract compute/communication op，保存所有下游
   legality需要的参数。
5. **建立physical versions**：创建allocation root、metadata view、typed physical encoding和valid domain；用显式
   fill/segment/mask/provider effect使selected invalid-lane proof可从IR重建。
6. **物化edge realization**：按第6节生成mapped transfer、local/staged movement、resident SSA或spill/reload。
7. **连接completion**：显式建立provider completion、multi-input join、last-consumer/reuse、collective wait和terminal drain。
8. **局部canonicalization**：只删除semantics/physical storage/effect均等价的no-op view或dead operation；不得重新决定
   implementation、encoding、cut、residency或order。
9. **验证并返回clone**：任一引用失效、unsupported materializer、relation不一致或verifier failure使整个clone失败。

候选物化后，whole-rank SPM/DDR/event/transport/ABI exact gates仍可能拒绝它。失败反馈给tasks/06的bounded search选择
下一个candidate；本层不得就地修补已经失败的候选。

## 9. Structured Semantic Coverage

扩展性按semantic family组织，不按source op两两组合：

| semantic family | materializer读取的通用事实 | 输出责任 |
| --- | --- | --- |
| contraction | iterator types、indexing maps、DPS init、combiner和numeric policy | selected GEMM/batch/psum/orientation op、accumulator chain |
| pointwise/relation/select/convert | elementwise iterator、scalar DAG、dtype和valid-domain policy | selected CT/composite task；保持或生成selected physical version |
| ordered reduction | reduction iterators、identity/combiner、source order和reassociation许可 | ordered composite或qualified native variant；completion后才暴露result |
| view/permutation/reshape | normalized IndexRelation、alias/effect和physical-isomorphism proof | metadata view、compute absorption或explicit movement |
| broadcast/slice/concat | indexing relation、static/piecewise domain和coverage | view、mapped transfer、local movement或structured failure |
| immutable tensor source | ConstantLike value、logical slice和selected encoding | immediate/fill或typed immutable resource load |
| structured control flow | block arguments、yields、loop-carried values和effects | 保留`scf`结构并显式传递memref/event |
| collective | rank group、local rank、shape/bytes、combiner和completion | target-abstract collective或selected Direct DTE task graph |

新增source op优先通过现有MLIR structured interfaces归一到这些family；只有新数学语义无法稳定表达时才扩interface/op。
不得为模型角色、算子名字、固定shape或参数顺序增加materializer分支。unsupported语义必须结构化失败，不留下半转换IR。

## 10. Control Flow、Event 和 Complete Traversal

`scf.if`和`scf.for`可以作为tile-dataflow structured control-flow container。tensor loop-carried value物化为显式memref
physical version；ready/free event必须和buffer slot一起loop-carried。没有可验证slot/parity/completion关系时，dynamic
ping-pong继续fail closed。

block order不替代异步完成：

```text
provider issue
  -> provider completion
  -> all consumers
  -> join(last consumer completion)
  -> next writer / range reuse
```

rank entry返回前，所有external write、communication和其它可观察effect都必须terminally drained。完整traversal可以用
compact loop表达，不能通过只物化representative iteration规避coverage、lifetime或cost。

## 11. Transform Dialect 控制面

Transform Dialect只承担选择payload scope、调用共享utility、编排确定性cleanup/verifier和复现diagnostic的控制面职责。
production named pipeline和可选Transform extension必须调用同一组：

```text
PhysicalDataflowPlanner
CandidateMaterializer
ExactCandidateGates
```

Transform op可以接收isolated rank root、target profile和search budget，并返回更新后的root handle或结构化failure；它不能：

- 为每种op/dtype/layout/tile定义一个transform op；
- 用`transform.alternatives`展开beam frontier；
- 把physical-version graph、SPM live set、cost vector或selected candidate长期存入TransformState；
- 让Transform script成为package、bundle或accepted IR的必要解释器。

`CandidateMaterializer`必须是独立C++ utility。production pass/driver和Transform extension只是调用方，不能各维护一套
rewrite或fallback。

## 12. Verifier

tile-dataflow/tile-region verifier至少检查：

- region argument/result、terminator和enclosing structured traversal类型一致；
- selected implementation的operand/result、dtype、shape、orientation、numeric和target capability合法；
- 每个physical version的logical shape、encoding、valid domain、allocation root和view relation可验证；
- metadata view保持physical storage同构，不能用reshape逃避真实movement；
- 每条跨task edge要么共享同一version，要么存在显式且lowerable的selected realization；
- external host-visible boundary保持compact ABI，mapped transfer只改变device-side realization；
- 从producer/movement的typed lane policy、offset/count和effects重建每个version的invalid-lane state，并证明padding/tail bits
  不会被未授权compute、reduce、store或后续consumer观察；
- async producer在completion前不能被读取或复用，rank exit前terminal effects完成；
- candidate覆盖完整static traversal，reduction顺序和tail coverage符合source numeric contract；
- 输出不包含planner/Transform side table依赖、模型名matcher或opaque implementation payload。

whole-rank verifier继续检查SPM/DDR footprint和offset、descriptor cover、event/transport、target geometry/narrowing和ABI。
这些gate决定candidate能否提交，不由tile-region verifier判断candidate是否最优。

## 13. 通用示例

以下只说明selected IR关系，不固定shape或workload：

```text
compact external/immutable source
  -> mapped boundary transfer to physical version P0
  -> contraction variant consumes P0 with typed orientation
  -> pointwise variant keeps result physical version P1
  -> one fanout consumer aliases/views P1
  -> another consumer gets explicit local movement P1 -> P2
  -> final mapped store writes compact host-visible output
```

同一合同适用于普通GEMM/MLP、卷积lowering后的contraction、attention、MoE分支和一般structured graph。具体shape、
tile、transposition、prepack、resident edge和spill位置都来自selected candidate，不是本文协议。
