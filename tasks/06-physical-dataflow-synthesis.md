# Wafer Physical-Dataflow Synthesis 与 Candidate Selection

状态：2026-07-17已收敛终态设计；实现由`tasks/progress.md`中的
`physical-dataflow-synthesis`任务跟踪。本文不定义或恢复`wafer.group`。

本文是 rank-local **physical-dataflow synthesis** 的唯一设计 owner。它联合选择等价计算形式、target
implementation、tile、physical encoding、storage realization、residency、buffering 和合法执行顺序，再把唯一选中方案
物化成 typed payload IR。07、08、10 分别拥有选中方案的 tile-dataflow IR、物理编码/传输路线和 target
implementation family；它们不维护第二个 planner。

当前 production 仍是 Q29 已完成的 bounded tile-dataflow scheduler：它使用确定性 source order，并在完整 rank clone
上比较 spill baseline 与 maximal full-buffer-resident alternative。该实现是迁移起点，不是本文终态算法。迁移期间不得
发布第二条 production pipeline；新 planner 必须先证明同一合法 baseline，再原子替换旧决策权，完成时删除 scope-prefix、
二选一 resident 后处理和 layout 独立决策旁路。

## 1. 核心结论

Wafer 的优化对象不是“把多少 graph op 塞进一个 group”，而是：

```text
rank-local structured semantics
  + numeric/effect policy
  + target capability families
  -> bounded physical-dataflow synthesis
  -> one explicit typed tile/dataflow program
  -> exact SPM/DDR/event/transport/instruction gates
  -> all-rank atomic executable bundle
```

关键原则如下：

1. **融合是结果，不是搜索变量。** producer/consumer 通过同一 SPM physical version 相连时即形成 resident
   dataflow；显式 store/reload、movement、effect 或数值 barrier 才切断它。IR 不增加 fused-kernel 名称或 group ID。
2. **layout 不是事后修补。** implementation、tile、encoding 和 transfer route 在同一个约束问题中选择；不先固定
   compute op，再插入一串 layout transform。
3. **目标相关、workload 无关。** planner 可以认识 Wafer engine、descriptor、SPM、Cx/NCx 和 event 能力，但不能认识
   Llama、Q/K/V、gate/up、参数名、文件名或固定 shape。
4. **搜索必须有合法 baseline 和硬上界。** 预算耗尽只降低优化质量，不能把合法输入误报为语义非法；所有候选均受相同
   verifier 和 exact resource gate。
5. **搜索状态不进入长期 IR。** relation、domain、Pareto frontier 和 cost trace 都是当前 IR 上可失效、可重算的 analysis；
   跨 stage 只保留选中方案的 typed compute、view、movement、buffer、event 和 offset。
6. **Transform Dialect 只作可选控制面。** production named pipeline 与可选 transform extension 调用同一 C++ planner/
   materializer；Transform IR 不承载 solver 内部状态，也不成为执行 artifact。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD 之后、按 logical rank 静态 specialize 的 verified Linalg/Tensor/SCF/Arith/Math
  structured tensor program；logical collective 已由 typed interface 表达。输入同时携带 validated
  ExecutionConfig、TargetProfileId、numeric policy 和可查询的 target capability provider。
- Current stage responsibility:
  从 structured iterator/indexing map、SSA use-def、shape/dtype、view relation、effect、control flow 和
  numeric policy 构造有限 semantic islands；按可证明等价规则和 target implementation family 惰性地产生
  decision domains，联合选择 implementation、tile、physical encoding、storage realization、residency、
  buffering 和 DAG-legal issue order；在 transformation-local complete rank clones 上物化候选并运行 exact
  instruction、SPM、DDR、completion、transport、geometry和14 registry/versioned-signature artifact-eligibility
  preflight（纯函数、不生成module/artifact）；对surviving candidates做有界
  Pareto selection，并原子提交一个 all-rank variant。
- Output artifact / IR:
  每个 rank 覆盖完整静态 traversal 的 accepted target-abstract tile/dataflow program；其中只含 typed
  memref/view、compute、movement、collective、buffer version 和 event relation。继续 lowering 后形成不含
  tensor/Linalg/search/group 残留的 instruction/memory/completion program；tasks/14 registry从winner最终instruction rows
  投影rank-local capability keys并形成all-rank canonical `RequiredCapabilitySet`，二者共同组成profile-bearing atomic
  ExecutableBundle。required set不是candidate/search state。
- Downstream consumer:
  07 物化 selected tile-dataflow IR；08 解释 selected physical encoding、view 和 transfer route；09/12 做
  exact SPM/DDR planning；10/11 lower target implementation 和 instruction；13 完成 transport；14/15 形成
  target artifact/package；16 做 completion evidence；17 的 SystemC/CModel 执行最终 target program。
  下游不读取 rejected candidate、search trace、Transform IR、历史 group 边界或 shadow plan。
- User-level driver / named pipeline:
  production 只经 wafer-compile 的 source-to-bundle named pipeline。可选 Wafer Transform extension 只能调用
  同一 planner/materializer library，用于受控实验、复现和诊断；不得形成不同语义或第二种 accepted artifact。
- Explicit non-goals:
  本 stage 不定义 framework 数学语义、SPMD partition、runtime manifest、raw packet/CRT ABI 或 cycle-accurate
  timing；不做 whole-model/unbounded equality saturation、任意 dynamic graph scheduling、名字驱动优化、模型专用
  pass、未获 numeric policy 许可的重结合，也不把 board 性能估计当 legality。首轮不承诺 dynamic-shape、动态
  multi-instance loop、未经 event 证明的 ping-pong 或全局 persistent weight cache。
- Completion gate:
  production planner 只消费 structured semantics/effect/numeric policy 和 target provider；parameterized family
  惰性实例化、constraint propagation、state canonicalization、hard budgets、baseline fallback 与 candidate-growth
  telemetry 均实际生效；selected proposal 只以 typed payload IR 跨阶段，旧 scope-prefix/layout-planner/
  maximal-resident 决策旁路清零。property、拓扑、多 dtype、多 workload、rank-count=1/16、7B scale、完整
  PyTorch/SystemC numerical、exact SPM/DDR/event/transport/ABI 和 atomic commit gates 全部通过。
```

## 3. 设计边界和稳定对象

### 3.1 Semantic island

`SemanticIsland` 是 planner 内部的有界 analysis scope，不是 IR op。它是 effect/control/numeric barrier 之间一段能从当前
IR 完整解释的 pure structured dataflow：

- roots、DPS inputs/inits/results 和 nested scalar regions 都可追溯；
- iterator domain、indexing relation、dtype、shape、broadcast/reduction 和 side effect 均已知；
- 外部 inputs/outputs、collective wait、observable store、unknown call 和 control-flow join 形成显式 boundary；
- scope 大小受 op/value/relation 数硬限制，超限时沿合法 boundary 切分并保留 baseline。

island 不是 fusion partition。planner 不枚举所有 partition；它先建立最大可分析 island，再由选中的 storage/movement/
effect edges 自然导出 resident components。

### 3.2 IndexRelation

`IndexRelation` 描述 consumer logical index 到 producer logical index 的关系，是从 structured IR 重算的 analysis 值：

```text
consumer index domain -> producer index domain + valid predicate
```

它至少覆盖 affine projected permutation、broadcast、static slice、reshape reassociation、concat piece 和 identity；支持组合、
逆关系存在性、image/preimage、bijective/injective/surjective 分类和 static box cover。动态或无法证明的关系保留显式
movement/spill baseline，不用 value/op 名恢复。

关系组合只服务：

- 判断 view 是否零拷贝；
- 把 consumer tile 反推为 producer dependent region；
- 验证多 operand pointwise 的 index 一致性；
- 计算 transfer descriptor cover；
- 证明等价改写覆盖 all-and-only logical elements。

`IndexRelation` 不复制 source IR 的长期事实，不写入 candidate attrs；08 负责把选中的关系解释为 physical map、view 或
movement。

### 3.3 ImplementationFamily

`ImplementationFamily` 是 10 中 target capability provider 暴露的参数化实现集合，不是按 op/dtype/layout 枚举的 case 表。
每个 family 声明：

- semantic kind 和 numeric contract；
- engine/instruction form 与 read/write/async effects；
- operand/result `IndexRelation` 约束；
- dtype、accumulator、rounding 和 exception domain；
- accepted physical encoding domain；
- tile geometry、alignment、tail、invalid-lane precondition和result transfer function；
- temporary/accumulator/resource requirements；
- parameter schema，例如 orientation、batch、vector width 或 algorithm variant；
- materializer 和 exact legality hook。

planner 先根据语义和粗约束获得 family descriptor，再在 frontier 中按需缩小参数 domain。它不得预先展开
`op × dtype × layout × tile × route × order` 的笛卡尔积。无法被 provider 资格化的能力 fail closed。

### 3.4 PhysicalEncoding

`PhysicalEncoding` 的详细合同由 08 拥有。本文只依赖其纯函数语义：

```text
phi(encoding, logical shape, dtype, logical index)
  -> physical bit offset + valid/padding domain
```

encoding 必须能回答 storage extent、alignment、block/tail、padding domain、view compatibility 和 transfer descriptor
cover；byte-addressable target field只能在bit offset可整除8后做checked byte投影，不能让bitpacked BOOL丢失byte内位置。
`Tensor`、`Cx`、`NCx` 是 target provider 的具体 encoding，不是 planner 内散落的字符串分支。

### 3.5 StorageRealization 和 PhysicalVersion

每条 logical value edge 最终选择一个 `StorageRealization`：

- `View`：相同 storage root 上的零拷贝 typed view；
- `ComputeAbsorbed`：index/layout relation 被选中 implementation 参数吸收；
- `ResidentPhysicalVersion`：producer/consumer 共享一个 SPM physical version；
- `BoundaryTransfer`：外部 DDR 与 SPM 间由 RDMA/WDMA/GS 等直接映射；
- `LocalMovement`：SPM 内显式 layout/repack；
- `StagedMovement`：直接路线不可覆盖时的有限 typed staging；
- `SpillReload`：producer 显式写 DDR、consumer 显式重载；
- `ImmutablePrepackedResource`：有稳定 owner、identity 和发布生命周期的 immutable physical payload。

这些名称是 planner 内部分类；accepted IR 只保留具体 memref/view/movement/compute/effect。每个 live physical version 由
`(storage root, logical tile region, encoding, valid domain, InvalidLaneState)`标识；其provider completion、consumer set和
last-use event可从selected IR重算。

`InvalidLaneState`是transformation-local有限小格，只区分`NoInvalidLanes`、`Unknown`和
`KnownSplat(typed raw value)`；若encoding存在固有piecewise padding class，也只能按hard-capped canonical segment class保存。
zero是`KnownSplat(0)`，某值是否neutral必须再结合implementation family、operand role和numeric policy判断，不能把“zero”
普遍等同“neutral”。bitpacked尾部未用bits同样属于invalid lanes。implementation family声明每个operand的
`requiredInvalidLanePredicate(role, state)`、access policy和result transfer function；route以`outState = inState`、
`Unknown`或`KnownSplat(v)`等有限transfer function声明写后状态。masked/unobserved属于consumer access proof，不进入buffer
内容state；incoming state不同的version不得canonical merge。

这个analysis状态不作为shadow attr进入accepted IR。selected candidate必须用显式fill、segmented valid-tail command、mask、
typed family valid-lane mode或完整physical write使verifier能从IR重建pre/post condition；例如只写valid segments的mapped load
产生`Unknown` padding，除非另有显式fill。无法把required neutral state物化为这些事实时，该choice不合法。

temporary compiler-owned layout tensor 不能伪装成 immutable resource。只有 package/artifact owner、logical identity、
encoding identity、read-only lifetime 和 invalidation 都明确时，才能选择 prepack。

### 3.6 Candidate

candidate 是一个 transformation-local complete proposal，至少包含：

- island 中各 root 的 implementation family 参数；
- collective roots的selected `CommunicationScheduleFamily`参数和all-rank compatibility constraints；
- shared tile-domain variables 和 dependent-region relations；
- values 的 selected physical encoding 与 storage realization；
- buffering slot 和 DAG-legal issue/order decisions；
- exact-materializable rank clone；
- legality status、resource vector、cost vector 和 deterministic tie-break signature。

candidate ID、score、beam parent 和 search trace 不写入 payload IR。accepted IR 是唯一事实源。

## 4. 等价变换合同

planner 只使用注册的、带 precondition 和 proof obligation 的通用规则。规则按强度分层：

### E0：bitwise index/view normalization

- identity view、连续 reshape reassociation、互逆 transpose/permutation、full static slice 等可证明关系的组合和消除；
- 不改变 logical iteration、运算次序、dtype 或 storage value；
- 只canonicalize logical relation；已物化storage间的movement仍须由08证明physical-isomorphism后才能删除；
- 使用唯一normal form和单调下降measure，无法证明下降时消耗明确rewrite fuel后停止，避免循环或无界表达式增长。

### E1：pointwise relation propagation

- pure pointwise op 可把同一 bijective relation 推过 result；
- 所有非 scalar operands 必须在 logical index 上一致，broadcast 必须显式证明；
- scalar body、rounding、NaN/Inf 和 conversion 顺序保持不变；
- padding lane 不属于 logical domain，除非 08/10 共同证明 selected implementation 的 padding invariant。

### E2：target-qualified implementation absorption

- transpose、orientation、vector packing 或 boundary mapping 只有在 family 明确提供参数并由 instruction/CRT/CModel 纵向
  资格化时才可吸收；
- 例如 GEMM transpose flag 不能因为底层 packet 疑似存在就进入 accepted candidate，必须由 10/11/14/17 的 typed
  capability 闭环支持；
- 吸收后的 logical result 和 numeric contract 必须与 source op 相同。

### E3：multi-root shared-input hyperedge

- 多个独立 contraction/compute roots 若共享相同 input region、numeric/effect 独立、tile domains 相容，可形成一个
  shared-load hyperedge；
- 该规则只共享 physical version、movement 和局部调度，不合并数学 result，也不要求固定 root 数量；
- fanout、不同 result shape 和不同 implementation family 通过 relation/constraint 处理，不匹配 QKV 或 gate/up 名称。

### E4：受 numeric policy 控制的代数变换

- reassociation、distribution、reduction tree、online reduction 或 FMA contraction 默认关闭；
- 只有 source IR 明确给出足够的 fast-math/overflow/identity 事实，且 target family 和 numerical verification 支持时才可
  进入 domain；
- 缺少许可时保持 source evaluation order 或 baseline，不用性能目标放宽正确性。

所有规则都必须能独立做 property/differential test。未注册的模型 pattern、固定 shape rewrite、operand-position 猜测和
字符串 role matcher 永远不进入 planner。

## 5. Tile、resident dataflow 与执行结构

### 5.1 空间和时间切分

空间 partition 已由 topology、execution mesh 和 Shardy/XLA SPMD 决定。planner 只处理一个显式 logical rank 的 local
program；Cx/NCx 是 rank 内 encoding，不是 mesh partition。全部 rank 独立 synthesis 和验证，即使代码相同也保留 typed
rank identity。

时间切分由以下变量共同决定：

- parallel iterator tile；
- 获 numeric policy 许可的 reduction chunk；
- producer/consumer 间的 dependent tile region；
- invariant physical version 的 reuse window；
- buffer slot、issue order 和 completion edge。

一个 task instance 由 source semantic root、logical tile coordinate、可选 reduction step 和 selected implementation
确定。主 traversal 使用 compact structured loops 和有限 static tail classes，不按 element 或所有静态 instance eager
展开。

### 5.2 Tile domain 生成

tile domain 只从通用事实产生：

- static extent 的 divisors、full extent 和 legal tail；
- target family 的 vector/block/geometry/alignment domain；
- dependent-region relation 的 compatibility；
- SPM lower bound、accumulator 和 staging threshold；
- descriptor count/cover threshold。

domain 以区间、divisibility、有限枚举和 relation constraints 保存。只有进入 frontier 的选择才惰性实例化具体整数；
不得生成从 1 到 extent 的全量 tile 表，也不得对每个 op 独立取 Cartesian product。

### 5.3 Resident dataflow 和 fusion result

selected IR 中：

```text
producer -> same SPM physical version -> consumer
```

表示 resident dataflow。producer 和 consumer 可以是不同 engine、不同 typed task、甚至位于不同
`wafer.tile.region`；region 边界不得自动 store/reload。以下情况形成 cut：

- observable external/ABI boundary；
- explicit spill/reload 或 staged movement；
- unknown/ordered effect 与 control-flow barrier；
- collective/provider completion 尚未闭合；
- relation、numeric、capacity、descriptor 或 instruction legality 不满足。

因此“最终融合组”只是从 selected physical versions 和 cut edges 派生的诊断视图，不是 allocator scope、candidate
identity 或 lowering protocol。

### 5.4 Dynamic loop 和 ping-pong

当前只接受可静态证明的 loop-carried version/lifetime。double buffering 必须物化两个不同 SPM slots，并明确
ready/free event、parity selection 和 last-use；若 dynamic multi-instance、nested tile-region capture 或 loop alias 不能从
IR 证明，planner 保留 single-buffer baseline 或 fail closed。不得以全局 summary 或 shadow loop schedule 掩盖缺失语义。

## 6. 有界联合搜索算法

### 6.1 总体流程

对每个 rank：

1. **Normalize**：执行 E0 canonicalization，建立 structured semantic graph、IndexRelation 和 effect/numeric barrier。
2. **Island formation**：建立最大有界 semantic islands；超限沿合法 boundary 切分。
3. **Capability query**：按 semantic root 查询 parameterized implementation、transfer route和communication schedule
   families，不展开组合；provider
   按canonical semantic signature排序，禁止依赖pointer、registration、DenseMap或parallel completion顺序。
4. **Backward region propagation**：从 outputs/roots 向 inputs 传播 dependent tile region，合并 shared-input constraints。
5. **Constraint propagation**：反复收紧 tile、family params、encoding、route、valid-domain 和 resource lower-bound domains；
   空 domain 立即剪枝。
6. **Reserved baseline**：每个production semantic family必须提供一个可判定的baseline implementation、canonical accepted
   encoding和由target geometry/capacity公式驱动的有界legality-directed safe-tile refinement；若provider没有baseline，该语义
   就不在该profile的production支持面，optimized family不得成为唯一正确性路径。为每rank baseline和all-rank baseline bundle
   保留不可被beam/top-K/optimization deadline淘汰的slot；先物化保持source order、external/boundary Tensor、family
   canonical encoding（如NE GEMM的Cx/NCx）、显式Tensor↔Cx/NCx materialization/spill的baseline，再完成分层exact gate。
7. **Pareto beam exploration**：baseline通过后，只对能消除movement、降低live bytes或启用合法implementation的frontier
   decision惰性分裂。
8. **Per-rank materialization/gates**：将bounded proposals写入隔离complete-rank clones，运行instruction/descriptor、SPM、
   local completion和per-rank可独立证明的DDR view/range gate，fresh recost后按all-rank compatibility signature分桶。
9. **Lazy variant coordination**：先传播shared transport variables，再baseline-first、best-first/factorized地惰性join各rank桶；
   不预先形成`K^R`。每个完整variant再运行whole-variant DDR/package eligibility、all-rank transport和14 registry/
   versioned-signature artifact-eligibility preflight；该pure preflight可临时投影prospective capability keys，但不进入frontier、
   accepted IR或artifact，且不生成module。winning Q16 commit后才从fresh final instruction rows形成canonical set，14只正式lower一次。
10. **Selection/commit**：只在同compatibility signature内做local Pareto dominance；从全部passing variants确定性选择winner，
    fresh recost，从winner final instruction IR重算每rank capability projection与all-rank set/digest，并与全部rank一起原子发布。

### 6.2 Canonical frontier state

beam state 只保留继续求解所需的最小规范形式：

```text
graph frontier
+ live physical versions:
    (storage root, tile region, encoding, valid domain, invalid-lane state)
+ constrained implementation/tile/route domains
+ SPM live signature and lower bound
+ engine/effect/completion frontier
+ all-rank compatibility signature
+ static resource/cost vector
```

已经完成且不再影响 future liveness/constraints 的历史决策不进入 key。等价状态按 relation、domain 和 live signature
canonicalize 后合并。cache使用完整规范化query signature：semantic/index relation、shape/dtype、tile/valid domain、source/
destination memory space与view offset、normalized alias/effect signature（same-root relative overlap、proven-disjoint或
unknown/may-alias）、family/route/encoding参数、按全部读写operand角色记录的incoming source/destination invalid-lane states
和target profile；unknown alias只可复用保守
fail-closed结果，不含op/value/
模型名。`all-rank compatibility signature`至少含collective/message bytes与segmentation、peer/order/completion和shared remote
buffer/resource requirements；不同signature不得做local dominance或canonical merge。即使signature相同，per-rank local
dominance也只比较whole-variant聚合函数已声明为componentwise monotone（例如sum/max）的exact dimensions；非单调或跨rank
目标进入compatibility signature或延迟到完整variant后比较。

### 6.3 惰性 decision 和候选顺序

planner 不枚举所有 fusion partitions、resident subsets、tile integers、physical versions 或 topological orders。decision
按预期收益/约束强度排序：

1. 能 canonicalize 或 compute-absorb 的 relation；
2. 大 boundary movement 的 direct route / resident opportunity；
3. shared-input reuse hyperedge；同relation/domain的全部compatible roots先形成deterministic maximal hyperedge，冲突只允许
   hard-capped domain split，不枚举`2^fanout` compatible subsets；
4. SPM peak 或 descriptor legality 所迫使的 tile refinement；
5. 少量能改变 reuse distance 的 ready-task alternatives；
6. optional buffering。

task scheduling 使用 deterministic resource-aware list scheduling。默认 tie-break 来自稳定 IR order；只有 ready set 中
resource/reuse signature 不同且仍有预算时，才分裂有限 alternative。绝不枚举 `N!` topological orders。

### 6.4 Hard legality gates

cheap constraints 只能提前剪枝，不能替代 materialized exact gate。per-rank gate为：

1. source semantic、numeric/equivalence、shape 和 dtype；
2. implementation family、encoding、tail 和 padding；
3. transfer relation、descriptor cover、offset/range/alignment；
4. whole-rank SPM alias/liveness/peak；
5. local provider completion、event join、buffer reuse和terminal drain；
6. per-rank instruction geometry、narrowing及可独立证明的DDR view/range。

只有完整all-rank variant才能继续验证：

7. whole-variant DDR demand/range/package；
8. all-rank collective/transport matching和shared resource closure；
9. versioned target ABI、artifact和package preflight。

任何未知能力或不完整 proof 均拒绝该优化 choice，而不是降低 verifier。baseline 若也失败，才报告 workload/target
不支持，并给出最早失败的 typed diagnostic。

### 6.5 Hard budgets 和 fallback

所有预算来自 compiler option/profile policy，但不进入 bundle identity 或 IR：

- 每 island 的 op/value/relation node 上限；
- 每 semantic root 的 family/parameter instantiation 上限；
- IndexRelation compose/normal-form rewrite fuel、piecewise image/preimage box和dependent-region fragment上限；
- descriptor segment split/fuel上限，以及shared-input hyperedge conflict split上限；
- tile refinement、ready-set alternative、beam width 和 exact top-K 上限；
- 每 rank 与 all-rank combination 的 candidate 上限；
- deterministic work/fuel/candidate caps、analysis cache memory上限，以及独立外部wall-time cancellation。

baseline使用独立预留的materialization/gate slot和hard CPU/memory/fuel/wall allowance，先于optimized exploration执行；
optimization work budget不得淘汰它，但baseline allowance本身也有总资源硬上界。若连baseline proof都在该allowance内
无法完成，返回明确`compiler_resource_exhausted`，不能标成target/workload illegal。baseline gate完成后，独立的optimization
work/fuel/candidate预算无条件生效，包括“已有候选仍在并行评估”的情况。生产可复现选择只由这些deterministic caps决定；
wall clock只作外部保护/取消，触发后丢弃所有optimized survivor、无条件返回reserved baseline并报告
`deadline_exceeded`，不按恰好完成的并行任务选择。跨线程determinism在相同work policy且未触发外部取消时验证。
optimization预算耗尽时：

- 停止新增 optimized states；
- 完成已 materialize 的 bounded exact gates；
- 若 baseline 合法则返回 baseline，并发出结构化 budget diagnostic；
- 若 baseline 不合法则返回真实 legality failure，不伪装成 timeout。

telemetry 至少记录每阶段 input states、domain reductions、deduplicated/pruned/materialized/passing counts、peak beam、
budget reason 和 baseline/selected cost vector。它是诊断输出，不是下游协议。

### 6.6 原子提交

candidate materialization、canonicalization、bufferization、physical-memory replanning 和 recost 全部在隔离 clone 上进行。
一个rank可按compatibility signature保留有限surviving frontier；all-rank coordinator先传播shared constraints，再用
baseline-first lazy/factorized join访问hard-capped完整组合，绝不先构造rank-frontier笛卡尔积。每个完整variant重放
transport/resource/ABI gate。只有winning variant的全部ranks同时通过后，才由tasks/14 registry从各rank final instruction
rows派生canonical `RequiredCapabilitySet`、验证all-and-only union/digest，并与source program替换及ExecutableBundle一起原子提交。
Q22.L target conversion从实际发射的TargetCall rows重算rank-local投影，Q17 readback并携带global set，Q18 schema-v4只序列化该set；
这些不是planner choice，也不能反向参与搜索。

局部 tile、单 op、单 rank、代表 rank 或部分 offsets 永不提交。rejected clone 不得泄漏 memref type、offset、event 或
resource mutation。

## 7. Cost、排序和板端校准

legality 与 cost 完全分离。板端校准前使用可从 selected IR 精确统计的 Pareto vector：

```text
(
  peak_spm_bytes,
  ddr_read_bytes,
  ddr_write_bytes,
  spm_movement_bytes,
  transport_bytes_by_route,
  transport_message_counts_by_route,
  compute_class_counts,
  transfer_descriptor_counts_by_engine,
  inner_contiguous_bytes_distribution_by_engine,
  route_temporary_peak_bytes,
  engine_command_counts,
  instruction_issue_counts,
  fence_wait_event_counts,
  padding_work,
  immutable_payload_bytes
)
```

`transport_*`同时覆盖NoC/DTE/collective和其它已注册transport route；08返回的physical read/write、descriptor、inner-span、
temporary、engine command和event metrics按上述同名维度聚合，09/12返回peak与DDR/SPM事实，13返回transport
bytes/message事实。unknown dimension 不得伪造为 0。先做 Pareto dominance，再用稳定 profile policy 作 deterministic
beam/tie-break；文档和diagnostic必须同时保留向量，不能把未校准 scalar estimate 宣称为硬件时间。

只有 IR/event 能证明 overlap，且对应 engine/queue/resource 能力已资格化时，resource DAG scheduler 才允许重叠；否则
保守串行。板端 Q9 只用 PMU/带宽/latency 数据校准合法候选的排序权重和 overlap 模型，不改变 semantic、numeric、
descriptor、capacity 或 ABI legality。

对外发布性能结论时分三层：

- 静态 IR 事实：bytes、peak、descriptor、issue、event；
- CModel wall time：host functional execution 性能，不代表板端；
- 板端实测：仅在配置、版本、PMU 和统计方法完整时报告。

## 8. Transform Dialect 的定位

MLIR Transform Dialect 对本设计有帮助，但它只解决 orchestration、匹配和可复现控制，不替代联合 solver。

推荐结构：

```text
production named pipeline ─┐
                           ├─> shared C++ physical-dataflow planner/materializer
optional Transform op  ────┘                 |
                                             v
                                  typed payload IR + diagnostics
```

可选扩展可以提供一个 coarse-grained `transform.wafer.synthesize_physical_dataflow`，对完整 rank-local static root 调用同一
library；也可提供只读 collect/report helper。它适合：

- 在测试/调优中选择 payload scope；
- 组合 standard canonicalization、bufferization 和 verification；
- 重放 profile、budget 和 deterministic policy version；
- 让研究性 pipeline 用同一终态 materializer 比较结果。

明确禁止：

- 为每个 dtype/layout/model/op pair 建 transform op；
- 用 transform handle/param/attr 表示 beam、SPM liveness、physical versions 或 cost frontier；
- 用 `transform.alternatives` 代替 top-K/Pareto 搜索——它是 first-success，不是 cost selector；
- 把 TransformState 或 transform module 保存为 package/accepted artifact；
- production 与 Transform extension 各维护一套 matcher、legality 或 materializer。

Transform extension 只有在核心 planner、typed IR 和 named production pipeline 已稳定后实施；它不是主线完成的前置。

## 9. Target capability 与硬件约束

planner 不直接散落硬件 magic number，而从 `TargetProfileId` 对应 provider 查询。当前 Wafer profile 至少应暴露：

- 每 tile 可用 SPM window `[0x10000, 0x2F0000)`，即 3,014,656 bytes，以及 alignment/reserved ranges；
- Tensor/Cx/NCx 的 block、tail、padding 和 physical map；当前非 int8 channel block 为 64，int8 为 128；
- RDMA source-strided/destination-sequential、WDMA inverse、GS 双侧映射及最多三层 outer stride/iteration 的 descriptor
  能力；
- compute engine 的 dtype、tile geometry、accumulator、layout/orientation 和 padding-lane contract；
- queue/resource/effect、provider completion、fence/wait 和 transport能力；
- instruction/CRT/CModel 已纵向资格化的字段。

每个 capability row 分开记录 `statically_representable`、`compiler_emittable`、`model_qualified` 和
`board_supported`，不能压成一个 `supported`。request 选择的 profile 决定可进入domain的最低证据级别：model-only
source-to-CModel vertical可以使用已完成typed ABI与formal/managed-reference资格的`model_qualified` row；真实board
publication必须进一步要求同target/environment的`board_supported` row。板端不可用不阻塞本任务的model-only实现闭环，
但也不能把model结果宣传成硬件已校准。

当前 packet/wrapper 虽存在 GEMM transpose flags，但 live Instr op、target call 和 CRT 默认路径尚未形成完整 typed
transposition capability；当前 load/store 也并非所有 layout/local-offset route 都可表达。因此这些只能作为实现
checkpoint，不能被 planner 假设为已支持。任何新 family 必须纵向补齐 10/11/14/17 后才进入 domain。

padding correctness 是 hard legality：若 physical lanes 超出 logical valid domain，compute 只有在所选 family 证明这些
lanes不被读取、使用 neutral value 或在结果前被 mask 时才可运行 full physical extent。`exp(0)=1`、scalar add 等会破坏
zero padding invariant；此时必须使用 segmented valid tail、mask-capable family 或 materialize Tensor，不能靠 case 经验。

## 10. Selected IR 和下游责任

选中 proposal 通过 07 物化后只表达以下长期事实：

- typed memref memory space 与 accepted physical encoding；
- `memref.subview`/reassociation 等可验证 view；
- target-abstract compute/movement/collective；
- explicit resident SSA、spill/reload 和 immutable resource；
- async issue、provider completion、join、reuse 和 terminal wait；
- structured static traversal、tail 和合法 reduction chain。

下游责任：

- 07：从 proposal 原子构造完整 tile-dataflow IR，并验证 coverage/use-def；
- 08：physical map、valid domain、descriptor cover、view/route/materialization；
- 09：whole-rank exact SPM lifetime、alias、offset 和 peak；
- 10：implementation family 与 target-abstract compute/movement ops；
- 11：typed instruction field、geometry、descriptor 和 narrowing legality；
- 12：whole-variant DDR demand、lifetime、offset 和 package range；
- 13：collective/DTE completion 和 all-rank matching；
- 14/15：从final instruction/TargetCall rows派生、readback并在schema-v4投影canonical `RequiredCapabilitySet`；
- 16/17：source oracle、SystemC/CModel、multi-dtype 与 board 分层 evidence。

analysis cache、choice domain、candidate score、fusion label、layout label graph、group、scope-prefix 和 shadow schedule
不得跨越该边界。

## 11. 通用示例和 7B scale case

### 11.1 示例参数与协议事实

通用 contraction chain、diamond、fanout/fanin、broadcast/reduce、multi-root shared-input、attention-like dataflow 和
branched expert-like dataflow 都由相同对象处理：

- iterator/indexing maps 产生 IndexRelation；
- target provider 产生 parameterized families；
- backward propagation 连接 dependent tiles；
- selected physical versions 决定 resident dataflow；
- effect/numeric/resource/descriptor gates 决定 cut；
- cost vector 和 hard budget 决定有限候选顺序。

这些是协议。某个模型的 op 名、root 数、shape、tile 数和最终融合视图只是输入实例。

### 11.2 Llama-2 7B 单 block 作为规模压力

标准 Llama-2 7B 单 block、batch 1、sequence 16、TP16、FP16 storage 只作 scale/evidence case：每 rank 的
attention projections、MLP contractions、normalization、RoPE、softmax、residual 和 collectives 必须走同一通用算法。

它用于验证：

- shared-input 多 root 不按 QKV/gate-up 名称识别；
- projection orientation/layout 可在 capability 闭合后由 implementation 吸收，而不是先写整块 DDR 转置；
- 很小的 pointwise/reduce task 能通过 resident physical version 与相邻 compute 连通，而非每 op 一个 DDR group；
- tile 缩小带来的 SPM headroom、重复 invariant load、descriptor/issue 增长全部进入同一 cost vector；
- collective 是 completion barrier，不自动成为 DDR barrier；
- rank-count=16 的 all-rank transport、package 和 complete output 数值不回退。

当前 Q29/Q28/Q30/Q31 evidence 只作为迁移 baseline：rank-0 high-water 曾为
2,725,568 / 3,014,656 bytes（90.411%），现有 layout/weight movement 和碎片化小 task 暴露了联合优化空间。该数字、
当前 tile shape 和最终 resident components 不进入规则；新 planner 必须重新从最终 selected IR 统计，并用完整 PyTorch
expected 与 SystemC/CModel 比较数值。

## 12. 实施边界和完成定义

实施计划位于 `tasks/plans/physical-dataflow-synthesis.md`。施工必须纵向推进，而不是先造一个脱离 accepted IR 的 solver：

1. 先冻结 shared capability/typed IR 合同和 fresh baseline；
2. 补齐 implementation/transfer family 到 Instr/CRT/SystemC 的最小纵向能力；
3. 实现 IndexRelation、PhysicalEncoding、valid-domain 和 descriptor-cover property core；
4. 让保守 baseline 经新 family/provider 物化并通过全部 exact gates；
5. 加入惰性 domain、constraint propagation、canonical frontier、Pareto beam、hard budget 和 telemetry；
6. 逐层启用 E0-E3 通用规则，E4 只在 numeric policy 完整时启用；
7. 切换 production，并删除旧 layout planner、implicit per-use materialization、scope-prefix/maximal-resident 决策旁路；
8. 可选地接入 Transform control plane，共用同一 C++ implementation。

完成必须同时满足：

1. 无 workload 名、固定 shape、parameter position 或 op-pair case table；
2. 无 fusion partition、tile integer、task permutation、resident subset 或 family Cartesian-product 全枚举；
3. hard budget、baseline fallback、canonical dedup 和 candidate-growth telemetry 有正负测试；
4. selected state 只以 typed payload IR 跨阶段，旧决策接口和重复事实源清零；
5. property tests 覆盖 dtype/block/tail/relation/descriptor/padding；拓扑覆盖 chain、diamond、fanout、fanin、
   multi-root、reduction、broadcast；workload 覆盖 GEMM/MLP、convolution/contraction、attention 和 branched/MoE-like；
6. rank-count=1/16、7B scale 和完整 PyTorch/SystemC numerical gate 不回退；
7. exact SPM/DDR/event/transport/instruction/ABI 和 atomic candidate/rank commit 不回退；
8. 预算耗尽仍返回合法 baseline 与结构化 diagnostic；
9. 板端 performance/calibration 仍由 later gate 拥有，不成为本任务虚假完成证据。

局部 FileCheck、只生成候选、只减少 layout op、只在 7B case 生效、单 rank 通过或旧 planner 与新 planner 并存，都不构成
完成。

## 13. 参考材料

以下工作只提供算法启发，不改变本文的 Wafer IR/target 合同：

- VTC：在fusion后graph上用virtual tensor/index mapping消除kernel间显式data movement，并全局选择virtual-tensor
  creation strategy；它明确与layout optimization/operator fusion互补。本文借鉴“跨kernel物理关系联合选择”，不复制其IR。
  <https://www.usenix.org/conference/osdi26/presentation/hu-muyan>
- Welder：从 consumer tile 反向传播 dependent region，并联合考虑 memory traffic；本文采用有界 region propagation。
  <https://www.usenix.org/conference/osdi23/presentation/shi>
- SmartMem：跨算子选择 layout 并减少 layout conversion；本文把 route/encoding 纳入 physical-dataflow synthesis。
  <https://arxiv.org/abs/2404.13528>
- TASO使用verified substitution与cost-based backtracking，TENSAT使用e-graph/equality saturation探索通用graph rewrite；
  本文只在小semantic island中使用有证明、hard-capped alternatives，不做whole-model equality saturation。
  <https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf>
  <https://proceedings.mlsys.org/paper_files/paper/2021/hash/cc427d934a7f6c0663e5923f49eba531-Abstract.html>
- ALT联合graph data layout与operator loop tuning；PBQP显式计入format-conversion cost做primitive/layout selection。
  本文借鉴参数域和约束传播，不引入第二份assignment IR。
  <https://arxiv.org/abs/2210.12415>
  <https://arxiv.org/abs/1710.01079>
- GraphTurbo展示DSA上的coarse subgraph partition/order与cross-layer scheduling；COSMA联合scratchpad accelerator的operator
  scheduling、memory allocation与tensor replacement；Korch在GPU上做operator fission与BLP kernel orchestration。本文只借鉴
  它们对计算/数据移动/资源联合决策的不同拆法，并用typed effects、exact resource gates与bounded frontier约束适用范围。
  <https://www.usenix.org/conference/osdi23/presentation/zhao>
  <https://arxiv.org/abs/2311.18246>
  <https://arxiv.org/abs/2406.09465>
- MLIR Transform Dialect：作为 transformation orchestration/control plane，而不是 solver state 或执行 artifact。
  <https://mlir.llvm.org/docs/Dialects/Transform/>
  <https://mlir.llvm.org/docs/Tutorials/transform/Ch4/>
