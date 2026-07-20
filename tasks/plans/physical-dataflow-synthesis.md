# Physical-Dataflow Synthesis 实施计划

状态：按 MLIR 原生接口、当前生产 candidate transaction 和真实优化收益重新基线。本计划取代旧的
provider/query、通用 frontier schema 和先造 solver 再造 rewrite 的施工顺序；动态任务状态仍以
`tasks/progress.md`为准。编号设计 owner 必须与本计划同批收敛后才能进入实现。

本计划中的 planner 只表示一个 compiler-private **候选编排器**：它从当前 structured IR 产生少量完整候选，
调用现有 materialization、memory planning 和 all-rank exact gates，再选择一个可原子提交的候选。它不是独立
语义层、通用约束求解语言、provider 注册系统或长期可序列化协议。

长期职责仍由编号文档拥有：

- 05：post-SPMD structured tensor semantics 和 normalization；
- 06：candidate 编排、选择和原子提交边界；
- 07、08、10、11：selected tile/physical/compute/movement/instruction IR；
- 09、12：从当前 candidate IR 重算的 SPM/DDR legality 和 offset；
- 13：从 typed communication IR 重算的 all-rank transport acceptance；
- 14、15：winner 之后的 target conversion、artifact 和 package；
- 16、17：source、SystemC/CModel、数值和后续 board 分层验证；
- 18：源码与依赖 ownership。

源码、pass、pipeline、CLI、diagnostic 和 artifact 使用稳定语义名称，不使用 checkpoint 或任务编号命名。

## 1. Pipeline Position

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD 后按 logical rank specialize、已通过 verifier 的 Linalg/Tensor/SCF/Arith/Math
  structured tensor program；validated ExecutionConfig、TargetProfileId、numeric/effect policy，以及当前
  Q29 production scheduler 形成的 source-to-bundle、rank-count=1/16 和 7B 数值基线。
- Current stage responsibility:
  直接通过 MLIR op/dialect/external interface、SSA use-def、indexing map、shape/type/effect 和按需构造的
  IndexRelation analysis 识别可优化 dataflow；在隔离的 complete-rank clone 上应用已验证的 typed rewrite
  mechanism，产生保守 baseline 和少量优化候选；复用现有 tile/instruction materialization、function-boundary
  finalization、SPM/DDR planning、completion/transport/resource/ABI exact gates和all-rank coordinator；在没有板端
  cost 校准前只提交静态 exact metrics 严格支配 baseline 的优化候选，最后原子形成 ExecutableBundle。
- Output artifact / IR:
  与当前 production 相同 schema 的 profile-bearing atomic ExecutableBundle；每个 rank 恰有一个 final
  placed/bound instruction/memory/completion program。accepted payload 不含 candidate ID、search trace、relation
  cache、fusion group、provider result、诊断统计、shadow plan 或其它 planner 状态。
- Downstream consumer:
  现有 target LLVM/module/artifact/package、no-card runtime、target-call/SystemC CModel，以及 later board
  numeric/cost/timing gates。下游只消费 winner typed IR/artifact，不读取 rejected candidate 或编译期分析对象。
- User-level driver / named pipeline:
  仅使用 wafer-compile 现有 source-to-bundle named production pipeline。focused wafer-opt/IR tests只重放同一
  mechanism 或既有 lowering/gate，不形成第二条用户级 pipeline。
- Explicit non-goals:
  不建立独立 provider/query/result/key/cache/schema；不复制 structured op 成 detached semantic descriptor 或 scalar DAG
  语义库；不序列化 canonical frontier、完整 instruction IR 或 candidate signature；不实现通用 Pareto/constraint
  solver、minimum-high-water packing objective、版本化诊断统计协议、Transform Dialect control plane、
  dynamic-shape/online scheduling、ping-pong、persistent weight cache、whole-model equality saturation或模型专用
  rewrite。mapped DMA、oriented GEMM、新 target ABI、RequiredCapabilitySet/package schema upgrade是独立target/runtime
  capability 工作，不是本计划 baseline、rewrite、planner 或 production cutover 的前置。
- Completion gate:
  至少两条通用 rewrite mechanism 在真实 production source 上实际改写；第一条 dependent tiling + resident
  handoff 必须在冻结 7B vertical 中删除真实 spill/reload 并降低最终 DDR bytes 或 instruction count。所有候选均
  通过现有 complete-rank 和 all-rank exact gates，rank-count=1/16、Q20/Q21、7B PyTorch/SystemC、双配置全量
  回归和 atomic failure fresh 通过；production 只剩一个 candidate owner，旧 scope-prefix、scalar estimated-time
  和 spill-vs-maximal-resident 决策旁路删除。没有证据需要 beam 时，固定小候选集即为完成形态。
```

## 2. 施工原则

1. **先有真实 rewrite，再抽象 planner。** 每个新增 analysis、interface、typed field 或 helper 必须被当批
   mechanism 直接消费，并在最终 IR 上产生可验证差异。
2. **IR 是唯一语义事实源。** op-local 能力由 op interface 或 external model 提供；跨 value 的关系从 SSA、
   indexing map 和 view op 重算；target-wide 几何事实由现有 typed target profile/helper 提供。普通 typed C++
   helper 可以存在，但不能演化成平行 provider/query 协议。
3. **Analysis 可失效、可重算。** IndexRelation、alias/effect、lifetime、cost 和候选统计不跨 rewrite 保存；
   rewrite 后从新 clone 的当前 IR fresh 构造。
4. **候选必须完整。** 便宜 bound 只能提前拒绝，不能代替 complete traversal、instruction、SPM/DDR、event、
   transport 或 ABI gate。rejected clone 不得污染 source 或其它候选。
5. **合法性与排序分离。** baseline 始终保留；板端校准前，存在任一静态 tradeoff、Unknown 或 Overflow 时优先
   baseline，不用自造 scalar time 或任意 lexicographic 权重宣称更快。
6. **复用现有 transaction。** rank candidate clone/finalization、Direct DTE acceptance、whole-variant resource gate
   和 ExecutableBundle atomic commit 是既有生产边界；本计划只替换候选生成和未校准选择，不重建第二套 transaction。
7. **复杂度来自证据。** 固定小候选集足够时不实现 beam/frontier；只有真实 source 证明候选增长或固定访问顺序
   丢失严格支配方案时，才进入 Checkpoint E。
8. **allocator 只做 legality。** Q34 fixed-capacity MiniMalloc、三态结果、validator 和 owner 原子 offset apply 保持
   唯一容量合同；allocator 不建议 tile/residency repair，也不承担 candidate cost search。

## 3. 明确排除的旧设计

本计划不实现、也不以删减名称重新包装下列对象：

- 统一的implementation/encoding/transfer/communication provider registry及canonical query/result/key/cache identity；
- 独立baseline composer、provider snapshot、ordered query trace和safe-tile proof codec；
- canonical frontier、placed/bound transport signature等planner-side serialization；transport correctness继续从
  candidate instruction IR由13/current coordinator重算；
- full-IR canonical bytes、candidate digest、search parent、beam trace或 shadow schedule；
- `PackingEnvelope`、minimum-height binary refinement、lower-bound proof、pressure-view-to-decision反馈和跨候选
  packing objective cache；
- 统一几十字段 deterministic work-policy schema和版本化planner统计协议。实现可保留少量
  invocation-local debug counters，但它们不版本化、不序列化、不参与选择或完成资格；
- Q32.T Transform control plane；
- mapped/oriented target extension、新target ABI revision、RequiredCapabilitySet或manifest schema升级。

这些能力未来若有独立需求，必须以实际 consumer、单独 pipeline contract 和独立完成 gate 重新排期，不能重新成为
本计划真实 rewrite 的前置。

## 4. Checkpoints

### Checkpoint A：Fresh baseline、MLIR 接口和最小 IndexRelation

输入：当前 production structured tensor IR、Q29 rank candidate pipeline、Q28/Q31冻结source/config/payload/expected，
以及现有 `TilingInterface`、`DestinationStyleOpInterface`、Linalg indexing maps和Wafer selected-op interfaces。

施工：

- fresh重放 Q20/Q21、rank-count=1/16 和标准7B，记录每rank最终 DDR read/write bytes、SPM movement、accepted
  SPM high-water、instruction/event、candidate数、source-to-bundle wall和明确仍存在的 spill/reload edge；指标只从
  final placed instruction IR重算；
- 建立第一条 rewrite 所需的 interface/consumer 矩阵。structured op 优先消费已有 Linalg/MLIR interface；无法直接
  修改上游op时注册 external model；只有确实缺失的 target implementation/encoding 局部能力才增加一个最小typed
  op或dialect interface method；
- 实现 transformation-local `IndexRelation` analysis 的最小闭包：从 Linalg indexing map、DPS relation和标准
  tensor view构造第一条 rewrite 实际需要的identity、projected permutation、static slice及必要组合；reshape或其它
  relation只有第一条真实edge确实经过时才加入；
- relation查询只提供 dependent-region preimage、exact equivalence和明确的Unsupported/Unknown，不定义通用wire schema、
  canonical bytes、跨pass cache或完整piecewise证明语言；优先复用MLIR affine/Presburger与structured-op utility；
- property/小shape穷举验证每种已支持relation的all-and-only index集合；动态、越界、非functional、effect barrier或无法
  精确表达时保持baseline，不猜测。

产出：fresh production基线、第一条真实edge的IR事实记录，以及能被Checkpoint B直接调用的最小interface和
IndexRelation analysis。production IR和package schema不变。

Gate：同一输入重复运行基线指标稳定；relation positive/negative/property tests通过；rewrite尚未启用时
development/target-model build、focused lit/unit和冻结7B source-to-package/SystemC/PyTorch基线fresh通过。

不算完成：只写接口但没有Checkpoint B consumer；复制一份structured语义descriptor；一次性实现concat、任意
piecewise relation、provider registry或canonical query；只保存IR dump而没有最终指标。

### Checkpoint B：第一条真实 dependent tiling + resident handoff

输入：Checkpoint A最小relation/interface、冻结baseline，以及一个从structured SSA/indexing relation识别的通用
producer-consumer edge；7B中的具体edge只作为规模case，不进入matcher或协议。

施工：

- 在隔离structured clone上，给定consumer tile后用IndexRelation精确反推producer dependent region；复用upstream
  `TilingInterface`和Linalg/SCF tile-and-fuse utility形成producer tile，不复制upstream whole pass或按op名字匹配；
- 将动作拆成两个typed cut：structured层只做dependent tiling/fusion；selected physical payload层只在producer result与
  consumer operand具有相同logical region、可兼容encoding、明确SSA/effect/completion关系且SPM lifetime合法时，共享同一
  physical version并删除对应store/reload；
- mechanism一次只改写一个稳定IR cut，返回Applied/NotApplicable/Failure；它不读取candidate score、搜索历史、模型名、
  operand名字或fallback policy；
- 保留未应用的原始baseline clone。任何relation、encoding、lifetime或exact gate未知时只拒绝优化candidate，不降低
  verifier或使整个合法输入失败；
- rewrite后丢弃旧analysis，从新clone重新materialize/lower，并fresh运行SPM/DDR、descriptor、completion、transport、
  instruction和ABI gate。

产出：一个policy-free、可单独测试的end-to-end mechanism，以及baseline和resident两个complete-rank候选。accepted IR中
只保留实际shared SPM root、view、compute/movement和event关系。

Gate：

- 通用chain、transpose/view boundary、tail和effect barrier正负例证明dependent region与rewrite coverage；
- 至少一个非7B真实source及冻结7B source实际命中rewrite；
- 7B winner最终IR至少删除一组真实WDMA/RDMA或等价store/reload，DDR bytes或instruction count严格下降，SPM peak仍在
  现有窗口内；
- rank-count=1/16、SystemC/PyTorch完整输出与baseline comparator一致，失败candidate无partial offset/event/bundle mutation。

不算完成：只在synthetic fixture改写；只打印“可resident”但最终IR不变；靠固定shape、Llama role、参数顺序或名字识别；
用SPM容量estimate代替真实planning；优化失败时没有baseline。

### Checkpoint C：接入现有 rank candidate，并完成 production-shaped test-seam vertical

输入：Checkpoint B的baseline/optimized complete-rank候选，以及当前
`buildScheduledRankCandidateFrontier`、shared finalization、whole-variant coordinator和ExecutableBundle transaction。

施工：

- 在现有compiler-private rank candidate seam中按稳定顺序产生baseline和Checkpoint B候选；不新增公开pass、CLI、driver
  mode或第二条production pipeline；
- 每个候选继续使用独立complete-rank clone，经过现有function-boundary finalization、SPM/DDR owner、fresh cost和
  verifier；候选对象只拥有module、baseline bit和少量稳定decision ordinal，不保存plan、SSA pointer、relation、offset副本或
  canonical serialized state；
- all-rank阶段直接复用current Direct DTE message matching/binding、whole-card resource acceptance和target ABI gate，所有
  message/range/completion事实从候选IR重算。若后续确有多个communication schedule，只用typed IR-derived的小型schedule
  identity分桶，不能复制placed/bound claim协议；
- 排序复用`InstructionProgramCost`逐维Known/Unknown facts。板端校准前，optimized candidate必须在相同transport语义下对
  baseline全部可比较维度不差且至少一维严格更好；否则选baseline。删除或绕开未校准`estimatedTimePs`对新候选的决策权；
- all-rank组合在clone上完成全部gate后才形成bundle；任一late rank、binding、resource或ABI失败均丢弃整组候选。

产出分两步映射任务队列：Q32.R先把Checkpoint B真实rewrite接入与旧候选完全相同的rank finalization/all-rank gate；
Q32.B再让它通过compiler-private、production-shaped test seam重放1/16-rank source-to-bundle/package/SystemC/PyTorch链并
实际胜出。Q32.B不改变用户默认pipeline；在Checkpoint F/Q32.G完成前，旧production producer仍是唯一默认。

Gate：rank-count=1/16、Q20/Q21和7B均证明baseline可恢复、优化候选实际胜出时满足严格静态支配；不同candidate
parallelism和重复运行选择相同；late-rank/transport/SPM/DDR/ABI注入失败无partial bundle/artifact/model result。

不算完成：新路径自己复制SPM/DDR或transport verifier；以serialized compatibility record代替current IR exact gate；
仅单rank成功；通过debug flag长期维护两条用户pipeline。

### Checkpoint D：第二条通用 tiled fanout reuse mechanism

输入：Checkpoint A-C的relation/interface、complete-candidate transaction，以及structured SSA中一个producer tile被两个或
更多consumer读取的通用fanout。

施工：

- 用SSA use-def和IndexRelation比较各consumer对producer的dependent region；只有region exact相同、encoding兼容、effect/
  completion不跨barrier且fresh SPM lifetime合法时，才让多个consumer共享一次producer tile compute/load和同一SPM physical
  version；
- compatible consumer集合按stable IR order形成单一确定性全体方案；出现互斥relation、不同encoding、unknown alias或容量
  tradeoff时保留baseline，不枚举`2^fanout`子集；
- structured reuse与selected physical-version reuse仍分别作为单一职责mechanism；二者由候选编排器顺序组合，不建立
  fusion group、hyperedge side table或provider domain；
- rewrite后fresh重算relation、alias/effect、SPM/DDR lifetime、completion和cost，并走Checkpoint C同一exact链。

产出：第二条独立、通用、policy-free mechanism；它与Checkpoint B共同证明候选编排器能组合多于一种真实优化，而不需要
通用solver。

Gate：diamond、2/3-way fanout、partial overlap、不同view/encoding、effect barrier、collective barrier和SPM overflow覆盖；
至少一个真实production source最终少一次compute/load或一组movement，且静态exact metrics严格支配baseline；1/16-rank与
完整数值回归通过。

不算完成：仅把两个use指向同一Value但没有删除最终重复work/movement；按固定root数、名字或operand位置特判；为了处理一个
冲突case引入subset search、beam或全局mutable reuse table。

### Checkpoint E：仅在证据要求时增加有限 frontier

输入：Checkpoint C/D收集的真实candidate数量、exact-gate失败分布、编译wall和“固定访问顺序是否漏掉严格支配候选”的
可复现实验。

进入条件：必须先有真实source证明至少一种情况：固定小候选集因hard cap跳过了稍后可生成且严格支配baseline的候选；或两条
已验证mechanism组合后候选增长已对compile wall/内存形成可测压力。没有这类证据时，不实现beam，本checkpoint以“保留固定
有界vector”的结论收口，不阻塞Checkpoint F。

若进入施工：

- 只在现有candidate producer内部增加有限vector/frontier；baseline占不可淘汰slot，候选由typed mechanism decision tuple和
  stable IR ordinal识别，不序列化structured/instruction IR或建立canonical shadow state；
- 只限制少量真实增长维度：每island候选数、每rank materialized候选数、rank frontier和whole-variant attempt；relation和
  Q34 packing继续使用各自owner预算，不建立统一几十字段work-policy schema；
- 可做同transport语义内的strict exact dominance剪枝；不存在板端校准时不在tradeoff候选间使用权重、伪时间或任意
  Pareto tie-break，incomparable时baseline优先；
- 并行只评估已按stable order形成的bounded batch，结果按原ordinal归并；future完成顺序不影响cap或winner；
- 只保留invocation-local普通统计，如generated/materialized/rejected-by-gate/peak-frontier/used-baseline，供debug日志和测试；
  不定义诊断统计schema、digest、artifact或qualification入口。

产出：若证据充分，得到最小bounded frontier；若证据不足，产出固定小候选集足够的fresh数据和明确不建设beam的结论。

Gate：压力图证明生成和materialization均不越明确小上界；不同线程数/重复运行winner一致；cap耗尽返回baseline；候选增长相对
Checkpoint C有可解释收益且没有显著恶化冻结7B compile wall。无证据分支则重放C/D证明固定候选集覆盖所有当前真实choice。

不算完成：因为“以后可能有更多实现”而建设beam；只限制最终top-K但允许前面无界生成；通过full-IR bytes、provider cache、
search parent或版本化统计协议维护frontier。

### Checkpoint F：Production cutover、旧路径删除和完整审计

输入：Checkpoint A-D全部通过，以及Checkpoint E的有限frontier或“不需要beam”结论。

施工：

- 让wafer-compile现有source-to-bundle pipeline只调用新候选编排器；不新增planner选择flag或silent fallback；
- 删除由新机制取代的旧scope-prefix/shared-input prefix、spill-vs-maximal-full-buffer-resident后处理、独立layout/materialization
  决策旁路、`estimatedTimePs`/scalar ranking/discovery-order recovery和只服务这些决策的tests/API；
- 保留现有typed Tile/Instr、SPM/DDR planning、Direct DTE binding、whole-card resource/ABI gate和atomic bundle transaction；
  不把合法性service误删为旧planner；
- 从winner current IR fresh重算并报告DDR read/write、SPM movement/high-water、compute、instruction、event和transport metrics；
  不发布板端性能或timing结论；
- current v1 target profile、target-call ABI和package schema保持不变。mapped/oriented/package schema upgrade如需恢复，另建独立
  pipeline contract和任务，不回插本计划；
- 同步受影响编号设计、progress、README和源码组织；清除旧provider/query/frontier/版本化统计/Transform前置叙述；有稳定经验才
  写memory；完成后归档计划并提交相关改动。

产出：单一production candidate owner；至少两条真实mechanism通过现有typed pipeline生成final all-rank bundle；accepted IR、
artifact和package没有planner/search/shadow状态。

Gate：

- development和target-model双配置fresh build，使用并行构建；lit/unit/CTest及unsupported/skipped清单核对；
- dependency、IR/source organization、CRT/conformance和文本一致性检查通过；
- Q20/Q21、rank-count=1/16、标准7B source-to-package-to-SystemC/PyTorch以及Q31 held-out seeds按受影响numeric policy重放；
- 第一条rewrite在7B final IR仍保留实际movement/instruction改善，第二条在真实production source保留实际改善；
- old decision API/option/pass consumer为零，任一late failure无partial bundle/artifact/model effect；
- 新路径不依赖provider/query/schema、packing objective、版本化统计协议、Transform control plane或新target ABI。

不算完成：新旧planner靠flag并存；只在默认pipeline不用旧入口但代码/tests仍保留；只证明单个pass改写而未进入all-rank bundle；
数值或SystemC gate skipped/unsupported；静态tradeoff candidate未经校准仍替换baseline。

## 5. Integrated Completion Audit

标记本任务完成前逐项确认：

- pipeline contract、编号设计、live code和唯一production named pipeline一致；
- op-local能力走MLIR interface/external model，跨value关系从当前IR重算；不存在detached descriptor/provider/query第二事实源；
- IndexRelation只覆盖被真实mechanism消费的闭包，每个新增relation kind有exact property/negative测试；
- 两条mechanism均先独立证明rewrite与fresh legality，再由候选编排器组合；policy中没有隐藏matcher或rewrite；
- baseline始终经过与optimized candidate相同的complete-rank/all-rank gate；优化失败或静态tradeoff返回baseline；
- accepted payload只含typed compute/view/movement/buffer/event/offset，不含relation、candidate、frontier、cost、诊断统计或
  compatibility sidecar；
- Q34 fixed-capacity packing仍是唯一SPM/DDR容量owner，没有minimum-height objective、repair recipe或candidate-feedback proof；
- all-rank correctness从final candidate IR重算，未复制placed/bound message signature协议；
- 没有数据证明beam必要时，固定候选集是正式终态而非未完成；若实现有限frontier，其生成/materialization cap和determinism有
  fresh压力测试；
- mapped DMA、oriented GEMM、RequiredCapabilitySet/package schema upgrade、Transform control和board cost/timing不在完成依赖中；
- fresh 1/16-rank、Q20/Q21、7B fixed/held-out numerical、SystemC、SPM/DDR/event/transport/instruction/ABI和atomic gates实际执行；
- 旧决策路径及兼容入口清零，文档不再把provider/query/版本化统计/shadow frontier作为当前实现合同；
- 稳定bug/workflow经验按需进入memory；计划归档、progress更新、相关改动提交。

任一项未满足时保持任务未完成，并准确记录缺口；单轮rewrite、局部FileCheck、candidate计数或“框架已经搭好”都不构成完成。
