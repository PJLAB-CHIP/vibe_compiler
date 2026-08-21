# Wafer Compiler Verification Contract

状态：本文只定义当前架构的验证层级和完成证明，不保存任务动态状态或历史case台账。稳定验证链为
`TensorProgram -> physical-dataflow selection -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
host局部gate、package roundtrip或一次source compile都不能把production任务提升为`board-ready`；没有真实设备matched
A/B改善时也不能标`done`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  当前source program、card-level GSPMD输出、normalized TensorProgram、selected CardModule/TileModule/TileRegion、
  final Instr、CardExecutable、target LLVM modules、linked ELF、target-ready immutable data、ExecutablePackage及其独立oracle。
- Current stage responsibility:
  在每个IR/output边界验证语义、coverage、physical identity、resource、completion、ABI、writing与execution；
  建立host、data-scale、no-card、model和真实板端证据之间不可越级的完成层级。
- Output IR / files:
  verifier diagnostics、fresh test/build结果、verified package/runtime plan、model result或真实board result；
  evidence不是IR sidecar，也不参与候选选择。
- Downstream consumer:
  tasks/progress状态判定、下一pipeline stage、board qualification和最终性能结论。
- User-level driver / named pipeline:
  wafer-opt named pipelines、wafer-compile `search|none`、wafer-run no-card/board和configured host test suites。
- Explicit non-goals:
  不用历史raw输出重签结论；不让fixture、FileCheck或shape dump代替主线source→package；不允许skipped/
  unsupported测试冒充通过；不为旧schema、旧ABI、旧pass或旧board harness保留测试入口。
- Completion gate:
  受影响边界的unit/lit/integration fresh通过；完整current package fresh no-card；涉及板端的任务达到
  board-ready后，再由当前构建/current case串行真实执行；Q53还要求matched性能改善。
```

## 2. Evidence levels

证据严格分层，低层不能代签高层：

1. **Static/compile evidence**：编译、ODS/verifier、unit、lit、source organization和文本一致性检查。
2. **Output evidence**：同一compile transaction生成并readback CardModule/Instr、CardExecutable、target module、
   ProgramDataHandoff、TargetTensor、target-ready program data和ExecutablePackage；记录source range reads、
   TargetTensor materialization与package byte closure。
3. **No-card evidence**：真实package经strict loader与runtime validation形成完整16-Tile invocation plan，且无provider effect。
4. **Functional model evidence**：同次owner-backed target module set经TargetCall frontend/SystemC执行，完整output与独立CPU expected比较。
5. **Board correctness evidence**：当前构建、当前package、当前payload在真实设备完成output/guard和lifecycle检查。
6. **Board performance evidence**：同一source/config/payload/ABI的baseline/winner做matched、重复、可解释的A/B。

只有第3层完成，且case、oracle、runner都齐全，板端任务才可写 `board-ready`。只有任务定义所需的第5/6层通过才可
写 `done`。instruction count、理论makespan、host wall time、SystemC event count和no-card成功都不是实卡性能证据。

`board-ready`只描述current producer、current package/runtime合同和current executable runner组成的可执行门禁，不是历史能力标签。
如果candidate owner、IR/output边界、package/ABI或runner已被替换，旧状态必须立即撤销：仍需要同一任务时重新生成current
package并fresh no-card；职责已由后继接管时直接从current queue移除旧任务，并把仍有效的source、oracle、effect witness、
计时或codec mechanics写入current owner。历史只由Git和archive保留，不能继续授权历史板端批次，也不建立旧任务状态索引。

## 3. Freshness 与执行纪律

- 每轮代码/测试改变后只使用本轮build和本轮输出；历史raw/log/report只作审计记录，不作为test input。
- producer/consumer合同或runner失去current executable路径时，即使source与oracle仍存在，也不能沿用旧`board-ready`；
  source-only、`executable=false`或pending-lowering资产只能等待current successor消费。
- 不重复执行已经有结论且代码/环境未变化的板端case；host/no-card仅在对应实现变化或异常归因时重跑。
- host build、unit、CTest、catalog与no-card默认使用 `nproc` 可用并行度；只有真实资源约束才降低并说明。
- 真实device launch始终单进程、逐case、bounded timeout；首个timeout/device异常后停止，不自动retry/reset/power。
- 同一重启会话且软硬件身份未变时，只资格化一次；后续case复用move-only qualified session。
- 默认板端数据类型为FP16/BF16；只有格式/ABI/转换/数值边界本身或真实source要求F32时才使用F32并记录理由。
- 测试报告必须核对实际执行数量、skip/unsupported清单和feature配置；`ctest passed`本身不证明关键lit或source vertical执行。

稳定host入口遵循当前CMake/lit配置，例如：

```text
cmake --build <configured-build> -j$(nproc)
ctest --test-dir <configured-build> -j$(nproc) --output-on-failure
```

任务结果只报告真正运行的target和case，不把构建一个object、收集到一个test或生成fixture写成端到端通过。

## 4. IR 与 physical-dataflow gates

### 4.1 CardModule / TileRegion / Instr

正例必须证明：

- `num_partitions`只描述card-level GSPMD domain；single-card current path固定 `num_partitions=1`；
- `wafer.card.module`拥有all-and-only 16个available `wafer.tile.module`；
- distinct Tile modules可含不同op、loop、temporal tile、region和执行长度；
- no-work Tile仍有合法entry并进入output/runtime domain；
- `(card_id, tile_id)`唯一，Tile-local SPM root不跨Tile SSA alias。

负例至少覆盖duplicate/unavailable/missing Tile、coverage hole/overlap、非法reduction overlap、cross-Tile SPM alias、
mismatched send/recv、payload/domain/encoding mismatch、缺失wait和premature release。

### 4.2 Physical-dataflow selection

同一个通用selection owner需要覆盖：

- single-op all-iterator多轴factor vector、非整除remainder、parallel/reduction partition与显式merge；
- 合法非最大、非矩形、非对称/非连通physical placement；compact/all-16只作为排序seed；
- chain中的producer/consumer wave pipeline；
- independent branch分配到disjoint Tile groups；
- diamond、fanout、fanin与reduction；
- partial co-location、mapping redistribution、multicast/gather/reduction；
- same-region不同temporal tile、selective spill/reload、recompute和region cut；
- compute、DDR、NoC与显式SPM movement在已有independence/buffering证明时的overlap。

每个spatial trial的logical demand gate必须消费完整consumer iteration/contribution domain、producer ownership domains、
data/init/support dependency role和IR epoch，不能以result axis、shard dimension或participant count代替。独立query正负测试至少
覆盖reduction input/init/partial contribution与merge role、broadcast、affine window的stride/dilation/halo/pad、strided
slice/view、multi-piece union、multi-result/fanout/fanin、explicit init root和multi-operand pure support graph。

candidate generation只能读取current structured semantics、indexing maps、SSA/effects、type/shape/dtype、explicit
physical communication和target capability。测试要搜索并拒绝framework/model/function/value-name、固定shape、operand
position、Attention/decode/mask专用matcher或公共pass残留。

### 4.3 Search correctness 与 exact gates

- semantic、spatial、temporal、TileRegion/融合、layout、movement、buffer、communication和order合法域从current IR惰性生成；
- temporal域同时覆盖完整all-iterator tile vector与Q50.D已选traversal内会改变reuse/lifetime/tail的有限
  wave-loop order；regular
  mapping、relation-derived reuse和coarse resource estimate只改变proposal顺序，开关后tiny accepted domain与winner不变；
- independent tiny reference domain enumerator不调用production domain builder，证明complete finite小图合法域完整；
- test-only flat reference composer先证明plan set/legality/cost；独立actualization runner才可从fresh source逐plan调用真实materializer与exact
  gates，且production search entry不可达该runner；
- cheap legality、exact coverage、topology symmetry、canonical dedup、已证明SPM lower bound和raw-work dominance只有在不删除合法最优解时才能在state expansion前剪枝；
- symbolic footprint只有proven must-coexist lower bound超过capacity才能早拒绝；ranking estimate、未经boundary-faithful proof
  的solver/model结果和nominal bandwidth projection不能签发packing、overlap或legality；exact局部solver的proof/proposal也须
  由selected typed materialization与正常gate复验；
- production planning不物化任何choice；只有selected full-proof plan构造一个CardModule并执行一次Tile→Instr、fresh completion、SPM/DDR、
  transport/resource/ABI和actual work parity；
- proven exact failure只拒绝对应causal assignment并消耗确定work unit，不触发late repair、retile、spill或另一selector；
  `ResourceExhausted`、solver timeout或internal failure属于indeterminate，必须保留合法state，不能形成no-good；
- logical demand outcome必须区分`satisfied`、带direct uncovered/role witness的`proven logical infeasible`、`unsupported semantic
  relation`和`indeterminate/compiler failure`；只有第二类可以删除当前placement trial，禁止将`FailureOr + string`压成
  legality bool或no-good；
- 对同一logical trial切换dense/strided/multi-piece carrier能力、layout或route可用性，exact demand与logical outcome必须
  extensionally相同；physical分解必须回证pieces union等于原set，carrier失败只拒绝对应representation/movement assignment；
- exact-demand cache跨immutable borrow、nested structural snapshot变化、consumer domain、producer ownership或
  partition/reduction/replication role变化时必须失效；`IREpoch`只拒绝跨borrow trial，不作为mutation detector或semantic key。
  任何窄cache key都要有extensional equivalence proof，不能以当前单轴fixture观察结果代签；
- serial/parallel proposal evaluation得到相同admitted set、winner和package identity；
- wall/RSS是回归证据，不设任意60秒硬gate；
- beam、candidate cap、随机启发式或其它会损失完整性/最优性的策略，只能在实际负载profiling后作为显式trade-off启用，
  并持续报告相对小图oracle和`none`的质量差异。
- top-k还必须报告`best-found@k`、winner recall@k、regret@k和estimate-vs-final recost误差；永久丢弃合法completion时结果只可
  标`budgeted-feasible`，外部系统的固定`k`不得成为本项目默认值。

`search`和`none`跨越同一output seam。`none`从正常上游IR自行完成deterministic conservative baseline的功能合法化并
materialize accepted executable；它不是只消费预选fixed assignment的validator。`search`从完整性能合法域惰性生成candidate。
测试不得把两者相同结果写成长期合同，也不得为某个case硬编码winner。

搜索结果分级必须与实际coverage一致：finite域与planning objective/global bound闭合才是`objective-optimal`；未展开completion仍由完整
exact continuation和admissible bound表示时可为`feasible-with-bound`；objective Unknown/incomparable时是`feasible-unranked`；缺bound或
已经丢弃合法completion时只能是`budgeted-feasible`。单纯预算中止不自动产生bound，也不能把planning optimum外推成硬件最优。

### 4.4 Cost model

每个comparison cohort预先确定统一typed target facts与enabled terms；exact work、admissible bound和estimate分开。测试必须证明：

- hard legality/capacity failure仍然fail closed；
- 缺性能参数、dynamic multiplicity和arithmetic overflow产生typed Unknown/Incomparable，不按0或极大值比较；
- per-Tile compute、card DDR、endpoint/minimum-hop/cut NoC、per-Tile explicit SPM movement和available control work保留各自knowledge；只有
  qualified exact route才有directed-link work；
- dependency phases相加，独立branch/资源取并发最大值，只有显式double/triple buffering才用steady-state II；
- estimate只排priority，不剪枝；lower bound逐prefix与flat completion minimum比较不高估；raw work与term source不写入selected IR。

## 5. Completion、memory 与 transport gates

### 5.1 Completion reconstruction

completion从final actual Instr的effects、worker issue domains、async tokens、control-flow boundaries和resource reuse重新
构造。旧completion必须先清除；fresh rewrite完成后验证：

- reuse不能跨未完成producer；
- loop backedge、branch merge、entry return和Direct-DTE exact wait闭合；
- `ReturnAfterLocalDrain`只声明Tile-local return条件，不冒充card-scoped barrier；
- CardExecutable成功需要16个Tile entry及transport obligations全部完成。

planning必须在J中证明completion；selected materialization仍缺失completion是compiler bug并终止compile，不能由runtime轮询、
统一entry尾等待或返回planner重选掩盖。

### 5.2 SPM/DDR

- search-time SPM live set包含inputs、resident intermediates、outputs、temporaries、layout buffers、NoC staging和rotating buffers；
- exact SPM packing只读final roots/lifetimes/conflicts，返回validated offsets或失败；
- DDR planning覆盖program resource、spill/reload、workspace、status和observable output，offset/alignment/range无overflow；
- allocator不改变selected tile、fusion、region、order、worker、communication或completion；
- logical element work、physical footprint与host payload分别检查，禁止用element count代替physical bytes。

### 5.3 Cross-Tile communication

- peer/collective只从显式physical source/destination、message identity、domain、encoding、bytes和topology派生；
- send/recv all-and-only matching，fanout lifetime覆盖最后consumer，fanin/reduction等待完整；
- local overlap、Direct-DTE event与NCC completion使用typed effect/resource关系；
- 不从logical partition、Tile编号算术、symbol名或容器顺序恢复route/algorithm；
- communication cost与staging footprint进入同一physical-dataflow candidate，不存在late profitability selector。

## 6. Target、package 与 runtime gates

### 6.1 Target LLVM/output

- exactly 16个Tile interfaces携带独立 `(card_id, tile_id, launch_slot)`；
- non-identity tile/slot mapping通过所有translation/readback；
- current target LLVM schema、target identity、runtime ABI、format、entry和typed Tile entry arguments逐项相等；
- Grid/Cluster aggregate materialization保持每Tile body与显式interface，dispatch不假设pid、slot和Tile ID相等；
- unsupported target call/dtype/layout/geometry、overflow、undefined symbol、bad digest均在writing前失败；
- 任一Tile失败时无部分target root可见。

### 6.2 ExecutablePackage

- ordinary package manifest与profile activation分别严格检查自己的current top-level identity和exact field set；所有文件均拒绝额外/缺失field，不保留version branch或旧reader；
- `card_count=1`、`tile_count=16`，entries覆盖all-and-only Tiles与dense launch slots；
- target identity/runtime ABI/module format与kernel launch mode/entry ABI/ordered phases同CardExecutable和module readback逐项相等，
  model launch kind或fallback输入必须拒绝；
- program data、ProgramTensor、TargetTensor、inputs、outputs、modules、entries和arguments all-and-only covered，
  无悬空、重复ID、未引用文件或source NPY/tree；
- ProgramTensor logical descriptor、TargetTensor `MemLayout`/shape/physical bytes/alignment和existing codec逐项一致；
  同一ProgramTensor的多个selected representations显式分离，current external port只有一个target descriptor且没有TargetTensor ID或package bytes；
- parameter/constant无需caller binding；每个package-owned TargetTensor只materialize一次，多个Tile arguments只有引用同一
  TargetTensor ID才能共享bytes；
- `program-data.bin`的deterministic offset、span、base/entry alignment、canonical zero padding和whole-file digest闭合；
  overlap、unaccounted gap、overflow、truncation、trailing bytes、wrong layout/alignment和自动identity合并均拒绝；
- TargetTensor为空时program data必须是canonical zero-byte member、`total_bytes=0`、`base_alignment=1`且runtime无对应provider call；
  非空时total bytes为正并覆盖全部TargetTensor；
- entry-local workspace/profile/status requirement只被对应entry引用；它们不进入ProgramTensor/TargetTensor表；
- entry completion只接受 `return_after_local_drain`；Direct-DTE status ABI/size/alignment/access/watchdog exact；
- canonical serialization、parse、semantic verification、module/data digest、whole-root tree closure和atomic writing roundtrip。

### 6.3 No-card/board runtime

no-card必须在任何provider side effect前闭合target/runtime capability、binding、memory plan、module/export/phase、transport
和16-Tile invocation plan。runtime与`wafer-run`均不提供按EntryId选择单个Tile的validation或执行入口；逐Tile记录只由
完整invocation内部构造，也不提供可传任意Tile count的独立空session资格化入口。board正负例验证：

- provider inventory中的显式tile/launch relation；
- package、显式device qualification与live inventory的Tile domain exact match，不接受更大domain中的16-Tile子集；
- non-empty program data一次大块allocation/整体H2D、empty program data零provider call、invocation memory一次大块allocation、
  TargetTensor地址=`base+offset`；
- input/output、每Tile workspace/profile/status和pointer rows的child ranges non-overlap且alignment正确；
- card-scoped phase submission、absolute deadline、completion observation、D2H和cleanup；
- partial/unknown accepted subset、timeout或不可信状态使session poisoned，且无后续provider call；
- profile instrumentation不存在可普通执行，存在但旧/stale/malformed必须pre-effect失败。

## 7. Target model gates

model必须消费与target writing相同的owner-backed target module set，不能重新lower或使用accepted-IR第二解释器。

验证包括：

- TargetCall descriptor registry与decoder对每种typed payload、worker和completion behavior闭合；
- frontend为每个transaction显式绑定target card/tile/launch slot和Tile-local issue ordinal；
- `begin`、16个Tile的`executeTile`与单次`finish`构成原子调用生命周期，任一失败`abort`且不返回partial result；
- SystemC一Tile一SC_THREAD，跨Tiledata-ready/completion关系由event表达，无OS thread或symbol恢复身份；
- private address spaces、range/alias/hazard、family-specific formal numeric与qualified bulk lane；formal/model从decoded TargetCall
  直接验证并执行或typed拒绝，不经过model profile、capability pattern或resolved-command registry；
- bulk qualification record只按concrete operation problem、physical payload、comparator、backend/environment和implementation
  evidence严格匹配；model support、bulk qualification与board correlation互不代签；
- complete output physical bytes解码为source dtype/shape，与独立CPU expected比较并检查NaN/Inf/tolerance policy。

SystemC是functional-event model，不声明cycle accuracy、板端吞吐或真实NoC arbitration。model pass不替代current manifest exact
package provider或真实board gate。

## 8. Source fidelity 与 workload matrix

frontend fixture必须直接使用真实source语义：operation、operand、constant、mask、RoPE、scalar flow、dtype、control flow
和function boundary原样进入compiler。RoPE table可以由模型按普通常量预计算；mask中的finite值或`-inf`原样lower。compiler
不注入、删除或特判这些值。

Q53无卡matrix至少包含：

- generic GEMM、elementwise、reduction、conv与mixed DAG；
- branch/fanin/fanout和layout-changing/stateful DAG；
- official HF prefill FP16/BF16；
- functional two-step KV-cache decode FP16/BF16，state通过普通inputs/results线程化；
- Llama-2 7B block FP16/BF16。

所有workload走同一public source→package path。case-specific harness只提供source、payload和oracle，不生成compiler marker、
special pass option、shape shortcut或手写替代graph。

## 9. Program data 与 whole-program scale

数据链和整图规模是两类证据，不能用单block互相代签：

1. tiny multi-binding golden验证ProgramDataSource→ProgramDataRange→TargetTensor→program-data offset、padding和digest；
   内容相同但identity不同的ProgramTensor不得自动合并，同一ProgramTensor的不同TargetTensor保持独立；
2. 最大单tensor验证checked大范围算术、bounded source/target window与无全tensor element-object materialization；
3. 完整大型parameter/external captured-constant inventory验证每个ProgramDataSource的bounded read/hash次数可解释、每次compile的
   package-owned TargetTensor只transform一次并占据一个deterministic file range；external input/output的compile-time transform和
   package byte计数为零，但不声明完整graph已编译；Q56 fake-provider/board另验证whole program data非空时只H2D一次、为空时零次；
4. 完整小模型graph验证embedding/多层或等价完整结构、final transform、output head及其all-and-only data bindings都进入同一public path；
5. Q61在Q53之后再以至少一个完整大图验证frontend、IR、search、target和package的共同规模行为。

每次compiler scale run至少记录source logical bytes、target physical bytes、package data bytes、ProgramTensor/ProgramDataSource/
ProgramDataRange/TargetTensor/file range计数，transform/write/read次数、bytes read/written、peak RSS、peak disk、最大live window、
alignment overhead，以及IR op、planning-state/work counts、stage wall time和typed failure分类。Q56 fake-provider/board runtime另记录H2D；
H2D不是Q58/Q61完成前置。完成不变量为：

```text
transform_count(package_owned_target_tensor) == 1
serialized_range_count(package_product, package_owned_target_tensor) == 1
transform_count(external_input_or_output_port) == 0
serialized_range_count(package_product, external_input_or_output_port) == 0
program_data_bytes == sum(target_tensor_physical_spans) + canonical_alignment_padding
additional_heap = O(max_live_window), not O(total_parameter_bytes * tile_count)
```

Q56 fake-provider/board的独立runtime不变量是
`one_shot_h2d_count(program_data) == (program_data_bytes > 0 ? 1 : 0)`；Q57另证明PreparedExecution lifetime内不重复上传。

Llama-2 7B完整inventory或完整graph可以作为可选named scale witness，但层数、模型名、attention/KV语义和serving policy不进入
通用合同，也不替代Q61 mandatory feasible large-graph case。Q61不默认要求full-model board execution；该named witness若因
current target容量、target-model能力或host预算不足，必须按stage报告typed unsupported/capacity，不得用缩成单block冒充，
但也不阻塞已经满足的通用Q61完成门禁。

## 10. Q49.P、Q50、Q51–Q53 completion gate

各队列项分别形成fresh证据，不能用后项的局部通过倒签前项：

1. Q49.P从current TensorProgram构造一个live canonical plan，调用闭包不包含search state/candidate、search-oriented domain/ranking
   evaluator、proposal order/group materializer或candidate统计，也不包含完整placement-option生成、domain propagation、recursive
   CSP/backtracking或任何“从option列表返回一个assignment”的helper。Q50.B-foundation先提供closed SpatialAssignment与canonical
   baseline producer，Q50.A只推导demand；随后迁其余baseline components并交付Q50.F closed-plan core，不等待完整search domain；
   每个coordinate只运行A/F等pure typed query，planning
   CardModule/Instr/Q50.0均为零；direct ExactRejection只能触发预定义、单调、不分支且不回溯的functional legalization transition，
   旧coordinate不作为alternative保留。FullFeasibilityProof关闭plan后，才沿TensorProgram→一个CardModule→TileRegion/Instr→fresh
   SPM/DDR→CardExecutable→ExecutablePackage执行一次actual链；actual resource problem与proof不一致或Q50.0失败是compiler bug，
   不继续fallback。
   每个baseline TileRegion恰有一个structured compute root和exact demand证明必要的non-root support，同Tile多root形成多个顺序region，
   跨root shaped dependency显式DDR；root cardinality按materialization relation映回structured DAG node，显式init producer不能伪装成
   support，一个root lower成多个compute/instruction op也不能误判成多root。plan-level resource query用typed relation表达所需
   region/function scope，不构造scratch IR；winner construction/bufferization后的relation必须仍指向actual current
   consumer/result/output buffer，不能以旧value仍live或retain/drop代替语义完整性。capacity witness将all-and-only conflict owner
   关联到当前single root及temporal assignment，equal-shape fanin不扩大归因。
   从没有selected assignment的正常TensorProgram开始，初始temporal tile超SPM时沿完整合法breakpoint lattice以pure query重建
   operand/halo/result/temporary/movement/alignment/bank/lifetime problem，缩到第一个full-proof fit后再生成package/no-card；
   multi-axis/tail/minimum-granularity受测，最小合法tile仍被完整proof拒绝才返回typed capacity failure，heuristic no-fit、solver耗尽
   与indeterminate不能冒充unsupported。fresh小图、五类relation、overfull-to-fit、轻量source-to-package/no-card及一轮FP16 LLaMA
   必须重新证明plan-only work count、actual/package/oracle；旧181.70秒LLaMA只作旧功能证据。Q49.P重新完成前不运行LLaMA search；
   完成后不再建立独立baseline板端任务，Q53只把current accepted baseline作为matched A/B一侧。
2. Q50.0：baseline与search共用无策略CardExecutable compile/verification boundary，任何lowering失败均不隐式repair；Q50.A：
   每份closed `SpatialAssignment`只携带exact execution shards、Tile embedding和per-output-piece reduction merge placement；
   func-scoped relation analysis以唯一operand-level SSA worklist派生program/constant/structured boundaries、final result owners、
   reconstruction和reduction merge requirements。M/N/K mixed、多reduction轴、multi-result/coupled reduction及init exactly-once由
   typed tests证明；partial contribution不得作为下游result owner。supported relation保持construction normal form并在操作前
   受work bound约束，不进入无界generic Presburger equality/subtraction；MLIR analysis invalidation替代manual epoch/fingerprint。
   outcome区分satisfied、unsupported semantics、indeterminate resource exhaustion和compiler contract error；cross-op demand不作
   普通placement no-good，carrier/layout/route失败不反写spatial legality。Q49.P canonical carrier与Q50.B、Q50.G/H/J消费同一
   proof。Q50.S：softmax-weighted-sum先归一为自包含semantic op；per-root domain只选择materialized-softmax或online recurrence，
   K/V block与partition分别由temporal/spatial轴证明。planning不物化算法point，winner由最终TileRegion
   materializer构造一次selected work；normalization、online recurrence、partial merge和failure atomicity各有direct witness。
3. Q50.B–Q50.K依次闭合spatial placement、single-root TileRegion、coupled traversal/region fusion、complete temporal tile与
   wave-loop order、
   scoped planning feasibility、layout/representation、movement、rotating buffers、event/resource schedule与conditional stage pipeline。
   Q50.B必须新增current尚不存在的reduction spatial factor、partial ownership与显式merge；Q50.E复用已经能物化的reduction
   temporal tile，但须以多reduction轴property/reference enumerator证明完整breakpoint与wave-loop domain。两项证据不得互相代签。
   probe缺少因果坐标时必须deferred，资源耗尽不得当作不可行；pipeline event structure形成后必须使旧calendar失效并
   重入event/resource schedule，由新assignment证明实际overlap。
   每项都需current接入点、production consumer、actual-IR witness和实际执行的正负测试；删除旧owner前还必须逐项映射其中独有
   algorithm/proof/diagnostic/test witness，旧owner删除和新domain存在都不能代替能力迁移。
4. Q51.Core：直接从immutable TensorProgram建立search session，不调用Q49.P、不接收baseline executable/actual cost，也不把
   baseline choices当未来轴default；Core闭合typed transition、deterministic frontier、stable dedup、global work/budget、
   dependency expansion和coverage/bound evidence。每轴independent reference
   enumerator随Q50.S/B–K交付，通用pure planning feasibility归Q50.F。Core从零建立new-search控制并接管public `search`，不适配旧
   candidate/generator/feedback/selector；旧控制链只能在能力迁移完成后删除。test-only flat exhaustive runner、全部联合维度、
   exact planning domain和new source-to-package链由Q51闭合。production search在typed state上选出一个winner，只允许该winner
   materialize并调用一次Q50.0；Q50.0失败是planning/lowering合同缺口，不得回到candidate set反复物化。selected IR必须有共享TileRegion，并以
   coupled traversal或明确retained SSA、tile-sized intermediate及无中间DDR round-trip证明有效融合；group字段不能代签。
5. Q52：它是Q51完整new-search链闭合后首个允许执行重型LLaMA `search`和质量profile的任务；Q49.P的单次fresh FP16 LLaMA
   `optimization-none`只证明baseline功能/materialization。generic/HF/LLaMA representative load只在显式、bounded profile批次
   记录work、wall、RSS和热点，不进入普通回归。基于实测引入的优化在小图oracle上不改变最优结果；search与同源`none`的
   质量差异只由独立matched A/B报告，没有固定shape/tile/fusion/buffer shortcut。
6. Q60先把framework adapter与pre-exported portable StableHLO接入同一product source contract。Q53从同一只读source bytes为
   `none`与`search`分别启动全新parse/import、ProgramData、output和package transaction；两条policy各自提交一个plan，绝不共享
   Module/analysis/executable或相互fallback。small structured DAG、conv mixed DAG、capacity-forced cross-Tile fanout/reduction、
   temporal-tail/storage、HF prefill、functional two-step decode和Llama block全部由产品入口fresh生成完整ExecutablePackage并
   fresh no-card。accepted Tile dataflow、Instr、Target LLVM、package和runtime plan按typed identity逐层对应；actual effectiveness
   证明region retained value无不必要DDR往返、exact temporal tail、physical version/use、cross-Tile payload/completion、storage
   lifetime/offset及event/resource order，而不是统计group/op/string。每个package包含all-and-only 16 Tile entries、current
   TileEntryArgument/program-data/completion闭包，runner、oracle、guard、decode continuation和fixed timeout在无板阶段全部完整；相关
   registered tests实际执行而非unsupported/skipped后才标`board-ready`。
7. 真实板端使用一个进程和一个`QualifiedBoardRuntimeSession`：第一次完整case同时完成device qualification，后续case不重复
   enumeration/selection/inventory；case串行，但每phase一次提交complete 16-Tile domain，`launch_slot`不表示Tile串行。small
   multi-Tile communication先通过correctness/lifecycle；Llama及prefill或functional decode代表分别用独立`none`/`search` package
   做settling和固定paired A/B，pair内顺序交替，所有sample逐次通过case oracle/guard。primary metric只取launch-to-completion，
   并结合completion observation resolution判定Improved/Regressed/Inconclusive；不删outlier、不见结果追加sample。Llama固定3 pairs、
   attention代表固定5 pairs且均为Improved后Q53才标`done`；Inconclusive/Regressed仍保持`board-ready`。timeout、device异常或poison
   立即停止，不retry/reset/power；历史输出、Q51 cost、Q52 profile和理论work不能代签。

Direct-DTE overlap、host search telemetry、framework capture/oracle和layout/movement能力直接由上述current
Q50/Q51/Q52/Q60/Q53边界拥有，不再建立独立旧任务门禁。Q47 current ABI correctness与Q56 program-data/one-shot runtime correctness
仍是低层独立板端合同；它们可以与Q53复用同一qualified device session串行执行，但不得合并completion语义或相互代签。

历史package、历史board raw、已删除harness、旧schema、instruction数量下降或理论估计都不能解除第1至第7项。
