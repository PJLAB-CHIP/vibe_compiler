# Wafer Compiler Verification Contract

状态：本文只定义当前架构的验证层级和完成证明，不保存任务动态状态或历史case台账。稳定验证链为
`TensorProgram -> physical-dataflow selection -> CardProgram/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
host局部gate、package roundtrip或一次source compile都不能把production任务提升为`board-ready`；没有真实设备matched
A/B改善时也不能标`done`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前source program、card-level GSPMD输出、normalized TensorProgram、selected CardProgram/TileProgram/TileRegion、
  final Instr、CardExecutable、target modules/artifacts、ExecutablePackage及其独立oracle。
- Current stage responsibility:
  在每个IR/artifact边界验证语义、coverage、physical identity、resource、completion、ABI、publication与execution；
  建立host、no-card、model和真实板端证据之间不可越级的完成层级。
- Output artifact / IR:
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
2. **Artifact evidence**：同一compile transaction生成并readback CardProgram/Instr、CardExecutable、target module和ExecutablePackage。
3. **No-card evidence**：真实package经strict loader与runtime preflight形成完整16-Tile invocation plan，且无provider effect。
4. **Functional model evidence**：同次owner-backed target module set经TargetCall frontend/SystemC执行，完整output与独立CPU expected比较。
5. **Board correctness evidence**：当前构建、当前package、当前payload在真实设备完成output/guard和lifecycle检查。
6. **Board performance evidence**：同一source/config/payload/ABI的baseline/winner做matched、重复、可解释的A/B。

只有第3层完成，且case、oracle、runner都齐全，板端任务才可写 `board-ready`。只有任务定义所需的第5/6层通过才可
写 `done`。instruction count、理论makespan、host wall time、SystemC event count和no-card成功都不是实卡性能证据。

## 3. Freshness 与执行纪律

- 每轮代码/测试改变后只使用本轮build和本轮输出；历史raw/log/report只作审计记录，不作为test input。
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

### 4.1 CardProgram / TileRegion / Instr

正例必须证明：

- `num_partitions`只描述card-level GSPMD domain；single-card current path固定 `num_partitions=1`；
- `wafer.card.program`拥有all-and-only 16个available `wafer.tile.program`；
- distinct Tile programs可含不同op、loop、temporal tile、region和执行长度；
- no-work Tile仍有合法entry并进入artifact/runtime domain；
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

candidate generation只能读取current structured semantics、indexing maps、SSA/effects、type/shape/dtype、explicit
physical communication和target capability。测试要搜索并拒绝framework/model/function/value-name、固定shape、operand
position、Attention/decode/mask专用matcher或公共pass残留。

### 4.3 Search correctness 与 exact gates

- semantic、spatial、temporal、TileRegion/融合、layout、movement、buffer、communication和order合法域从current IR惰性生成；
- independent tiny reference domain enumerator不调用production domain builder，证明complete finite小图合法域完整；
- production-mechanism flat exhaustive runner使用真实materializer、cost与exact gates全展开，再证明frontier、剪枝和winner；
- cheap legality、exact coverage、topology symmetry、canonical dedup、已证明SPM lower bound和raw-work dominance只有在不删除合法最优解时才能在clone前剪枝；
- 需要exact evaluation的choice才物化CardProgram actual candidate，任一时刻最多一个live actual clone；
- 所有materialized candidate执行同一Tile→Instr、fresh completion、SPM/DDR、transport/resource/ABI和final recost；
- proven exact failure只拒绝对应causal assignment并消耗确定work unit，不触发late repair、retile、spill或另一selector；
  `ResourceExhausted`、solver timeout或internal failure属于indeterminate，必须保留合法state，不能形成no-good；
- serial/parallel proposal evaluation得到相同admitted set、winner和package digest；
- wall/RSS是回归证据，不设任意60秒硬gate；
- beam、candidate cap、随机启发式或其它会损失完整性/最优性的策略，只能在实际负载profiling后作为显式trade-off启用，
  并持续报告相对小图oracle和`none`的质量差异。

`search`和`none`跨越同一artifact seam。`none`只materialize deterministic conservative baseline；`search`从完整合法域
惰性生成candidate。测试不得把两者相同结果写成长期合同，也不得为某个case硬编码winner。

搜索结果分级必须与实际coverage一致：finite域与global bound闭合才是`optimal-certified`；未展开completion仍由完整
exact continuation和admissible bound表示时可为`feasible-with-bound`；已经丢弃或未表示合法completion时只能是
`budgeted-feasible`。单纯预算中止不自动降级，也不能在exact continuation不完整时伪造bound。

### 4.4 Cost model

每个comparison cohort预先确定统一enabled terms：有实际参数用实际值，否则使用明确理论值，完全未知的性能项从
所有candidate同时删除。测试必须证明：

- hard legality/capacity failure仍然fail closed；
- optional性能参数缺失不会产生Unknown比较状态或阻塞候选排序；
- per-Tile compute、card DDR、directed-link NoC、per-Tile explicit SPM movement和available control work均为数值；
- dependency phases相加，独立branch/资源取并发最大值，只有显式double/triple buffering才用steady-state II；
- raw work与enabled-term source可诊断，但不写入selected IR/package。

## 5. Completion、memory 与 transport gates

### 5.1 Completion reconstruction

completion从final actual Instr的effects、worker issue domains、async tokens、control-flow boundaries和resource reuse重新
构造。旧completion必须先清除；fresh rewrite完成后验证：

- reuse不能跨未完成producer；
- loop backedge、branch merge、entry return和Direct-DTE exact wait闭合；
- `ReturnAfterLocalDrain`只声明Tile-local return条件，不冒充card-scoped barrier；
- CardExecutable成功需要16个Tile entry及transport obligations全部完成。

缺失completion是candidate failure，不能由runtime轮询或统一entry尾等待掩盖。

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

### 6.1 Target LLVM/artifact

- exactly 16个Tile interfaces携带独立 `(card_id, tile_id, launch_slot)`；
- non-identity tile/slot mapping通过所有translation/readback；
- current target LLVM schema、target identity、runtime ABI、format、entry和typed ABI slots逐项相等；
- Grid/Cluster aggregate materialization保持每Tile body与显式interface，dispatch不假设pid、slot和Tile ID相等；
- unsupported target call/dtype/layout/geometry、overflow、undefined symbol、bad digest均在publication前失败；
- 任一Tile失败时无部分target root可见。

### 6.2 ExecutablePackage

- ordinary package strict `schema_version=8`、profile companion strict `schema_version=9`，旧version和额外/缺失field拒绝；
- `card_count=1`、`tile_count=16`，entries覆盖all-and-only physical Tiles与dense launch slots；
- program-boundary resources为card scope并被16个entries引用；workspace/status为Tile scope且只被对应entry引用；
- resources、modules、entries和slots all-and-only covered，无悬空或重复ID；
- entry completion只接受 `return_after_local_drain`；Direct-DTE status ABI/size/alignment/access/watchdog exact；
- canonical serialization、parse、semantic verification、module digest和atomic publication roundtrip。

### 6.3 No-card/board runtime

no-card必须在任何provider side effect前闭合target/runtime capability、binding、resource、module/export/phase、transport
和16-Tile invocation plan。runtime与`wafer-run`均不提供按EntryId选择单个Tile的preflight或执行入口；逐Tile记录只由
完整invocation内部构造，也不提供可传任意Tile count的独立空session资格化入口。board正负例验证：

- provider inventory中的显式tile/launch relation；
- package、显式device qualification与live inventory的Tile domain exact match，不接受更大domain中的16-Tile子集；
- card-scoped resource单次分配/共享地址与Tile-scoped隔离；
- card-scoped phase submission、absolute deadline、completion observation、D2H和cleanup；
- partial/unknown accepted subset、timeout或不可信状态使session poisoned，且无后续provider call；
- profile companion不存在可普通执行，存在但旧/stale/malformed必须pre-effect失败。

## 7. Target model gates

model必须消费与target publication相同的owner-backed target module set，不能重新lower或使用accepted-IR第二解释器。

验证包括：

- TargetCall descriptor registry与decoder对每种typed payload、worker和completion behavior闭合；
- frontend为每个transaction显式绑定physical card/tile/launch slot和Tile-local issue ordinal；
- begin/execute 16 Tiles/prepareCommit/commit原子生命周期，任一失败abort且不发布partial effects；
- SystemC一Tile一SC_THREAD，跨Tiledata-ready/completion关系由event表达，无OS thread或symbol恢复身份；
- private address spaces、range/alias/hazard、formal numeric与qualified bulk lane；
- complete output physical bytes解码为source dtype/shape，与独立CPU expected比较并检查NaN/Inf/tolerance policy。

SystemC是functional-event model，不声明cycle accuracy、板端吞吐或真实NoC arbitration。model pass不替代schema-v8 exact
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

## 9. Q49/P、Q50、Q51–Q53 completion gate

各队列项分别形成fresh证据，不能用后项的局部通过倒签前项：

1. Q49：`none`通过current TensorProgram→CardProgram→TileRegion/Instr→fresh SPM/DDR→CardExecutable→ExecutablePackage链路；
   普通多op、spatially sharded compute和cross-Tile baseline package/no-card通过。Q49.P用fresh prefill/decode/Llama证明
   CardProgram、CardExecutable与package digest稳定、oracle/no-card通过，并以fresh阶段计时证明`none`不构造search对象、
   不重复全图materialization，且只对accepted baseline完整编译一次。
2. Q50.0：baseline与search共用无策略CardExecutable compile/admission boundary，任何lowering失败均不隐式repair；Q50.A：
   placement给定后从IndexRelation形成layout-independent exact logical demand，carrier/layout/route失败不反写spatial legality；
   Q50.S：typed proof和online/partitioned-KV等算法参数点均物化成真实TensorProgram alternatives。
3. Q50.B–Q50.K依次闭合spatial placement、single-op TileRegion、coupled traversal/region fusion、complete temporal tile、
   scoped actual probe、layout/representation、movement、rotating buffers、event/resource schedule与conditional stage pipeline。
   probe缺少因果坐标时必须deferred，资源耗尽不得当作不可行；pipeline event structure形成后必须使旧calendar失效并
   重入event/resource schedule，由新assignment证明实际overlap。
   每项都需current接入点、actual-IR witness和实际执行的正负测试；旧owner删除不能代替能力迁移。
4. Q51.Core：independent reference enumerator与production flat exhaustive runner、baseline incumbent、global ledger、budget与
   actual-probe seam闭合；Q51中两层oracle与`search` winner一致，全部联合维度实际参与选择，late exact failure回到同一
   frontier。selected IR必须有共享TileRegion，并以coupled traversal或明确retained SSA、tile-sized intermediate及无中间
   DDR round-trip证明有效融合；group字段不能代签。
5. Q52：generic/HF/Llama representative load的work、wall、RSS和热点fresh记录；基于实测引入的优化在小图oracle上
   不改变最优结果，代表负载不劣于同源`none`，没有固定shape/tile/fusion/buffer shortcut。
6. Q53：generic DAG与HF/Llama matrix全部由current source fresh生成完整ExecutablePackage并fresh no-card；每个package
   包含all-and-only 16 physical Tile entries、current ABI/resources/completion，runner与oracle完整；融合有效性证据闭合后
   才标`board-ready`。
7. 真实设备上Llama及一个prefill/decode代表分别做同源`none`/`search` matched A/B，exact output/guard通过，
   多次样本显示可重复实际改善，Q53才标`done`。

历史package、历史board raw、已删除harness、旧schema、instruction数量下降或理论估计都不能解除第1至第7项。
