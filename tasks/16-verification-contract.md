# Wafer Compiler Verification Contract

状态：本文只定义当前架构的验证层级和完成证明，不保存历史case台账。current whole-card主线已拆为Q49、
Q50.A–Q50.K及Q51–Q53；Q49为`board-ready`，当前只执行Q50.A。host局部gate、schema-v8 roundtrip或一次source compile都不能把Q53提升为
`board-ready`；没有真实设备matched A/B改善时也不能标`done`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前source program、card-level GSPMD输出、whole-card structured DAG、selected wafer.card.program/
  wafer.tile.program、final Instr、TargetLLVMModuleBundle、TargetArtifactBundle、schema-v8 package及其独立oracle。
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
2. **Artifact evidence**：同一production transaction生成并readback accepted IR、target module和schema-v8 package。
3. **No-card evidence**：真实package经strict loader与runtime preflight形成完整16-Tile invocation plan，且无provider effect。
4. **Functional model evidence**：同次TargetLLVMModuleBundle经TargetCall frontend/SystemC执行，完整output与独立CPU expected比较。
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

## 4. IR 与 whole-card synthesis gates

### 4.1 Card/Tile MPMD

正例必须证明：

- `num_partitions`只描述card-level GSPMD domain；single-card current path固定 `num_partitions=1`；
- `wafer.card.program`拥有all-and-only 16个available `wafer.tile.program`；
- distinct Tile programs可含不同op、loop、temporal tile、region和执行长度；
- no-work Tile仍有合法entry并进入artifact/runtime domain；
- `(card_id, tile_id)`唯一，Tile-local SPM root不跨Tile SSA alias。

负例至少覆盖duplicate/unavailable/missing Tile、coverage hole/overlap、非法reduction overlap、cross-Tile SPM alias、
mismatched send/recv、payload/domain/encoding mismatch、缺失wait和premature release。

### 4.2 Whole-DAG scheduler

同一个通用scheduler需要覆盖：

- single-op intra-op spatial mapping；
- chain中的producer/consumer wave pipeline；
- independent branch分配到disjoint Tile groups；
- diamond、fanout、fanin与reduction；
- partial co-location、mapping redistribution、multicast/gather/reduction；
- same-region不同temporal tile、selective spill/reload、recompute和region cut；
- compute、DDR、NoC与显式SPM movement在已有independence/buffering证明时的overlap。

candidate generation只能读取current structured semantics、indexing maps、SSA/effects、type/shape/dtype、explicit
physical communication和target capability。测试要搜索并拒绝framework/model/function/value-name、固定shape、operand
position、Attention/decode/mask专用matcher或公共pass残留。

### 4.3 Search bounds 与 exact gates

- cheap legality、coverage、topology symmetry、SPM lower bound和raw-work dominance在clone前剪枝；
- shortlist才物化whole-card actual candidate，任一时刻最多一个live actual clone；
- 所有materialized candidate执行同一Tile→Instr、fresh completion、SPM/DDR、transport/resource/ABI和final recost；
- exact failure只拒绝该candidate并消耗确定work unit，不触发late repair、retile、spill或另一selector；
- serial/parallel proposal evaluation得到相同admitted set、winner和package digest；
- wall/RSS是回归证据，不设任意60秒硬gate。

`search`和`none`跨越同一artifact seam。`none`只materialize deterministic conservative baseline；`search`
使用bounded search。测试不得把两者相同结果写成长期合同，也不得为某个case硬编码winner。

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
- `ReturnAfterLocalDrain`只声明Tile-local return条件，不冒充whole-card barrier；
- whole-card成功需要16个Tile entry及transport obligations全部完成。

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
- communication cost与staging footprint进入同一whole-DAG candidate，不存在late profitability selector。

## 6. Target、package 与 runtime gates

### 6.1 Target LLVM/artifact

- exactly 16个Tile interfaces携带独立 `(card_id, tile_id, launch_slot)`；
- non-identity tile/slot mapping通过所有translation/readback；
- current target LLVM schema、target identity、runtime ABI、format、entry和typed ABI slots逐项相等；
- Grid/Cluster aggregate materialization保持每Tile body与显式interface，dispatch不假设pid、slot和Tile ID相等；
- unsupported target call/dtype/layout/geometry、overflow、undefined symbol、bad digest均在publication前失败；
- 任一Tile失败时无部分target root可见。

### 6.2 Schema-v8 package

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
- whole-card phase submission、absolute deadline、completion observation、D2H和cleanup；
- partial/unknown accepted subset、timeout或不可信状态使session poisoned，且无后续provider call；
- profile companion不存在可普通执行，存在但旧/stale/malformed必须pre-effect失败。

## 7. Target model gates

model必须消费与target publication相同的 owner-backed `TargetLLVMModuleBundle`，不能重新lower或使用accepted-IR第二解释器。

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

## 9. Q49、Q50.A–Q50.K与Q51–Q53 completion gate

各队列项分别形成fresh证据，不能用后项的局部通过倒签前项：

1. Q49：`none`通过current source→CardProgram→physical Tile→Instr→fresh SPM/DDR→admission→package链路；
   普通多op、spatially sharded compute和cross-Tile baseline package/no-card通过。
2. Q50.A–Q50.K：每个子项只验收自己的current接入点、actual-IR witness和实际执行的正负测试并独立提交；
   旧owner删除不能代替能力迁移，某一子项通过也不能代签其它机制。
3. Q51：小图完整枚举oracle与`search` winner一致；全部联合维度实际参与选择，late exact failure回到同一frontier，
   且selected IR存在有效多op fusion group。
4. Q52：generic/HF/Llama representative load的work、wall、RSS和热点fresh记录；基于实测引入的优化在小图oracle上
   不改变最优结果，代表负载不劣于同源`none`，没有固定shape/tile/fusion/buffer shortcut。
5. Q53：generic DAG与HF/Llama matrix全部由current source fresh生成完整schema-v8 package并fresh no-card；每个package
   包含all-and-only 16 physical Tile entries、current ABI/resources/completion，runner与oracle完整；融合有效性证据闭合后
   才标`board-ready`。
6. 真实设备上Llama及一个prefill/decode代表分别做同源`none`/`search` matched A/B，exact output/guard通过，
   多次样本显示可重复实际改善，Q53才标`done`。

历史package、历史board raw、已删除harness、旧schema、instruction数量下降或理论估计都不能解除第1至第6项。
