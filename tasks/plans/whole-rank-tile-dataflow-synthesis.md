# Whole-Rank Tile Dataflow Synthesis 实施计划

状态：设计与施工边界已收敛，C0已完成，C1–C6待实施。任务状态以 `tasks/progress.md` 中
`whole-rank-tile-dataflow-synthesis` 为准。

本计划只拆解 `tasks/06-physical-dataflow-synthesis.md` 的施工顺序、迁移删除面和验证 checkpoint；算法、IR 和
pipeline contract以该设计文档为唯一owner；exact workload、收益和board门禁只由`tasks/16-verification-contract.md`
的对应gate拥有。Q32/Q38–Q41/Q46/Q47已有的relation、layout、actual-clone、worker、NoC、overlap、计时、packing和
current ABI能力全部复用，本任务不再发明替代层。

Checkpoint严格按`C0 → C1 → C2 → C3 → C4 → C5 → C6`推进；C3与C5可拆成多个可评审子提交，但不能越过前一checkpoint的gate。

## 施工边界

```text
post-SPMD complete-rank structured IR
  -> consumer-driven selected Tile/Dataflow IR
  -> terminal complete-rank Instr candidates for each typed collective/peer algorithm parameter
  -> worker / fixed-slot / ready-order siblings
  -> erase and fresh-rebuild completion
  -> SPM / DDR / post-memory Direct-DTE / ABI exact gates
  -> atomic all-rank ExecutableBundle
```

施工期间始终满足：

- candidate的语义主体是actual IR clone；允许current-IR上的transient component walk，不引入persistent、serialized或
  cross-stage component/fusion graph、layout plan、repair sidecar或新schema。
- 新算法只按 SSA、structured interfaces、`IndexRelation`、effects、typed numeric/target capability分派；禁止模型名、
  参数名、固定shape和字符串op-name matcher。
- `tile.region`物化selected dataflow中的maximal SPM residency/SSA containment domain；它不是独立physical
  arena、candidate-selection、completion或lowering单元。全部sibling regions仍由whole-rank allocator联合packing。
- allocator、conversion、completion和all-rank verifier仍是独立owner；只有06 decision owner生成sibling和选winner。
- baseline永远通过同一生产pipeline和late exact gates，不调用被退役的兼容路径。
- 每个checkpoint必须包含真实source到actual rewrite再到该阶段所有terminal gates；空框架、手工pass或局部fixture不算闭合。

## C0：Fresh Baseline 与可审计 Work Counters

目标：冻结实现前事实，建立能同时解释 DDR、GS、NCC/join、Instr 和 compile-search work 的同源计数。

实施：

1. 扩展现有`InstructionProgramCost`/`estimateStats`事实源和invocation-local diagnostics，不另造第二套count逻辑；独立dump parser
   只作test oracle。计数明确区分static site、static-trip loop-expanded execution work、conditional path lower/upper bound、
   symbolic/Unknown和Q9 runtime-measured count，不写manifest、package、schema或长期IR attr。
2. 分别记录每rank与all-rank aggregate：
   - RDMA/WDMA calls和bytes；
   - GS/pack/unpack calls和bytes；
   - NCC collective/peer engine work、Direct-DTE work、`NCCJoin`；
   - compute/recompute、各engine issue、dependency depth和`max(per-rank critical-path lower bound)`；
   - SPM high-water、DDR arena high-water、buffer/descriptor pressure。
3. 记录candidate expanded states、actual clones、terminal Tile→Instr lowerings、SPM/DDR pack calls、peak RSS和Release wall time。
   wall/RSS复用Q41的compile-transaction timer与process peak-RSS事实源：同一Release build、source/payload/options、`nproc`
   并发度和host身份下先做1次不计入的warm-up，再以3个独立process测量；冻结wall中位数、peak RSS最大值和每次search-work counters。
4. 对同一source、payload、launch、ABI和fresh build生成pre-Q49 optimized production baseline与`none`保守参照；两者只允许
   optimization policy不同。C0从当前pipeline dump post-SPMD structured IR与final Instr IR；selected complete-rank
   Tile/Dataflow IR必须在C1建立真实decision point后由同一driver直接dump。当前per-task pipeline在selection前已经破坏性lower
   Tile candidate，C0不得为补齐中间dump而保留/replay影子clone或建立sidecar。dump名和诊断不含任务号、case名或临时目录约定。
5. Fresh重放16的Q49 gate所定义的当前Llama primary，生成TP16 FP16完整package/no-card；只做本任务定向gate，不重跑无关catalog或历史板测。

Gate：

- fixed static loops的expanded work可从dump独立精确复算；conditional/dynamic control只报告bounds、symbolic或Unknown；
- serial/parallel compile得到相同work counts、winner digest和package；
- 当前output、guard、ABI与package contract不变；
- baseline记录至少包含本轮fresh RDMA/WDMA/GS/join/bytes、SPM high-water、compile work/RSS/wall；
- C1建立complete-rank decision point后，selected Tile/Dataflow dump能与同一次compile的post-SPMD和final Instr直接关联；
- 未满足以上条件，不得用旧日志或理论机会量签后续收益。

不算完成：只统计静态call site、只报全卡Instr总数、用aggregate/平均值隐藏逐rank与rank-maxima，或把未校准work加成伪cycle。

### C0 fresh evidence（2026-08-05）

同一Release build、同一PyTorch-exported FP16 Llama block source/payload、TP16 kernel launch与current ABI下，production和
`none`均从source生成完整16-rank package并fresh no-card。production三次独立正常并发compile的transaction wall为
`90269 / 93817 / 90493 ms`，中位数`90493 ms`；peak RSS为`2952700 / 2969012 / 2951464 KiB`，最大值
`2969012 KiB`。三次compiler work均严格相同：expanded states `37488`、terminal clones `48784`、terminal
Tile→Instr lowerings `49040`、SPM planning `60084`、DDR planning `46992`。

固定单CPU affinity的完整同源compile得到相同五项work、winner Instr与递归package bytes；其transaction wall为
`1352964 ms`、peak RSS为`814604 KiB`。正常并发三份package、单CPU package、manifest与rank-0 Instr digest分别逐字节一致；
manifest digest为`bead668201b76b7728813ed25a657f542ffda9093d03e2092c99b7451b16facb`，rank-0 Instr digest为
`445dd648b1d3f287e1ab9c1c8195778a0f36293cdfd85d4759f2870be3302d9f`。并发执行只改变吞吐和RSS，不改变search work或artifact。

final Instr同源counter如下；call/site与loop-expanded execution均保留在diagnostic中，表中列出最直接回答当前问题的exact work：

| all-rank exact work | production | `none` | production相对`none` |
| --- | ---: | ---: | ---: |
| Instr | 10324 | 15120 | -31.7% |
| GS calls | 3280 | 4064 | -19.3% |
| GS bytes | 885675520 | 904680960 | -2.1% |
| Direct-DTE send / receive / wait | 60 / 60 / 120 | 960 / 960 / 960 | DTE总work -91.7% |
| `NCCJoin` | 870 | 1792 | -51.5% |
| nonterminal `NCCJoin` | 854 | 1776 | -51.9% |
| DDR read / write bytes | 888076384 / 430446080 | 888076384 / 430446080 | 0% |
| SPM movement bytes | 2204197984 | 2223203424 | -0.9% |
| compiler-owned SPM / DDR buffers | 3824 / 832 | 4112 / 832 | SPM -7.0%，DDR 0% |
| summed SPM / DDR high-water bytes | 46850048 / 95428608 | 46850048 / 95428608 | 0% |

production的rank-0为`137 RDMA / 86 WDMA / 205 GS / 54 join`，rank-15为相同movement与`54 join`；各维度
rank-maxima为`658 Instr / 205 GS / 56 join / 24 DTE`，不是伪造的单一critical rank。现有dependency-depth analysis不能解释
该final Instr中的structured control，因而明确报告`Unknown(unsupported-control-flow)`，没有当作零或伪造cycle。

这组事实说明旧独立优化确实大量压低了Direct-DTE与约一半join，但没有减少任何DDR traffic/high-water，GS bytes也只下降
2.1%；因此后续收益不能再靠late局部cleanup，必须由C1开始把decision point移到complete-rank pre-Instr structured IR。
当前driver只能直接保存post-SPMD structured IR与final Instr；selected complete-rank Tile/Dataflow证据按上面的边界在C1建立，
没有为C0引入shadow clone、replay或sidecar。

## C1：Complete-Rank Decision Point

目标：在任何不可逆 Tile→Instr、SPM/DDR placement或terminal join之前，建立完整rank的structured decision scope。

实施：

1. 调整production orchestration，使scheduler先持有每个rank的完整post-SPMD structured clone；旧task/scope只能作为
   source IR中的结构，不再各自形成standalone function和lowered candidate。bring-up先在同一complete-rank clone内按现有scope
   保守物化Tile IR，再只执行一次complete-rank Tile→Instr；不得保留per-scope selection、standalone module、placement或join。
2. 从current SSA、`TilingInterface`、`IndexRelation`、effects、control flow和numeric contract派生transient component；
   对每条edge分别计算tile propagation、numeric、storage、completion和rank-coupling legality。
3. 冻结有限producer集合和严格拓扑progress measure；每次propagation只能消费尚未覆盖的domain/edge，避免single-tile
   fusion worklist重入或同一view链反复展开。
4. 含collective的actual clones不能rank-local误剪；coordinator可从current typed IR即时派生transient hash预筛，但每个tuple
   必须逐项fresh重证，hash不进入state、IR、schema或正确性判断；不用task/layout ordinal拼card-wide tuple。
5. 在winner仍选conservative baseline的bring-up阶段，保持selected IR和package等价，先证明新的decision point没有语义漂移。

Gate：

- chain、diamond、fanin/fanout、shared input、multi-root、structured control flow、unknown/observable effect的component与
  edge-legality正负例通过；
- 旧source scope、shape变化、layout不兼容和collective不会自动切component或强制intermediate store/reload；
  winner随后重建maximal SPM residency regions。最终sibling-region data edge必须是显式
  store→completion→load，不能保留SPM SSA；external input仍按storage semantics显式load，但不因此形成
  standalone lowering边界；
- final `tile.region` inputs/results为variadic，chain、diamond、fanin/fanout和multi-root不受人为边数上限；
- source-observable ordering只阻断对应schedule/completion action；unsupported numeric reorder、不可表示control flow及其它
  未知legality也只阻断其对应维度，只有所有可用实现都无法跨越时才成为component separator；
- production candidate construction不再调用单task独立Tile→Instr/SPM/DDR再import的路径；旧死代码统一在C6删除；
- rank-count 1/16 baseline经完整late gates和fresh no-card。

不算完成：只构建C++ component graph而未改变production边界，或暂时保留两条decision pipeline等待以后切换。

## C2：Consumer-Driven Tile Connection

目标：从consumer tile反推producer slice/iteration，统一生成fuse、share、recompute和materialize cut，不按算子堆pattern。

实施：

1. 从output/observable root的finite tile domain出发，用DPS、indexing maps和`TilingInterface`计算operand tiles，沿
   `IndexRelation`穿过view、slice、broadcast、permutation和reshape。
2. 对pure producer生成actual alternatives：direct SSA tile、shared version、consumer-local recompute和loop-invariant
   materialize；对不能direct传播的edge保留显式movement baseline。
3. fanout/fanin在同一live frontier内联合处理；shared input的load/layout placement随consumer loop construction决定，
   不在Instr层对effectful load做事后LICM。
4. 第一条production vertical使用通用indexing semantics把transpose/view relation吸收到oriented contraction actual clone，
   再对effect-proven read-only、原始逻辑shape的weight执行exact composed Tensor DDR mapped transfer到Cx；不得把它描述成
   identity DMA，也不得假设任意transpose可直接RDMA。
5. 在相同通用vertical中证明loop-invariant LHS load/layout位于N tile loop外，且不依赖QKV、gate/up、weight名或4096/11008。

Gate：

- chain、diamond、shared-input contraction、fanout/fanin、tail、multi-root和不同scalar body的`linalg.generic` held-out通过；
- full-weight runtime transpose的RDMA→GS→WDMA链在适用oriented-GEMM winner中消失；
- LHS RDMA/GS static-trip loop-expanded work不随N-loop trip count线性重复；
- resident direct edge没有中间WDMA/RDMA；
- 每个winner从真实source经Tile/Instr、SPM/DDR、descriptor、ABI、package和fresh no-card；
- heuristic冻结后才运行held-out graph，失败不得添加shape/op-name特例。

不算完成：只用late transfer elimination删一两个GS，或只对手写`wafer.tile.*` fixture产生预期IR。

## C3：Joint Physical Frontier 与 Capacity Feedback

目标：在完整tiled clone上共同选择唯一`MemLayout`、physical version、mapped/local/peer route、residency、spill和loop order。

实施：

1. 将现有layout PBQP迁为同一component/analysis epoch的factor reducer；proposal被保留后立即materialize到actual unplaced
   Tile clone并销毁，不保留per-task assignment或`physicalLayoutProposalOrdinal`。
2. 对未处理IR可观察的live-frontier facts完全相同的chain/tree state做frontier DP；一般小型DAG使用deterministic Pareto beam。
   该等价关系每次从actual clone fresh重算，transient hash只可预筛，不能形成新interface/schema或正确性key；dominance不能只看
   当前DDR/GS/Instr。
3. 把mapped load、local GS、secondary physical version、peer SPM、explicit spill/reload和loop placement实现为同一action algebra；
   canonicalization只删除winner里已经可证明的no-op。
4. structured frontier只使用从current Tile IR派生的capacity lower bound/Unknown，不提前调用Instr-level allocator。terminal
   SPM/DDR planner只返回validated placement/high-water或typed failure；decision owner从本次evaluation仍持有的无offset actual
   Tile parent生成有限retile、spill低收益edge、share→recompute、layout/route、loop order和double-buffer siblings；allocator不返回repair。
5. all-rank collective/peer alternatives共同物化；rank pruning保留可能组成whole-card winner的actual clones，tuple从current IR逐项重证。
6. SPM movement bytes保持final-IR exact work/pressure维度；删除现存NoC profitability中的历史per-tile
   `128 GB/s` SPM flat-duration项。bank phase只作allocator最后tie-break，不产生candidate-level latency或收益。

建议初始确定性预算（由C0 fresh work counters标定后冻结）：

- 每个live-frontier Pareto集合宽度不超过8；
- 每rank optimized terminal candidates不超过16；
- reserved baseline完整exact gate只执行一次；
- intermediate transitions只运行relation/type/effect/footprint lower bound和局部verifier；
- full Tile→Instr、SPM和DDR gate只对terminal survivors执行；
- budget按deterministic work units，不按wall-clock timeout；耗尽时保留baseline。

Gate：

- SPM不足只影响相关resident edges，不触发all-or-nothing full-buffer回退；
- fanout可选择共享version、有界K个consumer-compatible versions、分支local conversion或局部spill；K是search policy，不是IR语义上限；
- 三个consumer且存在多个互不兼容physical demands的held-out graph通过，不把初始primary+secondary覆盖写成通用上限；
- 每个survivor都是actual clone，factor/search对象销毁后verifier和lowering结论不变；
- packing failure不留下partial offsets，不使allocator修改tile/layout/order；
- serial/parallel frontier和winner确定一致；
- final cost不再包含SPM0/RAM_ACC 1024-bit接口外推的per-tile SPM flat duration；
- 不出现“region/task数 × recipe数”的full lowering笛卡尔积，expanded states、terminal lowering次数和RSS保持显式bounded；

不算完成：独立选layout、resident和NoC再做artifact Cartesian product，或用外部solver结果作为production语义输入。

## C4：Terminal Lowering、Worker 与 Fresh Completion

目标：只有terminal complete-rank Tile candidates才lower Instr；从最终worker/effect/range重建fixed-frontier latest-necessary
completion，再执行一次packing与独立alias验证。

实施：

1. 将candidate evaluation改为只接受complete-rank Tile clone；一次性Tile→Instr conversion和function-boundary bufferization。
2. 在unplaced Instr parent上派生有限worker、fixed-slot和ready-order siblings。pre-Instr search不保存worker/NCC shadow schedule。
3. completion API的canonical input不含旧`wafer.instr.ncc_join`：先删除clone中全部该op，再从current worker order、typed issue/effect、event token、alias/range、
   reuse和observer fresh rebuild。source-observable ordering由control-flow、SSA event/token和typed effects表达；DTE wait与
   group barrier按各自typed op语义处理。
4. 在固定worker/order/effect frontier上构造latest-necessary completion；每个join必须有hazard、protocol或observable-barrier witness，
   可安全coalesce的必须合并。地址复用只能生成bounded explicit siblings：separate slot、reorder、worker change、join或spill。
5. 每个completion sibling按`lifetime/conflict → SPM packing → physical-alias verification`执行一次。worker/order/slot/join sibling
   从无offset Instr parent生成；需要retile/spill/layout改变时回到同一evaluation的actual Tile parent生成新structured sibling。
   allocator不插join、不修改order，也不原地迭代到fixed point。它保持hard feasibility/exact verification owner，
   只允许在hard-valid且actual high-water/fragmentation/其它candidate-visible primary cost相同的placements内，
   使用offset-derived bank phase作最后tie-break；该偏好不得反馈或改变spill、region、
   DDR movement、worker/order或join。

Gate：

- standalone task不再执行Tile→Instr/SPM/DDR，task/region return不再自动插terminal `NCCJoin`；
- same-worker有序chain且无跨engine/复用/observer hazard时不插join；
- NCC→DTE/Kcore/external publication、cross-worker conflict、collective protocol和terminal observer保留精确completion；
- 每个剩余join都能从current IR重算hazard witness，join不替代DTE wait或group barrier；
- completion变化后的lifetime、SPM placement、physical range、descriptor和target gate全部fresh通过；
- late failure销毁完整clone，不恢复旧保守join pipeline。

不算完成：只调用现有“保留已有join”的normalizer，或在packing后无条件批量删除join。

## C5：Collective、Partial Reduction 与 Online Semantic Candidate

目标：让collective前后的SPM tile保持可选resident，并让短/长reduction通过typed语义选择native、partial或online实现。

实施可拆成三个顺序子提交：

1. **C5a collective residency + native/partial**：logical collective保持typed rank/group/payload semantics；Direct/Ring/Tree仅作为
   terminal conversion的typed参数逐点生成actual Instr sibling；参数/proposal在rewrite后立即销毁，actual Instr sibling继续进入
   worker/completion/memory/all-rank gates，只有失败时才销毁。compute tile可跨collective连接，但真实NCC/DTE wait、
   participant completion和all-rank resource matching不被融合删除。collective不自动
   产生DDR store/reload。reduction domain由一个legal tile覆盖时生成现有native typed reduce；跨tile时先复用
   `PartialReductionOpInterface`形成stable partial/two-pass actual clone。
2. **C5b typed online state/numeric/lowering**：online softmax/attention作为独立typed semantic candidate实现，不从普通tiling
   隐式假定合法。fused attention明确`has_value`与`(m,l,o)`的init、tile state、state-state merge及`y=o/l` finalize；
   scale/bias/boolean或additive mask在score域的应用顺序、broadcast relation、accumulator dtype、exp/rounding、
   NaN/Inf/signed-zero、tail、underflow/overflow、empty tile与fully-masked row均接入typed numeric proof/gate。只有SSA证明
   probability无额外observable use、dropout为identity且没有dynamic RNG时才生成fused candidate。standalone softmax若输出
   完整probability，必须在actual IR中显式二次遍历re-read/recompute，或保存并按global state重标定numerator；其SPM/DDR
   与work全部计价，不能用`(m,l)`单遍冒充。当前用现有SCF + tile reduce/elementwise/GEMM表达，不新增Instr/ABI；
   numeric或target合同未闭合时保留native/partial baseline，不按`S`阈值写算法分支。Q48以后只负责自动发现/证明更广variant。
3. **C5c workload matrix**：严格复用16的Q49 gate。read-only cached-attention/prefill使用真实PyTorch-exported attention source；
   长K/V在一次invocation内逐tile遍历/消费显式external K/V，不称runtime streaming resource，也不冒充persistent decode或完整7B block。

Gate：

- native、partial和online三类分别有typed positive/negative与numeric oracle；unsupported路径fail closed；
- collective本身不切region；collective前后仍resident的edge位于同一maximal region且没有DDR round-trip；
- score/partial/state traversal有complete coverage、tail和completion证明；
- 至少一个真实PyTorch-exported cached-attention或prefill case由production cost/legality实际选中partial或online sibling，且
  final IR证明没有完整score/probability tensor的DDR store→reload；仅让synthetic positive能生成candidate不算完成；
- 16规定的FP16/BF16 rank-count 1/16 source→package→model/no-card矩阵通过；
- held-out失败只能暴露interface/capability/resource问题，不能追加shape matcher。

不算完成：把现有block改成`S=1`冒充decode，或只用公式/文档宣称online softmax已支持。

## C6：Production Cutover 与旧路径删除

目标：默认driver只有一条complete-rank decision/materialization路径，旧owner consumer为零并删除代码和开关。

必须迁移或删除：

- standalone task function clone、per-task Tile→Instr→SPM/DDR evaluation和task candidate commit；
- task `func.return` terminal join；
- `taskAlternativeOrdinal`、`spmWorkingSetMultiplicity`和per-task layout ordinal；
- spill/spill-ready/resident/resident-ready等artifact-kind Cartesian machinery；
- all-or-nothing full-buffer residency decision owner；
- late NoC-resident tuple decision owner；typed peer/collective mechanics迁入06/13 action library；
- 历史NoC profitability中的per-tile SPM flat-rate duration及其配置/测试期望；SPM bytes保留为exact work，
  不用未经校准的单一rate计时；
- 被统一owner吸收的scope/layout/residency/NoC独立public optimization flags；
- 任何隐藏compatibility pipeline、手工pass拼接或名字/shape matcher。

实现审计至少点名`CandidateCommit.cpp`、`FullBufferHandoff.cpp`、`RankCandidateFrontier.h`的artifact enum、
`ScheduleTensorProgramInternal.h`的旧ordinal/multiplicity/layout fields、`NoCResidentDataflow.cpp`的decision部分和
`OptimizationConfig.h`旧独立flags。Q40的DTE-overlap/fixed-slot consumer迁到terminal Instr sibling；Q41的timing、bounded executor、
RSS和diagnostics继续复用，配置/CLI/help/tests同步收口为public `production`/`none`两种policy，不保留scope/residency/NoC等独立开关。
qualification选择仍可用compiler-private typed seam，但不是public optimization axis。

保留但降为单一职责：

- redundant transfer、GS coalescing、view folding和local LICM只作policy-free generic canonicalization；
- worker placement、fixed-slot、PBQP和NoC lowering保留为typed mechanics/action library；
- SPM/DDR allocator继续作为hard exact gate；SPM allocator只在hard-valid placements内用可重算bank phase作
  末级软偏好，不拥有candidate dataflow选择；
- `tile.region`继续表达maximal SPM residency、structured traversal和SSA containment，不因cutover删除。

Gate：

- `wafer-compile`是唯一production named pipeline，production/none A/B都走同一owner和late gates；
- 旧symbols、config fields、diagnostics和tests的consumer静态搜索为空；仅archive可保留历史文字；
- current typed capability、conservative baseline和negative failure均由新路径覆盖；
- Release model-scale compile按C0同一Q41测量协议复跑；三次search-work counters均满足显式上界，wall中位数不超过C0的110%，
  peak RSS最大值不超过C0的105%。若correctness bring-up暂时超出，只能在checkpoint内诊断，不能以减少重复次数、改变并发/
  workload、放宽容差或删除exact gate换取通过；
- compiler host/unit/lit/integration按`nproc`最大安全并发通过，unsupported/skipped单独核对；
- Llama与attention矩阵生成完整package并通过fresh no-card，达到`board-ready`。
- 板端matched baseline/winner均由cutover后同一fresh build和新pipeline重新生成；不得复用C0旧package。

不算完成：只关闭feature flag、留下死代码/旧test seam，或只让新路径在`wafer-opt`手工pass中工作。

## Workload 收益与验证门禁

所有exact workload、收益阈值、counter口径和board矩阵只引用`tasks/16-verification-contract.md`的Q49 gate；本计划不复制第二份
数字或shape事实源。比较基线是C0 fresh pre-Q49 optimized production，不是`none`；typed capability若证明某阈值不可达，先修改06/16并
说明证据，不能在implementation plan中静默放宽或追加case特化。

## 验证层次

### Unit / property

- edge-legality vector、component traversal、termination和complete coverage；
- IndexRelation pull/push、tail union、fanout/shared-input relation；
- current-IR frontier-fact equivalence、beam dominance、baseline retention和budget exhaustion；
- actual-clone atomicity、PBQP proposal销毁、packing sibling和无partial offset；
- completion hazard witness、erase+fresh rebuild、single-pass packing与physical-alias rejection；
- native/partial/online numeric和unsupported cases。

### IR / integration

- oriented GEMM + effect-proven read-only weight的exact composed mapped transfer，无runtime full transpose；
- LHS load/layout不随N-loop重复；
- spill消除后producer/consumer合并进入同一maximal region，intra-region resident edge无WDMA/RDMA；
- sibling region之间没有SPM memref/root/alias，且只由显式DDR store/completion/load连接；
- collective有正确wait/completion但无自动DDR；
- same-worker ordered chain不join，cross-worker/range reuse/observer必须join；
- chain、diamond、fanout/fanin、multi-root、reduction、control flow和effect barrier；
- FP16/BF16、整tile和tail；两个不同scalar body的generic op证明无case matcher。

### Source / package / model

- rank-count 1和TP16当前Llama full block；
- read-only cached-attention与long prefill主/held-out矩阵；
- CPU/PyTorch oracle、完整output/guard、SystemC/CModel、typed ABI、package和fresh no-card；
- 按rank分别报告static site、loop-expanded/bounded work、Q9 runtime-measured count和aggregate shared-resource demand。

### Board

无卡阶段必须准备完整case、oracle、runner、package并fresh no-card，状态最多为`board-ready`。真实板端按仓库规则单进程、
逐case、bounded timeout执行16的Q49 board矩阵。设备异常后停止，不自动retry/reset/power；长prefill由host/model承担泛化gate，
不把完整prefill板测设为唯一外部阻塞。

## 提交与收尾

每个checkpoint单独形成可评审提交，且同步：

- `tasks/progress.md`只更新当前row状态；
- 06及直接受影响的07–13、16、18 pipeline contract；
- 新的稳定bug pattern或workflow才写`memory/`，不写临时case状态；
- fresh验证命令、实际执行/unsupported清单和未完成风险进入提交说明或task evidence，不复制成第二份设计。

C0–C6、全部host/package/no-card gate和旧路径删除完成后只达到`board-ready`；真实板端matched correctness/performance通过后，
才可将任务标为`done`。
