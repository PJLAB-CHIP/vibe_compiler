# Wafer Compiler Task Queue

更新时间：2026-08-15

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。历史状态变化由Git保留。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级。
- 全局至多一个`doing`；`next`表示前置已满足但尚未开始，`queued`表示已经进入当前主线但仍等待直接前置闭合，
  `later`表示不进入当前主线。
- `done`只表示对应设计文档的completion gate已经满足；详细证据只链接owner，不复制到本文件。
- `blocked`必须写明尚缺的外部条件或上游任务，不能用工作过程代替阻塞原因。
- 新任务先进入本表并绑定编号设计owner；非小修再建立`tasks/plans/`实施计划。
- 状态变化只更新对应row；不得追加按日期、轮次或测试批次展开的worklog。
- 后续任务gate只验证直接影响面；除非任务本身改动相关ABI/runtime/firmware，否则禁止全量审计和无关回归。
- 包含板端验证的任务必须把`board-ready`写为无卡阶段门禁；`board-ready`不得标记为`done`。

## 当前调度

```text
Q32 + Q6.B -> Q9 profiler foundation                         [done]
Q32 + Q6.B + Q37 -> Q38 multi-engine software pipelining    [done]
Q38 -> Q39 NoC-resident tile dataflow                        [done]
Q43 Vibe Compiler collaboration review materials             [done]
Q42 test gate scope reduction                                [done]
Q42 + Q32.C -> Q41 compiler search scalability               [board-ready]
Q42 + Q39 -> Q40 composed search and DTE overlap             [board-ready]
Q15 + Q18 + Q35 -> Q44 PyTorch source board verticals        [board-ready]
Q32 + Q37 -> Q46 layout movement elimination                 [board-ready]
Q46 compiler closure -> Q47 target ABI retirement            [board-ready]
Q32 + Q38-Q41 + Q46/Q47 compiler mechanics
  -> Q49 current card baseline stabilization                 [board-ready]
Q49 -> Q50.0 actual executable compilation boundary           [done]
Q50.0 -> Q54 MLIR infrastructure conformance                  [done]
Q45 compiler terminology and naming                           [done]
Q9 + Q45 + Q47 current interfaces -> Q55 current-interface consolidation [done]
Q50.0 + Q54 + Q55 -> Q56 executable-package data closure      [doing]
Q56 board-ready + Q50.0 + Q54 + Q45
  -> Q49.P baseline control-flow isolation                     [queued]
Q50.0 + Q54 -> Q50.A exact placement-demand boundary          [queued]
Q50.0 + Q50.A -> Q51.Core common search kernel                [queued]
Q51.Core -> Q50.S structured semantic alternatives            [queued]
Q51.Core + Q50.A -> Q50.B spatial partition and placement    [queued]
Q50.B -> Q50.C maximal single-op TileRegion                  [queued]
Q50.C -> Q50.D coupled traversal and region fusion           [queued]
Q50.D -> Q50.E complete temporal tiling                      [queued]
Q50.E -> Q50.F scoped actual-region probe                    [queued]
Q50.F -> Q50.G layout and physical representation            [queued]
Q50.G -> Q50.H explicit data movement                        [queued]
Q50.H -> Q50.I rotating buffers                              [queued]
Q50.I -> Q50.J event and resource scheduling                 [queued]
Q50.J -> Q50.K conditional stage pipeline                    [queued]
Q50.S + Q50.B–Q50.K -> Q51 unified search correctness       [queued]
  -> Q52 workload-driven search scalability                  [queued]
Q52 + Q44 source mechanics + Q47 compiler ABI closure + Q56 board-ready
  -> Q53 current production board readiness                  [queued]
Q53 board-ready + Q56 board-ready
  -> Q57 resident static execution                             [later]
Q53 board-ready -> Q48 semantic superoptimization             [later]
```

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前工作与完成门禁 | 设计 / 计划 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `production-output-profiler` | `done` | Q32、Q6.B、configured board | 未插桩Primary TX stream launch-to-completion设备包络、分离的host diagnostics、Trace来源的五类NCC per-tile engine active ns/work-volume摘要、独立Direct-DTE cycles/raw activity、Count/Trace、逐次保留真实动态调用与rdcycle的16-tile timeline、exclusive语义成本与非加和Trace-only成本、final Instr静态work与硬件峰值下界对照、exact output、单一production output instrumentation、三文件专业UI和profile全树`0777`均闭合；不构造card-wide纯engine elapsed，静态cost不回灌ranking。Ranking feedback保留为后续独立门禁。 | 06、14-16；`tasks/archive/board-profiler.md` |
| Q38 | `multi-engine-software-pipelining` | `done` | Q32、Q6.B、Q37 | 历史任务已经证明typed multi-buffer、prologue/steady/epilogue、Direct-DTE issue/wait和exact range hazard可表达、可验证；旧fixed-slot candidate、worker selector及其专用测试已删除。Q50.I/Q50.J迁移机制后，Q51必须把buffering/worker作为同一physical-dataflow state的联合动作直接物化到selected Tile IR，不得恢复独立owner。 | 06、08-17；历史计划已归档 |
| Q39 | `noc-resident-tile-dataflow` | `done` | Q38历史表达/验证能力闭合；板端资格作为独立external gate | physical peer materialization、resident/spill mechanics和板端exact资格已经闭合；旧Tile collective/late selector、静态profitability owner及其output合同已删除，movement mechanism由Q50.H保全迁移，choice统一由Q51 physical-dataflow search拥有。 | 02-13、16-17；历史计划已归档 |
| Q43 | `compiler-collaboration-review-materials` | `done` | Q9、Q37-Q39完成证据 | 2026-08-03冻结的历史汇报材料已经闭合；它不作为current architecture合同。 | 01、08–16；`tasks/archive/vibe-compiler-collaboration-review.md` |
| Q42 | `test-load-reduction` | `done` | 无 | 默认lit/unit/CTest和owner integration均只判直接合同；历史catalog、批量测试矩阵、model-scale与重复package执行已退出默认入口。 | 16；`tasks/archive/test-gate-scope-reduction.md` |
| Q40 | `composed-choice-search-and-dte-overlap` | `board-ready` | Q39、Q42完成 | Direct-DTE issue→FP16/BF16 compute→exact wait结构witness、serialized baseline和buffered overlap qualification mechanics已经闭合；旧组合owner及package不再是current production evidence。真实板端matched A/B尚未执行，Q50.I/Q50.J迁移typed buffer、completion和overlap机制，Q51才统一选择。 | 06、08-13、16；历史计划已归档 |
| Q41 | `compiler-search-scalability` | `board-ready` | Q32.C bounded executor、Q42；M-sharded K=1024复现 | stage/pipeline/pass/analysis/candidate计时、RSS、bounded executor和低扰动诊断基础已经闭合；pre-current-card model-scale compile与package仅作历史性能背景，current性能证据由Q52、current package证据由Q53 fresh生成。public policy只保留`search`/`none`，剪枝不得按shape/op/name恢复语义。 | 06、14-16、18；`tasks/archive/compiler-search-scalability.md` |
| Q44 | `pytorch-source-board-verticals` | `board-ready` | Q15、Q18、Q35；Q41通用movement lowering与compile scalability | PyTorch/XLA capture、固定seed、case-owned dtype、同module eager oracle和原dtype/shape比较mechanics已经闭合；旧执行域package与ABI fixture已删除，GEMM、KV-cache decode和Llama workload必须由Q53 current source-to-ExecutablePackage pipeline fresh重建后才形成新的board-ready证据。 | 02、15-16；历史计划已归档 |
| Q46 | `layout-movement-elimination` | `board-ready` | Q32、Q37；复用Q41已闭合的compiler-side有界搜索与计时基础 | 唯一`MemLayout`事实源、exact relation motion、typed layout choice、fanout共享和cross-Tile physical payload gate mechanics已经闭合；其旧候选owner已退出production，Q50.G/Q50.H迁移机制后layout、physical encoding和movement只作为Q51同一候选的维度，structured op只走唯一direct typed lowering。真实板端matched性能尚未执行。 | 06-08、10-11、13-14、16-17；历史计划已归档 |
| Q49 | `card-baseline-stabilization` | `board-ready` | Q32 relation/actual-clone/exact-gate mechanics；当前CardModule实现 | `none`已由独立deterministic controller闭合：固定16-Tile、逐Linalg op独立temporal tiling、op间compiler-owned DDR、buffer=1、零fusion/零可选edge-action search；spatial shard不一致时只物化typed relation要求的deterministic peer fragments并在consumer端DDR assembly，不搜索通信方案。2026-08-11最终代码上FP16原始DAG prefill、两步functional decode和Llama均fresh生成完整package并通过no-card，且accepted baseline直接复用唯一exact-admitted executable。尚无本轮真实板端输出，因此不是`done`；Q50仍独立迁移搜索能力。 | 01、03-16、18；`tasks/plans/physical-dataflow-synthesis.md`；`test/Tools/wafer-compile-card-baseline.test` |
| Q50.0 | `actual-executable-compilation-boundary` | `done` | Q49 | 唯一无策略CardModule→CardExecutable边界已闭合：输入已选择并物化的CardModule，依次完成Tile module splitting、TileRegion-to-Instr、fresh completion、fixed-capacity SPM/DDR、transport/resource/ABI verification，并以typed result区分accepted、proven exact rejection与indeterminate failure。baseline与search共用该入口；lowering不枚举、retile、spill或rebuffer，未知allocator/internal failure不会形成candidate no-good。直接单测覆盖可重复exact SPM rejection与indeterminate invocation，source-to-package baseline/no-card gate通过。 | 01、06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q54 | `mlir-infrastructure-conformance` | `done` | Q50.0 | 已按19号合同完成全active compiler source整改：semantic Location pointer payload、raw semantic attr、ordinal/print relation和synthetic local wrapper退出active路径；TileRegion/Func/Module按真实scope组织，16个atomic pass组成7条常驻named semantic pipeline与1条Shardy条件pipeline，production只由统一runner建PM；RegionBranch/effect/call、ODS、AnalysisManager、transactional pattern/conversion、typed host error和pure target/MLIR adapter分层闭合。fresh core/model build通过；Q54定向单测168/168、其余非搜索单测629/629、受影响card关系用例1/1、lit 1/1、feature-on依赖/模型/link 17/17、IR/source organization和diff检查通过。Q49/Q52长搜索未冒充本任务全量验证；Q49.P继续只处理baseline控制流重复物化，Q50.I继续物化真实共同wave/stage loop。 | 18-19；`tasks/plans/mlir-engineering-remediation.md` |
| Q55 | `current-interface-consolidation` | `done` | Q9 profile合同、Q45命名规则、Q47 current target/package/runtime接口 | Wafer-owned frontend metadata、package、target ABI/CRT、profiler、Direct-DTE status、qualification evidence、dependency record与工具fixture均收敛为一种无编号current合同，旧reader/CLI/carrier/alias与版本分支退出active路径。Board calibration逐项迁移source、shape/dtype、oracle、guard、status、timeout、cleanup和profile要求：22个raw probe加complete-Tile add普通/profile共24个current-interface no-card CTest通过，10个真实Board入口进入同一串行runner但本轮未上板；full-4096 K-tiled与M-tiled profile由current global-source runner和catalog/inventory保留，并因global lowering/SPM门禁明确阻塞，不冒充board-ready。fresh configure/build、target unit 140/140、core lit 216/216、current-interface 24/24、Board host 32/32、Q55 tool/runtime lit 5/5、dependency/source/IR/diff检查通过。 | 02、11、14-17、20；`tasks/plans/interface-version-consolidation.md` |
| Q56 | `executable-package-data-closure` | `doing` | Q50.0、Q54、Q55 | 只改`CardExecutable`之后的compiler→package→one-shot runtime边界：把调用端口、package-owned immutable data和compiler-planned internal storage从单一resource role/host-visible协议中拆开；将allocation root、checked view、logical tensor与physical layout/span分离；parameter/constant由compiler按final target ABI物化为digest-bound package data，runtime不再要求caller逐次绑定；package root只含all-and-only runtime成员并完整readback。同步替换manifest、writer/loader、no-card、board one-shot、target model、CLI和fixtures，不引入旧reader。fresh source→package/no-card、package data正负例、fake-provider one-shot与完整板端case达到`board-ready`后，Q49.P/Q53才消费新current package。 | 14-17、20；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q49.P | `baseline-control-flow-isolation` | `queued` | Q56达到`board-ready`；Q50.0、Q54、Q45 | 不改变Q49已签发的正确性合同，只清理`none`控制流：不得构造search state/candidate family，不得为每个temporal尝试反复编译whole graph；scoped probe直接消费Q54形成的真实TileRegion或最近合法isolated-anchor seam，不得把每个region包装成synthetic Module/Func后重跑完整Tile memory planning和verification；最后只对accepted baseline编译一次完整CardExecutable。fresh prefill/decode/Llama证明CardModule、CardExecutable与package digest稳定，oracle/no-card通过，阶段计时证明重复materialization热点已移除。 | 06、14、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.A | `exact-placement-demand-boundary` | `queued` | Q50.0、Q54 | `IndexRelation.image()`与ownership coverage分析及正负测试已经存在，但production placement query仍会立即降成dense rectangle、layout fragments和route，因此本项重新打开。完成时placement transition只消费layout-independent logical demand；representation/movement carrier失败不得删除spatial placement，compatibility lowering只在对应坐标关闭后消费exact demand。 | 05-07、09-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51.Core | `physical-dataflow-search-kernel` | `queued` | Q50.0、Q50.A | 先建立唯一query-local assignment state、typed transition、baseline incumbent、global work ledger、scoped actual-probe seam和两层small oracle，再接入其它轴：独立reference enumerator证明tiny合法域完整，production flat exhaustive runner证明candidate set/剪枝/winner。真实负载从第一天支持budgeted anytime；exact continuation与admissible bound仍完整时返回`feasible-with-bound`，永久丢弃合法completion后才是`budgeted-feasible`。core不得携带局部winner、派生lifetime/calendar、固定beam/cap、workload matcher或shadow schedule。 | 06-08、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.S | `structured-semantic-alternatives` | `queued` | Q51.Core | 从typed SSA证明可用算法族并在isolated clone中物化真实TensorProgram alternative；普通DAG、online recurrence和partition/merge只是验证witness。算法参数改变图时必须先成为actual structured DAG，再进入同一Q51 root candidate set；不提供public独立selector/pass，不按名字、shape或参数位置恢复语义。 | 05-06、08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.B | `spatial-partition-and-placement` | `queued` | Q51.Core、Q50.A | 从structured iterator语义生成all-iterator多轴factor vector、非整除remainder、parallel/reduction partition与显式merge，以及合法非最大、非矩形、非对称/非连通physical placement的完整惰性域；ordered-factorized regular mapping与compact/all-16仅为seed，不能固定participant，只有改变实际partition/embedding的axis/factor nesting才形成不同choice，等价生成路径canonicalize。不同op/branch/wave可选择不同Tile集合，consumer placement直接查询Q50.A。mechanism只生成typed transition/actual CardModule，不选局部winner；两层tiny oracle覆盖chain、branch、diamond、reduction与partial redistribution。 | 05-08、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.C | `maximal-single-op-tile-region` | `queued` | Q50.B | 对已选spatial shard物化覆盖all-and-only local work的单op TileRegion，使baseline和search复用同一region materializer；支持multi-result、DPS init、reduction与effect boundary，不选择temporal tile、layout或buffer。 | 05-08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.D | `coupled-traversal-region-fusion` | `queued` | Q50.C | 从SSA、iterator/index relation和effect构造合法region boundary与consumer-driven coupled traversal；actual fused candidate必须处于共同TileRegion并避免中间值无意义DDR round-trip。maximal closure与合法cut都进入共同candidate set，不能用单边`fused`标记或group字段代替实际融合。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.E | `complete-temporal-tiling` | `queued` | Q50.D | 每个op在每个spatial shard上生成覆盖全部iterator的完整temporal tile向量，以及Q50.D已选traversal内部语义可区分的有限wave-loop nesting/order；从完整local extent惰性枚举所有改变work/legality/reuse/lifetime的breakpoint与order。容量失败只对包含全部因果坐标的branch产生新breakpoint，不能把`tile/2`、默认loop order、固定seed或最大可放下tile固化成全局限制。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.F | `scoped-actual-region-probe` | `queued` | Q50.E | isolated clone只在全部causal coordinates关闭后物化受影响TileRegion并执行lowering、fresh lifetime/completion与fixed-capacity packing；symbolic footprint只有proven must-coexist lower bound超过capacity才能早拒绝，普通estimate只排序。缺layout/movement/buffer/order等坐标必须返回`deferred(required coordinates)`，资源耗尽/内部失败返回indeterminate，只有proven infeasible或确定unsupported才是typed exact rejection。probe不修改parent、不做late repair；memo/no-good key包含结论依赖的所有轴，本局部gate不提前签发跨轴可行性。 | 06-11、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.G | `layout-and-physical-representation` | `queued` | Q50.F | 为scheduled value生成typed MemLayout、encoding、physical version和必要conversion的完整合法候选，并实际物化到CardModule/TileRegion；fanout可共享同一physical version。旧layout selector与late default assignment退出，lowering只消费selected representation。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.H | `explicit-data-movement` | `queued` | Q50.G | 由exact logical demand、representation、placement/topology及retention/release/boundary obligations生成同TileRegion retained reuse、跨region显式DDR、recompute、unicast/partial或多维multicast/collective等movement alternatives；query-local relation-derived reuse analysis只推导spatial/temporal invariance与payload/coverage，不写IR attr或选择最大broadcast winner。assignment形成后重算lifetime并使相关probe/schedule失效。每个actual action携带participant、payload、destination version、join/completion和resource work，不返回局部route winner。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.I | `rotating-buffer-materialization` | `queued` | Q50.H | 将buffer count、slot rotation、release、async consumer和prologue/steady/epilogue执行结构物化为可验证Instr；lifetime从actual roots、SSA、effect、order与completion派生。候选域由并发wave、capacity和alias推导，不预设double/triple为永久上限，也不单独选择overlap winner。 | 08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.J | `event-resource-scheduling` | `queued` | Q50.I | 从dependency、effect、data-ready/completion、worker与compute/SPM/NoC/DDR resource构造query-local event calendar和actual Instr order；复用现有target facts的分层resource projection只作ordering/proposal，nominal bandwidth或解析`max`不限制并行域、不签发overlap。可解的boundary-faithful局部子问题可返回alternatives/bound/no-good，但selected choice仍经actual gate；不保留全图shadow schedule或局部winner。 | 08、11、13-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.K | `conditional-stage-pipeline` | `queued` | Q50.J | 先从temporal waves、movement data-ready、rotating slots和effect/completion构造有限pipeline event-structure transition；该transition使旧Q50.J calendar失效并重入schedule mechanism，由新assignment实际证明resource overlap。serialized、same-region coupled traversal和pipeline共享parent；actual prologue/steady/epilogue正负witness闭合，不以估算重叠或boolean attr代替。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51 | `physical-dataflow-unified-search` | `queued` | Q50.S、Q50.B–Q50.K | 在Q51.Core上闭合唯一选择owner：同一可回溯assignment state联合表达semantic root、spatial mapping、TileRegion/融合、temporal tile与wave-loop order、layout/version、movement、buffer和resource-event choices，派生analysis不进入identity。tiny DAG以独立reference enumerator证明域完整、production flat exhaustive runner证明candidate set/剪枝/winner；真实负载以baseline incumbent和显式预算返回accepted actual winner并按coverage标结果等级。必须以共享TileRegion、coupled traversal或明确retained SSA、tile-sized intermediate和无中间DDR round-trip证明有效融合；`search`与`none`共用Q50.0 exact gate。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q52 | `physical-dataflow-search-scalability` | `queued` | Q51；复用Q41计时/RSS/work ledger | 先在通用DAG、HF prefill/decode和Llama representative load上记录状态数、重复率、失败原因、clone/packing/cost时间与峰值内存，并测regular-mapping/reuse proposal命中率、estimate-vs-final误差及`best-found@k`/winner recall/regret；再按热点引入canonical memo、DP、constraint/no-good cache、dominance/branch-and-bound或局部solver。启发式、top-k/diverse candidate set、遗传/退火等trade-off只能在实际组合爆炸后启用，并以小图最优oracle、`none`质量和融合保全约束校准；永久丢状态时标`budgeted-feasible`。10分钟以上必须分析热点，允许继续到30分钟；不得用固定tile/fusion/buffer/candidate上限冒充优化。 | 06、14-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q53 | `physical-dataflow-production-readiness` | `queued` | Q52、Q44 source/oracle mechanics、Q47 current ABI closure、Q56达到`board-ready` | 冻结`search`策略并fresh生成通用mixed DAG、official HF prefill、functional KV-cache decode和Llama block的FP16/BF16 package、oracle、runner与no-card证据；逐case证明selected actual IR中的多op融合、中间值SPM驻留、无无意义DDR round-trip及数值一致后标`board-ready`。真实板端只串行执行current matched baseline/winner；Llama及至少一个prefill/decode代表获得可重复改善后才`done`。 | 02、06-16、18；`tasks/plans/physical-dataflow-synthesis.md` |

## Later / External Gates

这些任务不会因当前主线完成自动启动；外部条件满足后先更新状态和对应设计。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9.R | `profile-writing-overhead` | `later` | Q9、进入profiler writing优化排期 | raw evidence只有一个canonical owner，analysis只发布ID/dictionary引用的摘要，HTML不再重复内嵌全量analysis/evidence，raw按需压缩加载；dense fixture证明规模随事件数线性并通过size gate。板端output readback是独立runtime开销，不计入report writing体积。当前generator和测试仍保留宽对象、pretty JSON及全量内嵌等旧路径，本项尚未实现。 | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B、configured numeric corpus | 按capability row冻结board区分向量、held-out和numeric comparator；现有workload证据不能单独代签。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32、configured simulator/ISS | 原样执行verified package及all-and-only RISC-V ELF；schema升级必须先独立完成。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22、owner-approved vendor package或公开规范 | 建立可引用的CRT/packet/MMIO provenance；缺失不阻塞functional CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C、validated PMU/timing environment | 校准LT/AT；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32、明确的external control-plane consumer | 复用现有rewrite/conversion；Transform IR不保存physical-dataflow search state，也不替代06的唯一candidate owner。 | 01、05-08、10、16、18 |
| Q47 | `target-abi-retirement` | `board-ready` | 用户明确将Q46后续验证与ABI收口合并推进 | TargetProfile、旧single-Tile launch form、旧pointer-block entry ABI及多schema reader已经删除；current worker-aware TargetCall/CRT、Grid/Cluster runtime ABI、current package和typed completion收口。fresh current source→package→model/no-card及板端普通/DTE路径通过后才能`done`。 | 11、14-17；旧施工记录见`tasks/archive/target-abi-retirement.md` |
| Q57 | `resident-static-execution` | `later` | Q56达到`board-ready`且Q53按新package合同达到`board-ready` | 在不改变selected `CardExecutable`语义的前提下建立owning device、完整whole-card loaded executable、device buffer/view和显式submission/completion；code与immutable data可驻留并按exact digest/physical descriptor/device qualification共享lease，one-shot执行只保留为同一路径便利封装。当前provider先固定single context、`max_inflight=1`、无cancel；板端必须证明load-once/run-many、immutable H2D一次、device output→下一次input连续性、terminal lifetime及poison后无破坏性cleanup。应用mutable state继续由显式buffer传递；runtime-owned state、multi-inflight、persistent device loop、跨卡和serving scheduler均不在本项。 | 12、14-17；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q48 | `semantic-superoptimization` | `later` | Q53按current CardModule/Tile合同达到`board-ready`，且current compiler/runtime closure可消费final Instr/TargetCall；当前前置未满足 | 从actual structured MLIR生成并物化verifier-legal TensorProgram alternatives，以query-local solver证明外部可观察value、memory与effect等价；所有actual roots进入Q51定义、Q52优化后的同一physical-dataflow candidate set与Q50.0 exact gate。Q48不在Instr层建立第二selector，不预设shortlist或固定候选cap。 | 05-08、10-11、16-18；`tasks/plans/semantic-superoptimization.md` |
| Q38.W | `multi-worker-production-promotion` | `later` | current typed ABI、actual clone和host/model资格已闭合，且出现需要隔离等待域并可能受益的独立命令链 | 以configured-board matched correctness/performance证明非零worker相对worker0流水的明确收益后才允许normal production promotion；不得为使用worker1/2而拆分已能在worker0并行的流水。 | 08、11、14-17 |
| Q3.6 | `crt-writeback-scalar` | `later` | 明确Count predicate及wrapper/target/model evidence | 独立闭合typed instruction、effect/completion、ABI/CRT、model和必要package readback。 | 11、14-17 |

不在当前DAG中的model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、
WCRE/global registry、capability lease、跨model state migration、共享weight cache、segmented MoE及
70B/100GB stress，恢复时必须先建立编号设计和completion gate。

## Done Index

本表只保留完成边界和证据入口。实现过程、测试数量和性能样本以对应owner为准。

| Tracking ID | Semantic key | 状态 | 完成边界 | 证据 owner |
| --- | --- | --- | --- | --- |
| Q45 | `compiler-terminology-and-naming` | `done` | active ODS、C++、pass/pipeline、file、diagnostic、CMake、tests和current docs已按真实IR/target语义收敛；范围限定只在logical/target、representation或Card/Tile强类型确有对照时保留，并通过fresh build、定向单测、CTest与组织检查。 | 01、04、06-08、18-19；`tasks/plans/compiler-terminology-and-naming.md` |
| Q14 | `architecture-baseline` | `done` | 单卡纵向架构、事实优先级和历史计划边界已重基线。 | 01、14-16；`tasks/archive/12-architecture-evidence-reset.md` |
| Q0 | `target-correctness` | `done` | Target conversion、结构保持、physical legality和原子失败边界闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | 固定source/config/payload/CPU expected corpus及独立oracle。 | 02、16 |
| Q15 | `compiler-driver` | `done` | Source到verified card-partition program及原子发布闭合；runtime launch kind只保留kernel/model。 | 01-06、14-16 |
| Q16 | `tile-executables` | `done` | 完整Tile executable集合与move-only资源控制闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `linked-target-modules` | `done` | Single-lowering target module、device link和原子目录替换闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Typed manifest、package readback和no-card validation闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | Typed kernel/model、grid/cluster和Direct-DTE launch/runtime lifecycle闭合。 | 13-16；`tasks/archive/runtime-board.md` |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct-DTE binding、completion、target activation和package projection闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | Linear/residual MLP的source、CPU expected和纵向mechanics闭合；current package证据由Q53重建。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | tiny Llama source、CPU expected和纵向mechanics闭合；current package证据由Q53重建。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Numeric、bulk、SystemC和host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | Typed target format legality及map/reduce lowering闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | Multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-modules` | `done` | Move-only Tile target LLVM modules和single-lowering device link闭合。 | 14、16、17；历史记录见对应archive文档。 |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime verification和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | Same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct-DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | Source-backed formal、bulk和multi-Tile完整输出组合mechanics闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | Model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | Instruction、tile-region lowering和target numeric按稳定职责拆分。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | Candidate、target LLVM、numeric、frontend和driver聚合实现拆分。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | Reference/model、compiler/output/package和frontend bridge残余聚合拆分。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | Path-sensitive lifetime/packing、loop backedge和DDR issue-to-typed-completion lifetime闭合。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享fixed-capacity canonical packing、validator及限定fallback闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | Accepted-IR第二解释器和旧CLI退役，CPU expected与model/board differential保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | Structured tiling与resource-gate mechanics闭合；其旧执行域和独立candidate owner在current card cutover工作树中已删除，但Q50 parity闭合前不构成能力完成迁移的证据。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | Llama-2 7B单block TP16 source→package→SystemC/PyTorch differential闭合。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持语义不变，收口static movement和physical codec host开销。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | ProgramTensor边界统计、多seed 7B characterization及source/model gate闭合。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | MLIR-native IndexRelation与typed lowering foundation闭合；旧implementation-choice接口在current cutover工作树中已删除，current替代能力由Q50.S与Q50.A–Q50.K迁移并由Q51统一选择。 | `tasks/archive/mlir-native-implementation-relation-foundation.md` |
| Q32.R | `physical-relation-realization` | `done` | Relation、encoding、transfer和resident boundary realization闭合。 | `tasks/archive/physical-relation-realization.md` |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Production-shaped spill/resident actual-clone seam及late gates闭合。 | `tasks/archive/physical-dataflow-test-seam-vertical.md` |
| Q32.V | `typed-target-capability-vertical` | `done` | Mapped transfer、physical fill和oriented GEMM typed capability纵向闭合。 | `tasks/archive/typed-target-capability-vertical.md` |
| Q32.M | `physical-mechanism-choice-closure` | `done` | 历史任务闭合了recompute、LICM、numeric、residency、ready-order和communication的表达/验证实验；旧独立candidate API已删除，current mechanism由Q50迁移。Q48只生成actual TensorProgram alternatives，physical-dataflow choice只能由Q51唯一owner接入。 | `tasks/archive/physical-mechanism-choice-closure.md` |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Bounded actual-clone组合、card Pareto和target-owned选择闭合。 | `tasks/archive/bounded-joint-physical-dataflow-selection.md` |
| Q32.G | `physical-dataflow-production-cutover` | `done` | `wafer-compile`成为唯一production decision owner，旧旁路退役。 | `tasks/archive/physical-dataflow-production-cutover.md` |
| Q32 | `physical-dataflow-synthesis` | `done` | MLIR-native actual-clone、implementation/relation/layout/storage/order/communication机制与exact-gate基础闭合；旧独立candidate set和per-task物化在current cutover工作树中已删除，current迁移边界与Q51统一owner只由06定义。 | `tasks/archive/physical-dataflow-synthesis-completion-audit.md` |
| Q32.C | `candidate-search-throughput` | `done` | Candidate search有界并发、parse复用和late-gate开销收口。 | `tasks/archive/whole-variant-search-throughput.md` |
| Q32.N | `numeric-algebraic-extension` | `done` | Supported floating algebraic candidates、typed tolerance和整数负例闭合。 | `tasks/archive/numeric-algebraic-extension.md` |
| Q36 | `topology-aware-collective-lowering` | `done` | 历史实验曾闭合Typed topology到Ring/ordered Tree mechanics；其Tile collective IR、late lowering和topology helper现已删除，不属于current architecture。 | `tasks/archive/topology-aware-collective-lowering.md` |
| Q35 | `k-sharded-gemm-board-vertical` | `done` | 16-rank K-sharded f16 GEMM、local compute、AllReduce和board exact纵向闭合。 | `tasks/archive/k-sharded-gemm-board-vertical.md` |
| Q37 | `tx81-compiler-hardware-calibration` | `done` | Compiler-sensitive硬件行为以supported/observed/unknown/excluded和保守策略闭合。 | `docs/tx81-compiler-hardware-calibration.md` |
| Q1 | `crt-surface-audit` | `done` | Compiler-emitted CRT symbol/prototype surface闭合。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | Production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence完成分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | Conformance工具不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical dependency root和CMake cache workflow闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理已归档。 | `tasks/README.md`、`tasks/archive/` |

## 导航

- Q49/P、Q50.0/Q50.S/Q50.A–Q50.K及Q51–Q53的baseline隔离、机制迁移、统一搜索、scalability与production readiness共用实施计划：
  `tasks/plans/physical-dataflow-synthesis.md`。
- Q54 MLIR工程化整改计划：`tasks/plans/mlir-engineering-remediation.md`；稳定工程合同由19拥有。
- Q56 package数据闭合与Q57设备驻留执行共用实施计划：
  `tasks/plans/executable-package-and-resident-runtime.md`；稳定package/runtime字段语义由15拥有。
- Q48语义驱动superoptimizer计划：`tasks/plans/semantic-superoptimization.md`。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
