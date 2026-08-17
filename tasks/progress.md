# Wafer Compiler Task Queue

更新时间：2026-08-17

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。历史状态变化由Git保留。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级。
- 全局至多一个`doing`；`next`表示前置已满足但尚未开始，`queued`表示已经进入当前主线但仍等待直接前置闭合，
  `later`表示不进入当前主线。
- `done`只表示对应设计文档的completion gate已经满足；详细证据只链接owner，不复制到本文件。
- current职责被后继吸收时，旧任务直接移出当前调度和状态表；可复用的source、oracle、mechanics与验证要求写入
  current owner，历史只由Git和`tasks/archive/`保留，不新增旧任务过渡状态或旧任务索引。
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
accepted deterministic CardModule baseline
  -> Q50.0 actual executable compilation boundary             [done]
Q50.0 -> Q54 MLIR infrastructure conformance                  [done]
Q45 compiler terminology and naming                           [done]
Q9 + Q45 + current target/package/runtime interfaces
  -> Q55 current-interface consolidation                      [done]
Q50.0 + Q54 + Q55 -> Q58 program data ownership                [done]
Q58 -> Q56 executable-package data closure               [board-ready]
Q56 board-ready -> Q59 compiler entry transaction closure     [done]
Q50.0 + Q54 -> Q50.A exact placement-demand boundary           [done]
Q59 + Q50.A + Q45
  -> Q49.P deterministic baseline functional closure           [doing]
Q50.0 + Q50.A + Q49.P -> Q51.Core common search kernel        [queued]
Q51.Core -> Q50.S structured semantic alternatives            [queued]
Q51.Core + Q50.A -> Q50.B spatial partition and placement    [queued]
Q50.B -> Q50.C maximal single-root TileRegion                [queued]
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
Q52 + current PyTorch/XLA capture/oracle mechanics + Q59
  -> Q60 frontend production entry                            [queued]
Q60 + Q55 current-interface closure + Q56 board-ready
  -> Q53 current production board readiness                  [queued]
Q53 board-ready + Q56 board-ready
  -> Q57 resident static execution                             [later]
Q53 board-ready + Q58 + Q60
  -> Q61 whole-program scale readiness                         [later]
Q53 board-ready -> Q48 semantic superoptimization             [later]
```

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前工作与完成门禁 | 设计 / 计划 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `production-output-profiler` | `done` | Q32、Q6.B、configured board | 未插桩Primary TX stream launch-to-completion设备包络、分离的host diagnostics、Trace来源的五类NCC per-tile engine active ns/work-volume摘要、独立Direct-DTE cycles/raw activity、Count/Trace、逐次保留真实动态调用与rdcycle的16-tile timeline、exclusive语义成本与非加和Trace-only成本、final Instr静态work与硬件峰值下界对照、exact output、单一production output instrumentation、三文件专业UI和profile全树`0777`均闭合；不构造card-wide纯engine elapsed，静态cost不回灌ranking。Ranking feedback保留为后续独立门禁。 | 06、14-16；`tasks/archive/board-profiler.md` |
| Q38 | `multi-engine-software-pipelining` | `done` | Q32、Q6.B、Q37 | 历史任务已经证明typed multi-buffer、prologue/steady/epilogue、Direct-DTE issue/wait和exact range hazard可表达、可验证；旧fixed-slot candidate、worker selector及其专用测试已删除。Q50.I/Q50.J迁移机制后，Q51必须把buffering/worker作为同一physical-dataflow state的联合动作直接物化到selected Tile IR，不得恢复独立owner。 | 06、08-17；历史计划已归档 |
| Q39 | `noc-resident-tile-dataflow` | `done` | Q38历史表达/验证能力闭合；板端资格作为独立external gate | physical peer materialization、resident/spill mechanics和板端exact资格已经闭合；旧Tile collective/late selector、静态profitability owner及其output合同已删除，movement mechanism由Q50.H保全迁移，choice统一由Q51 physical-dataflow search拥有。 | 02-13、16-17；历史计划已归档 |
| Q43 | `compiler-collaboration-review-materials` | `done` | Q9、Q37-Q39完成证据 | 2026-08-03冻结的历史汇报材料已经闭合；它不作为current architecture合同。 | 01、08–16；`tasks/archive/vibe-compiler-collaboration-review.md` |
| Q42 | `test-load-reduction` | `done` | 无 | 默认lit/unit/CTest和owner integration均只判直接合同；历史catalog、批量测试矩阵、model-scale与重复package执行已退出默认入口。 | 16；`tasks/archive/test-gate-scope-reduction.md` |
| Q50.0 | `actual-executable-compilation-boundary` | `done` | 已选择并物化的deterministic CardModule baseline | 唯一无策略CardModule→CardExecutable边界已闭合：输入已选择并物化的CardModule，依次完成Tile module splitting、TileRegion-to-Instr、fresh completion、fixed-capacity SPM/DDR、transport/resource/ABI verification，并以typed result区分accepted、proven exact rejection与indeterminate failure。baseline与search共用该入口；lowering不枚举、retile、spill或rebuffer，未知allocator/internal failure不会形成candidate no-good。直接单测覆盖可重复exact SPM rejection与indeterminate invocation，source-to-package baseline/no-card gate通过。 | 01、06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q54 | `mlir-infrastructure-conformance` | `done` | Q50.0 | 已按19号合同完成全active compiler source整改：semantic Location pointer payload、raw semantic attr、ordinal/print relation和synthetic local wrapper退出active路径；TileRegion/Func/Module按真实scope组织，16个atomic pass组成7条常驻named semantic pipeline与1条Shardy条件pipeline，production只由统一runner建PM；RegionBranch/effect/call、ODS、AnalysisManager、transactional pattern/conversion、typed host error和pure target/MLIR adapter分层闭合。fresh core/model build通过；Q54定向单测168/168、其余非搜索单测629/629、受影响card关系用例1/1、lit 1/1、feature-on依赖/模型/link 17/17、IR/source organization和diff检查通过。baseline/Q52长搜索未冒充本任务全量验证；Q49.P消费这些scope/pipeline seam继续闭合baseline deterministic functional fallback、search-policy隔离、single-root region、ancestor-scope probe、causal witness和整图物化，Q50.I继续物化真实共同wave/stage loop。 | 18-19；`tasks/plans/mlir-engineering-remediation.md` |
| Q55 | `current-interface-consolidation` | `done` | Q9 profile合同、Q45命名规则、current target/package/runtime接口 | Wafer-owned frontend metadata、package、target ABI/CRT、profiler、Direct-DTE status、qualification evidence、dependency record与工具fixture均收敛为一种无编号current合同，旧reader/CLI/carrier/alias与版本分支退出active路径。Board calibration逐项迁移source、shape/dtype、oracle、guard、status、timeout、cleanup和profile要求：22个raw probe加complete-Tile add普通/profile共24个current-interface no-card CTest通过，10个Q55-owned真实Board入口进入同一串行runner但本轮未上板；full-4096 K-tiled与M-tiled profile由current global-source runner和catalog/inventory保留，并因global lowering/SPM门禁明确阻塞，不冒充board-ready。fresh configure/build、target unit 140/140、core lit 216/216、current-interface 24/24、Board host 32/32、Q55 tool/runtime lit 5/5、dependency/source/IR/diff检查通过。 | 02、11、14-17、20；`tasks/plans/interface-version-consolidation.md` |
| Q58 | `program-data-ownership` | `done` | Q50.0、Q54、Q55 | 2026-08-16二次review缺口已闭合：handoff在稳定output parent下RAII拥有唯一目录，source持有move-safe只读handle并按handle→file→directory顺序清理，public compile返回前删除staging不再使CardExecutable悬空；establishment从同一source descriptor复制，并从owned descriptor重新验证header/exact extent和whole-file digest；最低层所有read强制≤1MiB且actual `file_opens/read_windows/read_bytes/maximum_read_window_bytes`进入共享账本；最后一次tensor verification后真实partition已adopt，全部未采用candidate在Card边界销毁。`ProgramDataTest` 15/15（含32KiB mmap阈值、move/staging lifetime、candidate文件清除、2.8MB range三window零新增open、16 Tile shared view），受影响filtered unit 52/52；小型FP16 parameter source→package/no-card 1/1，三档0.5/2/8MiB source→package/no-card 1/1，source/file opens恒定2/8、最大read window≤1MiB，payload增长16倍时peak RSS保持55,392–56,332KiB。feature-on returned-executable target-model参数回归已注册，但当前managed numeric-model record缺失而unsupported，未计入完成证明。证据与账本见计划checkpoint 2-6和4.2。 | 02、14、16、18；`tasks/plans/program-data-and-whole-program-scale.md` |
| Q56 | `executable-package-data-closure` | `board-ready` | Q58 | 二次review缺口已闭合：physical-order bounded codec覆盖多outer-row Cx/NCx与bitpacked BOOL；不同dtype走formal numeric conversion；strict verifier从shared codec重算descriptor并拒绝任意尾随字节；package closure包含目录拓扑；materialization按TargetTensor typed reader计数；Compiler/Package的Runtime header反向依赖已由source-organization gate封闭。专用FP16 parameter-add case、oracle和current board runner位于`test/Board/wafer_board_single_op_add_test.py`，并以`wafer-board-program-data-add`注册；同语义source→package→no-card、定向unit/lit、public link、SystemC/model和板端配置构建fresh通过。真实板测尚未执行，因此保持`board-ready`而非`done`。 | 14-18、20；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q59 | `compiler-entry-transaction-closure` | `done` | Q56达到`board-ready` | 2026-08-16 follow-up review缺口已闭合：package层只定义一个compiler/runtime共享的move-only `runtime::ExecutablePackage`；shared strict binder在publication前以read-backed immutable snapshots拥有canonical manifest、manifest顺序modules和program data，核对exact size/digest并关闭全部descriptor，同inode改写、删除或替换path不改变owner内容，board runtime不再按root reopen。profile transaction同样绑定exact activation/plan/site-map和两份capture package，runtime profile identity只对owned manifest bytes计算digest；唯一outer transaction在所有fallible验证完成后执行一次no-replace rename，committed public type只在rename成功后构造。全部test-only failure经public typed facade返回`CompilationFailure`；commit测试直接篡改staged program data/profile plan并证明package-commit分类、输出不可见和无staging残留。CLI current cutover为`--output-dir`，旧`--output-package-dir`无alias；profile输出仍是共同delivery root。fresh验证：受影响unit 120/120；request/internal-options/atomicity/commit-transaction/install-relocate lit 5/5；complete-Tile profile contract 1/1与fresh ordinary+profile source→package→no-card 1/1；public link smoke 3/3；main、board-runtime、SystemC/model三棵构建树及source/deps/IR/diff检查通过。既有32x32 dot被pinned helper拒绝与search长耗时仍分别归外部helper/Q52。 | 01、15、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q49.P | `deterministic-baseline-functional-closure` | `doing` | Q59；Q50.A；Q50.0、Q54、Q45 | current early-exit、真实TileRegion capacity query和一次完整CardExecutable compile只是起点。完成时`none`从未选placement/region/temporal/representation/buffer的正常TensorProgram自行完成确定性功能合法化：canonical controller按完整semantic tie-break遍历不被beam/cap/budget截断的有限baseline fallback，初始完整tile放不进SPM时重新推导operand/halo/result/temporary/movement/alignment/bank/lifetime并沿合法breakpoint缩小，直到第一个required scoped probe全部fit的canonical completion，再经一次完整Q50.0 gate成为accepted executable；selected assignment是该过程的输出而非前置输入。它仅复用policy-free placement-domain机制和Q50.A logical demand/coverage query，不依赖search state/candidate、search-oriented domain/ranking evaluator、proposal order/group materializer或candidate统计；每个baseline TileRegion恰有一个structured root和必要non-root support closure，同Tile多root形成多个顺序region，跨root shaped dependency显式DDR。Q50.A完成后已证明支持的reduction、broadcast、window/stride与multi-piece exact set由baseline canonical correctness carrier有限分解并走通，dense/descriptor carrier失败不得改判logical placement。region-local query需要call/function lifetime时提升到最近合法isolated ancestor；capacity witness把all-and-only conflict owner关联到当前single root的temporal assignment，unsupported witness命名typed lifetime/call relation和scope，均不得按type/shape、其它root或diagnostic字符串猜测。scoped query不造card-shaped/no-work-Tile wrapper；完整CardModule materialization与CardExecutable compilation各一次。resolved-assignment single-root apply与deterministic feasibility probe成为Q50.B/C/F复用的共同机制，不建立baseline-private路径；fresh overfull-to-fit、minimum-tile failure、prefill/decode/Llama证明IR/package digest、oracle/no-card、结构、调用闭包和work count；声明支持且baseline域有合法completion时必须产出accepted executable，accepted baseline由Q51直接复用为incumbent。已知既有失败（2026-08-16 HEAD即存在，非Q50.A引入）：`CardExecutableSynthesisTest.NoneJointlyRefinesExplicitProducerStageAndConsumerDemand`一次refinement后probe全fit但final spm-allocation失败，属本任务probe/final一致性gate case；`test/Tools/wafer-compile-card-baseline.test`的CROSS case（16-Tile transpose peer fragments）在baseline materialization验证（`validateSelectedTileLayouts`的每op×每relation递归`collectStorageRoots`，无memo）中数分钟级挂起，已证Q50.A前commit同样复现，属本任务baseline materialization域（详见`memory/bugs.md`）。 | 06、14、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.A | `exact-placement-demand-boundary` | `done` | Q50.0、Q54 | policy-free trial、typed four-state verdict、nested IR snapshot invalidation、bounded exact relation、all-and-only execution/ownership coverage、stable per-Tile payload、partial-reduction merge obligation和multi-result production carrier均已闭合。baseline support carrier直接消费同一per-destination exact demand/ownership intersections，empty destination不再被伪造为矩形或carrier失败。reduction、broadcast、affine window、strided view和multi-piece overwrite五类正常TensorProgram均经canonical baseline carrier完成一次完整Q50.0 CardModule compile并产出accepted CardExecutable；fresh relation/query/carrier回归62/62、五类production gate 5/5、Q50.0 baseline/稳定性/reduction 3/3、source-to-package 1/1和主构建通过。约163秒的既有placement枚举成本仍归Q52，不影响本logical boundary结论。 | 05-07、09-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51.Core | `physical-dataflow-search-kernel` | `queued` | Q50.0、Q50.A、Q49.P | 先直接接收Q49.P已准入baseline executable/actual cost作为incumbent，再建立唯一query-local assignment state、typed transition、global work ledger、scoped actual-probe seam和两层small oracle；不得用search carrier重新构造baseline。独立reference enumerator证明tiny合法域完整，production flat exhaustive runner证明candidate set/剪枝/winner。真实负载从第一天支持budgeted anytime；exact continuation与admissible bound仍完整时返回`feasible-with-bound`，永久丢弃合法completion后才是`budgeted-feasible`。core不得携带局部winner、派生lifetime/calendar、固定beam/cap、workload matcher或shadow schedule。 | 06-08、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.S | `structured-semantic-alternatives` | `queued` | Q51.Core | 从typed SSA证明可用算法族并在isolated clone中物化真实TensorProgram alternative；普通DAG、online recurrence和partition/merge只是验证witness。算法参数改变图时必须先成为actual structured DAG，再进入同一Q51 root candidate set；不提供public独立selector/pass，不按名字、shape或参数位置恢复语义。 | 05-06、08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.B | `spatial-partition-and-placement` | `queued` | Q51.Core、Q50.A | 从structured iterator语义生成all-iterator多轴factor vector、非整除remainder、parallel/reduction partition与显式merge，以及合法非最大、非矩形、非对称/非连通physical placement的完整惰性域；ordered-factorized regular mapping与compact/all-16仅为seed，不能固定participant，只有改变实际partition/embedding的axis/factor nesting才形成不同choice，等价生成路径canonicalize。不同op/branch/wave可选择不同Tile集合，consumer placement直接查询Q50.A。mechanism只生成typed transition/actual CardModule，不选局部winner；两层tiny oracle覆盖chain、branch、diamond、reduction与partial redistribution。 | 05-08、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.C | `maximal-single-root-tile-region` | `queued` | Q50.B | 对已选spatial shard物化覆盖all-and-only local work的single-root TileRegion，使baseline和search复用同一policy-free region materializer；支持multi-result、DPS init operand、reduction与effect boundary，但显式init producer若映回另一structured DAG node仍留在region boundary外；不选择temporal tile、layout或buffer。 | 05-08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.D | `coupled-traversal-region-fusion` | `queued` | Q50.C | 从SSA、iterator/index relation和effect构造合法region boundary与consumer-driven coupled traversal；actual fused candidate必须处于共同TileRegion并避免中间值无意义DDR round-trip。maximal closure与合法cut都进入共同candidate set，不能用单边`fused`标记或group字段代替实际融合。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.E | `complete-temporal-tiling` | `queued` | Q50.D | 每个op在每个spatial shard上生成覆盖全部iterator的完整temporal tile向量，以及Q50.D已选traversal内部语义可区分的有限wave-loop nesting/order；从完整local extent惰性枚举所有改变work/legality/reuse/lifetime的breakpoint与order，并与Q49.P canonical functional fallback复用同一policy-free legal-breakpoint/workset机制。容量失败只对包含全部因果坐标的branch产生新breakpoint，不能把baseline第一个fit、`tile/2`、默认loop order、固定seed或最大可放下tile固化成全局限制。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.F | `scoped-actual-region-probe` | `queued` | Q50.E | isolated clone只在全部causal coordinates关闭后物化受影响TileRegion并执行lowering、fresh lifetime/completion与fixed-capacity packing；symbolic footprint只有proven must-coexist lower bound超过capacity才能早拒绝，普通estimate只排序。缺layout/movement/buffer/order等坐标必须返回`deferred(required coordinates)`，资源耗尽/内部失败返回indeterminate，只有proven infeasible或确定unsupported才是typed exact rejection。probe不修改parent、不做late repair；memo/no-good key包含结论依赖的所有轴，本局部gate不提前签发跨轴可行性。 | 06-11、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.G | `layout-and-physical-representation` | `queued` | Q50.F | 为scheduled value生成唯一typed MemLayout/encoding事实、可共享的physical version和必要conversion完整合法域，并实际物化到CardModule/TileRegion；exact relation证明physical payload、fanout共享与destination coverage，layout变化使依赖的movement/lifetime/probe失效。旧layout selector、late default assignment和旁路payload owner退出，lowering只消费selected representation。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.H | `explicit-data-movement` | `queued` | Q50.G | 由exact logical demand、representation、placement/topology及retention/release/boundary obligations生成同TileRegion retained reuse、跨region显式DDR、recompute、unicast/partial或多维multicast/collective等movement alternatives；query-local relation-derived reuse analysis只推导spatial/temporal invariance与all-and-only payload/coverage，不写IR attr或选择最大broadcast winner。assignment形成后重算lifetime并使相关probe/schedule失效。每个actual action携带participant、source/destination physical version、exact payload、join/completion和resource work，并由current no-card/board case核对真正消除或改变的movement，不返回局部route winner。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.I | `rotating-buffer-materialization` | `queued` | Q50.H | 将typed buffer count、slot rotation、release、async consumer和prologue/steady/epilogue执行结构物化为可验证Instr；Direct-DTE issue/wait与每slot completion必须引用实际producer/consumer，lifetime从actual roots、SSA、effect、order与completion派生。候选域由并发wave、capacity和alias推导，不预设double/triple为永久上限，也不单独选择overlap winner。 | 08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.J | `event-resource-scheduling` | `queued` | Q50.I | 从dependency、effect、data-ready/completion、worker与compute/SPM/NoC/DDR resource构造query-local event calendar和actual Instr order；Direct-DTE issue/wait、compute及slot reuse必须形成可验证的先后或overlap witness。复用现有target facts的分层resource projection只作ordering/proposal，nominal bandwidth或解析`max`不限制并行域、不签发overlap。可解的boundary-faithful局部子问题可返回alternatives/bound/no-good，但selected choice仍经actual gate；不保留全图shadow schedule或局部winner。 | 08、11、13-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.K | `conditional-stage-pipeline` | `queued` | Q50.J | 先从temporal waves、movement data-ready、rotating slots和effect/completion构造有限pipeline event-structure transition；该transition使旧Q50.J calendar失效并重入schedule mechanism，由新assignment以actual issue/wait、compute与completion证明resource overlap。serialized、same-region coupled traversal和pipeline共享parent；actual prologue/steady/epilogue、至少两个可重叠stage及无重叠负例witness闭合，不以估算重叠或boolean attr代替。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51 | `physical-dataflow-unified-search` | `queued` | Q50.S、Q50.B–Q50.K | 在Q51.Core上闭合唯一选择owner：同一可回溯assignment state联合表达semantic root、spatial mapping、TileRegion/融合、temporal tile与wave-loop order、layout/version、movement、buffer和resource-event choices，派生analysis不进入identity。tiny DAG以独立reference enumerator证明域完整、production flat exhaustive runner证明candidate set/剪枝/winner；真实负载以baseline incumbent和显式预算返回accepted actual winner并按coverage标结果等级。必须以共享TileRegion、coupled traversal或明确retained SSA、tile-sized intermediate和无中间DDR round-trip证明有效融合；`search`与`none`共用Q50.0 exact gate。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q52 | `physical-dataflow-search-scalability` | `queued` | Q51；复用现有计时/RSS/work ledger | 先在通用DAG、HF prefill/decode和Llama representative load上按stage/pipeline/pass/analysis/candidate记录状态数、重复率、typed失败原因、clone/packing/cost时间、host wall time、峰值RSS和bounded executor work，并测regular-mapping/reuse proposal命中率、estimate-vs-final误差及`best-found@k`/winner recall/regret；计时与低扰动diagnostic不得改变candidate顺序或winner。再按热点引入canonical memo、DP、constraint/no-good cache、dominance/branch-and-bound或局部solver。启发式、top-k/diverse candidate set、遗传/退火等trade-off只能在实际组合爆炸后启用，并以小图最优oracle、`none`质量和融合保全约束校准；永久丢状态时标`budgeted-feasible`。10分钟以上必须分析热点，允许继续到30分钟；不得用固定tile/fusion/buffer/candidate上限冒充优化。 | 06、14-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q60 | `frontend-production-entry` | `queued` | Q52、current PyTorch/XLA capture/export资产、Q59 | 建立产品Python API `wafer.frontend.export_pytorch_program`、installed `wafer-verify-program`与pre-exported StableHLO入口：从现有资产吸收framework capture/export、graph-break/fallback detection、metadata/payload emission与canonical-equivalence机制；adapter只负这些产品职责和current source program writing，不携corpus dispatch、固定case/seed、CPU oracle/comparator、target/search选项或模型名分支。外部IR以StableHLO portable serialization进入同一compiler-owned snapshot/verifier，text MLIR只作diagnostic，不保留双reader。PyTorch/XLA adapter和pre-exported input必须进入同一`CompilationRequest`与source→package路径；graph break、fallback、unsupported dynamic和metadata mismatch在compile前fail closed。 | 01-02、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q53 | `physical-dataflow-production-readiness` | `queued` | Q60、Q55 current-interface closure、Q56达到`board-ready` | 冻结`search`策略，并由case owner以固定seed、case-owned dtype、framework eager oracle及原dtype/shape comparator，从Q60产品入口fresh生成通用mixed DAG、official HF prefill、functional KV-cache decode和Llama block的FP16/BF16 package、runner与no-card证据；逐case证明selected actual IR中的多op融合、中间值SPM驻留、layout/physical payload正确、movement实际消除或改变、Direct-DTE/compute overlap witness、无无意义DDR round-trip及数值一致后标`board-ready`。真实板端只串行执行同一current source/config/payload的matched baseline/winner；Llama及至少一个prefill/decode代表获得可重复改善后才`done`。 | 02、06-16、18；`tasks/plans/physical-dataflow-synthesis.md` |

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
| Q47 | `target-abi-retirement` | `later` | Q55、Q56、Q59 current interface/package/transaction闭合，且进入configured-board资格窗口 | 编译器侧ABI retirement已经闭合；启动时只用Q59之后的current package定向重签ordinary与Direct-DTE source→package→model/no-card，随后才标`board-ready`。真实板端只验证current ABI correctness、guard和lifecycle，通过后标`done`；不重复Q53性能A/B，也不执行任何历史package批次。 | 11、14-17、20 |
| Q57 | `resident-static-execution` | `later` | Q56达到`board-ready`且Q53按新package合同达到`board-ready` | 不改变selected `CardExecutable`、`TileEntryArgument`、program-data offset或Tile指令；把Q56的一次执行lifetime延长为`PreparedExecution -> submit* -> close`。PreparedExecution拥有loaded kernel modules、optional non-empty program data `BoardDeviceMemory`、invocation `BoardDeviceMemory`、pointer rows、transport/profile状态和provider failure state；submission/completion携带device generation。TX provider初始只single context、`max_inflight=1`、无cancel；板端证明load-once/run-many、non-empty program data只H2D一次、稳定地址、workspace初始化/覆盖、public terminal和poison quarantine。one-shot只保留同一路径便利封装。request scheduler、runtime-owned KV policy、multi-inflight、persistent device loop、跨卡和external execution-engine policy均不在本项。 | 12、14-17；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q61 | `whole-program-scale-readiness` | `later` | Q53达到`board-ready`、Q58、Q60 | 在主search、产品frontend和package data链闭合后，使用完整程序而非单算子/单block验证source→TensorProgram→search→`CardExecutable`→`ExecutablePackage`的规模行为；同时覆盖完整小模型语义、大型完整parameter inventory和至少一个完整大图，记录IR/op/candidate/work、wall、RSS、disk/IO、target/package bytes及失败分类。Llama-2 7B可作一个scale witness，但模型名、层数和LLM调度不进入合同；本项不以full-model board execution、resident runtime或serving integration作默认完成门禁。 | 01-02、06、14-18；`tasks/plans/program-data-and-whole-program-scale.md` |
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
| Q15 | `compiler-driver` | `done` | Source到verified card-partition program及原子发布闭合；历史实现保留kernel/model launch kind，current产品在Q56协调收口为kernel。 | 01-06、14-16 |
| Q16 | `tile-executables` | `done` | 完整Tile executable集合与move-only资源控制闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `linked-target-modules` | `done` | Single-lowering target module、device link和原子目录替换闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Typed manifest、package readback和no-card validation闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | 历史typed kernel/model、grid/cluster和Direct-DTE launch/runtime lifecycle证据闭合；current产品的model分支由Q56退役。 | 13-16；`tasks/archive/runtime-board.md` |
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

- Q49.P、Q50.0/Q50.S/Q50.A–Q50.K及Q51–Q53的baseline隔离、机制迁移、统一搜索、scalability与production readiness共用实施计划：
  `tasks/plans/physical-dataflow-synthesis.md`。
- Q54 MLIR工程化整改计划：`tasks/plans/mlir-engineering-remediation.md`；稳定工程合同由19拥有。
- Q56 package数据闭合与Q57设备常驻执行共用实施计划：
  `tasks/plans/executable-package-and-resident-runtime.md`；稳定package/runtime字段语义由15拥有。
- Q58 program data ownership与Q61 whole-program scale共用实施计划：
  `tasks/plans/program-data-and-whole-program-scale.md`；稳定source/data/target/verification语义由02、14-16拥有。
- Q59 compiler entry transaction与Q60 frontend production entry共用实施计划：
  `tasks/plans/compiler-entry-productization.md`；稳定driver/frontend/package/source组织语义由01-02、15、18-20拥有。
- Q48语义驱动superoptimizer计划：`tasks/plans/semantic-superoptimization.md`。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
