# Wafer Compiler Task Queue

更新时间：2026-08-19

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
Q55 + Q56 board-ready + current TargetCall/model
  -> Q62 target numeric contract reconstruction              [queued]
Q54 + current Instr/NCC target ABI
  -> Q63 NCC completion contract layering                     [queued]
Q50.0 + Q54 -> Q50.A exact placement-demand boundary           [done]
Q59 + Q50.A + Q45
  -> Q49.P deterministic baseline functional closure          [done]
Q49.P + Q50.0 -> Q51.Core search control kernel                [done]
Q51.Core -> Q50.S structured semantic alternatives            [doing]
Q51.Core + Q50.A -> Q50.B spatial partition and placement    [queued]
Q50.B -> Q50.C maximal single-root TileRegion                [queued]
Q50.C -> Q50.D coupled traversal and region fusion           [queued]
Q50.D -> Q50.E complete temporal tiling                      [queued]
Q50.E -> Q50.F scoped feasibility analysis                    [queued]
Q50.F -> Q50.G layout and physical representation            [queued]
Q50.G -> Q50.H explicit data movement                        [queued]
Q50.H -> Q50.I rotating buffers                              [queued]
Q50.I + Q63 -> Q50.J event and resource scheduling           [queued]
Q50.J -> Q50.K conditional stage pipeline                    [queued]
Q50.S + Q50.B–Q50.K -> Q51 unified search correctness       [queued]
  -> Q52 workload-driven search scalability                  [queued]
Q51.Core + Q62 + Q63 -> Q64 source registration truth        [queued]
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
| Q50.0 | `actual-executable-compilation-boundary` | `done` | 已选择并物化的deterministic CardModule baseline | 唯一无策略CardModule→CardExecutable边界已闭合：输入已选择并物化的CardModule，依次完成Tile module splitting、TileRegion-to-Instr、fresh completion、fixed-capacity SPM/DDR及transport/resource/runtime-launch verification，并以typed result区分accepted、proven exact rejection与indeterminate failure。target ABI/lowering/translation只在下游真正保留的target output执行。baseline与search共用该入口；lowering不枚举、retile、spill或rebuffer，未知allocator/internal failure不会形成candidate no-good。直接单测覆盖可重复exact SPM rejection与indeterminate invocation，source-to-package baseline/no-card gate通过。 | 01、06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q54 | `mlir-infrastructure-conformance` | `done` | Q50.0 | Checkpoint J已按普通pass、framework transaction、actual alternative、search evaluation和output fan-out逐类复核active root clone及dormant旧能力。本轮删除普通pass无合同root snapshot、required-NCC重复validation、production selected-buffer nested clone和CardExecutable discarded target gate；最终16 Tile target output每variant只clone/lower/translate一次并按LaunchSlot bounded并发归并。旧search/rank与未注册mechanism已绑定Q51.Core/Q50 extract-then-delete门禁，不冒充current production。fresh三树构建、非搜索unit、target/profile unit、lit、source-to-package/no-card、organization/link和显式work计数通过；Q54本身不替代后续Q49.P的baseline功能与模型级物化门禁，Q63独立承接NCC target/MLIR completion分层。 | 18-19；`tasks/plans/mlir-engineering-remediation.md` |
| Q55 | `current-interface-consolidation` | `done` | Q9 profile合同、Q45命名规则、current target/package/runtime接口 | Wafer-owned frontend metadata、package、target ABI/CRT、profiler、Direct-DTE status、qualification evidence、dependency record与工具fixture均收敛为一种无编号current合同，旧reader/CLI/carrier/alias与版本分支退出active路径。Board calibration逐项迁移source、shape/dtype、oracle、guard、status、timeout、cleanup和profile要求：22个raw probe加complete-Tile add普通/profile共24个current-interface no-card CTest通过，10个Q55-owned真实Board入口进入同一串行runner但本轮未上板；full-4096 K-tiled与M-tiled profile由current global-source runner和catalog/inventory保留，并因global lowering/SPM门禁明确阻塞，不冒充board-ready。fresh configure/build、target unit 140/140、core lit 216/216、current-interface 24/24、Board host 32/32、Q55 tool/runtime lit 5/5、dependency/source/IR/diff检查通过。 | 02、11、14-17、20；`tasks/plans/interface-version-consolidation.md` |
| Q58 | `program-data-ownership` | `done` | Q50.0、Q54、Q55 | 2026-08-16二次review缺口已闭合：handoff在稳定output parent下RAII拥有唯一目录，source持有move-safe只读handle并按handle→file→directory顺序清理，public compile返回前删除staging不再使CardExecutable悬空；establishment从同一source descriptor复制，并从owned descriptor重新验证header/exact extent和whole-file digest；最低层所有read强制≤1MiB且actual `file_opens/read_windows/read_bytes/maximum_read_window_bytes`进入共享账本；最后一次tensor verification后真实partition已adopt，全部未采用candidate在Card边界销毁。`ProgramDataTest` 15/15（含32KiB mmap阈值、move/staging lifetime、candidate文件清除、2.8MB range三window零新增open、16 Tile shared view），受影响filtered unit 52/52；小型FP16 parameter source→package/no-card 1/1，三档0.5/2/8MiB source→package/no-card 1/1，source/file opens恒定2/8、最大read window≤1MiB，payload增长16倍时peak RSS保持55,392–56,332KiB。feature-on returned-executable target-model参数回归已注册，但当前managed numeric-model record缺失而unsupported，未计入完成证明。证据与账本见计划checkpoint 2-6和4.2。 | 02、14、16、18；`tasks/plans/program-data-and-whole-program-scale.md` |
| Q56 | `executable-package-data-closure` | `board-ready` | Q58 | 二次review缺口已闭合：physical-order bounded codec覆盖多outer-row Cx/NCx与bitpacked BOOL；不同dtype的target-ready bytes已通过确定value conversion产生且禁止bit reinterpret，但当时借formal model route实现的反向依赖不构成current长期合同，由Q62改为显式TargetTensor materialization action；strict verifier从shared codec重算descriptor并拒绝任意尾随字节；package closure包含目录拓扑；materialization按TargetTensor typed reader计数；Compiler/Package的Runtime header反向依赖已由source-organization gate封闭。专用FP16 parameter-add case、oracle和current board runner位于`test/Board/wafer_board_single_op_add_test.py`，并以`wafer-board-program-data-add`注册；同语义source→package→no-card、定向unit/lit、public link、SystemC/model和板端配置构建fresh通过。真实板测尚未执行，因此保持`board-ready`而非`done`。 | 14-18、20；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q59 | `compiler-entry-transaction-closure` | `done` | Q56达到`board-ready` | 2026-08-16 follow-up review缺口已闭合：package层只定义一个compiler/runtime共享的move-only `runtime::ExecutablePackage`；shared strict binder在publication前以read-backed immutable snapshots拥有canonical manifest、manifest顺序modules和program data，核对exact size/digest并关闭全部descriptor，同inode改写、删除或替换path不改变owner内容，board runtime不再按root reopen。profile transaction同样绑定exact activation/plan/site-map和两份capture package，runtime profile identity只对owned manifest bytes计算digest；唯一outer transaction在所有fallible验证完成后执行一次no-replace rename，committed public type只在rename成功后构造。全部test-only failure经public typed facade返回`CompilationFailure`；commit测试直接篡改staged program data/profile plan并证明package-commit分类、输出不可见和无staging残留。CLI current cutover为`--output-dir`，旧`--output-package-dir`无alias；profile输出仍是共同delivery root。fresh验证：受影响unit 120/120；request/internal-options/atomicity/commit-transaction/install-relocate lit 5/5；complete-Tile profile contract 1/1与fresh ordinary+profile source→package→no-card 1/1；public link smoke 3/3；main、board-runtime、SystemC/model三棵构建树及source/deps/IR/diff检查通过。既有32x32 dot被pinned helper拒绝与search长耗时仍分别归外部helper/Q52。 | 01、15、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q62 | `target-numeric-contract-reconstruction` | `queued` | Q55、Q56达到`board-ready`、current TargetCall/SystemC/model mechanics | 保留target operation、physical codec、formal arithmetic、managed dependency验证和bulk qualification的真实能力，但拆掉Q22.N遗留的全局`NumericSemantics` umbrella。target command字段回到TargetOperation/TargetCall，physical tensor成为无model identity的descriptor，TargetTensor materialization显式携带转换；parser后的logical/target dtype原位切为closed typed value，字符串只留外部format边界；`NumericDependencyConformance`迁出always-built target execution public API，`WaferTargetModelCore -> WaferCompiler`反向link删除。formal/model按family直接执行或typed拒绝，bulk evidence只绑定concrete problem/payload/backend/environment；旧profile/pattern/resolver/digest/compat与专属registry测试删除。验证只跑fresh direct target/codec/package/formal/bulk/model/no-card、optional dependency和组织/link检查，不运行旧numeric registry、Q49.P/Q51长搜索或历史板端路径。 | 01、11、14、16-18；`tasks/plans/target-numeric-contract-reconstruction.md` |
| Q63 | `ncc-completion-contract-layering` | `queued` | Q54、current Instr/NCC target ABI | 将pure target issue/join/synchronous-writeback completion protocol、MLIR op interface/external model与query-local pending-worker analysis拆成三个单向owner；删除IR public header到TX81 ABI的include、`NCCSynchronizationContract` free `TypeSwitch`特殊case及runtime/model对WaferIR/Compiler的依赖。Lifetime、ScheduleCost、TargetScheduling、TileRegion lowering等consumer同批迁移；Q50.J只消费typed hard completion/resource facts，不恢复capability/profitability registry。 | 11、13、17-19；`tasks/plans/ncc-synchronization-contract-layering.md` |
| Q49.P | `deterministic-baseline-functional-closure` | `done` | Q54；Q59；Q50.A；Q50.0、Q45 | `none`只保留一个live canonical coordinate，不枚举placement候选或调用recursive CSP/search；temporal fallback是确定性贪心，reduction temporal tiling已闭合，reduction spatial partition/partial merge仍归Q50.B，完整temporal搜索域归Q50.E。current materializer按consumer operand exact demand在structured producer处截断，先证明local/peer/DDR coverage再一次性构造final single-root TileRegion；旧root/function SPM probe、完整DAG closure、post-hoc support rebuild/replay、accepted重物化和默认统计均退出。Card→Tile直接move大型body，16 Tile bounded并发且只共享current IR可重算analysis。compiler实现、私有头、unit与fixture已按Baseline/Planning/Executable/Pipeline/Search/Target/Transport等职责归位，baseline测试不再寄居旧search suite。fresh职责化unit 134/134、source organization、runner unit与构建通过；official FP16 LLaMA `optimization-none`已从fresh source完成16-Tile package、payload和no-card，CTest 1/1在181.70秒通过。显式timing发现的约22,968次TileRegion→Instr重复work转交Q52，禁止以actual IR cache/replay处理。 | 06、14、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.A | `exact-placement-demand-boundary` | `done` | Q50.0、Q54 | policy-free trial、typed four-state verdict、nested IR snapshot invalidation、structural relation limits与typed resource exhaustion、all-and-only execution/ownership coverage、stable per-Tile payload、partial-reduction merge obligation和multi-result production carrier均已闭合。baseline support carrier直接消费同一per-destination exact demand/ownership intersections，empty destination不再被伪造为矩形或carrier失败。reduction、broadcast、affine window、strided view和multi-piece overwrite五类正常TensorProgram均经canonical baseline carrier完成一次完整Q50.0 CardModule compile并产出accepted CardExecutable；fresh relation/query/carrier回归62/62、五类production gate 5/5、Q50.0 baseline/稳定性/reduction 3/3、source-to-package 1/1和主构建通过。follow-up确认结构limit不能单独保证generic Presburger equality的wall-time；这不改变本项logical proof/outcome，supported rectangle热路径的closed-form witness与single-coordinate work closure归Q49.P P6。约163秒的旧placement枚举不是本logical boundary门禁：baseline调用由Q49.P删除，旧search测试随Q51.Core清理，Q50.B只以新惰性domain/tiny oracle重新建立需要的轴。 | 05-07、09-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51.Core | `physical-dataflow-search-kernel` | `done` | Q49.P；复用Q50.0 | new-search control kernel已闭合typed evaluation、deterministic frontier/dedup、显式budget、未决state保留及stable incumbent替换；public `search`只经过新入口，当前空production mechanism直接move并返回Q49.P accepted baseline。旧单体search、字符串failure gate、feedback/beam/accepted cohort、shadow schedule、winner重物化、默认statistics、rank/coordinated与旧placement/schedule source/test island均在独有logical-demand、baseline placement closure和negative witness迁入current owner后删除；14处旧长链CTest、paired optimization driver/catalog、source-marker gate及旧stderr断言同步退出。轻量FP16 public `search` source→16-Tile package→no-card、Core state-graph unit、全量unit、host catalog和source organization验证通过；没有运行重型LLaMA search。 | 06、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.S | `structured-semantic-alternatives` | `doing` | Q51.Core | 从typed SSA证明可用算法族并在isolated clone中物化真实TensorProgram alternative；普通DAG、online recurrence和partition/merge只是验证witness。算法参数改变图时必须先成为actual structured DAG，再进入同一Q51 root candidate set；不提供public独立selector/pass，不按名字、shape或参数位置恢复语义。先核对并迁移现有Attention/structured alternative中的独有proof与actual-root能力，再删除旧local selector。 | 05-06、08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.B | `spatial-partition-and-placement` | `queued` | Q51.Core、Q50.A | current实现只把parallel iterator当spatial axis，reduction spatial factor保持1且没有partial-result merge，因此reduction轴的spatial切分尚未实现并明确由本项闭合。从structured iterator语义生成all-iterator多轴factor vector、非整除remainder、parallel/reduction partition与显式merge，以及合法非最大、非矩形、非对称/非连通physical placement的完整惰性域；ordered-factorized regular mapping与compact/all-16仅为seed，不能固定participant，只有改变实际partition/embedding的axis/factor nesting才形成不同choice，等价生成路径canonicalize。不同op/branch/wave可选择不同Tile集合，consumer placement直接查询Q50.A。mechanism只生成typed transition/actual CardModule，不选局部winner；独立tiny reference enumerator逐parent证明chain、branch、diamond、reduction与partial redistribution的spatial domain，跨轴flat exhaustive runner归Q51 closure。 | 05-08、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.C | `maximal-single-root-tile-region` | `queued` | Q50.B | 对已选spatial shard物化覆盖all-and-only local work的single-root TileRegion，使baseline和search复用同一policy-free region materializer；支持multi-result、DPS init operand、reduction与effect boundary，但显式init producer若映回另一structured DAG node仍留在region boundary外；不选择temporal tile、layout或buffer。 | 05-08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.D | `coupled-traversal-region-fusion` | `queued` | Q50.C | 从SSA、iterator/index relation和effect构造合法region boundary与consumer-driven coupled traversal；actual fused candidate必须处于共同TileRegion并避免中间值无意义DDR round-trip。maximal closure与合法cut都进入共同candidate set，不能用单边`fused`标记或group字段代替实际融合。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.E | `complete-temporal-tiling` | `queued` | Q50.D | current materializer与baseline complete gate已经携带reduction temporal tiles，fresh multi-reduction-axis materialization和`none` reduction-demand gate均通过；缺口不是“reduction轴不能temporal切分”，而是尚未闭合所有iterator breakpoint、wave-loop nesting/order及其完整可搜索域。每个op在每个spatial shard上生成覆盖全部iterator的完整temporal tile向量，以及Q50.D已选traversal内部语义可区分的有限wave-loop nesting/order；从完整local extent惰性枚举所有改变work/legality/reuse/lifetime的breakpoint与order，并与Q49.P canonical functional fallback复用同一policy-free legal-breakpoint/workset机制。容量失败只对包含全部因果坐标的branch产生新breakpoint，不能把baseline第一个fit、`tile/2`、默认loop order、固定seed或最大可放下tile固化成全局限制。 | 05-10、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.F | `scoped-feasibility-analysis` | `queued` | Q50.E | 只从immutable IR、target facts和显式partial assignment重算must-coexist lower bound、ordinary estimate、`deferred(required coordinates)`与causal taxonomy，不clone或lower IR，也不运行packer。只有boundary-faithful的proven lower bound可早拒绝；缺layout/movement/buffer/order等坐标必须deferred，资源耗尽/内部失败返回indeterminate。complete assignment只构造一个actual CardModule并由Q50.0完成唯一lowering/lifetime/packing结论；accepted executable直接进入incumbent，exact rejection销毁owner并回共同parent。memo/no-good key覆盖全部依赖轴且不缓存materialized IR。 | 06-11、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.G | `layout-and-physical-representation` | `queued` | Q50.F | 为scheduled value生成唯一typed MemLayout/encoding事实、可共享的physical version和必要conversion完整合法域，并实际物化到CardModule/TileRegion；exact relation证明physical payload、fanout共享与destination coverage，layout变化使依赖的movement/lifetime/feasibility analysis失效。旧layout selector、late default assignment和旁路payload owner退出，lowering只消费selected representation。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.H | `explicit-data-movement` | `queued` | Q50.G | 由exact logical demand、representation、placement/topology及retention/release/boundary obligations生成同TileRegion retained reuse、跨region显式DDR、recompute、unicast/partial或多维multicast/collective等movement alternatives；query-local relation-derived reuse analysis只推导spatial/temporal invariance与all-and-only payload/coverage，不写IR attr或选择最大broadcast winner。assignment形成后重算lifetime并使相关probe/schedule失效。每个actual action携带participant、source/destination physical version、exact payload、join/completion和resource work，并由current no-card/board case核对真正消除或改变的movement，不返回局部route winner。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.I | `rotating-buffer-materialization` | `queued` | Q50.H | 将typed buffer count、slot rotation、release、async consumer和prologue/steady/epilogue执行结构物化为可验证Instr；Direct-DTE issue/wait与每slot completion必须引用实际producer/consumer，lifetime从actual roots、SSA、effect、order与completion派生。候选域由并发wave、capacity和alias推导，不预设double/triple为永久上限，也不单独选择overlap winner。 | 08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.J | `event-resource-scheduling` | `queued` | Q50.I、Q63 | 从dependency、effect、data-ready/completion、worker与compute/SPM/NoC/DDR resource构造query-local event calendar和actual Instr order；Direct-DTE issue/wait、compute及slot reuse必须形成可验证先后/overlap witness。hard completion/resource facts来自Q63 pure target protocol与MLIR adapter；删除稀疏静态`TargetSchedulingCapabilityRegistry`及其pair/group legality/profitability row，profitability只归Q52 estimate。nominal bandwidth或解析`max`不限制并行域；局部solver只返回boundary-faithful alternatives/bound/no-good，selected choice仍经actual gate，不保留全图shadow schedule或局部winner。 | 08、11、13-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.K | `conditional-stage-pipeline` | `queued` | Q50.J | 先从temporal waves、movement data-ready、rotating slots和effect/completion构造有限pipeline event-structure transition；该transition使旧Q50.J calendar失效并重入schedule mechanism，由新assignment以actual issue/wait、compute与completion证明resource overlap。serialized、same-region coupled traversal和pipeline共享parent；actual prologue/steady/epilogue、至少两个可重叠stage及无重叠负例witness闭合，不以估算重叠或boolean attr代替。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51 | `physical-dataflow-unified-search` | `queued` | Q50.S、Q50.B–Q50.K | 在new Q51.Core上闭合完整typed domain：同一可回溯assignment state联合表达semantic root、spatial mapping、TileRegion/融合、temporal tile与wave-loop order、layout/version、movement和buffer/resource-event choices，派生analysis不进入identity。tiny DAG以逐轴独立reference enumerator证明域完整、production flat exhaustive runner证明new candidate set/剪枝/winner；真实负载以Q49.P baseline incumbent和显式预算返回accepted actual winner并按coverage标结果等级。new source-to-package链必须证明public `search`始终只进入新链且旧search实现已随Core和各Q50变更清理；public routing已由Core完成，Q51不对照旧winner或耗时。selected IR必须以共享TileRegion、coupled traversal或明确retained SSA、tile-sized intermediate和无中间DDR round-trip证明有效融合；`search`与`none`只共享Q50.0 exact gate和独立底层事实。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q52 | `physical-dataflow-search-scalability` | `queued` | Q51；复用现有显式计时/RSS/work计数 | 先拆`TargetScheduleCostPolicy`/`WaferTargetPolicy`跨owner aggregate：exact Instr resource metrics、hard memory facts、package/profile reference rates和search estimate priors分别归唯一owner；不保留`TileSearchEffort::{Quick,Default,Deep}`或固定candidate/beam cap。随后在通用DAG、HF prefill/decode和Llama representative load上显式记录state/work/clone/packing/cost/wall/RSS、estimate误差和best-found/recall/regret，再按真实热点引入memo、DP、no-good、dominance、branch-and-bound或局部solver。启发式/top-k/LNS等永久丢状态策略只在profile后启用并准确标`budgeted-feasible`；这些统计和长时间运行只属于本任务的显式profile批次，不进入普通编译或每次功能改动回归，也不得用旧search耗时或winner作基准。 | 06、14-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q64 | `source-registration-truth-closure` | `queued` | Q51.Core、Q62、Q63 | 在上述任务删除各自旧source/test island后，从repo filesystem与实际CMake target、unit/lit/CTest graph建立全仓registration mirror；每个translation unit/test必须active注册或由current编号任务明确唯一dormant能力与删除门禁。checker覆盖不再局限少数目录，未注册文件、public header无link symbol、test读取旧source marker或文档声称未注册test为gate均fail closed；不为通过检查机械激活旧实现或创建stub。 | 18；`tasks/plans/source-registration-truth-closure.md` |
| Q60 | `frontend-production-entry` | `queued` | Q52、current PyTorch/XLA capture/export资产、Q59 | 建立产品Python API `wafer.frontend.export_pytorch_program`、installed `wafer-verify-program`与pre-exported StableHLO入口：从现有资产吸收framework capture/export、graph-break/fallback detection、metadata/payload emission与canonical-equivalence机制；adapter只负这些产品职责和current source program writing，不携corpus dispatch、固定case/seed、CPU oracle/comparator、target/search选项或模型名分支。外部IR以StableHLO portable serialization进入同一compiler-owned snapshot/verifier，text MLIR只作diagnostic，不保留双reader。PyTorch/XLA adapter和pre-exported input必须进入同一`CompilationRequest`与source→package路径；graph break、fallback、unsupported dynamic和metadata mismatch在compile前fail closed。 | 01-02、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q53 | `physical-dataflow-production-readiness` | `queued` | Q60、Q55 current-interface closure、Q56达到`board-ready` | 冻结`search`策略，并由case owner以固定seed、case-owned dtype、framework eager oracle及原dtype/shape comparator，从Q60产品入口fresh生成通用mixed DAG、official HF prefill、functional KV-cache decode和Llama block的FP16/BF16 package、runner与no-card证据；逐case证明selected actual IR中的多op融合、中间值SPM驻留、layout/physical payload正确、movement实际消除或改变、Direct-DTE/compute overlap witness、无无意义DDR round-trip及数值一致后标`board-ready`。真实板端只串行执行同一current source/config/payload的matched baseline/winner；Llama及至少一个prefill/decode代表获得可重复改善后才`done`。 | 02、06-16、18；`tasks/plans/physical-dataflow-synthesis.md` |

## Later / External Gates

这些任务不会因当前主线完成自动启动；外部条件满足后先更新状态和对应设计。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9.R | `profile-writing-overhead` | `later` | Q9、进入profiler writing优化排期 | raw evidence只有一个canonical owner，analysis只发布ID/dictionary引用的摘要，HTML不再重复内嵌全量analysis/evidence，raw按需压缩加载；dense fixture证明规模随事件数线性并通过size gate。板端output readback是独立runtime开销，不计入report writing体积。当前generator和测试仍保留宽对象、pretty JSON及全量内嵌等旧路径，本项尚未实现。 | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B、configured numeric corpus | 按具体target operation、dtype/layout/parameter domain冻结board区分向量、held-out payload和comparator，并绑定current target/environment；不通过通用capability/profile registry签发。现有workload证据不能单独代签。 | 16、17 |
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
| Q22.N | `target-numeric-foundation` | `done` | 历史任务闭合了multi-dtype codec、formal arithmetic与受管oracle mechanics；当时建立的aggregate profile/registry不是current合同，现行重建入口为Q62。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-modules` | `done` | Move-only Tile target LLVM modules和single-lowering device link闭合。 | 14、16、17；历史记录见对应archive文档。 |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime verification和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | Same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct-DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | Source-backed formal、bulk和multi-Tile完整输出组合mechanics闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | 历史untimed functional-event/numeric model mechanics完成；current合同不保留model-only capability profile，按17号设计与Q62原位演进。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
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
- Q62 Target数值合同重建计划：`tasks/plans/target-numeric-contract-reconstruction.md`；稳定target operation、
  materialization、model/evidence与源码边界由01、11、14、16-18拥有。
- Q63 NCC completion合同分层计划：`tasks/plans/ncc-synchronization-contract-layering.md`；稳定Instr、communication、
  model和MLIR adapter边界由11、13、17-19拥有。
- Q64 source registration truth闭合计划：`tasks/plans/source-registration-truth-closure.md`；稳定source/library规则由18拥有。
- Q48语义驱动superoptimizer计划：`tasks/plans/semantic-superoptimization.md`。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
