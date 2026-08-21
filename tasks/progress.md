# Wafer Compiler Task Queue

更新时间：2026-08-21

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
  -> Q62 target numeric contract reconstruction                [done]
Q54 + current Instr/NCC target ABI
  -> Q63 NCC completion contract layering                       [done]
Q50.0 + Q54 -> Q50.B spatial-assignment representation foundation
                                                               [doing]
Q50.B foundation -> Q50.A exact placement-demand boundary    [queued]
Q50.A -> Q49.P remaining canonical-plan foundations + Q50.F closed-plan core
                                                               [queued]
Q49.P/F core -> Q49.P deterministic baseline plan-only closure
                                                              [queued]
Q49.P -> Q50.S attention semantic IR foundation               [queued]
Q50.S semantic foundation + Q50.A + Q64
  -> Q50.B full graph-level spatial domain                    [queued]
Q50.B full + Q50.A -> Q51.Core real-axis foundation           [queued]
Q51.Core foundation -> Q50.C root-work query / selected leaf emitter
                                                               [queued]
Q50.C -> Q50.D coupled-region domain/algorithm                [queued]
Q50.D -> Q50.E temporal domain/algorithm                      [queued]
Q50.E -> Q50.F partial-state feasibility foundation           [queued]
Q50.F partial foundation -> Q50.G layout constraint solver   [queued]
Q50.G -> Q50.H movement topology/proof                        [queued]
Q50.H -> Q50.I initial storage/slot domain                    [queued]
Q50.I initial + Q63 -> Q50.J event/resource foundation       [queued]
Q50.J foundation -> Q50.K execution-structure alternatives   [queued]
Q50.K -> Q50.I post-structure storage closure                 [queued]
Q50.I post-K closure -> Q50.J schedule closure               [queued]
Q50.J closure -> Q50.F full-coordinate feasibility closure   [queued]
Q49.P + Q50.S + Q50.F full closure + Q50.B–K + Q50.I post-K closure
  -> Q51.Core control closure
                                                               [queued]
Q51.Core closure -> Q51 tiny oracle and single-winner commit [queued]
Q51 -> Q52 workload-driven planning scalability              [queued]
Q64 source registration and library ownership                     [done]
current PyTorch/XLA capture/oracle mechanics + Q59
  -> Q60 frontend production entry                            [done]
Q51 + Q52 + Q60 + Q55 current-interface closure + Q56 board-ready
  -> Q53 current production board readiness                 [queued]
Q53 board-ready + Q56 board-ready
  -> Q57 resident static execution                             [later]
Q53 board-ready + Q58 + Q60
  -> Q61 whole-program scale readiness                         [later]
Q53 board-ready -> Q48 semantic superoptimization             [later]
```

Q50.B先只交付`SpatialPlan/SpatialAssignment`表示、structural close/validation和baseline canonical producer；它不实现完整domain。
Q50.A只消费这份closed assignment。Q50.S交付共同attention semantic foundation、Q50.A闭合exact demand后，Q50.B再交付full
domain/proposals，所以不存在A输入凭空产生或A/B循环；FA/FD不是B之前的search coordinate。
`Q49.P remaining canonical-plan foundations + Q50.F closed-plan core`同样是一个协调纵向：按
`C-root-work → D-singleton-region → E-canonical-temporal → G-canonical-representation → H-canonical-movement → K-Serialized → I-single-slot → J-canonical-schedule → F-core`
把baseline其余已真实消费的component迁入final typed owners，最后Q49 controller消费proof。该纵向不实现或宣称完整
Q50.S attention vertical与Q50.B–K search domains。K之后的I重闭是独立必经checkpoint，不能折叠进J或F。
Q51.Core也分两次：B-full/A后从`SpatialState`用真实domain建立foundation并让explicit `search`进入new owner；后续每个Q50 checkpoint同批扩Core。
F-full后只做control/coverage closure，不能把Core整体拖到所有mechanism之后。
Q50.S不能仅凭op注册或matcher unit标`done`：必须有B/A read-only consumer、C--K/F resource projection、single-winner
decomposition以及none/search纵向。Q50.B–K任何row不能仅凭domain/direct apply标`done`：必须同时有对应Core state/transition、
public search production caller和direct witness。反过来Core不得为attention建立algorithm state，也不得预声明尚无Q50
producer/consumer的nullable axis field。
F也只能对current Core中已经存在的coordinate返回Deferred；尚未实现的next axis由`IncompletePlanningDomain`负责。新axis提交必须
同批扩F observed dependency，禁止一次性预声明G–K missing列表。
上图Q50 checkpoint到Q51的箭头表示“query/domain/emitter已mechanism-ready”，不表示Q50 tracking row已经`done`。search-only
selected emitter的production consumer要到Q51 single-winner commit才真实可达；届时有source-to-package witness和donor迁移门禁的Q50 row
才能同批转`done`。因此Q51依赖mechanism-ready artifacts，不与Q50最终状态形成循环。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前工作与完成门禁 | 设计 / 计划 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `production-output-profiler` | `done` | Q32、Q6.B、configured board | 未插桩Primary TX stream launch-to-completion设备包络、分离的host diagnostics、Trace来源的五类NCC per-tile engine active ns/work-volume摘要、独立Direct-DTE cycles/raw activity、Count/Trace、逐次保留真实动态调用与rdcycle的16-tile timeline、exclusive语义成本与非加和Trace-only成本、final Instr静态work与硬件峰值下界对照、exact output、单一production output instrumentation、三文件专业UI和profile全树`0777`均闭合；不构造card-wide纯engine elapsed，静态cost不回灌ranking。Ranking feedback保留为后续独立门禁。 | 06、14-16；`tasks/archive/board-profiler.md` |
| Q38 | `multi-engine-software-pipelining` | `done` | Q32、Q6.B、Q37 | 历史任务证明typed multi-buffer、prologue/steady/epilogue、Direct-DTE issue/wait和exact range hazard曾可表达、可验证；这些是Q50.I/J/K重审必须逐项承接的donor evidence，不再声称旧fixed-slot/worker source删除已经由current search完成迁移。 | 06、08-17；历史计划已归档 |
| Q39 | `noc-resident-tile-dataflow` | `done` | Q38历史表达/验证能力闭合；板端资格作为独立external gate | 历史任务证明physical peer、resident/spill和collective mechanics；Q50.H/Q51重审正在核对旧NoC intermediate/partial-dataflow、alias/lifetime/slice proof的current owner，旧source删除不再作为迁移完成证据。 | 02-13、16-17；历史计划已归档 |
| Q43 | `compiler-collaboration-review-materials` | `done` | Q9、Q37-Q39完成证据 | 2026-08-03冻结的历史汇报材料已经闭合；它不作为current architecture合同。 | 01、08–16；`tasks/archive/vibe-compiler-collaboration-review.md` |
| Q42 | `test-load-reduction` | `done` | 无 | 默认lit/unit/CTest和owner integration均只判直接合同；历史catalog、批量测试矩阵、model-scale与重复package执行已退出默认入口。 | 16；`tasks/archive/test-gate-scope-reduction.md` |
| Q50.0 | `actual-executable-compilation-boundary` | `done` | 已选择并物化的CardModule | 唯一无策略CardModule→CardExecutable边界已闭合：依次完成Tile module splitting、TileRegion-to-Instr、fresh completion、fixed-capacity SPM/DDR及transport/resource/runtime-launch verification。baseline与search可复用同一lowerer，但production search只能在plan winner选定后调用一次；该边界的typed rejection用于暴露planning/lowering合同缺口，不再作为反复物化candidate或生成no-good的正常控制流。 | 01、06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q54 | `mlir-infrastructure-conformance` | `done` | Q50.0 | Checkpoint J已按普通pass、framework transaction、actual alternative、search evaluation和output fan-out逐类复核active root clone及dormant旧能力。本轮删除普通pass无合同root snapshot、required-NCC重复validation、production selected-buffer nested clone和CardExecutable discarded target gate；最终16 Tile target output每variant只clone/lower/translate一次并按LaunchSlot bounded并发归并。旧search/rank与未注册mechanism已绑定Q51.Core/Q50 extract-then-delete门禁，不冒充current production。fresh三树构建、非搜索unit、target/profile unit、lit、source-to-package/no-card、organization/link和显式work计数通过；Q54本身不替代后续Q49.P的baseline功能与模型级物化门禁，Q63独立承接NCC target/MLIR completion分层。 | 18-19；`tasks/plans/mlir-engineering-remediation.md` |
| Q55 | `current-interface-consolidation` | `done` | Q9 profile合同、Q45命名规则、current target/package/runtime接口 | Wafer-owned frontend metadata、package、target ABI/CRT、profiler、Direct-DTE status、qualification evidence、dependency record与工具fixture均收敛为一种无编号current合同，旧reader/CLI/carrier/alias与版本分支退出active路径。Board calibration逐项迁移source、shape/dtype、oracle、guard、status、timeout、cleanup和profile要求：22个raw probe加complete-Tile add普通/profile共24个current-interface no-card CTest通过，10个Q55-owned真实Board入口进入同一串行runner但本轮未上板；full-4096 K-tiled与M-tiled profile由current global-source runner和catalog/inventory保留，并因global lowering/SPM门禁明确阻塞，不冒充board-ready。fresh configure/build、target unit 140/140、core lit 216/216、current-interface 24/24、Board host 32/32、Q55 tool/runtime lit 5/5、dependency/source/IR/diff检查通过。 | 02、11、14-17、20；`tasks/plans/interface-version-consolidation.md` |
| Q58 | `program-data-ownership` | `done` | Q50.0、Q54、Q55 | 2026-08-16二次review缺口已闭合：handoff在稳定output parent下RAII拥有唯一目录，source持有move-safe只读handle并按handle→file→directory顺序清理，public compile返回前删除staging不再使CardExecutable悬空；establishment从同一source descriptor复制，并从owned descriptor重新验证header/exact extent和whole-file digest；最低层所有read强制≤1MiB，显式scale/test运行可选收集`file_opens/read_windows/read_bytes/maximum_read_window_bytes`，普通compile默认不构造I/O统计。最后一次tensor verification后真实partition已adopt，全部未采用helper payload在Card边界销毁。`ProgramDataTest` 15/15（含32KiB mmap阈值、move/staging lifetime、helper payload清除、2.8MB range三window零新增open、16 Tile shared view），受影响filtered unit 52/52；小型FP16 parameter source→package/no-card 1/1，三档0.5/2/8MiB source→package/no-card 1/1，source/file opens恒定2/8、最大read window≤1MiB，payload增长16倍时peak RSS保持55,392–56,332KiB。feature-on returned-executable target-model参数回归已注册但unsupported，未计入完成证明；详细证据见对应计划。 | 02、14、16、18；`tasks/plans/program-data-and-whole-program-scale.md` |
| Q56 | `executable-package-data-closure` | `board-ready` | Q58 | 二次review缺口已闭合：physical-order bounded codec覆盖多outer-row Cx/NCx与bitpacked BOOL；不同dtype的target-ready bytes已通过确定value conversion产生且禁止bit reinterpret，但当时借formal model route实现的反向依赖不构成current长期合同，由Q62改为显式TargetTensor materialization action；strict verifier从shared codec重算descriptor并拒绝任意尾随字节；package closure包含目录拓扑；materialization按TargetTensor typed reader计数；Compiler/Package的Runtime header反向依赖已由source-organization gate封闭。专用FP16 parameter-add case、oracle和current board runner位于`test/Board/wafer_board_single_op_add_test.py`，并以`wafer-board-program-data-add`注册；同语义source→package→no-card、定向unit/lit、public link、SystemC/model和板端配置构建fresh通过。真实板测尚未执行，因此保持`board-ready`而非`done`。 | 14-18、20；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q59 | `compiler-entry-transaction-closure` | `done` | Q56达到`board-ready` | 2026-08-16 follow-up review缺口已闭合：package层只定义一个compiler/runtime共享的move-only `runtime::ExecutablePackage`；shared strict binder在publication前以read-backed immutable snapshots拥有canonical manifest、manifest顺序modules和program data，核对exact size/digest并关闭全部descriptor，同inode改写、删除或替换path不改变owner内容，board runtime不再按root reopen。profile transaction同样绑定exact activation/plan/site-map和两份capture package，runtime profile identity只对owned manifest bytes计算digest；唯一outer transaction在所有fallible验证完成后执行一次no-replace rename，committed public type只在rename成功后构造。全部test-only failure经public typed facade返回`CompilationFailure`；commit测试直接篡改staged program data/profile plan并证明package-commit分类、输出不可见和无staging残留。CLI current cutover为`--output-dir`，旧`--output-package-dir`无alias；profile输出仍是共同delivery root。fresh验证：受影响unit 120/120；request/internal-options/atomicity/commit-transaction/install-relocate lit 5/5；complete-Tile profile contract 1/1与fresh ordinary+profile source→package→no-card 1/1；public link smoke 3/3；main、board-runtime、SystemC/model三棵构建树及source/deps/IR/diff检查通过。既有32x32 dot被pinned helper拒绝与search长耗时仍分别归外部helper/Q52。 | 01、15、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q62 | `target-numeric-contract-reconstruction` | `done` | Q55、Q56达到`board-ready`、current TargetCall/SystemC/model mechanics | `NumericSemantics` umbrella、profile/pattern/resolver及其digest已删除；rounding/elementwise/reduce字段归`TargetOperation`，convert只有一个typed optional parameter，physical descriptor无model identity。formal scalar/tensor与model backend按convert/elementwise/GEMM/reduce concrete type直连，不再有generic command/family switch；bulk record以concrete GEMM problem、payload、backend和environment绑定。NPY/JSON之后的program logical dtype、Tile ABI target dtype和package record均为closed typed value，字符串只在parse/serialize边界；TargetTensor action随同次Tile ABI metadata显式携带，非identity必须提供rounding/zero-point，package writer不再按dtype默选。`WaferTarget`、`WaferTargetScalarConversion`、`WaferTargetTensorMaterialization`与`WaferFormalNumeric`为单向library owner，CodeGen不链接formal。managed dependency验证由optional Python bootstrap/qualification owner及direct tests承接，always-built Target C++ API退出；`WaferTargetModelCore`不再链接Compiler，compiler integration位于独立leaf adapter。旧SystemC transport聚合目标没有被删除掩盖，而是迁为共享fixture与逐scenario executable；fresh feature-off全构建及9/9 direct/link gate、feature-on target/formal/bulk/model构建、16项current SystemC gate、756项unit、bulk CLI/readback和FP16 source→package/no-card通过；未运行search、LLaMA或板端。Q53继续处理baseline staged output writeback，不属于numeric registry合同。 | 01、11、14、16-18；`tasks/plans/target-numeric-contract-reconstruction.md` |
| Q63 | `ncc-completion-contract-layering` | `done` | Q54、current Instr/NCC target ABI | pure target `NCCCompletion`只拥有target worker/mask/completion kind；IR worker count从closed ODS enum推导且public header不再include TX81 ABI。join和peripheral通过`WaferNCCCompletionOpInterface`各自声明participant join或synchronous writeback，ordinary issue由现有issue interface统一适配，旧free concrete-op switch/classifier零残留。query-local `NCCCompletionAnalysis`从current if/for/TileRegion/direct-call IR重算per-op pending before/after和summary，递归/indirect/unsupported CFG fail closed；全部lifetime/cost/scheduling/lowering/buffering consumer迁移。runtime/model不include IR completion；剩余Model→Compiler宽link由Q62的invocation/numeric owner拆除。fresh相关unit、target public link、主构建及组织检查通过。 | 11、13、17-19；`tasks/plans/ncc-synchronization-contract-layering.md` |
| Q50.B | `spatial-partition-and-placement` | `doing` | representation foundation依赖Q50.0/Q54；full domain依赖Q50.S semantic foundation、Q50.A、Q64 | 当前只推进foundation：定义compact `SpatialPlan`、closed `SpatialAssignment`、structural validation和baseline canonical producer，给A真实输入但不枚举search domain；A/Q49后full stage覆盖全部regular partitions、available subsets/embeddings和per-output merge placements，并逐point调用A。attention mode只约束K2 spatial domain，不成为本轴前的choice。B-full/A成为Q51.Core foundation的首批production domain，Core caller就位前B-full不能标完成。仅topology automorphism可dedup，DP只作proposal/bound；旧trial和混杂signature退出。 | 05-08、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.A | `exact-placement-demand-boundary` | `queued` | Q50.B spatial-assignment representation foundation、Q50.0、Q54 | 本轮设计重审已确认旧完成结论无效。终态只消费B-foundation定义并structurally closed的`SpatialAssignment`，以operand-level SSA worklist派生`ExactDemandProof`、final owners和per-output-piece reduction requirements；它不定义或选择spatial assignment。B-full后续枚举每个plan并调用A，baseline则由B canonical producer调用A。旧`LogicalShardTrial`、partial-as-owner、node-wide merge、per-edge重复walk和manual epoch退出。 | 05-07、09-13、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q49.P | `deterministic-baseline-functional-closure` | `queued` | Q54、Q59、Q50.B foundation、Q50.A、Q50.0、Q45及Q50.F closed-plan core | B-foundation先提供baseline canonical `SpatialAssignment`，A再产生exact demand。随后把baseline其余canonical region/temporal/representation/movement/single-slot/Serialized/order事实迁入最终policy-free components，并交F closed-plan core证明resource；不等待完整search domains。最终删除per-coordinate CardModule/Q50.0，planning为零、selected各一次；旧181.70秒LLaMA只作功能背景，迁移后重跑定向oracle、轻量package/no-card和一轮fresh FP16 LLaMA才能恢复`done`。 | 06、14、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.S | `attention-algorithm-normalization` | `queued` | Q49.P plan-only closure；semantic IR foundation先于B-full/A integration | 在policy分叉前从typed SSA证明完整Q/K/V attention，创建一个`wafer.linalg_ext.attention`并确定FA/FD；none/search消费同一normalized TensorProgram。实现standard tiling/partial-reduction/effect/shape及窄coupled-state query，提供A–K/F所需的query-local work/resource description；Q51不增加algorithm axis。winner只展开一次selected Linalg/Tensor/SCF并转换到existing wafer.tile。旧attention/decode/Flash donor逐项迁移，whole-program clone、Materialized/Online domain和decode-only physical gate退出；prefill与two-step decode分别经两条policy完成package/no-card后才能done。 | 05-07、10、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51.Core | `physical-dataflow-search-kernel` | `queued` | foundation依赖Q49.P、Q50.S semantic foundation和B-full/A；control closure依赖C–K→I→J→F-full | 两次交付同一controller：foundation首次落地从`SpatialState`真实遍历B/A physical states、frontier和typed continuations，explicit public `search`切到new owner；Q50.S fixed graph facts只进入immutable problem，不形成state。source-only facts由与MLIR analysis wrapper共用的policy-free builder形成session-owned typed result，root/candidate key使用`SemanticRootKey`等typed identity，不持有`Analysis *`、pointer或ordinal。缺后续axis时返回`IncompletePlanningDomain`，不调用baseline、不commit。之后每个Q50 checkpoint同批扩closed variant/successor/invalidation与production caller，不预留nullable字段。F-full后补齐full-plan admission、cost/bound、causal rejection、coverage与controller oracle。全程planning零IR，proposal不移动canonical cursor，`none/search`只在final plan/materializer汇合。 | 06、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.C | `maximal-single-root-tile-region` | `queued` | canonical foundation依赖B-foundation/A；full mechanism依赖B-full/Core foundation | Q49纵向先迁`RootRegionWork` canonical query和single-root leaf primitive；任务不因此完成。B-full后扩全部root/merge work、multi-producer exact boundary和atomic emitter，同批接Core derived prefix；selected commit最终提供production apply witness。旧trial/second walker/implicit fallback和Module/Func/DAG clone退出。 | 05-08、10、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.D | `region-formation-and-producer-execution` | `queued` | singleton foundation依赖C-foundation/A；full domain依赖C-full/Core | Q49纵向先产生真实singleton `RegionPlan`、execution instance和use binding；full stage再枚举connected root partitions、top-level/nested execution、stored/direct/boundary use和recompute count，同批扩Core Region state。D不拥有temporal/layout/movement选择；旧greedy repair和whole-module apply退出。 | 05-10、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.E | `complete-temporal-tiling` | `queued` | canonical foundation依赖D-singleton；full domain依赖D-full/Core | Q49纵向先把baseline first-fit迁成真实`TemporalPlan`；full stage再覆盖每个execution scope所有合法iterator sizes、tail与全部structural loop orders，同批扩Core Temporal state和F-partial consumer。旧node-wide/global-max、preferred-size legality和allocation feedback退出。 | 05-10、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.F | `scoped-feasibility-analysis` | `queued` | closed-plan core依赖B-foundation/A、Q50.0与Q49 remaining canonical component extraction；partial foundation依赖Q50.E；full closure依赖K→I-post-K→J | 三次接入同一实现：先让Q49 canonical plan取得FullFeasibilityProof；E后以closed sum返回A–E可证明的minimum-storage/interference facts，但不预声明未来G–K coordinates；每个G/H/I/K/J checkpoint同批扩F对该真实axis的Deferred/observed dependency；最后对Q50.S fixed semantic facts与B–K→I→J完整plan闭合全部state/scratch/message/event resource proof与oracle。后续choice只扩同一problem schema，Q50.0 actual不一致是compiler bug。 | 06-11、16、18-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.G | `layout-and-physical-representation` | `queued` | canonical foundation依赖A/D/E canonical；full solver依赖F-partial/Core | Q49纵向先迁canonical `RepresentationPlan`和resource description；full stage再以`RegionValueVersionId/PhysicalVersionId` constraint graph覆盖primary/derived versions及per-use binding，同批扩Core和F。winner由`PhysicalVersionBuilder`一次构造；旧lookup-any、map save/restore和hidden allocation退出。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.H | `explicit-data-movement` | `queued` | canonical foundation依赖A/D/G canonical；full domain依赖G-full/Core | Q49纵向先迁canonical correctness carrier；full stage统一`TileCommunicationProblem`的local、DDR、Direct-DTE、Tile接收后转发及transfer/combine plans，同批扩Core/F/I/J。Ring/tree/Bruck等只作topology-aware proposals；opaque routing不伪造link，H不拥有retention/recompute/layout。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.I | `storage-binding-and-rotating-slots` | `queued` | canonical single-slot依赖E/H/K-Serialized；initial full domain依赖H-full/Core；post-K依赖K-full | Q49纵向先迁single-slot `BufferPlan`；I-initial再定义fresh/alias/reuse与`1..U` slots供J-foundation/K，K后必须I-post-K重闭multiplicity/rotation/ReuseAfterCompletion并扩Core/F。winner construction扩E/K loops；旧actual-loop scan、clone/reorder和wait/NCC混装退出。 | 08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.J | `event-resource-scheduling` | `queued` | canonical schedule依赖K-Serialized/I-single/Q63；foundation依赖I-initial；closure依赖K-full/I-post-K | Q49纵向先迁source-order/worker0 `ClosedSchedulePlan`；full J-foundation建立EventGraph供K，J-closure在K/I重闭后枚举order/worker/resource/completion并扩Core/F。winner `ScheduleEmitter`创建order/waits/joins/releases；calendar不进state。 | 08、11、13-19；`tasks/plans/physical-dataflow-synthesis.md` |
| Q50.K | `conditional-execution-structure` | `queued` | Serialized foundation依赖E canonical；full domain依赖J-foundation/I-initial/Core | Q49纵向先定义唯一Serialized identity供I/J；full stage再枚举Pipelined stage partition、launch distance和inter-iteration dependences，同批扩Core并强制I/J re-entry。selected builder只在新Card subtree机械构造SCF/periodic phases；multi-slot本身不触发pipeline，旧whole-Module fixed-slot路径退出。 | 06、08-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q51 | `physical-dataflow-unified-search` | `queued` | Q50.S、Q50.B–K、Q50.I post-K、Q50.J schedule closure、Q50.F full closure、Q51.Core | independent reference composer按B→D→E→G→H→I→K→I→J parent/full plan证明全部physical axes、F与cost；Q50.S fixed attention facts作为immutable input，不形成axis。test-only fresh actualization验证plan→selected Linalg→wafer.tile→Q50.0，不进入production。production planning对任意proposal/no-good/bound/parallel开关保持exact winner，CardModule/Instr/Q50.0均为零；selected `PhysicalDataflowPlan`经一个Card subtree transaction和一次Q50.0形成winner。旧Planning/Search每项能力迁入对应Q50/Q51 owner并有direct witness后整目录退役；public search不调用/fallback baseline。 | 06-13、16、18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q52 | `physical-dataflow-search-scalability` | `queued` | Q51 | 在完整Q51链上显式profile state/continuation/query/solver work、16-Tile等价问题、time-to-first-full-proof和RSS；基于完整`FutureBoundaryKey`加入session-local memo、separator/component DP和stronger admissible bounds，只共享pure result不强迫assignment。constructive lane先取得full-proof plan，typed LNS再联合释放耦合coordinates并复用同一transitions/F gate；永久丢state的policy最多BudgetedFeasible。production allowance/deadline由fresh曲线选择，无candidate-count/LNS CLI；代表LLaMA须在最终有限policy内返回plan并一次commit/package/no-card。 | 06、14-18；`tasks/plans/physical-dataflow-synthesis.md` |
| Q64 | `source-registration-truth-closure` | `done` | source registration与library ownership；Q62/Q63继续拥有各自语义清理 | 目录、CMake注册和library ownership重组本身保留；重审明确source organization只能证明文件归属与active build，不能证明被删算法/proof/test已迁移。Q51+相关“dormant/可删除”分类全部以当前donor-to-current审计重新判定，Q64不为Q50.S attention vertical、Q50.B–K或Q51完成状态背书。 | 18；`tasks/plans/source-registration-truth-closure.md` |
| Q60 | `frontend-production-entry` | `done` | current PyTorch/XLA capture/export资产、Q59 | 产品`wafer.frontend.export_pytorch_program(module, example_inputs, output_directory)`只拥有strict framework export、BF16 state保存与current source写入，无case/seed/oracle/target/search/model-name入口。`functions/forward.stablehlo.bc`是唯一source IR authority；旧text/generic bytecode在source中拒绝，post-SPMD text仅属compiler internal parser。`WaferStableHLOProgram`由installed `wafer-verify-program`与compiler transaction共同消费，后者仍独立snapshot并以owned payload fresh验证；显式text只进`wafer-opt`的`wafer-frontend-verification`。旧`wafer-compile-stablehlo`零alias；feature-off无stub，relocated install的adapter→verifier→compiler→no-card通过。fresh ordinary/branched/repeat/graph-break/pre-exported/corrupt/metadata/path/dynamic门禁及251/251 supported lit通过（4 configured unsupported）。 | 01-02、18-20；`tasks/plans/compiler-entry-productization.md` |
| Q53 | `physical-dataflow-production-readiness` | `queued` | Q51、Q52、Q60、Q55 current-interface closure、Q56达到`board-ready` | Q53-1以同一只读source的两份fresh transaction分别证明`none`/`search`单plan commit、accepted Tile dataflow→Instr→Target LLVM、typed effectiveness、package和no-card；small structured/communication/tail-storage与HF prefill、functional decode、Llama使用FP16/BF16，heavy只进显式campaign。Q53-2复用current `QualifiedBoardRuntimeSession`，第一次真实case确认一次device qualification，后续case串行而每phase完整16-Tile提交；small communication correctness后，Llama 3 pairs与prefill/decode代表5 pairs交替matched A/B，逐sample校验output/guard并按completion resolution分类。无板矩阵与runner完整才`board-ready`；两项matched均Improved才`done`。 | 02、06-16、18；`tasks/plans/physical-dataflow-synthesis.md` |

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
| Q48 | `semantic-superoptimization` | `later` | Q53按current CardModule/Tile合同达到`board-ready`，且current compiler/runtime closure可消费final Instr/TargetCall；当前前置未满足 | 从actual structured MLIR生成verifier-legal TensorProgram alternatives并以query-local solver证明外部可观察value、memory与effect等价；alternatives进入Q51 planning state，只有最终physical winner进入一次Q50.0。Q48不在Instr层建立第二selector，不预设shortlist或固定候选cap。 | 05-08、10-11、16-18；`tasks/plans/semantic-superoptimization.md` |
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
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | 历史MLIR-native IndexRelation与typed lowering foundation闭合；旧implementation-choice资产是否由Q50.S attention owner、Q50.B–K/Q51完整承接正在重审，不能再由current source删除直接推断迁移完成。 | `tasks/archive/mlir-native-implementation-relation-foundation.md` |
| Q32.R | `physical-relation-realization` | `done` | Relation、encoding、transfer和resident boundary realization闭合。 | `tasks/archive/physical-relation-realization.md` |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Production-shaped spill/resident actual-clone seam及late gates闭合。 | `tasks/archive/physical-dataflow-test-seam-vertical.md` |
| Q32.V | `typed-target-capability-vertical` | `done` | Mapped transfer、physical fill和oriented GEMM typed capability纵向闭合。 | `tasks/archive/typed-target-capability-vertical.md` |
| Q32.M | `physical-mechanism-choice-closure` | `done` | 历史任务闭合了recompute、LICM、numeric、residency、ready-order和communication实验；这些能力是Q50/Q51重审donor，旧API删除不等于current mechanism已迁移。 | `tasks/archive/physical-mechanism-choice-closure.md` |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Bounded actual-clone组合、card Pareto和target-owned选择闭合。 | `tasks/archive/bounded-joint-physical-dataflow-selection.md` |
| Q32.G | `physical-dataflow-production-cutover` | `done` | `wafer-compile`成为唯一production decision owner，旧旁路退役。 | `tasks/archive/physical-dataflow-production-cutover.md` |
| Q32 | `physical-dataflow-synthesis` | `done` | 历史任务证明若干implementation/relation/layout/storage/order/communication机制；current Q51+重审以其源码和测试作donor evidence，不继承旧candidate API、clone架构或历史完成结论。 | `tasks/archive/physical-dataflow-synthesis-completion-audit.md` |
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
