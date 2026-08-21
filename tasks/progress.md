# Wafer Compiler Task Queue

更新时间：2026-08-21

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。历史状态变化由Git保留。

## 队列规则

- `Q*`只标识稳定设计/能力owner，不表示pipeline层级，也不直接充当可反复进入的施工任务。
- current execution work item使用稳定semantic key；每个work item只有一个输入边界、一个输出边界，在当前调度中只出现一次。
- 全局至多一个work item为`doing`；`next`表示前置已满足但尚未开始，`queued`表示已经进入当前主线但仍等待直接前置闭合，
  `later`表示不进入当前主线。
- work item的`done`只表示该行输出合同已经满足；Q owner的整体完成只由owner map列出的全部work item及production gate汇总，
  不让owner row在`doing/queued`之间往返。
- current职责被后继吸收时，旧任务直接移出当前调度和状态表；可复用的source、oracle、mechanics与验证要求写入
  current owner，历史只由Git和`tasks/archive/`保留，不新增旧任务过渡状态或旧任务索引。
- `blocked`必须写明尚缺的外部条件或上游任务，不能用工作过程代替阻塞原因。
- 新work item先进入本表并绑定编号设计owner；非小修再更新`tasks/plans/`实施计划。
- 状态变化只更新对应row；不得追加按日期、轮次或测试批次展开的worklog。
- 后续任务gate只验证直接影响面；除非任务本身改动相关ABI/runtime/firmware，否则禁止全量审计和无关回归。
- 包含板端验证的任务必须把`board-ready`写为无卡阶段门禁；`board-ready`不得标记为`done`。

## 当前调度

当前执行队列只列一次性work item；顺序已经按artifact producer/consumer关系拓扑排序。Q50.0、Q54、Q55、Q56、Q59、Q60、Q62、
Q63和Q64等前置已满足，不在当前队列中重复展开。

| 顺序 | Work item | 状态 | 设计owner | 直接输入 | 完成输出 |
| --- | --- | --- | --- | --- | --- |
| 1 | `spatial-plan-schema` | `doing` | Q50.B | Q50.0、Q54 | `SpatialPlan/SpatialAssignment` schema、close与validator；不含root assignment或search domain |
| 2 | `attention-normalization` | `queued` | Q50.S | 05 normalizer、Q60产品输入 | self-contained attention op、FA/FD fixed facts及共同normalization |
| 3 | `attention-spatial-integration` | `queued` | Q50.S | spatial-plan-schema、attention-normalization | K1/K2 role及FA/FD canonical/full spatial constraints |
| 4 | `canonical-spatial-assignment` | `queued` | Q50.B | spatial-plan-schema、attention-normalization、attention-spatial-integration、topology | all normalized roots的deterministic closed assignment |
| 5 | `exact-demand-boundary` | `queued` | Q50.A | canonical-spatial-assignment | operand demand、final owners及per-output reduction requirements |
| 6 | `attention-demand-integration` | `queued` | Q50.S | attention-normalization、exact-demand-boundary | Q/K/V/mask demand及coupled contribution/merge |
| 7 | `canonical-root-work` | `queued` | Q50.C | canonical-spatial-assignment、attention-demand-integration | canonical `RootRegionWork`与single-root leaf primitive |
| 8 | `canonical-region-plan` | `queued` | Q50.D | canonical-root-work、exact-demand-boundary | singleton `RegionPlan`、execution instance及use binding |
| 9 | `canonical-temporal-plan` | `queued` | Q50.E | canonical-region-plan | canonical `TemporalPlan`、tail及loop order |
| 10 | `canonical-representation-plan` | `queued` | Q50.G | canonical-temporal-plan、attention-demand-integration | canonical `RepresentationPlan`与resource description |
| 11 | `canonical-movement-plan` | `queued` | Q50.H | canonical-region-plan、canonical-representation-plan | canonical local/DDR/peer correctness carrier |
| 12 | `serialized-execution` | `queued` | Q50.K | canonical-temporal-plan | unique Serialized execution identity |
| 13 | `canonical-storage-plan` | `queued` | Q50.I | canonical-movement-plan、serialized-execution | single-slot `BufferPlan`及lifetime facts |
| 14 | `canonical-schedule` | `queued` | Q50.J | canonical-storage-plan、serialized-execution、Q63 | source-order/worker0 `ClosedSchedulePlan` |
| 15 | `attention-work-projection` | `queued` | Q50.S | attention-normalization、canonical-root-work至canonical-schedule | `AttentionWorkDescription`及C–K/F resource projections |
| 16 | `canonical-feasibility-proof` | `queued` | Q50.F | canonical-spatial-assignment至canonical-schedule、attention-work-projection、Q50.0 | canonical problem/parity及`FullFeasibilityProof` |
| 17 | `attention-selected-decomposition` | `queued` | Q50.S | attention-work-projection、canonical-feasibility-proof | winner-only selected Linalg/Tensor/SCF→wafer.tile builder |
| 18 | `deterministic-baseline-closure` | `queued` | Q49.P | canonical-spatial-assignment至attention-selected-decomposition、Q59 | pure legalization、一次commit/Q50.0及fresh `none`纵向 |
| 19 | `spatial-domain` | `queued` | Q50.B | spatial-plan-schema、attention-spatial-integration、attention-demand-integration、exact-demand-boundary、Q64 | complete spatial successors、reference enumerator及proposal |
| 20 | `search-control-foundation` | `queued` | Q51.Core | spatial-domain、exact-demand-boundary、attention-demand-integration | SpatialState frontier/continuation及public `search` routing；missing axis typed incomplete |
| 21 | `root-work-domain` | `queued` | Q50.C | search-control-foundation、canonical-root-work | full root/merge work domain、Core consumer及selected emitter |
| 22 | `region-execution-domain` | `queued` | Q50.D | root-work-domain、canonical-region-plan | region/execution/use-binding domain及Core consumer |
| 23 | `temporal-domain` | `queued` | Q50.E | region-execution-domain、canonical-temporal-plan | complete temporal sizes/orders/tails及Core consumer |
| 24 | `partial-feasibility` | `queued` | Q50.F | temporal-domain、exact-demand-boundary | A–E minimum/interference、Deferred及causal query |
| 25 | `layout-domain` | `queued` | Q50.G | partial-feasibility、canonical-representation-plan | representation constraint solver、Core consumer及apply |
| 26 | `movement-domain` | `queued` | Q50.H | layout-domain、canonical-movement-plan | local/DDR/DTE/relay/collective domain、proof及Core consumer |
| 27 | `storage-domain` | `queued` | Q50.I | movement-domain、canonical-storage-plan | fresh/alias/reuse与`1..U` slot domain及Core consumer |
| 28 | `event-resource-foundation` | `queued` | Q50.J | storage-domain、Q63 | EventGraph、resource/recurrence facts及Core consumer |
| 29 | `execution-structure-domain` | `queued` | Q50.K | event-resource-foundation、storage-domain、serialized-execution | Serialized/Pipelined structure domain及Core consumer |
| 30 | `structure-specific-storage` | `queued` | Q50.I | execution-structure-domain、storage-domain | fixed-K occurrence、slot multiplicity、rotation及lifetime closure |
| 31 | `schedule-domain` | `queued` | Q50.J | structure-specific-storage、event-resource-foundation | fixed-K/I order、worker、resource与completion domain |
| 32 | `full-feasibility` | `queued` | Q50.F | schedule-domain及完整B–K→I→J plan | full resource proof、oracle及Core admission input |
| 33 | `search-control-closure` | `queued` | Q51.Core | full-feasibility、全部domain work items | full-plan admission、cost/bound、causal rejection、coverage及controller oracle |
| 34 | `unified-search-closure` | `queued` | Q51 | search-control-closure、attention-selected-decomposition、Q50.0 | tiny exhaustive oracle、single winner及一次production commit |
| 35 | `attention-production-closure` | `queued` | Q50.S | deterministic-baseline-closure、unified-search-closure | donor retirement及prefill/decode的none/search package/no-card |
| 36 | `search-scalability` | `queued` | Q52 | unified-search-closure、attention-production-closure | measured memo/DP/bound/LNS policy及有限预算LLaMA一次commit |
| 37 | `production-host-readiness` | `queued` | Q53 | search-scalability、Q60、Q55、Q56 board-ready | fresh source/IR/package/oracle/runner/no-card矩阵；Q53 `board-ready` |
失败留在当前work item修复；不跳过、不fallback，也不把owner整体状态提前标为完成。

### 设计owner映射

| 设计owner | 当前work items | Owner整体完成边界 |
| --- | --- | --- |
| Q50.B | spatial-plan-schema、canonical-spatial-assignment、spatial-domain | 三项均通过且spatial-domain在unified search中取得production witness |
| Q50.S | attention-normalization、attention-spatial-integration、attention-demand-integration、attention-work-projection、attention-selected-decomposition、attention-production-closure | 最后一项通过 |
| Q50.A | exact-demand-boundary | work item通过且所有physical consumers迁移 |
| Q50.C | canonical-root-work、root-work-domain | 两项通过且unified search取得selected witness |
| Q50.D | canonical-region-plan、region-execution-domain | 两项通过且unified search取得selected witness |
| Q50.E | canonical-temporal-plan、temporal-domain | 两项通过且unified search取得selected witness |
| Q50.F | canonical-feasibility-proof、partial-feasibility、full-feasibility | 三项及独立oracle通过 |
| Q50.G | canonical-representation-plan、layout-domain | 两项通过且unified search取得selected witness |
| Q50.H | canonical-movement-plan、movement-domain | 两项通过且unified search取得selected witness |
| Q50.I | canonical-storage-plan、storage-domain、structure-specific-storage | 三项通过且K re-entry witness闭合 |
| Q50.J | canonical-schedule、event-resource-foundation、schedule-domain | 三项通过且completion/resource witness闭合 |
| Q50.K | serialized-execution、execution-structure-domain | 两项通过且I/J re-entry闭合 |
| Q49.P | deterministic-baseline-closure | 对应work item通过 |
| Q51.Core | search-control-foundation、search-control-closure | 两项通过 |
| Q51 | unified-search-closure | 对应work item通过 |
| Q52 | search-scalability | 对应work item通过 |
| Q53 | production-host-readiness | 对应work item通过即为`board-ready`；真实板端不在当前目标 |

### 已满足前置

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
| Q64 | `source-registration-truth-closure` | `done` | source registration与library ownership；Q62/Q63继续拥有各自语义清理 | 目录、CMake注册和library ownership重组本身保留；重审明确source organization只能证明文件归属与active build，不能证明被删算法/proof/test已迁移。Q51+相关“dormant/可删除”分类全部以当前donor-to-current审计重新判定，Q64不为Q50.S attention vertical、Q50.B–K或Q51完成状态背书。 | 18；`tasks/plans/source-registration-truth-closure.md` |
| Q60 | `frontend-production-entry` | `done` | current PyTorch/XLA capture/export资产、Q59 | 产品`wafer.frontend.export_pytorch_program(module, example_inputs, output_directory)`只拥有strict framework export、BF16 state保存与current source写入，无case/seed/oracle/target/search/model-name入口。`functions/forward.stablehlo.bc`是唯一source IR authority；旧text/generic bytecode在source中拒绝，post-SPMD text仅属compiler internal parser。`WaferStableHLOProgram`由installed `wafer-verify-program`与compiler transaction共同消费，后者仍独立snapshot并以owned payload fresh验证；显式text只进`wafer-opt`的`wafer-frontend-verification`。旧`wafer-compile-stablehlo`零alias；feature-off无stub，relocated install的adapter→verifier→compiler→no-card通过。fresh ordinary/branched/repeat/graph-break/pre-exported/corrupt/metadata/path/dynamic门禁及251/251 supported lit通过（4 configured unsupported）。 | 01-02、18-20；`tasks/plans/compiler-entry-productization.md` |

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
