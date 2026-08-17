# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 `tasks/progress.md` 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序只读取`tasks/progress.md`中的状态和“必须满足的前置”；该列可为审计重复列出关键传递gate，不等同最小DAG边集。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | compiler stack主架构：production pipeline、library/CLI transaction、IR/module/package graph、跨层不变量、consumer分支和职责索引 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | StableHLO program directory、产品adapter边界、ProgramDataSource/ProgramDataRange lifetime与frontend验证 |
| 03 | `tasks/03-shardy-spmd.md` | Shardy/XLA的card-level GSPMD output与`num_partitions`；不绑定片内Tile |
| 04 | `tasks/04-topology-execution-mesh.md` | logical card partition mesh与独立target card/Tile topology |
| 05 | `tasks/05-local-compute-normalization.md` | card-partition-local structured compute normalization与tensor collective boundary |
| 06 | `tasks/06-physical-dataflow-synthesis.md` | card-local `TensorProgram -> CardModule/TileRegion/Instr -> CardExecutable`综合：统一决定spatial、temporal、fusion、physical representation、movement、buffering与调度；预算限制搜索工作而不预先截断合法域 |
| 07 | `tasks/07-tile-region.md` | selected tile/dataflow actual IR物化；一个或多个non-nested `tile.region`表达SPM residency domains，data边variadic DDR、SPM root不跨界，boundary不自动产生movement或join |
| 08 | `tasks/08-physical-realization.md` | physical encoding attr/type语义、valid domain、view、transfer realizability analysis、descriptor cover和selected physical realization |
| 09 | `tasks/09-spm-memory-planning.md` | SPM lifetime/coexistence、fixed-capacity MiniMalloc legality、all-root coverage、validated placement/headroom和accepted offsets；candidate choice仍由06拥有 |
| 10 | `tasks/10-compute-movement.md` | 窄source implementation OpInterface/external model、typed target-abstract compute/movement、standard MLIR effects/interface reuse和issue/token/fence/wait |
| 11 | `tasks/11-instruction-ir.md` | complete static Tile instruction program、current descriptor/geometry/range/narrowing legality及mapped/physical-fill/oriented typed extension |
| 12 | `tasks/12-ddr-memory-planning.md` | 当前DDR demand/accepted offsets；多DDR分区、state和streaming延后 |
| 13 | `tasks/13-communication.md` | selected Tile edge到typed p2p/staging/token/wait IR、Direct DTE card-scoped verification和completion；multi-card延后 |
| 14 | `tasks/14-target-code-generation.md` | `CardExecutable`的current target identity/format、accepted immutable data preparation、structure-preserving conversion、CRT ABI和atomic target-module writing |
| 15 | `tasks/15-launch-runtime-package.md` | `CardExecutable -> ExecutablePackage`、ProgramTensor/TargetTensor、program-data.bin、TileEntryArgument、runtime memory plan、no-card和board adapter边界 |
| 16 | `tasks/16-verification-contract.md` | `TensorProgram -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`的target correctness、data/whole-program scale、CPU oracle、target-model、no-card和board分层gate |
| 17 | `tasks/17-target-execution-model.md` | `CardExecutable`及其same-invocation target LLVM owner消费、multi-dtype numeric、oneDNN bulk、target-call/SystemC untimed CModel与板端numeric correlation边界 |
| 18 | `tasks/18-source-organization.md` | 跨pipeline的源码ownership、translation unit、compiler library/tool/install、构建依赖和测试镜像组织合同；不改变IR/output语义 |
| 19 | `tasks/19-mlir-engineering.md` | 跨IR层的ODS、standard interface、operation-scoped pass/analysis、rewrite/conversion和named nested pipeline工程合同；不重定义01–18语义 |
| 20 | `tasks/20-interface-evolution.md` | 跨compiler/runtime/tool的内部接口演进、持久化格式与ABI版本owner、集中兼容检查和current-only表示合同 |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program、产品adapter、program data ownership和当前program directory | 02 |
| pre-SPMD topology和execution mesh | 04 |
| Shardy/SPMD card-partition output | 03 |
| card-partition-local compute normalization与collective boundary | 05；跨stage正确性证据由16约束 |
| card-local multi-Tile physical-dataflow综合、selected Card/Tile MPMD materialization和physical encoding/transfer | 06、07、08；source implementation interface由10提供，instruction legality由11提供，exact resource/transport gate由09、12、13提供 |
| policy-free physical-dataflow rewrites | 06、07、08；upstream structured utility由05提供，source/direct typed lowering合同由10提供，源码ownership由18约束 |
| source implementation interface、target-abstract compute/movement和instruction legality | 10、11；transfer realizability/descriptor cover只由08拥有 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12；shared lifetime analysis的源码ownership和测试镜像由18约束 |
| Tile peer/collective materialization、Direct DTE completion、card-scoped acceptance与post-memory transport activation | 13；joint choice由06、target/package/verification consumer由14、15、16约束 |
| card-local multi-Tile时空调度、MPMD executable构造和`CardExecutable` | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、accepted immutable data preparation、CRT/device link和staged target module | 14 |
| target-ready data、typed manifest、RuntimeInvocationPlan和board launch | 15 |
| 横跨上述边界的completion evidence | 16 |
| target execution model、multi-dtype numeric/bulk、same-invocation target module消费、SystemC/CModel capability、板端numeric correlation和deferred timing | 17；target module形成与writing合同由14拥有，target/runtime/verification consumer由14、15、16约束 |
| 跨上述边界的源码与构建模块化 | 18；各IR/output语义仍由01-17拥有 |
| 跨上述IR层的MLIR operation scope、pass/analysis manager、interface和rewrite工程合同 | 19；各层具体语义仍由01-18拥有 |
| 跨compiler/runtime/tool的格式与ABI版本边界 | 20；具体字段语义仍由02、11、14-17拥有 |

## 实施计划导航

旧layout与ABI施工记录已移入`tasks/archive/`。current layout、target ABI、package和runtime合同只由06、08、
11、14-17编号设计文档拥有，状态只看`tasks/progress.md`；不得从旧计划恢复接口。

Q49.P、Q50、Q51–Q53共用card-local multi-Tile综合计划`tasks/plans/physical-dataflow-synthesis.md`。队列按可验证边界拆成：
Q50.0建立candidate compilation/verification seam；Q50.A先收口IndexRelation demand边界；Q49.P闭合从正常上游IR产生
accepted executable的baseline deterministic feasibility legalization、single-root TileRegion、
最窄scope probe、typed causal witness、整图物化与search-policy隔离；Q50.S把算法等价改写物化为actual `TensorProgram`；
Q50.B–Q50.K逐项接入
spatial、TileRegion/temporal/fusion、physical representation/movement、buffer/order/completion和条件式stage pipeline选择；
Q51.Core直接复用Q49.P accepted baseline作为session-level incumbent后，只建立typed assignment/transition、deterministic
frontier、ledger/budget和evaluation/result evidence的search control kernel，并原子接管public `search`的winner control；既有
bounded generation只作为显式lossy typed proposal source保全待迁移能力。Core不预声明未来轴、不拥有Q50.F actual probe，也不以
fixture签发真实domain完整性。Q50.S/B–K逐轴补齐mechanism与独立reference oracle，Q51再以full actual oracle闭合联合选择并删除
有损过渡proposal，Q52按实际负载优化scalability，Q53形成production
`board-ready`与真实板端证据。06仍是唯一联合决策设计owner；任务拆分只提供可验证接入checkpoint，不产生独立layout、
fusion、buffering、communication或worker selector。动态状态、依赖和完成门禁只看`tasks/progress.md`。

Q54 MLIR工程化整改计划见`tasks/plans/mlir-engineering-remediation.md`。19是横向工程合同owner：让现有operation/region
层级成为真实pass与analysis层级，收口typed ODS、standard interface、named nested pipeline和transactional rewrite，并
通过18定义的source truth gate；它不产生新IR stage或第二production driver。Q54优先于Q50.A，Q49.P再消费Q50.A与Q54的
current seam后由Q51继续施工，避免把semantic Location、whole-module local wrapper和手工analysis lifecycle固化进
baseline probe或新的candidate/search实现。

Q55接口版本收敛计划见`tasks/plans/interface-version-consolidation.md`。20只定义版本owner和兼容边界；02、11、14-17
继续拥有具体frontend、target、package、runtime、profiler与verification字段语义。Q55不建立compatibility mode，
只保留真实外围版本并让repo内同步接口回到一种current表示。

Q58 program data ownership与Q61 whole-program scale共用
`tasks/plans/program-data-and-whole-program-scale.md`。Q58先把verified source→外部SPMD helper→CardExecutable同事务
data handoff切换成`ProgramDataSource`、checked `ProgramDataRange`和move-only `ProgramDataHandoff`，删除整树和per-Tile
大payload复制，并以完整大型参数inventory量化RSS/disk/IO；它不选择target layout或定义package schema。Q61在Q53
`board-ready`后用完整程序验证frontend、IR、search、target和
package的共同规模，Llama 7B仅作可选named scale witness，不把LLM或serving协议写入compiler合同。

Q56 package数据闭合与Q57设备驻留执行共用`tasks/plans/executable-package-and-resident-runtime.md`。Q56消费Q58 handoff，
把`ProgramTensor`、selected `TargetTensor`、`data/program-data.bin` file range和`TileEntryArgument`闭合为唯一静态package合同；
runtime在side effect前预排program data与invocation memory，并以`BoardDeviceMemory base + offset`绑定，不逐tensor分配。
Q57只在Q53按新package合同达到`board-ready`后启动，把同一合同延长为`PreparedExecution -> submit* -> close`；当前仍以
single context、单inflight、无cancel和provider声明的loaded graph/module上限为事实，不把external execution engine、
multi-inflight或persistent device loop并入runtime。15继续是唯一package/runtime设计owner，实施计划不复制current schema。

Q59 compiler entry transaction与Q60 frontend production entry共用
`tasks/plans/compiler-entry-productization.md`。Q59在Q56达到`board-ready`后让compiler library primary result、package commit、CLI status
与install tree成为同一事务；Q60在Q52 search scalability后将最小framework adapter与portable StableHLO ingestion接入
同一source contract。Q53必须从Q60产品入口fresh生成证据，不能继续把test generator当成用户frontend。

Q48语义驱动superoptimizer计划见`tasks/plans/semantic-superoptimization.md`。它必须在Q53按card-local multi-Tile新合同
重新达到`board-ready`、Q47 current ABI可消费final Instr/TargetCall后启动，
复用05-08、10-11、16-18的现有IR、candidate、proof consumer、model和源码ownership合同；目标是把自动生成并证明的
actual MLIR统一物化为`TensorProgram` alternative并交给Q51唯一owner，不另建语义IR/interface/sidecar，也不预设独立
shortlist或固定候选cap。动态状态和
完成门禁只看`tasks/progress.md`。

Q9 profiler foundation已按`tasks/archive/board-profiler.md`完成：唯一public入口是
`wafer-compile --profile`；profile transaction只发布一个未插桩Primary production output，最后写入的activation把该
production manifest与instrumentation metadata exact-hash绑定，不另编一个关闭profile的ordinary package做逐字节对照。
`wafer-run`复用既有resource/expected/output binding，在一个qualified session内固定执行一次
未插桩Primary、一次Count和一次Trace；Primary主延迟是production launch前后同一TX stream event pair的
launch-to-completion设备包络，host submit与host launch→trusted-completion只作独立诊断。Trace header同时提供
entry-local span、五类NCC engine aggregate PMU和16-tile typed engine/DTE event。
tile clock未资格化时只展示16行entry-local timeline，不声称跨tile顺序。它尚不回写candidate cost；动态状态只看
`tasks/progress.md`。report writing体积收口另列Q9.R later；它不改变Q9采集协议已经完成的结论。

最新完成的compiler throughput收口为Q32.C，记录见
`tasks/archive/whole-variant-search-throughput.md`。它在当时的历史架构中保持candidate domain、hard cap、exact gate和winner语义，
删除passing ordinal之后的无消费者评估、逐batch线程/context churn、accepted module二次lowering、不可达owner
parse及不会进入fully-gated Pareto candidate set的ABI/LLVM lowering；性能记录只含优化后Release实测，不重跑旧二进制。
Q32.N numeric algebraic extension施工记录见
`tasks/archive/numeric-algebraic-extension.md`。它直接删除physical-dataflow algebraic、reduction/GEMM切分和
Ring collective中不必要的float类型门槛，以无额外标注的f16/bf16覆盖现有production pipeline，不增加
frontend mode、私有numeric policy或Tile/Instr carrier。Q36 topology-aware collective lowering已经闭合，
证据归档为`tasks/archive/topology-aware-collective-lowering.md`。Q35 full-4096 K-sharded GEMM board vertical也已完成，
实施与板端重复raw-exact证据归档为`tasks/archive/k-sharded-gemm-board-vertical.md`；它复用Q6.B cluster
Direct DTE路径闭合large-shape M/N tiling与communication，没有重开runtime ABI，也不单case完成Q22.C。
Q6.B board runtime完成计划已归档为
`tasks/archive/runtime-board.md`。Q32 integrated completion audit已归档为
`tasks/archive/physical-dataflow-synthesis-completion-audit.md`，完成后的实施计划归档为
`tasks/archive/physical-dataflow-synthesis.md`；Q32.G/S/M/V/B/R/I各checkpoint仍由对应独立归档记录保存详细变更和
验证证据。Q34 static memory packing已归档为`tasks/archive/static-memory-packing.md`。这些计划和记录只保存施工checkpoint、
验证/删除门槛与历史证据；动态blocked-by只看progress，算法与IR合同仍由01、05、06-18编号设计文档拥有，
横向MLIR工程合同由19拥有。

已完成Q31标准7B单block多seed数值表征和source/model gate收紧归档为
`tasks/archive/llama-block-numeric-characterization.md`；Q30标准7B单block production vertical性能收口归档为
`tasks/archive/llama-block-production-performance.md`，Q28 Llama-2 7B单block纵向归档为
`tasks/archive/llama-7b-block-vertical.md`，Q29
tile-dataflow scheduling归档为`tasks/archive/tile-dataflow-scheduling.md`。Q27 reference executor退役已归档为`tasks/archive/reference-executor-retirement.md`，
Q26 memory lifetime analysis已归档为`tasks/archive/memory-lifetime-analysis.md`，
Q25剩余聚合边界模块化已归档为`tasks/archive/residual-source-modularity.md`，Q24剩余热点模块化已归档为
`tasks/archive/remaining-source-modularity.md`，Q23首轮源码组织重构已归档为
`tasks/archive/source-organization-refactor.md`；
Q13.W依赖root一致性计划已归档为`tasks/archive/third-party-dependency-root-consistency.md`。Q22.S、Q22.V与最终
完成性审计计划已分别归档为`tasks/archive/systemc-functional-event-model.md`、
`tasks/archive/target-model-source-verticals.md`和`tasks/archive/target-model-completion-audit.md`。
Q22.B/Q22.L/Q22.N/Q0.L/Q22.H计划分别归档为`tasks/archive/target-bulk-qualification.md`、`tasks/archive/target-llvm-module-bundle.md`、
`tasks/archive/target-numeric-foundation.md`、`tasks/archive/target-command-legality-closure.md`和
`tasks/archive/target-call-functional-frontend.md`，Q22.R readiness归档为`tasks/archive/target-model-readiness.md`。
Q22.N+Q22.H已经在Q22.S SystemC event model汇合，Q22.B与Q22.S再由Q22.V source vertical闭合并完成Q22汇总；
后续代码任务开工前都需独立计划。动态执行状态只看`tasks/progress.md`。

## 归档文档

`tasks/archive/` 只保存历史 gap review、recovery、audit 和已收口的任务级记录。归档文档可以作为
实现背景或复盘材料，但不作为当前主线架构合同；如果归档内容和 numbered docs 冲突，以当前 numbered
docs、`tasks/progress.md` 和本轮已收敛设计结论为准。

归档正文中的编号文档basename、章节号、line range和命令按当时提交快照解释，不保证仍是当前可解析路径；当前owner路径
只从上面的“当前设计文档”表读取。重命名current owner时不机械改写archive，以免篡改历史审计证据。

| 文档 | 原性质 |
| --- | --- |
| `tasks/archive/whole-card-tile-dataflow-synthesis.md` | 2026-08-11至08-13的旧Q49/Q50完整施工计划；旧任务拆法、shortlist和owner合同不再有效 |
| `tasks/archive/whole-rank-tile-dataflow-synthesis.md` | 2026-08-08的structured-DAG/card历史施工计划；已由current physical-dataflow计划替代，旧public policy与bounded candidate set不再有效 |
| `tasks/archive/whole-variant-search-throughput.md` | 已完成Q32.C的passing-ordinal early stop、bounded persistent candidate executor、accepted-module owner import、exact attempt-plan selective parse和fully-gated Pareto前置late ABI/LLVM，并记录优化后Release单次实测 |
| `tasks/archive/k-sharded-gemm-board-vertical.md` | 已完成Q35 full-4096 f16 K-sharded GEMM的production tiling/SPM/Direct-DTE package、纯tiling隔离及16-rank重复板端raw-exact记录 |
| `tasks/archive/runtime-board.md` | 已完成Q6.B的typed TX board provider、kernel/model多tile launch、cluster Direct DTE、failure lifecycle及真实板端重复exact记录 |
| `tasks/archive/physical-dataflow-synthesis-completion-audit.md` | 已完成Q32的七checkpoint证据映射、双配置全量门禁、fixed/held-out 7B scale重放、单一production owner及剩余边界审计 |
| `tasks/archive/physical-dataflow-synthesis.md` | 已完成Q32 MLIR-native bounded physical-dataflow synthesis的施工checkpoint、hard-cap与integrated completion checklist |
| `tasks/archive/physical-dataflow-production-cutover.md` | 已完成Q32.G默认production winner cutover、旧decision surface删除、all-rank双键correspondence及source/bulk数值回归记录 |
| `tasks/archive/bounded-joint-physical-dataflow-selection.md` | 已完成Q32.S的actual-clone有界联合candidate set、reserved baseline、validated card exact cost、target static policy及逐producer whole-winner记录 |
| `tasks/archive/physical-mechanism-choice-closure.md` | 已完成Q32.M的actual-clone recompute/LICM/integer algebra、partial fanout、spill/resident/ready-order、communication alternatives及重复layout/resource/collective/instruction合同删除记录 |
| `tasks/archive/physical-relation-realization.md` | 已完成Q32.R的rich IndexRelation、physical encoding interface、TransferRealizability、destination-style load、relation-backed resident boundary和fresh 7B TP16数值纵向记录 |
| `tasks/archive/typed-target-capability-vertical.md` | 已完成Q32.V的mapped DMA双端offset、physical-footprint fill、source/Tile/Instr oriented GEMM及formal/SystemC fresh纵向记录 |
| `tasks/archive/mlir-native-implementation-relation-foundation.md` | 已完成Q32.I的fresh baseline、source implementation external model、真实reciprocal/division actual-clone纵向、MLIR-backed IndexRelation foundation和custom interface盘点/首轮删除记录 |
| `tasks/archive/static-memory-packing.md` | 已完成Q34的MiniMalloc默认fixed-capacity packing、精确conflict适配、宽松确定性work budget、typed outcome/fallback和SPM/DDR/7B纵向gate记录 |
| `tasks/archive/llama-block-numeric-characterization.md` | 已完成Q31的ProgramTensor逐rank abs/ULP统计、非verification多seed 7B重放及source/model comparator gate收紧记录 |
| `tasks/archive/llama-block-production-performance.md` | 已完成Q30的static movement/physical codec host性能收口、package等价性和完整7B双replay记录 |
| `tasks/archive/llama-7b-block-vertical.md` | 已完成Q28的标准Llama-2 7B单block TP16 source/package、repo-owned SystemC managed-reference和完整PyTorch eager output differential记录 |
| `tasks/archive/tile-dataflow-scheduling.md` | 已完成Q29历史structured tensor program直达bounded task/dataflow scheduling、当时的跨region SPM合同、card结果写入、旧group executable surface退役及7B TP16 compile-only验证记录；current region语义已由07及Q49.P/Q50/Q51边界替代 |
| `tasks/archive/reference-executor-retirement.md` | 已完成Q27的accepted-IR第二套解释器、oracle分支和旧CLI退役，以及CPU-expected到target CModel纵向gate收敛记录 |
| `tasks/archive/memory-lifetime-analysis.md` | 已完成Q26的共享structured lifetime/packing core、DDR issue-to-fence completion、两侧scope/source relation、offset一次性写入和双配置gate记录 |
| `tasks/archive/residual-source-modularity.md` | 已完成Q25的reference/model、numeric/bulk、compiler/output/package与frontend bridge共11个聚合实现模块化和双配置gate记录 |
| `tasks/archive/remaining-source-modularity.md` | 已完成Q24的group/candidate、target LLVM、numeric conformance、frontend program与compiler driver模块化和双配置gate记录 |
| `tasks/archive/source-organization-refactor.md` | 已完成Q23的instruction、tile-region到instruction、target numeric源码模块化和build/test组织gate记录 |
| `tasks/archive/third-party-dependency-root-consistency.md` | 已完成SystemC canonical third-party root、existing cache切换和双配置重放记录 |
| `tasks/archive/target-model-completion-audit.md` | 已完成Q22各分项字段、legality、runtime evidence、late-rank原子性和双配置全量证据复核记录 |
| `tasks/archive/target-model-source-verticals.md` | 已完成Q22.V的same-lowering product、typed source/model invocation、formal/exact-admitted bulk dispatch和五个固定source vertical实施记录 |
| `tasks/archive/target-bulk-qualification.md` | 已完成Q22.B的受管oneDNN、target-owned adapter、三阶段资格producer和runtime exact-match verification实施记录 |
| `tasks/archive/target-llvm-module-bundle.md` | 已完成Q22.L的all-rank LLVM module生命周期、typed readback和single-lowering device-link接入记录 |
| `tasks/archive/target-numeric-foundation.md` | 已完成Q22.N的typed numeric schema、受管formal依赖、13-format codec和formal kernel实施记录 |
| `tasks/archive/target-call-functional-frontend.md` | 已完成Q22.H的owner-safe host JIT、typed target-call registry/decoder和atomic sink实施记录 |
| `tasks/archive/target-model-readiness.md` | Q21 reduce、numeric/SystemC依赖与host-CRT seam implementation-readiness实证 |
| `tasks/archive/01-design-docs-gap-review.md` | 历史设计缺口审计 |
| `tasks/archive/02-source-organization-recovery.md` | 历史源码组织恢复记录 |
| `tasks/archive/03-dependency-layering-recovery.md` | 历史依赖分层恢复记录 |
| `tasks/archive/04-p0-p6-design-conformance-audit.md` | 历史 conformance audit |
| `tasks/archive/05-p0-p6-recovery-status.md` | 历史 recovery status |
| `tasks/archive/06-r2-recovery.md` | 历史 recovery 记录 |
| `tasks/archive/07-candidate-selection-task-design.md` | 已收口的 candidate-selection 任务记录 |
| `tasks/archive/08-committed-candidate-materialization-task-design.md` | 已收口的selected candidate materialization任务记录 |
| `tasks/archive/09-system-design-implementation-review.md` | 2026-07-10 系统设计与实现审计；只作风险和整改依据，不是架构合同 |
| `tasks/archive/10-target-crt-closure-plan.md` | 已完成并被当前路线替代的 CRT closure 实施记录 |
| `tasks/archive/11-target-crt-conformance-plan.md` | 已完成并被当前路线替代的 CRT conformance 实施记录 |
| `tasks/archive/target-command-legality-closure.md` | 已完成Q0.L的typed target format legality、map/reduce lowering和fresh source replay实施记录 |
| `tasks/archive/12-architecture-evidence-reset.md` | 2026-07-12架构事实重基线审计；只作证据和整改依据 |
| `tasks/archive/single-card-vertical-slice.md` | 已完成的单卡纵向切片实施计划；只保留历史checkpoint和验证记录 |
| `tasks/archive/2026-07-10-long-horizon-plans/` | 已被重基线取代的7份生成式长周期计划；non-normative |
