# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 `tasks/progress.md` 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序只读取`tasks/progress.md`中的状态和“必须满足的前置”；该列可为审计重复列出关键传递gate，不等同最小DAG边集。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | compiler stack主架构：production pipeline、IR/artifact DAG、跨层不变量、consumer分支和owner索引 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | 当前StableHLO program directory与frontend验证；typed state是后续扩展 |
| 03 | `tasks/03-shardy-spmd.md` | 当前Shardy/XLA SPMD artifact、显式rank identity；MPMD/rank class延后 |
| 04 | `tasks/04-topology-execution-mesh.md` | 当前topology/execution mesh；current target identity不进入mesh IR |
| 05 | `tasks/05-local-compute-normalization.md` | rank-local structured compute normalization与tensor collective handoff |
| 06 | `tasks/06-physical-dataflow-synthesis.md` | MLIR-native physical-dataflow synthesis：region partition、tile/loop、implementation/encoding/route/residency/spill/recompute/buffering/order/communication的bounded actual-clone selection与atomic commit |
| 07 | `tasks/07-tile-region.md` | selected tile/dataflow actual IR物化；一个或多个non-nested `tile.region`表达SPM residency domains，data边variadic DDR、SPM root不跨界，boundary不自动产生movement或join |
| 08 | `tasks/08-physical-realization.md` | physical encoding attr/type语义、valid domain、view、transfer realizability analysis、descriptor cover和selected physical realization |
| 09 | `tasks/09-spm-memory-planning.md` | SPM lifetime/coexistence、fixed-capacity MiniMalloc legality、all-root coverage、validated placement/headroom和accepted offsets；candidate choice仍由06拥有 |
| 10 | `tasks/10-compute-movement.md` | 窄source implementation OpInterface/external model、typed target-abstract compute/movement、standard MLIR effects/interface reuse和issue/token/fence/wait |
| 11 | `tasks/11-instruction-ir.md` | complete static rank instruction program、current descriptor/geometry/range/narrowing legality及mapped/physical-fill/oriented typed extension |
| 12 | `tasks/12-ddr-memory-planning.md` | 当前DDR demand/accepted offsets；multi-arena/state/streaming延后 |
| 13 | `tasks/13-communication.md` | logical collective lowering到typed p2p/staging/token/wait IR、Direct DTE all-rank acceptance和completion；segmented/multi-card延后 |
| 14 | `tasks/14-target-conversion-module-publication.md` | current target identity/format registry、structure-preserving target conversion、CRT ABI和atomic module publication；Q32.V扩展从typed winner rows派生 |
| 15 | `tasks/15-launch-runtime-package.md` | typed C++ manifest/PackageBundle、当前schema、canonical JSON、no-card RuntimeSession和board adapter边界；Q32.V仅在真实consumer需要时升级schema |
| 16 | `tasks/16-verification-contract.md` | 跨stage verification contract：target correctness、1/16-rank bundle、CPU oracle、target-model、scale、no-card和board分层gate |
| 17 | `tasks/17-target-execution-model.md` | multi-dtype numeric、oneDNN bulk、same-lowering target LLVM bundle消费、repo-owned target-call/SystemC untimed CModel、7B managed-reference scale、optional CRT/packet provenance、Q22.C板端numeric correlation、Q22.E exact-module和deferred Q22.P timing边界 |
| 18 | `tasks/18-source-organization.md` | 跨pipeline的源码ownership、translation unit、内部接口、构建依赖和测试镜像组织合同；不改变IR/artifact语义 |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program和当前program directory | 02 |
| pre-SPMD topology和execution mesh | 04 |
| Shardy/SPMD output和显式rank identity | 03 |
| component/rank-local compute normalization与collective handoff | 05；跨stage正确性证据由16约束 |
| complete-rank physical-dataflow bounded joint candidate生成/选择、all-rank协调、selected tile/dataflow materialization和physical encoding/transfer | 06、07、08；source implementation interface由10提供，instruction legality由11提供，exact resource/transport gate由09、12、13提供 |
| policy-free physical-dataflow rewrites | 06、07、08；upstream structured utility由05提供，source/selected implementation合同由10提供，源码ownership由18约束 |
| source implementation interface、target-abstract compute/movement和instruction legality | 10、11；transfer realizability/descriptor cover只由08拥有 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12；shared lifetime analysis的源码ownership和测试镜像由18约束 |
| logical collective direct/ring/tree candidate materialization、Direct DTE completion、all-rank acceptance与post-memory transport activation | 13；joint choice由06、target/package/verification consumer由14、15、16约束 |
| whole-rank/whole-variant candidate commit和typed executable bundle | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、CRT/device link和staged target module | 14 |
| typed manifest、launch和RuntimeSession | 15 |
| 横跨上述边界的completion evidence | 16 |
| target execution model、multi-dtype numeric/bulk、same-lowering bundle消费、SystemC/CModel capability、板端numeric correlation和deferred timing | 17；`TargetLLVMModuleBundle`的形成与publication合同由14拥有，target/runtime/verification consumer由14、15、16约束 |
| 跨上述边界的源码与构建模块化 | 18；各IR/artifact语义仍由01-17拥有 |

## 实施计划导航

Q46 layout movement elimination当前计划见`tasks/plans/layout-movement-elimination.md`；它复用06-08、10-11、13-14、
16-17的现有合同，动态状态和完成门禁只看`tasks/progress.md`。

Q47 Target ABI退役计划见`tasks/plans/target-abi-retirement.md`。它已在Q46 compiler-side closure后按独立ABI迁移窗口，
沿11、14-17的owner边界把TX81 target收口为唯一current ABI；它不包含SMT、候选生成或优化器改造，动态状态和
完成门禁只看`tasks/progress.md`。

Q49 whole-rank tile dataflow synthesis计划见`tasks/plans/whole-rank-tile-dataflow-synthesis.md`。它由06作为唯一
联合决策设计owner，复用01、07-13、16、18的selected IR、physical realization、memory、completion、communication和验证合同；
目标是在Instr lowering前联合搜索SPM-residency region partition、tiling/residency/materialization/communication，并让candidate generation与executable-finalization exact gate服从同一个all-rank
coordinator/global ledger，同时删除旧per-task提前物化路径。动态状态和完成门禁只看`tasks/progress.md`。

Q48语义驱动superoptimizer计划见`tasks/plans/semantic-superoptimization.md`。它必须在Q49达到`board-ready`且C0–C6
compiler cutover完成、Q47 current ABI可消费final Instr/TargetCall后启动，
复用05-08、10-11、16-18的现有IR、candidate、proof consumer、model和源码ownership合同；目标是自动生成并证明
actual MLIR clones，同时删除旧implementation抽象和重复numeric表示，不另建语义IR/interface/sidecar。动态状态和
完成门禁只看`tasks/progress.md`。

Q9 profiler foundation已按`tasks/archive/board-profiler.md`完成：唯一public入口是
`wafer-compile --profile`；profile transaction只发布一个未插桩Primary production artifact，最后写入的activation把该
production manifest与companion metadata exact-hash绑定，不另编一个关闭profile的ordinary package做逐字节对照。
`wafer-run`复用既有resource/expected/output binding，在一个qualified session内固定执行一次
未插桩Primary、一次Count和一次Trace；Primary主延迟是production launch前后同一TX stream event pair的
launch-to-completion设备包络，host submit与host launch→trusted-completion只作独立诊断。Trace header同时提供
entry-local span、五类NCC engine aggregate PMU和16-tile typed engine/DTE event。
tile clock未资格化时只展示16行entry-local timeline，不声称跨tile顺序。它尚不回写candidate cost；动态状态只看
`tasks/progress.md`。report publication体积收口另列Q9.R later；它不改变Q9采集协议已经完成的结论。

最新完成的compiler throughput收口为Q32.C，记录见
`tasks/archive/whole-variant-search-throughput.md`。它保持candidate domain、hard cap、exact gate和winner语义，
删除passing ordinal之后的无消费者评估、逐batch线程/context churn、accepted module二次lowering、不可达owner
parse及不会进入fully-gated Pareto frontier的ABI/LLVM lowering；性能记录只含优化后Release实测，不重跑旧二进制。
Q32.N numeric algebraic extension实施计划见
`tasks/plans/numeric-algebraic-extension.md`。它直接删除physical-dataflow algebraic、reduction/GEMM切分和
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
验证/删除门槛与历史证据；动态blocked-by只看progress，算法与IR合同仍由01、05、06-18编号设计文档拥有。

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
| `tasks/archive/whole-variant-search-throughput.md` | 已完成Q32.C的passing-ordinal early stop、bounded persistent candidate executor、accepted-module owner import、exact attempt-plan selective parse和fully-gated Pareto前置late ABI/LLVM，并记录优化后Release单次实测 |
| `tasks/archive/k-sharded-gemm-board-vertical.md` | 已完成Q35 full-4096 f16 K-sharded GEMM的production tiling/SPM/Direct-DTE package、纯tiling隔离及16-rank重复板端raw-exact记录 |
| `tasks/archive/runtime-board.md` | 已完成Q6.B的typed TX board provider、kernel/model多tile launch、cluster Direct DTE、failure lifecycle及真实板端重复exact记录 |
| `tasks/archive/physical-dataflow-synthesis-completion-audit.md` | 已完成Q32的七checkpoint证据映射、双配置全量门禁、fixed/held-out 7B scale重放、单一production owner及剩余边界审计 |
| `tasks/archive/physical-dataflow-synthesis.md` | 已完成Q32 MLIR-native bounded physical-dataflow synthesis的施工checkpoint、hard-cap与integrated completion checklist |
| `tasks/archive/physical-dataflow-production-cutover.md` | 已完成Q32.G默认production winner cutover、旧decision surface删除、all-rank双键correspondence及source/bulk数值回归记录 |
| `tasks/archive/bounded-joint-physical-dataflow-selection.md` | 已完成Q32.S的actual-clone有界联合frontier、reserved baseline、validated whole-card exact cost、target static policy及逐producer whole-winner记录 |
| `tasks/archive/physical-mechanism-choice-closure.md` | 已完成Q32.M的actual-clone recompute/LICM/integer algebra、partial fanout、spill/resident/ready-order、communication alternatives及重复layout/resource/collective/instruction合同删除记录 |
| `tasks/archive/physical-relation-realization.md` | 已完成Q32.R的rich IndexRelation、physical encoding interface、TransferRealizability、destination-style load、relation-backed resident handoff和fresh 7B TP16数值纵向记录 |
| `tasks/archive/typed-target-capability-vertical.md` | 已完成Q32.V的mapped DMA双端offset、physical-footprint fill、source/Tile/Instr/v2 oriented GEMM及formal/SystemC fresh纵向记录 |
| `tasks/archive/mlir-native-implementation-relation-foundation.md` | 已完成Q32.I的fresh baseline、source implementation external model、真实reciprocal/division actual-clone纵向、MLIR-backed IndexRelation foundation和custom interface盘点/首轮删除记录 |
| `tasks/archive/static-memory-packing.md` | 已完成Q34的MiniMalloc默认fixed-capacity packing、精确conflict适配、宽松确定性work budget、typed outcome/fallback和SPM/DDR/7B纵向gate记录 |
| `tasks/archive/llama-block-numeric-characterization.md` | 已完成Q31的ProgramTensor逐rank abs/ULP统计、非admission多seed 7B重放及source/model comparator gate收紧记录 |
| `tasks/archive/llama-block-production-performance.md` | 已完成Q30的static movement/physical codec host性能收口、package等价性和完整7B双replay记录 |
| `tasks/archive/llama-7b-block-vertical.md` | 已完成Q28的标准Llama-2 7B单block TP16 source/package、repo-owned SystemC managed-reference和完整PyTorch eager output differential记录 |
| `tasks/archive/tile-dataflow-scheduling.md` | 已完成Q29历史structured tensor program直达bounded task/dataflow scheduling、当时的跨region SPM合同、whole-variant commit、旧group executable surface退役及7B TP16 compile-only gate记录；当前region语义已由07/Q49替代 |
| `tasks/archive/reference-executor-retirement.md` | 已完成Q27的accepted-IR第二套解释器、oracle分支和旧CLI退役，以及CPU-expected到target CModel纵向gate收敛记录 |
| `tasks/archive/memory-lifetime-analysis.md` | 已完成Q26的共享structured lifetime/packing core、DDR issue-to-fence completion、两侧scope/provenance/原子commit和双配置gate记录 |
| `tasks/archive/residual-source-modularity.md` | 已完成Q25的reference/model、numeric/bulk、compiler/artifact/package与frontend bridge共11个聚合实现模块化和双配置gate记录 |
| `tasks/archive/remaining-source-modularity.md` | 已完成Q24的group/candidate、target LLVM、numeric conformance、frontend program与compiler driver模块化和双配置gate记录 |
| `tasks/archive/source-organization-refactor.md` | 已完成Q23的instruction、tile-region到instruction、target numeric源码模块化和build/test组织gate记录 |
| `tasks/archive/third-party-dependency-root-consistency.md` | 已完成SystemC canonical third-party root、existing cache切换和双配置重放记录 |
| `tasks/archive/target-model-completion-audit.md` | 已完成Q22各分项字段、legality、runtime evidence、late-rank原子性和双配置全量证据复核记录 |
| `tasks/archive/target-model-source-verticals.md` | 已完成Q22.V的same-lowering product、typed source/model invocation、formal/exact-admitted bulk dispatch和五个固定source vertical实施记录 |
| `tasks/archive/target-bulk-qualification.md` | 已完成Q22.B的受管oneDNN、target-owned adapter、三阶段资格producer和runtime exact-match admission实施记录 |
| `tasks/archive/target-llvm-module-bundle.md` | 已完成Q22.L的owner-backed all-rank LLVM module、typed readback和single-lowering device-link接入记录 |
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
| `tasks/archive/08-committed-candidate-materialization-task-design.md` | 已收口的 committed-materialization 任务记录 |
| `tasks/archive/09-system-design-implementation-review.md` | 2026-07-10 系统设计与实现审计；只作风险和整改依据，不是架构合同 |
| `tasks/archive/10-target-crt-closure-plan.md` | 已完成并被当前路线替代的 CRT closure 实施记录 |
| `tasks/archive/11-target-crt-conformance-plan.md` | 已完成并被当前路线替代的 CRT conformance 实施记录 |
| `tasks/archive/target-command-legality-closure.md` | 已完成Q0.L的typed target format legality、map/reduce lowering和fresh source replay实施记录 |
| `tasks/archive/12-architecture-evidence-reset.md` | 2026-07-12架构事实重基线审计；只作证据和整改依据 |
| `tasks/archive/single-card-vertical-slice.md` | 已完成的单卡纵向切片实施计划；只保留历史checkpoint和验证记录 |
| `tasks/archive/2026-07-10-long-horizon-plans/` | 已被重基线取代的7份生成式长周期计划；non-normative |
