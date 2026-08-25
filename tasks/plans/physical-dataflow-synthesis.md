# Physical Dataflow Planning 当前实施计划

状态：Q52 `search-scalability`正在执行，Q53 `production-host-readiness`等待Q52。Q49–Q51和Q54的施工、
审计与旧验证数字已归档；它们不是当前实现依据。动态状态只读`tasks/progress.md`，稳定IR和算法合同只读
`tasks/06-physical-dataflow-synthesis.md`及其直接下游编号设计。

历史详细记录见`tasks/archive/physical-dataflow-synthesis-working-history.md`。本计划只保留当前输入、输出、
线性步骤、逐项覆盖矩阵、删除边界和验收门禁。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current产品入口产生并验证的card-local TensorProgram；同一只读source artifact分别进入`none`和`search`
  独立编译事务。IR已经具有structured semantics、SSA、IndexRelation、effect、type、shape和dtype，尚未形成
  policy-complete Instr、SPM offsets或target output。
- Current stage responsibility:
  Q52先清理越界verifier和重复whole-IR验证，再分别修复baseline deterministic materializer与search selected
  materializer；关闭compute重复物化、root grouping、movement、layout/view和IR膨胀，删除旧shared/edge-driven
  路径。结构正确后才基于fresh profile优化search的memo、priority、component DP和LNS。Q53只把Q52结果从
  产品source推进到fresh package、host oracle、strict readback、no-card和完整board case准备，达到board-ready。
- Output IR / files:
  Q52分别产生policy-complete、verifier-legal Instr IR及其current relations，随后经实际SPM/DDR/transport/target
  leaf形成各自CardExecutable与ExecutablePackage。Q53产生本轮source、expected、package、no-card结果和可直接
  串行上板的case定义；不产生板端通过结论。
- Downstream consumer:
  Q52输出由actual PlanSPMMemory/MiniMalloc、DDR、transport、target和package owner消费；Q53输出由后续显式
  board qualification任务消费。
- User-level driver / named pipeline:
  `wafer-compile`的typed `none`与`search`产品入口；focused IR测试使用注册的named pipeline或同一compiler API。
- Explicit non-goals:
  不建立baseline/search共享的complete-candidate schema、preparation或materializer；不新增expected-inventory
  builder、IR膨胀legality、shape assume、全局verifier或第二份物化模型；不以footprint/shape/估算决定SPM合法性；
  不修改数值语义；不猜join/wait；当前计划不运行真实设备或判断板端性能。
- Completion criteria:
  Q52九个checkpoint按本文顺序全部通过，旧complete materializer和无producer surface清零；同一current FP16
  LLaMA block的`none`与`search`分别在15分钟Release门限内生成current package并通过strict readback/no-card，
  且两者实际进入Instr、MiniMalloc、DDR和target。随后Q53从产品入口完成fresh representative矩阵、host oracle、
  package/no-card和board runner准备，状态只到board-ready。
```

## 不可变边界

### Baseline与search是两套实现

| 边界 | Baseline / `none` | Search / `search` |
| --- | --- | --- |
| public owner | `compileCardBaseline`，不创建search session | `runUnifiedSearch`与search planning session，不调用baseline controller |
| planning | 一个live deterministic candidate；只有actual SPM capacity rejection可生成预定义的smaller temporal successor | typed partial/complete states；每个complete key actualize一次，controller比较actual result |
| region | 每个compute region恰一个semantic root；同Tile多root为多个顺序region；跨root shaped dependency显式materialize | selected `RegionPlan`可表达single/multi-root、nested、replica和coupled execution |
| representation/movement | baseline自己的deterministic layout、local/DDR和必要peer carrier | selected `RepresentationPlan`、`PhysicalVersionId`和`MovementPlan` |
| storage/schedule | baseline自己的deterministic buffer、order与completion | selected storage、execution-structure、schedule construction及plan/actual parity |
| materializer/verifier | baseline-owned | search-owned |
| accepted result | 第一个通过actual gate的candidate直接成为结果，无winner comparison | controller只发布一个retained actual winner |
| failure | typed baseline failure，不启动search | typed search failure/coverage，不fallback baseline |

两条路径可以复用source parsing/normalization、只读IndexRelation事实、单operation tiling/layout emission、
TileRegion-to-Instr conversion kernel、MiniMalloc、DDR/transport validation和target lowering。leaf API不能接收policy enum、
baseline/search plan variant、nullable callback或mode来切换语义。两条路径只在各自形成policy-complete、verifier-legal
Instr IR和current relation后汇合到actual memory/target leaf。

### SPM、同步与数值

- SPM合法性只由当前candidate实际Instr、allocation、alias/effect/completion/lifetime和唯一MiniMalloc结果决定；
  footprint、upper bound、buffer倍数、shape公式、synthetic demand和预测lifetime不进入admission或refinement。
- BodyEmitter只向caller-owned recorder报告实际buffer；无owner demand是compiler contract failure，不能按shape、名称、
  Location或root cardinality补猜。
- join/wait只来自current typed effect/token/control-flow/lifetime和硬件/ABI事实。unknown保持typed unknown；不得插固定
  worker join、结构化join、全worker drain或issue后立即await。
- 本任务不改变算术operation、dtype、reassociation或其它数值语义；既有数值case只作回归。

### 测试规模

Production-style positive默认rank至少3，主要迭代维度至少1024；partition、tiling、loop和resource链成对覆盖
1024及1025/1031，实际经过多Tile、多block/wave、remainder和tail。tiny只用于独立有界oracle、最小verifier负例、
scalar/zero-rank或单点定位，并必须有同机制真实规模对应项。shape只属于覆盖矩阵，不进入compiler策略或legality。

## Q52冻结证据与根因

### Search首个constructive candidate

| IR边界或关系 | 数量 | 相对关系 | 结论 |
| --- | ---: | ---: | --- |
| source StableHLO | 144 | 1.000x | 单个decoder block，包含current topology/mesh facts |
| post-SPMD StableHLO | 148 | 1.028x/source | 不是异常膨胀点 |
| Tensor/Linalg | 438 | 2.959x/post-SPMD | 106个structured compute，仍是一份logical program |
| selected execution | 1,696 | `106 * 16` | 当前plan要求的node/Tile execution |
| actual compute emission | 42,672 | 25.160x/selected execution | 同一execution被fragment carrier重复构造 |
| selected Tile dataflow | 560,032 ops | 1,278.612x/Tensor | 膨胀首次完整出现在actual candidate construction |

主要Tile op为`wafer.tile.extract_slice=92,976`、`bufferization.to_memref=57,824`、
`memref.global=57,600`、`memref.alloc=47,184`、`memref.subview=42,384`、`wafer.tile.load=27,184`、
`wafer.tile.materialize_layout=19,472`、`wafer.tile.insert_slice=16,608`和`wafer.tile.transpose=13,088`。
15分钟Release只处理约13%的slice，尚未调用actual SPM planner；timeout不能解释成SPM rejection。

### Baseline独立证据

同一source的fresh `none`在baseline CardModule postcondition失败，未进入TileRegion-to-Instr、SPM、DDR或target。
Tile 0已经形成31,911 ops、2,648个compute emission和6,426个extract slice；一个TileRegion包含15个semantic root，
违反baseline一root一region合同。该路径的`tile_memory_planning_invocations`、`tile_to_instr_lowerings`和
`spm_planning_invocations`均为0。Baseline不使用search plan，但复用了旧edge-driven materializer，因此同时具有
per-fragment compute重复和独立root grouping错误。

### 根因链

1. Search constructive spatial proposal逐node独立最大化Tile participation，没有沿producer/result到consumer/operand的
   exact IndexRelation协调相邻分片，产生大量peer fragment与RegionCut。
2. canonical coordinate进入旧`TileMaterializationSession`，非canonical coordinate进入
   `StructuredNodeShardGroup`，同一search controller存在两套active actual builder。
3. 旧edge carrier通过`getOrMaterializeSource -> materializeCandidateRootTileValue`按fragment反向创建producer
   traversal；cache以exact window为key，不能维护selected execution identity。
4. 重复compute继续生成slice、layout conversion、allocation、global和movement；internal partial window又被lower成
   真实gather/scatter。
5. selected movement仍依赖先生成donor send/recv/load再扫描、替换和erase；layout builder及full-buffer elimination没有
   production caller。
6. post-hoc relation和set-based verifier丢失multiplicity，掩盖了重复construction。

Frontier state数、MiniMalloc、SPM estimate、source模型规模和temporal tile过细均不是本次膨胀起点。修复不能依赖DCE、
descriptor cache、单Tile fallback或延长timeout。

## Verifier职责冻结分类

### Custom verifier hooks

当前inventory为52个`verify()`和3个`verifyRegions()`，共55个hook。施工只按下表处置，不因测试结果临时改变分类。

| Family | 数量 | 处置 |
| --- | ---: | --- |
| DTE message/binding attr | 2 | 保留local value/range/tuple与target-level固有字段范围 |
| `CardModuleOp` | 1 | 保留body、直接child、内部Tile ID和自有symbol table closure；parent topology、sibling card、available Tile/peer和全Tile ABI totality移到module/card stage |
| `TileModuleOp` | 1 | 保留local parent/body/ID，不重建card domain |
| attention op | 1 | 保留operand/result、indexing role、dtype、DPS和algorithm局部合同 |
| 5个collective op及2个combiner region hook | 7 | op保留DPS/shape/attrs/channel；相对execution mesh的全局partition bounds移到module stage；region hook保留local |
| topology/mesh op | 2 | 保留自身grid/shape/coordinate；parent关系优先ODS trait |
| Tile Move op | 7 | 保留source/result type及局部slice/reshape/transpose/broadcast relation |
| Tile Compute op | 7 | 保留operand/result、layout、shape、kind和DPS局部合同 |
| Tile Comm op | 2 | 保留buffer/peer/bytes/token；peer domain移到card stage |
| Tile View/Layout op | 2 | 保留ViewLike/layout局部关系；是否消除copy属于transform/test |
| `TileRegionOp` main/region hook | 2 | main保留outer/non-nested/input/result；递归def-use/SCF/ViewLike storage-root追踪移到bufferization/Instr-memory stage；region hook保留local |
| Storage load/store | 2 | 保留memory space、layout和logical tensor type |
| Instr movement/DTE/sync/peripheral/compute | 19 | 保留Instr可解释的memref、descriptor、field range、worker/token和compute合同 |
| 总计 | 55 | 48个hook保持local；5个collective op、CardModule和TileRegion main verifier拆分 |

### Whole-IR verification

当前有35个显式`mlir::verify`调用点：

| 文件族 | 数量 | 处置 |
| --- | ---: | --- |
| Driver/Frontend/helper边界 | 3 | parse/import/helper返回后的新IR epoch保留 |
| StableHLO normalization/attention | 4 | 保留pass或pipeline最终验证；已删除逐新op后又验证function的重复调用 |
| TensorProgram→Tile/Card | 14 | 旧materializer调用随路径删除；每个最终owned output scope验证一次 |
| selected preparation/schedule/structure | 3 | shared preparation随facade删除；schedule与structure已各自合并为一次稳定输出验证 |
| memory/executable/target | 10 | 真实mutation边界保留；callback删除后再去除同epoch相邻重复 |
| full-buffer transfer rewrite | 1 | 不逐rewrite attempt验证Module；fixed point完成后验证一次owned scope |
| 总计 | 35 | 不预设最终配额，只删除无mutation重复并保留真实stage边界 |

Materialized-plan检查的处置：

- `verifyMaterializedCardCandidate`和shared preparation中的Region/node/loop plan重放随旧facade删除；专项行为留在测试。
- schedule verifier删除builder order/worker全量重放，保留token ownership、pending-at-return和current IR有效性。
- execution-structure verifier把prologue/kernel/epilogue数量公式移入测试，保留temporary attr清理、SCF和直接结果有效性。
- storage inventory、`1+2*(slots-1)`和具体selector链移入builder测试；selected lifetime safety保留在Instr-memory stage。
- typed completion、SPM/DDR range、transport和ABI检查保留在各自actual stage。

## Q52线性实施顺序

下表每项都重复完整工作流程；任一项失败留在本项修复，不跳过、不fallback旧路径或另一policy。

| 顺序 | Work item | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- |
| 1 | `expansion-evidence-and-contract` | 冻结同一source identity、两条路径逐stage inventory、current call graph违规和目标独立materializer边界及删除清单 | instrumentation on/off结果一致；输入inventory一致；未到达stage明确标注；current shared/edge-driven调用点完整列出且第3/4/7项的清零目标可静态复核，不提前把违规写成已修复 | 读AGENTS/progress→读06/Q52与本项矩阵→读涉及硬件/ABI事实→调研成熟compiler的stage/owner instrumentation→查官方及pinned MLIR→改诊断/tests→fresh验证→重读设计并按MLIR规范复审→更新状态并提交 |
| 2 | `verifier-contract-cleanup` | 严格执行55/38/materialized-stage分类 | 分类零遗漏；必要负例仍在正确stage失败；无新增总体验证器 | 读AGENTS/progress→读19/Q52与本项矩阵→读硬件/ABI事实→复核LLVM/MLIR成熟实践→查pinned源码→逐类改代码/tests→fresh verify-each/build→重读设计/MLIR复审→更新并提交 |
| 3 | `search-execution-materializer` | canonical/noncanonical、generic/attention只走search execution-owned builder；carrier只消费SSA result/view | 旧conditional builder和carrier→compute tiler调用为0；fanout不增加producer compute；nested/replica/coupled/tail通过 | 读AGENTS/progress→读06/07/08/10与本项矩阵→调研MLIR/IREE/Triton materialization→查pinned MLIR→改代码/tests→fresh Card/Instr验证→重读设计/MLIR复审→更新并提交 |
| 4 | `baseline-root-materializer` | 独立baseline materializer按root创建region和independent-DDR carrier | 15-root region消失；每个compute region恰一root；baseline call graph不触及search owner | 读AGENTS/progress→读06/07/10与本项矩阵→读hardware completion/DDR事实→调研成熟baseline pipeline→查pinned MLIR→改代码/tests→fresh baseline Card/Instr验证→重读设计/MLIR复审→更新并提交 |
| 5 | `movement-and-spatial-closure` | carrier只消费已有SSA；接通local carrier；search proposal沿exact relation协调相邻partition | fanout只增加必要view/transport；compatible local edge零DDR；proposal不改变raw domain或SPM合法集合 | 读AGENTS/progress→读06/08/10/13与本项矩阵→读transport硬件事实→调研graph-coherent partition/fanout→查MLIR API→改代码/tests→actual Instr/SPM验证→重读设计/硬件/MLIR复审→更新并提交 |
| 6 | `layout-version-and-view-closure` | search production接通PhysicalVersion builder；baseline保留独立layout；view/alias不造copy；full-buffer elimination接入明确stage | production caller存在；shared use一份conversion；metadata view零target call；必要partial copy保留 | 读AGENTS/progress→读06/08/09/10与本项矩阵→调研MLIR/Triton layout propagation/removal→查pinned API→改代码/tests→fresh Instr/SPM验证→重读设计/MLIR复审→更新并提交 |
| 7 | `legacy-path-retirement` | 切换两个独立owner并删除shared facade、旧session、edge-owned compute、donor surgery和无producer action | baseline/search caller各自唯一；旧symbol/source/doc/test/CMake为0；失败无fallback | 读AGENTS/progress→读迁移/删除清单和本项矩阵→逐项对照old capability/new owner/test→调研API retirement→查MLIR ownership→切换删除→fresh build/static/downstream验证→重读设计/diff复审→更新并提交 |
| 8 | `scale-regression-and-q52-rebase` | 用真实规模重新profile，在新IR上只保留有证据的memo/DP/bound/priority/LNS | 完整Tile/Instr/target inventory；MiniMalloc实际运行；safe optimization on/off等价；相关unit/lit/named/driver parity通过 | 读AGENTS/progress→读06/Q52和本项矩阵→读resource事实→调研search scalability算法/实现→查pinned MLIR→实现或删除/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 9 | `llama-baseline-search-acceptance` | 同一current FP16 LLaMA source顺序运行独立`none`和`search`事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；均实际进入Instr/MiniMalloc/DDR/target | 读AGENTS/progress→重读06/Q52/Q53与本项矩阵→确认current source/tool→fresh顺序运行→逐项核对设计与MLIR规范→更新状态并提交 |

### 1. `expansion-evidence-and-contract`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| instrumentation off/on | generic chain/fanout及同一LLaMA source；rank 3–6，1024/1025/1031 | report sink失败不改变compiler result | source identity、stage reachability、op/relation inventory和failure类别一致 | 后续所有checkpoint使用同一冻结基线 |
| controller isolation | `none`和`search`独立process/owner；current shared facade作为冻结的待删违规 | instrumentation不改变现有typed result；漏记active shared/互调caller视为evidence contract failure | call graph、include和CMake owner及违规清单完整；未到达stage不记作0开销 | 第3/4项分别构造两条materializer，第7项静态清零 |

### 2. `verifier-contract-cleanup`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| local op与container closure | Card/Tile/collective/TileRegion正负例；1024/1025 actual Card | malformed local IR由op verifier拒绝；跨op错误由stage check拒绝 | 55个hook逐项有owner；leaf不读取parent/sibling/op外def-use | named/driver `verify-each`与actual memory/ABI gate |
| whole-IR mutation epoch | StableHLO→Tile→Instr→memory→target | 同epoch重复不是failure owner | 38个call site逐项处置，mutation后稳定边界仍验证 | 后续materializer不依赖旧plan replay verifier |

### 3. `search-execution-materializer`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| canonical/noncanonical | generic/attention、single/nested/replica/coupled、rank 3–6、1024/1025/1031 | selected group或fragment不可表达时typed materialization failure | 同一search builder；每个execution all-and-only一次 | search-owned storage/execution-structure/schedule及TileRegion-to-Instr |
| fanout | 1/2/15 destination和local consumer | fragment不属于selected result shard则fail closed | producer compute IR数量与destination数无关 | all-and-only view/send/recv/wait |

### 4. `baseline-root-materializer`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| root isolation | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031，16 Tile | missing/duplicate/multi-root region拒绝 | 每region一structured root；同Tile多root顺序独立；跨root DDR显式 | baseline-owned Instr/completion |
| actual feedback | initial over-capacity到smaller fit及minimum仍失败 | 只有actual capacity rejection推进；其它typed状态停止 | candidate CardModule次数等于actual planner次数，accepted不重建 | MiniMalloc offsets进入baseline result |

### 5. `movement-and-spatial-closure`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| graph-coherent partition | chain、matmul weight/activation、reshape/head、transpose；rank 3–5，1024/1025/1031 | relation无法传播时保留raw sibling，不伪造同分片 | proposal是raw-domain member；peer/RegionCut piece下降但合法集合不变 | MovementPlan读取同一exact demand |
| local/peer carrier | equal/partial window、layout compatible/incompatible、fanout/relay/gather | alias/effect/lifetime不明则保留explicit movement或typed unknown | compatible local edge零DDR；carrier不能调用compute tiler | actual DTE/DDR/completion与SPM lifetime |

### 6. `layout-version-and-view-closure`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| primary/derived/shared/alias | 1/2/15 uses，reshape/transpose，rank 3–6，1024/1025/1031 | physical relation不能证明时保留必要copy或typed unsupported | selected use按PhysicalVersionId绑定；shared conversion一次；alias/view零copy | bufferization、Instr descriptor和MiniMalloc |
| full/partial transfer | equivalent full buffer与partial/non-equivalent window | 不满足等价条件不得删除 | eliminator有production caller；只删除已证明冗余full transfer | target movement count与payload coverage |

### 7. `legacy-path-retirement`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| source/CMake/caller inventory | baseline/search、generic/attention及旧fixture | 仍需能力缺new owner时停止删除 | 每个donor映到new owner和test；旧symbol/include/CMake/test/doc为0 | fresh build、link、named/driver路径 |
| failure routing | 两条policy各自构造失败 | typed failure保持原policy | 无旧path、另一policy或hidden repair fallback | public CLI failure与output原子性 |

### 8. `scale-regression-and-q52-rebase`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| stage expansion | mixed DAG、HF prefill/decode、LLaMA；rank 3–6，1024/1025/1031 | stage未到达和typed compiler/resource failure分开 | compute duplicate为0；metadata view零target call；完整Tile/Instr/target inventory | actual MiniMalloc、package strict readback |
| search optimization | tiny exhaustive oracle及真实规模profile | timeout/resource exhaustion不伪装exact rejection | memo/DP/bound on/off保持exact set/objective/winner/coverage；有损策略只报BudgetedFeasible | retained search winner一次publication |

### 9. `llama-baseline-search-acceptance`覆盖矩阵

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output目录 | timeout/OOM/skip/fallback/未进入actual planner均失败 | 每次≤15分钟；source identity相同；plan/IR/result不共享；各自package唯一 | strict loader、host reference和no-card invocation |

## Q52删除清单与验收

终态至少删除或迁出production：

- shared `materializeCardCandidate(..., mode, ...)`及base/derived preparation callback；
- `usesSelectedRegions`或representation equality选择另一套builder的分支；
- `TileMaterializationSession`、`TileMaterializationSourceSession`和edge-owned complete Card builder；
- carrier侧`getOrMaterializeSource -> materializeCandidateRootTileValue`反向构造compute；
- movement中的`oldPairs`、`oldLoads`和donor扫描/替换/erase；
- post-hoc `appendRequiredExecutionNodeRelations`及按node set恢复execution identity；
- `RootRegionEmitter`从whole-function SSA重新选择root grouping；
- `BufferVersions/lookupAny`按first use或layout enum恢复selected version；
- 没有current producer的`RecursiveProducerTiling`、`SpillReload`和`Recompute`surface及only-purpose tests；
- active设计、memory、CMake和tests中把上述路径描述为current或supported的内容。

冻结LLaMA回归要求：search修复后Tile total、compute和slice相对560,032/42,672/92,976严格下降，execution duplicate为0；
baseline Tile 0相对31,911/2,648/6,426严格下降且15-root region消失。严格下降只作回归证据，不进入compiler
legality。最终必须同时通过专项结构断言、标准MLIR verifier、actual SPM/DDR/target和package/no-card。

## 结构修复后的search scalability

只有checkpoint 1–7关闭后才重新解释profile。普通compile不构造profile对象、字符串或报告。

- IR-derived只读facts由对应operation analysis或planning transaction持有并随IR mutation失效。
- session memo只保存pure typed query result；不保存Operation pointer、IR、offset、solver、diagnostic或actual result。
- memo key先包含query实际读取的完整typed prefix；字段投影必须有dependency perturbation证明。eviction/cache-off只增加work，
  不改变successor、actual result或winner。
- component DP只在separator包含全部cross-component value/version/use、Tile/DDR/DTE/FSM、alias/lifetime、completion、
  resource和observable effect时使用；opaque coupling不能伪装独立。无证明时回到base search traversal。
- symmetric Tile只能共享pure query result，不能共享assignment、actual IR、buffer/event或offset。
- priority、memo、safe bound和proposal只改变访问顺序或重复工作；fixed beam、Top-k、不可恢复eviction或LNS-only属于
  显式有损策略，结果最多为`BudgetedFeasible`。
- LNS只在已有actual Accepted incumbent后启动；destroy集合从typed dependency/resource graph取闭包，repair复用同一typed
  transitions，每个complete repair仍走normal actual gate。没有incumbent时不得让baseline补救。

实施时记录axis/query work、time-to-first、peak frontier/RSS、complete actualization、SPM/DDR调用和publication次数；
`complete candidates == candidate CardModules == actual admission invocations`，partial-state actualization为0，winner不重建。

## Q53 Production Host Readiness

Q53只消费Q52已经签发的current `none`与`search`路径，不改变search算法、deadline或candidate。一个case只导出一次
immutable source artifact；两种policy分别重新parse/import，在独立进程、work directory、ProgramData owner和output directory中
完成编译。source identity用于证明输入一致，不授权共享Module、analysis、plan、CardExecutable或package。

### Q53 work item

| Work item | 直接输入 | 完成输出 | 固定执行流程 |
| --- | --- | --- | --- |
| `production-host-readiness` | Q52、Q60产品入口、Q55 current interface、Q56 board-ready package/runtime | fresh source/IR/package/oracle/runner/no-card矩阵和可直接串行上板的case；Q53状态到`board-ready` | 读AGENTS/progress→读02/06/14–16及本项矩阵→读hardware/runtime/ABI事实→调研成熟compiler的host qualification与board-ready组织→查官方及pinned MLIR/API→改runner/tests→fresh host/no-card验证→重读设计并按MLIR/runtime规范复审→更新状态并提交 |

### Q53覆盖矩阵

| 输入等价类 | shape/dtype/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| generic structured | chain/diamond/fanout/reduction/mixed movement；rank 3–6，1024/1025/1031；FP16/BF16 | source/IR/resource/package/runtime分类保持 | exact coverage、owner、tail、movement、storage和completion all-and-only | current package strict loader/no-card |
| attention | HF prefill、functional two-step decode；1024/1025/1031；FP16/BF16 | unsupported/resource与compiler failure区分 | fixed FA/FD语义、selected physical plan、step continuation同policy独立 | host oracle、package/no-card和board case bindings |
| representative model | current LLaMA block；FP16 mandatory，BF16按current support矩阵 | timeout/OOM/skip不计通过 | `none`和`search`分别fresh生成package；无fallback或cross-policy state | strict readback、CPU reference、no-card |
| board-case preparation | representative communication、attention/decode和LLaMA的两种policy | 缺package/input/oracle/guard/deadline即非board-ready | package、payload、all outputs、guard、continuation、timeout和固定串行顺序完整 | 后续board runner无需修改source或临时补oracle |

Q53完成要求：registered case实际执行且非skip/unsupported；每个case使用本轮source和本轮package；strict loader验证canonical
manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、transport和completion
plan；host oracle与guard通过。当前任务不打开设备、不重放历史raw、不做matched A/B或性能结论。真实板测只能在Q53达到
board-ready后由新的显式任务和本轮qualified session执行，不能从历史Q53-2设想恢复为当前步骤。

## 收尾

- 状态只更新`tasks/progress.md`；本计划不维护第二份状态表。
- 每个checkpoint只提交本项相关代码、测试和文档；旧能力删除前必须有new owner、production caller和direct test。
- fresh验证必须确认关键lit/CTest实际执行；文档-only改动至少运行链接、旧术语、职责和格式检查。
- 临时profile数字、workload路径、build目录和单case偶发现象不进入稳定设计或memory。
