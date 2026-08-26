# Physical Dataflow Search 当前实施计划

状态：Q52 `search-scalability`正在执行，Q53 `production-host-readiness`等待Q52。稳定设计只看
`tasks/06-physical-dataflow-synthesis.md`及其直接下游编号文档；动态状态只看`tasks/progress.md`。
Q49–Q51和Q52已完成checkpoint的详细证据在`tasks/archive/physical-dataflow-synthesis-working-history.md`，
不再复制到current plan。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current产品入口产生并验证的card-local TensorProgram；同一只读source artifact分别进入none和search独立事务。
- Current stage responsibility:
  保留已完成的verifier清理和两条policy-owned structural materializer；删除在actual IR之前发明physical value、
  movement、storage、event、execution structure和schedule的shadow search chain。Search只保存显式choice，在结构
  choice后立即物化actual TileRegion IR，之后layout、movement、bufferization、Instr、completion和memory均读current IR。
- Output IR / files:
  none和search各自产生policy-complete、verifier-valid Instr IR，随后经同一actual SPM/DDR/transport/target leaf
  形成CardExecutable与ExecutablePackage。
- Downstream consumer:
  target conversion、device link、package emission和runtime launch。
- User-level driver / named pipeline:
  wafer-compile的typed `none`与`search`入口；focused测试使用注册named pipeline或同一compiler API。
- Explicit non-goals:
  不新建future-output IR、shadow candidate schema、expected-inventory builder、plan/actual parity verifier或兼容双路径；
  不以footprint/shape/推算值决定SPM legality；不猜join/wait；不修改数值语义；本计划不运行真实设备。
- Completion criteria:
  下表的13个checkpoint按线性顺序通过；current production不再有future SSA/buffer/event/schedule的跨stage owner；
  同一current FP16 LLaMA block的none和search分别在15分钟Release门限内生成package并通过strict readback/no-card，
  且两者实际进入Instr、MiniMalloc、DDR和target。
```

## 不可变边界

### Current IR是唯一事实源

```text
current IR
  -> typed choice
  -> candidate-owned transformation
  -> verifier
  -> fresh analysis
  -> next direct consumer
```

- Search key只保存spatial/region/temporal及当前transformation尚未消费的显式choice。
- Operation、SSA、buffer、alias、movement、lifetime、event、order和completion必须先存在于candidate-owned IR。
- Analysis只从current IR和显式target configuration重算；IR mutation后默认失效。
- Cost estimate只能排序choice，不代签actual op/buffer/Instr inventory、legality或memory结论。
- Failed candidate owner整体销毁；Accepted owner不rematerialize、不重建offset。

### Baseline与search是两套实现

| 边界 | Baseline / `none` | Search / `search` |
| --- | --- | --- |
| controller | `compileCardBaseline` | `runUnifiedSearch` |
| structural choice | deterministic single-root region与temporal successor | bounded spatial/region/temporal exploration |
| actual IR owner | baseline materializer | search structural materializer |
| downstream | 只消费baseline current IR | 只消费当前search candidate IR |
| feedback | 仅actual SPM capacity rejection生成smaller temporal choice | typed actual result返回frontier/controller |
| failure | 不启动search | 不fallback baseline |

两条policy可共享source analysis、IndexRelation、single-op rewrite/conversion和actual memory/target leaf，但不共享
controller、Card/Tile materializer、candidate owner、fallback或accepted result。

### SPM与completion

- SPM legality只由current Instr的actual allocation、alias/effect/completion/lifetime和唯一MiniMalloc决定。
- BodyEmitter只报告本次actual buffer relation，不推断owner或持有跨scope relation。
- Completion只从current Instr的effect/token/control-flow/lifetime和已证hardware/ABI事实fresh构造。
- Unknown保持typed unknown；不插入固定worker join、结构化join、全worker drain或issue后立即await。

### 测试规模

Production-style positive默认rank至少为3、主要迭代维至少1024；partition、tiling、loop和memory链成对覆盖
1024及1025/1031，实际经过多Tile、多block/wave、remainder和tail。Tiny只用于独立oracle、最小负例或
scalar/zero-rank，并必须有真实规模对应项。

## 已完成输出的当前用途

Q52前4项的详细证据已归档，current实施只消费下列输出：

| 已完成项 | 保留输出 | 不代签 |
| --- | --- | --- |
| expansion evidence | stage inventory、双materializer和IR膨胀根因证据 | 新materialization boundary正确 |
| verifier cleanup | local op verifier、container/stage check和verify-each边界 | 任何shadow plan/plan replay的正确性或其代签current IR的合法性 |
| search execution materializer | generic/attention的唯一execution construction donor | 物理值、movement、storage或schedule跨stage合同 |
| baseline root materializer | baseline-owned single-root region与actual feedback loop | search路径或Q52最终验收 |

## Q52线性实施顺序

任一项失败留在本项修复，不跳过、不fallback旧路径或另一policy。每项都重复完整工作流程。

| 顺序 | Work item | 状态 | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `expansion-evidence-and-contract` | `done` | 冻结同源两条policy的stage inventory、双路径和膨胀根因 | instrumentation on/off结果一致，违规caller和删除边界可静态复核 | 读AGENTS/progress→读06/Q52与本项矩阵→调研stage instrumentation→查官方及pinned MLIR→改诊断/tests→fresh验证→重读设计并按MLIR复审→更新并提交 |
| 2 | `verifier-contract-cleanup` | `done` | 将operation、container和materialized-stage检查收回正确owner | local verifier不跨scope，必要负例仍在直接stage失败，无总体parity verifier | 读AGENTS/progress→读19/Q52与本项矩阵→调研LLVM/MLIR verifier实践→查pinned源码→改代码/tests→fresh verify-each/build→重读设计/MLIR复审→更新并提交 |
| 3 | `search-execution-materializer` | `done` | search generic/attention使用一个execution-owned construction donor | canonical/noncanonical不选另一builder，producer compute不随fanout重复 | 读AGENTS/progress→读06/07/10与本项矩阵→调研MLIR/IREE/Triton materialization→查pinned API→改代码/tests→fresh Card/Instr验证→重读设计/MLIR复审→更新并提交 |
| 4 | `baseline-root-materializer` | `done` | baseline独立实root构造region并运行actual feedback | 每compute region恰一root，baseline call graph不进入search owner | 读AGENTS/progress→读06/07/10与本项矩阵→读DDR/completion事实→调研baseline materialization→查pinned API→改代码/tests→fresh Card/Instr/SPM验证→重读设计/MLIR复审→更新并提交 |
| 5 | `current-ir-contract-reset` | `done` | 重写active设计和施工顺序，冻结shadow owner删除清单 | AGENTS、01、05–19、Q52 plan/progress和memory只保留一套actual-IR合同；archive只作历史 | 读AGENTS/progress→读01/05–19/Q52与本项矩阵→调研VPlan/GlobalISel/Transform/Bufferization取舍→查pinned文档/源码→改current docs→文本检查→重读设计/MLIR复审→更新并提交 |
| 6 | `structural-candidate-boundary` | `pending` | spatial/region/temporal choice立即物化candidate-owned actual Card/TileRegion IR | 下游只接收actual SSA/ops；pre-structural state无future value/buffer/event；结构candidate verify-each通过 | 读AGENTS/progress→读05–08/10及本项矩阵→调研MLIR Transform alternatives/VPlan边界→查pinned Rewriter/IRMapping API→改代码/tests→fresh TileRegion验证→重读设计/MLIR复审→更新并提交 |
| 7 | `current-ir-layout-bufferization` | `pending` | 在current TileRegion SSA上完成layout assignment、view/alias、conversion和bufferization | 无PhysicalVersion production owner；shared conversion是一个SSA result；alias零copy/allocation；partial copy保留 | 读AGENTS/progress→读06/08–10及本项矩阵→调研One-Shot Bufferize/layout propagation→查pinned `BufferizableOpInterface`→改代码/tests→fresh Tile/Instr验证→重读设计/MLIR复审→更新并提交 |
| 8 | `current-ir-movement` | `pending` | 仅从current producer/use/relation生成local、DDR、peer/collective movement | compatible local edge零DDR；movement不创建compute；无future version或donor scan；all-and-only payload/effect闭合 | 读AGENTS/progress→读06/08/10/13及本项矩阵→读transport硬件事实→调研MLIR data movement/fanout→查pinned API→改代码/tests→fresh movement/Instr验证→重读设计/硬件/MLIR复审→更新并提交 |
| 9 | `current-instr-schedule-completion` | `pending` | TileRegion-to-Instr后从current Instr重建event/dependence，应用worker/order并fresh构造completion | EventGraph不跨mutation；无EventId→op反查或ClosedSchedulePlan；join/wait有actual effect/token/lifetime和hardware witness | 读AGENTS/progress→读06/10/11/13及本项矩阵→读NCC/DTE硬件/ABI事实→调研LLVM MachineScheduler/MLIR async→查pinned API→改代码/tests→fresh Instr验证→重读设计/硬件/MLIR复审→更新并提交 |
| 10 | `actual-memory-feedback` | `pending` | 从同一current Instr派生actual allocation/lifetime demand，运行唯一SPM/DDR/transport/target gate并返回typed feedback | MiniMalloc实际运行；capacity witness具current owner/conflict；其它failure不伪装capacity；accepted owner不重建 | 读AGENTS/progress→读06/09/11–14及本项矩阵→读memory/resource硬件事实→调研actual allocation/liveness实现→查pinned MLIR→改代码/tests→fresh Instr/SPM/DDR/target验证→重读设计/MLIR复审→更新并提交 |
| 11 | `shadow-path-retirement` | `pending` | 删除representation→movement→storage→structure→schedule旧search state、rebuild/parity和相关fixture/doc | 旧type/builder/domain/state/caller/CMake/test/current-doc零production残留；能力已映射到actual-IR owner和direct test | 读AGENTS/progress→读Q52删除清单/本项矩阵→逐项对照old capability/new owner/test→调研API retirement/MLIR ownership→查pinned API→同批切换删除→fresh build/static/downstream验证→重读设计/diff复审→更新并提交 |
| 12 | `scale-regression-and-inventory` | `pending` | 在新actual-IR pipeline上profile并仅保留有证据的memo/priority/DP/LNS，补齐融合与Instr只读汇总 | 完整Tile/TileRegion/buffer/movement/Instr/target inventory；instrumentation/safe optimization on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读06/Q52及本项矩阵→读resource事实→调研search scalability算法→查pinned MLIR→改代码/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 13 | `llama-baseline-search-acceptance` | `pending` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；均实际进入Instr/MiniMalloc/DDR/target | 读AGENTS/progress→重读06/Q52/Q53及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

## Q52逐项覆盖矩阵

### 1. Expansion evidence

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| instrumentation off/on | generic chain/fanout及同一LLaMA source；rank 3–6，1024/1025/1031 | report sink失败不改compiler result | source identity、stage reachability、op/relation inventory和failure类别一致 | 后续checkpoint的冻结输入 |

### 2. Verifier cleanup

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| local op/container/stage | Card/Tile/collective/TileRegion正负例；1024/1025 actual Card | malformed local IR由op verifier拒绝，跨op错误由stage check拒绝 | verifier不读parent sibling或重放未来IR | named/driver `verify-each`与actual memory/ABI gate |

### 3. Search execution materializer donor

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| generic/attention execution | single/nested/replica/coupled、fanout；rank 3–6，1024/1025/1031 | selected structural recipe不可表达时typed failure | 每execution all-and-only一次，fanout不重复compute | 第6项actual TileRegion producer |

### 4. Baseline root materializer

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| root isolation与actual feedback | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031，16 Tile | 只capacity rejection推进，其它typed状态停止 | 每region一root；candidate CardModule数等于actual planner数；accepted不重建 | baseline Instr/MiniMalloc/package |

### 5. Current-IR contract reset

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| active docs/code call graph | 01、05–13、Q52 plan/progress、memory；baseline/search两条路径 | 一个旧plan能力尚无actual owner时保留为待迁移，不伪装已删 | 每个字段分类为choice/current fact；active文档无正向shadow合同；删除清单有new owner/test | 第6–11项的唯一边界 |

### 6. Structural candidate boundary

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| spatial/region/temporal choice | chain/fanout/reduction/attention；rank 3–6，1024/1025/1031，16 Tile | rewrite前不可表达为silenceable/typed failure；mutation后失败擦除transaction | actual TileRegion/loop/SSA all-and-only；无future value/buffer/event state；winner不重建 | current-IR layout/movement transforms |

### 7. Current-IR layout and bufferization

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| primary/shared/alias/use-local需求 | 1/2/15 uses，reshape/transpose，Tensor/NTensor/Cx/NCx，rank 3–6，1024/1025/1031 | relation或bufferization不能证明时保留copy或typed unsupported | shared layout一个SSA definition；alias同storage；无unused conversion；actual allocation/effect完整 | movement transform与TileRegion-to-Instr |
| full/partial transfer | full equivalent与partial/permuted/layout-changing | 不满足exact条件不删 | 只删除current IR上证明冗余的full transfer | actual movement count与payload coverage |

### 8. Current-IR movement

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| local/DDR/peer/relay/gather | equal/partial window、fanout/fanin、reduction、attention；1024/1025/1031 | alias/effect/route不明时typed unknown/unsupported，不猜近似carrier | compatible local edge零DDR；compute数不随fragment增长；actual movement覆盖all-and-only | Instr movement/effect/token |

### 9. Current Instr schedule and completion

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| compute/DDR/DTE/control flow | straight-line、loop/tail、cross-worker、observable terminal；1024/1025/1031 | missing hardware/ABI语义为typed unknown；token/effect malformed为IR failure | event graph从current Instr fresh重算；应用order后失效；minimum/latest join/wait有直接witness | actual lifetime与MiniMalloc输入 |

### 10. Actual memory feedback

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| actual allocation/lifetime | baseline/search；aligned/ragged；1024/1025/1031；16 Tile | capacity、ResourceExhausted、timeout、unsupported、compiler error保持区分 | 具体化→Instr→completion→MiniMalloc链路；offset/conflict/owner来自current IR；删除估算不改legality set | DDR/transport/target与controller feedback |

### 11. Shadow path retirement

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| source/CMake/caller/test/doc inventory | generic/attention、none/search、旧fixture | 仍有独有能力没有new owner时停止删除 | 旧production symbol/caller/CMake/current-doc为0；无fallback或compatibility path | fresh build、named/driver、actual memory/target |

### 12. Scale regression and inventory

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3–6，1024/1025/1031，16 Tile | stage未到达与typed compiler/resource failure分开 | TileRegion/structured-execution/op分布、local/external use、actual buffer/movement、final Instr total/per-Tile/per-kind；compute duplicate为0 | actual MiniMalloc、package strict readback |
| search optimization | tiny exhaustive oracle与真实规模profile | timeout/resource不伪装exact rejection | safe optimization/instrumentation on/off保持result；有损策略只报BudgetedFeasible | retained actual winner一次publication |

### 13. LLaMA acceptance

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output directory | timeout/OOM/skip/fallback/未进actual planner均失败 | 每次≤15分钟；source identity相同；IR/result不共享；package唯一 | strict loader、host reference、no-card |

## Q52旧路径删除清单

下列对象不能在终态继续作为production cross-stage protocol。可复用的算法或negative test先迁到actual-IR owner，
然后在同一work item删除旧owner：

- `RepresentationState`、`PhysicalVersionId/Plan`、`RepresentationDomain`和production `PhysicalVersionBuilder`；
- 以future physical value为source/destination的`MovementState/MovementPlan`和canonical movement replay；
- `InitialBufferState`、`BufferState`、pre-IR `StorageObjectId`、`StorageDomain`、`StructureSpecificStorageDomain`和
  canonical storage/lifetime replay；
- `ExecutionStructureState/Plan`作为cross-stage candidate state的路径；actual Instr上的一次性scheduler analysis可保留；
- `ScheduledState`、`ClosedSchedulePlan`、`EventId -> operation`物化后反查和completion parity；
- 携带上述shadow facts的`CompleteCandidateKey/Plan`、`FullFeasibility` domain rebuild和`CompleteCandidatePreparation`；
- donor movement扫描/替换/erase、edge-owned compute重建、first-use layout恢复、无producer action surface和only-purpose fixtures；
- active设计、memory、CMake和tests中把这些对象当作current事实源或supported production contract的内容。

只在一次rewrite内使用的typed choice、`IRMapping`、SSA lookup或query-local event/lifetime graph不在该删除范围，但必须在
IR mutation后失效，不能成为下一stage的事实源。

## Search scalability

只有第11项完成后才重新解釈profile和优化search工作：

- memo只保存从immutable source IR或current candidate IR重算的pure typed query result，不保存IR owner、operation pointer、offset或actual result；
- component DP只在current IR可证明separator关闭时使用，无证明时回到base traversal；
- priority、memo和safe bound只改变choice访问顺序或重复工作；fixed beam/Top-k/LNS-only等有损策略只能返回
  `BudgetedFeasible`；
- LNS只在已有actual Accepted incumbent后启动，每个repair alternative仍必须物化为actual candidate并运行普通gate；
- 对accepted candidate记录TileRegion、buffer、movement、Instr、MiniMalloc/DDR/target工作和publication数，不记录预测IR inventory。

## Q53 Production Host Readiness

Q53只消费Q52签发的current none与search路径，不改变search算法、deadline或candidate。一个case只导出一次
immutable source artifact；两种policy分别重新parse/import，并在独立process、work directory、ProgramData owner和
output directory中完成编译。Source identity只证明输入一致，不授权共享Module、analysis、IR、CardExecutable或package。

### Q53 work item

| Work item | 直接输入 | 完成输出 | 固定执行流程 |
| --- | --- | --- | --- |
| `production-host-readiness` | Q52、Q60产品入口、Q55 current interface、Q56 board-ready package/runtime | fresh source/IR/package/oracle/runner/no-card矩阵和可直接串行上板的case；Q53状态到`board-ready` | 读AGENTS/progress→读02/06/14–16及本项矩阵→读hardware/runtime/ABI事实→调研host qualification与board-ready组织→查官方及pinned API→改runner/tests→fresh host/no-card验证→重读设计并按MLIR/runtime复审→更新并提交 |

### Q53覆盖矩阵

| 输入等价类 | shape/dtype/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| generic structured | chain/diamond/fanout/reduction/mixed movement；rank 3–6，1024/1025/1031；FP16/BF16 | source/IR/resource/package/runtime分类保持 | exact coverage、owner、tail、actual movement/buffer/completion all-and-only | current package strict loader/no-card |
| attention | HF prefill、functional two-step decode；1024/1025/1031；FP16/BF16 | unsupported/resource与compiler failure区分 | fixed FA/FD语义、actual physical IR和step continuation同policy独立 | host oracle、package/no-card和board case binding |
| representative model | current LLaMA block；FP16 mandatory | timeout/OOM/skip/fallback不计通过 | none和search分别fresh生成package；无cross-policy state | strict readback、CPU reference、no-card |
| board-case preparation | representative communication、attention/decode和LLaMA的两种policy | 缺package/input/oracle/guard/deadline即非board-ready | package、payload、all outputs、guard、continuation、timeout和固定串行顺序完整 | 后续board runner无需修改source或临时补oracle |

Q53完成要求registered case实际执行且非skip/unsupported；每个case使用本轮source和package；strict loader验证
canonical manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、
transport和completion；host oracle与guard通过。真实板测只能由Q53之后的显式任务执行。

## 收尾

- 状态只更新`tasks/progress.md`，本计划不维护第二份owner状态。
- 每个checkpoint只提交本项相关代码、测试和文档；旧能力删除前必须有new actual-IR owner和direct test。
- fresh验证必须确认关键lit/CTest实际执行；文档-only改动至少运行链接、旧术语、职责和格式检查。
- 临时profile数字、workload路径、build目录和单case偶发现象不进入稳定设计或memory。
