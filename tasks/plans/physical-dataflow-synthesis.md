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
  保留已完成的verifier清理、baseline direct materializer和search execution construction donor；删除在actual IR之前发明
  physical value、movement、storage、event、execution structure和schedule的shadow search chain。Search只保存显式choice，
  在结构choice后立即物化actual TileRegion IR，之后layout/bufferization、movement、execution structure、Instr/order/completion和
  memory均读各自current IR。
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
  下表的14个checkpoint按线性顺序通过；current production不再有future SSA/buffer/event/schedule的跨stage owner；
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
| construction input | current TensorProgram + fixed baseline rules；无search choice/domain/state | current TensorProgram + bounded spatial/region/temporal choice frontier |
| actual IR owner | baseline direct materializer | search structural materializer |
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

下表按**施工依赖**排列，不是把compiler pipeline从上游到下游照抄一遍。第6--10项先从最终actual leaf向上逐层
闭合直接consumer，并在每一项中立即替换baseline唯一production路径的对应片段；因此每项都由fresh TensorProgram
实际到达本项输出和已闭合下游，不允许只由手写fixture或未注册helper代签。第11项对这条baseline current-IR链运行
完整门禁；第12项最后建立search structural producer，并在同一修改中接通既有consumer、切换production和删除旧
shadow路径。最终search运行顺序固定为：

```text
structural choice
  -> 第12项：actual structural Card/TileRegion
  -> 第10项：current-IR layout/view/function-boundary/region-local bufferization
  -> 第9项：current-IR movement/staging/boundary closure
  -> 第8项：current Tile execution structure/rotating storage
  -> 第7项：TileRegion-to-Instr/order/completion
  -> 第6项：actual SPM/DDR/transport/target leaf
```

第12项以前不为同一policy注册第二条compiler pipeline，也不把未被真实上游和直接consumer共同读取的新stage标成完成。
Baseline和search仍是两条policy-owned路径；第6--10项只逐步替换baseline内部的shared downstream stages，search旧链只保留到
第12项原子cutover，不能作为这些新stage的完成证据。Structural producer、search production cutover和旧complete/shadow
路径删除是一个原子接口迁移，不能拆成并存的search production checkpoints。

| 顺序 | Work item | 状态 | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `expansion-evidence-and-contract` | `done` | 冻结同源两条policy的stage inventory、双路径和膨胀根因 | instrumentation on/off结果一致，违规caller和删除边界可静态复核 | 读AGENTS/progress→读06/Q52与本项矩阵→调研stage instrumentation→查官方及pinned MLIR→改诊断/tests→fresh验证→重读设计并按MLIR复审→更新并提交 |
| 2 | `verifier-contract-cleanup` | `done` | 将operation、container和materialized-stage检查收回正确owner | local verifier不跨scope，必要负例仍在直接stage失败，无总体parity verifier | 读AGENTS/progress→读19/Q52与本项矩阵→调研LLVM/MLIR verifier实践→查pinned源码→改代码/tests→fresh verify-each/build→重读设计/MLIR复审→更新并提交 |
| 3 | `search-execution-materializer` | `done` | search generic/attention使用一个execution-owned construction donor | canonical/noncanonical不选另一builder，producer compute不随fanout重复 | 读AGENTS/progress→读06/07/10与本项矩阵→调研MLIR/IREE/Triton materialization→查pinned API→改代码/tests→fresh Card/Instr验证→重读设计/MLIR复审→更新并提交 |
| 4 | `baseline-root-materializer` | `done` | baseline独立实root构造region并运行actual feedback | 每compute region恰一root，baseline call graph不进入search owner | 读AGENTS/progress→读06/07/10与本项矩阵→读DDR/completion事实→调研baseline materialization→查pinned API→改代码/tests→fresh Card/Instr/SPM验证→重读设计/MLIR复审→更新并提交 |
| 5 | `current-ir-contract-reset` | `done` | 重写active设计和施工顺序，冻结shadow owner删除清单 | AGENTS、01、05–19、Q52 plan/progress和memory只保留一套actual-IR合同；archive只作历史 | 读AGENTS/progress→读01/05–19/Q52与本项矩阵→调研VPlan/GlobalISel/Transform/Bufferization取舍→查pinned文档/源码→改current docs→文本检查→重读设计/MLIR复审→更新并提交 |
| 6 | `actual-leaf-current-ir-contract` | `pending` | 从现有Card→Executable入口抽出completion-closed canonical Instr→actual SPM/DDR/transport/target的唯一typed leaf；baseline原入口立即调用该leaf，不重新实现allocator或target | leaf不运行function-boundary bufferization、join/wait rebuild或其它completion mutation，不读取policy/preparation/shadow plan；MiniMalloc实际运行；typed failure和Accepted owner保持；fresh baseline source实际进入该leaf | 读AGENTS/progress→读09/11–14及本项矩阵→读memory/resource硬件事实→调研actual allocation/liveness与typed allocator feedback→查官方及pinned MLIR→拆分唯一stage boundary/tests→fresh baseline Instr/SPM/DDR/target验证→重读设计/MLIR复审→更新并提交 |
| 7 | `current-instr-schedule-completion` | `pending` | 消费第8项给出的function-boundary-bufferized、execution-structure-closed physical TileRegion；安装relation listener并执行TileRegion-to-Instr，随后从current Instr重建dependence，应用worker/order并fresh构造completion，输出直接进入第6项 | 本stage及baseline调用链不运行bufferization，不消费EventId→op反查或ClosedSchedulePlan；EventGraph不跨mutation；relation在conversion后仍属于current IR；join/wait有actual effect/token/lifetime和hardware witness；第6项不再修改completion；fresh baseline source经第7→6项成功 | 读AGENTS/progress→读06/09–11/13及本项矩阵→读NCC/DTE硬件/ABI事实→调研LLVM MachineScheduler/MLIR async→查pinned API→改唯一baseline stage/tests→fresh baseline TileRegion→Instr→第6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 8 | `current-tile-execution-structure` | `pending` | 在movement-closed physical TileRegion上立即物化serialized或selected software pipeline、prefix/steady/tail、chunk control、rotating allocation roots和slot SSA relation，输出直接进入第7项 | 本stage不消费ExecutionStructurePlan/BufferPlan跨stage事实，不创建completion或offset；Serialized不改IR；pipelined case的loop、slot、movement/compute occurrence all-and-only；fresh actual source经第8→7→6项成功 | 读AGENTS/progress→读06–11及本项矩阵→读worker/completion/memory硬件事实→调研MLIR/LLVM software pipelining与rotating-buffer实现→查pinned SCF/Rewriter API→改current-IR transform/tests→fresh actual source pipeline→第7→6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 9 | `current-ir-movement-boundary-closure` | `pending` | 先冻结layout-resolved endpoint→movement-closed physical TileRegion合同，再在baseline唯一路径中把movement从materializer内部选择/修补拆成current producer/use/endpoint/relation transformation，闭合staging、boundary和effect | 本stage及baseline调用链不消费future version或donor scan；compatible local edge零DDR；movement不创建compute或重选layout；payload/effect all-and-only；fresh baseline source经第9→8→7→6项成功 | 读AGENTS/progress→读06–10/13及本项矩阵→读transport硬件事实→调研MLIR data movement/fanout→查pinned API→先更新阶段合同再改唯一baseline stage/tests→fresh baseline movement→第8→7→6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 10 | `current-ir-layout-bufferization` | `pending` | 冻结structural TileRegion→layout-resolved endpoint合同，在baseline唯一路径中一次完成function-boundary与region-local bufferization、layout、view/alias和shared conversion；跨region/tile logical boundary交给第9项 | 本stage及baseline调用链不消费PhysicalVersion；function boundary、allocation、alias和effect在movement前闭合；shared conversion是一个SSA result；alias零copy/allocation；不预建route/staging；fresh baseline source经第10→9→8→7→6项成功 | 读AGENTS/progress→读06–10/19及本项矩阵→调研One-Shot Bufferize/layout propagation→查pinned `BufferizableOpInterface`→先更新阶段合同再改唯一baseline stage/tests→fresh baseline layout→第9→8→7→6项验证→重读设计/MLIR复审→更新并提交 |
| 11 | `baseline-current-ir-integration-gate` | `pending` | 对第6--10项已逐步接通的baseline唯一路径运行完整none门禁；不首次接线、不重做root构造 | baseline不调用search state/domain/materializer；每region一semantic root；第10→9→8→7→6项和actual MiniMalloc均到达；fresh current FP16 LLaMA none在15分钟内生成package并strict readback/no-card | 读AGENTS/progress→读06–16及本项矩阵→确认current baseline/source/package边界→读相关硬件/ABI事实→调研deterministic baseline集成验证→查pinned API→补完整gate/tests→fresh focused矩阵和none package/no-card→重读设计/MLIR复审→更新并提交 |
| 12 | `search-current-ir-cutover-and-shadow-retirement` | `pending` | spatial/region/temporal choice物化candidate-owned structural Card/TileRegion；layout、movement、execution structure和Instr schedule choice分别在current IR上枚举后立即变换并进入第6项；同批重接controller、切换production并删除旧shadow/complete路径 | pre-structural state无future value/buffer/event；各current-IR choice只有一个transformation owner；typed actual feedback回到不依赖CompleteCandidateKey的controller；Accepted owner不重建；旧type/builder/domain/state/caller/CMake/test/current-doc零production残留；baseline路径不变 | 读AGENTS/progress→读05–16/Q52删除清单与本项矩阵→调研MLIR Transform alternatives/VPlan边界、current-IR search transaction和原子API migration→查pinned Rewriter/IRMapping/ownership API→实现producer/current-IR枚举/controller接线并同批删除→fresh search全链及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 13 | `scale-regression-and-inventory` | `pending` | 在新actual-IR pipeline上profile并仅保留有证据的memo/priority/DP/LNS，补齐融合与Instr只读汇总 | 完整Tile/TileRegion/buffer/movement/execution-structure/Instr/target inventory；instrumentation/safe optimization on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读06/Q52及本项矩阵→读resource事实→调研search scalability算法→查pinned MLIR→改代码/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 14 | `llama-baseline-search-acceptance` | `pending` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；均实际进入Instr/MiniMalloc/DDR/target | 读AGENTS/progress→重读06/Q52/Q53及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

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
| generic/attention execution | single/nested/replica/coupled、fanout；rank 3–6，1024/1025/1031 | selected structural recipe不可表达时typed failure | 每execution all-and-only一次，fanout不重复compute | 第12项search structural producer与原子cutover |

### 4. Baseline root materializer

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| root isolation与actual feedback | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031，16 Tile | 只capacity rejection推进，其它typed状态停止 | 每region一root；candidate CardModule数等于actual planner数；accepted不重建 | baseline Instr/MiniMalloc/package |

### 5. Current-IR contract reset

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| active docs/code call graph | 01、05–19、Q52 plan/progress、memory；baseline/search两条路径 | 一个旧plan能力尚无actual owner时保留为待迁移，不伪装已删 | 每个字段分类为choice/current fact；active文档无正向shadow合同；删除清单有new owner/test | 第6–12项consumer-first施工与原子cutover的唯一边界 |

### 6. Actual leaf current-IR contract

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| completion-closed canonical unplaced Instr、actual allocation/alias/effect/lifetime | baseline focused current source；aligned/ragged；rank 3–6，1024/1025/1031；16 Tile | capacity、ResourceExhausted、timeout、unsupported、compiler error保持区分 | leaf不改变function boundary或join/wait；MiniMalloc/offset/conflict/owner只来自current IR；读取policy/preparation/shadow state次数为0；Accepted owner不重建 | baseline现有Card→Executable入口直接调用该leaf；第11项完整none与第12项search controller消费同一typed outcome |

### 7. Current Instr schedule and completion

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| baseline function-boundary-bufferized、execution-structure-closed TileRegion中的compute/DDR/DTE/control flow | straight-line、loop/tail、cross-worker、observable terminal；rank 3–6，1024/1025/1031 | missing hardware/ABI语义为typed unknown；token/effect malformed为IR failure | 本stage不运行bufferization；TileRegion-to-Instr只转换一次且relation保持current；event graph fresh重算，order后失效；minimum/latest join/wait有直接hardware/effect/lifetime witness；第6项不再重建completion | fresh TensorProgram经baseline materializer和第8项输出current Instr，直接运行第6项leaf |

### 8. Current Tile execution structure

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| movement-closed physical TileRegion上的Serialized与selected pipeline | straight-line、nested loop、multi-wave、prefix/steady/tail、observable publication；rank 3–6，1024/1025/1031 | recurrence、effect、slot reuse或required completion obligation无法由current SSA/effect/token表达时typed unknown/unsupported，不退回预测buffer | Serialized IR byte-equivalent；pipeline的chunk occurrence all-and-only；prefix/steady/tail cover完整；rotating roots、slot selection、loop-carried SSA和reuse obligation显式；不创建join/offset | fresh actual TensorProgram经同一transformation的Serialized与pipelined choice后顺序通过第7→6项 |

### 9. Current-IR movement and boundary closure

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| baseline current producer/use与layout-resolved endpoint；local/DDR/peer/relay/gather | equal/partial window、fanout/fanin、reduction、attention；rank 3–6，1024/1025/1031 | alias/effect/route不明时typed unknown/unsupported，不猜近似carrier或改layout | compatible local edge零DDR；compute数不随fragment增长；staging/boundary、payload、token和effect all-and-only；physical TileRegion无未闭合tensor boundary；baseline不消费旧movement plan | fresh TensorProgram经baseline materializer和本stage输出physical TileRegion，顺序通过第8→7→6项 |
| layout-resolved→physical form transition | direct region chain、cross-region DDR、cross-Tile peer/collective；1024/1025/1031 | malformed bridge、missing endpoint或wrong-form residue由直接stage typed拒绝 | 不使用phase attr；所有logical tensor boundary恰好闭合一次；SPM memref不成为region I/O | physical-form stage verifier与第8项execution-structure consumer |

### 10. Current-IR layout and bufferization

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| baseline structural TileRegion的function boundary与primary/shared/alias/use-local需求 | 1/2/15 uses，reshape/transpose，Tensor/NTensor/Cx/NCx，rank 3–6，1024/1025/1031 | relation或bufferization不能证明时保留explicit materialization或typed unsupported，不建立alias | function boundary与region-local bufferization只运行一次；shared layout一个SSA definition；alias同storage；allocation/effect完整；无unused conversion、route或staging；每个logical boundary有current endpoint；baseline不消费PhysicalVersion | fresh TensorProgram经baseline materializer和本stage产生layout-resolved endpoint，顺序通过第9→8→7→6项 |
| structural→layout-resolved form transition | single/multi-region、chain/fanout、tensor boundary；rank 3–6，1024/1025/1031 | missing current endpoint、arbitrary external tensor use或SPM memref boundary由直接stage typed拒绝 | 不使用phase attr；op verifier只查局部关系；card stage只允许direct TileRegion boundary chain；每个bridge有current SSA owner | 第9项movement transformation和layout-resolved stage verifier |
| current full/partial transfer cleanup | full equivalent与partial/permuted/layout-changing | 不满足exact条件不删 | 只删除current IR上证明冗余的full transfer；partial copy保留 | 第9项actual movement count与payload coverage |

### 11. Baseline current-IR integration gate

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| current TensorProgram + fixed baseline rules | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031；16 Tile；current FP16 LLaMA block | 只有actual capacity rejection构造smaller temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；每region一semantic root；第10→9→8→7→6项均到达；fresh none package唯一 | 第12项cutover的baseline隔离回归与第14项同源两policy验收 |

### 12. Search current-IR cutover and shadow retirement

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| spatial/region/temporal choice及其current-IR layout/movement/structure/schedule alternatives | chain/fanout/reduction/attention；rank 3–6，1024/1025/1031；16 Tile | rewrite前不可表达为typed failure；mutation后失败销毁candidate owner；第10→6项typed outcome原样返回controller | actual TileRegion/loop/SSA all-and-only；每层choice选中后立即变成current IR；pre-structural state无future value/buffer/event；Accepted owner不重建 | 第10→9→8→7→6项正式search production调用以及第13项scale inventory |
| source/CMake/caller/test/current-doc inventory | generic/attention、none/search、旧fixture | 仍有独有能力没有current-IR owner和direct test时停止cutover，不保留compatibility path | 旧production type/builder/domain/state/caller/CMake/current-doc为0；controller不依赖CompleteCandidateKey；search不fallback旧path/baseline；baseline调用链不变 | fresh build、named/driver、baseline隔离、actual memory/target和第14项验收 |

### 13. Scale regression and inventory

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3–6，1024/1025/1031，16 Tile | stage未到达与typed compiler/resource failure分开 | TileRegion/structured-execution/op分布、local/external use、actual buffer/movement/execution-structure、final Instr total/per-Tile/per-kind；compute duplicate为0 | actual MiniMalloc、package strict readback |
| search optimization | tiny exhaustive oracle与真实规模profile | timeout/resource不伪装exact rejection | safe optimization/instrumentation on/off保持result；有损策略只报BudgetedFeasible | retained actual winner一次publication |

### 14. LLaMA acceptance

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
- `ExecutionStructureState/Plan`作为cross-stage candidate state的路径；第8项只保留从current physical TileRegion枚举并立即应用的
  query-local choice，以及物化后的actual loop/rotating roots/slot SSA；
- `ScheduledState`、`ClosedSchedulePlan`、`EventId -> operation`物化后反查和completion parity；
- 携带上述shadow facts的`CompleteCandidateKey/Plan`、`FullFeasibility` domain rebuild和`CompleteCandidatePreparation`；
- donor movement扫描/替换/erase、edge-owned compute重建、first-use layout恢复、无producer action surface和only-purpose fixtures；
- active设计、memory、CMake和tests中把这些对象当作current事实源或supported production contract的内容。

只在一次rewrite内使用的typed choice、`IRMapping`、SSA lookup或query-local event/lifetime graph不在该删除范围，但必须在
IR mutation后失效，不能成为下一stage的事实源。

## Search scalability

只有第12项完成后才重新解释profile和优化search工作：

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
